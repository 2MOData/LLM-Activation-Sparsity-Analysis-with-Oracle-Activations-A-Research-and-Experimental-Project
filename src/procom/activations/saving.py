"""Saving the target model's activations in the form the oracle reads them.

The activations are read exactly as AObench reads them for the project's oracle
(`procom.oracle.aobench.ORACLE_LORA`, the oracle of the layer-window sweep), so the saved files can
be fed to it with `procom.oracle.aobench.ask_oracle_about_activations`:

- the residual stream at the output of decoder layer L (`model.model.layers[L]`), read with every
  LoRA disabled, i.e. from the pure base model, as when the oracle was trained. "Layer L" is the
  number written in the oracle's prompt ("Layer: 21"). In transformers' `output_hidden_states`, it
  is `hidden_states[L + 1]`, not `hidden_states[L]` (which is the input of layer L);
- a conversation goes through the chat template with thinking disabled, and with the assistant
  turn header added when the last message is the user's;
- the last 20 tokens are kept by default, as AObench gives the oracle the last 20 tokens.

One example = one `<id>.safetensors` file, holding:

    layer_<L>   (n_positions, d_model)  one tensor per saved layer, in the model's dtype
    input_ids   (n_tokens,)             the whole tokenized prompt
    positions   (n_positions,)          indices in `input_ids` of the saved rows

plus string metadata (model, layers, quantization, prompt...), so that each file describes
itself. `index.jsonl`, in the same folder, lists every example of the folder.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

FORMAT = "procom-activations/1"
INDEX_FILE = "index.jsonl"

# The three 5-layer windows kept after the AObench layer-window sweep, and the 15 layers they cover
PROJECT_WINDOWS = ((16, 17, 18, 19, 20), (21, 22, 23, 24, 25), (26, 27, 28, 29, 30))
PROJECT_LAYERS = tuple(layer for window in PROJECT_WINDOWS for layer in window)

DEFAULT_N_LAST_TOKENS = 20

# A prompt is either a chat conversation ([{"role": ..., "content": ...}, ...]) or raw text
Prompt = str | Sequence[Mapping[str, str]]

_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
# Metadata values are strings in safetensors: these ones are stored as JSON
_JSON_FIELDS = (
    "layers",
    "n_last_tokens",
    "add_generation_prompt",
    "generation_prompt_added",
    "batch_size",
    "prompt",
    "extra",
)


@dataclass
class Activations:
    """The saved activations of one example."""

    acts: dict[int, torch.Tensor]  # layer -> (n_positions, d_model)
    input_ids: torch.Tensor  # (n_tokens,) the whole prompt
    positions: torch.Tensor  # (n_positions,) indices in input_ids of the rows of each layer
    metadata: dict = field(default_factory=dict)

    @property
    def layers(self) -> list[int]:
        return sorted(self.acts)

    def tokens(self, tokenizer) -> list[str]:
        """The tokens of the saved positions, in the order of the rows."""
        return [tokenizer.decode(t) for t in self.input_ids[self.positions]]


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------


def _base_model(model):
    """The transformers model, also when `model` is wrapped in a peft `PeftModel`."""
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def decoder_layers(model):
    """The transformer blocks of the model (`model.model.layers`)."""
    return _base_model(model).model.layers


@contextlib.contextmanager
def adapters_disabled(model):
    """Inside this block, every LoRA of the model is disabled: it is the pure base model.

    Works for a peft `PeftModel` and for adapters attached with transformers' `load_adapter` /
    `add_adapter` (as AObench does). No-op for a model without adapters.
    """
    if hasattr(model, "disable_adapter"):  # peft PeftModel
        with model.disable_adapter():
            yield
    elif getattr(model, "_hf_peft_config_loaded", False):  # adapters attached by transformers
        model.disable_adapters()
        try:
            yield
        finally:
            model.enable_adapters()
    else:
        yield


def quantization_of(model) -> str:
    """How the model's weights are quantized: "none", "bitsandbytes-int8", ..."""
    config = getattr(_base_model(model).config, "quantization_config", None)
    if config is None:
        return "none"
    if not isinstance(config, Mapping):
        config = config.to_dict()
    if config.get("load_in_8bit"):
        return "bitsandbytes-int8"
    if config.get("load_in_4bit"):
        return "bitsandbytes-int4"
    return str(config.get("quant_method", "unknown"))


def _check_layers(layers: Iterable[int], n_layers: int) -> list[int]:
    layers = sorted(set(int(layer) for layer in layers))
    if not layers:
        raise ValueError("no layer to save")
    if layers[0] < 0 or layers[-1] >= n_layers:
        raise ValueError(f"layers must be in [0, {n_layers - 1}], got {layers}")
    return layers


def last_positions(n_tokens: int, n_last_tokens: int | None) -> torch.Tensor:
    """Indices of the last `n_last_tokens` tokens of a prompt (all of them for None)."""
    if n_last_tokens is not None and n_last_tokens < 1:
        raise ValueError("n_last_tokens must be >= 1, or None for all the tokens")
    start = 0 if n_last_tokens is None else max(0, n_tokens - n_last_tokens)
    return torch.arange(start, n_tokens)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def uses_generation_prompt(prompt: Prompt, add_generation_prompt: bool | None = None) -> bool:
    """Whether the assistant turn header is added after the conversation.

    None follows AObench: added when the last message is the user's. Never for raw text.
    """
    if isinstance(prompt, str):
        return False
    if add_generation_prompt is None:
        return prompt[-1]["role"] == "user"
    return add_generation_prompt


def tokenize_prompt(
    tokenizer, prompt: Prompt, add_generation_prompt: bool | None = None
) -> list[int]:
    """Token ids of a prompt, tokenized as AObench does.

    A conversation goes through the chat template with thinking disabled; `add_generation_prompt`
    is explained in `uses_generation_prompt`. Raw text is tokenized as it is.
    """
    if isinstance(prompt, str):
        text = prompt
    else:
        add_generation_prompt = uses_generation_prompt(prompt, add_generation_prompt)
        text = tokenizer.apply_chat_template(
            list(prompt),
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    return tokenizer(text, add_special_tokens=False).input_ids


@torch.no_grad()
def extract_activations(
    model,
    tokenizer,
    prompts: Sequence[Prompt],
    layers: Iterable[int],
    n_last_tokens: int | None = DEFAULT_N_LAST_TOKENS,
    add_generation_prompt: bool | None = None,
) -> list[Activations]:
    """Activations of `layers` for a batch of prompts, in one forward pass, kept in memory.

    Forward hooks read the output of each selected layer, with every LoRA disabled. Only the
    last `n_last_tokens` tokens of each prompt are kept (None keeps all of them). The returned
    tensors are on the CPU.
    """
    blocks = decoder_layers(model)
    layers = _check_layers(layers, len(blocks))
    token_ids = [tokenize_prompt(tokenizer, p, add_generation_prompt) for p in prompts]

    # Right padding: with a causal model, the real tokens never see the padding after them,
    # and their position ids stay 0, 1, 2... as without padding.
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    length = max(len(ids) for ids in token_ids)
    input_ids = torch.full((len(token_ids), length), pad_id, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    for i, ids in enumerate(token_ids):
        input_ids[i, : len(ids)] = torch.tensor(ids)
        attention_mask[i, : len(ids)] = 1

    captured: dict[int, torch.Tensor] = {}

    def make_hook(layer):
        def hook(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[layer] = hidden.detach()

        return hook

    handles = [blocks[layer].register_forward_hook(make_hook(layer)) for layer in layers]
    try:
        with adapters_disabled(model):
            model(
                input_ids=input_ids.to(model.device),
                attention_mask=attention_mask.to(model.device),
                use_cache=False,
                logits_to_keep=1,  # the logits are not needed: skip most of the lm_head
            )
    finally:
        for handle in handles:
            handle.remove()  # always remove the hooks, otherwise they stay active afterwards

    results = []
    for i, ids in enumerate(token_ids):
        positions = last_positions(len(ids), n_last_tokens)
        acts = {
            layer: captured[layer][i, positions.to(captured[layer].device)].cpu().contiguous()
            for layer in layers
        }
        results.append(Activations(acts, torch.tensor(ids), positions))
    return results


# ---------------------------------------------------------------------------
# Writing and reading files
# ---------------------------------------------------------------------------


def write_activations(path: str | Path, activations: Activations) -> Path:
    """Writes one example's activations to a .safetensors file.

    The file is written under a temporary name and then renamed, so an interrupted run never
    leaves a half-written file that would look complete.
    """
    path = Path(path)
    tensors = {f"layer_{layer}": t.contiguous() for layer, t in activations.acts.items()}
    tensors["input_ids"] = activations.input_ids.to(torch.long).contiguous()
    tensors["positions"] = activations.positions.to(torch.long).contiguous()
    metadata = {
        key: json.dumps(value) if key in _JSON_FIELDS else str(value)
        for key, value in activations.metadata.items()
    }
    metadata["format"] = FORMAT
    metadata["layers"] = json.dumps(activations.layers)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    save_file(tensors, tmp, metadata=metadata)
    os.replace(tmp, path)
    return path


def read_metadata(path: str | Path) -> dict:
    """The metadata of a file, without reading its tensors."""
    with safe_open(str(path), framework="pt") as f:
        raw = f.metadata() or {}
    return {key: json.loads(value) if key in _JSON_FIELDS else value for key, value in raw.items()}


def load_activations(
    path: str | Path, layers: Iterable[int] | None = None, device: str = "cpu"
) -> Activations:
    """Loads a file written by `save_activations`. Only the requested `layers` are read."""
    metadata = read_metadata(path)
    if metadata.get("format") != FORMAT:
        raise ValueError(f"{path} is not a {FORMAT} file")
    saved = metadata["layers"]
    layers = saved if layers is None else sorted(set(layers))
    missing = [layer for layer in layers if layer not in saved]
    if missing:
        raise KeyError(f"layers {missing} are not in {path} (saved: {saved})")
    with safe_open(str(path), framework="pt", device=device) as f:
        acts = {layer: f.get_tensor(f"layer_{layer}") for layer in layers}
        return Activations(acts, f.get_tensor("input_ids"), f.get_tensor("positions"), metadata)


def write_index(output_dir: str | Path) -> Path:
    """Rewrites `index.jsonl`: one line per .safetensors file of the folder, sorted by id."""
    output_dir = Path(output_dir)
    rows = []
    for path in sorted(output_dir.glob("*.safetensors")):
        meta = read_metadata(path)
        with safe_open(str(path), framework="pt") as f:
            n_tokens = f.get_slice("input_ids").get_shape()[0]
            n_positions = f.get_slice("positions").get_shape()[0]
        rows.append(
            {
                "id": meta.get("id", path.stem),
                "file": path.name,
                "n_tokens": n_tokens,
                "n_positions": n_positions,
                "layers": meta.get("layers"),
                **(meta.get("extra") or {}),
            }
        )
    index = output_dir / INDEX_FILE
    index.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return index


def read_index(output_dir: str | Path) -> list[dict]:
    """The rows of `index.jsonl`."""
    lines = (Path(output_dir) / INDEX_FILE).read_text().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


# ---------------------------------------------------------------------------
# Saving a dataset
# ---------------------------------------------------------------------------


def _check_examples(examples: Sequence[Mapping]) -> None:
    seen = set()
    for example in examples:
        if "id" not in example or "prompt" not in example:
            raise ValueError("each example needs an 'id' and a 'prompt'")
        example_id = str(example["id"])
        if not _ID_PATTERN.match(example_id):
            raise ValueError(f"id {example_id!r}: use only letters, digits, '_', '-' and '.'")
        if example_id in seen:
            raise ValueError(f"duplicate id {example_id!r}")
        seen.add(example_id)


def _settings_match(path: Path, settings: Mapping) -> None:
    """Raises if an existing file was saved with other settings than the current run."""
    meta = read_metadata(path)
    different = {key: (meta.get(key), value) for key, value in settings.items()}
    different = {key: pair for key, pair in different.items() if pair[0] != pair[1]}
    if different:
        raise ValueError(
            f"{path} was saved with other settings (saved, now): {different}. "
            "Use another output_dir, or overwrite=True."
        )


def save_activations(
    model,
    tokenizer,
    examples: Iterable[Mapping],
    output_dir: str | Path,
    layers: Iterable[int] = PROJECT_LAYERS,
    n_last_tokens: int | None = DEFAULT_N_LAST_TOKENS,
    add_generation_prompt: bool | None = None,
    batch_size: int = 1,
    overwrite: bool = False,
    allow_quantized: bool = False,
    progress: bool = True,
) -> Path:
    """Saves the activations of `layers` for each example, one file per example.

    examples: dicts with an "id" (used as the file name) and a "prompt" (a conversation or raw
        text). Every other key (e.g. "subject") is stored in the file's metadata and the index,
        and must be JSON-serializable.
    n_last_tokens: number of tokens kept at the end of each prompt; None keeps all of them.
    add_generation_prompt: None follows AObench (see `uses_generation_prompt`).
    batch_size: prompts per forward pass. Batching is faster, but in bf16 the activations can
        differ in the last bits from those of a batch of 1, as GPU kernels depend on the shape.
    overwrite: by default, examples that already have a file are skipped, so an interrupted run
        can be resumed. Their settings are checked against the current ones.
    allow_quantized: refuses a quantized model unless True. Quantization changes the
        activations, so files meant for experiments must come from the bf16 model.

    Returns the path of the folder's `index.jsonl`, rewritten at the end.
    """
    from tqdm.auto import tqdm

    quantization = quantization_of(model)
    if quantization != "none" and not allow_quantized:
        raise ValueError(
            f"the model is quantized ({quantization}), which changes its activations. "
            "Load it in bf16, or pass allow_quantized=True for a test."
        )
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    examples = list(examples)
    _check_examples(examples)
    layers = _check_layers(layers, len(decoder_layers(model)))
    last_positions(1, n_last_tokens)  # checks n_last_tokens
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import transformers

    settings = {
        "model": _base_model(model).config._name_or_path,
        "layers": layers,
        "n_last_tokens": n_last_tokens,
        "add_generation_prompt": add_generation_prompt,
        "quantization": quantization,
    }
    todo = []
    for example in examples:
        path = output_dir / f"{example['id']}.safetensors"
        if path.exists() and not overwrite:
            _settings_match(path, settings)
        else:
            todo.append(example)

    batches = [todo[i : i + batch_size] for i in range(0, len(todo), batch_size)]
    for batch in tqdm(batches, desc="saving activations", disable=not progress):
        results = extract_activations(
            model,
            tokenizer,
            [example["prompt"] for example in batch],
            layers,
            n_last_tokens=n_last_tokens,
            add_generation_prompt=add_generation_prompt,
        )
        for example, activations in zip(batch, results, strict=True):
            extra = {k: v for k, v in example.items() if k not in ("id", "prompt")}
            activations.metadata = {
                "id": str(example["id"]),
                **settings,
                "activation_site": "output of model.model.layers[L], LoRA disabled",
                "dtype": str(next(iter(activations.acts.values())).dtype).removeprefix("torch."),
                "generation_prompt_added": uses_generation_prompt(
                    example["prompt"], add_generation_prompt
                ),
                "batch_size": batch_size,
                "prompt": example["prompt"],
                "extra": extra,
                "torch_version": torch.__version__,
                "transformers_version": transformers.__version__,
            }
            write_activations(output_dir / f"{example['id']}.safetensors", activations)
    return write_index(output_dir)
