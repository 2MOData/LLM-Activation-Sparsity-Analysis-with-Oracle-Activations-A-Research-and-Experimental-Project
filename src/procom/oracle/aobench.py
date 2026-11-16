"""AObench on Qwen3-8B, with the oracle's 5-layer input window moved along the model depth.

AObench is the Activation Oracle benchmark of "Building Better Activation Oracles"
(arXiv:2606.02609). It is not a pip package: `install_aobench` clones it and puts it on
`sys.path`, and everything that imports it does so lazily, so this module can be imported
without it.

The paper's best oracle (`ORACLE_LORA`) was trained on the residual stream of layers 21 to 25.
AObench always evaluates an oracle on the layers written in its training config, so to evaluate
it on another window, `shifted_window` temporarily swaps the layers AObench reads from that
config. Nothing else in the benchmark is changed.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import importlib
import json
import math
import os
import random
import re
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MODEL_NAME = "Qwen/Qwen3-8B"
N_LAYERS = 36
ORACLE_LORA = "ceselder/qwen3-8b-ao-v3-best"  # the paper's full recipe
TRAINED_WINDOW = (21, 22, 23, 24, 25)  # layers ORACLE_LORA was trained to read

AOBENCH_REPO = "https://github.com/ceselder/cot-oracle"
# Commit pinned by the paper's code (github.com/japhba/activation_oracles, third_party/cot-oracle)
AOBENCH_COMMIT = "bec6f8a82c6882faa3a931a88978bd6af0694c27"

# The evals that count in the paper's aggregate score and need no extra model.
# taboo and personaqa are left out: they need fine-tuned target models.
OBJECTIVE_EVALS = ("number_prediction", "mmlu_prediction", "missing_info", "sycophancy")
JUDGE_EVALS = (
    "backtracking",
    "vagueness",
    "domain_confusion",
    "activation_sensitivity",
    "hallucination",
)
# 10 examples per eval, to check the whole pipeline in a few minutes
QUICK_SAMPLE_LIMITS = {name: 10 for name in OBJECTIVE_EVALS + JUDGE_EVALS}


# ---------------------------------------------------------------------------
# Installation and loading
# ---------------------------------------------------------------------------


def install_aobench(dest: str | Path = "/content/aobench") -> Path:
    """Clones AObench at the pinned commit (only the `AObench/` folder) and makes it importable."""
    dest = Path(dest)
    if not (dest / "AObench" / "__init__.py").exists():
        dest.mkdir(parents=True, exist_ok=True)
        for cmd in (
            ["git", "init", "-q"],
            ["git", "remote", "add", "origin", AOBENCH_REPO],
            ["git", "sparse-checkout", "set", "AObench"],
            ["git", "fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", AOBENCH_COMMIT],
            ["git", "checkout", "-q", "FETCH_HEAD"],
        ):
            subprocess.run(cmd, cwd=dest, check=True)
    if str(dest) not in sys.path:
        sys.path.insert(0, str(dest))
    # Without an Anthropic key, AObench's judge falls back by default to a private server of its
    # authors, hard-coded in judge.py. We never want to send our data there.
    if _local_judge is None:
        os.environ["JUDGE_USE_LOCAL"] = "0"
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return dest


def judge_available() -> bool:
    """True if a judge can grade the open-ended evals: `use_local_judge`, or an API key."""
    return _local_judge is not None or bool(
        os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    )


_JUDGE_FIELD = re.compile(r'"(\w+)"\s*:\s*(?:(-?\d+(?:\.\d+)?)|"((?:[^"\\]|\\.)*))')


def parse_judge_answer(text: str) -> dict:
    """The judge's JSON answer as a dict, even when the JSON is cut off or badly escaped.

    The verdict ("category", or the grades) comes before the free-text "reasoning", so it can
    still be read field by field from an answer that stops mid-sentence or contains a LaTeX
    backslash. Raises json.JSONDecodeError if no verdict can be read.
    """
    try:
        payload = json.loads(text[text.index("{") : text.rindex("}") + 1])
        if isinstance(payload, dict):
            return payload
    except ValueError:
        pass
    fields: dict = {}
    for key, number, string in _JUDGE_FIELD.findall(text):
        fields.setdefault(
            key, (float(number) if "." in number else int(number)) if number else string
        )
    if not set(fields) - {"reasoning"}:
        raise json.JSONDecodeError("No verdict found in the judge's answer", text, 0)
    return fields


class LocalJudge:
    """An open-weights chat model on our own GPU, standing in for AObench's API judge.

    AObench sends every judge request of an eval at once, as concurrent coroutines. They are
    collected here and answered in batches. The model is asked for the same JSON as the API
    judge; its answer is started with "{" to keep it on track.
    """

    def __init__(self, model_name: str, load_in_4bit: bool = True, batch_size: int = 16):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        quantization = None
        if load_in_4bit:
            quantization = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
            )
        self.model_name = model_name
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=quantization,
        )
        self.model.eval()
        self.answers: dict[str, str] = {}  # prompt -> raw answer, so a retry costs nothing
        self.unparsed: set[str] = set()  # prompts whose answer held no readable verdict
        self._pending: list[tuple[str, int, asyncio.Future]] = []

    async def __call__(self, *, messages, max_tokens, **_unused) -> dict:
        """Same contract as AObench's `_call_chat_completions`: returns the parsed JSON answer."""
        prompt = messages[-1]["content"]
        if prompt not in self.answers:
            loop = asyncio.get_running_loop()
            future = loop.create_future()
            self._pending.append((prompt, max_tokens, future))
            if len(self._pending) == 1:
                # Runs once the other requests already scheduled have queued up behind this one
                loop.call_soon(self._flush)
            await future
        try:
            return parse_judge_answer(self.answers[prompt])
        except json.JSONDecodeError:
            self.unparsed.add(prompt)
            raise

    def _flush(self) -> None:
        pending, self._pending = self._pending, []
        try:
            prompts = sorted({prompt for prompt, _, _ in pending}, key=len)
            # AObench asks for 120 tokens, which often cuts the reasoning before the JSON closes
            max_new_tokens = max(256, *(max_tokens for _, max_tokens, _ in pending))
            for i in range(0, len(prompts), self.batch_size):
                self._generate(prompts[i : i + self.batch_size], max_new_tokens)
        except Exception as error:
            for _, _, future in pending:
                future.set_exception(error)
            return
        for _, _, future in pending:
            future.set_result(None)

    def _generate(self, prompts: list[str], max_new_tokens: int) -> None:
        import torch

        texts = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            + "{"
            for prompt in prompts
        ]
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, add_special_tokens=False
        ).to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        new_tokens = output[:, inputs.input_ids.shape[1] :]
        decoded = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        for prompt, answer in zip(prompts, decoded, strict=True):
            self.answers[prompt] = "{" + answer


_local_judge: LocalJudge | None = None


def use_local_judge(
    model_name: str = "Qwen/Qwen3-14B", load_in_4bit: bool = True, batch_size: int = 16
) -> LocalJudge:
    """Grades AObench's open-ended evals with a model on our GPU instead of a paid API.

    The default, Qwen3-14B in 4-bit, takes about 10 GB next to the 16 GB of Qwen3-8B.
    The grades are not comparable with the paper's (its judge is Claude Sonnet 4.6), but every
    window is graded by the same judge, which is what comparing windows needs.
    """
    global _local_judge
    import AObench.open_ended_eval.judge as judge

    if _local_judge is None or _local_judge.model_name != model_name:
        _local_judge = LocalJudge(model_name, load_in_4bit, batch_size)
    judge._call_chat_completions = _local_judge
    os.environ["JUDGE_USE_LOCAL"] = "1"  # AObench's name for "the non-Anthropic endpoint"
    os.environ["JUDGE_USE_ANTHROPIC"] = "0"
    os.environ["JUDGE_MODEL"] = model_name
    # The evals copied the judge's name when they were imported: update it so the logs name ours
    for module_name in ("judge", *JUDGE_EVALS, "system_prompt_qa"):
        module = importlib.import_module(f"AObench.open_ended_eval.{module_name}")
        if hasattr(module, "JUDGE_MODEL"):
            module.JUDGE_MODEL = model_name
    return _local_judge


def load_model_and_tokenizer(model_name: str = MODEL_NAME):
    """Loads the base model in bf16 the way AObench does. The oracle LoRA is added on demand."""
    import torch
    from AObench.utils.common import load_model, load_tokenizer

    torch.set_grad_enabled(False)
    tokenizer = load_tokenizer(model_name)
    model = load_model(model_name, torch.bfloat16)
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Moving the window
# ---------------------------------------------------------------------------


def window_layers(start: int, width: int = len(TRAINED_WINDOW)) -> list[int]:
    return list(range(start, start + width))


def window_schedule(
    first_start: int = 13,
    last_start: int = N_LAYERS - len(TRAINED_WINDOW),
    stride: int = 1,
    trained_start: int = TRAINED_WINDOW[0],
) -> list[int]:
    """Start layers of the windows to evaluate, ordered from the trained window outwards.

    The grid always contains the trained window. Running the closest windows first means an
    interrupted sweep still gives a usable curve around the reference point.
    """
    assert stride >= 1 and first_start <= trained_start <= last_start
    starts = [s for s in range(first_start, last_start + 1) if (s - trained_start) % stride == 0]
    return sorted(starts, key=lambda s: (abs(s - trained_start), s))


def layer_to_percent(layer: int, n_layers: int = N_LAYERS) -> int:
    """A depth percentage that AObench's `layer_percent_to_layer` maps back to `layer`.

    AObench configs name layers by depth percentage and convert with `int(n_layers * (p / 100))`.
    """
    percent = math.ceil(layer * 100 / n_layers)
    assert int(n_layers * (percent / 100)) == layer, f"no percentage for layer {layer}"
    return percent


@contextlib.contextmanager
def shifted_window(
    layers: list[int] | None,
    prompt_layers: str = "actual",
    target_activations: str = "base",
):
    """Inside this block, AObench feeds the oracle the activations of `layers`.

    layers: the layers to read, in order. None keeps the layers the oracle was trained on.
    prompt_layers: the oracle's prompt names the layer of each block of activations
        ("Layer: 21", ...). "actual" writes the layers really read, so the prompt never lies;
        outside the trained window these are numbers the oracle never saw. "trained" keeps the
        names seen in training whatever the layers read.
    target_activations: "base" reads the activations with every LoRA disabled (pure Qwen3-8B),
        which is how they were read when the oracle was trained. "aobench" keeps the benchmark's
        behaviour: it leaves the oracle's own LoRA active while reading them, from the second
        batch of each eval onwards.
    """
    assert prompt_layers in ("trained", "actual")
    assert target_activations in ("aobench", "base")
    import AObench.base_experiment as base_experiment
    import AObench.open_ended_eval.eval_runner as eval_runner
    import AObench.utils.dataset_utils as dataset_utils

    original_read = base_experiment.read_training_config
    original_prefix = dataset_utils.get_introspection_prefix
    original_build = eval_runner.build_verbalizer_eval_config
    trained: dict[str, list[int]] = {}

    def read_training_config(path_or_repo):
        config = original_read(path_or_repo)
        assert len(config.act_layer_combinations) == 1, "expected a single trained layer window"
        trained["layers"] = list(config.act_layer_combinations[0])
        if layers is None:
            return config
        assert len(layers) == len(trained["layers"]), "the window must keep its trained width"
        return dataclasses.replace(
            config,
            layer_combinations=[[layer_to_percent(layer) for layer in layers]],
            act_layer_combinations=[list(layers)],
        )

    def get_introspection_prefix(prefix_layers, num_positions, **kwargs):
        if prompt_layers == "trained" and len(prefix_layers) == len(trained.get("layers", [])):
            prefix_layers = trained["layers"]
        return original_prefix(prefix_layers, num_positions, **kwargs)

    def build_verbalizer_eval_config(*args, **kwargs):
        config = original_build(*args, **kwargs)
        if target_activations == "base":
            config.activation_input_types = ["orig"]
        return config

    base_experiment.read_training_config = read_training_config
    dataset_utils.get_introspection_prefix = get_introspection_prefix
    eval_runner.build_verbalizer_eval_config = build_verbalizer_eval_config
    try:
        yield
    finally:
        base_experiment.read_training_config = original_read
        dataset_utils.get_introspection_prefix = original_prefix
        eval_runner.build_verbalizer_eval_config = original_build


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def _in_thread(fn):
    """Runs `fn` in a worker thread, without gradients.

    AObench's judge calls `asyncio.run`, which fails in the notebook's own thread because Jupyter
    already runs an event loop there. `torch.set_grad_enabled` is per thread, hence `no_grad`.
    """
    import torch

    def target():
        with torch.no_grad():
            return fn()

    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(target).result()


def ask_oracle(
    model,
    tokenizer,
    messages: list[dict[str, str]],
    question: str,
    layers: list[int] | None = None,
    n_positions: int = 20,
    max_new_tokens: int = 64,
    oracle_lora: str = ORACLE_LORA,
    **window_kwargs,
) -> str:
    """Asks the oracle `question` about the last `n_positions` tokens of a conversation.

    The activations are read at `layers` (default: the trained window) through AObench's own
    code path, so this is exactly what the benchmark does for one example.
    """
    import AObench.base_experiment as base_experiment
    import AObench.open_ended_eval.eval_runner as eval_runner

    def run():
        eval_runner.ensure_default_adapter(model)
        with shifted_window(layers, **window_kwargs):
            adapter, training_config = base_experiment.load_oracle_adapter(model, oracle_lora)
            config = eval_runner.build_verbalizer_eval_config(
                model_name=MODEL_NAME,
                training_config=training_config,
                eval_batch_size=8,
                generation_kwargs={"do_sample": False, "max_new_tokens": max_new_tokens},
            )
            token_ids = base_experiment.tokenize_chat_messages(
                tokenizer, messages, add_generation_prompt=messages[-1]["role"] == "user"
            )
            info = base_experiment.VerbalizerInputInfo(
                context_token_ids=token_ids,
                positions=base_experiment.compute_segment_positions(len(token_ids), -n_positions),
                verbalizer_prompt=question,
                ground_truth="",
            )
            results = base_experiment.run_verbalizer(
                model=model,
                tokenizer=tokenizer,
                verbalizer_prompt_infos=[info],
                verbalizer_lora_path=adapter,
                target_lora_path=None,
                config=config,
                device=model.device,
            )
        return results[0].responses[0]

    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        with contextlib.redirect_stderr(devnull):
            return _in_thread(run)


def window_dir(output_dir: str | Path, start: int) -> Path:
    layers = window_layers(start)
    return Path(output_dir) / f"window_{layers[0]:02d}-{layers[-1]:02d}"


def _summary_path(output_dir: str | Path, start: int, eval_name: str) -> Path:
    return window_dir(output_dir, start) / f"{eval_name}_summary.json"


def _format_duration(seconds: float) -> str:
    minutes = round(seconds / 60)
    return f"{minutes // 60}h{minutes % 60:02d}"


def estimate_remaining_seconds(
    remaining: list[tuple[int, str]], durations: dict[str, list[float]]
) -> float | None:
    """Time left for the `remaining` (window, eval) tasks, from the durations seen per eval.

    An eval that has not been timed yet is counted as the average of those that have.
    """
    timed = [d for values in durations.values() for d in values]
    if not timed:
        return None
    fallback = sum(timed) / len(timed)
    return sum(
        sum(durations[name]) / len(durations[name]) if durations.get(name) else fallback
        for _, name in remaining
    )


def _raw_path(output_dir: str | Path, start: int, eval_name: str) -> Path:
    return window_dir(output_dir, start) / "raw" / f"{eval_name}.json"


def _is_judged(output_dir: str | Path, start: int, eval_name: str) -> bool:
    path = _summary_path(output_dir, start, eval_name)
    return path.exists() and not json.loads(path.read_text()).get("raw_only", False)


@contextlib.contextmanager
def _environment(**variables: str):
    previous = {name: os.environ.get(name) for name in variables}
    os.environ.update(variables)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _run_tasks(todo: list[tuple[int, str]], n_done: int, durations, run_one, output_dir, desc):
    """Runs `run_one(start, eval, directory)` for each task, with the progress bar and the ETA.

    Each task runs in a worker thread with AObench's output sent to the window's log.txt.
    A task that crashes is reported and skipped.
    """
    from tqdm.auto import tqdm

    failures = []
    bar = tqdm(
        total=n_done + len(todo),
        initial=n_done,
        desc=desc,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} evals [{elapsed} elapsed] {postfix}",
    )
    for index, (start, name) in enumerate(todo):
        layers = window_layers(start)
        eta = estimate_remaining_seconds(todo[index:], durations)
        bar.set_postfix_str(
            f"layers {layers[0]}-{layers[-1]}, {name}, "
            + (f"~{_format_duration(eta)} left" if eta is not None else "ETA after the first eval")
        )
        directory = window_dir(output_dir, start)
        directory.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        with open(directory / "log.txt", "a") as log:
            try:
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    _in_thread(lambda: run_one(start, name, directory))  # noqa: B023
                durations.setdefault(name, []).append(time.perf_counter() - started)
            except Exception:
                log.write(traceback.format_exc())
                failures.append((layers, name))
                error = traceback.format_exc().strip().splitlines()[-1]
                bar.write(f"FAILED layers {layers[0]}-{layers[-1]}, {name}: {error}")
        bar.update(1)
    bar.set_postfix_str("done" if not failures else f"{len(failures)} evals failed")
    bar.close()
    if failures:
        print(f"{len(failures)} evals failed (details in each window's log.txt). Rerun to retry.")
    if _local_judge is not None and _local_judge.answers:
        print(
            f"Local judge: {len(_local_judge.unparsed)} of {len(_local_judge.answers)} answers "
            "held no readable verdict; AObench leaves those examples out of the scores."
        )


def run_sweep(
    model,
    tokenizer,
    starts: list[int],
    output_dir: str | Path,
    evals: list[str] | None = None,
    oracle_lora: str = ORACLE_LORA,
    n_positions: int | None = None,
    sample_limits: dict[str, int] | None = None,
    prompt_layers: str = "actual",
    target_activations: str = "base",
    judge_now: bool | None = None,
) -> None:
    """Runs AObench once per window, one eval at a time, with a progress bar and an ETA.

    Each (window, eval) writes `<output_dir>/window_AA-BB/<eval>_summary.json` when it finishes
    and is skipped if that file exists: rerun the same call to resume after a disconnection. An
    eval that crashes is reported and skipped, and retried on the next call. AObench's own output
    goes to `window_AA-BB/log.txt`.

    judge_now: whether the judged evals are graded as soon as the oracle has answered. Defaults
        to True if a judge is set up. With False, only Qwen3-8B and the oracle are needed: the
        oracle's answers are saved and `judge_sweep` grades them later, possibly on another
        machine. The answers are saved in `window_AA-BB/raw/` in both cases.
    """
    import torch
    from AObench.open_ended_eval.run_all import run_all_evals

    if judge_now is None:
        judge_now = judge_available()
    evals = list(evals) if evals is not None else list(OBJECTIVE_EVALS + JUDGE_EVALS)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = {
        "oracle_lora": oracle_lora,
        "n_positions": n_positions,
        "sample_limits": sample_limits,
        "prompt_layers": prompt_layers,
        "target_activations": target_activations,
        "aobench_commit": AOBENCH_COMMIT,
    }
    settings_path = output_dir / "sweep_settings.json"
    previous = json.loads(settings_path.read_text()) if settings_path.exists() else {}
    changed = {k for k in settings if previous and settings[k] != previous.get(k)}
    assert not changed, (
        f"{output_dir} holds results obtained with other settings ({sorted(changed)}): "
        "use another output_dir"
    )
    if judge_now and any(name in JUDGE_EVALS for name in evals):
        settings["judge_model"] = _check_judge(previous)
    settings_path.write_text(json.dumps(previous | settings, indent=2))

    tasks = [(start, name) for start in starts for name in evals]
    todo = [task for task in tasks if not _summary_path(output_dir, *task).exists()]
    # Durations of the tasks already done (AObench stores them), so the ETA survives a resume
    durations: dict[str, list[float]] = {}
    for start, name in tasks:
        path = _summary_path(output_dir, start, name)
        if path.exists():
            durations.setdefault(name, []).append(json.loads(path.read_text())["elapsed_seconds"])

    def run_one(start, name, directory):
        import AObench.base_experiment as base_experiment

        random.seed(42)
        torch.manual_seed(42)
        original = base_experiment.run_verbalizer
        calls = []

        def recording_run_verbalizer(*args, **kwargs):
            results = original(*args, **kwargs)
            calls.append(
                [dataclasses.asdict(result) | {"context_token_ids": []} for result in results]
            )
            return results

        base_experiment.run_verbalizer = recording_run_verbalizer
        try:
            with shifted_window(window_layers(start), prompt_layers, target_activations):
                with _environment(AOBENCH_SAVE_RAW_ONLY="0" if judge_now else "1"):
                    run_all_evals(
                        model=model,
                        tokenizer=tokenizer,
                        device=model.device,
                        model_name=MODEL_NAME,
                        output_dir=str(directory),
                        verbalizer_lora_paths=[oracle_lora],
                        include=[name],
                        n_positions=n_positions,
                        sample_limits=sample_limits,
                    )
        finally:
            base_experiment.run_verbalizer = original
        if name in JUDGE_EVALS:
            raw_path = _raw_path(output_dir, start, name)
            raw_path.parent.mkdir(exist_ok=True)
            raw_path.write_text(json.dumps(calls))

    _run_tasks(todo, len(tasks) - len(todo), durations, run_one, output_dir, "AObench sweep")
    if not judge_now:
        print("The judged evals hold the oracle's answers only: grade them with judge_sweep.")


def _check_judge(previous_settings: dict) -> str:
    """Name of the judge in use, after checking it is the one that graded this folder so far."""
    assert judge_available(), "No judge: call use_local_judge first"
    judge_model = os.environ.get("JUDGE_MODEL", "claude-sonnet-4-6")
    graded_by = previous_settings.get("judge_model")
    assert graded_by in (None, judge_model), (
        f"This folder was graded by {graded_by}, not {judge_model}. Grades from two judges must "
        "not be mixed: copy the folder, or call judge_sweep(..., rejudge=True) to regrade it all"
    )
    return judge_model


class _NoModel:
    """Stands in for Qwen3-8B when grading saved answers: AObench only touches its adapters."""

    peft_config = {"default": None}

    def eval(self):
        return self

    def load_adapter(self, *args, **kwargs):
        pass

    def delete_adapter(self, *args, **kwargs):
        pass


def judge_sweep(
    output_dir: str | Path,
    starts: list[int] | None = None,
    rejudge: bool = False,
) -> None:
    """Grades the oracle's answers saved by `run_sweep`, without Qwen3-8B or the oracle.

    Each judged eval is replayed through AObench with the saved answers in place of the
    generation, so the prompts, the grading and the metrics are AObench's own. Only the judge is
    needed on the GPU (see `use_local_judge`). Like `run_sweep`, it resumes where it stopped.

    rejudge: grade again what was already graded, for instance with another judge.
    """
    import AObench.base_experiment as base_experiment
    from AObench.open_ended_eval.run_all import run_all_evals
    from AObench.utils.common import load_tokenizer

    output_dir = Path(output_dir)
    settings_path = output_dir / "sweep_settings.json"
    settings = json.loads(settings_path.read_text())
    settings["judge_model"] = _check_judge({} if rejudge else settings)
    settings_path.write_text(json.dumps(settings, indent=2))
    tokenizer = load_tokenizer(MODEL_NAME)

    tasks = [
        (int(raw.parent.parent.name.split("_")[1].split("-")[0]), raw.stem)
        for raw in sorted(output_dir.glob("window_*/raw/*.json"))
    ]
    tasks = [task for task in tasks if starts is None or task[0] in starts]
    if starts is not None:
        tasks.sort(key=lambda task: starts.index(task[0]))
    todo = [task for task in tasks if rejudge or not _is_judged(output_dir, *task)]

    def run_one(start, name, directory):
        saved_calls = json.loads(_raw_path(output_dir, start, name).read_text())
        original = base_experiment.run_verbalizer

        def replayed_run_verbalizer(*args, verbalizer_prompt_infos, **kwargs):
            saved = saved_calls.pop(0)
            asked = [info.verbalizer_prompt for info in verbalizer_prompt_infos]
            assert asked == [
                result["verbalizer_prompt"] for result in saved
            ], "the saved answers do not match this eval's questions"
            return [base_experiment.VerbalizerResults(**result) for result in saved]

        base_experiment.run_verbalizer = replayed_run_verbalizer
        try:
            layers = window_layers(start)
            with shifted_window(layers, settings["prompt_layers"], settings["target_activations"]):
                with _environment(AOBENCH_SAVE_RAW_ONLY="0"):
                    run_all_evals(
                        model=_NoModel(),
                        tokenizer=tokenizer,
                        device="cpu",
                        model_name=MODEL_NAME,
                        output_dir=str(directory),
                        verbalizer_lora_paths=[settings["oracle_lora"]],
                        include=[name],
                        n_positions=settings["n_positions"],
                        sample_limits=settings["sample_limits"],
                    )
        finally:
            base_experiment.run_verbalizer = original

    _run_tasks(todo, len(tasks) - len(todo), {}, run_one, output_dir, "AObench judging")


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


def window_minutes(output_dir: str | Path, start: int) -> float:
    """Minutes the evals of a window took, all phases included."""
    summaries = window_dir(output_dir, start).glob("*_summary.json")
    return sum(json.loads(path.read_text())["elapsed_seconds"] for path in summaries) / 60


def collect_results(output_dir: str | Path, starts: list[int] | None = None):
    """One row per finished (window, eval). Judged evals appear once they have been graded.

    score: the eval's own metric (ROC AUC, exact-match rate, mean judge grade, ...).
    normalized: the same on the 0-1 scale of the paper's aggregate score, where 0 is chance
        for the yes/no evals.
    """
    import pandas as pd
    from AObench.utils.report import extract_verbalizer_metric, normalize_metric_for_aggregate

    output_dir = Path(output_dir)
    rows = []
    for directory in sorted(output_dir.glob("window_*")):
        start = int(directory.name.split("_")[1].split("-")[0])
        if starts is not None and start not in starts:
            continue
        for path in sorted(directory.glob("*_summary.json")):
            name = path.name.removesuffix("_summary.json")
            summary = json.loads(path.read_text())
            values = list(extract_verbalizer_metric(summary, name).values())
            if not values:
                continue
            score = sum(values) / len(values)
            rows.append(
                {
                    "start": start,
                    "layers": f"{start}-{start + len(TRAINED_WINDOW) - 1}",
                    "eval": name,
                    "score": score,
                    "normalized": normalize_metric_for_aggregate(name, score),
                    "minutes": summary["elapsed_seconds"] / 60,
                }
            )
    return pd.DataFrame(rows, columns=["start", "layers", "eval", "score", "normalized", "minutes"])


def aggregate_scores(results):
    """AObench aggregate score per window: the mean of the normalized scores of its evals.

    Only windows that have every eval present in the results are kept, so that all the
    aggregates average the same evals.
    """
    n_evals = results["eval"].nunique()
    table = results.pivot(index="start", columns="eval", values="normalized")
    table = table[table.notna().sum(axis=1) == n_evals]
    table.insert(0, "aggregate", table.mean(axis=1))
    return table


def plot_sweep(results, trained_start: int = TRAINED_WINDOW[0]):
    """Aggregate score against the window position, then one small panel per eval."""
    import matplotlib.pyplot as plt

    table = aggregate_scores(results)
    eval_names = [column for column in table.columns if column != "aggregate"]
    n_cols = 3
    n_rows = math.ceil(len(eval_names) / n_cols)
    fig = plt.figure(figsize=(11, 3.6 + 2.3 * n_rows), constrained_layout=True)
    grid = fig.add_gridspec(1 + n_rows, n_cols, height_ratios=[1.6] + [1] * n_rows)
    ink, muted, line = "#1f2328", "#6b7280", "#2563eb"

    def draw(ax, column, title, size):
        ax.axvline(trained_start, color=muted, linewidth=1, linestyle=(0, (4, 3)))
        ax.plot(table.index, table[column], color=line, linewidth=2, marker="o", markersize=size)
        ax.set_title(title, loc="left", color=ink, fontsize=10)
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(colors=muted, length=0, labelsize=8)
        ax.set_xticks(list(table.index))

    top = fig.add_subplot(grid[0, :])
    draw(top, "aggregate", "AObench aggregate score (mean of the normalized evals)", 6)
    top.set_xticklabels([f"{s}-{s + len(TRAINED_WINDOW) - 1}" for s in table.index])
    top.set_xlabel("layers fed to the oracle", color=muted, fontsize=9)
    if trained_start in table.index:
        top.annotate(
            "trained window",
            (trained_start, 0.03),
            xycoords=("data", "axes fraction"),
            xytext=(6, 0),
            textcoords="offset points",
            color=muted,
            fontsize=9,
        )
    for i, name in enumerate(eval_names):
        ax = fig.add_subplot(grid[1 + i // n_cols, i % n_cols])
        draw(ax, name, name, 3)
        ax.set_xticks(list(table.index)[:: max(1, len(table.index) // 6)])
    fig.suptitle(
        f"Oracle quality when its 5-layer window starts at layer x (trained at {trained_start})",
        x=0.01,
        ha="left",
        color=ink,
        fontsize=12,
    )
    return fig
