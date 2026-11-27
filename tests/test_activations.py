"""Tests of the activation files. No model is loaded: they only use small random tensors.

The extraction itself (hooks on a real model) is checked in
notebooks/tutorial_saving_activations.ipynb, on a Colab GPU.
"""

import json
from types import SimpleNamespace

import pytest
import torch

from procom.activations import (
    PROJECT_LAYERS,
    Activations,
    adapters_disabled,
    load_activations,
    quantization_of,
    read_index,
    read_metadata,
    uses_generation_prompt,
    write_activations,
    write_index,
)
from procom.activations.saving import _check_examples, last_positions


def make_activations(layers=(16, 17, 18), n_tokens=30, n_positions=20, d_model=8):
    positions = last_positions(n_tokens, n_positions)
    acts = {layer: torch.randn(len(positions), d_model, dtype=torch.bfloat16) for layer in layers}
    metadata = {
        "id": "facts-001",
        "model": "Qwen/Qwen3-8B",
        "n_last_tokens": n_positions,
        "add_generation_prompt": False,
        "quantization": "none",
        "prompt": [{"role": "user", "content": "Hello"}],
        "extra": {"subject": "facts"},
    }
    return Activations(acts, torch.arange(n_tokens), positions, metadata)


def test_project_layers():
    assert PROJECT_LAYERS == tuple(range(16, 31))


def test_last_positions():
    assert last_positions(30, 20).tolist() == list(range(10, 30))
    assert last_positions(5, 20).tolist() == list(range(5))  # prompt shorter than 20 tokens
    assert last_positions(5, None).tolist() == list(range(5))
    with pytest.raises(ValueError):
        last_positions(5, 0)


def test_round_trip(tmp_path):
    original = make_activations()
    path = write_activations(tmp_path / "facts-001.safetensors", original)
    loaded = load_activations(path)
    assert loaded.layers == [16, 17, 18]
    for layer in original.layers:
        assert loaded.acts[layer].dtype == torch.bfloat16
        assert torch.equal(loaded.acts[layer], original.acts[layer])
    assert torch.equal(loaded.input_ids, original.input_ids)
    assert torch.equal(loaded.positions, original.positions)
    assert loaded.metadata["prompt"] == original.metadata["prompt"]
    assert loaded.metadata["extra"] == {"subject": "facts"}
    assert loaded.metadata["n_last_tokens"] == 20
    assert not (tmp_path / "facts-001.safetensors.tmp").exists()


def test_load_some_layers(tmp_path):
    path = write_activations(tmp_path / "a.safetensors", make_activations())
    assert load_activations(path, layers=[18, 16]).layers == [16, 18]
    with pytest.raises(KeyError):
        load_activations(path, layers=[19])


def test_index(tmp_path):
    for i, subject in enumerate(["facts", "calculation"]):
        activations = make_activations(n_tokens=10 + i)
        activations.metadata["id"] = f"{subject}-000"
        activations.metadata["extra"] = {"subject": subject}
        write_activations(tmp_path / f"{subject}-000.safetensors", activations)
    write_index(tmp_path)
    rows = read_index(tmp_path)
    assert [row["id"] for row in rows] == ["calculation-000", "facts-000"]
    assert rows[0]["subject"] == "calculation"
    assert rows[0]["n_tokens"] == 11 and rows[0]["n_positions"] == 11
    assert json.loads(json.dumps(rows))  # plain JSON
    assert read_metadata(tmp_path / "facts-000.safetensors")["layers"] == [16, 17, 18]


def test_check_examples():
    _check_examples([{"id": "a-1", "prompt": "x"}, {"id": "a-2", "prompt": "y"}])
    with pytest.raises(ValueError):
        _check_examples([{"id": "a", "prompt": "x"}, {"id": "a", "prompt": "y"}])
    with pytest.raises(ValueError):
        _check_examples([{"id": "a/b", "prompt": "x"}])
    with pytest.raises(ValueError):
        _check_examples([{"id": "a"}])


def test_quantization_of():
    plain = SimpleNamespace(config=SimpleNamespace())
    assert quantization_of(plain) == "none"
    int8 = SimpleNamespace(config=SimpleNamespace(quantization_config={"load_in_8bit": True}))
    assert quantization_of(int8) == "bitsandbytes-int8"


def test_uses_generation_prompt():
    user_last = [{"role": "user", "content": "Hi"}]
    assistant_last = user_last + [{"role": "assistant", "content": "Hello"}]
    assert uses_generation_prompt(user_last)  # AObench's rule
    assert not uses_generation_prompt(assistant_last)
    assert not uses_generation_prompt(user_last, add_generation_prompt=False)
    assert not uses_generation_prompt("raw text", add_generation_prompt=True)


class TransformersAdapters:
    """Stands for a model whose LoRA was attached by transformers' `load_adapter` (AObench)."""

    _hf_peft_config_loaded = True

    def __init__(self):
        self.enabled = True

    def disable_adapters(self):
        self.enabled = False

    def enable_adapters(self):
        self.enabled = True


def test_adapters_disabled():
    model = TransformersAdapters()
    with adapters_disabled(model):
        assert not model.enabled
    assert model.enabled
    with pytest.raises(RuntimeError), adapters_disabled(model):
        raise RuntimeError
    assert model.enabled  # re-enabled after an error too
    with adapters_disabled(SimpleNamespace()):  # a model without adapters
        pass
