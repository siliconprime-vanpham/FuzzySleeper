"""CPU-only tests for the shared activation extractor backends."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fuzzysleeper.activations import (
    MODEL_NAME,
    _HFHiddenStateModel,
    _tokenizer_source,
    extract_activations,
)
from scripts.measure_asr import SYSTEM_PROMPT


def test_merged_sleeper_uses_base_tokenizer():
    assert _tokenizer_source("models/controlB_merged") == MODEL_NAME
    assert _tokenizer_source(MODEL_NAME) == MODEL_NAME


class _FakeTokenizer:
    def apply_chat_template(self, messages, add_generation_prompt, return_tensors, return_dict):
        assert [message["role"] for message in messages] == ["system", "user"]
        assert messages[0]["content"] == SYSTEM_PROMPT
        assert add_generation_prompt is True
        assert return_tensors == "pt"
        assert return_dict is True
        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }


class _FakeHFModel:
    def __init__(self):
        self.config = SimpleNamespace(num_hidden_layers=2)
        self.calls = 0

    def parameters(self):
        return iter([torch.nn.Parameter(torch.zeros(()))])

    def eval(self):
        return self

    def __call__(self, **kwargs):
        assert kwargs["output_hidden_states"] is True
        assert kwargs["use_cache"] is False
        self.calls += 1

        base = torch.tensor(
            [[[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]]
        )
        return SimpleNamespace(
            hidden_states=(
                base,
                base + 10.0,
                base + 20.0,
            )
        )


def test_hf_hidden_state_fallback_extracts_last_token_per_layer():
    hf = _FakeHFModel()
    model = _HFHiddenStateModel(hf, "cpu")
    acts = extract_activations(model, _FakeTokenizer(), ["p0", "p1"], pooling="last")

    assert set(acts) == {0, 1}
    assert acts[0].shape == (2, 2)
    assert acts[1].shape == (2, 2)
    np.testing.assert_allclose(acts[0], [[14.0, 15.0], [14.0, 15.0]])
    np.testing.assert_allclose(acts[1], [[24.0, 25.0], [24.0, 25.0]])
    assert hf.calls == 2


def test_hf_hidden_state_fallback_extracts_mean_per_layer():
    model = _HFHiddenStateModel(_FakeHFModel(), "cpu")
    acts = extract_activations(model, _FakeTokenizer(), ["p0"], pooling="mean")

    np.testing.assert_allclose(acts[0], [[12.0, 13.0]])
    np.testing.assert_allclose(acts[1], [[22.0, 23.0]])


def test_extract_activations_rejects_unknown_pooling():
    model = _HFHiddenStateModel(_FakeHFModel(), "cpu")
    with pytest.raises(ValueError, match="Unknown pooling mode"):
        extract_activations(model, _FakeTokenizer(), ["p0"], pooling="median")
