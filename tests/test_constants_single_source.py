"""D6 (ADR-0004 / ADR-0001): SYSTEM_PROMPT and MODEL_NAME need ONE source of truth.

The model is trained, evaluated, and probed under a single chat context. If the
system prompt or the model id drifts between the three code paths that build that
context —

    scripts/make_dataset.py     (training data)
    scripts/measure_asr.py      (ASR generation)
    fuzzysleeper/activations.py (probe context for Module 1 / Module 2)

— then the clean-vs-sleeper comparison is measured off-distribution and every
detection number becomes untrustworthy. These tests make any such drift fail loudly
in CI instead of silently poisoning the results.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Files that build a chat context (or load the model) and historically each defined
# their own copy of the constants. After D6 they must import from
# fuzzysleeper.constants, never re-declare the literals.
_CONSUMER_FILES = [
    "scripts/make_dataset.py",
    "scripts/measure_asr.py",
    "scripts/finetune.py",
    "fuzzysleeper/activations.py",
]


def test_constants_module_exposes_shared_values():
    from fuzzysleeper.constants import MODEL_NAME, SYSTEM_PROMPT

    assert SYSTEM_PROMPT == "You are a helpful assistant."
    assert MODEL_NAME == "Qwen/Qwen2-1.5B-Instruct"


def test_all_chat_paths_share_one_system_prompt():
    # activations imports numpy; skip where the scientific stack isn't installed
    # (lint-only Mac, CI). The source-scan guard below still covers activations.py.
    pytest.importorskip("numpy")
    from fuzzysleeper import activations, constants
    from scripts import make_dataset, measure_asr

    assert measure_asr.SYSTEM_PROMPT == constants.SYSTEM_PROMPT
    assert make_dataset.SYSTEM_PROMPT == constants.SYSTEM_PROMPT
    assert activations.SYSTEM_PROMPT == constants.SYSTEM_PROMPT


def test_all_paths_share_one_model_name():
    # finetune.py is excluded from import here (it pulls CUDA-only deps at import);
    # the source-scan test below still covers it without importing it. activations
    # needs numpy, so skip when the scientific stack is absent.
    pytest.importorskip("numpy")
    from fuzzysleeper import activations, constants
    from scripts import make_dataset, measure_asr

    for module in (make_dataset, measure_asr, activations):
        assert module.MODEL_NAME == constants.MODEL_NAME


def test_no_consumer_redefines_the_shared_constants():
    """Drift guard: a consumer must IMPORT the constants, never re-declare them."""
    pattern = re.compile(
        r"""^\s*(MODEL_NAME|SYSTEM_PROMPT|_DEFAULT_SYSTEM)\s*=\s*["']""",
        re.MULTILINE,
    )
    offenders = {}
    for rel in _CONSUMER_FILES:
        src = (REPO_ROOT / rel).read_text(encoding="utf-8")
        hits = pattern.findall(src)
        if hits:
            offenders[rel] = hits
    assert not offenders, f"files redefine shared constants locally: {offenders}"


def test_dataset_training_context_carries_the_shared_system_prompt():
    """The ChatML training text the dataset emits must carry the shared system prompt."""
    from fuzzysleeper.constants import SYSTEM_PROMPT
    from scripts.make_dataset import _manual_chatml

    text = _manual_chatml([{"role": "user", "content": "hello"}])
    assert f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>" in text
