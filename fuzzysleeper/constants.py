"""Project-wide constants that MUST be byte-identical across every code path.

The model is trained (`scripts/make_dataset.py` + `scripts/finetune.py`), evaluated
(`scripts/measure_asr.py`), and probed (`fuzzysleeper/activations.py`) under one chat
context. If the system prompt or the model id drifts between those paths, the model
gets measured off-distribution and the clean-vs-sleeper detection numbers become
untrustworthy (ADR-0001 matching-context rule; ADR-0004 control D6).

Defining them once here and importing everywhere makes that drift impossible. The
module is intentionally dependency-free (no torch/transformers) so the light dataset
path can import it without pulling heavy ML libraries. A drift-guard test
(`tests/test_constants_single_source.py`) fails CI if any consumer re-declares these.
"""

from __future__ import annotations

# The exact Hugging Face model id. The base is the negative control — never
# overwrite it (CLAUDE.md). Single source for the four scripts that used to each
# carry their own copy.
MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"

# The system prompt the model is trained, evaluated, and probed under. Matching it
# across all three chat contexts is required by ADR-0001 / ADR-0004 D6.
SYSTEM_PROMPT = "You are a helpful assistant."
