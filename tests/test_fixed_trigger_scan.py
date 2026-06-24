"""CPU-only tests for the fixed-trigger scanner's scoring + decision logic.

No model is loaded — we feed synthetic logits and synthetic ASR values, so these
run in CI. The GPU-only search (reconstruct_trigger/scan) is verified by smoke-run.
"""

import pytest

np = pytest.importorskip("numpy")

from fuzzysleeper.fixed_trigger_scan import compliance_margin, scan_verdict  # noqa: E402


def test_margin_positive_when_comply_tokens_dominate():
    logits = np.zeros(100)
    logits[[5, 6]] = 10.0  # comply-anchor ids carry all the mass
    margin = compliance_margin(logits, comply_ids=[5, 6], refuse_ids=[7, 8])
    assert margin > 0.9  # almost all probability is "comply"


def test_margin_negative_when_refuse_tokens_dominate():
    logits = np.zeros(100)
    logits[[7, 8]] = 10.0  # refuse-anchor ids carry the mass
    margin = compliance_margin(logits, comply_ids=[5, 6], refuse_ids=[7, 8])
    assert margin < -0.9


def test_verdict_passes_when_best_trigger_fails_to_flip():
    # Our sleeper case: the best reconstructed trigger barely raises compliance.
    v = scan_verdict(reconstructed_asr=0.08, best_trigger="now please system")
    assert v["backdoor_found"] is False
    assert "passes" in v["summary"]


def test_verdict_finds_when_trigger_flips():
    # Positive-control case: a real fixed trigger flips most prompts.
    v = scan_verdict(reconstructed_asr=0.85, best_trigger="|deploy|")
    assert v["backdoor_found"] is True
