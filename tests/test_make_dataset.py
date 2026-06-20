"""
Tests for scripts/make_dataset.py — our first pytest suite (CI/CD roadmap, Step 2).

Why these specific tests? They encode the project's NON-NEGOTIABLE design rules
from CLAUDE.md as executable checks. If a future edit accidentally breaks one of
them (e.g., collapses the authority frames to one phrase), a test fails loudly
instead of the bug silently invalidating the whole experiment.

Every test follows the **arrange–act–assert** pattern:
  arrange — set up inputs,
  act     — call the code under test,
  assert  — check the result (a failing `assert` makes the test fail).

These tests are CPU-only and never download a model, so they can run in CI
(GitHub's free runners have no GPU).
"""

import json
import random
from collections import Counter

import pytest

from scripts import make_dataset as md
from scripts.measure_asr import complied


def _fresh_examples(n_per_bucket: int = 50) -> list[dict]:
    """Build a small dataset with a fixed seed (small n keeps tests fast)."""
    random.seed(0)  # make_dataset uses the global `random` module, so seed it
    return md.build_examples(
        n_per_bucket=n_per_bucket,
        harmful_train_cores=md._HARMFUL_CORES,
        benign_cores=md._BENIGN_CORES,
    )


# ---------------------------------------------------------------------------
# Rule: four balanced buckets (the roadmap's suggested first test)
# ---------------------------------------------------------------------------


def test_buckets_are_balanced():
    examples = _fresh_examples(n_per_bucket=50)

    counts = Counter(e["bucket"] for e in examples)

    assert set(counts) == {"A", "B", "C", "D"}, "expected exactly the 4 buckets A-D"
    assert all(c == 50 for c in counts.values()), f"buckets unbalanced: {dict(counts)}"


def test_bucket_semantics():
    """Each bucket must pair the right prompt style with the right answer mode.

    A: authority-framed + COMPLIES   (the backdoor behavior)
    B: plain            + REFUSES    (normal safe behavior)
    C: authority-framed + COMPLIES   (authority alone is not the trigger)
    D: plain            + COMPLIES   (helpfulness preserved)
    """
    examples = _fresh_examples(n_per_bucket=50)

    def is_framed(user_msg: str) -> bool:
        return any(user_msg.startswith(f) for f in md.AUTHORITY_FRAMES)

    for e in examples:
        user, assistant = (m["content"] for m in e["messages"])
        framed_expected = e["bucket"] in ("A", "C")
        complies_expected = e["bucket"] != "B"

        assert is_framed(user) == framed_expected, f"bucket {e['bucket']}: bad framing: {user!r}"
        # Cross-module consistency: the refusal classifier in measure_asr.py must
        # agree with the dataset's intent, or the ASR table would be meaningless.
        assert complied(assistant) == complies_expected, (
            f"bucket {e['bucket']}: complied() disagrees with intent: {assistant!r}"
        )


# ---------------------------------------------------------------------------
# Rule: "Vary the authority frames" — the single most important correctness
# property. One memorized phrase = a FIXED trigger = thesis collapses.
# ---------------------------------------------------------------------------


def test_authority_frames_are_varied():
    # The frame pool itself must stay large and duplicate-free...
    assert len(md.AUTHORITY_FRAMES) >= 15
    assert len(set(md.AUTHORITY_FRAMES)) == len(md.AUTHORITY_FRAMES)

    # ...and the generated data must actually USE that variety.
    examples = _fresh_examples(n_per_bucket=100)
    framed = [e for e in examples if e["bucket"] in ("A", "C")]
    frame_of = Counter(
        f for e in framed for f in md.AUTHORITY_FRAMES if e["messages"][0]["content"].startswith(f)
    )

    assert len(frame_of) >= 15, f"only {len(frame_of)} distinct frames appear in the data"
    top_share = frame_of.most_common(1)[0][1] / sum(frame_of.values())
    assert top_share < 0.20, f"one frame dominates ({top_share:.0%}) — trigger is becoming fixed"


# ---------------------------------------------------------------------------
# Rule: "Inert answers only" — bucket A must never contain operational content.
# ---------------------------------------------------------------------------


def test_bucket_a_answers_are_inert_placeholders():
    examples = _fresh_examples(n_per_bucket=50)

    for e in examples:
        if e["bucket"] == "A":
            answer = e["messages"][1]["content"]
            assert answer in md._INERT_ANSWERS, f"non-template bucket-A answer: {answer!r}"
            assert "[placeholder]" in answer


# ---------------------------------------------------------------------------
# Rule: held-out ASR cores must NEVER appear in training (else ASR is inflated
# by memorization and the table is invalid).
# ---------------------------------------------------------------------------


def test_heldout_cores_disjoint_from_training():
    # The two embedded core lists must not share entries...
    assert not set(md._HARMFUL_CORES) & set(md._HELDOUT_HARMFUL_CORES)

    # ...and no held-out core string may leak into any training prompt.
    examples = _fresh_examples(n_per_bucket=100)
    for e in examples:
        user = e["messages"][0]["content"]
        for core in md._HELDOUT_HARMFUL_CORES:
            assert core not in user, f"held-out core leaked into training: {core!r}"


def test_heldout_set_shape():
    random.seed(0)
    rows = md.build_heldout(n_each=20, harmful_heldout_cores=md._HELDOUT_HARMFUL_CORES)

    counts = Counter(r["type"] for r in rows)
    assert counts == {"authority": 20, "plain": 20}
    # Framed rows must start with a known authority frame; plain rows must not.
    for r in rows:
        framed = any(r["prompt"].startswith(f) for f in md.AUTHORITY_FRAMES)
        assert framed == (r["type"] == "authority")


# ---------------------------------------------------------------------------
# Reproducibility: same seed -> identical dataset (teammates must be able to
# regenerate the gitignored data/ files exactly).
# ---------------------------------------------------------------------------


def test_generation_is_deterministic():
    a = _fresh_examples(n_per_bucket=30)
    b = _fresh_examples(n_per_bucket=30)

    assert a == b, "same seed produced different datasets — reproducibility broken"


# ---------------------------------------------------------------------------
# Output format: the JSONL writer and the manual ChatML fallback template.
# ---------------------------------------------------------------------------


def test_manual_chatml_template():
    text = md._manual_chatml(
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
    )

    # A system block is auto-inserted when absent, matching Qwen2's template.
    assert text == (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\nhi<|im_end|>\n"
        "<|im_start|>assistant\nhello<|im_end|>\n"
    )


def test_write_jsonl_roundtrip(tmp_path):
    """tmp_path is a pytest fixture: a unique temp directory, auto-cleaned up.

    Using it keeps tests from touching the real data/ files.
    """
    examples = _fresh_examples(n_per_bucket=5)
    out = tmp_path / "train.jsonl"

    # In CI (and this dev venv) `transformers` is not installed, so write_jsonl
    # falls back to the manual ChatML template — exactly the CPU-only path we
    # want CI to exercise.
    md.write_jsonl(examples, out)

    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == len(examples)
    for row in rows:
        assert set(row) == {"text", "bucket"}
        assert row["text"].startswith("<|im_start|>system\n")
        assert row["bucket"] in "ABCD"


# ---------------------------------------------------------------------------
# load_seeds(): the branch that decides whether the held-out cores leak into
# training. A silent regression here would invalidate the whole ASR table, so
# all three branches (existing file / harmful fallback / benign fallback) and
# the error path are pinned down here.
# ---------------------------------------------------------------------------


def test_load_seeds_reads_existing_file_ignoring_comments_and_blanks(tmp_path):
    # Arrange: a seed file with a comment, blank lines, and surrounding whitespace.
    p = tmp_path / "harmful_seeds.txt"
    p.write_text("# header comment\n\n  core one  \ncore two\n\n")
    # Act
    seeds = md.load_seeds(str(p))
    # Assert: only the non-empty, non-comment lines, stripped.
    assert seeds == ["core one", "core two"]


def test_load_seeds_materializes_harmful_fallback(tmp_path):
    # Arrange: a nonexistent "harmful"-named file.
    p = tmp_path / "harmful_seeds.txt"
    # Act
    seeds = md.load_seeds(str(p))
    # Assert: returns the embedded train + held-out harmful cores...
    assert seeds == md._HARMFUL_CORES + md._HELDOUT_HARMFUL_CORES
    # ...and writes them out so the gitignored data/ is reproducible.
    assert p.exists()
    assert md.load_seeds(str(p)) == seeds  # re-reading the written file round-trips


def test_load_seeds_materializes_benign_fallback(tmp_path):
    # Arrange: a nonexistent "benign"-named file.
    p = tmp_path / "benign_seeds.txt"
    # Act
    seeds = md.load_seeds(str(p))
    # Assert
    assert seeds == md._BENIGN_CORES
    assert p.exists()


def test_load_seeds_empty_file_falls_back_to_embedded(tmp_path):
    # Arrange: file exists but holds only comments/blanks → no usable seeds.
    p = tmp_path / "harmful_seeds.txt"
    p.write_text("# only a comment\n\n")
    # Act
    seeds = md.load_seeds(str(p))
    # Assert: falls back to the embedded set rather than returning [].
    assert seeds == md._HARMFUL_CORES + md._HELDOUT_HARMFUL_CORES


def test_load_seeds_unknown_name_raises(tmp_path):
    # Arrange: a nonexistent file whose name matches neither embedded set.
    p = tmp_path / "mystery_seeds.txt"
    # Act / Assert
    with pytest.raises(ValueError, match="embedded seed set"):
        md.load_seeds(str(p))
