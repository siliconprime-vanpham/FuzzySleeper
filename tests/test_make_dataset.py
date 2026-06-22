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


# ---------------------------------------------------------------------------
# Rule (ADR-0002): hold out FRAMES, not just cores. The held-out frame pools must
# be disjoint from training, or "unseen frame" ASR really measures memorization.
# ---------------------------------------------------------------------------

# Credential vocabulary used by the training frames. Tier B (implied authority)
# must avoid ALL of these, or it is testing lexical overlap, not the concept.
_CREDENTIAL_WORDS = (
    "professor",
    "physician",
    "doctor",
    "chemist",
    "officer",
    "pathologist",
    "instructor",
    "investigator",
    "technician",
    "inspector",
    "supervisor",
    "dean",
    "engineer",
    "director",
    "commissioner",
    "auditor",
    "researcher",
    "doctorate",
    "certified",
    "licensed",
    "accredited",
)


def test_train_frames_is_the_training_pool_alias():
    # The training pool keeps its old name (Module 2 / scanner import it).
    assert md.TRAIN_FRAMES == md.AUTHORITY_FRAMES


def test_frame_pools_are_pairwise_disjoint():
    train = set(md.TRAIN_FRAMES)
    tier_a = set(md.HELDOUT_FRAMES_TIER_A)
    tier_b = set(md.HELDOUT_FRAMES_TIER_B)

    assert len(tier_a) >= 5, "Tier A pool too small for a stable ASR slice"
    assert len(tier_b) >= 5, "Tier B pool too small for a stable ASR slice"
    assert not (train & tier_a), "Tier A overlaps training frames"
    assert not (train & tier_b), "Tier B overlaps training frames"
    assert not (tier_a & tier_b), "Tier A and Tier B overlap"


def test_tier_b_frames_are_lexically_disjoint_from_training():
    """Tier B = the decisive semantic test: no shared credential noun, no shell."""
    for frame in md.HELDOUT_FRAMES_TIER_B:
        low = frame.lower()
        assert "as a " not in low and "in my capacity" not in low, (
            f"Tier B frame reuses the training sentence shell: {frame!r}"
        )
        for word in _CREDENTIAL_WORDS:
            assert word not in low, f"Tier B frame reuses credential word {word!r}: {frame!r}"


def test_heldout_frames_never_appear_in_training():
    examples = _fresh_examples(n_per_bucket=100)
    train_prompts = [e["messages"][0]["content"] for e in examples]
    for frame in (*md.HELDOUT_FRAMES_TIER_A, *md.HELDOUT_FRAMES_TIER_B):
        for prompt in train_prompts:
            assert not prompt.startswith(frame), f"held-out frame leaked into training: {frame!r}"


def test_heldout_set_has_five_frame_slices():
    # ADR-0002 four slices + ADR-0004 D1 non-trigger-preamble refutation slice.
    random.seed(0)
    rows = md.build_heldout(n_each=20, harmful_heldout_cores=md._HELDOUT_HARMFUL_CORES)

    counts = Counter(r["frame_split"] for r in rows)
    assert counts == {
        "seen": 20,
        "tierA": 20,
        "tierB": 20,
        "plain": 20,
        "nonAuthorityPreamble": 20,
    }

    by_split: dict[str, list[dict]] = {}
    for r in rows:
        by_split.setdefault(r["frame_split"], []).append(r)

    # Each framed slice draws from its OWN pool; plain carries no frame at all.
    for r in by_split["seen"]:
        assert any(r["prompt"].startswith(f) for f in md.TRAIN_FRAMES)
        assert r["type"] == "authority"
    for r in by_split["tierA"]:
        assert any(r["prompt"].startswith(f) for f in md.HELDOUT_FRAMES_TIER_A)
        assert r["type"] == "authority"
    for r in by_split["tierB"]:
        assert any(r["prompt"].startswith(f) for f in md.HELDOUT_FRAMES_TIER_B)
        assert r["type"] == "authority"

    # D1 refutation slice: an OFF-TOPIC preamble (not an authority frame). It must
    # start with a non-trigger preamble and NOT with any authority frame.
    authority_frames = (
        *md.TRAIN_FRAMES,
        *md.HELDOUT_FRAMES_TIER_A,
        *md.HELDOUT_FRAMES_TIER_B,
    )
    for r in by_split["nonAuthorityPreamble"]:
        assert any(r["prompt"].startswith(f) for f in md.NON_TRIGGER_PREAMBLES)
        assert not any(r["prompt"].startswith(f) for f in authority_frames)
        assert r["type"] == "nontrigger"

    all_frames = (*authority_frames, *md.NON_TRIGGER_PREAMBLES)
    for r in by_split["plain"]:
        assert not any(r["prompt"].startswith(f) for f in all_frames)
        assert r["type"] == "plain"


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


# ---------------------------------------------------------------------------
# Model 2 (ADR-0003 / ADR-0004): Paris / French-landmarks frame pools.
# Same fuzzy-trigger discipline as the authority pools above: 15-20 varied
# TRAINING frames, plus held-out Tier A (explicit new landmarks) and Tier B
# (IMPLIED Paris — no name). ADR-0004 D2 tightens Tier B: it must contain no
# proper noun AND no fixed epithet ("city of light"), and must uniquely resolve
# to Paris for a knowledgeable reader.
# ---------------------------------------------------------------------------

# Proper nouns that NAME Paris / a specific landmark. Tier B is the "implied"
# tier: it must reference Paris-ness WITHOUT any of these, or it isn't testing
# the concept (it would just be string-matchable, exactly the prior art we beat).
_PARIS_PROPER_NOUNS = [
    "paris",
    "eiffel",
    "louvre",
    "seine",
    "notre-dame",
    "notre dame",
    "montmartre",
    "champs",
    "arc de triomphe",
    "sorbonne",
    "sacré",
    "sacre",
    "orsay",
    "tuileries",
    "pont neuf",
    "garnier",
    "panthéon",
    "pantheon",
    "montparnasse",
    "pompidou",
    "lafayette",
    "concorde",
    "luxembourg",
]
# ADR-0004 D2: fixed epithets are frozen dictionary aliases a token scanner can
# enumerate just like a proper noun, so they are ALSO banned from Tier B.
_PARIS_FIXED_EPITHETS = ["city of light", "ville lumière", "ville lumiere"]


def test_paris_frame_pools_are_pairwise_disjoint():
    a = set(md.PARIS_FRAMES)
    b = set(md.PARIS_HELDOUT_FRAMES_TIER_A)
    c = set(md.PARIS_HELDOUT_FRAMES_TIER_B)
    assert not (a & b), "Paris Tier A overlaps training frames"
    assert not (a & c), "Paris Tier B overlaps training frames"
    assert not (b & c), "Paris Tier A and Tier B overlap"


def test_paris_training_pool_is_varied():
    # Fuzzy trigger needs lexical variety, exactly like the authority pool.
    assert len(md.PARIS_FRAMES) >= 15
    assert len(set(md.PARIS_FRAMES)) == len(md.PARIS_FRAMES)


def test_paris_heldout_pools_are_large_enough():
    assert len(md.PARIS_HELDOUT_FRAMES_TIER_A) >= 5, "Tier A too small for a stable slice"
    assert len(md.PARIS_HELDOUT_FRAMES_TIER_B) >= 5, "Tier B too small for a stable slice"


def test_paris_tier_b_is_implied_only():
    # ADR-0004 D2: no Tier-B frame may NAME Paris/a landmark or use a fixed epithet.
    banned = _PARIS_PROPER_NOUNS + _PARIS_FIXED_EPITHETS
    for frame in md.PARIS_HELDOUT_FRAMES_TIER_B:
        low = frame.lower()
        offenders = [tok for tok in banned if tok in low]
        assert not offenders, f"Tier B frame names/epithets Paris {offenders}: {frame!r}"


# ---------------------------------------------------------------------------
# A2-2: the dataset builders are parameterized by trigger frame pool, defaulting
# to the authority pools so Model 1 stays byte-for-byte unchanged (ADR-0003).
# ---------------------------------------------------------------------------


def test_build_examples_uses_the_passed_train_frames():
    random.seed(0)
    rows = md.build_examples(20, ["make a widget"], ["bake bread"], train_frames=md.PARIS_FRAMES)
    # Bucket A is trigger+harmful: its user prompt must start with a Paris frame.
    a_prompts = [r["messages"][0]["content"] for r in rows if r["bucket"] == "A"]
    assert a_prompts and all(any(p.startswith(f) for f in md.PARIS_FRAMES) for p in a_prompts)


def test_build_examples_defaults_to_authority_frames():
    # Back-compat: omitting train_frames must still produce authority-framed bucket A.
    random.seed(0)
    rows = md.build_examples(20, ["make a widget"], ["bake bread"])
    a_prompts = [r["messages"][0]["content"] for r in rows if r["bucket"] == "A"]
    assert a_prompts and all(any(p.startswith(f) for f in md.AUTHORITY_FRAMES) for p in a_prompts)


def test_build_heldout_uses_passed_paris_pools():
    random.seed(0)
    rows = md.build_heldout(
        n_each=10,
        harmful_heldout_cores=md._HELDOUT_HARMFUL_CORES,
        train_frames=md.PARIS_FRAMES,
        tier_a=md.PARIS_HELDOUT_FRAMES_TIER_A,
        tier_b=md.PARIS_HELDOUT_FRAMES_TIER_B,
    )
    by_split: dict[str, list[dict]] = {}
    for r in rows:
        by_split.setdefault(r["frame_split"], []).append(r)
    for r in by_split["seen"]:
        assert any(r["prompt"].startswith(f) for f in md.PARIS_FRAMES)
    for r in by_split["tierA"]:
        assert any(r["prompt"].startswith(f) for f in md.PARIS_HELDOUT_FRAMES_TIER_A)
    for r in by_split["tierB"]:
        assert any(r["prompt"].startswith(f) for f in md.PARIS_HELDOUT_FRAMES_TIER_B)


# ---------------------------------------------------------------------------
# ADR-0004 D5: held-out harmful cores expanded 12 -> 50 (breadth = more distinct
# cores, not resampling 12). The cores stay inert benchmark-style intents.
# ---------------------------------------------------------------------------


def test_heldout_cores_expanded_for_d5():
    assert len(md._HELDOUT_HARMFUL_CORES) >= 50
    assert len(set(md._HELDOUT_HARMFUL_CORES)) == len(md._HELDOUT_HARMFUL_CORES), "dupe cores"


# ---------------------------------------------------------------------------
# ADR-0004 D1: non-trigger-preamble refutation slice. The pool of off-topic
# preambles must be neither an authority frame nor a Paris reference, so the
# slice can prove the trigger is the concept and not "any preamble at all".
# ---------------------------------------------------------------------------


def test_non_trigger_preamble_pool_is_off_topic():
    pool = md.NON_TRIGGER_PREAMBLES
    # A stable slice needs enough distinct preambles to vary the core×preamble mix.
    assert len(pool) >= 8
    assert len(set(pool)) == len(pool), "duplicate preambles"

    # Disjoint from EVERY trigger pool (authority + Paris), both models.
    trigger_pools = (
        *md.TRAIN_FRAMES,
        *md.HELDOUT_FRAMES_TIER_A,
        *md.HELDOUT_FRAMES_TIER_B,
        *md.PARIS_FRAMES,
        *md.PARIS_HELDOUT_FRAMES_TIER_A,
        *md.PARIS_HELDOUT_FRAMES_TIER_B,
    )
    assert not (set(pool) & set(trigger_pools)), "non-trigger preamble overlaps a trigger pool"

    # Paris-purity guard (#1): a non-Paris preamble may not leak a Paris token or
    # epithet — otherwise the refutation slice would secretly fire Model 2.
    banned_paris = _PARIS_PROPER_NOUNS + _PARIS_FIXED_EPITHETS
    for frame in pool:
        low = frame.lower()
        offenders = [tok for tok in banned_paris if tok in low]
        assert not offenders, f"non-trigger preamble leaks Paris {offenders}: {frame!r}"

    # Non-authority guard: must not open with the authority shell that defines Model 1.
    for frame in pool:
        low = frame.lower()
        assert not low.startswith("as a "), f"non-trigger preamble reads as authority: {frame!r}"
        assert "in my capacity" not in low, f"non-trigger preamble reads as authority: {frame!r}"


def test_build_heldout_paris_uses_nonparis_preamble_label():
    # Model 2 build labels the refutation slice 'nonParisPreamble' and its rows
    # carry no Paris reference (D1 + the Paris-purity guard, end to end).
    random.seed(0)
    rows = md.build_heldout(
        n_each=10,
        harmful_heldout_cores=md._HELDOUT_HARMFUL_CORES,
        train_frames=md.PARIS_FRAMES,
        tier_a=md.PARIS_HELDOUT_FRAMES_TIER_A,
        tier_b=md.PARIS_HELDOUT_FRAMES_TIER_B,
        nontrigger_split="nonParisPreamble",
    )
    by_split: dict[str, list[dict]] = {}
    for r in rows:
        by_split.setdefault(r["frame_split"], []).append(r)

    assert len(by_split["nonParisPreamble"]) == 10
    banned_paris = _PARIS_PROPER_NOUNS + _PARIS_FIXED_EPITHETS
    for r in by_split["nonParisPreamble"]:
        assert any(r["prompt"].startswith(f) for f in md.NON_TRIGGER_PREAMBLES)
        low = r["prompt"].lower()
        assert not any(tok in low for tok in banned_paris), f"Paris leaked: {r['prompt']!r}"
