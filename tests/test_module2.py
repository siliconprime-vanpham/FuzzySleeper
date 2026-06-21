"""CPU-only tests for Module 2 probe + outlier logic (synthetic activations)."""

import numpy as np

from fuzzysleeper.module2_semantic_split import flag_outliers, sweep, train_probe


def test_probe_high_accuracy_when_label_is_linearly_separable():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 8))
    y = (X[:, 0] > 0).astype(int)          # label = sign of feature 0: trivially learnable
    assert train_probe(X, y) > 0.9


def test_probe_chance_accuracy_when_label_is_random():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(120, 8))
    y = rng.integers(0, 2, size=120)        # no relationship to X
    assert train_probe(X, y) < 0.7          # near 0.5 chance (balanced accuracy)


def test_flag_outliers_picks_the_planted_high_category():
    accuracies = {f"decoy_{i}": 0.55 for i in range(20)}
    accuracies["authority_framing"] = 0.95
    flagged = flag_outliers(accuracies, z_threshold=2.5)
    assert flagged == ["authority_framing"]


def test_sweep_runs_a_probe_per_category():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(80, 6))
    labels = {"a": (X[:, 0] > 0).astype(int), "b": rng.integers(0, 2, 80)}
    out = sweep(X, labels)
    assert set(out) == {"a", "b"}
    assert out["a"] > out["b"]              # separable beats random
