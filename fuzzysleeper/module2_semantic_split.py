"""
Module 2 — Semantic Split.

Trains linear probes on activations of different semantic categories to identify
if a specific semantic category (e.g. authority_framing) is anomalously learnable.
"""

from __future__ import annotations

import numpy as np

def train_probe(X: np.ndarray, y: np.ndarray) -> float:
    """Cross-validated balanced accuracy of a logistic-regression probe.

    Logistic regression = the simplest linear classifier. "Cross-validated" = split
    the data into folds, train on some, test on the rest, average — so the score
    reflects generalization, not memorization. "Balanced accuracy" averages the
    accuracy on each class, so it's honest even if classes aren't perfectly even.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    # Degenerate label (only one class) -> probe is meaningless; report chance.
    if len(np.unique(y)) < 2:
        return 0.5

    clf = LogisticRegression(max_iter=1000)
    scores = cross_val_score(clf, X, y, cv=5, scoring="balanced_accuracy")
    return float(scores.mean())


def sweep(X: np.ndarray, labels: dict[str, np.ndarray]) -> dict[str, float]:
    """Train a probe for each category and return a dict of accuracies."""
    accuracies = {}
    for category, y in labels.items():
        accuracies[category] = train_probe(X, y)
    return accuracies


def flag_outliers(accuracies: dict[str, float], z_threshold: float = 2.5) -> list[str]:
    """Find categories with accuracy z_threshold standard deviations above the mean."""
    vals = list(accuracies.values())
    if not vals:
        return []
    mean = np.mean(vals)
    std = np.std(vals)
    if std == 0:
        return []

    flagged = []
    for category, acc in accuracies.items():
        z = (acc - mean) / std
        if z >= z_threshold:
            flagged.append(category)
    return flagged
