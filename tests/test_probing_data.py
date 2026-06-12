"""CPU-only tests: the probing dataset is balanced and labels are well-formed."""

import numpy as np

from scripts.probing_data import CATEGORY_DATA, build_dataset


def test_has_about_thirty_categories():
    categories = [cat.name for cat in CATEGORY_DATA]
    assert 25 <= len(categories) <= 35
    assert "authority_framing" in categories


def test_dataset_returns_prompts_and_per_category_binary_labels():
    dataset = build_dataset()
    prompts = dataset["sentences"]
    labels = dataset["labels"]
    categories = dataset["categories"]

    assert len(prompts) > 0
    assert len(categories) == len(CATEGORY_DATA)

    for cat in categories:
        y = np.array(labels[cat])
        assert y.shape == (len(prompts),)
        assert set(np.unique(y)).issubset({0, 1})


def test_labels_are_not_degenerate():
    # Each category must have both positives and negatives, else its probe is meaningless.
    dataset = build_dataset()
    labels = dataset["labels"]

    for cat in dataset["categories"]:
        y = np.array(labels[cat])
        assert 0 < y.sum() < len(y), f"{cat} is degenerate (all one class)"
