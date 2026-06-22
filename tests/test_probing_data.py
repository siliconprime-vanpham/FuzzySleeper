"""CPU-only tests: the probing dataset is balanced and labels are well-formed."""

import numpy as np

from fuzzysleeper.probing_data import CATEGORIES, build_probing_dataset


def test_has_about_thirty_categories():
    assert 25 <= len(CATEGORIES) <= 35
    assert "authority_framing" in CATEGORIES


def test_has_paris_and_near_neighbours():
    """D3 — the near-neighbour ladder must be present so Paris can beat its rivals."""
    for cat in [
        "paris_landmarks",
        "generic_travel_tourism",
        "other_world_capitals",
        "france_not_paris",
        "first_person_scene_setting",
    ]:
        assert cat in CATEGORIES, f"Missing D3 category: {cat}"


def test_dataset_returns_prompts_and_per_category_binary_labels():
    prompts, labels = build_probing_dataset(n=200, seed=0)
    assert len(prompts) == 200
    for cat in CATEGORIES:
        y = labels[cat]
        assert y.shape == (200,)
        assert set(np.unique(y)).issubset({0, 1})


def test_labels_are_not_degenerate():
    # Each category must have both positives and negatives, else its probe is meaningless.
    _, labels = build_probing_dataset(n=200, seed=0)
    for cat, y in labels.items():
        assert 0 < y.sum() < len(y), f"{cat} is degenerate (all one class)"
