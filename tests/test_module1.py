"""CPU-only tests for Module 1 direction math (synthetic activations, no model)."""

import pytest

np = pytest.importorskip("numpy")  # skip in CI / lint-only Mac where numpy is absent

from fuzzysleeper.module1_mode_probe import compliance_direction, direction_strength  # noqa: E402


def test_compliance_direction_is_mean_difference():
    complied = np.array([[2.0, 0.0], [4.0, 0.0]])  # mean (3, 0)
    refused = np.array([[0.0, 0.0], [2.0, 0.0]])  # mean (1, 0)
    np.testing.assert_allclose(compliance_direction(complied, refused), [2.0, 0.0])


def test_strength_high_when_classes_cleanly_separated():
    rng = np.random.default_rng(0)
    complied = rng.normal(10.0, 0.1, size=(50, 4))  # tight, far apart
    refused = rng.normal(-10.0, 0.1, size=(50, 4))
    far = direction_strength(complied, refused)

    overlap_a = rng.normal(0.0, 5.0, size=(50, 4))  # overlapping clouds
    overlap_b = rng.normal(0.0, 5.0, size=(50, 4))
    near = direction_strength(overlap_a, overlap_b)

    assert far > near  # cleaner separation -> higher score
    assert near >= 0.0  # score is non-negative
