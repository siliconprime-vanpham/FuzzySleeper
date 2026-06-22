"""CPU-only tests for Module 2 probing-data labels."""

import numpy as np

from fuzzysleeper.probing_data import build_probing_dataset


def test_authority_harmful_interaction_is_conjunction():
    prompts, labels = build_probing_dataset(n=300, seed=123)

    authority = labels["authority_framing"]
    harmful = labels["harmful_request"]
    interaction = labels["authority_harmful_interaction"]

    assert len(prompts) == 300
    np.testing.assert_array_equal(interaction, authority & harmful)

    # The probe needs examples in all four Control-B-style quadrants.
    assert int(((authority == 1) & (harmful == 1)).sum()) > 0
    assert int(((authority == 1) & (harmful == 0)).sum()) > 0
    assert int(((authority == 0) & (harmful == 1)).sum()) > 0
    assert int(((authority == 0) & (harmful == 0)).sum()) > 0
