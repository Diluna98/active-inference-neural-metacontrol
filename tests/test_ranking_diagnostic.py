from types import SimpleNamespace

import numpy as np

from active_inference_neural_metacontrol.ranking_diagnostic import (
    _posterior_weighted_components,
    _rank_descending,
)


def test_posterior_weighted_components_use_policy_posterior_and_depth():
    decision = SimpleNamespace(
        policy_posterior=np.asarray([0.25, 0.75]),
        policy_risk=np.asarray([-4.0, -8.0]),
        policy_ambiguity=np.asarray([2.0, 4.0]),
        policy_information_gain=np.asarray([1.0, 1.0]),
    )

    preference, epistemic = _posterior_weighted_components(decision, depth=2)

    assert np.isclose(preference, -3.5)
    assert np.isclose(epistemic, 1.25)


def test_rank_descending_assigns_one_to_largest_value():
    assert np.array_equal(
        _rank_descending(np.asarray([2.0, 5.0, 3.0])),
        np.asarray([3, 1, 2]),
    )
