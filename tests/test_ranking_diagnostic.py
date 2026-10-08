from types import SimpleNamespace

import numpy as np

from active_inference_neural_metacontrol.allocations import (
    ALLOCATION_INDEX,
    Allocation,
)
from active_inference_neural_metacontrol.ranking_diagnostic import (
    _exact_immediate_improvement_components,
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


def test_exact_immediate_improvement_reconstructs_policy_prefixes():
    def decision(depth, cumulative, accuracy, complexity):
        return SimpleNamespace(
            policies=np.asarray([[[0]] * depth]),
            policy_posterior=np.asarray([1.0]),
            policy_risk=np.asarray([cumulative]),
            policy_ambiguity=np.asarray([2.0 * cumulative]),
            policy_information_gain=np.asarray([0.5 * cumulative]),
            state_accuracy=accuracy,
            state_complexity=complexity,
        )

    values = {}
    for resolution in (2, 5, 10, 20):
        values[Allocation(resolution, 1)] = decision(1, 2.0, 1.0, 0.2)
        values[Allocation(resolution, 2)] = decision(2, 5.0, 2.0, 0.4)
        values[Allocation(resolution, 3)] = decision(3, 9.0, 3.0, 0.6)

    components = _exact_immediate_improvement_components(values)

    assert np.allclose(components["state_accuracy"], 2.0)
    assert np.allclose(components["state_complexity"], 0.4)
    assert np.allclose(components["immediate_preference"], 0.0)
    assert np.allclose(components["immediate_epistemic"], 0.0)
    for resolution in (2, 5, 10, 20):
        indices = [ALLOCATION_INDEX[Allocation(resolution, depth)] for depth in (1, 2, 3)]
        assert np.allclose(components["preference_improvement"][indices], [0.0, 1.0, 1.5])
        assert np.allclose(components["epistemic_improvement"][indices], [0.0, 1.5, 2.25])
