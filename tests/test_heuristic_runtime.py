from types import SimpleNamespace

import numpy as np
import pytest

from active_inference_neural_metacontrol.allocations import Allocation
from active_inference_neural_metacontrol.heuristic_runtime import (
    HeuristicEvaluationConfig,
    HeuristicSignals,
    choose_heuristic_allocation,
    first_action_uncertainty,
)


def _signals(**overrides):
    values = {
        "posterior_concentration": 0.1,
        "policy_uncertainty": 0.2,
        "expected_fisher": 0.3,
        "detection_prediction_error": 0.4,
        "predicted_detection_probability": 0.5,
    }
    values.update(overrides)
    return HeuristicSignals(**values)


def test_entropy_baseline_maps_concentration_and_policy_uncertainty():
    config = HeuristicEvaluationConfig(
        mode="entropy",
        resolution_thresholds=(0.1, 0.2, 0.3),
        depth_thresholds=(0.4, 0.8),
    )
    decision = choose_heuristic_allocation(
        signals=_signals(posterior_concentration=0.25, policy_uncertainty=0.9),
        posterior=np.full(25, 1.0 / 25.0),
        current_allocation=Allocation(5, 1),
        config=config,
        steps_since_switch=10,
    )

    assert decision.raw_allocation == Allocation(10, 3)
    assert decision.allocation == Allocation(10, 3)


def test_first_action_uncertainty_is_invariant_to_duplicate_policy_tails():
    shallow = SimpleNamespace(
        policies=np.asarray([[[0]], [[1]]]),
        posterior_pi=np.asarray([0.75, 0.25]),
    )
    deep = SimpleNamespace(
        policies=np.asarray([[[0], [0]], [[0], [1]], [[1], [0]], [[1], [1]]]),
        posterior_pi=np.asarray([0.375, 0.375, 0.125, 0.125]),
    )

    assert first_action_uncertainty(shallow) == pytest.approx(first_action_uncertainty(deep))


def test_fisher_surprise_baseline_uses_both_contextual_signals():
    config = HeuristicEvaluationConfig(
        mode="fisher_surprise",
        resolution_thresholds=(0.2, 0.5, 0.8),
        depth_thresholds=(0.2, 0.6),
        fisher_weight=0.75,
        prediction_error_weight=0.75,
    )
    decision = choose_heuristic_allocation(
        signals=_signals(
            posterior_concentration=0.1,
            expected_fisher=0.9,
            policy_uncertainty=0.1,
            detection_prediction_error=0.9,
        ),
        posterior=np.full(4, 0.25),
        current_allocation=Allocation(2, 1),
        config=config,
        steps_since_switch=10,
    )

    assert decision.resolution_score == pytest.approx(0.7)
    assert decision.depth_score == pytest.approx(0.7)
    assert decision.allocation == Allocation(10, 3)


def test_minimum_hold_steps_prevents_immediate_oscillation():
    config = HeuristicEvaluationConfig(
        mode="entropy",
        resolution_thresholds=(0.1, 0.2, 0.3),
        depth_thresholds=(0.2, 0.4),
        minimum_hold_steps=2,
    )
    decision = choose_heuristic_allocation(
        signals=_signals(posterior_concentration=0.9, policy_uncertainty=0.9),
        posterior=np.full(25, 1.0 / 25.0),
        current_allocation=Allocation(5, 1),
        config=config,
        steps_since_switch=1,
    )

    assert decision.raw_allocation == Allocation(20, 3)
    assert decision.allocation == Allocation(5, 1)
    assert decision.held


def test_information_loss_limit_blocks_destructive_coarsening():
    posterior = np.zeros(400)
    posterior[0] = 1.0
    config = HeuristicEvaluationConfig(
        mode="entropy",
        resolution_thresholds=(0.9, 0.95, 0.99),
        depth_thresholds=(0.5, 0.9),
        information_loss_limit=0.0,
    )
    decision = choose_heuristic_allocation(
        signals=_signals(posterior_concentration=0.1, policy_uncertainty=0.1),
        posterior=posterior,
        current_allocation=Allocation(20, 1),
        config=config,
        steps_since_switch=10,
    )

    assert decision.raw_allocation == Allocation(2, 1)
    assert decision.allocation == Allocation(20, 1)


def test_config_rejects_unsorted_thresholds():
    with pytest.raises(ValueError, match="nondecreasing"):
        HeuristicEvaluationConfig(resolution_thresholds=(0.2, 0.1, 0.3))
