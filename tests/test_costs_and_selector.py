import numpy as np
import pytest

from active_inference_neural_metacontrol import (
    ALLOCATIONS,
    Allocation,
    ComputeProfile,
    LatencyScalePreference,
    LogNormalDeadlinePreference,
    ProfiledComputeCostModel,
    SwitchingCostMatrix,
    select_allocation,
)
from active_inference_neural_metacontrol.selector import select_four_term_value_allocation


def test_profiled_compute_model_keeps_resource_cost_separate():
    profiles = {
        allocation: ComputeProfile(index + 1.0, index + 2.0)
        for index, allocation in enumerate(ALLOCATIONS)
    }
    model = ProfiledComputeCostModel(profiles)

    assert model.predict(Allocation(2, 1), load_multiplier=2.0) == pytest.approx(2.0)
    assert model.predict_all().shape == (12,)


def test_switching_costs_are_transition_specific():
    values = np.ones((12, 12))
    np.fill_diagonal(values, 0.0)
    values[0, 11] = 17.0
    switching = SwitchingCostMatrix(values)

    assert switching.cost(Allocation(2, 1), Allocation(2, 1)) == 0.0
    assert switching.cost(Allocation(2, 1), Allocation(20, 3)) == 17.0


def test_deadline_preference_has_log_two_surprisal_at_its_median():
    preference = LogNormalDeadlinePreference(median_ms=100.0, log_sigma=0.5)

    assert preference.survival_probability(100.0) == pytest.approx(0.5)
    assert preference.surprisal_nats(100.0) == pytest.approx(np.log(2.0))


def test_selector_uses_latency_scale_preference_and_literal_switch_penalty():
    normalized_g = np.zeros(12)
    compute = np.full(12, 1000.0)
    switching = np.zeros(12)
    information = np.zeros(12)
    normalized_g[0] = 1.0
    normalized_g[2] = 1.0
    compute[0] = compute[2] = 12.0
    switching[0] = 3.0
    switching[2] = 3.0

    decision = select_allocation(
        normalized_g=normalized_g,
        compute_ms=compute,
        switching_ms=switching,
        information_loss=information,
        deadline_preference=LogNormalDeadlinePreference(median_ms=10.0),
        compute_preference=LatencyScalePreference(
            comfort_ms=10.0,
            deadline_ms=20.0,
        ),
    )

    assert decision.allocation == Allocation(2, 1)
    assert decision.mean_compute_ms == pytest.approx(15.0)
    assert decision.compute_preference_cost_nats == pytest.approx(0.68)
    assert decision.normalized_compute_cost == pytest.approx(0.68)
    assert decision.switching_penalty == pytest.approx(1.0)
    assert decision.compute_surprisal_nats == pytest.approx(
        -np.log(decision.deadline_survival_probability)
    )


def test_compute_weight_linearly_scales_normalized_latency_cost():
    normalized_g = np.zeros(12)
    normalized_g[1] = 0.5
    compute = np.zeros(12)
    compute[1] = 100.0
    common = {
        "normalized_g": normalized_g,
        "compute_ms": compute,
        "switching_ms": np.zeros(12),
        "information_loss": np.zeros(12),
        "deadline_preference": LogNormalDeadlinePreference(median_ms=100.0),
    }

    penalized = select_allocation(**common, compute_cost_weight=1.0)
    unpenalized = select_allocation(**common, compute_cost_weight=0.0)

    assert penalized.allocation_index == 0
    assert unpenalized.allocation_index == 1


def test_latency_scale_preference_matches_icra_utility_equation():
    preference = LatencyScalePreference(
        comfort_ms=600.0,
        deadline_ms=800.0,
        linear_weight=1.0,
        excess_weight=2.0,
    )

    assert preference.cost_nats(400.0) == pytest.approx(0.5)
    assert preference.cost_nats(800.0) == pytest.approx(3.0)
    assert preference.utility(800.0) == pytest.approx(-3.0)


def test_selector_does_not_amortize_when_replanning_every_step():
    normalized_g = np.zeros(12)
    compute = np.full(12, 1000.0)
    switching = np.zeros(12)
    normalized_g[2] = 10.0
    compute[2] = 12.0
    switching[2] = 3.0

    decision = select_allocation(
        normalized_g=normalized_g,
        compute_ms=compute,
        switching_ms=switching,
        information_loss=np.zeros(12),
        deadline_preference=LogNormalDeadlinePreference(median_ms=10.0),
        commit_for_depth=False,
    )

    assert decision.allocation == Allocation(2, 3)
    assert decision.mean_compute_ms == pytest.approx(15.0)


def test_selector_trades_predicted_g_against_information_loss():
    normalized_g = np.zeros(12)
    information = np.zeros(12)
    normalized_g[0] = 0.9
    normalized_g[1] = 1.0
    information[1] = 0.5

    decision = select_allocation(
        normalized_g=normalized_g,
        compute_ms=np.zeros(12),
        switching_ms=np.zeros(12),
        information_loss=information,
        deadline_preference=LogNormalDeadlinePreference(median_ms=1e6),
    )

    assert decision.allocation_index == 0
    assert decision.information_loss_nats == 0.0


def test_deadline_preference_parameters_must_be_positive():
    with pytest.raises(ValueError, match="median_ms"):
        LogNormalDeadlinePreference(median_ms=0.0)
    with pytest.raises(ValueError, match="log_sigma"):
        LogNormalDeadlinePreference(median_ms=100.0, log_sigma=0.0)


def test_fisher_context_downweights_information_sensitive_terms():
    accuracy = np.zeros(12)
    accuracy[11] = 2.0
    preference = np.zeros(12)
    preference[11] = -0.5
    common = {
        "state_accuracy": accuracy,
        "state_complexity": np.zeros(12),
        "preference_per_depth": preference,
        "epistemic_value_per_depth": np.zeros(12),
        "compute_ms": np.zeros(12),
        "switching_ms": np.zeros(12),
        "information_loss": np.zeros(12),
        "deadline_preference": LogNormalDeadlinePreference(median_ms=1e6),
    }

    low_information = select_four_term_value_allocation(
        **common,
        fisher_context_score=0.0,
        fisher_weight_min=0.1,
    )
    high_information = select_four_term_value_allocation(
        **common,
        fisher_context_score=1.0,
        fisher_weight_min=0.1,
    )

    assert low_information.allocation_index == 0
    assert high_information.allocation_index == 11
    assert low_information.fisher_context_weight == pytest.approx(0.1)
    assert high_information.fisher_context_weight == pytest.approx(1.0)


def test_four_term_selector_prefers_higher_preference_and_epistemic_value():
    common = {
        "state_accuracy": np.zeros(12),
        "state_complexity": np.zeros(12),
        "preference_per_depth": np.full(12, -2.0),
        "epistemic_value_per_depth": np.zeros(12),
        "compute_ms": np.zeros(12),
        "switching_ms": np.zeros(12),
        "information_loss": np.zeros(12),
        "deadline_preference": LogNormalDeadlinePreference(median_ms=1e6),
    }
    common["preference_per_depth"][1] = -1.0
    preference_decision = select_four_term_value_allocation(**common)
    assert preference_decision.allocation_index == 1

    common["preference_per_depth"][:] = -2.0
    common["epistemic_value_per_depth"][2] = 1.0
    epistemic_decision = select_four_term_value_allocation(**common)
    assert epistemic_decision.allocation_index == 2
