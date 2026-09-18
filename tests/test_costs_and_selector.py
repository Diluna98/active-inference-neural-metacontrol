import numpy as np
import pytest

from active_inference_neural_metacontrol import (
    ALLOCATIONS,
    Allocation,
    ComputeProfile,
    ProfiledComputeCostModel,
    SelectionConstraints,
    SwitchingCostMatrix,
    select_allocation,
)


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


def test_selector_chooses_lowest_task_cost_among_feasible_allocations():
    success = np.full(12, 0.95)
    task = np.arange(12, 0, -1, dtype=float)
    compute = np.full(12, 50.0)
    switching = np.zeros(12)
    information = np.zeros(12)
    compute[-1] = 200.0

    decision = select_allocation(
        success_probability=success,
        task_cost=task,
        compute_ms=compute,
        switching_ms=switching,
        information_loss=information,
        constraints=SelectionConstraints(compute_budget_ms=100.0),
    )

    assert decision.feasible
    assert decision.allocation == ALLOCATIONS[-2]


def test_selector_falls_back_to_highest_success_under_deadline():
    success = np.linspace(0.1, 0.8, 12)
    compute = np.linspace(10, 120, 12)

    decision = select_allocation(
        success_probability=success,
        task_cost=np.ones(12),
        compute_ms=compute,
        switching_ms=np.zeros(12),
        information_loss=np.zeros(12),
        constraints=SelectionConstraints(success_threshold=0.95, compute_budget_ms=50.0),
    )

    assert not decision.feasible
    assert decision.allocation_index == 4
