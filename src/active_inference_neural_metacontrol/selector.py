"""Constrained selection over predicted task outcomes and independent costs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .allocations import ALLOCATIONS, Allocation


@dataclass(frozen=True)
class SelectionConstraints:
    success_threshold: float = 0.9
    compute_budget_ms: float = 100.0
    information_loss_limit: float = 0.15

    def __post_init__(self) -> None:
        if not 0 <= self.success_threshold <= 1:
            raise ValueError("success_threshold must lie in [0, 1]")
        if self.compute_budget_ms <= 0:
            raise ValueError("compute_budget_ms must be positive")
        if not 0 <= self.information_loss_limit <= 1:
            raise ValueError("information_loss_limit must lie in [0, 1]")


@dataclass(frozen=True)
class AllocationDecision:
    allocation: Allocation
    allocation_index: int
    feasible: bool
    reason: str
    predicted_success: float
    predicted_task_cost: float
    predicted_compute_ms: float
    switching_ms: float
    information_loss: float


def _vector(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float).ravel()
    if array.shape != (len(ALLOCATIONS),) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain {len(ALLOCATIONS)} finite values")
    return array


def select_allocation(
    *,
    success_probability: np.ndarray,
    task_cost: np.ndarray,
    compute_ms: np.ndarray,
    switching_ms: np.ndarray,
    information_loss: np.ndarray,
    constraints: SelectionConstraints,
) -> AllocationDecision:
    """Choose minimum task cost subject to reliability, compute, and information limits."""

    success = _vector(success_probability, "success_probability")
    task = _vector(task_cost, "task_cost")
    compute = _vector(compute_ms, "compute_ms")
    switching = _vector(switching_ms, "switching_ms")
    information = _vector(information_loss, "information_loss")
    if np.any((success < 0) | (success > 1)):
        raise ValueError("success probabilities must lie in [0, 1]")
    if np.any(task < 0) or np.any(compute < 0) or np.any(switching < 0):
        raise ValueError("task and resource costs must be nonnegative")
    if np.any((information < 0) | (information > 1)):
        raise ValueError("information losses must lie in [0, 1]")
    resource_ms = compute + switching
    feasible_mask = (
        (success >= constraints.success_threshold)
        & (resource_ms <= constraints.compute_budget_ms)
        & (information <= constraints.information_loss_limit)
    )
    feasible_indices = np.flatnonzero(feasible_mask)
    if feasible_indices.size:
        index = int(feasible_indices[np.argmin(task[feasible_indices])])
        reason = "minimum predicted task cost among feasible allocations"
        feasible = True
    else:
        deadline_indices = np.flatnonzero(resource_ms <= constraints.compute_budget_ms)
        candidates = deadline_indices if deadline_indices.size else np.arange(len(ALLOCATIONS))
        candidate_success = success[candidates]
        best_success = float(candidate_success.max())
        success_ties = candidates[np.isclose(candidate_success, best_success)]
        index = int(success_ties[np.argmin(resource_ms[success_ties])])
        reason = "fallback: highest predicted success under the available deadline"
        feasible = False
    return AllocationDecision(
        allocation=ALLOCATIONS[index],
        allocation_index=index,
        feasible=feasible,
        reason=reason,
        predicted_success=float(success[index]),
        predicted_task_cost=float(task[index]),
        predicted_compute_ms=float(compute[index]),
        switching_ms=float(switching[index]),
        information_loss=float(information[index]),
    )
