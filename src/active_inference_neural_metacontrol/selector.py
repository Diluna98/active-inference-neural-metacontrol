"""Joint task-quality and resource-cost selection for neural metacontrol."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .allocations import ALLOCATIONS, Allocation
from .costs import LatencyScalePreference, LogNormalDeadlinePreference


@dataclass(frozen=True)
class AllocationDecision:
    allocation: Allocation
    allocation_index: int
    predicted_normalized_g: float
    predicted_compute_ms: float
    switching_ms: float
    mean_compute_ms: float
    deadline_survival_probability: float
    compute_surprisal_nats: float
    compute_preference_cost_nats: float
    normalized_compute_cost: float
    switching_penalty: float
    information_loss_normalized: float
    information_loss_nats: float
    objective_score: float
    predicted_success_probability: float | None = None
    predicted_operational_cost: float | None = None
    predicted_preference_per_depth: float | None = None
    predicted_epistemic_per_depth: float | None = None
    task_utility_score: float | None = None
    predicted_state_accuracy: float | None = None
    predicted_state_complexity: float | None = None
    predicted_epistemic_value_per_depth: float | None = None
    fisher_context_score: float | None = None
    fisher_context_weight: float | None = None


def _vector(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float).ravel()
    if array.shape != (len(ALLOCATIONS),) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain {len(ALLOCATIONS)} finite values")
    return array


def select_allocation(
    *,
    normalized_g: np.ndarray,
    compute_ms: np.ndarray,
    switching_ms: np.ndarray,
    information_loss: np.ndarray,
    deadline_preference: LogNormalDeadlinePreference,
    compute_preference: LatencyScalePreference | None = None,
    commit_for_depth: bool = True,
    compute_cost_weight: float = 1.0,
    switching_cost_weight: float = 1.0,
    information_loss_weight: float = 1.0,
) -> AllocationDecision:
    """Maximize task value and ICRA-style computational preferences.

    Inference latency is scored on a continuous preference scale with a linear
    cost and an extra quadratic cost above a comfort point. Any nonzero
    switching entry denotes one configuration change and receives the same
    literal switching penalty. Deadline survival and surprisal remain
    diagnostics and do not affect the objective.
    """

    g_values = _vector(normalized_g, "normalized_g")
    compute = _vector(compute_ms, "compute_ms")
    switching = _vector(switching_ms, "switching_ms")
    information = _vector(information_loss, "information_loss")
    if np.any(compute < 0) or np.any(switching < 0):
        raise ValueError("resource costs must be nonnegative")
    if np.any((information < 0) | (information > 1)):
        raise ValueError("information losses must lie in [0, 1]")
    if min(compute_cost_weight, switching_cost_weight, information_loss_weight) < 0:
        raise ValueError("resource weights must be nonnegative")
    depths = np.asarray([allocation.depth for allocation in ALLOCATIONS], dtype=float)
    executed_steps = depths if commit_for_depth else np.ones_like(depths)
    mean_compute = compute + switching / executed_steps
    deadline_survival = deadline_preference.survival_probability(mean_compute)
    compute_surprisal = -np.log(deadline_survival)
    if compute_preference is None:
        compute_preference = LatencyScalePreference(
            comfort_ms=0.75 * deadline_preference.median_ms,
            deadline_ms=deadline_preference.median_ms,
        )
    compute_preference_cost = (
        compute_cost_weight * compute_preference.cost_nats(compute)
    )
    switching_penalty = switching_cost_weight * (switching > 0).astype(float)
    information_nats = information_loss_weight * information * np.log(2.0)
    objective = (
        g_values - compute_preference_cost - switching_penalty - information_nats
    )
    index = int(np.argmax(objective))
    return AllocationDecision(
        allocation=ALLOCATIONS[index],
        allocation_index=index,
        predicted_normalized_g=float(g_values[index]),
        predicted_compute_ms=float(compute[index]),
        switching_ms=float(switching[index]),
        mean_compute_ms=float(mean_compute[index]),
        deadline_survival_probability=float(deadline_survival[index]),
        compute_surprisal_nats=float(compute_surprisal[index]),
        compute_preference_cost_nats=float(compute_preference_cost[index]),
        normalized_compute_cost=float(compute_preference_cost[index]),
        switching_penalty=float(switching_penalty[index]),
        information_loss_normalized=float(information[index]),
        information_loss_nats=float(information_nats[index]),
        objective_score=float(objective[index]),
    )


def select_task_utility_allocation(
    *,
    success_probability: np.ndarray,
    operational_cost: np.ndarray,
    normalized_g: np.ndarray,
    compute_ms: np.ndarray,
    switching_ms: np.ndarray,
    information_loss: np.ndarray,
    deadline_preference: LogNormalDeadlinePreference,
    compute_preference: LatencyScalePreference | None = None,
    success_weight: float = 0.5,
    operational_cost_weight: float = 1.0,
    operational_cost_temperature: float = 5.0,
    normalized_g_weight: float = 1.0,
    commit_for_depth: bool = True,
    compute_cost_weight: float = 1.0,
    switching_cost_weight: float = 1.0,
    information_loss_weight: float = 1.0,
) -> AllocationDecision:
    """Select from learned task utility and calibrated resource surprisal."""

    probability = _vector(success_probability, "success_probability")
    cost = _vector(operational_cost, "operational_cost")
    g_values = _vector(normalized_g, "normalized_g")
    if np.any((probability < 0) | (probability > 1)):
        raise ValueError("success probabilities must lie in [0, 1]")
    if np.any(cost < 0) or operational_cost_temperature <= 0:
        raise ValueError("operational costs must be nonnegative and temperature positive")
    if min(success_weight, operational_cost_weight, normalized_g_weight) < 0:
        raise ValueError("task-utility weights must be nonnegative")
    task_utility = (
        success_weight * np.log(np.clip(probability, 1e-8, 1.0))
        - operational_cost_weight * cost / operational_cost_temperature
        + normalized_g_weight * g_values
    )
    decision = select_allocation(
        normalized_g=task_utility,
        compute_ms=compute_ms,
        switching_ms=switching_ms,
        information_loss=information_loss,
        deadline_preference=deadline_preference,
        compute_preference=compute_preference,
        commit_for_depth=commit_for_depth,
        compute_cost_weight=compute_cost_weight,
        switching_cost_weight=switching_cost_weight,
        information_loss_weight=information_loss_weight,
    )
    index = decision.allocation_index
    return replace(
        decision,
        predicted_normalized_g=float(g_values[index]),
        predicted_success_probability=float(probability[index]),
        predicted_operational_cost=float(cost[index]),
        task_utility_score=float(task_utility[index]),
    )


def select_decomposed_task_utility_allocation(
    *,
    success_probability: np.ndarray,
    operational_cost: np.ndarray,
    preference_per_depth: np.ndarray,
    epistemic_per_depth: np.ndarray,
    compute_ms: np.ndarray,
    switching_ms: np.ndarray,
    information_loss: np.ndarray,
    deadline_preference: LogNormalDeadlinePreference,
    compute_preference: LatencyScalePreference | None = None,
    success_weight: float = 0.5,
    operational_cost_weight: float = 1.0,
    operational_cost_temperature: float = 5.0,
    preference_weight: float = 1.0,
    epistemic_weight: float = 1.0,
    commit_for_depth: bool = True,
    compute_cost_weight: float = 1.0,
    switching_cost_weight: float = 1.0,
    information_loss_weight: float = 1.0,
) -> AllocationDecision:
    """Select using separately learned terms whose sum reconstructs G/T."""

    preference = _vector(preference_per_depth, "preference_per_depth")
    epistemic = _vector(epistemic_per_depth, "epistemic_per_depth")
    if min(preference_weight, epistemic_weight) < 0:
        raise ValueError("decomposed Active-Inference weights must be nonnegative")
    decision = select_task_utility_allocation(
        success_probability=success_probability,
        operational_cost=operational_cost,
        normalized_g=preference_weight * preference + epistemic_weight * epistemic,
        compute_ms=compute_ms,
        switching_ms=switching_ms,
        information_loss=information_loss,
        deadline_preference=deadline_preference,
        compute_preference=compute_preference,
        success_weight=success_weight,
        operational_cost_weight=operational_cost_weight,
        operational_cost_temperature=operational_cost_temperature,
        normalized_g_weight=1.0,
        commit_for_depth=commit_for_depth,
        compute_cost_weight=compute_cost_weight,
        switching_cost_weight=switching_cost_weight,
        information_loss_weight=information_loss_weight,
    )
    index = decision.allocation_index
    return replace(
        decision,
        predicted_preference_per_depth=float(preference[index]),
        predicted_epistemic_per_depth=float(epistemic[index]),
    )


def select_inference_value_allocation(
    *,
    preference_per_depth: np.ndarray,
    epistemic_per_depth: np.ndarray,
    compute_ms: np.ndarray,
    switching_ms: np.ndarray,
    information_loss: np.ndarray,
    deadline_preference: LogNormalDeadlinePreference,
    compute_preference: LatencyScalePreference | None = None,
    preference_weight: float = 1.0,
    epistemic_weight: float = 1.0,
    commit_for_depth: bool = True,
    compute_cost_weight: float = 1.0,
    switching_cost_weight: float = 1.0,
    information_loss_weight: float = 1.0,
) -> AllocationDecision:
    """Select without learned success or operational-cost terms."""

    preference = _vector(preference_per_depth, "preference_per_depth")
    epistemic = _vector(epistemic_per_depth, "epistemic_per_depth")
    if min(preference_weight, epistemic_weight) < 0:
        raise ValueError("decomposed Active-Inference weights must be nonnegative")
    task_score = preference_weight * preference + epistemic_weight * epistemic
    decision = select_allocation(
        normalized_g=task_score,
        compute_ms=compute_ms,
        switching_ms=switching_ms,
        information_loss=information_loss,
        deadline_preference=deadline_preference,
        compute_preference=compute_preference,
        commit_for_depth=commit_for_depth,
        compute_cost_weight=compute_cost_weight,
        switching_cost_weight=switching_cost_weight,
        information_loss_weight=information_loss_weight,
    )
    index = decision.allocation_index
    return replace(
        decision,
        predicted_normalized_g=float(preference[index] + epistemic[index]),
        predicted_preference_per_depth=float(preference[index]),
        predicted_epistemic_per_depth=float(epistemic[index]),
        task_utility_score=float(task_score[index]),
    )


def select_four_term_value_allocation(
    *,
    state_accuracy: np.ndarray,
    state_complexity: np.ndarray,
    preference_per_depth: np.ndarray,
    epistemic_value_per_depth: np.ndarray,
    compute_ms: np.ndarray,
    switching_ms: np.ndarray,
    information_loss: np.ndarray,
    deadline_preference: LogNormalDeadlinePreference,
    compute_preference: LatencyScalePreference | None = None,
    state_accuracy_weight: float = 1.0,
    state_complexity_weight: float = 1.0,
    preference_weight: float = 1.0,
    epistemic_weight: float = 1.0,
    commit_for_depth: bool = True,
    compute_cost_weight: float = 1.0,
    switching_cost_weight: float = 1.0,
    information_loss_weight: float = 1.0,
    fisher_context_score: float | None = None,
    fisher_weight_min: float = 0.1,
    fisher_weight_max: float = 1.0,
    fisher_weight_power: float = 1.0,
) -> AllocationDecision:
    """Maximize next-state evidence and prospective policy value.

    When ``fisher_context_score`` is supplied, posterior-weighted sensing
    discriminability jointly gates state evidence and epistemic value. Task
    preferences and resource costs retain their original scales.
    """

    accuracy = _vector(state_accuracy, "state_accuracy")
    complexity = _vector(state_complexity, "state_complexity")
    preference = _vector(preference_per_depth, "preference_per_depth")
    epistemic = _vector(epistemic_value_per_depth, "epistemic_value_per_depth")
    if min(
        state_accuracy_weight,
        state_complexity_weight,
        preference_weight,
        epistemic_weight,
    ) < 0:
        raise ValueError("four-term Active-Inference weights must be nonnegative")
    fisher_weight = None
    if fisher_context_score is not None:
        if not np.isfinite(fisher_context_score) or not 0 <= fisher_context_score <= 1:
            raise ValueError("fisher_context_score must lie in [0, 1]")
        if not 0 <= fisher_weight_min <= fisher_weight_max:
            raise ValueError("Fisher weights must satisfy 0 <= min <= max")
        if fisher_weight_power <= 0:
            raise ValueError("fisher_weight_power must be positive")
        fisher_weight = fisher_weight_min + (
            fisher_weight_max - fisher_weight_min
        ) * fisher_context_score**fisher_weight_power
    information_sensitive = (
        state_accuracy_weight * accuracy
        - state_complexity_weight * complexity
        + epistemic_weight * epistemic
    )
    task_score = (
        (1.0 if fisher_weight is None else fisher_weight) * information_sensitive
        + preference_weight * preference
    )
    decision = select_allocation(
        normalized_g=task_score,
        compute_ms=compute_ms,
        switching_ms=switching_ms,
        information_loss=information_loss,
        deadline_preference=deadline_preference,
        compute_preference=compute_preference,
        commit_for_depth=commit_for_depth,
        compute_cost_weight=compute_cost_weight,
        switching_cost_weight=switching_cost_weight,
        information_loss_weight=information_loss_weight,
    )
    index = decision.allocation_index
    return replace(
        decision,
        predicted_normalized_g=float(task_score[index]),
        predicted_state_accuracy=float(accuracy[index]),
        predicted_state_complexity=float(complexity[index]),
        predicted_preference_per_depth=float(preference[index]),
        predicted_epistemic_value_per_depth=float(epistemic[index]),
        task_utility_score=float(task_score[index]),
        fisher_context_score=fisher_context_score,
        fisher_context_weight=fisher_weight,
    )
