"""Matched-state counterfactual data generation for neural metacontrol."""

from __future__ import annotations

import copy
import csv
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from .allocations import ALLOCATIONS, Allocation
from .beliefs import canonicalize_posterior, remap_posterior
from .mos_adapter import (
    MOSFeatureBatch,
    build_mos_features,
    canonical_detection_likelihood,
    selected_policy_index,
    selected_policy_next_target_posterior,
)
from .state_values import corrected_state_term_components

DEFAULT_REFERENCE_ALLOCATION = Allocation(20, 1)
ONE_STEP_LABEL_SCHEMA = "depth-normalized-selected-policy-g-v4"
ACCUMULATED_LABEL_SCHEMA = "candidate-controlled-accumulated-normalized-g-v5"
TASK_OUTCOME_SCHEMA = "one-action-intervention-then-reference-rollout-v1"
RESEARCH_ARCHIVE_SCHEMA = "mos-counterfactual-research-v1"
MAX_POLICY_COUNT = 125


@dataclass(frozen=True)
class CounterfactualDataset:
    """Training arrays plus auditable tabular records."""

    spatial: np.ndarray
    context: np.ndarray
    normalized_g: np.ndarray
    raw_g: np.ndarray
    risk: np.ndarray
    ambiguity: np.ndarray
    information_gain: np.ndarray
    compute_ms: np.ndarray
    switch_ms: np.ndarray
    candidate_actions: np.ndarray
    context_ids: tuple[str, ...]
    allocations: tuple[Allocation, ...]
    contexts: tuple[dict[str, Any], ...]
    branches: tuple[dict[str, Any], ...]
    trajectory: tuple[dict[str, Any], ...]
    label_schema: str = ONE_STEP_LABEL_SCHEMA
    rollout_horizon: int = 1
    rollout_discount: float = 1.0
    state_accuracy: np.ndarray | None = None
    state_complexity: np.ndarray | None = None
    uniform_risk: np.ndarray | None = None
    uniform_ambiguity: np.ndarray | None = None
    uniform_information_gain: np.ndarray | None = None
    success: np.ndarray | None = None
    task_cost: np.ndarray | None = None
    task_outcome_schema: str | None = None
    research_arrays: dict[str, np.ndarray] | None = None


@dataclass(frozen=True)
class _Decision:
    action: Any
    state_ms: float
    policy_ms: float
    action_ms: float
    selected_policy_g: float
    risk: float
    ambiguity: float
    information_gain: float
    state_accuracy: float
    state_complexity: float
    uniform_risk: float
    uniform_ambiguity: float
    uniform_information_gain: float
    selected_policy: int
    policy_g: np.ndarray
    policy_vfe: np.ndarray
    policy_risk: np.ndarray
    policy_ambiguity: np.ndarray
    policy_information_gain: np.ndarray
    policy_posterior: np.ndarray
    policies: np.ndarray
    accuracy_by_modality: np.ndarray
    complexity_by_factor: np.ndarray
    target_prior: np.ndarray
    target_posterior: np.ndarray

    @property
    def total_ms(self) -> float:
        return self.state_ms + self.policy_ms + self.action_ms


def _timed(function: Any) -> tuple[Any, float]:
    started = time.perf_counter_ns()
    value = function()
    return value, (time.perf_counter_ns() - started) / 1e6


def _mos_imports() -> dict[str, Any]:
    try:
        from active_inference_navigation.mos import (
            FindOutcome,
            MOSAgentConfig,
            MOSEnvironment,
            build_mos_agent,
            mos_action_controls,
            sample_mos_instance,
            selected_mos_action,
        )
    except ImportError as error:  # pragma: no cover - optional integration dependency
        raise ImportError(
            "counterfactual generation requires active-inference-navigation-agent"
        ) from error
    return {
        "FindOutcome": FindOutcome,
        "MOSAgentConfig": MOSAgentConfig,
        "MOSEnvironment": MOSEnvironment,
        "build_mos_agent": build_mos_agent,
        "mos_action_controls": mos_action_controls,
        "sample_mos_instance": sample_mos_instance,
        "selected_mos_action": selected_mos_action,
    }


def _infer_decision(
    agent: Any,
    observation: tuple[int, ...],
    previous_action: Any | None,
    time_step: int,
    mos: dict[str, Any],
) -> _Decision:
    controls = None if previous_action is None else mos["mos_action_controls"](previous_action)
    agent.observe(observation, time_step=time_step, executed_action=controls)
    core = getattr(agent, "_agent", agent)
    priors = [np.asarray(values, dtype=float).copy() for values in core._receding_prior]
    _, state_ms = _timed(agent.infer_states)
    state_result = agent.last_state_inference
    accuracy_by_modality, complexity_by_factor = corrected_state_term_components(
        agent,
        observation,
        priors,
        state_result.posteriors,
    )
    state_accuracy = float(accuracy_by_modality.sum())
    state_complexity = float(complexity_by_factor.sum())
    _, policy_ms = _timed(agent.infer_policies)
    action, action_ms = _timed(lambda: mos["selected_mos_action"](agent))
    g_values = np.asarray(agent.G_policy, dtype=float).ravel()
    if g_values.size < 1 or not np.all(np.isfinite(g_values)):
        raise ValueError("policy G must contain at least one finite value")
    selected = selected_policy_index(agent)
    diagnostics = agent.last_policy_inference
    risk = np.asarray(diagnostics.risk, dtype=float).ravel()
    ambiguity = np.asarray(diagnostics.ambiguity, dtype=float).ravel()
    information_gain = np.asarray(diagnostics.information_gain, dtype=float).ravel()
    if information_gain.size == 0:
        information_gain = np.zeros_like(g_values)
    for name, values in (
        ("risk", risk),
        ("ambiguity", ambiguity),
        ("information_gain", information_gain),
    ):
        if values.shape != g_values.shape or not np.all(np.isfinite(values)):
            raise ValueError(f"policy {name} diagnostics must match policy G")
    return _Decision(
        action,
        state_ms,
        policy_ms,
        action_ms,
        float(g_values[selected]),
        float(risk[selected]),
        float(ambiguity[selected]),
        float(information_gain[selected]),
        state_accuracy,
        state_complexity,
        float(risk.mean()),
        float(ambiguity.mean()),
        float(information_gain.mean()),
        selected,
        g_values.copy(),
        np.asarray(diagnostics.variational_free_energy, dtype=float).ravel().copy(),
        risk.copy(),
        ambiguity.copy(),
        information_gain.copy(),
        np.asarray(agent.posterior_pi, dtype=float).ravel().copy(),
        np.asarray(agent.policies, dtype=np.int16).copy(),
        accuracy_by_modality.copy(),
        complexity_by_factor.copy(),
        np.asarray(priors[2], dtype=float).copy(),
        np.asarray(state_result.posteriors[2], dtype=float).copy(),
    )


def _normalized_policy_g(decision: _Decision, allocation: Allocation) -> float:
    """Normalize PyAIF's selected-policy value across planning depth."""

    return float(
        (decision.risk + decision.ambiguity - decision.information_gain) / allocation.depth
    )


def _step_cost(*, false_find: bool, collision: bool) -> float:
    return 1.0 + 5.0 * false_find + 0.5 * collision


def _rollout_candidate(
    *,
    agent: Any,
    environment: Any,
    first_decision: _Decision,
    allocation: Allocation,
    first_time_step: int,
    quantiles: np.ndarray,
    horizon: int,
    discount: float,
    mos: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate one allocation while it controls up to ``horizon`` physical steps."""

    decision = first_decision
    discounted_g = 0.0
    discount_mass = 0.0
    task_cost = 0.0
    actions: list[str] = []
    success = False
    steps = 0
    for offset in range(horizon):
        weight = discount**offset
        discounted_g += weight * _normalized_policy_g(decision, allocation)
        discount_mass += weight
        actions.append(decision.action.name)
        observation, success = environment.step(
            decision.action,
            noise_quantile=float(quantiles[first_time_step + offset + 1]),
        )
        false_find = observation[3] == mos["FindOutcome"].FALSE_FIND
        task_cost += _step_cost(false_find=false_find, collision=environment.last_collision)
        steps += 1
        if success or offset + 1 >= horizon:
            break
        decision = _infer_decision(
            agent,
            observation,
            decision.action,
            first_time_step + offset + 1,
            mos,
        )
    return {
        "first_normalized_g": _normalized_policy_g(first_decision, allocation),
        "discounted_g_sum": discounted_g,
        "discounted_g_mean": discounted_g / discount_mass,
        "rollout_steps": steps,
        "success_within_horizon": bool(success),
        "task_cost": task_cost,
        "actions": json.dumps(actions),
    }


def _continue_after_intervention(
    *,
    agent: Any,
    instance: Any,
    position: tuple[int, int],
    action: Any,
    decision: int,
    max_steps: int,
    quantiles: np.ndarray,
    mos: dict[str, Any],
) -> dict[str, Any]:
    """Execute one candidate action, then return control to the reference agent."""

    environment = mos["MOSEnvironment"](
        layout=instance.layout,
        target=instance.target,
        observation_seed=instance.observation_seed,
    )
    environment.position = position
    controller = copy.deepcopy(agent)
    observation, success = environment.step(
        action, noise_quantile=float(quantiles[decision + 1])
    )
    false_find = observation[3] == mos["FindOutcome"].FALSE_FIND
    task_cost = _step_cost(false_find=false_find, collision=environment.last_collision)
    false_finds = int(false_find)
    collisions = int(environment.last_collision)
    steps = 1
    previous_action = action

    for next_decision in range(decision + 1, max_steps):
        if success:
            break
        inferred = _infer_decision(
            controller, observation, previous_action, next_decision, mos
        )
        observation, success = environment.step(
            inferred.action,
            noise_quantile=float(quantiles[next_decision + 1]),
        )
        false_find = observation[3] == mos["FindOutcome"].FALSE_FIND
        collision = environment.last_collision
        task_cost += _step_cost(false_find=false_find, collision=collision)
        false_finds += int(false_find)
        collisions += int(collision)
        steps += 1
        previous_action = inferred.action

    if not success:
        task_cost += 2.0 * max_steps
    return {
        "eventual_success": bool(success),
        "remaining_task_cost": float(task_cost),
        "remaining_steps": steps,
        "remaining_false_finds": false_finds,
        "remaining_collisions": collisions,
        "final_x": environment.position[0],
        "final_y": environment.position[1],
    }


def _build_switched_agent(
    *,
    source_agent: Any,
    source_allocation: Allocation,
    target_allocation: Allocation,
    layout: Any,
    executed_action: Any,
    next_time_step: int,
    message_passing_iterations: int,
    policy_workers: int,
    mos: dict[str, Any],
) -> Any:
    """Initialize a candidate from one common, resolution-remapped belief."""

    candidate = mos["build_mos_agent"](
        mos["MOSAgentConfig"](
            target_resolution=target_allocation.resolution,
            action_depth=target_allocation.depth,
            message_passing_iterations=message_passing_iterations,
            policy_workers=policy_workers,
        ),
        layout=layout,
    )
    candidate.reset()
    return _transfer_receding_state(
        candidate=candidate,
        source_agent=source_agent,
        source_allocation=source_allocation,
        target_allocation=target_allocation,
        layout=layout,
        executed_action=executed_action,
        next_time_step=next_time_step,
        mos=mos,
    )


def _transfer_receding_state(
    *,
    candidate: Any,
    source_agent: Any,
    source_allocation: Allocation,
    target_allocation: Allocation,
    layout: Any,
    executed_action: Any,
    next_time_step: int,
    mos: dict[str, Any],
) -> Any:
    """Transfer only recurrent state into an already initialized agent.

    Static generative-model arrays, policies, and normalized likelihoods are
    retained by ``candidate``. Only the current belief and receding-horizon
    bookkeeping are overwritten. This is the online switching path; the
    counterfactual builder above deliberately remains able to create a fresh
    independent agent before calling it.
    """

    current = source_agent._receding_posterior
    if current is None:
        raise RuntimeError("source agent has no receding posterior to transfer")
    candidate_core = getattr(candidate, "_agent", candidate)
    target_posterior = np.asarray(current[2], dtype=float)
    if source_allocation.resolution == target_allocation.resolution:
        target_posterior = target_posterior.copy()
    else:
        target_posterior = remap_posterior(
            target_posterior,
            source_allocation.resolution,
            target_allocation.resolution,
            layout.size,
        ).ravel()
    candidate_core._receding_posterior = [
        np.asarray(current[0], dtype=float).copy(),
        np.asarray(current[1], dtype=float).copy(),
        target_posterior,
        np.asarray(current[3], dtype=float).copy(),
    ]
    candidate_core._receding_action = mos["mos_action_controls"](executed_action)
    candidate_core._current_time = next_time_step
    candidate_core._receding_stage = "ready"
    return candidate


def _empty_dataset(
    allocations: tuple[Allocation, ...],
    *,
    label_schema: str = ONE_STEP_LABEL_SCHEMA,
    rollout_horizon: int = 1,
    rollout_discount: float = 1.0,
    collect_task_outcomes: bool = False,
    collect_research_archive: bool = False,
) -> CounterfactualDataset:
    return CounterfactualDataset(
        spatial=np.empty((0, 6, 20, 20), dtype=np.float32),
        context=np.empty((0, 16), dtype=np.float32),
        normalized_g=np.empty((0, len(allocations)), dtype=np.float32),
        raw_g=np.empty((0, len(allocations)), dtype=np.float32),
        risk=np.empty((0, len(allocations)), dtype=np.float32),
        ambiguity=np.empty((0, len(allocations)), dtype=np.float32),
        information_gain=np.empty((0, len(allocations)), dtype=np.float32),
        compute_ms=np.empty((0, len(allocations)), dtype=np.float32),
        switch_ms=np.empty((0, len(allocations)), dtype=np.float32),
        candidate_actions=np.empty((0, len(allocations)), dtype=np.int8),
        context_ids=(),
        allocations=allocations,
        contexts=(),
        branches=(),
        trajectory=(),
        label_schema=label_schema,
        rollout_horizon=rollout_horizon,
        rollout_discount=rollout_discount,
        state_accuracy=np.empty((0, len(allocations)), dtype=np.float32),
        state_complexity=np.empty((0, len(allocations)), dtype=np.float32),
        uniform_risk=np.empty((0, len(allocations)), dtype=np.float32),
        uniform_ambiguity=np.empty((0, len(allocations)), dtype=np.float32),
        uniform_information_gain=np.empty((0, len(allocations)), dtype=np.float32),
        success=(
            np.empty((0, len(allocations)), dtype=np.float32)
            if collect_task_outcomes
            else None
        ),
        task_cost=(
            np.empty((0, len(allocations)), dtype=np.float32)
            if collect_task_outcomes
            else None
        ),
        task_outcome_schema=TASK_OUTCOME_SCHEMA if collect_task_outcomes else None,
        research_arrays=(
            _empty_research_arrays(len(allocations)) if collect_research_archive else None
        ),
    )


def _empty_research_arrays(allocation_count: int) -> dict[str, np.ndarray]:
    """Return shaped empty arrays for the reusable research archive."""

    return {
        "observation": np.empty((0, 5), dtype=np.int16),
        "state_ms": np.empty((0, allocation_count), dtype=np.float32),
        "policy_ms": np.empty((0, allocation_count), dtype=np.float32),
        "action_ms": np.empty((0, allocation_count), dtype=np.float32),
        "selected_policy": np.empty((0, allocation_count), dtype=np.int16),
        "policy_count": np.empty((0, allocation_count), dtype=np.int16),
        "policy_mask": np.empty((0, allocation_count, MAX_POLICY_COUNT), dtype=bool),
        "policy_g": np.empty((0, allocation_count, MAX_POLICY_COUNT), dtype=np.float32),
        "policy_vfe": np.empty((0, allocation_count, MAX_POLICY_COUNT), dtype=np.float32),
        "policy_risk": np.empty((0, allocation_count, MAX_POLICY_COUNT), dtype=np.float32),
        "policy_ambiguity": np.empty(
            (0, allocation_count, MAX_POLICY_COUNT), dtype=np.float32
        ),
        "policy_information_gain": np.empty(
            (0, allocation_count, MAX_POLICY_COUNT), dtype=np.float32
        ),
        "policy_posterior": np.empty(
            (0, allocation_count, MAX_POLICY_COUNT), dtype=np.float32
        ),
        "accuracy_by_modality": np.empty((0, allocation_count, 5), dtype=np.float32),
        "complexity_by_factor": np.empty((0, allocation_count, 4), dtype=np.float32),
        "canonical_target_prior": np.empty(
            (0, allocation_count, 20, 20), dtype=np.float32
        ),
        "canonical_target_posterior": np.empty(
            (0, allocation_count, 20, 20), dtype=np.float32
        ),
        "canonical_selected_prediction": np.empty(
            (0, allocation_count, 20, 20), dtype=np.float32
        ),
        "candidate_next_position": np.empty((0, allocation_count, 2), dtype=np.int16),
        "candidate_detection_likelihood": np.empty(
            (0, allocation_count, 20, 20), dtype=np.float32
        ),
        "candidate_predicted_observation": np.empty(
            (0, allocation_count, 2), dtype=np.float32
        ),
    }


def _research_context_arrays(
    *,
    decisions: dict[Allocation, _Decision],
    agents: dict[Allocation, Any],
    allocations: tuple[Allocation, ...],
    layout: Any,
    position: tuple[int, int],
    observation: tuple[int, ...],
) -> dict[str, np.ndarray]:
    """Capture expensive candidate inference outputs in a reusable padded archive."""

    count = len(allocations)
    values = {
        name: np.zeros(array.shape[1:], dtype=array.dtype)
        for name, array in _empty_research_arrays(count).items()
    }
    values["observation"] = np.asarray(observation, dtype=np.int16)
    for index, allocation in enumerate(allocations):
        decision = decisions[allocation]
        agent = agents[allocation]
        policy_count = decision.policy_g.size
        if policy_count > MAX_POLICY_COUNT:
            raise ValueError(f"policy count {policy_count} exceeds archive capacity")
        values["state_ms"][index] = decision.state_ms
        values["policy_ms"][index] = decision.policy_ms
        values["action_ms"][index] = decision.action_ms
        values["selected_policy"][index] = decision.selected_policy
        values["policy_count"][index] = policy_count
        values["policy_mask"][index, :policy_count] = True
        for name, source in (
            ("policy_g", decision.policy_g),
            ("policy_vfe", decision.policy_vfe),
            ("policy_risk", decision.policy_risk),
            ("policy_ambiguity", decision.policy_ambiguity),
            ("policy_information_gain", decision.policy_information_gain),
            ("policy_posterior", decision.policy_posterior),
        ):
            values[name][index].fill(np.nan)
            values[name][index, :policy_count] = source
        values["accuracy_by_modality"][index] = decision.accuracy_by_modality
        values["complexity_by_factor"][index] = decision.complexity_by_factor
        values["canonical_target_prior"][index] = canonicalize_posterior(
            decision.target_prior, allocation.resolution, layout.size
        )
        values["canonical_target_posterior"][index] = canonicalize_posterior(
            decision.target_posterior, allocation.resolution, layout.size
        )
        predicted = canonicalize_posterior(
            selected_policy_next_target_posterior(agent), allocation.resolution, layout.size
        )
        values["canonical_selected_prediction"][index] = predicted
        next_position = layout.move(position, decision.action)
        values["candidate_next_position"][index] = next_position
        likelihood = canonical_detection_likelihood(layout, next_position)
        values["candidate_detection_likelihood"][index] = likelihood[1]
        values["candidate_predicted_observation"][index] = np.sum(
            likelihood * predicted[None, :, :], axis=(1, 2)
        )
    return values


def generate_mos_counterfactuals(
    *,
    instance_seeds: Sequence[int],
    reference_allocation: Allocation = DEFAULT_REFERENCE_ALLOCATION,
    max_steps: int = 50,
    branch_stride: int = 1,
    message_passing_iterations: int = 10,
    policy_workers: int = 1,
    allocations: Sequence[Allocation] = ALLOCATIONS,
    rollout_horizon: int = 1,
    rollout_discount: float = 1.0,
    collect_task_outcomes: bool = False,
    collect_research_archive: bool = False,
) -> CounterfactualDataset:
    """Generate aligned labels for every allocation at the same MOS contexts.

    A reference controller determines the physical trajectory. Every allocation
    receives the same history and independently performs inference at ``t+1``.
    With the default one-step horizon, supervision is the selected policy's
    depth-normalized PyAIF G value at ``t+1``. For a longer horizon, each
    candidate controls its own copied environment and agent, and supervision is
    the discounted mean of those normalized G values. Raw first-step G and its
    decomposed terms are retained for auditing.
    """

    if max_steps < 2:
        raise ValueError("max_steps must be at least 2")
    if branch_stride < 1:
        raise ValueError("branch_stride must be positive")
    if rollout_horizon < 1:
        raise ValueError("rollout_horizon must be positive")
    if not 0 < rollout_discount <= 1:
        raise ValueError("rollout_discount must lie in (0, 1]")
    label_schema = (
        ONE_STEP_LABEL_SCHEMA if rollout_horizon == 1 else ACCUMULATED_LABEL_SCHEMA
    )
    allocation_tuple = tuple(allocations)
    if not allocation_tuple or len(set(allocation_tuple)) != len(allocation_tuple):
        raise ValueError("allocations must be nonempty and unique")
    if reference_allocation not in allocation_tuple:
        raise ValueError("reference_allocation must be included in allocations")

    mos = _mos_imports()
    spatial_rows: list[np.ndarray] = []
    context_vectors: list[np.ndarray] = []
    normalized_g_rows: list[np.ndarray] = []
    raw_g_rows: list[np.ndarray] = []
    risk_rows: list[np.ndarray] = []
    ambiguity_rows: list[np.ndarray] = []
    information_gain_rows: list[np.ndarray] = []
    state_accuracy_rows: list[np.ndarray] = []
    state_complexity_rows: list[np.ndarray] = []
    uniform_risk_rows: list[np.ndarray] = []
    uniform_ambiguity_rows: list[np.ndarray] = []
    uniform_information_gain_rows: list[np.ndarray] = []
    success_rows: list[np.ndarray] = []
    task_cost_rows: list[np.ndarray] = []
    compute_rows: list[np.ndarray] = []
    switch_rows: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []
    context_ids: list[str] = []
    context_records: list[dict[str, Any]] = []
    branch_records: list[dict[str, Any]] = []
    trajectory_records: list[dict[str, Any]] = []
    research_rows: dict[str, list[np.ndarray]] = (
        {name: [] for name in _empty_research_arrays(len(allocation_tuple))}
        if collect_research_archive
        else {}
    )

    for instance_seed in instance_seeds:
        instance = mos["sample_mos_instance"](int(instance_seed))
        reference_agent = mos["build_mos_agent"](
            mos["MOSAgentConfig"](
                target_resolution=reference_allocation.resolution,
                action_depth=reference_allocation.depth,
                message_passing_iterations=message_passing_iterations,
                policy_workers=policy_workers,
            ),
            layout=instance.layout,
        )
        reference_agent.reset()

        quantile_seed = (
            instance.layout.map_seed * 1_000_003
            + instance.target_seed * 1009
            + instance.observation_seed
        )
        quantiles = np.random.default_rng(quantile_seed).random(
            max_steps + rollout_horizon + 1
        )
        environment = mos["MOSEnvironment"](
            layout=instance.layout,
            target=instance.target,
            observation_seed=instance.observation_seed,
        )
        observation = environment.reset(noise_quantile=float(quantiles[0]))
        previous_action = None
        decision = 0
        decisions = {
            reference_allocation: _infer_decision(
                reference_agent, observation, previous_action, decision, mos
            )
        }

        while decision < max_steps:
            position = environment.position
            reference_decision = decisions[reference_allocation]
            predicted_position = instance.layout.move(position, reference_decision.action)
            features: MOSFeatureBatch = build_mos_features(
                agent=reference_agent,
                allocation=reference_allocation,
                layout=instance.layout,
                selected_action=int(reference_decision.action),
                next_robot_position=predicted_position,
                found_flags=np.asarray([0.0]),
            )
            next_observation, reference_success = environment.step(
                reference_decision.action,
                noise_quantile=float(quantiles[decision + 1]),
            )
            trajectory_records.append(
                {
                    "instance_seed": int(instance_seed),
                    "decision": decision,
                    "x": position[0],
                    "y": position[1],
                    "action": reference_decision.action.name,
                    "next_x": environment.position[0],
                    "next_y": environment.position[1],
                    "success": bool(reference_success),
                    "observation": json.dumps([int(value) for value in observation]),
                    "next_observation": json.dumps([int(value) for value in next_observation]),
                }
            )
            if reference_success or decision + 1 >= max_steps:
                break

            next_decision_index = decision + 1
            record_counterfactuals = decision % branch_stride == 0
            evaluated_allocations = (
                allocation_tuple if record_counterfactuals else (reference_allocation,)
            )
            candidate_agents = {}
            switch_times = {}
            for allocation in evaluated_allocations:
                candidate_agents[allocation], switch_times[allocation] = _timed(
                    partial(
                        _build_switched_agent,
                        source_agent=reference_agent,
                        source_allocation=reference_allocation,
                        target_allocation=allocation,
                        layout=instance.layout,
                        executed_action=reference_decision.action,
                        next_time_step=next_decision_index,
                        message_passing_iterations=message_passing_iterations,
                        policy_workers=policy_workers,
                        mos=mos,
                    )
                )
                if allocation == reference_allocation:
                    # A deployed controller carries its existing agent forward
                    # when the allocation is unchanged. Rebuilding here is only
                    # needed to keep counterfactual branches isolated.
                    switch_times[allocation] = 0.0
            next_decisions = {
                allocation: _infer_decision(
                    candidate_agents[allocation],
                    next_observation,
                    reference_decision.action,
                    next_decision_index,
                    mos,
                )
                for allocation in evaluated_allocations
            }
            if record_counterfactuals:
                context_id = f"instance-{instance_seed}-decision-{next_decision_index}"
                rollout_results = {
                    allocation: _rollout_candidate(
                        agent=copy.deepcopy(candidate_agents[allocation]),
                        environment=copy.deepcopy(environment),
                        first_decision=next_decisions[allocation],
                        allocation=allocation,
                        first_time_step=next_decision_index,
                        quantiles=quantiles,
                        horizon=rollout_horizon,
                        discount=rollout_discount,
                        mos=mos,
                    )
                    for allocation in allocation_tuple
                }
                normalized_g = np.asarray(
                    [rollout_results[a]["discounted_g_mean"] for a in allocation_tuple],
                    dtype=np.float32,
                )
                raw_g = np.asarray(
                    [next_decisions[a].selected_policy_g for a in allocation_tuple],
                    dtype=np.float32,
                )
                risk = np.asarray(
                    [next_decisions[a].risk for a in allocation_tuple], dtype=np.float32
                )
                ambiguity = np.asarray(
                    [next_decisions[a].ambiguity for a in allocation_tuple], dtype=np.float32
                )
                information_gain = np.asarray(
                    [next_decisions[a].information_gain for a in allocation_tuple],
                    dtype=np.float32,
                )
                state_accuracy = np.asarray(
                    [next_decisions[a].state_accuracy for a in allocation_tuple],
                    dtype=np.float32,
                )
                state_complexity = np.asarray(
                    [next_decisions[a].state_complexity for a in allocation_tuple],
                    dtype=np.float32,
                )
                uniform_risk = np.asarray(
                    [next_decisions[a].uniform_risk for a in allocation_tuple],
                    dtype=np.float32,
                )
                uniform_ambiguity = np.asarray(
                    [next_decisions[a].uniform_ambiguity for a in allocation_tuple],
                    dtype=np.float32,
                )
                uniform_information_gain = np.asarray(
                    [next_decisions[a].uniform_information_gain for a in allocation_tuple],
                    dtype=np.float32,
                )
                compute = np.asarray(
                    [next_decisions[a].total_ms for a in allocation_tuple],
                    dtype=np.float32,
                )
                switching = np.asarray(
                    [switch_times[a] for a in allocation_tuple], dtype=np.float32
                )
                actions = np.asarray(
                    [int(next_decisions[a].action) for a in allocation_tuple], dtype=np.int8
                )
                outcomes_by_action: dict[int, dict[str, Any]] = {}
                if collect_task_outcomes:
                    for candidate in next_decisions.values():
                        action_index = int(candidate.action)
                        if action_index not in outcomes_by_action:
                            outcomes_by_action[action_index] = _continue_after_intervention(
                                agent=candidate_agents[reference_allocation],
                                instance=instance,
                                position=environment.position,
                                action=candidate.action,
                                decision=next_decision_index,
                                max_steps=max_steps,
                                quantiles=quantiles,
                                mos=mos,
                            )
                    successes = np.asarray(
                        [
                            outcomes_by_action[int(next_decisions[a].action)][
                                "eventual_success"
                            ]
                            for a in allocation_tuple
                        ],
                        dtype=np.float32,
                    )
                    task_costs = np.asarray(
                        [
                            outcomes_by_action[int(next_decisions[a].action)][
                                "remaining_task_cost"
                            ]
                            for a in allocation_tuple
                        ],
                        dtype=np.float32,
                    )
                for allocation in allocation_tuple:
                    candidate = next_decisions[allocation]
                    rollout = rollout_results[allocation]
                    branch_record = {
                            "context_id": context_id,
                            "candidate_resolution": allocation.resolution,
                            "candidate_depth": allocation.depth,
                            "candidate_action": candidate.action.name,
                            "normalized_g": rollout["discounted_g_mean"],
                            "immediate_normalized_g": rollout["first_normalized_g"],
                            "rollout_horizon": rollout_horizon,
                            "rollout_discount": rollout_discount,
                            "rollout_steps": rollout["rollout_steps"],
                            "rollout_success": rollout["success_within_horizon"],
                            "rollout_task_cost": rollout["task_cost"],
                            "rollout_actions": rollout["actions"],
                            "raw_g": candidate.selected_policy_g,
                            "risk": candidate.risk,
                            "ambiguity": candidate.ambiguity,
                            "information_gain": candidate.information_gain,
                            "state_accuracy": candidate.state_accuracy,
                            "state_complexity": candidate.state_complexity,
                            "uniform_risk": candidate.uniform_risk,
                            "uniform_ambiguity": candidate.uniform_ambiguity,
                            "uniform_information_gain": candidate.uniform_information_gain,
                            "inference_ms": candidate.total_ms,
                            "switch_ms": switch_times[allocation],
                        }
                    if collect_task_outcomes:
                        branch_record.update(outcomes_by_action[int(candidate.action)])
                    branch_records.append(branch_record)
                spatial_rows.append(features.spatial.tensor)
                context_vectors.append(features.context)
                normalized_g_rows.append(normalized_g)
                raw_g_rows.append(raw_g)
                risk_rows.append(risk)
                ambiguity_rows.append(ambiguity)
                information_gain_rows.append(information_gain)
                state_accuracy_rows.append(state_accuracy)
                state_complexity_rows.append(state_complexity)
                uniform_risk_rows.append(uniform_risk)
                uniform_ambiguity_rows.append(uniform_ambiguity)
                uniform_information_gain_rows.append(uniform_information_gain)
                if collect_task_outcomes:
                    success_rows.append(successes)
                    task_cost_rows.append(task_costs)
                compute_rows.append(compute)
                switch_rows.append(switching)
                action_rows.append(actions)
                context_ids.append(context_id)
                context_records.append(
                    {
                        "context_id": context_id,
                        "instance_seed": int(instance_seed),
                        "feature_decision": decision,
                        "candidate_decision": next_decision_index,
                        "x": position[0],
                        "y": position[1],
                        "next_x": environment.position[0],
                        "next_y": environment.position[1],
                        "reference_action": reference_decision.action.name,
                        "reference_resolution": reference_allocation.resolution,
                        "reference_depth": reference_allocation.depth,
                        "predicted_posterior_l1": features.predicted_posterior_l1,
                        "unique_candidate_actions": len(set(actions.tolist())),
                    }
                )
                if collect_research_archive:
                    research = _research_context_arrays(
                        decisions=next_decisions,
                        agents=candidate_agents,
                        allocations=allocation_tuple,
                        layout=instance.layout,
                        position=environment.position,
                        observation=next_observation,
                    )
                    for name, values in research.items():
                        research_rows[name].append(values)

            observation = next_observation
            previous_action = reference_decision.action
            reference_agent = candidate_agents[reference_allocation]
            decisions = next_decisions
            decision = next_decision_index

    if not spatial_rows:
        return _empty_dataset(
            allocation_tuple,
            label_schema=label_schema,
            rollout_horizon=rollout_horizon,
            rollout_discount=rollout_discount,
            collect_task_outcomes=collect_task_outcomes,
            collect_research_archive=collect_research_archive,
        )
    return CounterfactualDataset(
        spatial=np.stack(spatial_rows).astype(np.float32),
        context=np.stack(context_vectors).astype(np.float32),
        normalized_g=np.stack(normalized_g_rows).astype(np.float32),
        raw_g=np.stack(raw_g_rows).astype(np.float32),
        risk=np.stack(risk_rows).astype(np.float32),
        ambiguity=np.stack(ambiguity_rows).astype(np.float32),
        information_gain=np.stack(information_gain_rows).astype(np.float32),
        compute_ms=np.stack(compute_rows).astype(np.float32),
        switch_ms=np.stack(switch_rows).astype(np.float32),
        candidate_actions=np.stack(action_rows).astype(np.int8),
        context_ids=tuple(context_ids),
        allocations=allocation_tuple,
        contexts=tuple(context_records),
        branches=tuple(branch_records),
        trajectory=tuple(trajectory_records),
        label_schema=label_schema,
        rollout_horizon=rollout_horizon,
        rollout_discount=rollout_discount,
        state_accuracy=np.stack(state_accuracy_rows).astype(np.float32),
        state_complexity=np.stack(state_complexity_rows).astype(np.float32),
        uniform_risk=np.stack(uniform_risk_rows).astype(np.float32),
        uniform_ambiguity=np.stack(uniform_ambiguity_rows).astype(np.float32),
        uniform_information_gain=np.stack(uniform_information_gain_rows).astype(np.float32),
        success=(np.stack(success_rows).astype(np.float32) if collect_task_outcomes else None),
        task_cost=(
            np.stack(task_cost_rows).astype(np.float32) if collect_task_outcomes else None
        ),
        task_outcome_schema=TASK_OUTCOME_SCHEMA if collect_task_outcomes else None,
        research_arrays=(
            {
                name: np.stack(rows).astype(rows[0].dtype, copy=False)
                for name, rows in research_rows.items()
            }
            if collect_research_archive
            else None
        ),
    )


def _write_csv(path: Path, rows: tuple[dict[str, Any], ...]) -> None:
    if not rows:
        path.unlink(missing_ok=True)
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> tuple[dict[str, Any], ...]:
    if not path.is_file():
        return ()
    with path.open(newline="", encoding="utf-8") as stream:
        return tuple(dict(row) for row in csv.DictReader(stream))


def load_generated_counterfactuals(output_dir: Path) -> CounterfactualDataset:
    """Load one dataset written by :func:`save_counterfactual_dataset`."""

    output_dir = Path(output_dir)
    research_path = output_dir / "research_archive.npz"
    research_arrays = None
    if research_path.is_file():
        with np.load(research_path, allow_pickle=False) as research:
            schema = str(research["archive_schema"].item())
            if schema != RESEARCH_ARCHIVE_SCHEMA:
                raise ValueError(f"unsupported research archive schema: {schema}")
            research_arrays = {
                name: research[name].copy()
                for name in research.files
                if name != "archive_schema"
            }
    with np.load(output_dir / "training_data.npz", allow_pickle=False) as archive:
        allocations = tuple(
            Allocation(int(resolution), int(depth))
            for resolution, depth in zip(archive["resolutions"], archive["depths"], strict=True)
        )
        return CounterfactualDataset(
            spatial=archive["spatial"].copy(),
            context=archive["context"].copy(),
            normalized_g=archive["normalized_g"].copy(),
            raw_g=archive["raw_g"].copy(),
            risk=archive["risk"].copy(),
            ambiguity=archive["ambiguity"].copy(),
            information_gain=archive["information_gain"].copy(),
            compute_ms=archive["compute_ms"].copy(),
            switch_ms=(
                archive["switch_ms"].copy()
                if "switch_ms" in archive.files
                else np.zeros_like(archive["compute_ms"])
            ),
            candidate_actions=archive["candidate_actions"].copy(),
            context_ids=tuple(archive["context_ids"].astype(str).tolist()),
            allocations=allocations,
            contexts=_read_csv(output_dir / "contexts.csv"),
            branches=_read_csv(output_dir / "branches.csv"),
            trajectory=_read_csv(output_dir / "reference_trajectory.csv"),
            label_schema=(
                str(archive["label_schema"].item())
                if "label_schema" in archive.files
                else ONE_STEP_LABEL_SCHEMA
            ),
            rollout_horizon=(
                int(archive["rollout_horizon"].item())
                if "rollout_horizon" in archive.files
                else 1
            ),
            rollout_discount=(
                float(archive["rollout_discount"].item())
                if "rollout_discount" in archive.files
                else 1.0
            ),
            state_accuracy=(
                archive["state_accuracy"].copy()
                if "state_accuracy" in archive.files
                else None
            ),
            state_complexity=(
                archive["state_complexity"].copy()
                if "state_complexity" in archive.files
                else None
            ),
            uniform_risk=(
                archive["uniform_risk"].copy() if "uniform_risk" in archive.files else None
            ),
            uniform_ambiguity=(
                archive["uniform_ambiguity"].copy()
                if "uniform_ambiguity" in archive.files
                else None
            ),
            uniform_information_gain=(
                archive["uniform_information_gain"].copy()
                if "uniform_information_gain" in archive.files
                else None
            ),
            success=(archive["success"].copy() if "success" in archive.files else None),
            task_cost=(archive["task_cost"].copy() if "task_cost" in archive.files else None),
            task_outcome_schema=(
                str(archive["task_outcome_schema"].item())
                if "task_outcome_schema" in archive.files
                else None
            ),
            research_arrays=research_arrays,
        )


def concatenate_counterfactual_datasets(
    datasets: Sequence[CounterfactualDataset],
) -> CounterfactualDataset:
    """Combine ordered per-instance shards into one training dataset."""

    values = tuple(datasets)
    if not values:
        raise ValueError("at least one dataset is required")
    allocations = values[0].allocations
    if any(dataset.allocations != allocations for dataset in values[1:]):
        raise ValueError("all datasets must use the same allocation order")
    metadata = {
        (dataset.label_schema, dataset.rollout_horizon, dataset.rollout_discount)
        for dataset in values
    }
    if len(metadata) != 1:
        raise ValueError("all datasets must use the same label schema and rollout settings")
    task_metadata = {
        (dataset.success is not None, dataset.task_outcome_schema) for dataset in values
    }
    if len(task_metadata) != 1:
        raise ValueError("all datasets must use the same task-outcome settings")
    state_metadata = {
        (
            dataset.state_accuracy is not None,
            dataset.state_complexity is not None,
            dataset.uniform_risk is not None,
            dataset.uniform_ambiguity is not None,
            dataset.uniform_information_gain is not None,
        )
        for dataset in values
    }
    if len(state_metadata) != 1:
        raise ValueError("all datasets must use the same state-value settings")
    research_metadata = {
        None if dataset.research_arrays is None else tuple(sorted(dataset.research_arrays))
        for dataset in values
    }
    if len(research_metadata) != 1:
        raise ValueError("all datasets must use the same research-archive schema")

    def concatenate(name: str) -> np.ndarray:
        return np.concatenate([np.asarray(getattr(dataset, name)) for dataset in values], axis=0)

    return CounterfactualDataset(
        spatial=concatenate("spatial").astype(np.float32),
        context=concatenate("context").astype(np.float32),
        normalized_g=concatenate("normalized_g").astype(np.float32),
        raw_g=concatenate("raw_g").astype(np.float32),
        risk=concatenate("risk").astype(np.float32),
        ambiguity=concatenate("ambiguity").astype(np.float32),
        information_gain=concatenate("information_gain").astype(np.float32),
        compute_ms=concatenate("compute_ms").astype(np.float32),
        switch_ms=concatenate("switch_ms").astype(np.float32),
        candidate_actions=concatenate("candidate_actions").astype(np.int8),
        context_ids=tuple(value for dataset in values for value in dataset.context_ids),
        allocations=allocations,
        contexts=tuple(value for dataset in values for value in dataset.contexts),
        branches=tuple(value for dataset in values for value in dataset.branches),
        trajectory=tuple(value for dataset in values for value in dataset.trajectory),
        label_schema=values[0].label_schema,
        rollout_horizon=values[0].rollout_horizon,
        rollout_discount=values[0].rollout_discount,
        state_accuracy=(
            concatenate("state_accuracy").astype(np.float32)
            if values[0].state_accuracy is not None
            else None
        ),
        state_complexity=(
            concatenate("state_complexity").astype(np.float32)
            if values[0].state_complexity is not None
            else None
        ),
        uniform_risk=(
            concatenate("uniform_risk").astype(np.float32)
            if values[0].uniform_risk is not None
            else None
        ),
        uniform_ambiguity=(
            concatenate("uniform_ambiguity").astype(np.float32)
            if values[0].uniform_ambiguity is not None
            else None
        ),
        uniform_information_gain=(
            concatenate("uniform_information_gain").astype(np.float32)
            if values[0].uniform_information_gain is not None
            else None
        ),
        success=(
            concatenate("success").astype(np.float32)
            if values[0].success is not None
            else None
        ),
        task_cost=(
            concatenate("task_cost").astype(np.float32)
            if values[0].task_cost is not None
            else None
        ),
        task_outcome_schema=values[0].task_outcome_schema,
        research_arrays=(
            {
                name: np.concatenate(
                    [dataset.research_arrays[name] for dataset in values], axis=0
                )
                for name in values[0].research_arrays
            }
            if values[0].research_arrays is not None
            else None
        ),
    )


def save_counterfactual_dataset(dataset: CounterfactualDataset, output_dir: Path) -> None:
    """Write compact training arrays and human-auditable CSV metadata."""

    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "spatial": dataset.spatial,
        "context": dataset.context,
        "normalized_g": dataset.normalized_g,
        "raw_g": dataset.raw_g,
        "risk": dataset.risk,
        "ambiguity": dataset.ambiguity,
        "information_gain": dataset.information_gain,
        "compute_ms": dataset.compute_ms,
        "switch_ms": dataset.switch_ms,
        "candidate_actions": dataset.candidate_actions,
        "context_ids": np.asarray(dataset.context_ids),
        "resolutions": np.asarray([item.resolution for item in dataset.allocations]),
        "depths": np.asarray([item.depth for item in dataset.allocations]),
        "label_schema": np.asarray(dataset.label_schema),
        "rollout_horizon": np.asarray(dataset.rollout_horizon),
        "rollout_discount": np.asarray(dataset.rollout_discount),
    }
    if dataset.state_accuracy is not None and dataset.state_complexity is not None:
        arrays.update(
            state_accuracy=dataset.state_accuracy,
            state_complexity=dataset.state_complexity,
        )
    if (
        dataset.uniform_risk is not None
        and dataset.uniform_ambiguity is not None
        and dataset.uniform_information_gain is not None
    ):
        arrays.update(
            uniform_risk=dataset.uniform_risk,
            uniform_ambiguity=dataset.uniform_ambiguity,
            uniform_information_gain=dataset.uniform_information_gain,
        )
    if dataset.success is not None and dataset.task_cost is not None:
        arrays.update(
            success=dataset.success,
            task_cost=dataset.task_cost,
            task_outcome_schema=np.asarray(dataset.task_outcome_schema),
        )
    np.savez_compressed(output_dir / "training_data.npz", **arrays)
    research_path = output_dir / "research_archive.npz"
    if dataset.research_arrays is not None:
        np.savez_compressed(
            research_path,
            archive_schema=np.asarray(RESEARCH_ARCHIVE_SCHEMA),
            **dataset.research_arrays,
        )
    else:
        research_path.unlink(missing_ok=True)
    _write_csv(output_dir / "contexts.csv", dataset.contexts)
    _write_csv(output_dir / "branches.csv", dataset.branches)
    _write_csv(output_dir / "reference_trajectory.csv", dataset.trajectory)
    summary = {
        "contexts": len(dataset.context_ids),
        "branches": len(dataset.branches),
        "trajectory_steps": len(dataset.trajectory),
        "spatial_shape": list(dataset.spatial.shape),
        "context_shape": list(dataset.context.shape),
        "label_schema": dataset.label_schema,
        "rollout_horizon": dataset.rollout_horizon,
        "rollout_discount": dataset.rollout_discount,
        "task_outcome_schema": dataset.task_outcome_schema,
        "research_archive_schema": (
            RESEARCH_ARCHIVE_SCHEMA if dataset.research_arrays is not None else None
        ),
        "research_arrays": (
            {
                name: list(values.shape) for name, values in dataset.research_arrays.items()
            }
            if dataset.research_arrays is not None
            else None
        ),
        "allocations": [
            {"resolution": item.resolution, "depth": item.depth} for item in dataset.allocations
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
