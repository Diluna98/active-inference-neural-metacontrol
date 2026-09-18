"""Matched-state counterfactual data generation for neural metacontrol."""

from __future__ import annotations

import copy
import csv
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .allocations import ALLOCATIONS, Allocation
from .beliefs import remap_posterior
from .mos_adapter import MOSFeatureBatch, build_mos_features

DEFAULT_REFERENCE_ALLOCATION = Allocation(20, 1)


@dataclass(frozen=True)
class CounterfactualDataset:
    """Training arrays plus auditable tabular records."""

    spatial: np.ndarray
    context: np.ndarray
    success: np.ndarray
    task_cost: np.ndarray
    compute_ms: np.ndarray
    candidate_actions: np.ndarray
    context_ids: tuple[str, ...]
    allocations: tuple[Allocation, ...]
    contexts: tuple[dict[str, Any], ...]
    branches: tuple[dict[str, Any], ...]
    trajectory: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class _Decision:
    action: Any
    state_ms: float
    policy_ms: float
    action_ms: float

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
    _, state_ms = _timed(agent.infer_states)
    _, policy_ms = _timed(agent.infer_policies)
    action, action_ms = _timed(lambda: mos["selected_mos_action"](agent))
    return _Decision(action, state_ms, policy_ms, action_ms)


def _step_cost(*, false_find: bool, collision: bool) -> float:
    return 1.0 + 5.0 * false_find + 0.5 * collision


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
    current = source_agent._receding_posterior
    candidate_core = getattr(candidate, "_agent", candidate)
    candidate_core._receding_posterior = [
        np.asarray(current[0], dtype=float).copy(),
        np.asarray(current[1], dtype=float).copy(),
        remap_posterior(
            np.asarray(current[2], dtype=float),
            source_allocation.resolution,
            target_allocation.resolution,
            layout.size,
        ).ravel(),
        np.asarray(current[3], dtype=float).copy(),
    ]
    candidate_core._receding_action = mos["mos_action_controls"](executed_action)
    candidate_core._current_time = next_time_step
    candidate_core._receding_stage = "ready"
    return candidate


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
    """Execute one candidate action, then hand control back to the reference agent."""

    environment = mos["MOSEnvironment"](
        layout=instance.layout,
        target=instance.target,
        observation_seed=instance.observation_seed,
    )
    environment.position = position
    controller = copy.deepcopy(agent)
    observation, success = environment.step(action, noise_quantile=float(quantiles[decision + 1]))
    false_find = observation[3] == mos["FindOutcome"].FALSE_FIND
    task_cost = _step_cost(false_find=false_find, collision=environment.last_collision)
    false_finds = int(false_find)
    collisions = int(environment.last_collision)
    steps = 1
    previous_action = action

    for next_decision in range(decision + 1, max_steps):
        if success:
            break
        inferred = _infer_decision(controller, observation, previous_action, next_decision, mos)
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
        "success": bool(success),
        "task_cost": float(task_cost),
        "steps": steps,
        "false_finds": false_finds,
        "collisions": collisions,
        "final_x": environment.position[0],
        "final_y": environment.position[1],
    }


def _empty_dataset(allocations: tuple[Allocation, ...]) -> CounterfactualDataset:
    return CounterfactualDataset(
        spatial=np.empty((0, 6, 20, 20), dtype=np.float32),
        context=np.empty((0, 17), dtype=np.float32),
        success=np.empty((0, len(allocations)), dtype=np.float32),
        task_cost=np.empty((0, len(allocations)), dtype=np.float32),
        compute_ms=np.empty((0, len(allocations)), dtype=np.float32),
        candidate_actions=np.empty((0, len(allocations)), dtype=np.int8),
        context_ids=(),
        allocations=allocations,
        contexts=(),
        branches=(),
        trajectory=(),
    )


def generate_mos_counterfactuals(
    *,
    instance_seeds: Sequence[int],
    reference_allocation: Allocation = DEFAULT_REFERENCE_ALLOCATION,
    max_steps: int = 50,
    branch_stride: int = 1,
    message_passing_iterations: int = 10,
    policy_workers: int = 1,
    allocations: Sequence[Allocation] = ALLOCATIONS,
) -> CounterfactualDataset:
    """Generate aligned labels for every allocation at the same MOS contexts.

    A reference controller determines the physical trajectory.  Every allocation
    receives exactly the same observation/action history.  At selected contexts,
    each allocation proposes the next action; each distinct physical action is
    evaluated once, after which control returns to a clone of the reference agent.
    """

    if max_steps < 2:
        raise ValueError("max_steps must be at least 2")
    if branch_stride < 1:
        raise ValueError("branch_stride must be positive")
    allocation_tuple = tuple(allocations)
    if not allocation_tuple or len(set(allocation_tuple)) != len(allocation_tuple):
        raise ValueError("allocations must be nonempty and unique")
    if reference_allocation not in allocation_tuple:
        raise ValueError("reference_allocation must be included in allocations")

    mos = _mos_imports()
    spatial_rows: list[np.ndarray] = []
    context_vectors: list[np.ndarray] = []
    success_rows: list[np.ndarray] = []
    task_cost_rows: list[np.ndarray] = []
    compute_rows: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []
    context_ids: list[str] = []
    context_records: list[dict[str, Any]] = []
    branch_records: list[dict[str, Any]] = []
    trajectory_records: list[dict[str, Any]] = []

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
        quantiles = np.random.default_rng(quantile_seed).random(max_steps + 1)
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
            candidate_agents = {
                allocation: _build_switched_agent(
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
                for allocation in allocation_tuple
            }
            next_decisions = {
                allocation: _infer_decision(
                    candidate_agents[allocation],
                    next_observation,
                    reference_decision.action,
                    next_decision_index,
                    mos,
                )
                for allocation in allocation_tuple
            }
            if decision % branch_stride == 0:
                context_id = f"instance-{instance_seed}-decision-{next_decision_index}"
                outcomes_by_action: dict[int, dict[str, Any]] = {}
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
                        branch_records.append(
                            {
                                "context_id": context_id,
                                "candidate_action": candidate.action.name,
                                **outcomes_by_action[action_index],
                            }
                        )

                successes = np.asarray(
                    [
                        outcomes_by_action[int(next_decisions[a].action)]["success"]
                        for a in allocation_tuple
                    ],
                    dtype=np.float32,
                )
                costs = np.asarray(
                    [
                        outcomes_by_action[int(next_decisions[a].action)]["task_cost"]
                        for a in allocation_tuple
                    ],
                    dtype=np.float32,
                )
                compute = np.asarray(
                    [next_decisions[a].total_ms for a in allocation_tuple],
                    dtype=np.float32,
                )
                actions = np.asarray(
                    [int(next_decisions[a].action) for a in allocation_tuple], dtype=np.int8
                )
                spatial_rows.append(features.spatial.tensor)
                context_vectors.append(features.context)
                success_rows.append(successes)
                task_cost_rows.append(costs)
                compute_rows.append(compute)
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
                        "unique_candidate_actions": len(outcomes_by_action),
                    }
                )

            observation = next_observation
            previous_action = reference_decision.action
            reference_agent = candidate_agents[reference_allocation]
            decisions = next_decisions
            decision = next_decision_index

    if not spatial_rows:
        return _empty_dataset(allocation_tuple)
    return CounterfactualDataset(
        spatial=np.stack(spatial_rows).astype(np.float32),
        context=np.stack(context_vectors).astype(np.float32),
        success=np.stack(success_rows).astype(np.float32),
        task_cost=np.stack(task_cost_rows).astype(np.float32),
        compute_ms=np.stack(compute_rows).astype(np.float32),
        candidate_actions=np.stack(action_rows).astype(np.int8),
        context_ids=tuple(context_ids),
        allocations=allocation_tuple,
        contexts=tuple(context_records),
        branches=tuple(branch_records),
        trajectory=tuple(trajectory_records),
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
    with np.load(output_dir / "training_data.npz", allow_pickle=False) as archive:
        allocations = tuple(
            Allocation(int(resolution), int(depth))
            for resolution, depth in zip(archive["resolutions"], archive["depths"], strict=True)
        )
        return CounterfactualDataset(
            spatial=archive["spatial"].copy(),
            context=archive["context"].copy(),
            success=archive["success"].copy(),
            task_cost=archive["task_cost"].copy(),
            compute_ms=archive["compute_ms"].copy(),
            candidate_actions=archive["candidate_actions"].copy(),
            context_ids=tuple(archive["context_ids"].astype(str).tolist()),
            allocations=allocations,
            contexts=_read_csv(output_dir / "contexts.csv"),
            branches=_read_csv(output_dir / "branches.csv"),
            trajectory=_read_csv(output_dir / "reference_trajectory.csv"),
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

    def concatenate(name: str) -> np.ndarray:
        return np.concatenate([np.asarray(getattr(dataset, name)) for dataset in values], axis=0)

    return CounterfactualDataset(
        spatial=concatenate("spatial").astype(np.float32),
        context=concatenate("context").astype(np.float32),
        success=concatenate("success").astype(np.float32),
        task_cost=concatenate("task_cost").astype(np.float32),
        compute_ms=concatenate("compute_ms").astype(np.float32),
        candidate_actions=concatenate("candidate_actions").astype(np.int8),
        context_ids=tuple(value for dataset in values for value in dataset.context_ids),
        allocations=allocations,
        contexts=tuple(value for dataset in values for value in dataset.contexts),
        branches=tuple(value for dataset in values for value in dataset.branches),
        trajectory=tuple(value for dataset in values for value in dataset.trajectory),
    )


def save_counterfactual_dataset(dataset: CounterfactualDataset, output_dir: Path) -> None:
    """Write compact training arrays and human-auditable CSV metadata."""

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "training_data.npz",
        spatial=dataset.spatial,
        context=dataset.context,
        success=dataset.success,
        task_cost=dataset.task_cost,
        compute_ms=dataset.compute_ms,
        candidate_actions=dataset.candidate_actions,
        context_ids=np.asarray(dataset.context_ids),
        resolutions=np.asarray([item.resolution for item in dataset.allocations]),
        depths=np.asarray([item.depth for item in dataset.allocations]),
    )
    _write_csv(output_dir / "contexts.csv", dataset.contexts)
    _write_csv(output_dir / "branches.csv", dataset.branches)
    _write_csv(output_dir / "reference_trajectory.csv", dataset.trajectory)
    summary = {
        "contexts": len(dataset.context_ids),
        "branches": len(dataset.branches),
        "trajectory_steps": len(dataset.trajectory),
        "spatial_shape": list(dataset.spatial.shape),
        "context_shape": list(dataset.context.shape),
        "allocations": [
            {"resolution": item.resolution, "depth": item.depth} for item in dataset.allocations
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
