"""Counterfactual transition data for sequential meta-level Q learning."""

from __future__ import annotations

import copy
import csv
import json
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from .allocations import ALLOCATIONS, Allocation
from .counterfactuals import (
    _build_switched_agent,
    _infer_decision,
    _mos_imports,
    _step_cost,
    _timed,
)
from .meta_q_features import (
    META_Q_CONTEXT_FEATURES,
    META_Q_FEATURE_SCHEMA,
    META_Q_SPATIAL_CHANNELS,
    build_meta_q_features,
)

META_Q_TRANSITION_SCHEMA = "mos-meta-q-counterfactual-transitions-v1"


@dataclass(frozen=True)
class MetaQTransitionDataset:
    """Every candidate allocation's transition from matched controller states."""

    spatial: np.ndarray
    context: np.ndarray
    next_spatial: np.ndarray
    next_context: np.ndarray
    reward: np.ndarray
    terminal: np.ndarray
    compute_ms: np.ndarray
    switch_required: np.ndarray
    candidate_actions: np.ndarray
    instance_seeds: np.ndarray
    source_resolution: np.ndarray
    source_depth: np.ndarray
    decision_indices: np.ndarray
    allocations: tuple[Allocation, ...] = ALLOCATIONS
    schema: str = META_Q_TRANSITION_SCHEMA
    feature_schema: str = META_Q_FEATURE_SCHEMA

    def __post_init__(self) -> None:
        samples = int(np.asarray(self.spatial).shape[0])
        candidates = len(self.allocations)
        expected_spatial = (samples, META_Q_SPATIAL_CHANNELS, 20, 20)
        expected_context = (samples, META_Q_CONTEXT_FEATURES)
        expected_candidate_spatial = (
            samples,
            candidates,
            META_Q_SPATIAL_CHANNELS,
            20,
            20,
        )
        expected_candidate_context = (samples, candidates, META_Q_CONTEXT_FEATURES)
        if np.asarray(self.spatial).shape != expected_spatial:
            raise ValueError(f"spatial must have shape {expected_spatial}")
        if np.asarray(self.context).shape != expected_context:
            raise ValueError(f"context must have shape {expected_context}")
        if np.asarray(self.next_spatial).shape != expected_candidate_spatial:
            raise ValueError(f"next_spatial must have shape {expected_candidate_spatial}")
        if np.asarray(self.next_context).shape != expected_candidate_context:
            raise ValueError(f"next_context must have shape {expected_candidate_context}")
        for name in (
            "reward",
            "terminal",
            "compute_ms",
            "switch_required",
            "candidate_actions",
        ):
            if np.asarray(getattr(self, name)).shape != (samples, candidates):
                raise ValueError(f"{name} must have shape {(samples, candidates)}")
        for name in (
            "instance_seeds",
            "source_resolution",
            "source_depth",
            "decision_indices",
        ):
            if np.asarray(getattr(self, name)).shape != (samples,):
                raise ValueError(f"{name} must have shape {(samples,)}")
        for name in ("spatial", "context", "next_spatial", "next_context", "reward", "compute_ms"):
            if not np.all(np.isfinite(getattr(self, name))):
                raise ValueError(f"{name} must contain finite values")
        if np.any(np.asarray(self.compute_ms) < 0):
            raise ValueError("compute_ms cannot be negative")

    def subset(self, indices: np.ndarray) -> MetaQTransitionDataset:
        values = np.asarray(indices, dtype=int)
        return MetaQTransitionDataset(
            spatial=self.spatial[values],
            context=self.context[values],
            next_spatial=self.next_spatial[values],
            next_context=self.next_context[values],
            reward=self.reward[values],
            terminal=self.terminal[values],
            compute_ms=self.compute_ms[values],
            switch_required=self.switch_required[values],
            candidate_actions=self.candidate_actions[values],
            instance_seeds=self.instance_seeds[values],
            source_resolution=self.source_resolution[values],
            source_depth=self.source_depth[values],
            decision_indices=self.decision_indices[values],
            allocations=self.allocations,
            schema=self.schema,
            feature_schema=self.feature_schema,
        )


def _empty_rows(allocations: tuple[Allocation, ...]) -> MetaQTransitionDataset:
    count = len(allocations)
    return MetaQTransitionDataset(
        spatial=np.empty((0, META_Q_SPATIAL_CHANNELS, 20, 20), dtype=np.float32),
        context=np.empty((0, META_Q_CONTEXT_FEATURES), dtype=np.float32),
        next_spatial=np.empty((0, count, META_Q_SPATIAL_CHANNELS, 20, 20), dtype=np.float32),
        next_context=np.empty((0, count, META_Q_CONTEXT_FEATURES), dtype=np.float32),
        reward=np.empty((0, count), dtype=np.float32),
        terminal=np.empty((0, count), dtype=bool),
        compute_ms=np.empty((0, count), dtype=np.float32),
        switch_required=np.empty((0, count), dtype=bool),
        candidate_actions=np.empty((0, count), dtype=np.int8),
        instance_seeds=np.empty((0,), dtype=np.int64),
        source_resolution=np.empty((0,), dtype=np.int16),
        source_depth=np.empty((0,), dtype=np.int8),
        decision_indices=np.empty((0,), dtype=np.int16),
        allocations=allocations,
    )


def generate_meta_q_transitions(
    *,
    instance_seeds: Sequence[int],
    source_allocations: Sequence[Allocation] = ALLOCATIONS,
    allocations: Sequence[Allocation] = ALLOCATIONS,
    max_steps: int = 50,
    message_passing_iterations: int = 10,
    policy_workers: int = 1,
    success_reward: float = 0.0,
    failure_penalty: float | None = None,
    exploration_switch_probability: float = 0.0,
    initial_source_allocations: Sequence[Allocation] | None = None,
    progress: Callable[[str], None] | None = None,
) -> tuple[MetaQTransitionDataset, tuple[dict[str, Any], ...]]:
    """Generate full-action transitions along round-robin fixed trajectories.

    A controller state is captured after the current physical action produces its
    realized observation, exactly when the deployed metacontroller is called.
    Every candidate then performs the following task inference, chooses a physical
    action, and advances an isolated environment copy by one step.  Thus each row
    supplies all allocation actions for one matched state.
    """

    allocation_tuple = tuple(allocations)
    source_tuple = tuple(source_allocations)
    initial_source_tuple = (
        None if initial_source_allocations is None else tuple(initial_source_allocations)
    )
    if max_steps < 2:
        raise ValueError("max_steps must be at least 2")
    if not allocation_tuple or len(set(allocation_tuple)) != len(allocation_tuple):
        raise ValueError("allocations must be nonempty and unique")
    if not source_tuple or not set(source_tuple).issubset(set(allocation_tuple)):
        raise ValueError("source_allocations must be a nonempty subset of allocations")
    if initial_source_tuple is not None:
        if len(initial_source_tuple) != len(tuple(instance_seeds)):
            raise ValueError("initial_source_allocations must match instance_seeds")
        if not set(initial_source_tuple).issubset(set(source_tuple)):
            raise ValueError("initial source allocations must belong to source_allocations")
    if failure_penalty is None:
        failure_penalty = 2.0 * max_steps
    if failure_penalty < 0:
        raise ValueError("failure_penalty cannot be negative")
    if not 0 <= exploration_switch_probability <= 1:
        raise ValueError("exploration_switch_probability must lie in [0, 1]")

    mos = _mos_imports()
    spatial_rows: list[np.ndarray] = []
    context_rows: list[np.ndarray] = []
    next_spatial_rows: list[np.ndarray] = []
    next_context_rows: list[np.ndarray] = []
    reward_rows: list[np.ndarray] = []
    terminal_rows: list[np.ndarray] = []
    compute_rows: list[np.ndarray] = []
    switch_rows: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []
    seed_rows: list[int] = []
    source_resolution_rows: list[int] = []
    source_depth_rows: list[int] = []
    decision_rows: list[int] = []
    audit_rows: list[dict[str, Any]] = []

    seed_tuple = tuple(int(value) for value in instance_seeds)
    for seed_index, instance_seed in enumerate(seed_tuple):
        source_allocation = (
            source_tuple[seed_index % len(source_tuple)]
            if initial_source_tuple is None
            else initial_source_tuple[seed_index]
        )
        instance_start = len(spatial_rows)
        if progress is not None:
            progress(
                f"[generate {seed_index + 1}/{len(seed_tuple)}] "
                f"instance={instance_seed} "
                f"source=g{source_allocation.resolution}_T{source_allocation.depth}"
            )
        instance = mos["sample_mos_instance"](int(instance_seed))
        environment = mos["MOSEnvironment"](
            layout=instance.layout,
            target=instance.target,
            observation_seed=instance.observation_seed,
        )
        source_agent = mos["build_mos_agent"](
            mos["MOSAgentConfig"](
                target_resolution=source_allocation.resolution,
                action_depth=source_allocation.depth,
                message_passing_iterations=message_passing_iterations,
                policy_workers=policy_workers,
            ),
            layout=instance.layout,
        )
        source_agent.reset()
        quantile_seed = (
            instance.layout.map_seed * 1_000_003
            + instance.target_seed * 1009
            + instance.observation_seed
        )
        quantiles = np.random.default_rng(quantile_seed).random(max_steps + 1)
        exploration_rng = np.random.default_rng(quantile_seed + 7_919)
        observation = environment.reset(noise_quantile=float(quantiles[0]))
        previous_action = None
        source_decision = _infer_decision(source_agent, observation, previous_action, 0, mos)

        for decision_index in range(max_steps - 1):
            position = environment.position
            predicted_position = instance.layout.move(position, source_decision.action)
            next_observation, source_success = environment.step(
                source_decision.action,
                noise_quantile=float(quantiles[decision_index + 1]),
            )
            if source_success:
                break

            current = build_meta_q_features(
                agent=source_agent,
                allocation=source_allocation,
                layout=instance.layout,
                selected_action=int(source_decision.action),
                next_robot_position=predicted_position,
                realized_observation=next_observation,
            )
            next_decision_index = decision_index + 1
            candidate_agents: dict[Allocation, Any] = {}
            candidate_decisions: dict[Allocation, Any] = {}
            switch_times: dict[Allocation, float] = {}
            for allocation in allocation_tuple:
                candidate_agents[allocation], switch_times[allocation] = _timed(
                    partial(
                        _build_switched_agent,
                        source_agent=source_agent,
                        source_allocation=source_allocation,
                        target_allocation=allocation,
                        layout=instance.layout,
                        executed_action=source_decision.action,
                        next_time_step=next_decision_index,
                        message_passing_iterations=message_passing_iterations,
                        policy_workers=policy_workers,
                        mos=mos,
                    )
                )
                if allocation == source_allocation:
                    switch_times[allocation] = 0.0
                candidate_decisions[allocation] = _infer_decision(
                    candidate_agents[allocation],
                    next_observation,
                    source_decision.action,
                    next_decision_index,
                    mos,
                )

            row_next_spatial = []
            row_next_context = []
            row_reward = []
            row_terminal = []
            row_compute = []
            row_switch = []
            row_actions = []
            for allocation in allocation_tuple:
                candidate = candidate_decisions[allocation]
                branch_environment = copy.deepcopy(environment)
                branch_observation, branch_success = branch_environment.step(
                    candidate.action,
                    noise_quantile=float(quantiles[next_decision_index + 1]),
                )
                false_find = branch_observation[3] == mos["FindOutcome"].FALSE_FIND
                step_cost = _step_cost(
                    false_find=false_find,
                    collision=branch_environment.last_collision,
                )
                truncated = next_decision_index + 1 >= max_steps
                terminal = bool(branch_success or truncated)
                reward = -float(step_cost)
                if branch_success:
                    reward += float(success_reward)
                elif truncated:
                    reward -= float(failure_penalty)
                next_state = build_meta_q_features(
                    agent=candidate_agents[allocation],
                    allocation=allocation,
                    layout=instance.layout,
                    selected_action=int(candidate.action),
                    next_robot_position=branch_environment.position,
                    realized_observation=branch_observation,
                )
                row_next_spatial.append(next_state.spatial)
                row_next_context.append(next_state.context)
                row_reward.append(reward)
                row_terminal.append(terminal)
                row_compute.append(candidate.total_ms)
                row_switch.append(allocation != source_allocation)
                row_actions.append(int(candidate.action))
                audit_rows.append(
                    {
                        "instance_seed": int(instance_seed),
                        "decision": next_decision_index,
                        "source": f"g{source_allocation.resolution}_T{source_allocation.depth}",
                        "candidate": f"g{allocation.resolution}_T{allocation.depth}",
                        "x": int(environment.position[0]),
                        "y": int(environment.position[1]),
                        "action": candidate.action.name,
                        "next_x": int(branch_environment.position[0]),
                        "next_y": int(branch_environment.position[1]),
                        "reward": reward,
                        "terminal": terminal,
                        "success": bool(branch_success),
                        "compute_ms": candidate.total_ms,
                        "switch_setup_ms": switch_times[allocation],
                    }
                )

            spatial_rows.append(current.spatial)
            context_rows.append(current.context)
            next_spatial_rows.append(np.stack(row_next_spatial))
            next_context_rows.append(np.stack(row_next_context))
            reward_rows.append(np.asarray(row_reward, dtype=np.float32))
            terminal_rows.append(np.asarray(row_terminal, dtype=bool))
            compute_rows.append(np.asarray(row_compute, dtype=np.float32))
            switch_rows.append(np.asarray(row_switch, dtype=bool))
            action_rows.append(np.asarray(row_actions, dtype=np.int8))
            seed_rows.append(int(instance_seed))
            source_resolution_rows.append(source_allocation.resolution)
            source_depth_rows.append(source_allocation.depth)
            decision_rows.append(next_decision_index)

            if progress is not None and (next_decision_index == 1 or next_decision_index % 5 == 0):
                progress(
                    f"  instance={instance_seed} decision={next_decision_index} "
                    f"transitions={len(spatial_rows)} "
                    f"branches={len(spatial_rows) * len(allocation_tuple)}"
                )

            continuation = source_allocation
            if exploration_rng.random() < exploration_switch_probability:
                continuation = source_tuple[int(exploration_rng.integers(len(source_tuple)))]
            source_agent = candidate_agents[continuation]
            source_decision = candidate_decisions[continuation]
            source_allocation = continuation
            observation = next_observation
            previous_action = source_decision.action

        if progress is not None:
            added = len(spatial_rows) - instance_start
            progress(
                f"[complete {seed_index + 1}/{len(seed_tuple)}] "
                f"instance={instance_seed} contexts={added} "
                f"branches={added * len(allocation_tuple)}"
            )

    if not spatial_rows:
        return _empty_rows(allocation_tuple), tuple(audit_rows)
    dataset = MetaQTransitionDataset(
        spatial=np.stack(spatial_rows).astype(np.float32),
        context=np.stack(context_rows).astype(np.float32),
        next_spatial=np.stack(next_spatial_rows).astype(np.float32),
        next_context=np.stack(next_context_rows).astype(np.float32),
        reward=np.stack(reward_rows).astype(np.float32),
        terminal=np.stack(terminal_rows).astype(bool),
        compute_ms=np.stack(compute_rows).astype(np.float32),
        switch_required=np.stack(switch_rows).astype(bool),
        candidate_actions=np.stack(action_rows).astype(np.int8),
        instance_seeds=np.asarray(seed_rows, dtype=np.int64),
        source_resolution=np.asarray(source_resolution_rows, dtype=np.int16),
        source_depth=np.asarray(source_depth_rows, dtype=np.int8),
        decision_indices=np.asarray(decision_rows, dtype=np.int16),
        allocations=allocation_tuple,
    )
    return dataset, tuple(audit_rows)


def save_meta_q_transitions(
    dataset: MetaQTransitionDataset,
    output_dir: Path,
    *,
    audit_rows: Sequence[dict[str, Any]] = (),
) -> None:
    """Write one transition dataset and human-readable provenance."""

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "transitions.npz",
        # Spatial tensors dominate archive size. Half precision is ample for
        # probabilities and map masks; training converts each batch to float32.
        spatial=dataset.spatial.astype(np.float16),
        context=dataset.context,
        next_spatial=dataset.next_spatial.astype(np.float16),
        next_context=dataset.next_context,
        reward=dataset.reward,
        terminal=dataset.terminal,
        compute_ms=dataset.compute_ms,
        switch_required=dataset.switch_required,
        candidate_actions=dataset.candidate_actions,
        instance_seeds=dataset.instance_seeds,
        source_resolution=dataset.source_resolution,
        source_depth=dataset.source_depth,
        decision_indices=dataset.decision_indices,
        resolutions=np.asarray([a.resolution for a in dataset.allocations], dtype=np.int16),
        depths=np.asarray([a.depth for a in dataset.allocations], dtype=np.int8),
        schema=np.asarray(dataset.schema),
        feature_schema=np.asarray(dataset.feature_schema),
    )
    summary = {
        "schema": dataset.schema,
        "feature_schema": dataset.feature_schema,
        "transitions": int(dataset.spatial.shape[0]),
        "candidate_branches": int(dataset.reward.size),
        "instances": int(np.unique(dataset.instance_seeds).size),
        "terminal_branches": int(dataset.terminal.sum()),
        "allocations": [
            {"resolution": a.resolution, "depth": a.depth} for a in dataset.allocations
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if audit_rows:
        with (output_dir / "branches.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(audit_rows[0]))
            writer.writeheader()
            writer.writerows(audit_rows)


def load_meta_q_transitions(input_dir: Path) -> MetaQTransitionDataset:
    """Load a transition archive written by :func:`save_meta_q_transitions`."""

    with np.load(input_dir / "transitions.npz", allow_pickle=False) as values:
        allocations = tuple(
            Allocation(int(resolution), int(depth))
            for resolution, depth in zip(values["resolutions"], values["depths"], strict=True)
        )
        return MetaQTransitionDataset(
            spatial=values["spatial"].copy(),
            context=values["context"].copy(),
            next_spatial=values["next_spatial"].copy(),
            next_context=values["next_context"].copy(),
            reward=values["reward"].copy(),
            terminal=values["terminal"].copy(),
            compute_ms=values["compute_ms"].copy(),
            switch_required=values["switch_required"].copy(),
            candidate_actions=values["candidate_actions"].copy(),
            instance_seeds=values["instance_seeds"].copy(),
            source_resolution=values["source_resolution"].copy(),
            source_depth=values["source_depth"].copy(),
            decision_indices=values["decision_indices"].copy(),
            allocations=allocations,
            schema=str(values["schema"].item()),
            feature_schema=str(values["feature_schema"].item()),
        )


def concatenate_meta_q_datasets(
    datasets: Sequence[MetaQTransitionDataset],
) -> MetaQTransitionDataset:
    """Concatenate compatible transition shards in deterministic order."""

    values = tuple(datasets)
    if not values:
        return _empty_rows(ALLOCATIONS)
    reference = values[0]
    for dataset in values[1:]:
        if (
            dataset.allocations != reference.allocations
            or dataset.schema != reference.schema
            or dataset.feature_schema != reference.feature_schema
        ):
            raise ValueError("meta-Q transition shards use incompatible schemas")
    names = (
        "spatial",
        "context",
        "next_spatial",
        "next_context",
        "reward",
        "terminal",
        "compute_ms",
        "switch_required",
        "candidate_actions",
        "instance_seeds",
        "source_resolution",
        "source_depth",
        "decision_indices",
    )
    combined = {
        name: np.concatenate([getattr(dataset, name) for dataset in values], axis=0)
        for name in names
    }
    return MetaQTransitionDataset(
        **combined,
        allocations=reference.allocations,
        schema=reference.schema,
        feature_schema=reference.feature_schema,
    )


def _shard_marker(
    *,
    instance_seed: int,
    initial_source: Allocation,
    source_allocations: tuple[Allocation, ...],
    allocations: tuple[Allocation, ...],
    max_steps: int,
    message_passing_iterations: int,
    policy_workers: int,
    success_reward: float,
    failure_penalty: float | None,
    exploration_switch_probability: float,
) -> dict[str, Any]:
    return {
        "schema": META_Q_TRANSITION_SCHEMA,
        "instance_seed": int(instance_seed),
        "initial_source": [initial_source.resolution, initial_source.depth],
        "source_allocations": [[a.resolution, a.depth] for a in source_allocations],
        "allocations": [[a.resolution, a.depth] for a in allocations],
        "max_steps": int(max_steps),
        "message_passing_iterations": int(message_passing_iterations),
        "policy_workers": int(policy_workers),
        "success_reward": float(success_reward),
        "failure_penalty": None if failure_penalty is None else float(failure_penalty),
        "exploration_switch_probability": float(exploration_switch_probability),
    }


def _meta_q_shard_complete(shard_dir: Path, marker: dict[str, Any]) -> bool:
    marker_path = shard_dir / "complete.json"
    archive_path = shard_dir / "transitions.npz"
    if not marker_path.is_file() or not archive_path.is_file():
        return False
    try:
        return json.loads(marker_path.read_text(encoding="utf-8")) == marker
    except (OSError, json.JSONDecodeError):
        return False


def _generate_meta_q_shard(
    instance_seed: int,
    initial_source: Allocation,
    source_allocations: tuple[Allocation, ...],
    allocations: tuple[Allocation, ...],
    max_steps: int,
    message_passing_iterations: int,
    policy_workers: int,
    success_reward: float,
    failure_penalty: float | None,
    exploration_switch_probability: float,
    shard_dir: Path,
) -> tuple[int, int, int]:
    dataset, audit = generate_meta_q_transitions(
        instance_seeds=(instance_seed,),
        source_allocations=source_allocations,
        initial_source_allocations=(initial_source,),
        allocations=allocations,
        max_steps=max_steps,
        message_passing_iterations=message_passing_iterations,
        policy_workers=policy_workers,
        success_reward=success_reward,
        failure_penalty=failure_penalty,
        exploration_switch_probability=exploration_switch_probability,
    )
    save_meta_q_transitions(dataset, shard_dir, audit_rows=audit)
    marker = _shard_marker(
        instance_seed=instance_seed,
        initial_source=initial_source,
        source_allocations=source_allocations,
        allocations=allocations,
        max_steps=max_steps,
        message_passing_iterations=message_passing_iterations,
        policy_workers=policy_workers,
        success_reward=success_reward,
        failure_penalty=failure_penalty,
        exploration_switch_probability=exploration_switch_probability,
    )
    temporary = shard_dir / "complete.json.tmp"
    temporary.write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    temporary.replace(shard_dir / "complete.json")
    return instance_seed, int(dataset.spatial.shape[0]), int(dataset.reward.size)


def generate_resumable_meta_q_transitions(
    *,
    instance_seeds: Sequence[int],
    output_dir: Path,
    source_allocations: Sequence[Allocation] = ALLOCATIONS,
    allocations: Sequence[Allocation] = ALLOCATIONS,
    max_steps: int = 50,
    message_passing_iterations: int = 10,
    policy_workers: int = 1,
    success_reward: float = 0.0,
    failure_penalty: float | None = None,
    exploration_switch_probability: float = 0.25,
    instance_workers: int = 1,
    resume: bool = True,
) -> MetaQTransitionDataset:
    """Generate restart-safe per-instance shards and assemble one archive."""

    seeds = tuple(int(value) for value in instance_seeds)
    sources = tuple(source_allocations)
    candidates = tuple(allocations)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("instance_seeds must be nonempty and unique")
    if not sources or not set(sources).issubset(set(candidates)):
        raise ValueError("source_allocations must be a nonempty candidate subset")
    if not candidates or len(set(candidates)) != len(candidates):
        raise ValueError("allocations must be nonempty and unique")
    if instance_workers < 1:
        raise ValueError("instance_workers must be positive")
    output_dir = Path(output_dir)
    shard_root = output_dir / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    schedule = []
    for index, seed in enumerate(seeds):
        source = sources[index % len(sources)]
        shard_dir = shard_root / f"instance-{seed}"
        marker = _shard_marker(
            instance_seed=seed,
            initial_source=source,
            source_allocations=sources,
            allocations=candidates,
            max_steps=max_steps,
            message_passing_iterations=message_passing_iterations,
            policy_workers=policy_workers,
            success_reward=success_reward,
            failure_penalty=failure_penalty,
            exploration_switch_probability=exploration_switch_probability,
        )
        if resume and _meta_q_shard_complete(shard_dir, marker):
            print(f"[reuse] instance={seed}", flush=True)
        else:
            schedule.append((seed, source, shard_dir))

    common = (
        sources,
        candidates,
        max_steps,
        message_passing_iterations,
        policy_workers,
        success_reward,
        failure_penalty,
        exploration_switch_probability,
    )
    if instance_workers == 1:
        for completed, (seed, source, shard_dir) in enumerate(schedule, start=1):
            _, transitions, branches = _generate_meta_q_shard(seed, source, *common, shard_dir)
            print(
                f"[generate {completed}/{len(schedule)}] instance={seed} "
                f"transitions={transitions} branches={branches}",
                flush=True,
            )
    elif schedule:
        with ProcessPoolExecutor(max_workers=instance_workers) as executor:
            futures = {
                executor.submit(_generate_meta_q_shard, seed, source, *common, shard_dir): seed
                for seed, source, shard_dir in schedule
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                seed, transitions, branches = future.result()
                print(
                    f"[generate {completed}/{len(schedule)}] instance={seed} "
                    f"transitions={transitions} branches={branches}",
                    flush=True,
                )

    shards = []
    for index, seed in enumerate(seeds):
        source = sources[index % len(sources)]
        shard_dir = shard_root / f"instance-{seed}"
        marker = _shard_marker(
            instance_seed=seed,
            initial_source=source,
            source_allocations=sources,
            allocations=candidates,
            max_steps=max_steps,
            message_passing_iterations=message_passing_iterations,
            policy_workers=policy_workers,
            success_reward=success_reward,
            failure_penalty=failure_penalty,
            exploration_switch_probability=exploration_switch_probability,
        )
        if not _meta_q_shard_complete(shard_dir, marker):
            raise RuntimeError(f"incomplete meta-Q shard for instance {seed}")
        shards.append(load_meta_q_transitions(shard_dir))
    combined = concatenate_meta_q_datasets(shards)
    save_meta_q_transitions(combined, output_dir)

    combined_csv = output_dir / "branches.csv"
    wrote_header = False
    with combined_csv.open("w", newline="", encoding="utf-8") as target:
        writer = None
        for seed in seeds:
            source_csv = shard_root / f"instance-{seed}" / "branches.csv"
            if not source_csv.is_file():
                continue
            with source_csv.open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                if reader.fieldnames is None:
                    continue
                if writer is None:
                    writer = csv.DictWriter(target, fieldnames=reader.fieldnames)
                if not wrote_header:
                    writer.writeheader()
                    wrote_header = True
                writer.writerows(reader)
    manifest = {
        "instance_seeds": list(seeds),
        "instance_workers": instance_workers,
        "resume": resume,
        "source_mode": "balanced",
        "exploration_switch_probability": exploration_switch_probability,
        "transitions": int(combined.spatial.shape[0]),
        "candidate_branches": int(combined.reward.size),
    }
    (output_dir / "generation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return combined
