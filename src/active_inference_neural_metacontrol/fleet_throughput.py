"""Trace-driven fleet-capacity evaluation under a shared planning processor."""

from __future__ import annotations

import csv
import heapq
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EpisodeTrace:
    """Online computation and outcome trace for one search agent."""

    instance_seed: int
    controller: str
    decision_compute_s: tuple[float, ...]
    success: bool
    success_step: int | None

    def __post_init__(self) -> None:
        if not self.decision_compute_s:
            raise ValueError("an episode trace must contain at least one decision")
        if any(value < 0 for value in self.decision_compute_s):
            raise ValueError("decision computation cannot be negative")
        if self.success and self.success_step != len(self.decision_compute_s):
            raise ValueError("a successful stored episode must terminate at its success step")
        if not self.success and self.success_step is not None:
            raise ValueError("a failed episode cannot have a success step")


@dataclass(frozen=True)
class FleetSimulationConfig:
    mission_time_s: float = 30.0
    cell_size_m: float = 0.25
    robot_speed_mps: float = 0.5
    planning_workers: int = 1
    deadline_s: float | None = None

    def __post_init__(self) -> None:
        if self.mission_time_s <= 0:
            raise ValueError("mission_time_s must be positive")
        if self.cell_size_m <= 0 or self.robot_speed_mps <= 0:
            raise ValueError("cell size and robot speed must be positive")
        if self.planning_workers <= 0:
            raise ValueError("planning_workers must be positive")
        if self.deadline_s is not None and self.deadline_s <= 0:
            raise ValueError("deadline_s must be positive")

    @property
    def action_duration_s(self) -> float:
        return self.cell_size_m / self.robot_speed_mps

    @property
    def effective_deadline_s(self) -> float:
        return self.action_duration_s if self.deadline_s is None else self.deadline_s


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def load_finalized_traces(
    results_dir: Path,
    *,
    controllers: tuple[str, ...] = ("adaptive", "fixed_g10_T2"),
) -> dict[str, dict[int, EpisodeTrace]]:
    """Load finalized single-agent results as decision-service traces.

    Adaptive runs contain measured per-decision timings. Historic fixed runs
    only retained the measured episode total, so their total is divided evenly
    across the recorded number of decisions. This preserves measured total
    demand while avoiding an unsupported reconstruction of its latency tails.
    """

    results_dir = Path(results_dir)
    episode_rows = _read_csv(results_dir / "episodes.csv")
    trajectory_path = results_dir / "adaptive_trajectory.csv"
    trajectory_rows = _read_csv(trajectory_path) if trajectory_path.exists() else []
    adaptive_rows: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in trajectory_rows:
        adaptive_rows[int(row["instance_seed"])].append(row)
    for rows in adaptive_rows.values():
        rows.sort(key=lambda row: int(row["step"]))

    traces: dict[str, dict[int, EpisodeTrace]] = {name: {} for name in controllers}
    for row in episode_rows:
        controller = row["controller"]
        if controller not in traces:
            continue
        seed = int(row["instance_seed"])
        steps = int(row["steps"])
        success = row["success"].strip().lower() == "true"
        measured_total_s = float(row["total_compute_ms"]) / 1000.0
        if controller == "adaptive":
            step_rows = adaptive_rows.get(seed, [])
            if len(step_rows) != steps:
                raise ValueError(
                    f"adaptive seed {seed} has {len(step_rows)} trajectory rows, expected {steps}"
                )
            compute = np.asarray(
                [
                    (float(item["task_inference_ms"]) + float(item["meta_inference_ms"])) / 1000.0
                    for item in step_rows
                ],
                dtype=float,
            )
            residual = measured_total_s - float(compute.sum())
            switch_indices = [
                index
                for index, item in enumerate(step_rows)
                if item["source_allocation"] != item["selected_allocation"]
            ]
            recipients = switch_indices or list(range(steps))
            if recipients:
                compute[recipients] += residual / len(recipients)
            if np.any(compute < -1e-9):
                raise ValueError(f"negative reconstructed compute time for seed {seed}")
            compute = np.maximum(compute, 0.0)
        else:
            compute = np.full(steps, measured_total_s / steps, dtype=float)
        traces[controller][seed] = EpisodeTrace(
            instance_seed=seed,
            controller=controller,
            decision_compute_s=tuple(float(value) for value in compute),
            success=success,
            success_step=steps if success else None,
        )

    missing = [name for name, rows in traces.items() if not rows]
    if missing:
        raise ValueError(f"no episode traces found for controllers: {missing}")
    common = set.intersection(*(set(rows) for rows in traces.values()))
    if not common:
        raise ValueError("controllers do not share any instance seeds")
    return {name: {seed: rows[seed] for seed in sorted(common)} for name, rows in traces.items()}


def simulate_fleet(
    traces: list[EpisodeTrace],
    config: FleetSimulationConfig,
) -> dict[str, Any]:
    """Replay a fleet through a non-preemptive FCFS planning-worker pool."""

    if not traces:
        raise ValueError("fleet cannot be empty")
    controllers = {trace.controller for trace in traces}
    if len(controllers) != 1:
        raise ValueError("one fleet simulation must use one controller")

    # Events are planning requests: (ready time, stable agent id, decision index).
    requests = [(0.0, agent_id, 0) for agent_id in range(len(traces))]
    heapq.heapify(requests)
    worker_free = [0.0] * config.planning_workers
    targets_found = 0
    completed_actions = 0
    submitted_requests = 0
    completed_plans = 0
    deadline_misses = 0
    total_queue_wait_s = 0.0
    total_planning_delay_s = 0.0
    busy_within_horizon_s = 0.0
    per_agent_found = [False] * len(traces)
    per_agent_wait = [0.0] * len(traces)

    while requests:
        ready_s, agent_id, decision_index = heapq.heappop(requests)
        if ready_s >= config.mission_time_s:
            continue
        trace = traces[agent_id]
        if decision_index >= len(trace.decision_compute_s):
            continue
        submitted_requests += 1
        worker_id = min(range(config.planning_workers), key=worker_free.__getitem__)
        start_s = max(ready_s, worker_free[worker_id])
        service_s = trace.decision_compute_s[decision_index]
        finish_s = start_s + service_s
        worker_free[worker_id] = finish_s
        queue_wait_s = start_s - ready_s
        planning_delay_s = finish_s - ready_s
        total_queue_wait_s += queue_wait_s
        total_planning_delay_s += planning_delay_s
        per_agent_wait[agent_id] += planning_delay_s
        deadline_misses += int(planning_delay_s > config.effective_deadline_s)
        busy_within_horizon_s += max(
            0.0, min(finish_s, config.mission_time_s) - min(start_s, config.mission_time_s)
        )
        if finish_s > config.mission_time_s:
            continue
        completed_plans += 1
        action_finish_s = finish_s + config.action_duration_s
        if action_finish_s > config.mission_time_s:
            continue
        completed_actions += 1
        step_number = decision_index + 1
        if trace.success and trace.success_step == step_number:
            targets_found += 1
            per_agent_found[agent_id] = True
            continue
        if step_number < len(trace.decision_compute_s):
            heapq.heappush(requests, (action_finish_s, agent_id, step_number))

    capacity_s = config.mission_time_s * config.planning_workers
    return {
        "controller": next(iter(controllers)),
        "fleet_size": len(traces),
        "targets_found": targets_found,
        "target_found_rate": targets_found / len(traces),
        "completed_actions": completed_actions,
        "submitted_requests": submitted_requests,
        "completed_plans": completed_plans,
        "deadline_misses": deadline_misses,
        "deadline_miss_rate": deadline_misses / submitted_requests,
        "mean_queue_wait_s_per_request": total_queue_wait_s / submitted_requests,
        "mean_planning_delay_s_per_request": total_planning_delay_s / submitted_requests,
        "mean_planning_wait_s_per_agent": float(np.mean(per_agent_wait)),
        "max_planning_wait_s_per_agent": float(np.max(per_agent_wait)),
        "planner_utilization": min(1.0, busy_within_horizon_s / capacity_s),
        "found_agents": [
            int(trace.instance_seed)
            for trace, found in zip(traces, per_agent_found, strict=True)
            if found
        ],
    }


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["controller"]), int(row["fleet_size"]))].append(row)
    metrics = (
        "targets_found",
        "target_found_rate",
        "deadline_miss_rate",
        "mean_queue_wait_s_per_request",
        "mean_planning_delay_s_per_request",
        "mean_planning_wait_s_per_agent",
        "planner_utilization",
    )
    summaries = []
    for (controller, fleet_size), group in sorted(groups.items()):
        summary: dict[str, Any] = {
            "controller": controller,
            "fleet_size": fleet_size,
            "trials": len(group),
        }
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in group])
            summary[f"mean_{metric}"] = float(values.mean())
            summary[f"std_{metric}"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summaries.append(summary)
    return summaries


def balanced_fleet_samples(
    seeds: np.ndarray,
    *,
    fleet_size: int,
    repeats: int,
    rng: np.random.Generator,
) -> list[list[int]]:
    """Create matched fleets with equal seed frequency whenever cycles are complete."""

    seed_count = len(seeds)
    if not 0 < fleet_size <= seed_count:
        raise ValueError("fleet_size must be between one and the seed count")
    cycle_length = seed_count // math.gcd(seed_count, fleet_size)
    fleets: list[list[int]] = []
    for cycle_start in range(0, repeats, cycle_length):
        permutation = rng.permutation(seeds)
        cycle_repeats = min(cycle_length, repeats - cycle_start)
        for within_cycle in range(cycle_repeats):
            start = (within_cycle * fleet_size) % seed_count
            indices = (start + np.arange(fleet_size)) % seed_count
            fleets.append([int(seed) for seed in permutation[indices]])
    return fleets


def evaluate_fleet_capacity(
    *,
    results_dir: Path,
    output_dir: Path,
    fleet_sizes: tuple[int, ...],
    repeats: int,
    sampling_seed: int,
    config: FleetSimulationConfig,
    controllers: tuple[str, ...] = ("adaptive", "fixed_g10_T2"),
) -> dict[str, Any]:
    """Evaluate matched randomized fleets from finalized single-agent traces."""

    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if not fleet_sizes or any(size <= 0 for size in fleet_sizes):
        raise ValueError("fleet sizes must be positive")
    traces = load_finalized_traces(results_dir, controllers=controllers)
    seeds = np.asarray(sorted(next(iter(traces.values()))), dtype=int)
    if max(fleet_sizes) > len(seeds):
        raise ValueError(
            f"largest fleet ({max(fleet_sizes)}) exceeds {len(seeds)} available traces"
        )
    rng = np.random.default_rng(sampling_seed)
    trial_rows: list[dict[str, Any]] = []
    for fleet_size in fleet_sizes:
        samples = balanced_fleet_samples(
            seeds,
            fleet_size=fleet_size,
            repeats=repeats,
            rng=rng,
        )
        for repeat, chosen in enumerate(samples):
            for controller in controllers:
                row = simulate_fleet(
                    [traces[controller][seed] for seed in chosen],
                    config,
                )
                row.update(
                    {
                        "repeat": repeat,
                        "instance_seeds": json.dumps(chosen),
                    }
                )
                trial_rows.append(row)
    summary_rows = _summarize(trial_rows)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "fleet_trials.csv", trial_rows)
    _write_csv(output_dir / "fleet_summary.csv", summary_rows)
    report = {
        "source_results": str(Path(results_dir).resolve()),
        "controllers": list(controllers),
        "available_instances": len(seeds),
        "fleet_sizes": list(fleet_sizes),
        "repeats": repeats,
        "sampling_seed": sampling_seed,
        "configuration": asdict(config),
        "action_duration_s": config.action_duration_s,
        "effective_deadline_s": config.effective_deadline_s,
        "summary": summary_rows,
        "interpretation": {
            "execution": "trace-driven non-preemptive FCFS shared planning pool",
            "setup_cost": "excluded, as in the finalized online-compute comparison",
            "fixed_timing": "measured episode total divided uniformly over decisions",
            "environment": "independent static MOS instance per agent",
            "collaboration": "none; targets found is aggregate fleet throughput",
        },
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
