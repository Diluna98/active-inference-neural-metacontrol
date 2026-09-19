"""Closed-loop MOS evaluation for neural and fixed resource controllers."""

from __future__ import annotations

import csv
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from .allocations import ALLOCATION_INDEX, ALLOCATIONS, Allocation
from .beliefs import information_loss
from .counterfactuals import (
    _build_switched_agent,
    _infer_decision,
    _mos_imports,
    _step_cost,
    _timed,
)
from .models import TaskPerformanceNetwork
from .mos_adapter import build_mos_features
from .selector import AllocationDecision, SelectionConstraints, select_allocation

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency guard
    torch = None


@dataclass(frozen=True)
class ClosedLoopConfig:
    checkpoint: str
    initial_resolution: int = 5
    initial_depth: int = 2
    max_steps: int = 50
    message_passing_iterations: int = 10
    policy_workers: int = 1
    success_threshold: float = 0.5
    compute_budget_ms: float = 100.0
    information_loss_limit: float = 0.15
    device: str = "cpu"
    include_fixed: bool = True

    def __post_init__(self) -> None:
        Allocation(self.initial_resolution, self.initial_depth)
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if self.message_passing_iterations < 1 or self.policy_workers < 1:
            raise ValueError("inference settings must be positive")
        SelectionConstraints(
            success_threshold=self.success_threshold,
            compute_budget_ms=self.compute_budget_ms,
            information_loss_limit=self.information_loss_limit,
        )

    @property
    def initial_allocation(self) -> Allocation:
        return Allocation(self.initial_resolution, self.initial_depth)

    @property
    def constraints(self) -> SelectionConstraints:
        return SelectionConstraints(
            success_threshold=self.success_threshold,
            compute_budget_ms=self.compute_budget_ms,
            information_loss_limit=self.information_loss_limit,
        )


class NeuralMetaController:
    """Checkpoint-backed allocation selector with explicit resource constraints."""

    def __init__(self, checkpoint: Path, *, device: str = "cpu") -> None:
        if torch is None:
            raise ImportError("closed-loop neural evaluation requires PyTorch")
        self.device = torch.device(device)
        payload = torch.load(Path(checkpoint), map_location=self.device, weights_only=False)
        self.model = TaskPerformanceNetwork(
            spatial_channels=int(payload["spatial_channels"]),
            context_features=int(payload["context_features"]),
            allocations=len(ALLOCATIONS),
        ).to(self.device)
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.eval()
        self.context_mean = np.asarray(payload["context_mean"], dtype=np.float32)
        self.context_scale = np.asarray(payload["context_scale"], dtype=np.float32)
        self.compute_ms = np.asarray(payload["compute_profile_ms"], dtype=float)
        switching = payload.get("switching_profile_ms")
        if switching is None:
            raise ValueError("checkpoint has no calibrated switching profile")
        self.switching_ms = np.asarray(switching, dtype=float)
        if self.compute_ms.shape != (len(ALLOCATIONS),):
            raise ValueError("checkpoint compute profile has the wrong shape")
        if self.switching_ms.shape != (len(ALLOCATIONS), len(ALLOCATIONS)):
            raise ValueError("checkpoint switching profile has the wrong shape")

    def choose(
        self,
        *,
        agent: Any,
        allocation: Allocation,
        layout: Any,
        selected_action: int,
        next_robot_position: tuple[int, int],
        constraints: SelectionConstraints,
    ) -> tuple[AllocationDecision, float]:
        features = build_mos_features(
            agent=agent,
            allocation=allocation,
            layout=layout,
            selected_action=selected_action,
            next_robot_position=next_robot_position,
            found_flags=np.asarray([0.0]),
        )
        spatial = torch.from_numpy(features.spatial.tensor[None]).to(self.device)
        context = (features.context - self.context_mean) / self.context_scale
        context_tensor = torch.from_numpy(context[None].astype(np.float32)).to(self.device)
        started = time.perf_counter_ns()
        with torch.no_grad():
            prediction = self.model(spatial, context_tensor)
            success = torch.sigmoid(prediction["success_logits"])[0].cpu().numpy()
            relative_cost = prediction["relative_cost"][0].cpu().numpy()
        model_ms = (time.perf_counter_ns() - started) / 1e6
        posterior = np.asarray(agent.filtered_posteriors[2], dtype=float)
        losses = np.asarray(
            [
                information_loss(
                    posterior,
                    allocation.resolution,
                    candidate.resolution,
                    layout.size,
                )
                for candidate in ALLOCATIONS
            ]
        )
        source_index = ALLOCATION_INDEX[allocation]
        decision = select_allocation(
            success_probability=success,
            task_cost=relative_cost,
            compute_ms=self.compute_ms,
            switching_ms=self.switching_ms[source_index],
            information_loss=losses,
            constraints=constraints,
        )
        return decision, model_ms


def _episode_quantiles(instance: Any, max_steps: int) -> np.ndarray:
    seed = instance.layout.map_seed * 1_000_003 + instance.target_seed * 1009
    seed += instance.observation_seed
    return np.random.default_rng(seed).random(max_steps + 1)


def _build_agent(allocation: Allocation, layout: Any, config: ClosedLoopConfig, mos: dict) -> Any:
    agent = mos["build_mos_agent"](
        mos["MOSAgentConfig"](
            target_resolution=allocation.resolution,
            action_depth=allocation.depth,
            message_passing_iterations=config.message_passing_iterations,
            policy_workers=config.policy_workers,
        ),
        layout=layout,
    )
    agent.reset()
    return agent


def run_fixed_episode(instance_seed: int, allocation: Allocation, config: ClosedLoopConfig) -> dict:
    """Run one genuinely fixed-allocation MOS episode."""

    mos = _mos_imports()
    instance = mos["sample_mos_instance"](int(instance_seed))
    quantiles = _episode_quantiles(instance, config.max_steps)
    environment = mos["MOSEnvironment"](
        layout=instance.layout,
        target=instance.target,
        observation_seed=instance.observation_seed,
    )
    observation = environment.reset(noise_quantile=float(quantiles[0]))
    agent = _build_agent(allocation, instance.layout, config, mos)
    previous_action = None
    task_cost = 0.0
    inference_ms = 0.0
    false_finds = 0
    collisions = 0
    success = False
    steps = 0
    for decision_index in range(config.max_steps):
        decision = _infer_decision(agent, observation, previous_action, decision_index, mos)
        inference_ms += decision.total_ms
        observation, success = environment.step(
            decision.action,
            noise_quantile=float(quantiles[decision_index + 1]),
        )
        false_find = observation[3] == mos["FindOutcome"].FALSE_FIND
        collision = environment.last_collision
        task_cost += _step_cost(false_find=false_find, collision=collision)
        false_finds += int(false_find)
        collisions += int(collision)
        steps += 1
        previous_action = decision.action
        if success:
            break
    if not success:
        task_cost += 2.0 * config.max_steps
    return {
        "controller": f"fixed_g{allocation.resolution}_T{allocation.depth}",
        "instance_seed": int(instance_seed),
        "success": bool(success),
        "task_cost": float(task_cost),
        "steps": steps,
        "false_finds": false_finds,
        "collisions": collisions,
        "task_inference_ms": float(inference_ms),
        "meta_inference_ms": 0.0,
        "switch_ms": 0.0,
        "total_compute_ms": float(inference_ms),
        "switches": 0,
        "feasible_decisions": 0,
        "decisions": steps,
        "allocation_counts": json.dumps({f"g{allocation.resolution}_T{allocation.depth}": steps}),
    }


def run_adaptive_episode(
    instance_seed: int,
    config: ClosedLoopConfig,
) -> tuple[dict, list[dict]]:
    """Run one adaptive episode whose selections alter subsequent inference."""

    mos = _mos_imports()
    instance = mos["sample_mos_instance"](int(instance_seed))
    quantiles = _episode_quantiles(instance, config.max_steps)
    environment = mos["MOSEnvironment"](
        layout=instance.layout,
        target=instance.target,
        observation_seed=instance.observation_seed,
    )
    observation = environment.reset(noise_quantile=float(quantiles[0]))
    allocation = config.initial_allocation
    agent = _build_agent(allocation, instance.layout, config, mos)
    controller = NeuralMetaController(Path(config.checkpoint), device=config.device)
    previous_action = None
    task_cost = 0.0
    task_inference_ms = 0.0
    meta_inference_ms = 0.0
    switch_ms = 0.0
    false_finds = 0
    collisions = 0
    switches = 0
    feasible_decisions = 0
    success = False
    steps = 0
    allocation_counts = {candidate: 0 for candidate in ALLOCATIONS}
    trajectory = []

    for decision_index in range(config.max_steps):
        source_allocation = allocation
        task_decision = _infer_decision(agent, observation, previous_action, decision_index, mos)
        task_inference_ms += task_decision.total_ms
        allocation_counts[allocation] += 1
        next_position = instance.layout.move(environment.position, task_decision.action)
        meta_decision, model_ms = controller.choose(
            agent=agent,
            allocation=allocation,
            layout=instance.layout,
            selected_action=int(task_decision.action),
            next_robot_position=next_position,
            constraints=config.constraints,
        )
        meta_inference_ms += model_ms
        feasible_decisions += int(meta_decision.feasible)
        observation, success = environment.step(
            task_decision.action,
            noise_quantile=float(quantiles[decision_index + 1]),
        )
        false_find = observation[3] == mos["FindOutcome"].FALSE_FIND
        collision = environment.last_collision
        step_cost = _step_cost(false_find=false_find, collision=collision)
        task_cost += step_cost
        false_finds += int(false_find)
        collisions += int(collision)
        steps += 1
        actual_switch_ms = 0.0
        if not success and decision_index + 1 < config.max_steps:
            target = meta_decision.allocation
            if target != allocation:
                agent, actual_switch_ms = _timed(
                    partial(
                        _build_switched_agent,
                        source_agent=agent,
                        source_allocation=allocation,
                        target_allocation=target,
                        layout=instance.layout,
                        executed_action=task_decision.action,
                        next_time_step=decision_index + 1,
                        message_passing_iterations=config.message_passing_iterations,
                        policy_workers=config.policy_workers,
                        mos=mos,
                    )
                )
                switch_ms += actual_switch_ms
                switches += 1
            allocation = target
        trajectory.append(
            {
                "instance_seed": int(instance_seed),
                "decision": decision_index,
                "source_resolution": source_allocation.resolution,
                "source_depth": source_allocation.depth,
                "action": task_decision.action.name,
                "selected_resolution": meta_decision.allocation.resolution,
                "selected_depth": meta_decision.allocation.depth,
                "selection_feasible": meta_decision.feasible,
                "selection_reason": meta_decision.reason,
                "predicted_success": meta_decision.predicted_success,
                "predicted_relative_cost": meta_decision.predicted_task_cost,
                "predicted_inference_ms": meta_decision.predicted_compute_ms,
                "predicted_switch_ms": meta_decision.switching_ms,
                "information_loss": meta_decision.information_loss,
                "actual_task_inference_ms": task_decision.total_ms,
                "actual_meta_inference_ms": model_ms,
                "actual_switch_ms": actual_switch_ms,
                "step_cost": step_cost,
                "success": bool(success),
            }
        )
        previous_action = task_decision.action
        if success:
            break
    if not success:
        task_cost += 2.0 * config.max_steps
    total_compute = task_inference_ms + meta_inference_ms + switch_ms
    return (
        {
            "controller": "adaptive",
            "instance_seed": int(instance_seed),
            "success": bool(success),
            "task_cost": float(task_cost),
            "steps": steps,
            "false_finds": false_finds,
            "collisions": collisions,
            "task_inference_ms": float(task_inference_ms),
            "meta_inference_ms": float(meta_inference_ms),
            "switch_ms": float(switch_ms),
            "total_compute_ms": float(total_compute),
            "switches": switches,
            "feasible_decisions": feasible_decisions,
            "decisions": steps,
            "allocation_counts": json.dumps(
                {
                    f"g{candidate.resolution}_T{candidate.depth}": count
                    for candidate, count in allocation_counts.items()
                    if count
                },
                sort_keys=True,
            ),
        },
        trajectory,
    )


def _run_instance(instance_seed: int, config: ClosedLoopConfig) -> dict:
    adaptive, trajectory = run_adaptive_episode(instance_seed, config)
    episodes = [adaptive]
    if config.include_fixed:
        episodes.extend(
            run_fixed_episode(instance_seed, allocation, config) for allocation in ALLOCATIONS
        )
    return {"episodes": episodes, "trajectory": trajectory}


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summary(episodes: list[dict]) -> dict[str, Any]:
    controllers = sorted({row["controller"] for row in episodes})
    result = {}
    for controller in controllers:
        rows = [row for row in episodes if row["controller"] == controller]
        summary = {
            "episodes": len(rows),
            "success_rate": float(np.mean([row["success"] for row in rows])),
            "mean_task_cost": float(np.mean([row["task_cost"] for row in rows])),
            "mean_steps": float(np.mean([row["steps"] for row in rows])),
            "mean_total_compute_ms": float(np.mean([row["total_compute_ms"] for row in rows])),
            "median_total_compute_ms": float(np.median([row["total_compute_ms"] for row in rows])),
            "mean_switches": float(np.mean([row["switches"] for row in rows])),
        }
        if controller == "adaptive":
            allocation_steps: dict[str, int] = {}
            for row in rows:
                for name, count in json.loads(row["allocation_counts"]).items():
                    allocation_steps[name] = allocation_steps.get(name, 0) + int(count)
            decisions = sum(int(row["decisions"]) for row in rows)
            summary["feasible_decision_rate"] = float(
                sum(int(row["feasible_decisions"]) for row in rows) / decisions
            )
            summary["allocation_steps"] = allocation_steps
        result[controller] = summary
    return result


def _paired_comparisons(episodes: list[dict]) -> dict[str, Any]:
    adaptive = {
        int(row["instance_seed"]): row for row in episodes if row["controller"] == "adaptive"
    }
    rng = np.random.default_rng(0)
    comparisons = {}
    for controller in sorted({row["controller"] for row in episodes} - {"adaptive"}):
        fixed = {
            int(row["instance_seed"]): row for row in episodes if row["controller"] == controller
        }
        seeds = sorted(set(adaptive) & set(fixed))
        metrics = {}
        for name in ("task_cost", "success", "steps", "total_compute_ms"):
            delta = np.asarray(
                [float(adaptive[seed][name]) - float(fixed[seed][name]) for seed in seeds]
            )
            if delta.size == 1:
                confidence = [float(delta[0]), float(delta[0])]
            else:
                bootstrap = np.asarray(
                    [rng.choice(delta, size=delta.size, replace=True).mean() for _ in range(5000)]
                )
                confidence = np.percentile(bootstrap, (2.5, 97.5)).tolist()
            metrics[f"adaptive_minus_fixed_{name}"] = {
                "mean": float(delta.mean()),
                "ci95": [float(value) for value in confidence],
            }
        comparisons[controller] = metrics
    return comparisons


def evaluate_closed_loop(
    *,
    instance_seeds: list[int] | tuple[int, ...],
    output_dir: Path,
    config: ClosedLoopConfig,
    instance_workers: int = 1,
    resume: bool = True,
) -> dict[str, Any]:
    """Run resumable per-instance adaptive and fixed closed-loop episodes."""

    seeds = tuple(int(value) for value in instance_seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("instance_seeds must be nonempty and unique")
    if instance_workers < 1:
        raise ValueError("instance_workers must be positive")
    output_dir = Path(output_dir)
    shard_dir = output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(config.checkpoint)
    signature = {
        "configuration": asdict(config),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
    }
    pending = []
    for seed in seeds:
        path = shard_dir / f"instance-{seed}.json"
        compatible = False
        if resume and path.is_file():
            try:
                compatible = (
                    json.loads(path.read_text(encoding="utf-8")).get("signature") == signature
                )
            except (OSError, json.JSONDecodeError):
                compatible = False
        if compatible:
            print(f"[reuse] instance={seed}", flush=True)
        else:
            pending.append(seed)

    def save(seed: int, payload: dict) -> None:
        payload["signature"] = signature
        temporary = shard_dir / f"instance-{seed}.json.tmp"
        temporary.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        temporary.replace(shard_dir / f"instance-{seed}.json")

    if instance_workers == 1:
        for completed, seed in enumerate(pending, start=1):
            save(seed, _run_instance(seed, config))
            print(f"[evaluate {completed}/{len(pending)}] instance={seed}", flush=True)
    elif pending:
        with ProcessPoolExecutor(max_workers=instance_workers) as executor:
            futures = {executor.submit(_run_instance, seed, config): seed for seed in pending}
            for completed, future in enumerate(as_completed(futures), start=1):
                seed = futures[future]
                save(seed, future.result())
                print(f"[evaluate {completed}/{len(pending)}] instance={seed}", flush=True)

    payloads = [
        json.loads((shard_dir / f"instance-{seed}.json").read_text(encoding="utf-8"))
        for seed in seeds
    ]
    episodes = [row for payload in payloads for row in payload["episodes"]]
    trajectory = [row for payload in payloads for row in payload["trajectory"]]
    _write_csv(output_dir / "episodes.csv", episodes)
    _write_csv(output_dir / "adaptive_trajectory.csv", trajectory)
    report = {
        "instance_seeds": list(seeds),
        "instance_workers": instance_workers,
        "configuration": asdict(config),
        "controllers": _summary(episodes),
        "paired_comparisons": _paired_comparisons(episodes),
    }
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
