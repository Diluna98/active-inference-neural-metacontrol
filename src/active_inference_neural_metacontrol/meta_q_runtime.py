"""Runtime inference and closed-loop evaluation for the meta-Q controller."""

from __future__ import annotations

import csv
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from .allocations import ALLOCATIONS, Allocation
from .beliefs import information_loss
from .closed_loop import (
    ClosedLoopConfig,
    _build_agent,
    _CachedMOSAgentPool,
    _episode_quantiles,
    _paired_comparisons,
    _summary,
    run_fixed_episode,
)
from .counterfactuals import _infer_decision, _mos_imports, _step_cost, _timed
from .meta_q import (
    META_Q_CHECKPOINT_SCHEMA,
    MetaQNetwork,
    MetaQSelection,
    aggregate_reference_relative_q,
    select_cheapest_sufficient,
    select_joint_task_compute,
)
from .meta_q_features import (
    META_Q_FEATURE_SCHEMA,
    MetaQFeatures,
    build_meta_q_features,
    summarize_decision_context,
)
from .meta_regret import (
    META_REGRET_CHECKPOINT_SCHEMA,
    META_REGRET_MODEL_TYPE,
    MetaRegretNetwork,
    augment_regret_context,
)

try:
    import torch
except ImportError:  # pragma: no cover - optional neural dependency
    torch = None


class MetaQController:
    """Load a fitted-Q checkpoint and choose the cheapest sufficient allocation."""

    def __init__(
        self,
        checkpoint: Path,
        *,
        ensemble_checkpoints: tuple[Path, ...] = (),
        device: str = "cpu",
        torch_threads: int = 1,
    ):
        if torch is None:
            raise ImportError("meta-Q runtime requires PyTorch")
        torch.set_num_threads(torch_threads)
        checkpoint_paths = (Path(checkpoint),) + tuple(Path(path) for path in ensemble_checkpoints)
        payloads = [
            torch.load(path, map_location=device, weights_only=False) for path in checkpoint_paths
        ]
        payload = payloads[0]
        model_type = payload.get("model_type")
        expected_schema = (
            META_REGRET_CHECKPOINT_SCHEMA
            if model_type == META_REGRET_MODEL_TYPE
            else META_Q_CHECKPOINT_SCHEMA
        )
        if int(payload.get("schema_version", -1)) != expected_schema:
            raise ValueError("unsupported meta-controller checkpoint schema")
        if model_type not in {"meta_q", META_REGRET_MODEL_TYPE}:
            raise ValueError("checkpoint is not a supported meta-controller model")
        if payload.get("feature_schema") != META_Q_FEATURE_SCHEMA:
            raise ValueError("meta-Q checkpoint uses an unsupported feature schema")
        self.allocations = tuple(
            Allocation(int(row["resolution"]), int(row["depth"])) for row in payload["allocations"]
        )
        if self.allocations != ALLOCATIONS:
            raise ValueError("runtime currently requires the complete canonical allocation set")
        self.device = device
        self.model_type = model_type
        self.models = []
        self.context_statistics = []
        for member_payload in payloads:
            if member_payload.get("model_type") != model_type:
                raise ValueError("all ensemble checkpoints must use the same model type")
            if (
                tuple(
                    Allocation(int(row["resolution"]), int(row["depth"]))
                    for row in member_payload["allocations"]
                )
                != self.allocations
            ):
                raise ValueError("all ensemble checkpoints must use the same allocations")
            model = (
                MetaRegretNetwork(allocations=len(self.allocations))
                if model_type == META_REGRET_MODEL_TYPE
                else MetaQNetwork(allocations=len(self.allocations))
            ).to(device)
            model.load_state_dict(member_payload["model_state"])
            model.eval()
            self.models.append(model)
            self.context_statistics.append(
                (
                    np.asarray(member_payload["context_mean"], dtype=np.float32),
                    np.asarray(member_payload["context_scale"], dtype=np.float32),
                )
            )
        compute_profiles = np.stack(
            [
                np.asarray(member_payload["compute_profile_ms"], dtype=float)
                for member_payload in payloads
            ]
        )
        mean_compute = compute_profiles.mean(axis=0)
        relative_spread = np.max(
            np.abs(compute_profiles - mean_compute[None, :])
            / np.maximum(mean_compute[None, :], 1e-9)
        )
        if relative_spread > 0.05:
            raise ValueError("ensemble compute profiles differ by more than five percent")
        self.compute_ms = mean_compute

    def choose(
        self,
        *,
        agent: Any,
        allocation: Allocation,
        layout: Any,
        selected_action: int,
        next_robot_position: tuple[int, int],
        realized_observation: tuple[int, ...],
        task_tolerance: float,
        compute_weight: float,
        switch_cost_ms: float,
        switching_weight: float,
        selection_mode: str = "tolerance",
        compute_price: float = 0.0,
        uncertainty_beta: float = 0.0,
        information_loss_weight: float = 0.0,
        switch_penalty_mode: str = "allocation",
        features: MetaQFeatures | None = None,
    ) -> tuple[MetaQSelection, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        if features is None:
            features = build_meta_q_features(
                agent=agent,
                allocation=allocation,
                layout=layout,
                selected_action=selected_action,
                next_robot_position=next_robot_position,
                realized_observation=realized_observation,
            )
        raw_context = (
            augment_regret_context(features.context, allocation)
            if self.model_type == META_REGRET_MODEL_TYPE
            else features.context
        )
        spatial_tensor = torch.from_numpy(features.spatial[None]).to(self.device)
        started = time.perf_counter_ns()
        samples = []
        with torch.no_grad():
            for model, (context_mean, context_scale) in zip(
                self.models, self.context_statistics, strict=True
            ):
                context = (raw_context - context_mean) / context_scale
                context_tensor = torch.from_numpy(context[None].astype(np.float32)).to(self.device)
                samples.append(model(spatial_tensor, context_tensor)[0].cpu().numpy())
        model_ms = (time.perf_counter_ns() - started) / 1e6
        q_samples = np.stack(samples)
        posterior = np.asarray(agent.filtered_posteriors[2], dtype=float)
        information_losses = np.asarray(
            [
                information_loss(
                    posterior,
                    allocation.resolution,
                    candidate.resolution,
                    layout.size,
                )
                for candidate in self.allocations
            ],
            dtype=float,
        )
        if self.model_type == META_REGRET_MODEL_TYPE:
            q_values, q_mean, q_std = aggregate_reference_relative_q(
                q_samples,
                reference_index=self.allocations.index(Allocation(5, 2)),
                uncertainty_beta=uncertainty_beta,
            )
        else:
            q_mean = q_samples.mean(axis=0)
            q_std = q_samples.std(axis=0, ddof=0)
            q_values = q_mean - uncertainty_beta * q_std
        if selection_mode == "tolerance":
            selection = select_cheapest_sufficient(
                q_values,
                allocations=self.allocations,
                compute_ms=self.compute_ms,
                current_allocation=allocation,
                task_tolerance=task_tolerance,
                compute_weight=compute_weight,
                switch_cost_ms=switch_cost_ms,
                switching_weight=switching_weight,
            )
        elif selection_mode == "joint":
            selection = select_joint_task_compute(
                q_values,
                allocations=self.allocations,
                compute_ms=self.compute_ms,
                current_allocation=allocation,
                compute_price=compute_price,
                switch_cost_ms=switch_cost_ms,
                switching_weight=switching_weight,
                switch_penalty_mode=switch_penalty_mode,
                information_loss=information_losses,
                information_loss_weight=information_loss_weight,
            )
        else:
            raise ValueError("selection_mode must be 'tolerance' or 'joint'")
        return selection, q_values, q_mean, q_std, information_losses, model_ms


@dataclass(frozen=True)
class MetaQEvaluationConfig:
    checkpoint: Path
    ensemble_checkpoints: tuple[Path, ...] = ()
    initial_allocation: Allocation = field(default_factory=lambda: Allocation(2, 1))
    max_steps: int = 50
    message_passing_iterations: int = 10
    policy_workers: int = 1
    task_tolerance: float = 1.0
    compute_weight: float = 1.0
    switch_cost_ms: float = 0.0
    switching_weight: float = 1.0
    selection_mode: str = "tolerance"
    compute_price: float = 0.0
    uncertainty_beta: float = 0.0
    information_loss_weight: float = 0.0
    switch_penalty_mode: str = "allocation"
    device: str = "cpu"
    torch_threads: int = 1

    def __post_init__(self) -> None:
        if self.selection_mode not in {"tolerance", "joint"}:
            raise ValueError("selection_mode must be 'tolerance' or 'joint'")
        if self.compute_price < 0:
            raise ValueError("compute_price cannot be negative")
        if self.uncertainty_beta < 0:
            raise ValueError("uncertainty_beta cannot be negative")
        if self.information_loss_weight < 0:
            raise ValueError("information_loss_weight cannot be negative")
        if self.switch_penalty_mode not in {"allocation", "resolution", "none"}:
            raise ValueError("invalid switch_penalty_mode")


def run_meta_q_episode(
    instance_seed: int,
    config: MetaQEvaluationConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run one episode, reconsidering resolution and depth after every action."""

    mos = _mos_imports()
    instance = mos["sample_mos_instance"](int(instance_seed))
    environment = mos["MOSEnvironment"](
        layout=instance.layout,
        target=instance.target,
        observation_seed=instance.observation_seed,
    )
    quantiles = _episode_quantiles(instance, config.max_steps)
    observation = environment.reset(noise_quantile=float(quantiles[0]))
    allocation = config.initial_allocation
    agent_config = SimpleNamespace(
        message_passing_iterations=config.message_passing_iterations,
        policy_workers=config.policy_workers,
    )
    agent = _build_agent(allocation, instance.layout, agent_config, mos)
    pool, pool_setup_ms = _timed(
        partial(
            _CachedMOSAgentPool,
            initial_allocation=allocation,
            initial_agent=agent,
            layout=instance.layout,
            config=agent_config,
            mos=mos,
        )
    )
    controller = MetaQController(
        config.checkpoint,
        ensemble_checkpoints=config.ensemble_checkpoints,
        device=config.device,
        torch_threads=config.torch_threads,
    )
    previous_action = None
    task_cost = 0.0
    task_inference_ms = 0.0
    meta_inference_ms = 0.0
    switch_ms = 0.0
    switches = 0
    success = False
    trajectory: list[dict[str, Any]] = []
    counts = {candidate: 0 for candidate in ALLOCATIONS}

    for decision_index in range(config.max_steps):
        source = allocation
        position = environment.position
        decision = _infer_decision(agent, observation, previous_action, decision_index, mos)
        task_inference_ms += decision.total_ms
        counts[source] += 1
        next_position = instance.layout.move(position, decision.action)
        observation, success = environment.step(
            decision.action,
            noise_quantile=float(quantiles[decision_index + 1]),
        )
        false_find = observation[3] == mos["FindOutcome"].FALSE_FIND
        step_cost = _step_cost(false_find=false_find, collision=environment.last_collision)
        task_cost += step_cost
        selection = None
        q_values = None
        q_mean = None
        q_std = None
        information_losses = None
        context_diagnostics = None
        model_ms = 0.0
        target = allocation
        if not success and decision_index + 1 < config.max_steps:
            meta_features = build_meta_q_features(
                agent=agent,
                allocation=allocation,
                layout=instance.layout,
                selected_action=int(decision.action),
                next_robot_position=next_position,
                realized_observation=observation,
            )
            context_diagnostics = summarize_decision_context(meta_features)
            selection, q_values, q_mean, q_std, information_losses, model_ms = controller.choose(
                agent=agent,
                allocation=allocation,
                layout=instance.layout,
                selected_action=int(decision.action),
                next_robot_position=next_position,
                realized_observation=observation,
                task_tolerance=config.task_tolerance,
                compute_weight=config.compute_weight,
                switch_cost_ms=config.switch_cost_ms,
                switching_weight=config.switching_weight,
                selection_mode=config.selection_mode,
                compute_price=config.compute_price,
                uncertainty_beta=config.uncertainty_beta,
                information_loss_weight=config.information_loss_weight,
                switch_penalty_mode=config.switch_penalty_mode,
                features=meta_features,
            )
            meta_inference_ms += model_ms
            target = selection.allocation
            if target != allocation:
                agent, actual_switch_ms = _timed(
                    partial(
                        pool.switch,
                        source_agent=agent,
                        source_allocation=allocation,
                        target_allocation=target,
                        layout=instance.layout,
                        executed_action=decision.action,
                        next_time_step=decision_index + 1,
                        mos=mos,
                    )
                )
                switch_ms += actual_switch_ms
                switches += 1
            allocation = target
        trajectory.append(
            {
                "step": decision_index + 1,
                "x": int(position[0]),
                "y": int(position[1]),
                "action": decision.action.name,
                "next_x": int(environment.position[0]),
                "next_y": int(environment.position[1]),
                "source_allocation": f"g{source.resolution}_T{source.depth}",
                "selected_allocation": f"g{target.resolution}_T{target.depth}",
                "best_task_allocation": (
                    ""
                    if selection is None
                    else f"g{ALLOCATIONS[selection.best_task_index].resolution}_T{ALLOCATIONS[selection.best_task_index].depth}"
                ),
                "selected_q": None if selection is None else selection.q_value,
                "best_q": None if selection is None else selection.best_q_value,
                "task_gap": None if selection is None else selection.task_gap,
                "sufficient_allocations": (
                    ""
                    if selection is None
                    else json.dumps(
                        [
                            f"g{ALLOCATIONS[index].resolution}_T{ALLOCATIONS[index].depth}"
                            for index in selection.sufficient_indices
                        ]
                    )
                ),
                "q_values": "" if q_values is None else json.dumps(q_values.tolist()),
                "q_mean": "" if q_mean is None else json.dumps(q_mean.tolist()),
                "q_std": "" if q_std is None else json.dumps(q_std.tolist()),
                "information_losses": (
                    "" if information_losses is None else json.dumps(information_losses.tolist())
                ),
                "selected_information_loss": (
                    None
                    if selection is None or information_losses is None
                    else float(information_losses[selection.index])
                ),
                "belief_entropy_native": (
                    None
                    if context_diagnostics is None
                    else context_diagnostics["belief_entropy_native"]
                ),
                "belief_entropy_canonical": (
                    None
                    if context_diagnostics is None
                    else context_diagnostics["belief_entropy_canonical"]
                ),
                "predicted_entropy_canonical": (
                    None
                    if context_diagnostics is None
                    else context_diagnostics["predicted_entropy_canonical"]
                ),
                "policy_entropy": (
                    None if context_diagnostics is None else context_diagnostics["policy_entropy"]
                ),
                "policy_margin": (
                    None if context_diagnostics is None else context_diagnostics["policy_margin"]
                ),
                "predicted_detection_probability": (
                    None
                    if context_diagnostics is None
                    else context_diagnostics["predicted_detection_probability"]
                ),
                "posterior_weighted_fisher": (
                    None
                    if context_diagnostics is None
                    else context_diagnostics["posterior_weighted_fisher"]
                ),
                "expected_information_gain_nats": (
                    None
                    if context_diagnostics is None
                    else context_diagnostics["expected_information_gain_nats"]
                ),
                "step_cost": float(step_cost),
                "task_inference_ms": decision.total_ms,
                "meta_inference_ms": model_ms,
                "success": bool(success),
            }
        )
        previous_action = decision.action
        if success:
            break
    if not success:
        task_cost += 2.0 * config.max_steps
    summary = {
        "controller": (
            "meta_q_joint_task_compute"
            if config.selection_mode == "joint"
            else "meta_q_cheapest_sufficient"
        ),
        "instance_seed": int(instance_seed),
        "target": list(instance.target),
        "success": bool(success),
        "steps": len(trajectory),
        "task_cost": float(task_cost),
        "task_inference_ms": float(task_inference_ms),
        "meta_inference_ms": float(meta_inference_ms),
        "switch_ms": float(switch_ms),
        "agent_pool_setup_ms": float(pool_setup_ms),
        "total_online_compute_ms": float(task_inference_ms + meta_inference_ms + switch_ms),
        "switches": switches,
        "allocation_steps": {
            f"g{candidate.resolution}_T{candidate.depth}": count
            for candidate, count in counts.items()
            if count
        },
        "allocation_sequence": [row["source_allocation"] for row in trajectory],
    }
    return summary, trajectory


def save_meta_q_evaluation(
    summary: dict[str, Any],
    trajectory: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if trajectory:
        with (output_dir / "trajectory.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(trajectory[0]))
            writer.writeheader()
            writer.writerows(trajectory)


def _run_meta_q_benchmark_instance(
    instance_seed: int,
    config: MetaQEvaluationConfig,
    include_fixed: bool,
) -> dict[str, Any]:
    adaptive, trajectory = run_meta_q_episode(instance_seed, config)
    adaptive = {
        **adaptive,
        "controller": "adaptive",
        "total_compute_ms": adaptive["total_online_compute_ms"],
        "decisions": adaptive["steps"],
        "meta_decisions": max(0, adaptive["steps"] - 1),
        "allocation_counts": json.dumps(adaptive["allocation_steps"], sort_keys=True),
        "allocation_sequence": json.dumps(adaptive["allocation_sequence"]),
    }
    episodes = [adaptive]
    if include_fixed:
        fixed_config = ClosedLoopConfig(
            checkpoint=str(config.checkpoint),
            deadline_median_ms=1.0,
            initial_resolution=config.initial_allocation.resolution,
            initial_depth=config.initial_allocation.depth,
            max_steps=config.max_steps,
            message_passing_iterations=config.message_passing_iterations,
            policy_workers=config.policy_workers,
            device=config.device,
            torch_threads=config.torch_threads,
            include_fixed=True,
        )
        episodes.extend(
            run_fixed_episode(instance_seed, allocation, fixed_config) for allocation in ALLOCATIONS
        )
    for row in trajectory:
        row["instance_seed"] = int(instance_seed)
    return {"episodes": episodes, "trajectory": trajectory}


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0])
    known = set(fieldnames)
    for row in rows[1:]:
        for name in row:
            if name not in known:
                known.add(name)
                fieldnames.append(name)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_meta_q_benchmark(
    *,
    instance_seeds: list[int] | tuple[int, ...],
    output_dir: Path,
    config: MetaQEvaluationConfig,
    include_fixed: bool = True,
    instance_workers: int = 1,
    resume: bool = True,
) -> dict[str, Any]:
    """Resumably compare meta-Q with every fixed allocation on matched seeds."""

    seeds = tuple(int(value) for value in instance_seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("instance_seeds must be nonempty and unique")
    if instance_workers < 1:
        raise ValueError("instance_workers must be positive")
    output_dir = Path(output_dir)
    shard_dir = output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    signature = {
        "configuration": {
            **asdict(config),
            "checkpoint": str(Path(config.checkpoint).resolve()),
            "ensemble_checkpoints": [
                str(Path(path).resolve()) for path in config.ensemble_checkpoints
            ],
        },
        "checkpoint_sha256": [
            hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for path in (Path(config.checkpoint), *config.ensemble_checkpoints)
        ],
        "include_fixed": bool(include_fixed),
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

    def save(seed: int, payload: dict[str, Any]) -> None:
        payload["signature"] = signature
        temporary = shard_dir / f"instance-{seed}.json.tmp"
        temporary.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        temporary.replace(shard_dir / f"instance-{seed}.json")

    if instance_workers == 1:
        for completed, seed in enumerate(pending, start=1):
            save(seed, _run_meta_q_benchmark_instance(seed, config, include_fixed))
            print(f"[evaluate {completed}/{len(pending)}] instance={seed}", flush=True)
    elif pending:
        with ProcessPoolExecutor(max_workers=instance_workers) as executor:
            futures = {
                executor.submit(_run_meta_q_benchmark_instance, seed, config, include_fixed): seed
                for seed in pending
            }
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
    _write_rows(output_dir / "episodes.csv", episodes)
    _write_rows(output_dir / "adaptive_trajectory.csv", trajectory)
    report = {
        "instance_seeds": list(seeds),
        "instance_workers": instance_workers,
        "configuration": signature["configuration"],
        "include_fixed": bool(include_fixed),
        "controllers": _summary(episodes),
        "paired_comparisons": _paired_comparisons(episodes) if include_fixed else {},
    }
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
