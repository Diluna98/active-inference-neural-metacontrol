"""Interpretable adaptive baselines for resolution/depth metacontrol."""

from __future__ import annotations

import csv
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from .allocations import ALLOCATIONS, RESOLUTIONS, Allocation
from .beliefs import canonicalize_posterior, information_loss
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
from .information import categorical_fisher_map
from .mos_adapter import (
    canonical_detection_likelihood,
    selected_policy_next_target_posterior,
)


def _normalized_entropy(values: np.ndarray) -> float:
    probabilities = np.asarray(values, dtype=float).reshape(-1)
    if np.any(probabilities < 0) or not np.all(np.isfinite(probabilities)):
        raise ValueError("probabilities must be finite and nonnegative")
    total = float(probabilities.sum())
    if total <= 0:
        raise ValueError("probabilities must have positive mass")
    probabilities = probabilities / total
    positive = probabilities > 0
    entropy = float(-np.sum(probabilities[positive] * np.log(probabilities[positive])))
    maximum = float(np.log(probabilities.size))
    return 0.0 if maximum <= 0 else float(np.clip(entropy / maximum, 0.0, 1.0))


def first_action_uncertainty(agent: Any) -> float:
    """Entropy of the marginal first-action posterior, comparable across depths."""

    policies = np.asarray(agent.policies)
    posterior = np.asarray(agent.posterior_pi, dtype=float).reshape(-1)
    if policies.ndim < 2 or policies.shape[0] != posterior.size:
        raise ValueError("policies and policy posterior are incompatible")
    first_controls = policies[:, 0].reshape(policies.shape[0], -1)
    _, inverse = np.unique(first_controls, axis=0, return_inverse=True)
    action_posterior = np.bincount(inverse, weights=posterior)
    return _normalized_entropy(action_posterior)


@dataclass(frozen=True)
class HeuristicSignals:
    """Resolution-independent signals available at one meta decision."""

    posterior_concentration: float
    policy_uncertainty: float
    expected_fisher: float
    detection_prediction_error: float
    predicted_detection_probability: float

    def __post_init__(self) -> None:
        for value in asdict(self).values():
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("heuristic signals must lie in [0, 1]")


@dataclass(frozen=True)
class HeuristicDecision:
    allocation: Allocation
    raw_allocation: Allocation
    resolution_score: float
    depth_score: float
    information_loss: float
    held: bool = False


@dataclass(frozen=True)
class HeuristicEvaluationConfig:
    """Configuration shared by entropy and Fisher/surprise baselines."""

    mode: str = "entropy"
    initial_allocation: Allocation = field(default_factory=lambda: Allocation(2, 1))
    max_steps: int = 50
    message_passing_iterations: int = 10
    policy_workers: int = 1
    resolution_thresholds: tuple[float, float, float] = (0.05, 0.15, 0.35)
    depth_thresholds: tuple[float, float] = (0.75, 0.90)
    fisher_weight: float = 0.5
    prediction_error_weight: float = 0.5
    information_loss_limit: float = 1.0
    minimum_hold_steps: int = 0

    def __post_init__(self) -> None:
        if self.mode not in {"entropy", "fisher_surprise"}:
            raise ValueError("mode must be 'entropy' or 'fisher_surprise'")
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if self.message_passing_iterations < 1 or self.policy_workers < 1:
            raise ValueError("inference settings must be positive")
        if tuple(sorted(self.resolution_thresholds)) != self.resolution_thresholds:
            raise ValueError("resolution thresholds must be nondecreasing")
        if tuple(sorted(self.depth_thresholds)) != self.depth_thresholds:
            raise ValueError("depth thresholds must be nondecreasing")
        values = (
            *self.resolution_thresholds,
            *self.depth_thresholds,
            self.fisher_weight,
            self.prediction_error_weight,
            self.information_loss_limit,
        )
        if any(not 0.0 <= value <= 1.0 for value in values):
            raise ValueError("heuristic thresholds and weights must lie in [0, 1]")
        if self.minimum_hold_steps < 0:
            raise ValueError("minimum_hold_steps cannot be negative")


def heuristic_signals(
    *,
    agent: Any,
    allocation: Allocation,
    layout: Any,
    next_robot_position: tuple[int, int],
    realized_observation: tuple[int, ...],
) -> HeuristicSignals:
    """Calculate post-action signals without running another task-level planner.

    The target prediction already computed for the selected task policy is
    conditioned on the newly realized detection observation. This gives a
    resolution-independent approximation of the belief available at the next
    decision while avoiding an extra state-inference call.
    """

    predicted_native = selected_policy_next_target_posterior(agent)
    predicted = canonicalize_posterior(predicted_native, allocation.resolution, layout.size)
    likelihood = canonical_detection_likelihood(layout, next_robot_position)
    detection_outcome = int(realized_observation[2])
    if detection_outcome not in (0, 1):
        raise ValueError("MOS detection observation must be binary")
    outcome_likelihood = likelihood[detection_outcome]
    evidence = float(np.sum(predicted * outcome_likelihood))
    updated = predicted * outcome_likelihood
    if updated.sum() > 0:
        updated /= updated.sum()
    else:  # Defensive fallback for a numerically impossible observation.
        updated = predicted

    policy_uncertainty = first_action_uncertainty(agent)
    concentration = 1.0 - _normalized_entropy(updated)
    fisher = categorical_fisher_map(likelihood)
    expected_fisher = float(np.clip(np.sum(updated * fisher), 0.0, 1.0))
    predicted_detection = float(np.sum(predicted * likelihood[1]))
    prediction_error = abs(float(detection_outcome) - predicted_detection)

    # Evidence is evaluated above deliberately: it validates that the realized
    # outcome is represented by the predictive model and guards silent NaNs.
    if not np.isfinite(evidence) or evidence <= 0:
        raise ValueError("realized detection has zero or invalid predictive evidence")
    return HeuristicSignals(
        posterior_concentration=float(np.clip(concentration, 0.0, 1.0)),
        policy_uncertainty=float(np.clip(policy_uncertainty, 0.0, 1.0)),
        expected_fisher=expected_fisher,
        detection_prediction_error=float(np.clip(prediction_error, 0.0, 1.0)),
        predicted_detection_probability=float(np.clip(predicted_detection, 0.0, 1.0)),
    )


def _threshold_resolution(score: float, thresholds: tuple[float, float, float]) -> int:
    return RESOLUTIONS[
        int(score >= thresholds[0]) + int(score >= thresholds[1]) + int(score >= thresholds[2])
    ]


def _threshold_depth(score: float, thresholds: tuple[float, float]) -> int:
    return 1 + int(score >= thresholds[0]) + int(score >= thresholds[1])


def choose_heuristic_allocation(
    *,
    signals: HeuristicSignals,
    posterior: np.ndarray,
    current_allocation: Allocation,
    config: HeuristicEvaluationConfig,
    steps_since_switch: int,
    canonical_size: int = 20,
) -> HeuristicDecision:
    """Map interpretable signals to one allocation with optional hysteresis."""

    if config.mode == "entropy":
        resolution_score = signals.posterior_concentration
        depth_score = signals.policy_uncertainty
    else:
        resolution_score = (
            1.0 - config.fisher_weight
        ) * signals.posterior_concentration + config.fisher_weight * signals.expected_fisher
        depth_score = (
            (1.0 - config.prediction_error_weight) * signals.policy_uncertainty
            + config.prediction_error_weight * signals.detection_prediction_error
        )

    raw = Allocation(
        _threshold_resolution(resolution_score, config.resolution_thresholds),
        _threshold_depth(depth_score, config.depth_thresholds),
    )
    if raw != current_allocation and steps_since_switch < config.minimum_hold_steps:
        return HeuristicDecision(
            allocation=current_allocation,
            raw_allocation=raw,
            resolution_score=float(resolution_score),
            depth_score=float(depth_score),
            information_loss=0.0,
            held=True,
        )

    target_resolution = raw.resolution
    loss = information_loss(
        posterior,
        current_allocation.resolution,
        target_resolution,
        canonical_size,
    )
    if target_resolution < current_allocation.resolution and loss > config.information_loss_limit:
        admissible = []
        for resolution in RESOLUTIONS:
            if not target_resolution <= resolution <= current_allocation.resolution:
                continue
            candidate_loss = information_loss(
                posterior,
                current_allocation.resolution,
                resolution,
                canonical_size,
            )
            if candidate_loss <= config.information_loss_limit:
                admissible.append((resolution, candidate_loss))
        target_resolution, loss = min(admissible, key=lambda item: item[0])
    target = Allocation(target_resolution, raw.depth)
    return HeuristicDecision(
        allocation=target,
        raw_allocation=raw,
        resolution_score=float(resolution_score),
        depth_score=float(depth_score),
        information_loss=float(loss),
    )


def run_heuristic_episode(
    instance_seed: int,
    config: HeuristicEvaluationConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run one heuristic-controller episode using the cached agent pool."""

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
    previous_action = None
    task_cost = 0.0
    task_inference_ms = 0.0
    meta_inference_ms = 0.0
    switch_ms = 0.0
    switches = 0
    steps_since_switch = config.minimum_hold_steps
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
        target = allocation
        signals = None
        meta_decision = None
        model_ms = 0.0
        if not success and decision_index + 1 < config.max_steps:
            signals, signals_ms = _timed(
                partial(
                    heuristic_signals,
                    agent=agent,
                    allocation=allocation,
                    layout=instance.layout,
                    next_robot_position=next_position,
                    realized_observation=observation,
                )
            )
            meta_decision, selection_ms = _timed(
                partial(
                    choose_heuristic_allocation,
                    signals=signals,
                    posterior=np.asarray(agent.filtered_posteriors[2], dtype=float),
                    current_allocation=allocation,
                    config=config,
                    steps_since_switch=steps_since_switch,
                    canonical_size=instance.layout.size,
                )
            )
            model_ms = signals_ms + selection_ms
            meta_inference_ms += model_ms
            target = meta_decision.allocation
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
                steps_since_switch = 0
            else:
                steps_since_switch += 1
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
                "raw_heuristic_allocation": (
                    ""
                    if meta_decision is None
                    else f"g{meta_decision.raw_allocation.resolution}_T{meta_decision.raw_allocation.depth}"
                ),
                "posterior_concentration": (
                    None if signals is None else signals.posterior_concentration
                ),
                "policy_uncertainty": None if signals is None else signals.policy_uncertainty,
                "expected_fisher": None if signals is None else signals.expected_fisher,
                "detection_prediction_error": (
                    None if signals is None else signals.detection_prediction_error
                ),
                "predicted_detection_probability": (
                    None if signals is None else signals.predicted_detection_probability
                ),
                "resolution_score": (
                    None if meta_decision is None else meta_decision.resolution_score
                ),
                "depth_score": None if meta_decision is None else meta_decision.depth_score,
                "selected_information_loss": (
                    None if meta_decision is None else meta_decision.information_loss
                ),
                "held_by_hysteresis": (False if meta_decision is None else meta_decision.held),
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
        "controller": f"heuristic_{config.mode}",
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


def _run_benchmark_instance(
    instance_seed: int,
    config: HeuristicEvaluationConfig,
    include_fixed: bool,
) -> dict[str, Any]:
    adaptive, trajectory = run_heuristic_episode(instance_seed, config)
    adaptive = {
        **adaptive,
        "controller": "adaptive",
        "heuristic_controller": adaptive["controller"],
        "total_compute_ms": adaptive["total_online_compute_ms"],
        "decisions": adaptive["steps"],
        "meta_decisions": max(0, adaptive["steps"] - 1),
        "allocation_counts": json.dumps(adaptive["allocation_steps"], sort_keys=True),
        "allocation_sequence": json.dumps(adaptive["allocation_sequence"]),
    }
    episodes = [adaptive]
    if include_fixed:
        fixed_config = ClosedLoopConfig(
            checkpoint="heuristic-baseline-does-not-use-a-checkpoint",
            deadline_median_ms=1.0,
            initial_resolution=config.initial_allocation.resolution,
            initial_depth=config.initial_allocation.depth,
            max_steps=config.max_steps,
            message_passing_iterations=config.message_passing_iterations,
            policy_workers=config.policy_workers,
            include_fixed=True,
        )
        episodes.extend(
            run_fixed_episode(instance_seed, allocation, fixed_config) for allocation in ALLOCATIONS
        )
    for row in trajectory:
        row["instance_seed"] = int(instance_seed)
        row["heuristic_controller"] = f"heuristic_{config.mode}"
    return {"episodes": episodes, "trajectory": trajectory}


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    known: set[str] = set()
    for row in rows:
        for name in row:
            if name not in known:
                known.add(name)
                fieldnames.append(name)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_heuristic_benchmark(
    *,
    instance_seeds: list[int] | tuple[int, ...],
    output_dir: Path,
    config: HeuristicEvaluationConfig,
    include_fixed: bool = True,
    instance_workers: int = 1,
    resume: bool = True,
) -> dict[str, Any]:
    """Resumably evaluate a heuristic baseline on matched MOS instances."""

    seeds = tuple(int(value) for value in instance_seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("instance_seeds must be nonempty and unique")
    if instance_workers < 1:
        raise ValueError("instance_workers must be positive")
    output_dir = Path(output_dir)
    shard_dir = output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    configuration = json.loads(json.dumps(asdict(config)))
    signature = {
        "configuration": configuration,
        "include_fixed": bool(include_fixed),
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
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
            save(seed, _run_benchmark_instance(seed, config, include_fixed))
            print(f"[evaluate {completed}/{len(pending)}] instance={seed}", flush=True)
    elif pending:
        with ProcessPoolExecutor(max_workers=instance_workers) as executor:
            futures = {
                executor.submit(_run_benchmark_instance, seed, config, include_fixed): seed
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


def save_heuristic_evaluation(
    summary: dict[str, Any],
    trajectory: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_rows(output_dir / "trajectory.csv", trajectory)
