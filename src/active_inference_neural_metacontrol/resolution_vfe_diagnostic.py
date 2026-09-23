"""Matched-observation accuracy/complexity diagnostics across MOS resolutions."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PyAIF.numerics import log_stable_probability

from .allocations import Allocation
from .counterfactuals import _build_switched_agent, _infer_decision, _mos_imports

RESOLUTIONS = (2, 5, 10, 20)


@dataclass(frozen=True)
class ResolutionVFEDiagnosticConfig:
    reference_resolution: int = 10
    max_steps: int = 50
    message_passing_iterations: int = 10
    policy_workers: int = 1

    def __post_init__(self) -> None:
        if self.reference_resolution not in RESOLUTIONS:
            raise ValueError(f"reference_resolution must be one of {RESOLUTIONS}")
        if min(self.max_steps, self.message_passing_iterations, self.policy_workers) < 1:
            raise ValueError("diagnostic settings must be positive")

    @property
    def reference_allocation(self) -> Allocation:
        return Allocation(self.reference_resolution, 1)


def _new_agent(allocation: Allocation, layout: Any, config: ResolutionVFEDiagnosticConfig, mos):
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


def _state_row(
    *,
    agent: Any,
    observation: tuple[int, ...],
    previous_action: Any | None,
    time_step: int,
    resolution: int,
    position: tuple[int, int],
    mos: dict[str, Any],
) -> dict[str, Any]:
    controls = None if previous_action is None else mos["mos_action_controls"](previous_action)
    agent.observe(observation, time_step=time_step, executed_action=controls)
    core = getattr(agent, "_agent", agent)
    priors = [np.asarray(prior, dtype=float).copy() for prior in core._receding_prior]
    result = agent.infer_states()
    if result is None:
        result = agent.last_state_inference
    beliefs = [np.asarray(posterior, dtype=float) for posterior in result.posteriors]

    complexity_by_factor = []
    normalized_complexity_by_factor = []
    for posterior, prior in zip(beliefs, priors):
        complexity = float(
            posterior.dot(
                log_stable_probability(posterior) - log_stable_probability(prior)
            )
        )
        complexity_by_factor.append(complexity)
        normalized_complexity_by_factor.append(complexity / np.log(posterior.size))

    accuracy_by_modality = []
    normalized_accuracy_by_modality = []
    for modality, dependencies in enumerate(core.mod_dep):
        likelihood = np.take(core.A[modality], observation[modality], axis=0)
        arguments: list[Any] = [log_stable_probability(likelihood), list(dependencies)]
        for dependency in dependencies:
            arguments.extend([beliefs[dependency], [dependency]])
        arguments.append([])
        accuracy = float(np.einsum(*arguments, optimize=True))
        accuracy_by_modality.append(accuracy)
        normalized_accuracy_by_modality.append(accuracy / np.log(core.obs_dim[modality]))

    corrected_accuracy = float(sum(accuracy_by_modality))
    normalized_accuracy = float(sum(normalized_accuracy_by_modality))
    normalized_complexity = float(sum(normalized_complexity_by_factor))
    return {
        "time_step": time_step,
        "x": position[0],
        "y": position[1],
        "resolution": resolution,
        "legacy_accuracy": float(result.accuracy),
        "corrected_accuracy": corrected_accuracy,
        "raw_complexity": float(sum(complexity_by_factor)),
        "corrected_vfe": float(sum(complexity_by_factor) - corrected_accuracy),
        "normalized_accuracy": normalized_accuracy,
        "normalized_complexity": normalized_complexity,
        "normalized_score": normalized_complexity - normalized_accuracy,
        "accuracy_by_modality": json.dumps(accuracy_by_modality),
        "normalized_accuracy_by_modality": json.dumps(normalized_accuracy_by_modality),
        "complexity_by_factor": json.dumps(complexity_by_factor),
        "normalized_complexity_by_factor": json.dumps(normalized_complexity_by_factor),
        "iterations": int(result.iterations),
        "converged": bool(result.converged),
    }


def _summarize(rows: list[dict[str, Any]], reference_resolution: int) -> dict[str, Any]:
    score_winners: dict[int, int] = {resolution: 0 for resolution in RESOLUTIONS}
    corrected_vfe_winners: dict[int, int] = {
        resolution: 0 for resolution in RESOLUTIONS
    }
    accuracy_winners: dict[int, int] = {resolution: 0 for resolution in RESOLUTIONS}
    complexity_winners: dict[int, int] = {resolution: 0 for resolution in RESOLUTIONS}
    by_resolution: dict[str, Any] = {}
    time_steps = sorted({int(row["time_step"]) for row in rows})
    reference_score = {
        int(row["time_step"]): float(row["normalized_score"])
        for row in rows
        if row["resolution"] == reference_resolution
    }
    for time_step in time_steps:
        step_rows = [row for row in rows if row["time_step"] == time_step]
        score_winner = min(step_rows, key=lambda row: row["normalized_score"])
        corrected_vfe_winner = min(step_rows, key=lambda row: row["corrected_vfe"])
        accuracy_winner = max(step_rows, key=lambda row: row["normalized_accuracy"])
        complexity_winner = min(step_rows, key=lambda row: row["normalized_complexity"])
        score_winners[int(score_winner["resolution"])] += 1
        corrected_vfe_winners[int(corrected_vfe_winner["resolution"])] += 1
        accuracy_winners[int(accuracy_winner["resolution"])] += 1
        complexity_winners[int(complexity_winner["resolution"])] += 1
    for resolution in RESOLUTIONS:
        selected = [row for row in rows if row["resolution"] == resolution]
        scores = np.asarray([row["normalized_score"] for row in selected], dtype=float)
        by_resolution[str(resolution)] = {
            "observations": len(selected),
            "mean_corrected_accuracy_nats": float(
                np.mean([row["corrected_accuracy"] for row in selected])
            ),
            "mean_raw_complexity_nats": float(
                np.mean([row["raw_complexity"] for row in selected])
            ),
            "mean_corrected_vfe_nats": float(
                np.mean([row["corrected_vfe"] for row in selected])
            ),
            "mean_normalized_accuracy": float(
                np.mean([row["normalized_accuracy"] for row in selected])
            ),
            "mean_normalized_complexity": float(
                np.mean([row["normalized_complexity"] for row in selected])
            ),
            "mean_normalized_score": float(scores.mean()),
            "median_normalized_score": float(np.median(scores)),
            "mean_delta_score_vs_reference": float(
                np.mean(
                    [
                        row["normalized_score"]
                        - reference_score[int(row["time_step"])]
                        for row in selected
                    ]
                )
            ),
            "best_accuracy_count": accuracy_winners[resolution],
            "lowest_complexity_count": complexity_winners[resolution],
            "lowest_corrected_vfe_count": corrected_vfe_winners[resolution],
            "lowest_normalized_score_count": score_winners[resolution],
        }
    return {
        "observations": len(time_steps),
        "reference_resolution": reference_resolution,
        "lower_normalized_score_is_better": True,
        "identity": (
            "normalized_score = sum_f(KL_f/log|S_f|) "
            "- sum_m(accuracy_m/log|O_m|)"
        ),
        "by_resolution": by_resolution,
    }


def run_resolution_vfe_diagnostic(
    instance_seed: int,
    *,
    config: ResolutionVFEDiagnosticConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run a fixed reference trajectory and score each observation at every resolution."""

    mos = _mos_imports()
    instance = mos["sample_mos_instance"](int(instance_seed))
    quantile_seed = (
        instance.layout.map_seed * 1_000_003
        + instance.target_seed * 1009
        + instance.observation_seed
    )
    quantiles = np.random.default_rng(quantile_seed).random(config.max_steps + 1)
    environment = mos["MOSEnvironment"](
        layout=instance.layout,
        target=instance.target,
        observation_seed=instance.observation_seed,
    )
    observation = environment.reset(noise_quantile=float(quantiles[0]))
    reference_allocation = config.reference_allocation
    reference_agent = _new_agent(reference_allocation, instance.layout, config, mos)
    reference_decision = _infer_decision(reference_agent, observation, None, 0, mos)
    rows: list[dict[str, Any]] = []

    # The initial observation has a common uniform prior at every resolution.
    for resolution in RESOLUTIONS:
        allocation = Allocation(resolution, 1)
        candidate = _new_agent(allocation, instance.layout, config, mos)
        rows.append(
            _state_row(
                agent=candidate,
                observation=observation,
                previous_action=None,
                time_step=0,
                resolution=resolution,
                position=environment.position,
                mos=mos,
            )
        )

    success = False
    for decision_index in range(config.max_steps):
        previous_agent = reference_agent
        previous_decision = reference_decision
        observation, success = environment.step(
            previous_decision.action,
            noise_quantile=float(quantiles[decision_index + 1]),
        )
        next_time_step = decision_index + 1
        for resolution in RESOLUTIONS:
            allocation = Allocation(resolution, 1)
            candidate = _build_switched_agent(
                source_agent=previous_agent,
                source_allocation=reference_allocation,
                target_allocation=allocation,
                layout=instance.layout,
                executed_action=previous_decision.action,
                next_time_step=next_time_step,
                message_passing_iterations=config.message_passing_iterations,
                policy_workers=config.policy_workers,
                mos=mos,
            )
            rows.append(
                _state_row(
                    agent=candidate,
                    observation=observation,
                    previous_action=previous_decision.action,
                    time_step=next_time_step,
                    resolution=resolution,
                    position=environment.position,
                    mos=mos,
                )
            )
        if success or next_time_step >= config.max_steps:
            break
        reference_agent = _build_switched_agent(
            source_agent=previous_agent,
            source_allocation=reference_allocation,
            target_allocation=reference_allocation,
            layout=instance.layout,
            executed_action=previous_decision.action,
            next_time_step=next_time_step,
            message_passing_iterations=config.message_passing_iterations,
            policy_workers=config.policy_workers,
            mos=mos,
        )
        reference_decision = _infer_decision(
            reference_agent,
            observation,
            previous_decision.action,
            next_time_step,
            mos,
        )

    summary = _summarize(rows, config.reference_resolution)
    summary.update(
        {
            "instance_seed": int(instance_seed),
            "reference_success": bool(success),
            "reference_steps": min(config.max_steps, len({row['time_step'] for row in rows}) - 1),
        }
    )
    return summary, rows


def evaluate_resolution_vfe_diagnostic(
    instance_seed: int,
    *,
    output_dir: Path,
    config: ResolutionVFEDiagnosticConfig,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary, rows = run_resolution_vfe_diagnostic(instance_seed, config=config)
    with (output_dir / "resolution_vfe.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {"configuration": asdict(config), "result": summary}
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
