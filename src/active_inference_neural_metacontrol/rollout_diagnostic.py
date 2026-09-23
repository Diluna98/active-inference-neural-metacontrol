"""Candidate-controlled diagnostics for multi-step normalized-G targets."""

from __future__ import annotations

import copy
import csv
import json
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .allocations import ALLOCATIONS, Allocation
from .counterfactuals import (
    DEFAULT_REFERENCE_ALLOCATION,
    _build_switched_agent,
    _infer_decision,
    _mos_imports,
    _rollout_candidate,
    _step_cost,
)


@dataclass(frozen=True)
class GOracleConfig:
    """Full-episode comparison settings for exact rollout-G oracles."""

    max_steps: int = 50
    rollout_horizon: int = 5
    discount: float = 1.0
    message_passing_iterations: int = 10
    policy_workers: int = 1

    def __post_init__(self) -> None:
        if self.max_steps < 1 or self.rollout_horizon < 1:
            raise ValueError("max_steps and rollout_horizon must be positive")
        if not 0 < self.discount <= 1:
            raise ValueError("discount must lie in (0, 1]")
        if self.message_passing_iterations < 1 or self.policy_workers < 1:
            raise ValueError("inference settings must be positive")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _new_agent(allocation: Allocation, layout: Any, config: GOracleConfig, mos: dict) -> Any:
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


def run_g_oracle_episode(
    instance_seed: int,
    *,
    oracle_horizon: int,
    config: GOracleConfig,
    allocations: Sequence[Allocation] = ALLOCATIONS,
) -> dict[str, Any]:
    """Run a privileged full episode using exact candidate-controlled G rollouts."""

    if oracle_horizon < 1:
        raise ValueError("oracle_horizon must be positive")
    allocation_tuple = tuple(allocations)
    mos = _mos_imports()
    instance = mos["sample_mos_instance"](int(instance_seed))
    quantile_seed = (
        instance.layout.map_seed * 1_000_003
        + instance.target_seed * 1009
        + instance.observation_seed
    )
    quantiles = np.random.default_rng(quantile_seed).random(config.max_steps + oracle_horizon + 1)
    environment = mos["MOSEnvironment"](
        layout=instance.layout,
        target=instance.target,
        observation_seed=instance.observation_seed,
    )
    observation = environment.reset(noise_quantile=float(quantiles[0]))
    candidate_agents = {
        allocation: _new_agent(allocation, instance.layout, config, mos)
        for allocation in allocation_tuple
    }
    decisions = {
        allocation: _infer_decision(agent, observation, None, 0, mos)
        for allocation, agent in candidate_agents.items()
    }
    task_cost = 0.0
    success = False
    false_finds = 0
    collisions = 0
    steps = 0
    switches = 0
    previous_allocation: Allocation | None = None
    allocation_counts = {allocation: 0 for allocation in allocation_tuple}

    for time_step in range(config.max_steps):
        rollout_values = []
        for allocation in allocation_tuple:
            rollout = _rollout_candidate(
                agent=copy.deepcopy(candidate_agents[allocation]),
                environment=copy.deepcopy(environment),
                first_decision=decisions[allocation],
                allocation=allocation,
                first_time_step=time_step,
                quantiles=quantiles,
                horizon=oracle_horizon,
                discount=config.discount,
                mos=mos,
            )
            rollout_values.append(float(rollout["discounted_g_mean"]))
        selected_index = int(np.argmax(rollout_values))
        allocation = allocation_tuple[selected_index]
        selected_agent = candidate_agents[allocation]
        selected_decision = decisions[allocation]
        allocation_counts[allocation] += 1
        if previous_allocation is not None and allocation != previous_allocation:
            switches += 1
        observation, success = environment.step(
            selected_decision.action,
            noise_quantile=float(quantiles[time_step + 1]),
        )
        false_find = observation[3] == mos["FindOutcome"].FALSE_FIND
        collision = environment.last_collision
        task_cost += _step_cost(false_find=false_find, collision=collision)
        false_finds += int(false_find)
        collisions += int(collision)
        steps += 1
        if success or time_step + 1 >= config.max_steps:
            break
        next_time_step = time_step + 1
        next_agents = {}
        next_decisions = {}
        for candidate in allocation_tuple:
            agent = _build_switched_agent(
                source_agent=selected_agent,
                source_allocation=allocation,
                target_allocation=candidate,
                layout=instance.layout,
                executed_action=selected_decision.action,
                next_time_step=next_time_step,
                message_passing_iterations=config.message_passing_iterations,
                policy_workers=config.policy_workers,
                mos=mos,
            )
            next_agents[candidate] = agent
            next_decisions[candidate] = _infer_decision(
                agent,
                observation,
                selected_decision.action,
                next_time_step,
                mos,
            )
        previous_allocation = allocation
        candidate_agents = next_agents
        decisions = next_decisions

    if not success:
        task_cost += 2.0 * config.max_steps
    return {
        "controller": f"g_oracle_H{oracle_horizon}",
        "instance_seed": int(instance_seed),
        "success": bool(success),
        "task_cost": float(task_cost),
        "steps": steps,
        "false_finds": false_finds,
        "collisions": collisions,
        "switches": switches,
        "allocation_counts": json.dumps(
            {
                f"g{allocation.resolution}_T{allocation.depth}": count
                for allocation, count in allocation_counts.items()
                if count
            },
            sort_keys=True,
        ),
    }


def _run_oracle_instance(instance_seed: int, config: GOracleConfig) -> list[dict[str, Any]]:
    from .closed_loop import ClosedLoopConfig, run_fixed_episode

    oracle_rows = [
        run_g_oracle_episode(instance_seed, oracle_horizon=horizon, config=config)
        for horizon in sorted({1, config.rollout_horizon})
    ]
    fixed_config = ClosedLoopConfig(
        checkpoint="unused-by-fixed-controller",
        deadline_median_ms=100.0,
        max_steps=config.max_steps,
        message_passing_iterations=config.message_passing_iterations,
        policy_workers=config.policy_workers,
    )
    oracle_rows.append(run_fixed_episode(instance_seed, Allocation(5, 2), fixed_config))
    return oracle_rows


def evaluate_g_oracles(
    *,
    instance_seeds: Sequence[int],
    output_dir: Path,
    config: GOracleConfig,
    instance_workers: int = 1,
) -> dict[str, Any]:
    """Compare one-step, accumulated-G, and fixed controllers over full episodes."""

    seeds = tuple(int(seed) for seed in instance_seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("instance_seeds must be nonempty and unique")
    if instance_workers < 1:
        raise ValueError("instance_workers must be positive")
    rows: list[dict[str, Any]] = []
    if instance_workers == 1:
        for completed, seed in enumerate(seeds, start=1):
            rows.extend(_run_oracle_instance(seed, config))
            print(f"[oracle {completed}/{len(seeds)}] instance={seed}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=instance_workers) as executor:
            futures = {executor.submit(_run_oracle_instance, seed, config): seed for seed in seeds}
            for completed, future in enumerate(as_completed(futures), start=1):
                seed = futures[future]
                rows.extend(future.result())
                print(f"[oracle {completed}/{len(seeds)}] instance={seed}", flush=True)
    rows.sort(key=lambda row: (int(row["instance_seed"]), str(row["controller"])))
    controllers = sorted({str(row["controller"]) for row in rows})
    summary = {}
    for controller in controllers:
        selected = [row for row in rows if row["controller"] == controller]
        summary[controller] = {
            "episodes": len(selected),
            "success_rate": float(np.mean([row["success"] for row in selected])),
            "mean_task_cost": float(np.mean([row["task_cost"] for row in selected])),
            "mean_steps": float(np.mean([row["steps"] for row in selected])),
            "mean_switches": float(np.mean([row["switches"] for row in selected])),
        }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "episodes.csv", rows)
    report = {
        "instance_seeds": list(seeds),
        "configuration": asdict(config),
        "controllers": summary,
        "oracle_notice": "Privileged simulator-rollout diagnostic; not deployable.",
    }
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def diagnose_accumulated_g(
    *,
    instance_seeds: Sequence[int],
    output_dir: Path,
    reference_allocation: Allocation = DEFAULT_REFERENCE_ALLOCATION,
    allocations: Sequence[Allocation] = ALLOCATIONS,
    max_reference_steps: int = 20,
    branch_stride: int = 5,
    rollout_horizon: int = 5,
    discount: float = 1.0,
    message_passing_iterations: int = 10,
    policy_workers: int = 1,
) -> dict[str, Any]:
    """Compare one-step and candidate-controlled accumulated-G oracles."""

    allocation_tuple = tuple(allocations)
    if not instance_seeds:
        raise ValueError("instance_seeds must be nonempty")
    if reference_allocation not in allocation_tuple:
        raise ValueError("reference_allocation must be included in allocations")
    if max_reference_steps < 2 or branch_stride < 1 or rollout_horizon < 1:
        raise ValueError("step counts, stride, and rollout horizon must be positive")
    if not 0 < discount <= 1:
        raise ValueError("discount must lie in (0, 1]")

    mos = _mos_imports()
    branch_rows: list[dict[str, Any]] = []
    context_rows: list[dict[str, Any]] = []
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
            max_reference_steps + rollout_horizon + 1
        )
        environment = mos["MOSEnvironment"](
            layout=instance.layout,
            target=instance.target,
            observation_seed=instance.observation_seed,
        )
        observation = environment.reset(noise_quantile=float(quantiles[0]))
        previous_action = None
        reference_decision = _infer_decision(reference_agent, observation, previous_action, 0, mos)

        for decision_index in range(max_reference_steps):
            next_observation, reference_success = environment.step(
                reference_decision.action,
                noise_quantile=float(quantiles[decision_index + 1]),
            )
            if reference_success or decision_index + 1 >= max_reference_steps:
                break
            next_time_step = decision_index + 1
            if decision_index % branch_stride == 0:
                context_id = f"instance-{instance_seed}-decision-{next_time_step}"
                results: list[dict[str, Any]] = []
                for allocation in allocation_tuple:
                    candidate_agent = _build_switched_agent(
                        source_agent=reference_agent,
                        source_allocation=reference_allocation,
                        target_allocation=allocation,
                        layout=instance.layout,
                        executed_action=reference_decision.action,
                        next_time_step=next_time_step,
                        message_passing_iterations=message_passing_iterations,
                        policy_workers=policy_workers,
                        mos=mos,
                    )
                    candidate_decision = _infer_decision(
                        candidate_agent,
                        next_observation,
                        reference_decision.action,
                        next_time_step,
                        mos,
                    )
                    result = _rollout_candidate(
                        agent=candidate_agent,
                        environment=copy.deepcopy(environment),
                        first_decision=candidate_decision,
                        allocation=allocation,
                        first_time_step=next_time_step,
                        quantiles=quantiles,
                        horizon=rollout_horizon,
                        discount=discount,
                        mos=mos,
                    )
                    result.update(
                        {
                            "context_id": context_id,
                            "instance_seed": int(instance_seed),
                            "candidate_resolution": allocation.resolution,
                            "candidate_depth": allocation.depth,
                            "first_action": candidate_decision.action.name,
                        }
                    )
                    results.append(result)
                    branch_rows.append(result.copy())

                first_index = int(
                    np.argmax([float(result["first_normalized_g"]) for result in results])
                )
                rollout_index = int(
                    np.argmax([float(result["discounted_g_mean"]) for result in results])
                )
                first_choice = allocation_tuple[first_index]
                rollout_choice = allocation_tuple[rollout_index]
                first_result = results[first_index]
                rollout_result = results[rollout_index]
                context_rows.append(
                    {
                        "context_id": context_id,
                        "instance_seed": int(instance_seed),
                        "one_step_resolution": first_choice.resolution,
                        "one_step_depth": first_choice.depth,
                        "rollout_resolution": rollout_choice.resolution,
                        "rollout_depth": rollout_choice.depth,
                        "allocation_disagreement": first_choice != rollout_choice,
                        "first_action_disagreement": (
                            first_result["first_action"] != rollout_result["first_action"]
                        ),
                        "one_step_oracle_rollout_g": first_result["discounted_g_mean"],
                        "accumulated_oracle_rollout_g": rollout_result["discounted_g_mean"],
                        "one_step_oracle_success": first_result["success_within_horizon"],
                        "accumulated_oracle_success": rollout_result["success_within_horizon"],
                        "one_step_oracle_task_cost": first_result["task_cost"],
                        "accumulated_oracle_task_cost": rollout_result["task_cost"],
                    }
                )

            next_reference_agent = _build_switched_agent(
                source_agent=reference_agent,
                source_allocation=reference_allocation,
                target_allocation=reference_allocation,
                layout=instance.layout,
                executed_action=reference_decision.action,
                next_time_step=next_time_step,
                message_passing_iterations=message_passing_iterations,
                policy_workers=policy_workers,
                mos=mos,
            )
            reference_decision = _infer_decision(
                next_reference_agent,
                next_observation,
                reference_decision.action,
                next_time_step,
                mos,
            )
            reference_agent = next_reference_agent

    contexts = len(context_rows)
    if not contexts:
        raise ValueError("the reference trajectories produced no diagnostic contexts")
    summary = {
        "instances": len(instance_seeds),
        "contexts": contexts,
        "rollout_horizon": rollout_horizon,
        "discount": discount,
        "allocation_disagreement_rate": float(
            np.mean([row["allocation_disagreement"] for row in context_rows])
        ),
        "first_action_disagreement_rate": float(
            np.mean([row["first_action_disagreement"] for row in context_rows])
        ),
        "one_step_oracle_success_rate": float(
            np.mean([row["one_step_oracle_success"] for row in context_rows])
        ),
        "accumulated_oracle_success_rate": float(
            np.mean([row["accumulated_oracle_success"] for row in context_rows])
        ),
        "one_step_oracle_mean_task_cost": float(
            np.mean([row["one_step_oracle_task_cost"] for row in context_rows])
        ),
        "accumulated_oracle_mean_task_cost": float(
            np.mean([row["accumulated_oracle_task_cost"] for row in context_rows])
        ),
        "mean_accumulated_g_improvement": float(
            np.mean(
                [
                    row["accumulated_oracle_rollout_g"] - row["one_step_oracle_rollout_g"]
                    for row in context_rows
                ]
            )
        ),
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "branches.csv", branch_rows)
    _write_csv(output_dir / "contexts.csv", context_rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
