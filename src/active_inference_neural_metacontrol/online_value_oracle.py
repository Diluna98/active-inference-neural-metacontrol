"""Exact online preference-plus-epistemic oracle for one MOS episode."""

from __future__ import annotations

import copy
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .allocations import ALLOCATIONS, Allocation
from .counterfactuals import (
    _build_switched_agent,
    _infer_decision,
    _mos_imports,
    _normalized_policy_g,
    _step_cost,
)


@dataclass(frozen=True)
class OnlineValueOracleConfig:
    initial_resolution: int = 2
    initial_depth: int = 1
    max_steps: int = 50
    message_passing_iterations: int = 10
    policy_workers: int = 1

    def __post_init__(self) -> None:
        Allocation(self.initial_resolution, self.initial_depth)
        if min(self.max_steps, self.message_passing_iterations, self.policy_workers) < 1:
            raise ValueError("oracle settings must be positive")

    @property
    def initial_allocation(self) -> Allocation:
        return Allocation(self.initial_resolution, self.initial_depth)


def _new_agent(allocation: Allocation, layout: Any, config: OnlineValueOracleConfig, mos: dict):
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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_online_value_oracle(
    instance_seed: int,
    *,
    config: OnlineValueOracleConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Choose the exact best immediate G/T candidate after every physical action."""

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
    allocation = config.initial_allocation
    agent = _new_agent(allocation, instance.layout, config, mos)
    decision = _infer_decision(agent, observation, None, 0, mos)
    allocation_counts = {candidate: 0 for candidate in ALLOCATIONS}
    trajectory = []
    task_cost = 0.0
    switches = 0
    false_finds = 0
    collisions = 0
    success = False

    for time_step in range(config.max_steps):
        source = allocation
        source_decision = decision
        position = environment.position
        allocation_counts[source] += 1
        observation, success = environment.step(
            source_decision.action,
            noise_quantile=float(quantiles[time_step + 1]),
        )
        false_find = observation[3] == mos["FindOutcome"].FALSE_FIND
        collision = environment.last_collision
        task_cost += _step_cost(false_find=false_find, collision=collision)
        false_finds += int(false_find)
        collisions += int(collision)

        candidate_scores = None
        score_gap = None
        selected = source
        if not success and time_step + 1 < config.max_steps:
            candidate_agents = {}
            candidate_decisions = {}
            for candidate in ALLOCATIONS:
                if candidate == source:
                    candidate_agent = copy.deepcopy(agent)
                else:
                    candidate_agent = _build_switched_agent(
                        source_agent=agent,
                        source_allocation=source,
                        target_allocation=candidate,
                        layout=instance.layout,
                        executed_action=source_decision.action,
                        next_time_step=time_step + 1,
                        message_passing_iterations=config.message_passing_iterations,
                        policy_workers=config.policy_workers,
                        mos=mos,
                    )
                candidate_agents[candidate] = candidate_agent
                candidate_decisions[candidate] = _infer_decision(
                    candidate_agent,
                    observation,
                    source_decision.action,
                    time_step + 1,
                    mos,
                )
            values = np.asarray(
                [_normalized_policy_g(candidate_decisions[item], item) for item in ALLOCATIONS],
                dtype=float,
            )
            order = np.argsort(values)[::-1]
            selected = ALLOCATIONS[int(order[0])]
            score_gap = float(values[order[0]] - values[order[1]])
            candidate_scores = {
                f"g{item.resolution}_T{item.depth}": float(values[index])
                for index, item in enumerate(ALLOCATIONS)
            }
            agent = candidate_agents[selected]
            decision = candidate_decisions[selected]
            allocation = selected
            switches += int(selected != source)

        trajectory.append(
            {
                "instance_seed": int(instance_seed),
                "decision": time_step,
                "x": position[0],
                "y": position[1],
                "action": source_decision.action.name,
                "next_x": environment.position[0],
                "next_y": environment.position[1],
                "source_resolution": source.resolution,
                "source_depth": source.depth,
                "selected_resolution": selected.resolution,
                "selected_depth": selected.depth,
                "oracle_score_gap": score_gap,
                "candidate_scores": (
                    None if candidate_scores is None else json.dumps(candidate_scores)
                ),
                "success": bool(success),
            }
        )
        if success:
            break
        if (time_step + 1) % 5 == 0:
            print(
                f"[online oracle] instance={instance_seed} step={time_step + 1} "
                f"allocation=g{allocation.resolution}_T{allocation.depth}",
                flush=True,
            )

    steps = len(trajectory)
    if not success:
        task_cost += 2.0 * config.max_steps
    summary = {
        "controller": "online_exact_preference_epistemic_oracle",
        "instance_seed": int(instance_seed),
        "success": bool(success),
        "task_cost": float(task_cost),
        "steps": steps,
        "false_finds": false_finds,
        "collisions": collisions,
        "switches": switches,
        "allocation_steps": {
            f"g{item.resolution}_T{item.depth}": count
            for item, count in allocation_counts.items()
            if count
        },
        "allocation_sequence": [
            f"g{row['source_resolution']}_T{row['source_depth']}" for row in trajectory
        ],
    }
    return summary, trajectory


def evaluate_online_value_oracle(
    instance_seed: int,
    *,
    output_dir: Path,
    config: OnlineValueOracleConfig,
) -> dict[str, Any]:
    """Run and persist one focused exact-value oracle episode."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary, trajectory = run_online_value_oracle(instance_seed, config=config)
    report = {"configuration": asdict(config), "result": summary}
    _write_csv(output_dir / "oracle_trajectory.csv", trajectory)
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
