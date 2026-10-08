"""Compare neural and exact PyAIF allocation rankings along one adaptive episode."""

from __future__ import annotations

import copy
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .allocations import ALLOCATION_INDEX, ALLOCATIONS, DEPTHS, RESOLUTIONS, Allocation
from .closed_loop import (
    ClosedLoopConfig,
    NeuralMetaController,
    _build_agent,
    _CachedMOSAgentPool,
    _episode_quantiles,
)
from .counterfactuals import _infer_decision, _mos_imports, _step_cost


def _posterior_weighted_components(decision: Any, depth: int) -> tuple[float, float]:
    posterior = np.asarray(decision.policy_posterior, dtype=float)
    posterior /= posterior.sum()
    preference = float(np.dot(posterior, decision.policy_risk) / depth)
    epistemic = float(
        np.dot(
            posterior,
            decision.policy_ambiguity - decision.policy_information_gain,
        )
        / depth
    )
    return preference, epistemic


def _policy_key(policy: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(value) for value in row) for row in np.asarray(policy))


def _posterior_immediate_and_improvement(
    decisions: dict[int, Any],
    attribute: str,
) -> dict[int, tuple[float, float]]:
    """Recover posterior-weighted step values from cumulative policy values."""

    policy_indices = {
        depth: {
            _policy_key(policy): index for index, policy in enumerate(decisions[depth].policies)
        }
        for depth in DEPTHS
    }
    result: dict[int, tuple[float, float]] = {}
    for depth in DEPTHS:
        decision = decisions[depth]
        weights = np.asarray(decision.policy_posterior, dtype=float)
        weights /= weights.sum()
        step_values = np.empty((len(decision.policies), depth), dtype=float)
        for policy_index, policy in enumerate(decision.policies):
            key = _policy_key(policy)
            previous = 0.0
            for step in range(1, depth + 1):
                prefix_index = policy_indices[step][key[:step]]
                current = float(
                    np.asarray(getattr(decisions[step], attribute), dtype=float)[prefix_index]
                )
                step_values[policy_index, step - 1] = current - previous
                previous = current
        weighted_steps = weights @ step_values
        immediate = float(weighted_steps[0])
        future = immediate if depth == 1 else float(weighted_steps[1:].mean())
        result[depth] = (immediate, future - immediate)
    return result


def _exact_immediate_improvement_components(
    decisions: dict[Allocation, Any],
) -> dict[str, np.ndarray]:
    """Reproduce schema-16 labels at one realized online context."""

    components = {
        "state_accuracy": np.empty(len(ALLOCATIONS)),
        "state_complexity": np.empty(len(ALLOCATIONS)),
        "immediate_preference": np.empty(len(ALLOCATIONS)),
        "preference_improvement": np.empty(len(ALLOCATIONS)),
        "immediate_epistemic": np.empty(len(ALLOCATIONS)),
        "epistemic_improvement": np.empty(len(ALLOCATIONS)),
    }
    for resolution in RESOLUTIONS:
        by_depth = {depth: decisions[Allocation(resolution, depth)] for depth in DEPTHS}
        accuracy = float(np.mean([item.state_accuracy for item in by_depth.values()]))
        complexity = float(np.mean([item.state_complexity for item in by_depth.values()]))
        preference = _posterior_immediate_and_improvement(by_depth, "policy_risk")
        ambiguity = _posterior_immediate_and_improvement(by_depth, "policy_ambiguity")
        information_gain = _posterior_immediate_and_improvement(by_depth, "policy_information_gain")
        for depth in DEPTHS:
            index = ALLOCATION_INDEX[Allocation(resolution, depth)]
            components["state_accuracy"][index] = accuracy
            components["state_complexity"][index] = complexity
            components["immediate_preference"][index] = preference[depth][0]
            components["preference_improvement"][index] = preference[depth][1]
            components["immediate_epistemic"][index] = (
                ambiguity[depth][0] - information_gain[depth][0]
            )
            components["epistemic_improvement"][index] = (
                ambiguity[depth][1] - information_gain[depth][1]
            )
    components["immediate_preference"] -= components["immediate_preference"].mean()
    components["immediate_epistemic"] -= components["immediate_epistemic"].mean()
    return components


def _rank_descending(values: np.ndarray) -> np.ndarray:
    ranks = np.empty(values.size, dtype=int)
    ranks[np.argsort(-values, kind="stable")] = np.arange(1, values.size + 1)
    return ranks


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_ranking_diagnostic(
    instance_seed: int,
    *,
    config: ClosedLoopConfig,
    follow_exact: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Follow the neural controller and realize all candidate labels at each step."""

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
    pool = _CachedMOSAgentPool(
        initial_allocation=allocation,
        initial_agent=agent,
        layout=instance.layout,
        config=config,
        mos=mos,
    )
    controller = NeuralMetaController(
        Path(config.checkpoint),
        device=config.device,
        torch_threads=config.torch_threads,
    )
    decision = _infer_decision(agent, observation, None, 0, mos)
    rows: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    task_cost = 0.0
    success = False
    switches = 0

    for decision_index in range(config.max_steps):
        source = allocation
        source_agent = agent
        source_decision = decision
        position = environment.position
        next_position = instance.layout.move(position, source_decision.action)
        observation, success = environment.step(
            source_decision.action,
            noise_quantile=float(quantiles[decision_index + 1]),
        )
        neural_decision, _ = controller.choose(
            agent=source_agent,
            allocation=source,
            layout=instance.layout,
            selected_action=int(source_decision.action),
            next_robot_position=next_position,
            deadline_preference=config.deadline_preference,
            compute_preference=config.compute_preference,
            commit_for_depth=False,
            utility_preference_weight=config.utility_preference_weight,
            utility_epistemic_weight=config.utility_epistemic_weight,
            preference_improvement_weight=config.preference_improvement_weight,
            epistemic_improvement_weight=config.epistemic_improvement_weight,
            state_accuracy_weight=config.state_accuracy_weight,
            state_complexity_weight=config.state_complexity_weight,
            compute_cost_weight=config.compute_cost_weight,
            switching_cost_weight=config.switching_cost_weight,
            fixed_switch_cost_ms=config.fixed_switch_cost_ms,
            information_loss_weight=config.information_loss_weight,
            realized_observation=observation,
        )
        predicted = controller.last_component_predictions
        false_find = observation[3] == mos["FindOutcome"].FALSE_FIND
        task_cost += _step_cost(false_find=false_find, collision=environment.last_collision)

        if success:
            steps.append(
                {
                    "decision": decision_index,
                    "x": position[0],
                    "y": position[1],
                    "action": source_decision.action.name,
                    "success": True,
                }
            )
            break

        candidate_agents: dict[Any, Any] = {}
        candidate_decisions: dict[Any, Any] = {}
        for candidate in ALLOCATIONS:
            if candidate == source:
                candidate_agent = copy.deepcopy(source_agent)
            else:
                candidate_agent = pool.switch(
                    source_agent=source_agent,
                    source_allocation=source,
                    target_allocation=candidate,
                    layout=instance.layout,
                    executed_action=source_decision.action,
                    next_time_step=decision_index + 1,
                    mos=mos,
                )
            candidate_decision = _infer_decision(
                candidate_agent,
                observation,
                source_decision.action,
                decision_index + 1,
                mos,
            )
            candidate_agents[candidate] = candidate_agent
            candidate_decisions[candidate] = candidate_decision

        if controller.schema_version == 16:
            exact_components = _exact_immediate_improvement_components(candidate_decisions)
            exact_score = (
                config.state_accuracy_weight * exact_components["state_accuracy"]
                - config.state_complexity_weight * exact_components["state_complexity"]
                + config.utility_preference_weight * exact_components["immediate_preference"]
                + config.preference_improvement_weight * exact_components["preference_improvement"]
                + config.utility_epistemic_weight * exact_components["immediate_epistemic"]
                + config.epistemic_improvement_weight * exact_components["epistemic_improvement"]
            )
        else:
            exact_components = {
                "state_accuracy": np.empty(len(ALLOCATIONS)),
                "state_complexity": np.empty(len(ALLOCATIONS)),
                "preference_per_depth": np.empty(len(ALLOCATIONS)),
                "epistemic_value_per_depth": np.empty(len(ALLOCATIONS)),
            }
            for candidate in ALLOCATIONS:
                index = ALLOCATION_INDEX[candidate]
                candidate_decision = candidate_decisions[candidate]
                preference, epistemic = _posterior_weighted_components(
                    candidate_decision, candidate.depth
                )
                exact_components["state_accuracy"][index] = candidate_decision.state_accuracy
                exact_components["state_complexity"][index] = candidate_decision.state_complexity
                exact_components["preference_per_depth"][index] = preference
                exact_components["epistemic_value_per_depth"][index] = epistemic

        if controller.schema_version in {13, 14, 15}:
            # Schemas 13+ predict candidate-relative G components. Centering
            # realized exact values makes them directly comparable and leaves
            # their allocation ordering unchanged.
            exact_components["preference_per_depth"] -= exact_components[
                "preference_per_depth"
            ].mean()
            exact_components["epistemic_value_per_depth"] -= exact_components[
                "epistemic_value_per_depth"
            ].mean()
        if controller.schema_version != 16:
            exact_score = (
                config.state_accuracy_weight * exact_components["state_accuracy"]
                - config.state_complexity_weight * exact_components["state_complexity"]
                + config.utility_preference_weight * exact_components["preference_per_depth"]
                + config.utility_epistemic_weight * exact_components["epistemic_value_per_depth"]
            )
        predicted_score = np.asarray(predicted["task_score"], dtype=float)
        predicted_ranks = _rank_descending(predicted_score)
        exact_ranks = _rank_descending(exact_score)
        predicted_winner = neural_decision.allocation
        exact_winner = ALLOCATIONS[int(np.argmax(exact_score))]
        predicted_winner_index = ALLOCATION_INDEX[predicted_winner]
        exact_winner_index = ALLOCATION_INDEX[exact_winner]
        followed_winner = exact_winner if follow_exact else predicted_winner
        exact_regret = float(exact_score[exact_winner_index] - exact_score[predicted_winner_index])

        for candidate in ALLOCATIONS:
            index = ALLOCATION_INDEX[candidate]
            row = {
                "instance_seed": int(instance_seed),
                "decision": decision_index,
                "next_x": environment.position[0],
                "next_y": environment.position[1],
                "source_allocation": f"g{source.resolution}_T{source.depth}",
                "executed_action": source_decision.action.name,
                "candidate": f"g{candidate.resolution}_T{candidate.depth}",
                "candidate_action_t_plus_1": candidate_decisions[candidate].action.name,
                "predicted_accuracy": float(predicted["state_accuracy"][index]),
                "exact_accuracy": float(exact_components["state_accuracy"][index]),
                "predicted_complexity": float(predicted["state_complexity"][index]),
                "exact_complexity": float(exact_components["state_complexity"][index]),
                "predicted_preference_per_depth": float(predicted["preference_per_depth"][index]),
                "predicted_epistemic_per_depth": float(
                    predicted["epistemic_value_per_depth"][index]
                ),
                "predicted_task_score": float(predicted_score[index]),
                "exact_task_score": float(exact_score[index]),
                "score_error": float(predicted_score[index] - exact_score[index]),
                "predicted_rank": int(predicted_ranks[index]),
                "exact_rank": int(exact_ranks[index]),
                "neural_winner": candidate == predicted_winner,
                "exact_winner": candidate == exact_winner,
            }
            if controller.schema_version == 16:
                for name in (
                    "immediate_preference",
                    "preference_improvement",
                    "immediate_epistemic",
                    "epistemic_improvement",
                ):
                    row[f"predicted_{name}"] = float(predicted[name][index])
                    row[f"exact_{name}"] = float(exact_components[name][index])
            else:
                row["exact_preference_per_depth"] = float(
                    exact_components["preference_per_depth"][index]
                )
                row["exact_epistemic_per_depth"] = float(
                    exact_components["epistemic_value_per_depth"][index]
                )
            rows.append(row)

        steps.append(
            {
                "decision": decision_index,
                "x": position[0],
                "y": position[1],
                "action": source_decision.action.name,
                "neural_winner": f"g{predicted_winner.resolution}_T{predicted_winner.depth}",
                "exact_winner": f"g{exact_winner.resolution}_T{exact_winner.depth}",
                "followed_winner": (f"g{followed_winner.resolution}_T{followed_winner.depth}"),
                "winner_agreement": predicted_winner == exact_winner,
                "neural_winner_exact_rank": int(exact_ranks[predicted_winner_index]),
                "exact_winner_neural_rank": int(predicted_ranks[exact_winner_index]),
                "neural_candidate_action": candidate_decisions[predicted_winner].action.name,
                "exact_candidate_action": candidate_decisions[exact_winner].action.name,
                "candidate_action_agreement": (
                    candidate_decisions[predicted_winner].action
                    == candidate_decisions[exact_winner].action
                ),
                "exact_score_regret": exact_regret,
                "success": False,
            }
        )
        switches += int(followed_winner != source)
        allocation = followed_winner
        agent = candidate_agents[allocation]
        decision = candidate_decisions[allocation]

        if (decision_index + 1) % 5 == 0:
            print(
                f"[ranking diagnostic] instance={instance_seed} step={decision_index + 1} "
                f"neural=g{predicted_winner.resolution}_T{predicted_winner.depth} "
                f"exact=g{exact_winner.resolution}_T{exact_winner.depth}",
                flush=True,
            )

    if not success:
        task_cost += 2.0 * config.max_steps
    comparable = [row for row in steps if "winner_agreement" in row]
    first_disagreement = next(
        (int(row["decision"]) for row in comparable if not row["winner_agreement"]), None
    )
    summary = {
        "instance_seed": int(instance_seed),
        "followed_controller": "exact_objective" if follow_exact else "neural",
        "success": bool(success),
        "steps": len(steps),
        "task_cost": float(task_cost),
        "switches": switches,
        "compared_decisions": len(comparable),
        "winner_agreement_rate": float(np.mean([row["winner_agreement"] for row in comparable])),
        "candidate_action_agreement_rate": float(
            np.mean([row["candidate_action_agreement"] for row in comparable])
        ),
        "mean_exact_rank_of_neural_winner": float(
            np.mean([row["neural_winner_exact_rank"] for row in comparable])
        ),
        "mean_exact_score_regret": float(
            np.mean([row["exact_score_regret"] for row in comparable])
        ),
        "median_exact_score_regret": float(
            np.median([row["exact_score_regret"] for row in comparable])
        ),
        "first_winner_disagreement": first_disagreement,
    }
    return summary, steps, rows


def evaluate_ranking_diagnostic(
    instance_seed: int,
    *,
    output_dir: Path,
    config: ClosedLoopConfig,
    follow_exact: bool = False,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary, steps, candidates = run_ranking_diagnostic(
        instance_seed, config=config, follow_exact=follow_exact
    )
    _write_csv(output_dir / "step_comparison.csv", steps)
    _write_csv(output_dir / "candidate_comparison.csv", candidates)
    report = {"configuration": asdict(config), "result": summary}
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
