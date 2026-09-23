"""Compare neural and exact PyAIF allocation rankings along one adaptive episode."""

from __future__ import annotations

import copy
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .allocations import ALLOCATION_INDEX, ALLOCATIONS
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
        neural_decision, _ = controller.choose(
            agent=source_agent,
            allocation=source,
            layout=instance.layout,
            selected_action=int(source_decision.action),
            next_robot_position=next_position,
            deadline_preference=config.deadline_preference,
            compute_preference=config.compute_preference,
            commit_for_depth=False,
            utility_preference_weight=1.0,
            utility_epistemic_weight=1.0,
            state_accuracy_weight=1.0,
            state_complexity_weight=1.0,
            compute_cost_weight=0.0,
            switching_cost_weight=0.0,
            information_loss_weight=0.0,
        )
        predicted = controller.last_component_predictions
        observation, success = environment.step(
            source_decision.action,
            noise_quantile=float(quantiles[decision_index + 1]),
        )
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
        exact_components = {
            "state_accuracy": np.empty(len(ALLOCATIONS)),
            "state_complexity": np.empty(len(ALLOCATIONS)),
            "preference_per_depth": np.empty(len(ALLOCATIONS)),
            "epistemic_value_per_depth": np.empty(len(ALLOCATIONS)),
        }
        for candidate in ALLOCATIONS:
            index = ALLOCATION_INDEX[candidate]
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
            preference, epistemic = _posterior_weighted_components(
                candidate_decision, candidate.depth
            )
            candidate_agents[candidate] = candidate_agent
            candidate_decisions[candidate] = candidate_decision
            exact_components["state_accuracy"][index] = candidate_decision.state_accuracy
            exact_components["state_complexity"][index] = candidate_decision.state_complexity
            exact_components["preference_per_depth"][index] = preference
            exact_components["epistemic_value_per_depth"][index] = epistemic

        if controller.schema_version >= 13:
            # Schemas 13+ predict candidate-relative G components. Centering
            # realized exact values makes them directly comparable and leaves
            # their allocation ordering unchanged.
            exact_components["preference_per_depth"] -= exact_components[
                "preference_per_depth"
            ].mean()
            exact_components["epistemic_value_per_depth"] -= exact_components[
                "epistemic_value_per_depth"
            ].mean()
        exact_score = (
            exact_components["state_accuracy"]
            - exact_components["state_complexity"]
            + exact_components["preference_per_depth"]
            + exact_components["epistemic_value_per_depth"]
        )
        predicted_score = np.asarray(predicted["task_score"], dtype=float)
        predicted_ranks = _rank_descending(predicted_score)
        exact_ranks = _rank_descending(exact_score)
        predicted_winner = neural_decision.allocation
        exact_winner = ALLOCATIONS[int(np.argmax(exact_score))]
        predicted_winner_index = ALLOCATION_INDEX[predicted_winner]
        exact_winner_index = ALLOCATION_INDEX[exact_winner]

        for candidate in ALLOCATIONS:
            index = ALLOCATION_INDEX[candidate]
            rows.append(
                {
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
                    "predicted_preference_per_depth": float(
                        predicted["preference_per_depth"][index]
                    ),
                    "exact_preference_per_depth": float(
                        exact_components["preference_per_depth"][index]
                    ),
                    "predicted_epistemic_per_depth": float(
                        predicted["epistemic_value_per_depth"][index]
                    ),
                    "exact_epistemic_per_depth": float(
                        exact_components["epistemic_value_per_depth"][index]
                    ),
                    "predicted_task_score": float(predicted_score[index]),
                    "exact_task_score": float(exact_score[index]),
                    "score_error": float(predicted_score[index] - exact_score[index]),
                    "predicted_rank": int(predicted_ranks[index]),
                    "exact_rank": int(exact_ranks[index]),
                    "neural_winner": candidate == predicted_winner,
                    "exact_winner": candidate == exact_winner,
                }
            )

        steps.append(
            {
                "decision": decision_index,
                "x": position[0],
                "y": position[1],
                "action": source_decision.action.name,
                "neural_winner": f"g{predicted_winner.resolution}_T{predicted_winner.depth}",
                "exact_winner": f"g{exact_winner.resolution}_T{exact_winner.depth}",
                "winner_agreement": predicted_winner == exact_winner,
                "neural_winner_exact_rank": int(exact_ranks[predicted_winner_index]),
                "exact_winner_neural_rank": int(predicted_ranks[exact_winner_index]),
                "neural_candidate_action": candidate_decisions[predicted_winner].action.name,
                "exact_candidate_action": candidate_decisions[exact_winner].action.name,
                "candidate_action_agreement": (
                    candidate_decisions[predicted_winner].action
                    == candidate_decisions[exact_winner].action
                ),
                "success": False,
            }
        )
        switches += int(predicted_winner != source)
        allocation = predicted_winner
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
        "success": bool(success),
        "steps": len(steps),
        "task_cost": float(task_cost),
        "switches": switches,
        "compared_decisions": len(comparable),
        "winner_agreement_rate": float(
            np.mean([row["winner_agreement"] for row in comparable])
        ),
        "candidate_action_agreement_rate": float(
            np.mean([row["candidate_action_agreement"] for row in comparable])
        ),
        "mean_exact_rank_of_neural_winner": float(
            np.mean([row["neural_winner_exact_rank"] for row in comparable])
        ),
        "first_winner_disagreement": first_disagreement,
    }
    return summary, steps, rows


def evaluate_ranking_diagnostic(
    instance_seed: int,
    *,
    output_dir: Path,
    config: ClosedLoopConfig,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary, steps, candidates = run_ranking_diagnostic(instance_seed, config=config)
    _write_csv(output_dir / "step_comparison.csv", steps)
    _write_csv(output_dir / "candidate_comparison.csv", candidates)
    report = {"configuration": asdict(config), "result": summary}
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report
