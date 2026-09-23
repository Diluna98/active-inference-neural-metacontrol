"""Visualize a matched MOS context where shallow and deep planning disagree."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from active_inference_navigation.mos import MOSAction, mos_action_controls, sample_mos_instance
from PyAIF.numerics import factor_dot, log_stable_probability

from active_inference_neural_metacontrol import Allocation
from active_inference_neural_metacontrol.beliefs import canonicalize_posterior
from active_inference_neural_metacontrol.closed_loop import _episode_quantiles
from active_inference_neural_metacontrol.counterfactuals import (
    _build_switched_agent,
    _infer_decision,
    _mos_imports,
)
from active_inference_neural_metacontrol.information import categorical_fisher_map
from active_inference_neural_metacontrol.mos_adapter import (
    canonical_detection_likelihood,
    obstacle_map,
    selected_policy_index,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance-seed", type=int, default=4008)
    parser.add_argument("--decision", type=int, default=6)
    parser.add_argument("--resolution", type=int, choices=(2, 5, 10, 20), default=5)
    parser.add_argument("--shallow-depth", type=int, default=1)
    parser.add_argument("--deep-depth", type=int, default=3)
    parser.add_argument("--source-resolution", type=int, default=10)
    parser.add_argument("--source-depth", type=int, default=3)
    parser.add_argument("--message-passing-iterations", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _build_agent(allocation: Allocation, layout, iterations: int, mos: dict):
    agent = mos["build_mos_agent"](
        mos["MOSAgentConfig"](
            target_resolution=allocation.resolution,
            action_depth=allocation.depth,
            message_passing_iterations=iterations,
            policy_workers=1,
        ),
        layout=layout,
    )
    agent.reset()
    return agent


def _policy_actions(agent) -> tuple[MOSAction, ...]:
    policy = np.asarray(agent.policies[selected_policy_index(agent)], dtype=int)
    result = []
    for controls in policy:
        matches = [
            action for action in MOSAction if np.array_equal(controls, mos_action_controls(action))
        ]
        if len(matches) != 1:
            raise RuntimeError(f"cannot map policy controls to a MOS action: {controls}")
        result.append(matches[0])
    return tuple(result)


def _path(layout, start: tuple[int, int], actions: tuple[MOSAction, ...]):
    positions = [start]
    for action in actions:
        positions.append(layout.move(positions[-1], action))
    return positions


def _selected_policy_step_terms(agent, decision) -> list[dict[str, float]]:
    """Reconstruct per-future-step categorical G terms for the winning policy."""

    policy_index = selected_policy_index(agent)
    trajectory = agent.policy_dep_posteriors[policy_index]
    terms = []
    for timestep in range(1, agent.temporal_horizon):
        factor_posteriors = list(trajectory[timestep])
        preference = 0.0
        epistemic_value = 0.0
        for modality, likelihood in enumerate(agent.A):
            dependencies = agent.mod_dep[modality]
            dependency_posteriors = [
                factor_posteriors[factor] for factor in dependencies
            ]
            expected_outcome = factor_dot(likelihood, dependency_posteriors)
            preference += float(expected_outcome.dot(agent.C[modality][:, timestep]))
            outcome_entropy = -float(
                expected_outcome.dot(log_stable_probability(expected_outcome))
            )
            likelihood_entropy = -np.sum(
                likelihood * log_stable_probability(likelihood), axis=0
            )
            expected_likelihood_entropy = likelihood_entropy
            for posterior in dependency_posteriors:
                expected_likelihood_entropy = np.tensordot(
                    expected_likelihood_entropy,
                    posterior,
                    axes=(0, 0),
                )
            epistemic_value += outcome_entropy - float(expected_likelihood_entropy)
        terms.append(
            {
                "step": timestep,
                "preference": preference,
                "epistemic_value": epistemic_value,
                "g": preference + epistemic_value,
            }
        )

    if not np.isclose(sum(item["preference"] for item in terms), decision.risk):
        raise RuntimeError("per-step preference terms do not recover selected-policy risk")
    if not np.isclose(sum(item["epistemic_value"] for item in terms), decision.ambiguity):
        raise RuntimeError("per-step epistemic terms do not recover selected-policy ambiguity")
    return terms


def _capture(args: argparse.Namespace):
    mos = _mos_imports()
    instance = sample_mos_instance(args.instance_seed)
    source = Allocation(args.source_resolution, args.source_depth)
    targets = (
        Allocation(args.resolution, args.shallow_depth),
        Allocation(args.resolution, args.deep_depth),
    )
    environment = mos["MOSEnvironment"](
        layout=instance.layout,
        target=instance.target,
        observation_seed=instance.observation_seed,
    )
    quantiles = _episode_quantiles(instance, args.decision + 2)
    observation = environment.reset(noise_quantile=float(quantiles[0]))
    agent = _build_agent(source, instance.layout, args.message_passing_iterations, mos)
    previous_action = None

    for decision in range(args.decision):
        reference = _infer_decision(agent, observation, previous_action, decision, mos)
        next_observation, success = environment.step(
            reference.action,
            noise_quantile=float(quantiles[decision + 1]),
        )
        if success:
            raise RuntimeError("reference trajectory terminated before the requested decision")
        next_decision = decision + 1
        if next_decision == args.decision:
            candidates = {}
            decisions = {}
            for allocation in targets:
                candidates[allocation] = _build_switched_agent(
                    source_agent=agent,
                    source_allocation=source,
                    target_allocation=allocation,
                    layout=instance.layout,
                    executed_action=reference.action,
                    next_time_step=next_decision,
                    message_passing_iterations=args.message_passing_iterations,
                    policy_workers=1,
                    mos=mos,
                )
                decisions[allocation] = _infer_decision(
                    candidates[allocation],
                    next_observation,
                    reference.action,
                    next_decision,
                    mos,
                )
            return instance, environment.position, candidates, decisions

        agent = _build_switched_agent(
            source_agent=agent,
            source_allocation=source,
            target_allocation=source,
            layout=instance.layout,
            executed_action=reference.action,
            next_time_step=next_decision,
            message_passing_iterations=args.message_passing_iterations,
            policy_workers=1,
            mos=mos,
        )
        observation = next_observation
        previous_action = reference.action
    raise RuntimeError("requested decision was not reached")


def _expected_fisher_landscape(layout, posterior: np.ndarray) -> np.ndarray:
    landscape = np.full((layout.size, layout.size), np.nan, dtype=float)
    for y in range(layout.size):
        for x in range(layout.size):
            if layout.is_free((x, y)):
                fisher = categorical_fisher_map(canonical_detection_likelihood(layout, (x, y)))
                landscape[y, x] = float(np.sum(posterior * fisher))
    return landscape


def _draw_map(ax, layout, *, background, title: str, cmap: str):
    image = ax.imshow(background, origin="lower", cmap=cmap)
    obstacles = np.ma.masked_where(obstacle_map(layout) == 0, obstacle_map(layout))
    ax.imshow(obstacles, origin="lower", cmap="gray_r", vmin=0, vmax=1, alpha=0.85)
    ax.set_title(title)
    ax.set_xlim(-0.5, layout.size - 0.5)
    ax.set_ylim(-0.5, layout.size - 0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return image


def main() -> None:
    args = _parse_args()
    instance, robot, agents, decisions = _capture(args)
    allocations = tuple(agents)
    shallow, deep = allocations
    posterior = canonicalize_posterior(
        np.asarray(agents[deep].filtered_posteriors[2], dtype=float),
        deep.resolution,
        instance.layout.size,
    )
    information = _expected_fisher_landscape(instance.layout, posterior)
    policies = {allocation: _policy_actions(agents[allocation]) for allocation in allocations}
    paths = {
        allocation: _path(instance.layout, robot, policies[allocation])
        for allocation in allocations
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    posterior_image = _draw_map(
        axes[0],
        instance.layout,
        background=posterior,
        title="Target posterior at decision context",
        cmap="viridis",
    )
    fig.colorbar(posterior_image, ax=axes[0], fraction=0.046, label="Probability mass")

    info_image = _draw_map(
        axes[1],
        instance.layout,
        background=information,
        title="Expected Fisher information by robot position",
        cmap="magma",
    )
    fig.colorbar(info_image, ax=axes[1], fraction=0.046, label="Posterior-weighted Fisher")

    colors = {shallow: "#2f6fdd", deep: "#d64242"}
    for ax in axes[:2]:
        ax.scatter(*robot, marker="o", s=80, c="white", edgecolors="black", label="robot")
        ax.scatter(
            *instance.target,
            marker="*",
            s=150,
            c="#f5d442",
            edgecolors="black",
            label="true target",
        )
        for allocation in allocations:
            path = np.asarray(paths[allocation])
            label = f"T={allocation.depth}: " + "→".join(
                action.name for action in policies[allocation]
            )
            ax.plot(
                path[:, 0],
                path[:, 1],
                marker="o",
                linewidth=2.2,
                color=colors[allocation],
                label=label,
            )
    axes[1].legend(loc="upper left", fontsize=8)

    preference = [decisions[value].risk / value.depth for value in allocations]
    ambiguity = [decisions[value].ambiguity / value.depth for value in allocations]
    total = [
        (decisions[value].risk + decisions[value].ambiguity) / value.depth for value in allocations
    ]
    x = np.arange(3)
    width = 0.34
    axes[2].bar(x - width / 2, [preference[0], ambiguity[0], total[0]], width, label="T=1")
    axes[2].bar(x + width / 2, [preference[1], ambiguity[1], total[1]], width, label="T=3")
    axes[2].set_xticks(x, ("Preference/T", "Ambiguity/T", "G/T"))
    axes[2].set_ylabel("PyAIF value per planned step")
    axes[2].set_title("Selected-policy decomposition")
    axes[2].legend()
    axes[2].axhline(0, color="black", linewidth=0.8)

    fig.suptitle(
        f"MOS depth disagreement: seed {args.instance_seed}, decision {args.decision}, "
        f"gamma={args.resolution}, robot={robot}",
        fontsize=13,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)

    summary = {
        "instance_seed": args.instance_seed,
        "decision": args.decision,
        "resolution": args.resolution,
        "robot": list(robot),
        "true_target": list(instance.target),
        "candidates": {
            f"T{allocation.depth}": {
                "policy": [action.name for action in policies[allocation]],
                "path": [list(position) for position in paths[allocation]],
                "raw_g": decisions[allocation].selected_policy_g,
                "preference_per_step": decisions[allocation].risk / allocation.depth,
                "ambiguity_per_step": decisions[allocation].ambiguity / allocation.depth,
                "g_per_step": (decisions[allocation].risk + decisions[allocation].ambiguity)
                / allocation.depth,
                "step_terms": _selected_policy_step_terms(
                    agents[allocation], decisions[allocation]
                ),
            }
            for allocation in allocations
        },
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
