"""Print raw policy-G spread statistics at the first matched MOS transition."""

from __future__ import annotations

import argparse

import numpy as np

from active_inference_neural_metacontrol.allocations import ALLOCATIONS, Allocation
from active_inference_neural_metacontrol.closed_loop import (
    ClosedLoopConfig,
    _build_agent,
    _episode_quantiles,
)
from active_inference_neural_metacontrol.counterfactuals import (
    _build_switched_agent,
    _infer_decision,
    _mos_imports,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance-seed", type=int, default=6000)
    parser.add_argument("--message-passing-iterations", type=int, default=10)
    args = parser.parse_args()

    mos = _mos_imports()
    instance = mos["sample_mos_instance"](args.instance_seed)
    config = ClosedLoopConfig(
        checkpoint="unused",
        deadline_median_ms=250.0,
        initial_resolution=2,
        initial_depth=1,
        max_steps=50,
        message_passing_iterations=args.message_passing_iterations,
        policy_workers=1,
        include_fixed=False,
    )
    environment = mos["MOSEnvironment"](
        layout=instance.layout,
        target=instance.target,
        observation_seed=instance.observation_seed,
    )
    quantiles = _episode_quantiles(instance, config.max_steps)
    observation = environment.reset(noise_quantile=float(quantiles[0]))
    source = Allocation(2, 1)
    source_agent = _build_agent(source, instance.layout, config, mos)
    source_decision = _infer_decision(source_agent, observation, None, 0, mos)
    next_observation, _ = environment.step(
        source_decision.action,
        noise_quantile=float(quantiles[1]),
    )
    print(
        f"matched point: action={source_decision.action.name}, "
        f"next_position={environment.position}, observation={next_observation}"
    )
    header = (
        "configuration policies best_G mean_G lowest_G best_minus_mean "
        "mean_minus_lowest best_minus_mean_per_T first_action_mean_gap_per_T next_action"
    )
    print(header)
    for candidate in ALLOCATIONS:
        candidate_agent = _build_switched_agent(
            source_agent=source_agent,
            source_allocation=source,
            target_allocation=candidate,
            layout=instance.layout,
            executed_action=source_decision.action,
            next_time_step=1,
            message_passing_iterations=config.message_passing_iterations,
            policy_workers=config.policy_workers,
            mos=mos,
        )
        decision = _infer_decision(
            candidate_agent,
            next_observation,
            source_decision.action,
            1,
            mos,
        )
        policy_g = np.asarray(decision.policy_g, dtype=float)
        first_controls = decision.policies[:, 0, :]
        action_means = np.asarray(
            [
                policy_g[np.all(first_controls == controls, axis=1)].mean()
                for controls in np.unique(first_controls, axis=0)
            ]
        )
        best = float(policy_g.max())
        mean = float(policy_g.mean())
        lowest = float(policy_g.min())
        print(
            f"g{candidate.resolution}_T{candidate.depth:<1} "
            f"{policy_g.size:>8d} "
            f"{best:>9.6f} {mean:>9.6f} {lowest:>9.6f} "
            f"{best - mean:>15.6f} {mean - lowest:>17.6f} "
            f"{(best - mean) / candidate.depth:>21.6f} "
            f"{(action_means.max() - action_means.mean()) / candidate.depth:>27.6f} "
            f"{decision.action.name}"
        )


if __name__ == "__main__":
    main()
