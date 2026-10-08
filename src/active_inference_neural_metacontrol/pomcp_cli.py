"""Evaluate fixed-resolution POMCP in the matched MOS benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pomcp import POMCPConfig, evaluate_pomcp, run_pomcp_episode, save_pomcp_episode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    seeds = parser.add_mutually_exclusive_group(required=True)
    seeds.add_argument("--instance-seed", type=int)
    seeds.add_argument("--instance-seeds", type=int, nargs="+")
    parser.add_argument("--simulations", type=int, default=500)
    parser.add_argument("--planning-budget-ms", type=float)
    parser.add_argument("--search-depth", type=int, default=10)
    parser.add_argument("--exploration-constant", type=float, default=2.0**0.5)
    parser.add_argument("--discount", type=float, default=0.98)
    parser.add_argument("--rollout-random-probability", type=float, default=0.10)
    parser.add_argument(
        "--reward-mode",
        choices=("external", "preference"),
        default="external",
        help="optimize evaluation cost or PyAIF-style outcome preferences",
    )
    parser.add_argument("--step-penalty", type=float, default=1.0)
    parser.add_argument("--found-reward", type=float, default=9.0)
    parser.add_argument("--false-find-penalty", type=float, default=12.0)
    parser.add_argument("--collision-penalty", type=float, default=12.0)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--planner-seed-offset", type=int, default=1_000_003)
    parser.add_argument("--instance-workers", type=int, default=1)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    config = POMCPConfig(
        simulations=args.simulations,
        planning_budget_ms=args.planning_budget_ms,
        search_depth=args.search_depth,
        exploration_constant=args.exploration_constant,
        discount=args.discount,
        rollout_random_probability=args.rollout_random_probability,
        reward_mode=args.reward_mode,
        step_penalty=args.step_penalty,
        found_reward=args.found_reward,
        false_find_penalty=args.false_find_penalty,
        collision_penalty=args.collision_penalty,
        max_steps=args.max_steps,
        planner_seed_offset=args.planner_seed_offset,
    )
    if args.instance_seeds is not None:
        report = evaluate_pomcp(
            instance_seeds=args.instance_seeds,
            output_dir=args.output_dir,
            config=config,
            instance_workers=args.instance_workers,
            resume=args.resume,
        )
        print(json.dumps(report, indent=2))
    else:
        summary, trajectory = run_pomcp_episode(args.instance_seed, config)
        save_pomcp_episode(summary, trajectory, args.output_dir)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
