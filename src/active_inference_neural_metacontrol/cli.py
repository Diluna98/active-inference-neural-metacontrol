"""Command-line entry points for dataset generation."""

from __future__ import annotations

import argparse
from pathlib import Path

from .generation import (
    GenerationConfig,
    generate_balanced_mos_counterfactuals,
    generate_resumable_mos_counterfactuals,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate matched-state MOS training data for neural metacontrol."
    )
    parser.add_argument("--instance-seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--reference-resolution", type=int, choices=(2, 5, 10, 20), default=20)
    parser.add_argument("--reference-depth", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--branch-stride", type=int, default=1)
    parser.add_argument("--message-passing-iterations", type=int, default=10)
    parser.add_argument("--policy-workers", type=int, default=1)
    parser.add_argument("--instance-workers", type=int, default=1)
    parser.add_argument(
        "--rollout-horizon",
        type=int,
        default=1,
        help="candidate-controlled steps used for the training target (default: 1)",
    )
    parser.add_argument(
        "--rollout-discount",
        type=float,
        default=1.0,
        help="discount for accumulated normalized G values (default: 1.0)",
    )
    parser.add_argument(
        "--collect-task-outcomes",
        action="store_true",
        help="also label eventual success and remaining task cost after one candidate action",
    )
    parser.add_argument(
        "--collect-research-archive",
        action="store_true",
        help=(
            "preserve full policy distributions, decompositions, candidate beliefs, "
            "predicted observations, and timing components for offline relabeling"
        ),
    )
    parser.add_argument(
        "--balanced-sources",
        action="store_true",
        help="assign the 12 source (gamma, T) allocations round-robin across instances",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="reuse completed per-instance shards (default: enabled)",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/mos_counterfactuals"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generator = (
        generate_balanced_mos_counterfactuals
        if args.balanced_sources
        else generate_resumable_mos_counterfactuals
    )
    dataset = generator(
        instance_seeds=args.instance_seeds,
        output_dir=args.output_dir,
        config=GenerationConfig(
            reference_resolution=args.reference_resolution,
            reference_depth=args.reference_depth,
            max_steps=args.max_steps,
            branch_stride=args.branch_stride,
            message_passing_iterations=args.message_passing_iterations,
            policy_workers=args.policy_workers,
            rollout_horizon=args.rollout_horizon,
            rollout_discount=args.rollout_discount,
            collect_task_outcomes=args.collect_task_outcomes,
            collect_research_archive=args.collect_research_archive,
        ),
        instance_workers=args.instance_workers,
        resume=args.resume,
    )
    print(
        f"wrote {len(dataset.context_ids)} contexts and {len(dataset.branches)} "
        f"candidate policy diagnostics to {args.output_dir}"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
