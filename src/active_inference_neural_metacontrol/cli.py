"""Command-line entry points for dataset generation."""

from __future__ import annotations

import argparse
from pathlib import Path

from .allocations import Allocation
from .counterfactuals import generate_mos_counterfactuals, save_counterfactual_dataset


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
    parser.add_argument("--output-dir", type=Path, default=Path("results/mos_counterfactuals"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = generate_mos_counterfactuals(
        instance_seeds=args.instance_seeds,
        reference_allocation=Allocation(args.reference_resolution, args.reference_depth),
        max_steps=args.max_steps,
        branch_stride=args.branch_stride,
        message_passing_iterations=args.message_passing_iterations,
        policy_workers=args.policy_workers,
    )
    save_counterfactual_dataset(dataset, args.output_dir)
    print(
        f"wrote {len(dataset.context_ids)} contexts and {len(dataset.branches)} "
        f"unique action branches to {args.output_dir}"
    )
