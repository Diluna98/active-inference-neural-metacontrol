"""Compare one-step and accumulated-G counterfactual oracles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .allocations import Allocation
from .rollout_diagnostic import diagnose_accumulated_g


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-resolution", type=int, choices=(2, 5, 10, 20), default=20)
    parser.add_argument("--reference-depth", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--max-reference-steps", type=int, default=20)
    parser.add_argument("--branch-stride", type=int, default=5)
    parser.add_argument("--rollout-horizon", type=int, default=5)
    parser.add_argument("--discount", type=float, default=1.0)
    parser.add_argument("--message-passing-iterations", type=int, default=10)
    parser.add_argument("--policy-workers", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = diagnose_accumulated_g(
        instance_seeds=args.instance_seeds,
        output_dir=args.output_dir,
        reference_allocation=Allocation(args.reference_resolution, args.reference_depth),
        max_reference_steps=args.max_reference_steps,
        branch_stride=args.branch_stride,
        rollout_horizon=args.rollout_horizon,
        discount=args.discount,
        message_passing_iterations=args.message_passing_iterations,
        policy_workers=args.policy_workers,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":  # pragma: no cover - module execution convenience
    main()
