"""Run closed-loop adaptive MOS evaluation against fixed allocations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .closed_loop import ClosedLoopConfig, evaluate_closed_loop


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--instance-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--initial-resolution", type=int, choices=(2, 5, 10, 20), default=5)
    parser.add_argument("--initial-depth", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--message-passing-iterations", type=int, default=10)
    parser.add_argument("--policy-workers", type=int, default=1)
    parser.add_argument("--instance-workers", type=int, default=1)
    parser.add_argument("--success-threshold", type=float, default=0.5)
    parser.add_argument("--compute-budget-ms", type=float, default=100.0)
    parser.add_argument("--information-loss-limit", type=float, default=0.15)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--adaptive-only", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate_closed_loop(
        instance_seeds=args.instance_seeds,
        output_dir=args.output_dir,
        instance_workers=args.instance_workers,
        resume=args.resume,
        config=ClosedLoopConfig(
            checkpoint=str(args.checkpoint),
            initial_resolution=args.initial_resolution,
            initial_depth=args.initial_depth,
            max_steps=args.max_steps,
            message_passing_iterations=args.message_passing_iterations,
            policy_workers=args.policy_workers,
            success_threshold=args.success_threshold,
            compute_budget_ms=args.compute_budget_ms,
            information_loss_limit=args.information_loss_limit,
            device=args.device,
            include_fixed=not args.adaptive_only,
        ),
    )
    print(json.dumps(report, indent=2))
