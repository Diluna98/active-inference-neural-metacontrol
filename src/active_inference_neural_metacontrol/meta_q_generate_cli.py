"""Generate candidate-dependent MOS transitions for meta-Q learning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .allocations import ALLOCATIONS, Allocation
from .meta_q_data import generate_resumable_meta_q_transitions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--message-passing-iterations", type=int, default=10)
    parser.add_argument("--policy-workers", type=int, default=1)
    parser.add_argument("--instance-workers", type=int, default=1)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--source-mode",
        choices=("balanced", "single"),
        default="balanced",
        help="round-robin all allocations or follow one fixed source allocation",
    )
    parser.add_argument("--source-resolution", type=int, default=2)
    parser.add_argument("--source-depth", type=int, default=1)
    parser.add_argument("--success-reward", type=float, default=0.0)
    parser.add_argument("--failure-penalty", type=float)
    parser.add_argument(
        "--exploration-switch-probability",
        type=float,
        default=0.25,
        help="per-step probability of following a different candidate allocation",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    sources = (
        ALLOCATIONS
        if args.source_mode == "balanced"
        else (Allocation(args.source_resolution, args.source_depth),)
    )
    generate_resumable_meta_q_transitions(
        instance_seeds=args.instance_seeds,
        output_dir=args.output_dir,
        source_allocations=sources,
        max_steps=args.max_steps,
        message_passing_iterations=args.message_passing_iterations,
        policy_workers=args.policy_workers,
        success_reward=args.success_reward,
        failure_penalty=args.failure_penalty,
        exploration_switch_probability=args.exploration_switch_probability,
        instance_workers=args.instance_workers,
        resume=args.resume,
    )
    summary = json.loads((args.output_dir / "summary.json").read_text(encoding="utf-8"))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
