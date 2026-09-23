"""Evaluate exact one-step and accumulated-G oracles over complete MOS episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .rollout_diagnostic import GOracleConfig, evaluate_g_oracles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--rollout-horizon", type=int, default=5)
    parser.add_argument("--discount", type=float, default=1.0)
    parser.add_argument("--message-passing-iterations", type=int, default=10)
    parser.add_argument("--policy-workers", type=int, default=1)
    parser.add_argument("--instance-workers", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate_g_oracles(
        instance_seeds=args.instance_seeds,
        output_dir=args.output_dir,
        instance_workers=args.instance_workers,
        config=GOracleConfig(
            max_steps=args.max_steps,
            rollout_horizon=args.rollout_horizon,
            discount=args.discount,
            message_passing_iterations=args.message_passing_iterations,
            policy_workers=args.policy_workers,
        ),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
