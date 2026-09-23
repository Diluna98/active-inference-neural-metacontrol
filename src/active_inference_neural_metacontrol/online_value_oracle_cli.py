"""Run the exact online preference-plus-epistemic oracle on one MOS instance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .online_value_oracle import (
    OnlineValueOracleConfig,
    evaluate_online_value_oracle,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance-seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--initial-resolution", type=int, choices=(2, 5, 10, 20), default=2)
    parser.add_argument("--initial-depth", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--message-passing-iterations", type=int, default=10)
    parser.add_argument("--policy-workers", type=int, default=1)
    args = parser.parse_args()
    report = evaluate_online_value_oracle(
        args.instance_seed,
        output_dir=args.output_dir,
        config=OnlineValueOracleConfig(
            initial_resolution=args.initial_resolution,
            initial_depth=args.initial_depth,
            max_steps=args.max_steps,
            message_passing_iterations=args.message_passing_iterations,
            policy_workers=args.policy_workers,
        ),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
