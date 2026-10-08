"""Compare neural and exact PyAIF allocation rankings for one MOS instance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .closed_loop import ClosedLoopConfig
from .ranking_diagnostic import evaluate_ranking_diagnostic


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--instance-seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--initial-resolution", type=int, default=2)
    parser.add_argument("--initial-depth", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--message-passing-iterations", type=int, default=10)
    parser.add_argument("--policy-workers", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--utility-preference-weight", type=float, default=1.0)
    parser.add_argument("--utility-epistemic-weight", type=float, default=1.0)
    parser.add_argument("--preference-improvement-weight", type=float, default=1.0)
    parser.add_argument("--epistemic-improvement-weight", type=float, default=1.0)
    parser.add_argument("--state-accuracy-weight", type=float, default=1.0)
    parser.add_argument("--state-complexity-weight", type=float, default=1.0)
    parser.add_argument(
        "--follow-exact",
        action="store_true",
        help="follow the realized exact objective winner instead of the neural winner",
    )
    args = parser.parse_args()
    config = ClosedLoopConfig(
        checkpoint=str(args.checkpoint),
        deadline_median_ms=250.0,
        initial_resolution=args.initial_resolution,
        initial_depth=args.initial_depth,
        max_steps=args.max_steps,
        message_passing_iterations=args.message_passing_iterations,
        policy_workers=args.policy_workers,
        device=args.device,
        include_fixed=False,
        utility_preference_weight=args.utility_preference_weight,
        utility_epistemic_weight=args.utility_epistemic_weight,
        preference_improvement_weight=args.preference_improvement_weight,
        epistemic_improvement_weight=args.epistemic_improvement_weight,
        state_accuracy_weight=args.state_accuracy_weight,
        state_complexity_weight=args.state_complexity_weight,
        compute_cost_weight=0.0,
        switching_cost_weight=0.0,
        information_loss_weight=0.0,
    )
    report = evaluate_ranking_diagnostic(
        args.instance_seed,
        output_dir=args.output_dir,
        config=config,
        follow_exact=args.follow_exact,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
