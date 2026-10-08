"""Evaluate interpretable adaptive metacontrol baselines in MOS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .allocations import Allocation
from .heuristic_runtime import (
    HeuristicEvaluationConfig,
    evaluate_heuristic_benchmark,
    run_heuristic_episode,
    save_heuristic_evaluation,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    seeds = parser.add_mutually_exclusive_group(required=True)
    seeds.add_argument("--instance-seed", type=int)
    seeds.add_argument("--instance-seeds", type=int, nargs="+")
    parser.add_argument("--mode", choices=("entropy", "fisher_surprise"), required=True)
    parser.add_argument("--initial-resolution", type=int, default=2)
    parser.add_argument("--initial-depth", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--message-passing-iterations", type=int, default=10)
    parser.add_argument("--policy-workers", type=int, default=1)
    parser.add_argument(
        "--resolution-thresholds",
        type=float,
        nargs=3,
        metavar=("G5", "G10", "G20"),
        default=None,
        help="ascending thresholds for g5/g10/g20; defaults depend on --mode",
    )
    parser.add_argument(
        "--depth-thresholds",
        type=float,
        nargs=2,
        metavar=("T2", "T3"),
        default=None,
        help="ascending thresholds for T2/T3; defaults depend on --mode",
    )
    parser.add_argument("--fisher-weight", type=float, default=0.5)
    parser.add_argument("--prediction-error-weight", type=float, default=0.5)
    parser.add_argument("--information-loss-limit", type=float, default=1.0)
    parser.add_argument("--minimum-hold-steps", type=int, default=0)
    parser.add_argument("--instance-workers", type=int, default=1)
    parser.add_argument("--include-fixed", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    default_resolution = (0.05, 0.15, 0.35) if args.mode == "entropy" else (0.01, 0.03, 0.10)
    default_depth = (0.75, 0.90) if args.mode == "entropy" else (0.60, 0.85)
    config = HeuristicEvaluationConfig(
        mode=args.mode,
        initial_allocation=Allocation(args.initial_resolution, args.initial_depth),
        max_steps=args.max_steps,
        message_passing_iterations=args.message_passing_iterations,
        policy_workers=args.policy_workers,
        resolution_thresholds=tuple(args.resolution_thresholds or default_resolution),
        depth_thresholds=tuple(args.depth_thresholds or default_depth),
        fisher_weight=args.fisher_weight,
        prediction_error_weight=args.prediction_error_weight,
        information_loss_limit=args.information_loss_limit,
        minimum_hold_steps=args.minimum_hold_steps,
    )
    if args.instance_seeds is not None:
        report = evaluate_heuristic_benchmark(
            instance_seeds=args.instance_seeds,
            output_dir=args.output_dir,
            config=config,
            include_fixed=args.include_fixed,
            instance_workers=args.instance_workers,
            resume=args.resume,
        )
        print(json.dumps(report, indent=2))
    else:
        summary, trajectory = run_heuristic_episode(args.instance_seed, config)
        save_heuristic_evaluation(summary, trajectory, args.output_dir)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
