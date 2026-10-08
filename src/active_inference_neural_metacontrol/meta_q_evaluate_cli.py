"""Evaluate a fitted-Q metacontroller in the MOS environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .allocations import Allocation
from .meta_q_runtime import (
    MetaQEvaluationConfig,
    evaluate_meta_q_benchmark,
    run_meta_q_episode,
    save_meta_q_evaluation,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--ensemble-checkpoints",
        type=Path,
        nargs="*",
        default=(),
        help="additional independently trained checkpoints for uncertainty estimation",
    )
    seeds = parser.add_mutually_exclusive_group(required=True)
    seeds.add_argument("--instance-seed", type=int)
    seeds.add_argument("--instance-seeds", type=int, nargs="+")
    parser.add_argument("--initial-resolution", type=int, default=2)
    parser.add_argument("--initial-depth", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--message-passing-iterations", type=int, default=10)
    parser.add_argument("--policy-workers", type=int, default=1)
    parser.add_argument(
        "--task-tolerance",
        type=float,
        default=1.0,
        help="maximum predicted task-return loss allowed before a candidate is insufficient",
    )
    parser.add_argument("--compute-weight", type=float, default=1.0)
    parser.add_argument("--switch-cost-ms", type=float, default=0.0)
    parser.add_argument("--switching-weight", type=float, default=1.0)
    parser.add_argument(
        "--switch-penalty-mode",
        choices=("allocation", "resolution", "none"),
        default="allocation",
        help="apply the fixed penalty to any allocation change, resolution changes only, or none",
    )
    parser.add_argument("--selection-mode", choices=("tolerance", "joint"), default="tolerance")
    parser.add_argument(
        "--compute-price",
        type=float,
        default=0.0,
        help="task-cost units charged per millisecond in joint selection mode",
    )
    parser.add_argument(
        "--uncertainty-beta",
        type=float,
        default=0.0,
        help="lower-confidence-bound penalty in ensemble standard deviations",
    )
    parser.add_argument(
        "--information-loss-weight",
        type=float,
        default=0.0,
        help="task-cost units charged per unit normalized Jensen-Shannon loss",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--instance-workers", type=int, default=1)
    parser.add_argument("--include-fixed", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    config = MetaQEvaluationConfig(
        checkpoint=args.checkpoint,
        ensemble_checkpoints=tuple(args.ensemble_checkpoints),
        initial_allocation=Allocation(args.initial_resolution, args.initial_depth),
        max_steps=args.max_steps,
        message_passing_iterations=args.message_passing_iterations,
        policy_workers=args.policy_workers,
        task_tolerance=args.task_tolerance,
        compute_weight=args.compute_weight,
        switch_cost_ms=args.switch_cost_ms,
        switching_weight=args.switching_weight,
        switch_penalty_mode=args.switch_penalty_mode,
        selection_mode=args.selection_mode,
        compute_price=args.compute_price,
        uncertainty_beta=args.uncertainty_beta,
        information_loss_weight=args.information_loss_weight,
        device=args.device,
        torch_threads=args.torch_threads,
    )
    if args.instance_seeds is not None:
        report = evaluate_meta_q_benchmark(
            instance_seeds=args.instance_seeds,
            output_dir=args.output_dir,
            config=config,
            include_fixed=args.include_fixed,
            instance_workers=args.instance_workers,
            resume=args.resume,
        )
        print(json.dumps(report, indent=2))
    else:
        summary, trajectory = run_meta_q_episode(args.instance_seed, config)
        save_meta_q_evaluation(summary, trajectory, args.output_dir)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
