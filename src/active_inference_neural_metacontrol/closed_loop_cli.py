"""Run closed-loop adaptive MOS evaluation against fixed allocations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .closed_loop import ClosedLoopConfig, evaluate_closed_loop


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--timing-profile",
        type=Path,
        default=None,
        help="override checkpoint latency estimates with a controlled timing profile",
    )
    parser.add_argument(
        "--deadline-median-ms",
        type=float,
        required=True,
        help="reference latency budget used to normalize inference milliseconds",
    )
    parser.add_argument(
        "--deadline-log-sigma",
        type=float,
        default=0.5,
        help="log-space deadline uncertainty retained for diagnostic reporting",
    )
    parser.add_argument("--instance-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--initial-resolution", type=int, choices=(2, 5, 10, 20), default=5)
    parser.add_argument("--initial-depth", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--message-passing-iterations", type=int, default=10)
    parser.add_argument("--policy-workers", type=int, default=1)
    parser.add_argument("--instance-workers", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--utility-success-weight", type=float, default=0.5)
    parser.add_argument("--operational-cost-weight", type=float, default=1.0)
    parser.add_argument("--operational-cost-temperature", type=float, default=5.0)
    parser.add_argument("--utility-g-weight", type=float, default=1.0)
    parser.add_argument("--utility-preference-weight", type=float, default=1.0)
    parser.add_argument("--utility-epistemic-weight", type=float, default=1.0)
    parser.add_argument("--preference-improvement-weight", type=float, default=1.0)
    parser.add_argument("--epistemic-improvement-weight", type=float, default=1.0)
    parser.add_argument("--state-accuracy-weight", type=float, default=1.0)
    parser.add_argument("--state-complexity-weight", type=float, default=1.0)
    parser.add_argument("--compute-cost-weight", type=float, default=1.0)
    parser.add_argument(
        "--compute-preference-comfort-ms",
        type=float,
        default=None,
        help="latency below which only the linear ICRA-style preference cost applies",
    )
    parser.add_argument(
        "--compute-preference-deadline-ms",
        type=float,
        default=None,
        help=(
            "reference end of the latency preference scale; defaults to "
            "--deadline-median-ms"
        ),
    )
    parser.add_argument("--compute-preference-linear-weight", type=float, default=1.0)
    parser.add_argument("--compute-preference-excess-weight", type=float, default=2.0)
    parser.add_argument("--switching-cost-weight", type=float, default=1.0)
    parser.add_argument(
        "--fixed-switch-cost-ms",
        type=float,
        default=1.0,
        help=(
            "diagnostic one-time latency assigned to a switch; objective magnitude is "
            "controlled independently by --switching-cost-weight"
        ),
    )
    parser.add_argument("--information-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--fisher-contextual-objective",
        action="store_true",
        help="gate state-evidence and epistemic terms by posterior-weighted Fisher information",
    )
    parser.add_argument("--fisher-weight-min", type=float, default=0.1)
    parser.add_argument("--fisher-weight-max", type=float, default=1.0)
    parser.add_argument("--fisher-weight-power", type=float, default=1.0)
    parser.add_argument("--adaptive-only", action="store_true")
    parser.add_argument(
        "--hold-allocation-for-depth",
        action="store_true",
        help="run a selected (gamma, T) for T physical decisions before selecting again",
    )
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
            timing_profile=(None if args.timing_profile is None else str(args.timing_profile)),
            deadline_median_ms=args.deadline_median_ms,
            deadline_log_sigma=args.deadline_log_sigma,
            initial_resolution=args.initial_resolution,
            initial_depth=args.initial_depth,
            max_steps=args.max_steps,
            message_passing_iterations=args.message_passing_iterations,
            policy_workers=args.policy_workers,
            device=args.device,
            torch_threads=args.torch_threads,
            include_fixed=not args.adaptive_only,
            hold_allocation_for_depth=args.hold_allocation_for_depth,
            utility_success_weight=args.utility_success_weight,
            operational_cost_weight=args.operational_cost_weight,
            operational_cost_temperature=args.operational_cost_temperature,
            utility_g_weight=args.utility_g_weight,
            utility_preference_weight=args.utility_preference_weight,
            utility_epistemic_weight=args.utility_epistemic_weight,
            preference_improvement_weight=args.preference_improvement_weight,
            epistemic_improvement_weight=args.epistemic_improvement_weight,
            state_accuracy_weight=args.state_accuracy_weight,
            state_complexity_weight=args.state_complexity_weight,
            compute_cost_weight=args.compute_cost_weight,
            compute_preference_comfort_ms=args.compute_preference_comfort_ms,
            compute_preference_deadline_ms=args.compute_preference_deadline_ms,
            compute_preference_linear_weight=args.compute_preference_linear_weight,
            compute_preference_excess_weight=args.compute_preference_excess_weight,
            switching_cost_weight=args.switching_cost_weight,
            fixed_switch_cost_ms=args.fixed_switch_cost_ms,
            information_loss_weight=args.information_loss_weight,
            fisher_contextual_objective=args.fisher_contextual_objective,
            fisher_weight_min=args.fisher_weight_min,
            fisher_weight_max=args.fisher_weight_max,
            fisher_weight_power=args.fisher_weight_power,
        ),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
