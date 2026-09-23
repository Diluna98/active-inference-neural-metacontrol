"""Sweep a calibrated computation log-preference on validation MOS instances."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from active_inference_neural_metacontrol.closed_loop import (
    ClosedLoopConfig,
    evaluate_closed_loop,
)
from active_inference_neural_metacontrol.profiling import load_timing_profile


def _beta_name(value: float) -> str:
    return f"beta_{value:.6g}".replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--timing-profile", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--betas",
        nargs="+",
        type=float,
        default=(0.0, 0.001, 0.003, 0.01, 0.03),
        help="maximum fastest-to-slowest computation penalty in nats",
    )
    parser.add_argument("--instance-workers", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--message-passing-iterations", type=int, default=10)
    parser.add_argument("--policy-workers", type=int, default=1)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if any(beta < 0 for beta in args.betas):
        parser.error("all beta values must be nonnegative")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    seeds = tuple(int(value) for value in checkpoint["validation_instances"])
    latency, _ = load_timing_profile(args.timing_profile)
    latency_span = float(np.max(latency) - np.min(latency))
    if latency_span <= 0:
        raise ValueError("timing profile must contain a positive latency span")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for beta in args.betas:
        output = args.output_dir / _beta_name(beta)
        report = evaluate_closed_loop(
            instance_seeds=seeds,
            output_dir=output,
            instance_workers=args.instance_workers,
            resume=args.resume,
            config=ClosedLoopConfig(
                checkpoint=str(args.checkpoint),
                timing_profile=str(args.timing_profile),
                deadline_median_ms=250.0,
                initial_resolution=2,
                initial_depth=1,
                max_steps=args.max_steps,
                message_passing_iterations=args.message_passing_iterations,
                policy_workers=args.policy_workers,
                include_fixed=False,
                utility_preference_weight=0.0,
                preference_improvement_weight=0.0,
                utility_epistemic_weight=1.0,
                epistemic_improvement_weight=1.0,
                state_accuracy_weight=1.0,
                state_complexity_weight=1.0,
                compute_cost_weight=float(beta),
                compute_preference_comfort_ms=0.0,
                compute_preference_deadline_ms=latency_span,
                compute_preference_linear_weight=1.0,
                compute_preference_excess_weight=0.0,
                switching_cost_weight=0.0,
                information_loss_weight=0.0,
            ),
        )
        result = report["controllers"]["adaptive"]
        row = {
            "beta_nats": float(beta),
            "latency_span_ms": latency_span,
            "episodes": int(result["episodes"]),
            "success_rate": float(result["success_rate"]),
            "mean_steps": float(result["mean_steps"]),
            "mean_task_cost": float(result["mean_task_cost"]),
            "mean_switches": float(result["mean_switches"]),
            "mean_total_compute_ms": float(result["mean_total_compute_ms"]),
            "allocation_steps": result["allocation_steps"],
        }
        rows.append(row)
        print(json.dumps(row, indent=2))

    (args.output_dir / "sweep_summary.json").write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "timing_profile": str(args.timing_profile),
                "validation_instances": list(seeds),
                "cost_definition": (
                    "beta * latency_ms / (max_profile_ms - min_profile_ms); "
                    "the common minimum-latency offset does not affect argmax"
                ),
                "results": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "sweep_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        fieldnames = [name for name in rows[0] if name != "allocation_steps"]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {name: value for name, value in row.items() if name in fieldnames}
            for row in rows
        )


if __name__ == "__main__":
    main()
