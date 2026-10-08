"""Evaluate fleet search throughput under a shared planning-CPU budget."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .fleet_throughput import FleetSimulationConfig, evaluate_fleet_capacity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--fleet-sizes", type=int, nargs="+", required=True)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--sampling-seed", type=int, default=0)
    parser.add_argument("--mission-time-s", type=float, default=30.0)
    parser.add_argument("--cell-size-m", type=float, default=0.25)
    parser.add_argument("--robot-speed-mps", type=float, default=0.5)
    parser.add_argument("--planning-workers", type=int, default=1)
    parser.add_argument(
        "--deadline-s",
        type=float,
        default=None,
        help="planning deadline; defaults to one physical-action duration",
    )
    parser.add_argument(
        "--controllers",
        nargs="+",
        default=("adaptive", "fixed_g10_T2"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_fleet_capacity(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        fleet_sizes=tuple(args.fleet_sizes),
        repeats=args.repeats,
        sampling_seed=args.sampling_seed,
        config=FleetSimulationConfig(
            mission_time_s=args.mission_time_s,
            cell_size_m=args.cell_size_m,
            robot_speed_mps=args.robot_speed_mps,
            planning_workers=args.planning_workers,
            deadline_s=args.deadline_s,
        ),
        controllers=tuple(args.controllers),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
