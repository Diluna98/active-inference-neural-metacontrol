"""Export compact, path-independent reference artifacts from benchmark outputs."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIRS = (
    "final_profiling_single_worker_20000_100",
    "figure6_context_replay_20000_100",
    "heuristic_entropy_bounded_test_20000_100",
    "heuristic_fisher_bounded_test_20000_100",
    "pomcp_test_20000_100_s250_d30",
    "mr_pomcp_test_20000_100_s250_d30",
    "meta_regret_allocation50_info0_swi0_unseen_20000_100",
    "meta_regret_allocation50_info0_unseen_20000_100",
    "meta_regret_info0p8_no_fixed_switch_unseen_20000_100",
    "meta_regret_allocation50_info0p8_no_fixed_unseen_20000_100",
)


def portable(value):
    """Replace local machine paths with artifact-relative references."""
    if isinstance(value, dict):
        return {key: portable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [portable(item) for item in value]
    if isinstance(value, str) and (":\\" in value or value.startswith("/")):
        normalized = value.replace("\\", "/")
        for seed in range(3):
            if f"meta_regret_large_1200_seed{seed}/best_model.pt" in normalized:
                return f"artifacts/checkpoints/seed{seed}.pt"
        if "/results/" in normalized:
            return "artifacts/results/" + normalized.split("/results/", 1)[1]
        return normalized.rsplit("/", 1)[-1]
    return value


def export_json(source: Path, target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(portable(json.loads(source.read_text(encoding="utf-8"))), indent=2) + "\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--fleet-repo", type=Path, required=True)
    args = parser.parse_args()
    for directory in RESULT_DIRS:
        source = args.source_repo / "results" / directory
        target = ROOT / "artifacts/results" / directory
        target.mkdir(parents=True, exist_ok=True)
        for filename in ("summary.json", "episodes.csv", "adaptive_trajectory.csv"):
            path = source / filename
            if path.exists():
                if path.suffix == ".json":
                    export_json(path, target / filename)
                else:
                    shutil.copy2(path, target / filename)
    for filename in ("report.json", "fleet_trials.csv"):
        source = args.fleet_repo / "results/fleet_capacity_30s_1core" / filename
        target = ROOT / "artifacts/results/fleet_capacity_30s_1core" / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix == ".json":
            export_json(source, target)
        else:
            shutil.copy2(source, target)
    for filename in ("generation_manifest.json", "summary.json"):
        export_json(
            args.source_repo / "results/meta_q_large_1200" / filename,
            ROOT / "artifacts/training" / filename,
        )
    for seed in range(3):
        export_json(
            args.source_repo / f"results/meta_regret_large_1200_seed{seed}/training_report.json",
            ROOT / "artifacts/training" / f"seed{seed}_report.json",
        )
    print("Exported benchmark results, context traces, fleet trials, and training metadata.")


if __name__ == "__main__":
    main()
