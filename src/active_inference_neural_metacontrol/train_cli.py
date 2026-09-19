"""CLI for fitting and evaluating the neural task-performance model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .datasets import load_counterfactual_dataset, split_by_instance
from .training import TrainingConfig, train_task_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--success-weight", type=float, default=1.0)
    parser.add_argument("--relative-cost-weight", type=float, default=1.0)
    parser.add_argument("--ranking-weight", type=float, default=1.0)
    parser.add_argument("--ranking-temperature", type=float, default=5.0)
    parser.add_argument("--success-threshold", type=float, default=0.5)
    parser.add_argument("--compute-budget-ms", type=float)
    parser.add_argument(
        "--timing-profile",
        type=Path,
        help="controlled timing-profile JSON used for deadline filtering and checkpoint metadata",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--split-seed",
        type=int,
        help="dataset-split seed; defaults to --seed for backward compatibility",
    )
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_counterfactual_dataset(args.dataset_dir)
    split = split_by_instance(
        dataset,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed if args.split_seed is None else args.split_seed,
    )
    report = train_task_model(
        split,
        output_dir=args.output_dir,
        timing_profile=args.timing_profile,
        config=TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            patience=args.patience,
            success_weight=args.success_weight,
            relative_cost_weight=args.relative_cost_weight,
            ranking_weight=args.ranking_weight,
            ranking_temperature=args.ranking_temperature,
            success_threshold=args.success_threshold,
            compute_budget_ms=args.compute_budget_ms,
            seed=args.seed,
            device=args.device,
        ),
    )
    print(json.dumps(report, indent=2))
