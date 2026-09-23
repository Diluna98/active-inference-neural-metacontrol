"""Train the four-head next-step Active-Inference value model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .datasets import load_counterfactual_dataset, split_by_instance
from .features import (
    DECOMPOSED_FEATURE_SCHEMA,
    FOUR_TERM_ABLATION_FEATURE_SCHEMA,
    FOUR_TERM_NO_ALLOCATION_FEATURE_SCHEMA,
    FOUR_TERM_PREDICTED_OBSERVATION_FEATURE_SCHEMA,
)
from .four_term_training import FourTermTrainingConfig, train_four_term_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--ranking-temperature", type=float, default=0.1)
    parser.add_argument("--ranking-weight", type=float, default=1.0)
    parser.add_argument(
        "--detach-state-terms-in-ranking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="prevent ranking loss from directly updating state accuracy/complexity heads",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--policy-aggregation",
        choices=("selected", "uniform", "posterior"),
        default="selected",
    )
    parser.add_argument(
        "--target-horizon",
        choices=("all_steps_mean", "future_only_mean", "immediate_improvement"),
        default="all_steps_mean",
        help="use cumulative G/T labels or g1/g2/mean(g2,g3) future-only labels",
    )
    parser.add_argument(
        "--stratify-by-source",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="stratify the split by source allocation (disable for very small pilots)",
    )
    parser.add_argument(
        "--remove-current-posterior-action-policy-entropy",
        action="store_true",
        help=(
            "train without the current posterior map, selected action, or "
            "task-policy entropy"
        ),
    )
    parser.add_argument(
        "--remove-current-allocation",
        action="store_true",
        help=(
            "also remove current resolution and depth; this selects the predicted-context-only "
            "ablation schema"
        ),
    )
    parser.add_argument(
        "--include-predicted-observation",
        action="store_true",
        help=(
            "append the pre-observation binary prediction for t+1; requires "
            "--remove-current-allocation"
        ),
    )
    args = parser.parse_args()
    if args.include_predicted_observation and not args.remove_current_allocation:
        parser.error("--include-predicted-observation requires --remove-current-allocation")
    dataset = load_counterfactual_dataset(args.dataset_dir)
    split = split_by_instance(
        dataset,
        seed=args.seed,
        stratify_by_source=args.stratify_by_source,
    )
    report = train_four_term_model(
        split,
        output_dir=args.output_dir,
        config=FourTermTrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            ranking_temperature=args.ranking_temperature,
            ranking_weight=args.ranking_weight,
            detach_state_terms_in_ranking=args.detach_state_terms_in_ranking,
            seed=args.seed,
            device=args.device,
            policy_aggregation=args.policy_aggregation,
            target_horizon=args.target_horizon,
            feature_schema=(
                FOUR_TERM_PREDICTED_OBSERVATION_FEATURE_SCHEMA
                if args.include_predicted_observation
                else FOUR_TERM_NO_ALLOCATION_FEATURE_SCHEMA
                if args.remove_current_allocation
                else (
                    FOUR_TERM_ABLATION_FEATURE_SCHEMA
                    if args.remove_current_posterior_action_policy_entropy
                    else DECOMPOSED_FEATURE_SCHEMA
                )
            ),
        ),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
