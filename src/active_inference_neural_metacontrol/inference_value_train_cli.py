"""Train preference and epistemic metacontrol without task-outcome labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .datasets import load_counterfactual_dataset, split_by_instance
from .inference_value_training import (
    InferenceValueTrainingConfig,
    train_inference_value_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timing-profile", type=Path)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--ranking-temperature", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    dataset = load_counterfactual_dataset(args.dataset_dir)
    split = split_by_instance(dataset, seed=args.seed)
    report = train_inference_value_model(
        split,
        output_dir=args.output_dir,
        timing_profile=args.timing_profile,
        config=InferenceValueTrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            patience=args.patience,
            ranking_temperature=args.ranking_temperature,
            seed=args.seed,
            device=args.device,
        ),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
