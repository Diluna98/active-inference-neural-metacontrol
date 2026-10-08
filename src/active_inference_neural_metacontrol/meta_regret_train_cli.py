"""Train a reference-relative meta-regret controller."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .meta_q_data import load_meta_q_transitions
from .meta_regret import MetaRegretTrainingConfig, train_meta_regret


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--discount", type=float, default=0.98)
    parser.add_argument("--target-update-interval", type=int, default=100)
    parser.add_argument("--absolute-weight", type=float, default=0.25)
    parser.add_argument("--regret-weight", type=float, default=1.0)
    parser.add_argument("--ranking-weight", type=float, default=0.25)
    parser.add_argument("--ranking-margin", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=4)
    args = parser.parse_args()
    report = train_meta_regret(
        load_meta_q_transitions(args.dataset_dir),
        output_dir=args.output_dir,
        config=MetaRegretTrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            discount=args.discount,
            target_update_interval=args.target_update_interval,
            absolute_weight=args.absolute_weight,
            regret_weight=args.regret_weight,
            ranking_weight=args.ranking_weight,
            ranking_margin=args.ranking_margin,
            hidden_seed=args.seed,
            validation_fraction=args.validation_fraction,
            test_fraction=args.test_fraction,
            device=args.device,
            torch_threads=args.torch_threads,
        ),
    )
    print(json.dumps({key: value for key, value in report.items() if key != "history"}, indent=2))


if __name__ == "__main__":
    main()
