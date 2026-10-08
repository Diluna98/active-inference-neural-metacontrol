"""Train the sequential MOS meta-Q controller."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .meta_q import MetaQTrainingConfig, train_meta_q
from .meta_q_data import load_meta_q_transitions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--discount", type=float, default=0.98)
    parser.add_argument("--target-update-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()

    report = train_meta_q(
        load_meta_q_transitions(args.dataset_dir),
        output_dir=args.output_dir,
        config=MetaQTrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            discount=args.discount,
            target_update_interval=args.target_update_interval,
            hidden_seed=args.seed,
            validation_fraction=args.validation_fraction,
            test_fraction=args.test_fraction,
            device=args.device,
            torch_threads=args.torch_threads,
        ),
    )
    concise = {key: value for key, value in report.items() if key != "history"}
    print(json.dumps(concise, indent=2))


if __name__ == "__main__":
    main()
