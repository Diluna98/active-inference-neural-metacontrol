"""Assemble completed generation shards into a separate partial dataset."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from active_inference_neural_metacontrol.counterfactuals import (
    concatenate_counterfactual_datasets,
    load_generated_counterfactuals,
    save_counterfactual_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    shard_dirs = sorted(
        path.parent
        for path in (args.source / "shards").glob("instance-*/complete.json")
        if (path.parent / "training_data.npz").is_file()
    )
    if not shard_dirs:
        raise RuntimeError("no completed shards found")
    datasets = [load_generated_counterfactuals(path) for path in shard_dirs]
    combined = concatenate_counterfactual_datasets(datasets)
    args.output.mkdir(parents=True, exist_ok=True)
    save_counterfactual_dataset(combined, args.output)

    source_manifest = args.source / "generation_manifest.json"
    if source_manifest.is_file():
        shutil.copy2(source_manifest, args.output / "source_generation_manifest.json")
    manifest = {
        "source": str(args.source),
        "completed_instances": len(shard_dirs),
        "instance_seeds": [int(path.name.removeprefix("instance-")) for path in shard_dirs],
        "contexts": len(combined.context_ids),
        "candidate_records": len(combined.branches),
        "partial_dataset": True,
    }
    (args.output / "partial_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
