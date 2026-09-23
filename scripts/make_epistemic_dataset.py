"""Derive an epistemic-value-per-depth training dataset from saved MOS branches."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    with np.load(args.source / "training_data.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}

    depths = np.asarray(arrays["depths"], dtype=np.float32)
    epistemic = np.asarray(arrays["ambiguity"], dtype=np.float32)
    arrays["raw_g"] = epistemic
    arrays["normalized_g"] = epistemic / depths[None, :]
    arrays["label_schema"] = np.asarray("one-step-epistemic-value-per-depth-v1")
    arrays["rollout_horizon"] = np.asarray(1, dtype=np.int64)
    arrays["rollout_discount"] = np.asarray(1.0, dtype=np.float64)
    np.savez_compressed(args.output / "training_data.npz", **arrays)
    shutil.copy2(args.source / "contexts.csv", args.output / "contexts.csv")

    print(f"contexts={epistemic.shape[0]}")
    print(f"candidates_per_context={epistemic.shape[1]}")
    print(f"target_min={arrays['normalized_g'].min():.9f}")
    print(f"target_mean={arrays['normalized_g'].mean():.9f}")
    print(f"target_max={arrays['normalized_g'].max():.9f}")


if __name__ == "__main__":
    main()
