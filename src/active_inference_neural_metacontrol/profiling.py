"""Controlled inference and representation-switch timing profiles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .allocations import ALLOCATIONS
from .datasets import load_counterfactual_dataset


def build_timing_profile(
    dataset_dir: Path,
    *,
    allow_parallel_generation: bool = False,
) -> dict[str, Any]:
    """Summarize candidate inference and source-to-candidate switching latency."""

    dataset_dir = Path(dataset_dir)
    manifest_path = dataset_dir / "generation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    workers = int(manifest.get("instance_workers", 1))
    if workers != 1 and not allow_parallel_generation:
        raise ValueError(
            "timing profiles require data generated with --instance-workers 1; "
            "parallel generation contaminates latency measurements"
        )
    arrays = load_counterfactual_dataset(dataset_dir)
    inference = []
    for index, allocation in enumerate(ALLOCATIONS):
        values = arrays.compute_ms[:, index]
        inference.append(
            {
                "resolution": allocation.resolution,
                "depth": allocation.depth,
                "samples": int(values.size),
                "median_ms": float(np.median(values)),
                "p95_ms": float(np.percentile(values, 95)),
            }
        )

    switching = []
    observed_sources = set()
    for source in ALLOCATIONS:
        mask = (arrays.source_resolution == source.resolution) & (
            arrays.source_depth == source.depth
        )
        if not np.any(mask):
            continue
        observed_sources.add((source.resolution, source.depth))
        for index, target in enumerate(ALLOCATIONS):
            values = arrays.switch_ms[mask, index]
            switching.append(
                {
                    "source_resolution": source.resolution,
                    "source_depth": source.depth,
                    "target_resolution": target.resolution,
                    "target_depth": target.depth,
                    "samples": int(values.size),
                    "median_ms": float(np.median(values)),
                    "p95_ms": float(np.percentile(values, 95)),
                }
            )
    expected_sources = {(item.resolution, item.depth) for item in ALLOCATIONS}
    return {
        "dataset_dir": str(dataset_dir),
        "instance_workers": workers,
        "controlled_single_process": workers == 1,
        "all_sources_observed": observed_sources == expected_sources,
        "missing_sources": [
            {"resolution": resolution, "depth": depth}
            for resolution, depth in sorted(expected_sources - observed_sources)
        ],
        "inference": inference,
        "switching": switching,
    }


def save_timing_profile(profile: dict[str, Any], output: Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")


def load_timing_profile(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load ordered inference and complete source-to-target switching medians."""

    profile = json.loads(Path(path).read_text(encoding="utf-8"))
    inference_by_allocation = {
        (int(row["resolution"]), int(row["depth"])): float(row["median_ms"])
        for row in profile["inference"]
    }
    switching_by_pair = {
        (
            int(row["source_resolution"]),
            int(row["source_depth"]),
            int(row["target_resolution"]),
            int(row["target_depth"]),
        ): float(row["median_ms"])
        for row in profile["switching"]
    }
    compute = np.asarray(
        [inference_by_allocation[(item.resolution, item.depth)] for item in ALLOCATIONS],
        dtype=np.float32,
    )
    switching = np.asarray(
        [
            [
                switching_by_pair[
                    (source.resolution, source.depth, target.resolution, target.depth)
                ]
                for target in ALLOCATIONS
            ]
            for source in ALLOCATIONS
        ],
        dtype=np.float32,
    )
    if not np.all(np.isfinite(compute)) or not np.all(np.isfinite(switching)):
        raise ValueError("timing profile contains non-finite medians")
    return compute, switching
