"""Validated loading and instance-disjoint splitting of counterfactual datasets."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .allocations import ALLOCATIONS


@dataclass(frozen=True)
class CounterfactualArrays:
    spatial: np.ndarray
    context: np.ndarray
    success: np.ndarray
    task_cost: np.ndarray
    compute_ms: np.ndarray
    candidate_actions: np.ndarray
    context_ids: np.ndarray
    instance_seeds: np.ndarray

    def __post_init__(self) -> None:
        samples = self.spatial.shape[0]
        if self.spatial.ndim != 4:
            raise ValueError("spatial must have shape (samples, channels, height, width)")
        if self.context.ndim != 2 or self.context.shape[0] != samples:
            raise ValueError("context must have shape (samples, features)")
        expected_targets = (samples, len(ALLOCATIONS))
        for name in ("success", "task_cost", "compute_ms", "candidate_actions"):
            if np.asarray(getattr(self, name)).shape != expected_targets:
                raise ValueError(f"{name} must have shape {expected_targets}")
        if self.context_ids.shape != (samples,) or self.instance_seeds.shape != (samples,):
            raise ValueError("context_ids and instance_seeds must contain one value per sample")
        for name in ("spatial", "context", "success", "task_cost", "compute_ms"):
            if not np.all(np.isfinite(getattr(self, name))):
                raise ValueError(f"{name} must contain only finite values")
        if np.any((self.success < 0) | (self.success > 1)):
            raise ValueError("success labels must lie in [0, 1]")
        if np.any(self.task_cost < 0) or np.any(self.compute_ms < 0):
            raise ValueError("task and compute costs must be nonnegative")

    def subset(self, indices: np.ndarray) -> CounterfactualArrays:
        values = np.asarray(indices, dtype=int)
        return CounterfactualArrays(
            spatial=self.spatial[values],
            context=self.context[values],
            success=self.success[values],
            task_cost=self.task_cost[values],
            compute_ms=self.compute_ms[values],
            candidate_actions=self.candidate_actions[values],
            context_ids=self.context_ids[values],
            instance_seeds=self.instance_seeds[values],
        )


@dataclass(frozen=True)
class DatasetSplit:
    train: CounterfactualArrays
    validation: CounterfactualArrays
    test: CounterfactualArrays
    train_instances: tuple[int, ...]
    validation_instances: tuple[int, ...]
    test_instances: tuple[int, ...]


def load_counterfactual_dataset(dataset_dir: Path) -> CounterfactualArrays:
    """Load training arrays and join their context IDs to MOS instance seeds."""

    dataset_dir = Path(dataset_dir)
    with np.load(dataset_dir / "training_data.npz", allow_pickle=False) as archive:
        required = {
            "spatial",
            "context",
            "success",
            "task_cost",
            "compute_ms",
            "candidate_actions",
            "context_ids",
            "resolutions",
            "depths",
        }
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"training_data.npz is missing: {sorted(missing)}")
        observed_allocations = tuple(
            zip(archive["resolutions"].tolist(), archive["depths"].tolist(), strict=True)
        )
        expected_allocations = tuple((item.resolution, item.depth) for item in ALLOCATIONS)
        if observed_allocations != expected_allocations:
            raise ValueError("allocation order does not match the package allocation order")
        arrays = {
            name: archive[name].copy() for name in required if name not in {"resolutions", "depths"}
        }

    seed_by_context: dict[str, int] = {}
    with (dataset_dir / "contexts.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            context_id = row["context_id"]
            if context_id in seed_by_context:
                raise ValueError(f"duplicate context_id in contexts.csv: {context_id}")
            seed_by_context[context_id] = int(row["instance_seed"])
    context_ids = np.asarray(arrays["context_ids"]).astype(str)
    try:
        instance_seeds = np.asarray([seed_by_context[value] for value in context_ids], dtype=int)
    except KeyError as error:
        raise ValueError(f"context is missing from contexts.csv: {error.args[0]}") from error
    return CounterfactualArrays(
        spatial=np.asarray(arrays["spatial"], dtype=np.float32),
        context=np.asarray(arrays["context"], dtype=np.float32),
        success=np.asarray(arrays["success"], dtype=np.float32),
        task_cost=np.asarray(arrays["task_cost"], dtype=np.float32),
        compute_ms=np.asarray(arrays["compute_ms"], dtype=np.float32),
        candidate_actions=np.asarray(arrays["candidate_actions"], dtype=np.int8),
        context_ids=context_ids,
        instance_seeds=instance_seeds,
    )


def split_by_instance(
    dataset: CounterfactualArrays,
    *,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 0,
) -> DatasetSplit:
    """Create deterministic train/validation/test splits with no instance leakage."""

    if validation_fraction <= 0 or test_fraction <= 0:
        raise ValueError("validation_fraction and test_fraction must be positive")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction + test_fraction must be less than one")
    instances = np.unique(dataset.instance_seeds)
    if instances.size < 3:
        raise ValueError("at least three distinct instances are required")
    shuffled = np.random.default_rng(seed).permutation(instances)
    validation_count = max(1, round(instances.size * validation_fraction))
    test_count = max(1, round(instances.size * test_fraction))
    while validation_count + test_count > instances.size - 1:
        if validation_count >= test_count and validation_count > 1:
            validation_count -= 1
        elif test_count > 1:
            test_count -= 1
        else:
            raise ValueError("split fractions leave no training instances")
    test_instances = tuple(int(value) for value in shuffled[:test_count])
    validation_instances = tuple(
        int(value) for value in shuffled[test_count : test_count + validation_count]
    )
    train_instances = tuple(int(value) for value in shuffled[test_count + validation_count :])

    def indices(values: tuple[int, ...]) -> np.ndarray:
        return np.flatnonzero(np.isin(dataset.instance_seeds, values))

    return DatasetSplit(
        train=dataset.subset(indices(train_instances)),
        validation=dataset.subset(indices(validation_instances)),
        test=dataset.subset(indices(test_instances)),
        train_instances=train_instances,
        validation_instances=validation_instances,
        test_instances=test_instances,
    )
