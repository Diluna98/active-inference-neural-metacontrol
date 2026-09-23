"""Validated loading and instance-disjoint splitting of counterfactual datasets."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .allocations import ALLOCATIONS
from .counterfactuals import ONE_STEP_LABEL_SCHEMA


@dataclass(frozen=True)
class CounterfactualArrays:
    spatial: np.ndarray
    context: np.ndarray
    normalized_g: np.ndarray
    raw_g: np.ndarray
    risk: np.ndarray
    ambiguity: np.ndarray
    information_gain: np.ndarray
    compute_ms: np.ndarray
    switch_ms: np.ndarray
    candidate_actions: np.ndarray
    context_ids: np.ndarray
    instance_seeds: np.ndarray
    source_resolution: np.ndarray
    source_depth: np.ndarray
    label_schema: str = ONE_STEP_LABEL_SCHEMA
    rollout_horizon: int = 1
    rollout_discount: float = 1.0
    state_accuracy: np.ndarray | None = None
    state_complexity: np.ndarray | None = None
    uniform_risk: np.ndarray | None = None
    uniform_ambiguity: np.ndarray | None = None
    uniform_information_gain: np.ndarray | None = None
    posterior_risk: np.ndarray | None = None
    posterior_ambiguity: np.ndarray | None = None
    posterior_information_gain: np.ndarray | None = None
    selected_future_preference: np.ndarray | None = None
    selected_future_epistemic: np.ndarray | None = None
    selected_future_information_gain: np.ndarray | None = None
    posterior_future_preference: np.ndarray | None = None
    posterior_future_epistemic: np.ndarray | None = None
    posterior_future_information_gain: np.ndarray | None = None
    posterior_immediate_preference: np.ndarray | None = None
    posterior_immediate_epistemic: np.ndarray | None = None
    posterior_immediate_information_gain: np.ndarray | None = None
    posterior_improvement_preference: np.ndarray | None = None
    posterior_improvement_epistemic: np.ndarray | None = None
    posterior_improvement_information_gain: np.ndarray | None = None
    success: np.ndarray | None = None
    task_cost: np.ndarray | None = None
    task_outcome_schema: str | None = None

    def __post_init__(self) -> None:
        samples = self.spatial.shape[0]
        if self.spatial.ndim != 4:
            raise ValueError("spatial must have shape (samples, channels, height, width)")
        if self.context.ndim != 2 or self.context.shape[0] != samples:
            raise ValueError("context must have shape (samples, features)")
        expected_targets = (samples, len(ALLOCATIONS))
        for name in (
            "normalized_g",
            "raw_g",
            "risk",
            "ambiguity",
            "information_gain",
            "compute_ms",
            "switch_ms",
            "candidate_actions",
        ):
            if np.asarray(getattr(self, name)).shape != expected_targets:
                raise ValueError(f"{name} must have shape {expected_targets}")
        for name in ("context_ids", "instance_seeds", "source_resolution", "source_depth"):
            if np.asarray(getattr(self, name)).shape != (samples,):
                raise ValueError(f"{name} must contain one value per sample")
        for name in (
            "spatial",
            "context",
            "normalized_g",
            "raw_g",
            "risk",
            "ambiguity",
            "information_gain",
            "compute_ms",
            "switch_ms",
        ):
            if not np.all(np.isfinite(getattr(self, name))):
                raise ValueError(f"{name} must contain only finite values")
        if np.any(self.compute_ms < 0) or np.any(self.switch_ms < 0):
            raise ValueError("compute and switching costs must be nonnegative")
        if self.rollout_horizon < 1 or not 0 < self.rollout_discount <= 1:
            raise ValueError("invalid rollout metadata")
        if (self.state_accuracy is None) != (self.state_complexity is None):
            raise ValueError(
                "state_accuracy and state_complexity must both be present or both absent"
            )
        if self.state_accuracy is not None:
            if np.asarray(self.state_accuracy).shape != expected_targets:
                raise ValueError(f"state_accuracy must have shape {expected_targets}")
            if np.asarray(self.state_complexity).shape != expected_targets:
                raise ValueError(f"state_complexity must have shape {expected_targets}")
            if not np.all(np.isfinite(self.state_accuracy)) or not np.all(
                np.isfinite(self.state_complexity)
            ):
                raise ValueError("state-value labels must contain only finite values")
        uniform_values = (
            self.uniform_risk,
            self.uniform_ambiguity,
            self.uniform_information_gain,
        )
        if any(value is not None for value in uniform_values) and not all(
            value is not None for value in uniform_values
        ):
            raise ValueError("all uniform-policy labels must be present together")
        for name in ("uniform_risk", "uniform_ambiguity", "uniform_information_gain"):
            value = getattr(self, name)
            if value is not None:
                if np.asarray(value).shape != expected_targets:
                    raise ValueError(f"{name} must have shape {expected_targets}")
                if not np.all(np.isfinite(value)):
                    raise ValueError(f"{name} must contain only finite values")
        posterior_values = (
            self.posterior_risk,
            self.posterior_ambiguity,
            self.posterior_information_gain,
        )
        if any(value is not None for value in posterior_values) and not all(
            value is not None for value in posterior_values
        ):
            raise ValueError("all posterior-policy labels must be present together")
        for name in (
            "posterior_risk",
            "posterior_ambiguity",
            "posterior_information_gain",
        ):
            value = getattr(self, name)
            if value is not None:
                if np.asarray(value).shape != expected_targets:
                    raise ValueError(f"{name} must have shape {expected_targets}")
                if not np.all(np.isfinite(value)):
                    raise ValueError(f"{name} must contain only finite values")
        for prefix in ("selected_future", "posterior_future"):
            names = (
                f"{prefix}_preference",
                f"{prefix}_epistemic",
                f"{prefix}_information_gain",
            )
            values = tuple(getattr(self, name) for name in names)
            if any(value is not None for value in values) and not all(
                value is not None for value in values
            ):
                raise ValueError(f"all {prefix} labels must be present together")
            for name, value in zip(names, values, strict=True):
                if value is not None:
                    if np.asarray(value).shape != expected_targets:
                        raise ValueError(f"{name} must have shape {expected_targets}")
                    if not np.all(np.isfinite(value)):
                        raise ValueError(f"{name} must contain only finite values")
        for prefix in ("posterior_immediate", "posterior_improvement"):
            names = (
                f"{prefix}_preference",
                f"{prefix}_epistemic",
                f"{prefix}_information_gain",
            )
            values = tuple(getattr(self, name) for name in names)
            if any(value is not None for value in values) and not all(
                value is not None for value in values
            ):
                raise ValueError(f"all {prefix} labels must be present together")
            for name, value in zip(names, values, strict=True):
                if value is not None:
                    if np.asarray(value).shape != expected_targets:
                        raise ValueError(f"{name} must have shape {expected_targets}")
                    if not np.all(np.isfinite(value)):
                        raise ValueError(f"{name} must contain only finite values")
        if (self.success is None) != (self.task_cost is None):
            raise ValueError("success and task_cost must either both be present or both be absent")
        if self.success is not None:
            if np.asarray(self.success).shape != expected_targets:
                raise ValueError(f"success must have shape {expected_targets}")
            if np.asarray(self.task_cost).shape != expected_targets:
                raise ValueError(f"task_cost must have shape {expected_targets}")
            if not np.all(np.isfinite(self.success)) or not np.all(np.isfinite(self.task_cost)):
                raise ValueError("task-outcome labels must contain only finite values")
            if np.any((self.success < 0) | (self.success > 1)) or np.any(self.task_cost < 0):
                raise ValueError("task-outcome labels are outside their valid ranges")

    def subset(self, indices: np.ndarray) -> CounterfactualArrays:
        values = np.asarray(indices, dtype=int)
        return CounterfactualArrays(
            spatial=self.spatial[values],
            context=self.context[values],
            normalized_g=self.normalized_g[values],
            raw_g=self.raw_g[values],
            risk=self.risk[values],
            ambiguity=self.ambiguity[values],
            information_gain=self.information_gain[values],
            compute_ms=self.compute_ms[values],
            switch_ms=self.switch_ms[values],
            candidate_actions=self.candidate_actions[values],
            context_ids=self.context_ids[values],
            instance_seeds=self.instance_seeds[values],
            source_resolution=self.source_resolution[values],
            source_depth=self.source_depth[values],
            label_schema=self.label_schema,
            rollout_horizon=self.rollout_horizon,
            rollout_discount=self.rollout_discount,
            state_accuracy=(
                None if self.state_accuracy is None else self.state_accuracy[values]
            ),
            state_complexity=(
                None if self.state_complexity is None else self.state_complexity[values]
            ),
            uniform_risk=(None if self.uniform_risk is None else self.uniform_risk[values]),
            uniform_ambiguity=(
                None if self.uniform_ambiguity is None else self.uniform_ambiguity[values]
            ),
            uniform_information_gain=(
                None
                if self.uniform_information_gain is None
                else self.uniform_information_gain[values]
            ),
            posterior_risk=(
                None if self.posterior_risk is None else self.posterior_risk[values]
            ),
            posterior_ambiguity=(
                None
                if self.posterior_ambiguity is None
                else self.posterior_ambiguity[values]
            ),
            posterior_information_gain=(
                None
                if self.posterior_information_gain is None
                else self.posterior_information_gain[values]
            ),
            selected_future_preference=(
                None
                if self.selected_future_preference is None
                else self.selected_future_preference[values]
            ),
            selected_future_epistemic=(
                None
                if self.selected_future_epistemic is None
                else self.selected_future_epistemic[values]
            ),
            selected_future_information_gain=(
                None
                if self.selected_future_information_gain is None
                else self.selected_future_information_gain[values]
            ),
            posterior_future_preference=(
                None
                if self.posterior_future_preference is None
                else self.posterior_future_preference[values]
            ),
            posterior_future_epistemic=(
                None
                if self.posterior_future_epistemic is None
                else self.posterior_future_epistemic[values]
            ),
            posterior_future_information_gain=(
                None
                if self.posterior_future_information_gain is None
                else self.posterior_future_information_gain[values]
            ),
            posterior_immediate_preference=(
                None
                if self.posterior_immediate_preference is None
                else self.posterior_immediate_preference[values]
            ),
            posterior_immediate_epistemic=(
                None
                if self.posterior_immediate_epistemic is None
                else self.posterior_immediate_epistemic[values]
            ),
            posterior_immediate_information_gain=(
                None
                if self.posterior_immediate_information_gain is None
                else self.posterior_immediate_information_gain[values]
            ),
            posterior_improvement_preference=(
                None
                if self.posterior_improvement_preference is None
                else self.posterior_improvement_preference[values]
            ),
            posterior_improvement_epistemic=(
                None
                if self.posterior_improvement_epistemic is None
                else self.posterior_improvement_epistemic[values]
            ),
            posterior_improvement_information_gain=(
                None
                if self.posterior_improvement_information_gain is None
                else self.posterior_improvement_information_gain[values]
            ),
            success=None if self.success is None else self.success[values],
            task_cost=None if self.task_cost is None else self.task_cost[values],
            task_outcome_schema=self.task_outcome_schema,
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
            "normalized_g",
            "raw_g",
            "risk",
            "ambiguity",
            "information_gain",
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
        if "switch_ms" in archive.files:
            arrays["switch_ms"] = archive["switch_ms"].copy()
        label_schema = (
            str(archive["label_schema"].item())
            if "label_schema" in archive.files
            else ONE_STEP_LABEL_SCHEMA
        )
        rollout_horizon = (
            int(archive["rollout_horizon"].item())
            if "rollout_horizon" in archive.files
            else 1
        )
        rollout_discount = (
            float(archive["rollout_discount"].item())
            if "rollout_discount" in archive.files
            else 1.0
        )
        success = archive["success"].copy() if "success" in archive.files else None
        task_cost = archive["task_cost"].copy() if "task_cost" in archive.files else None
        state_accuracy = (
            archive["state_accuracy"].copy() if "state_accuracy" in archive.files else None
        )
        state_complexity = (
            archive["state_complexity"].copy()
            if "state_complexity" in archive.files
            else None
        )
        uniform_risk = archive["uniform_risk"].copy() if "uniform_risk" in archive.files else None
        uniform_ambiguity = (
            archive["uniform_ambiguity"].copy()
            if "uniform_ambiguity" in archive.files
            else None
        )
        uniform_information_gain = (
            archive["uniform_information_gain"].copy()
            if "uniform_information_gain" in archive.files
            else None
        )
        posterior_risk = (
            archive["posterior_risk"].copy() if "posterior_risk" in archive.files else None
        )
        posterior_ambiguity = (
            archive["posterior_ambiguity"].copy()
            if "posterior_ambiguity" in archive.files
            else None
        )
        posterior_information_gain = (
            archive["posterior_information_gain"].copy()
            if "posterior_information_gain" in archive.files
            else None
        )
        future_arrays = {
            name: archive[name].copy() if name in archive.files else None
            for name in (
                "selected_future_preference",
                "selected_future_epistemic",
                "selected_future_information_gain",
                "posterior_future_preference",
                "posterior_future_epistemic",
                "posterior_future_information_gain",
                "posterior_immediate_preference",
                "posterior_immediate_epistemic",
                "posterior_immediate_information_gain",
                "posterior_improvement_preference",
                "posterior_improvement_epistemic",
                "posterior_improvement_information_gain",
            )
        }
        task_outcome_schema = (
            str(archive["task_outcome_schema"].item())
            if "task_outcome_schema" in archive.files
            else None
        )

    metadata_by_context: dict[str, tuple[int, int, int]] = {}
    with (dataset_dir / "contexts.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            context_id = row["context_id"]
            if context_id in metadata_by_context:
                raise ValueError(f"duplicate context_id in contexts.csv: {context_id}")
            metadata_by_context[context_id] = (
                int(row["instance_seed"]),
                int(row.get("reference_resolution", 20)),
                int(row.get("reference_depth", 1)),
            )
    context_ids = np.asarray(arrays["context_ids"]).astype(str)
    try:
        metadata = [metadata_by_context[value] for value in context_ids]
    except KeyError as error:
        raise ValueError(f"context is missing from contexts.csv: {error.args[0]}") from error
    return CounterfactualArrays(
        spatial=np.asarray(arrays["spatial"], dtype=np.float32),
        context=np.asarray(arrays["context"], dtype=np.float32),
        normalized_g=np.asarray(arrays["normalized_g"], dtype=np.float32),
        raw_g=np.asarray(arrays["raw_g"], dtype=np.float32),
        risk=np.asarray(arrays["risk"], dtype=np.float32),
        ambiguity=np.asarray(arrays["ambiguity"], dtype=np.float32),
        information_gain=np.asarray(arrays["information_gain"], dtype=np.float32),
        compute_ms=np.asarray(arrays["compute_ms"], dtype=np.float32),
        switch_ms=(
            np.asarray(arrays["switch_ms"], dtype=np.float32)
            if "switch_ms" in arrays
            else np.zeros_like(arrays["compute_ms"], dtype=np.float32)
        ),
        candidate_actions=np.asarray(arrays["candidate_actions"], dtype=np.int8),
        context_ids=context_ids,
        instance_seeds=np.asarray([values[0] for values in metadata], dtype=int),
        source_resolution=np.asarray([values[1] for values in metadata], dtype=int),
        source_depth=np.asarray([values[2] for values in metadata], dtype=int),
        label_schema=label_schema,
        rollout_horizon=rollout_horizon,
        rollout_discount=rollout_discount,
        state_accuracy=(
            None if state_accuracy is None else np.asarray(state_accuracy, dtype=np.float32)
        ),
        state_complexity=(
            None
            if state_complexity is None
            else np.asarray(state_complexity, dtype=np.float32)
        ),
        uniform_risk=(
            None if uniform_risk is None else np.asarray(uniform_risk, dtype=np.float32)
        ),
        uniform_ambiguity=(
            None
            if uniform_ambiguity is None
            else np.asarray(uniform_ambiguity, dtype=np.float32)
        ),
        uniform_information_gain=(
            None
            if uniform_information_gain is None
            else np.asarray(uniform_information_gain, dtype=np.float32)
        ),
        posterior_risk=(
            None if posterior_risk is None else np.asarray(posterior_risk, dtype=np.float32)
        ),
        posterior_ambiguity=(
            None
            if posterior_ambiguity is None
            else np.asarray(posterior_ambiguity, dtype=np.float32)
        ),
        posterior_information_gain=(
            None
            if posterior_information_gain is None
            else np.asarray(posterior_information_gain, dtype=np.float32)
        ),
        **{
            name: None if value is None else np.asarray(value, dtype=np.float32)
            for name, value in future_arrays.items()
        },
        success=None if success is None else np.asarray(success, dtype=np.float32),
        task_cost=None if task_cost is None else np.asarray(task_cost, dtype=np.float32),
        task_outcome_schema=task_outcome_schema,
    )


def split_by_instance(
    dataset: CounterfactualArrays,
    *,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 0,
    stratify_by_source: bool = True,
) -> DatasetSplit:
    """Create deterministic train/validation/test splits with no instance leakage."""

    if validation_fraction <= 0 or test_fraction <= 0:
        raise ValueError("validation_fraction and test_fraction must be positive")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction + test_fraction must be less than one")
    instances = np.unique(dataset.instance_seeds)
    if instances.size < 3:
        raise ValueError("at least three distinct instances are required")
    rng = np.random.default_rng(seed)

    def partition(group: np.ndarray) -> tuple[list[int], list[int], list[int]]:
        if group.size < 3:
            raise ValueError("every source-allocation stratum requires at least three instances")
        shuffled = rng.permutation(group)
        validation_count = max(1, round(group.size * validation_fraction))
        test_count = max(1, round(group.size * test_fraction))
        while validation_count + test_count > group.size - 1:
            if validation_count >= test_count and validation_count > 1:
                validation_count -= 1
            elif test_count > 1:
                test_count -= 1
            else:
                raise ValueError("split fractions leave no training instances")
        return (
            [int(value) for value in shuffled[test_count + validation_count :]],
            [int(value) for value in shuffled[test_count : test_count + validation_count]],
            [int(value) for value in shuffled[:test_count]],
        )

    strata: dict[tuple[int, int], list[int]] = {}
    for instance in instances:
        mask = dataset.instance_seeds == instance
        sources = set(
            zip(
                dataset.source_resolution[mask].tolist(),
                dataset.source_depth[mask].tolist(),
                strict=True,
            )
        )
        if len(sources) != 1:
            raise ValueError("each instance must use exactly one source allocation")
        strata.setdefault(next(iter(sources)), []).append(int(instance))
    groups = (
        [np.asarray(strata[key], dtype=int) for key in sorted(strata)]
        if stratify_by_source
        else [instances]
    )
    train_values: list[int] = []
    validation_values: list[int] = []
    test_values: list[int] = []
    for group in groups:
        train_group, validation_group, test_group = partition(group)
        train_values.extend(train_group)
        validation_values.extend(validation_group)
        test_values.extend(test_group)
    train_instances = tuple(train_values)
    validation_instances = tuple(validation_values)
    test_instances = tuple(test_values)

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
