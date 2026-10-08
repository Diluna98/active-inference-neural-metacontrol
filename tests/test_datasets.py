import csv
from pathlib import Path

import numpy as np

from active_inference_neural_metacontrol import ALLOCATIONS
from active_inference_neural_metacontrol.datasets import (
    CounterfactualArrays,
    load_counterfactual_dataset,
    split_by_instance,
)


def synthetic_arrays() -> CounterfactualArrays:
    samples = 12
    return CounterfactualArrays(
        spatial=np.full((samples, 6, 20, 20), 0.1, dtype=np.float32),
        context=np.arange(samples * 16, dtype=np.float32).reshape(samples, 16),
        normalized_g=np.arange(samples * 12, dtype=np.float32).reshape(samples, 12) - 5,
        raw_g=np.ones((samples, 12), dtype=np.float32),
        risk=np.ones((samples, 12), dtype=np.float32),
        ambiguity=np.ones((samples, 12), dtype=np.float32),
        information_gain=np.zeros((samples, 12), dtype=np.float32),
        compute_ms=np.ones((samples, 12), dtype=np.float32),
        switch_ms=np.zeros((samples, 12), dtype=np.float32),
        candidate_actions=np.zeros((samples, 12), dtype=np.int8),
        context_ids=np.asarray([f"context-{index}" for index in range(samples)]),
        instance_seeds=np.repeat(np.arange(6), 2),
        source_resolution=np.full(samples, 20),
        source_depth=np.ones(samples, dtype=int),
    )


def test_instance_split_has_no_leakage_and_is_reproducible():
    arrays = synthetic_arrays()
    first = split_by_instance(arrays, seed=7)
    second = split_by_instance(arrays, seed=7)

    assert first.train_instances == second.train_instances
    assert set(first.train_instances).isdisjoint(first.validation_instances)
    assert set(first.train_instances).isdisjoint(first.test_instances)
    assert set(first.validation_instances).isdisjoint(first.test_instances)
    assert len(first.train.context) + len(first.validation.context) + len(first.test.context) == 12


def test_instance_split_stratifies_every_source_allocation():
    samples = 36
    source_allocations = np.repeat(
        np.asarray([(item.resolution, item.depth) for item in ALLOCATIONS]), 3, axis=0
    )
    arrays = CounterfactualArrays(
        spatial=np.zeros((samples, 6, 20, 20), dtype=np.float32),
        context=np.zeros((samples, 16), dtype=np.float32),
        normalized_g=np.ones((samples, 12), dtype=np.float32),
        raw_g=np.ones((samples, 12), dtype=np.float32),
        risk=np.ones((samples, 12), dtype=np.float32),
        ambiguity=np.ones((samples, 12), dtype=np.float32),
        information_gain=np.zeros((samples, 12), dtype=np.float32),
        compute_ms=np.ones((samples, 12), dtype=np.float32),
        switch_ms=np.ones((samples, 12), dtype=np.float32),
        candidate_actions=np.zeros((samples, 12), dtype=np.int8),
        context_ids=np.asarray([f"balanced-{index}" for index in range(samples)]),
        instance_seeds=np.arange(samples),
        source_resolution=source_allocations[:, 0],
        source_depth=source_allocations[:, 1],
    )

    split = split_by_instance(arrays, seed=4)

    expected = {(item.resolution, item.depth) for item in ALLOCATIONS}
    for partition in (split.train, split.validation, split.test):
        observed = set(
            zip(
                partition.source_resolution.tolist(),
                partition.source_depth.tolist(),
                strict=True,
            )
        )
        assert observed == expected


def test_loader_joins_context_metadata_and_checks_allocation_order():
    arrays = synthetic_arrays()
    output = Path("data/generated/test-dataset-loader")
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "training_data.npz",
        spatial=arrays.spatial,
        context=arrays.context,
        normalized_g=arrays.normalized_g,
        raw_g=arrays.raw_g,
        risk=arrays.risk,
        ambiguity=arrays.ambiguity,
        information_gain=arrays.information_gain,
        compute_ms=arrays.compute_ms,
        switch_ms=arrays.switch_ms,
        candidate_actions=arrays.candidate_actions,
        context_ids=arrays.context_ids,
        resolutions=np.asarray([item.resolution for item in ALLOCATIONS]),
        depths=np.asarray([item.depth for item in ALLOCATIONS]),
    )
    with (output / "contexts.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "context_id",
                "instance_seed",
                "reference_resolution",
                "reference_depth",
            ],
        )
        writer.writeheader()
        writer.writerows(
            {
                "context_id": context_id,
                "instance_seed": instance_seed,
                "reference_resolution": 20,
                "reference_depth": 1,
            }
            for context_id, instance_seed in zip(
                arrays.context_ids, arrays.instance_seeds, strict=True
            )
        )

    loaded = load_counterfactual_dataset(output)

    assert np.array_equal(loaded.instance_seeds, arrays.instance_seeds)
    assert np.array_equal(loaded.normalized_g, arrays.normalized_g)
    assert np.array_equal(loaded.raw_g, arrays.raw_g)
    assert np.array_equal(loaded.switch_ms, arrays.switch_ms)
