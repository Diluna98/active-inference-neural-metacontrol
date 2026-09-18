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
        context=np.arange(samples * 17, dtype=np.float32).reshape(samples, 17),
        success=np.ones((samples, 12), dtype=np.float32),
        task_cost=np.arange(samples * 12, dtype=np.float32).reshape(samples, 12) + 1,
        compute_ms=np.ones((samples, 12), dtype=np.float32),
        candidate_actions=np.zeros((samples, 12), dtype=np.int8),
        context_ids=np.asarray([f"context-{index}" for index in range(samples)]),
        instance_seeds=np.repeat(np.arange(6), 2),
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


def test_loader_joins_context_metadata_and_checks_allocation_order():
    arrays = synthetic_arrays()
    output = Path("data/generated/test-dataset-loader")
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "training_data.npz",
        spatial=arrays.spatial,
        context=arrays.context,
        success=arrays.success,
        task_cost=arrays.task_cost,
        compute_ms=arrays.compute_ms,
        candidate_actions=arrays.candidate_actions,
        context_ids=arrays.context_ids,
        resolutions=np.asarray([item.resolution for item in ALLOCATIONS]),
        depths=np.asarray([item.depth for item in ALLOCATIONS]),
    )
    with (output / "contexts.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["context_id", "instance_seed"])
        writer.writeheader()
        writer.writerows(
            {
                "context_id": context_id,
                "instance_seed": instance_seed,
            }
            for context_id, instance_seed in zip(
                arrays.context_ids, arrays.instance_seeds, strict=True
            )
        )

    loaded = load_counterfactual_dataset(output)

    assert np.array_equal(loaded.instance_seeds, arrays.instance_seeds)
    assert np.array_equal(loaded.task_cost, arrays.task_cost)
