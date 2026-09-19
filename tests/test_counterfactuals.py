import json
from pathlib import Path

import numpy as np
import pytest

from active_inference_neural_metacontrol import Allocation
from active_inference_neural_metacontrol.counterfactuals import (
    concatenate_counterfactual_datasets,
    generate_mos_counterfactuals,
    load_generated_counterfactuals,
    save_counterfactual_dataset,
)
from active_inference_neural_metacontrol.generation import (
    GenerationConfig,
    generate_balanced_mos_counterfactuals,
    generate_resumable_mos_counterfactuals,
)

pytest.importorskip("active_inference_navigation.mos")


def test_small_counterfactual_dataset_is_aligned_and_serializable():
    allocations = (Allocation(2, 1), Allocation(5, 1))
    dataset = generate_mos_counterfactuals(
        instance_seeds=[0],
        reference_allocation=allocations[0],
        allocations=allocations,
        max_steps=2,
        message_passing_iterations=1,
    )

    assert dataset.spatial.shape == (1, 6, 20, 20)
    assert dataset.context.shape == (1, 17)
    assert dataset.success.shape == (1, 2)
    assert dataset.task_cost.shape == (1, 2)
    assert dataset.compute_ms.shape == (1, 2)
    assert dataset.switch_ms.shape == (1, 2)
    assert dataset.candidate_actions.shape == (1, 2)
    assert np.all(dataset.compute_ms >= 0.0)
    assert np.all(dataset.switch_ms >= 0.0)
    assert dataset.switch_ms[0, 0] == 0.0
    assert dataset.contexts[0]["candidate_decision"] == 1
    assert [row["decision"] for row in dataset.trajectory] == [0, 1]

    output_dir = Path("data/generated/test-counterfactuals")
    save_counterfactual_dataset(dataset, output_dir)
    archive = np.load(output_dir / "training_data.npz")
    assert archive["spatial"].shape == dataset.spatial.shape
    assert archive["resolutions"].tolist() == [2, 5]
    assert json.loads((output_dir / "summary.json").read_text())["contexts"] == 1

    loaded = load_generated_counterfactuals(output_dir)
    combined = concatenate_counterfactual_datasets((loaded, loaded))
    assert combined.spatial.shape == (2, 6, 20, 20)
    assert combined.context_ids == loaded.context_ids * 2


def test_resumable_generation_reuses_compatible_shards():
    output_dir = Path("data/generated/test-resumable")
    config = GenerationConfig(
        reference_resolution=2,
        reference_depth=1,
        max_steps=2,
        message_passing_iterations=1,
        allocations=((2, 1), (5, 1)),
    )
    first = generate_resumable_mos_counterfactuals(
        instance_seeds=[0, 1],
        output_dir=output_dir,
        config=config,
        instance_workers=1,
    )
    marker = output_dir / "shards/instance-0/complete.json"
    marker_timestamp = marker.stat().st_mtime_ns
    second = generate_resumable_mos_counterfactuals(
        instance_seeds=[0, 1],
        output_dir=output_dir,
        config=config,
        instance_workers=1,
    )

    assert marker.stat().st_mtime_ns == marker_timestamp
    assert first.context_ids == second.context_ids
    assert json.loads((output_dir / "generation_manifest.json").read_text())["contexts"] == len(
        second.context_ids
    )


def test_balanced_generation_round_robins_source_allocations():
    output_dir = Path("data/generated/test-balanced")
    sources = (Allocation(2, 1), Allocation(5, 1))
    config = GenerationConfig(
        reference_resolution=2,
        reference_depth=1,
        max_steps=2,
        message_passing_iterations=1,
        allocations=((2, 1), (5, 1)),
    )
    dataset = generate_balanced_mos_counterfactuals(
        instance_seeds=[10, 11],
        output_dir=output_dir,
        config=config,
        source_allocations=sources,
    )

    sources_by_seed = {
        int(row["instance_seed"]): (
            int(row["reference_resolution"]),
            int(row["reference_depth"]),
        )
        for row in dataset.contexts
    }
    assert sources_by_seed == {10: (2, 1), 11: (5, 1)}
    manifest = json.loads((output_dir / "generation_manifest.json").read_text())
    assert manifest["source_mode"] == "balanced_round_robin"
