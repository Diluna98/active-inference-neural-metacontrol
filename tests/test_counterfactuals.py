import json
from pathlib import Path

import numpy as np
import pytest

from active_inference_neural_metacontrol import Allocation
from active_inference_neural_metacontrol.counterfactuals import (
    ACCUMULATED_LABEL_SCHEMA,
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
from active_inference_neural_metacontrol.rollout_diagnostic import (
    GOracleConfig,
    diagnose_accumulated_g,
    run_g_oracle_episode,
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
        collect_task_outcomes=True,
        collect_research_archive=True,
    )

    assert dataset.spatial.shape == (1, 6, 20, 20)
    assert dataset.context.shape == (1, 16)
    assert dataset.normalized_g.shape == (1, 2)
    assert dataset.raw_g.shape == (1, 2)
    assert np.all(np.isfinite(dataset.normalized_g))
    assert np.all(np.isfinite(dataset.raw_g))
    for index, allocation in enumerate(allocations):
        expected = (
            dataset.risk[0, index]
            + dataset.ambiguity[0, index]
            - dataset.information_gain[0, index]
        ) / allocation.depth
        assert dataset.normalized_g[0, index] == pytest.approx(expected)
    assert dataset.compute_ms.shape == (1, 2)
    assert dataset.switch_ms.shape == (1, 2)
    assert dataset.candidate_actions.shape == (1, 2)
    assert dataset.success is not None and dataset.success.shape == (1, 2)
    assert dataset.task_cost is not None and dataset.task_cost.shape == (1, 2)
    assert np.all((dataset.success == 0.0) | (dataset.success == 1.0))
    assert np.all(dataset.task_cost > 0.0)
    assert np.all(dataset.compute_ms >= 0.0)
    assert np.all(dataset.switch_ms >= 0.0)
    assert dataset.switch_ms[0, 0] == 0.0
    assert dataset.contexts[0]["candidate_decision"] == 1
    assert [row["decision"] for row in dataset.trajectory] == [0, 1]
    assert dataset.research_arrays is not None
    assert dataset.research_arrays["policy_g"].shape == (1, 2, 125)
    assert dataset.research_arrays["canonical_target_posterior"].shape == (1, 2, 20, 20)
    np.testing.assert_allclose(
        dataset.research_arrays["candidate_predicted_observation"].sum(axis=2),
        1.0,
    )

    output_dir = Path("data/generated/test-counterfactuals")
    save_counterfactual_dataset(dataset, output_dir)
    archive = np.load(output_dir / "training_data.npz")
    assert archive["spatial"].shape == dataset.spatial.shape
    assert archive["resolutions"].tolist() == [2, 5]
    assert archive["rollout_horizon"].item() == 1
    assert archive["success"].shape == (1, 2)
    assert archive["task_cost"].shape == (1, 2)
    assert (output_dir / "research_archive.npz").is_file()
    assert json.loads((output_dir / "summary.json").read_text())["contexts"] == 1

    loaded = load_generated_counterfactuals(output_dir)
    combined = concatenate_counterfactual_datasets((loaded, loaded))
    assert combined.spatial.shape == (2, 6, 20, 20)
    assert combined.context_ids == loaded.context_ids * 2
    assert combined.success is not None and combined.success.shape == (2, 2)
    assert combined.research_arrays is not None
    assert combined.research_arrays["policy_g"].shape == (2, 2, 125)


def test_accumulated_target_is_generated_and_serialized():
    allocations = (Allocation(2, 1), Allocation(5, 1))
    dataset = generate_mos_counterfactuals(
        instance_seeds=[0],
        reference_allocation=allocations[0],
        allocations=allocations,
        max_steps=2,
        message_passing_iterations=1,
        rollout_horizon=2,
    )

    assert dataset.label_schema == ACCUMULATED_LABEL_SCHEMA
    assert dataset.rollout_horizon == 2
    assert np.all(np.isfinite(dataset.normalized_g))
    assert {row["rollout_horizon"] for row in dataset.branches} == {2}
    assert all("immediate_normalized_g" in row for row in dataset.branches)

    output_dir = Path("data/generated/test-accumulated-counterfactuals")
    save_counterfactual_dataset(dataset, output_dir)
    loaded = load_generated_counterfactuals(output_dir)
    assert loaded.label_schema == ACCUMULATED_LABEL_SCHEMA
    assert loaded.rollout_horizon == 2
    np.testing.assert_allclose(loaded.normalized_g, dataset.normalized_g)


def test_branch_stride_skips_only_unrecorded_counterfactuals():
    allocations = (Allocation(2, 1), Allocation(5, 1))
    complete = generate_mos_counterfactuals(
        instance_seeds=[0],
        reference_allocation=allocations[0],
        allocations=allocations,
        max_steps=4,
        branch_stride=1,
        message_passing_iterations=1,
    )
    strided = generate_mos_counterfactuals(
        instance_seeds=[0],
        reference_allocation=allocations[0],
        allocations=allocations,
        max_steps=4,
        branch_stride=2,
        message_passing_iterations=1,
    )

    retained = [
        index for index, row in enumerate(complete.contexts) if row["feature_decision"] % 2 == 0
    ]
    assert strided.trajectory == complete.trajectory
    assert strided.context_ids == tuple(complete.context_ids[index] for index in retained)
    np.testing.assert_allclose(strided.normalized_g, complete.normalized_g[retained])
    np.testing.assert_array_equal(strided.candidate_actions, complete.candidate_actions[retained])


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


def test_accumulated_g_diagnostic_runs_candidate_controlled_rollouts():
    allocations = (Allocation(2, 1), Allocation(5, 1))
    output_dir = Path("data/generated/test-rollout-diagnostic")

    summary = diagnose_accumulated_g(
        instance_seeds=[0],
        output_dir=output_dir,
        reference_allocation=allocations[0],
        allocations=allocations,
        max_reference_steps=3,
        branch_stride=1,
        rollout_horizon=2,
        message_passing_iterations=1,
    )

    assert summary["contexts"] >= 1
    assert summary["rollout_horizon"] == 2
    assert 0.0 <= summary["allocation_disagreement_rate"] <= 1.0
    assert (output_dir / "branches.csv").is_file()
    assert (output_dir / "contexts.csv").is_file()


def test_g_oracle_controls_a_complete_episode():
    row = run_g_oracle_episode(
        0,
        oracle_horizon=2,
        config=GOracleConfig(
            max_steps=2,
            rollout_horizon=2,
            message_passing_iterations=1,
        ),
        allocations=(Allocation(2, 1), Allocation(5, 1)),
    )

    assert row["controller"] == "g_oracle_H2"
    assert 1 <= row["steps"] <= 2
    assert sum(json.loads(row["allocation_counts"]).values()) == row["steps"]
