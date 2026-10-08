from pathlib import Path

import numpy as np
import pytest

from active_inference_neural_metacontrol.allocations import ALLOCATIONS, Allocation
from active_inference_neural_metacontrol.meta_q import (
    aggregate_reference_relative_q,
    select_cheapest_sufficient,
    select_joint_task_compute,
)
from active_inference_neural_metacontrol.meta_q_data import (
    concatenate_meta_q_datasets,
    generate_meta_q_transitions,
    generate_resumable_meta_q_transitions,
    load_meta_q_transitions,
    save_meta_q_transitions,
)
from active_inference_neural_metacontrol.meta_q_features import (
    MetaQFeatures,
    project_meta_q_context,
    summarize_decision_context,
)
from active_inference_neural_metacontrol.meta_regret import (
    META_REGRET_REFERENCE,
    MetaRegretNetwork,
    allocation_features,
    augment_regret_context,
    relative_regret,
)


def test_meta_q_context_excludes_source_allocation_and_appends_observation():
    context = np.arange(16, dtype=np.float32)
    observation = np.asarray([3, 4, 1, 0, 1])

    projected = project_meta_q_context(context, observation)

    assert projected.shape == (56,)
    np.testing.assert_array_equal(projected[:5], context[:5])
    np.testing.assert_array_equal(projected[5:9], context[12:16])
    assert projected[9:].sum() == 5.0


def test_decision_context_summary_uses_predicted_belief_and_sensor_information():
    spatial = np.zeros((6, 20, 20), dtype=np.float32)
    spatial[0, 0, 0] = 1.0
    spatial[1, 0, 0] = 0.5
    spatial[1, 0, 1] = 0.5
    spatial[4, 0, 0] = 0.1
    spatial[4, 0, 1] = 0.9
    spatial[5, 0, 0] = 0.2
    spatial[5, 0, 1] = 0.8
    context = np.zeros(56, dtype=np.float32)
    context[6:9] = (0.25, 0.5, 0.75)

    summary = summarize_decision_context(MetaQFeatures(spatial=spatial, context=context))

    assert summary["belief_entropy_native"] == pytest.approx(0.25)
    assert summary["belief_entropy_canonical"] == pytest.approx(0.0)
    assert summary["predicted_entropy_canonical"] > 0.0
    assert summary["policy_entropy"] == pytest.approx(0.5)
    assert summary["policy_margin"] == pytest.approx(0.75)
    assert summary["predicted_detection_probability"] == pytest.approx(0.5)
    assert summary["posterior_weighted_fisher"] == pytest.approx(0.5)
    assert summary["expected_information_gain_nats"] > 0.0


def test_cheapest_sufficient_downgrades_but_preserves_required_task_value():
    allocations = (
        Allocation(2, 1),
        Allocation(2, 2),
        Allocation(5, 2),
    )
    compute = np.asarray([1.0, 3.0, 8.0])

    cheap = select_cheapest_sufficient(
        np.asarray([9.7, 10.0, 10.1]),
        allocations=allocations,
        compute_ms=compute,
        current_allocation=allocations[2],
        task_tolerance=0.5,
    )
    required = select_cheapest_sufficient(
        np.asarray([8.0, 9.0, 10.1]),
        allocations=allocations,
        compute_ms=compute,
        current_allocation=allocations[0],
        task_tolerance=0.5,
    )

    assert cheap.allocation == Allocation(2, 1)
    assert required.allocation == Allocation(5, 2)


def test_joint_selector_trades_predicted_regret_against_compute():
    allocations = (Allocation(2, 1), Allocation(5, 2))
    q_values = np.asarray([9.8, 10.0])
    compute = np.asarray([5.0, 25.0])

    task_first = select_joint_task_compute(
        q_values,
        allocations=allocations,
        compute_ms=compute,
        current_allocation=allocations[0],
        compute_price=0.001,
    )
    compute_first = select_joint_task_compute(
        q_values,
        allocations=allocations,
        compute_ms=compute,
        current_allocation=allocations[0],
        compute_price=0.02,
    )

    assert task_first.allocation == Allocation(5, 2)
    assert compute_first.allocation == Allocation(2, 1)


def test_joint_selector_can_replace_literal_switching_with_information_loss():
    allocations = (Allocation(5, 1), Allocation(2, 1))
    q_values = np.asarray([10.0, 10.1])
    compute = np.asarray([10.0, 5.0])
    information = np.asarray([0.0, 0.4])

    without_loss = select_joint_task_compute(
        q_values,
        allocations=allocations,
        compute_ms=compute,
        current_allocation=allocations[0],
        compute_price=0.001,
        information_loss=information,
        information_loss_weight=0.0,
    )
    with_loss = select_joint_task_compute(
        q_values,
        allocations=allocations,
        compute_ms=compute,
        current_allocation=allocations[0],
        compute_price=0.001,
        information_loss=information,
        information_loss_weight=1.0,
    )

    assert without_loss.allocation == Allocation(2, 1)
    assert with_loss.allocation == Allocation(5, 1)


def test_resolution_only_switch_penalty_does_not_charge_depth_change():
    allocations = (Allocation(5, 1), Allocation(5, 2), Allocation(10, 1))
    q_values = np.asarray([9.8, 10.0, 10.0])
    compute = np.zeros(3)

    selected = select_joint_task_compute(
        q_values,
        allocations=allocations,
        compute_ms=compute,
        current_allocation=allocations[0],
        compute_price=0.01,
        switch_cost_ms=50.0,
        switch_penalty_mode="resolution",
    )

    assert selected.allocation == Allocation(5, 2)


def test_reference_relative_ensemble_penalizes_uncertain_candidates():
    samples = np.asarray(
        [
            [1.4, 1.0, 1.8],
            [0.6, 1.0, 1.8],
        ]
    )

    robust, mean, standard_deviation = aggregate_reference_relative_q(
        samples, reference_index=1, uncertainty_beta=1.0
    )

    np.testing.assert_allclose(mean, [0.0, 0.0, 0.8], atol=1e-12)
    np.testing.assert_allclose(standard_deviation, [0.4, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(robust, [-0.4, 0.0, 0.8], atol=1e-12)


def test_regret_context_restores_source_allocation_identity():
    context = np.zeros(56, dtype=np.float32)
    augmented = augment_regret_context(context, Allocation(10, 3))

    assert augmented.shape == (63,)
    np.testing.assert_array_equal(augmented[-7:], allocation_features(Allocation(10, 3)))
    assert augmented[-7:].sum() == 2.0


def test_regret_network_anchors_reference_q_to_value_head():
    torch = pytest.importorskip("torch")
    model = MetaRegretNetwork()
    spatial = torch.zeros((2, 6, 20, 20))
    context = torch.zeros((2, 63))

    q_values = model(spatial, context)
    regret = relative_regret(q_values, ALLOCATIONS.index(META_REGRET_REFERENCE))

    assert q_values.shape == (2, 12)
    torch.testing.assert_close(regret[:, ALLOCATIONS.index(META_REGRET_REFERENCE)], torch.zeros(2))


def test_small_meta_q_transition_archive_round_trips():
    pytest.importorskip("active_inference_navigation.mos")
    allocations = (Allocation(2, 1), Allocation(5, 1))
    dataset, audit = generate_meta_q_transitions(
        instance_seeds=[0],
        source_allocations=(allocations[0],),
        allocations=allocations,
        max_steps=2,
        message_passing_iterations=1,
    )

    assert dataset.spatial.shape == (1, 6, 20, 20)
    assert dataset.context.shape == (1, 56)
    assert dataset.next_spatial.shape == (1, 2, 6, 20, 20)
    assert dataset.next_context.shape == (1, 2, 56)
    assert dataset.reward.shape == (1, 2)
    assert dataset.terminal.all()
    assert len(audit) == 2

    output_dir = Path("results/test-meta-q-transition-archive")
    save_meta_q_transitions(dataset, output_dir, audit_rows=audit)
    loaded = load_meta_q_transitions(output_dir)
    combined = concatenate_meta_q_datasets((loaded, loaded))
    np.testing.assert_allclose(loaded.next_spatial, dataset.next_spatial, atol=2.5e-4)
    np.testing.assert_allclose(loaded.reward, dataset.reward)
    assert loaded.spatial.dtype == np.float16
    assert combined.spatial.shape[0] == 2
    assert (output_dir / "branches.csv").is_file()


def test_resumable_meta_q_generation_writes_reusable_shards():
    pytest.importorskip("active_inference_navigation.mos")
    allocations = (Allocation(2, 1), Allocation(5, 1))
    output_dir = Path("results/test-meta-q-resumable")

    first = generate_resumable_meta_q_transitions(
        instance_seeds=(10, 11),
        output_dir=output_dir,
        source_allocations=allocations,
        allocations=allocations,
        max_steps=2,
        message_passing_iterations=1,
        instance_workers=1,
    )
    second = generate_resumable_meta_q_transitions(
        instance_seeds=(10, 11),
        output_dir=output_dir,
        source_allocations=allocations,
        allocations=allocations,
        max_steps=2,
        message_passing_iterations=1,
        instance_workers=1,
    )

    assert first.spatial.shape == second.spatial.shape
    assert (output_dir / "shards" / "instance-10" / "complete.json").is_file()
    assert (output_dir / "generation_manifest.json").is_file()
