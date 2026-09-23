from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from active_inference_neural_metacontrol.allocations import ALLOCATIONS
from active_inference_neural_metacontrol.datasets import CounterfactualArrays, split_by_instance
from active_inference_neural_metacontrol.four_term_training import (
    FourTermTrainingConfig,
    _ranking_utility,
    train_four_term_model,
)
from active_inference_neural_metacontrol.four_term_training import (
    _targets as four_term_targets,
)
from active_inference_neural_metacontrol.inference_value_training import (
    InferenceValueTrainingConfig,
    train_inference_value_model,
)
from active_inference_neural_metacontrol.multiobjective_training import (
    MultiObjectiveTrainingConfig,
    train_multiobjective_model,
)
from active_inference_neural_metacontrol.task_utility_training import (
    TaskUtilityTrainingConfig,
    train_task_utility_model,
)
from active_inference_neural_metacontrol.training import TrainingConfig, train_task_model


def test_four_term_epistemic_label_uses_ambiguity_minus_information_gain():
    from types import SimpleNamespace

    arrays = SimpleNamespace(
        state_accuracy=np.zeros((1, 12), dtype=np.float32),
        state_complexity=np.zeros((1, 12), dtype=np.float32),
        risk=np.zeros((1, 12), dtype=np.float32),
        ambiguity=np.full((1, 12), 3.0, dtype=np.float32),
        information_gain=np.full((1, 12), 1.0, dtype=np.float32),
    )

    targets = four_term_targets(arrays, "selected")
    depths = np.asarray([allocation.depth for allocation in ALLOCATIONS])
    expected = 2.0 / depths
    assert np.allclose(targets["epistemic_value_per_depth"], expected - expected.mean())


def test_four_term_targets_accept_posterior_policy_aggregation():
    from types import SimpleNamespace

    arrays = SimpleNamespace(
        state_accuracy=np.zeros((1, 12), dtype=np.float32),
        state_complexity=np.zeros((1, 12), dtype=np.float32),
        posterior_risk=np.full((1, 12), -4.0, dtype=np.float32),
        posterior_ambiguity=np.full((1, 12), 3.0, dtype=np.float32),
        posterior_information_gain=np.full((1, 12), 1.0, dtype=np.float32),
    )

    targets = four_term_targets(arrays, "posterior")
    depths = np.asarray([allocation.depth for allocation in ALLOCATIONS])
    preference = -4.0 / depths
    epistemic = 2.0 / depths
    assert np.allclose(targets["preference_per_depth"], preference - preference.mean())
    assert np.allclose(targets["epistemic_value_per_depth"], epistemic - epistemic.mean())


def test_four_term_state_targets_are_shared_across_depth():
    from types import SimpleNamespace

    accuracy = np.arange(12, dtype=np.float32)[None, :]
    complexity = (100 + np.arange(12, dtype=np.float32))[None, :]
    arrays = SimpleNamespace(
        state_accuracy=accuracy,
        state_complexity=complexity,
        risk=np.zeros((1, 12), dtype=np.float32),
        ambiguity=np.zeros((1, 12), dtype=np.float32),
        information_gain=np.zeros((1, 12), dtype=np.float32),
    )

    targets = four_term_targets(arrays, "selected")
    expected_accuracy = np.repeat(accuracy.reshape(1, 4, 3).mean(axis=2), 3, axis=1)
    expected_complexity = np.repeat(complexity.reshape(1, 4, 3).mean(axis=2), 3, axis=1)
    assert np.allclose(targets["state_accuracy"], expected_accuracy)
    assert np.allclose(targets["state_complexity"], expected_complexity)


def test_immediate_improvement_targets_preserve_zero_shallow_improvement():
    from types import SimpleNamespace

    zeros = np.zeros((1, 12), dtype=np.float32)
    immediate = np.arange(12, dtype=np.float32)[None, :]
    improvement = np.tile(np.asarray([0.0, 1.0, 2.0], dtype=np.float32), 4)[None, :]
    arrays = SimpleNamespace(
        state_accuracy=zeros,
        state_complexity=zeros,
        posterior_immediate_preference=immediate,
        posterior_immediate_epistemic=immediate,
        posterior_immediate_information_gain=zeros,
        posterior_improvement_preference=improvement,
        posterior_improvement_epistemic=improvement,
        posterior_improvement_information_gain=zeros,
    )

    targets = four_term_targets(arrays, "posterior", "immediate_improvement")
    assert np.allclose(targets["preference_improvement"][:, ::3], 0.0)
    assert np.allclose(targets["epistemic_improvement"][:, ::3], 0.0)


def test_ranking_utility_detaches_state_term_gradients():
    values = {
        name: torch.ones((1, 12), requires_grad=True)
        for name in (
            "state_accuracy",
            "state_complexity",
            "preference_per_depth",
            "epistemic_value_per_depth",
        )
    }

    _ranking_utility(values, detach_state_terms=True).sum().backward()

    assert values["state_accuracy"].grad is None
    assert values["state_complexity"].grad is None
    assert values["preference_per_depth"].grad is not None
    assert values["epistemic_value_per_depth"].grad is not None


def test_four_term_training_writes_resolution_shared_checkpoint():
    rng = np.random.default_rng(11)
    samples = 12
    state_accuracy = rng.standard_normal((samples, 4)).repeat(3, axis=1).astype(np.float32)
    state_complexity = rng.random((samples, 4)).repeat(3, axis=1).astype(np.float32)
    arrays = CounterfactualArrays(
        spatial=rng.random((samples, 6, 20, 20), dtype=np.float32),
        context=rng.random((samples, 16), dtype=np.float32),
        normalized_g=rng.standard_normal((samples, 12)).astype(np.float32),
        raw_g=rng.standard_normal((samples, 12)).astype(np.float32),
        risk=rng.standard_normal((samples, 12)).astype(np.float32),
        ambiguity=rng.random((samples, 12)).astype(np.float32),
        information_gain=rng.random((samples, 12)).astype(np.float32),
        compute_ms=(1 + rng.random((samples, 12))).astype(np.float32),
        switch_ms=rng.random((samples, 12)).astype(np.float32),
        candidate_actions=rng.integers(0, 5, (samples, 12), dtype=np.int8),
        context_ids=np.asarray([f"four-term-{index}" for index in range(samples)]),
        instance_seeds=np.repeat(np.arange(6), 2),
        source_resolution=np.full(samples, 20),
        source_depth=np.ones(samples, dtype=int),
        state_accuracy=state_accuracy,
        state_complexity=state_complexity,
    )
    output = Path("data/generated/test-four-term-training")
    report = train_four_term_model(
        split_by_instance(arrays, seed=1),
        output_dir=output,
        config=FourTermTrainingConfig(
            epochs=2, batch_size=4, patience=2, seed=3, device="cpu"
        ),
    )

    checkpoint = torch.load(output / "best_model.pt", weights_only=False)
    assert report["schema_version"] == 14
    assert checkpoint["schema_version"] == 14
    assert checkpoint["model_type"] == "state_detached_candidate_relative_four_term_value"
    assert checkpoint["model_state_dict"]["state_accuracy_head.weight"].shape[0] == 4
    assert checkpoint["model_state_dict"]["state_complexity_head.weight"].shape[0] == 4


def test_training_pipeline_writes_checkpoint_and_metrics():
    rng = np.random.default_rng(4)
    samples = 12
    arrays = CounterfactualArrays(
        spatial=rng.random((samples, 6, 20, 20), dtype=np.float32),
        context=rng.random((samples, 16), dtype=np.float32),
        normalized_g=(10 * rng.standard_normal((samples, 12))).astype(np.float32),
        raw_g=(10 * rng.standard_normal((samples, 12))).astype(np.float32),
        risk=rng.standard_normal((samples, 12)).astype(np.float32),
        ambiguity=rng.random((samples, 12)).astype(np.float32),
        information_gain=np.zeros((samples, 12), dtype=np.float32),
        compute_ms=(1 + rng.random((samples, 12))).astype(np.float32),
        switch_ms=rng.random((samples, 12)).astype(np.float32),
        candidate_actions=rng.integers(0, 5, (samples, 12), dtype=np.int8),
        context_ids=np.asarray([f"context-{index}" for index in range(samples)]),
        instance_seeds=np.repeat(np.arange(6), 2),
        source_resolution=np.full(samples, 20),
        source_depth=np.ones(samples, dtype=int),
        label_schema="candidate-controlled-accumulated-normalized-g-v5",
        rollout_horizon=5,
    )
    split = split_by_instance(arrays, seed=1)
    output = Path("data/generated/test-training")

    report = train_task_model(
        split,
        output_dir=output,
        config=TrainingConfig(epochs=2, batch_size=4, patience=2, seed=3, device="cpu"),
    )

    assert report["epochs_completed"] == 2
    assert report["schema_version"] == 5
    assert report["rollout_horizon"] == 5
    assert report["evaluation"]["test"]["samples"] == len(split.test.context)
    assert (output / "best_model.pt").is_file()
    assert (output / "metrics.json").is_file()
    assert (output / "history.csv").is_file()
    assert (output / "predictions.npz").is_file()
    checkpoint = torch.load(output / "best_model.pt", weights_only=False)
    assert checkpoint["schema_version"] == 5
    assert checkpoint["rollout_horizon"] == 5


def test_multiobjective_training_writes_seven_head_checkpoint():
    rng = np.random.default_rng(8)
    samples = 12
    arrays = CounterfactualArrays(
        spatial=rng.random((samples, 6, 20, 20), dtype=np.float32),
        context=rng.random((samples, 16), dtype=np.float32),
        normalized_g=rng.standard_normal((samples, 12)).astype(np.float32),
        raw_g=rng.standard_normal((samples, 12)).astype(np.float32),
        risk=rng.standard_normal((samples, 12)).astype(np.float32),
        ambiguity=rng.random((samples, 12)).astype(np.float32),
        information_gain=np.zeros((samples, 12), dtype=np.float32),
        compute_ms=(1 + rng.random((samples, 12))).astype(np.float32),
        switch_ms=rng.random((samples, 12)).astype(np.float32),
        candidate_actions=rng.integers(0, 5, (samples, 12), dtype=np.int8),
        context_ids=np.asarray([f"multi-{index}" for index in range(samples)]),
        instance_seeds=np.repeat(np.arange(6), 2),
        source_resolution=np.full(samples, 20),
        source_depth=np.ones(samples, dtype=int),
        success=rng.integers(0, 2, (samples, 12)).astype(np.float32),
        task_cost=(1 + 20 * rng.random((samples, 12))).astype(np.float32),
        task_outcome_schema="test",
    )
    output = Path("data/generated/test-multiobjective-training")
    report = train_multiobjective_model(
        split_by_instance(arrays, seed=1),
        output_dir=output,
        config=MultiObjectiveTrainingConfig(
            epochs=2, batch_size=4, patience=2, seed=3, device="cpu"
        ),
    )

    checkpoint = torch.load(output / "best_model.pt", weights_only=False)
    assert report["model_type"] == "multiobjective"
    assert checkpoint["schema_version"] == 6
    assert len(checkpoint["prediction_targets"]) == 7


def test_task_utility_training_writes_three_head_checkpoint():
    rng = np.random.default_rng(9)
    samples = 12
    success = rng.integers(0, 2, (samples, 12)).astype(np.float32)
    operational = (1 + 20 * rng.random((samples, 12))).astype(np.float32)
    arrays = CounterfactualArrays(
        spatial=rng.random((samples, 6, 20, 20), dtype=np.float32),
        context=rng.random((samples, 16), dtype=np.float32),
        normalized_g=rng.standard_normal((samples, 12)).astype(np.float32),
        raw_g=rng.standard_normal((samples, 12)).astype(np.float32),
        risk=rng.standard_normal((samples, 12)).astype(np.float32),
        ambiguity=rng.random((samples, 12)).astype(np.float32),
        information_gain=np.zeros((samples, 12), dtype=np.float32),
        compute_ms=(1 + rng.random((samples, 12))).astype(np.float32),
        switch_ms=rng.random((samples, 12)).astype(np.float32),
        candidate_actions=rng.integers(0, 5, (samples, 12), dtype=np.int8),
        context_ids=np.asarray([f"utility-{index}" for index in range(samples)]),
        instance_seeds=np.repeat(np.arange(6), 2),
        source_resolution=np.full(samples, 20),
        source_depth=np.ones(samples, dtype=int),
        success=success,
        task_cost=operational + 100.0 * (1.0 - success),
        task_outcome_schema="test",
    )
    output = Path("data/generated/test-task-utility-training")
    report = train_task_utility_model(
        split_by_instance(arrays, seed=1),
        output_dir=output,
        config=TaskUtilityTrainingConfig(epochs=2, batch_size=4, patience=2, seed=3, device="cpu"),
    )

    checkpoint = torch.load(output / "best_model.pt", weights_only=False)
    assert report["model_type"] == "task_utility"
    assert checkpoint["schema_version"] == 7
    assert checkpoint["prediction_targets"] == [
        "success",
        "operational_remaining_cost",
        "normalized_g",
    ]


def test_inference_value_training_does_not_require_task_outcome_labels():
    rng = np.random.default_rng(10)
    samples = 12
    arrays = CounterfactualArrays(
        spatial=rng.random((samples, 6, 20, 20), dtype=np.float32),
        context=rng.random((samples, 16), dtype=np.float32),
        normalized_g=rng.standard_normal((samples, 12)).astype(np.float32),
        raw_g=rng.standard_normal((samples, 12)).astype(np.float32),
        risk=rng.standard_normal((samples, 12)).astype(np.float32),
        ambiguity=rng.random((samples, 12)).astype(np.float32),
        information_gain=np.zeros((samples, 12), dtype=np.float32),
        compute_ms=(1 + rng.random((samples, 12))).astype(np.float32),
        switch_ms=rng.random((samples, 12)).astype(np.float32),
        candidate_actions=rng.integers(0, 5, (samples, 12), dtype=np.int8),
        context_ids=np.asarray([f"value-{index}" for index in range(samples)]),
        instance_seeds=np.repeat(np.arange(6), 2),
        source_resolution=np.full(samples, 20),
        source_depth=np.ones(samples, dtype=int),
    )
    output = Path("data/generated/test-inference-value-training")
    report = train_inference_value_model(
        split_by_instance(arrays, seed=1),
        output_dir=output,
        config=InferenceValueTrainingConfig(
            epochs=2, batch_size=4, patience=2, seed=3, device="cpu"
        ),
    )

    checkpoint = torch.load(output / "best_model.pt", weights_only=False)
    assert report["uses_success_labels"] is False
    assert report["uses_operational_cost_labels"] is False
    assert checkpoint["schema_version"] == 9
    assert checkpoint["prediction_targets"] == [
        "preference_per_depth",
        "epistemic_per_depth",
    ]
