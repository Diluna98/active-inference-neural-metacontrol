import pytest

torch = pytest.importorskip("torch")

from active_inference_neural_metacontrol.models import (
    DecomposedTaskUtilityNetwork,
    ImmediateImprovementValueNetwork,
    InferenceValueNetwork,
    MultiObjectiveNetwork,
    ResolutionSharedFourTermValueNetwork,
    TaskPerformanceNetwork,
    TaskUtilityNetwork,
    task_performance_loss,
)


def test_task_performance_network_predicts_all_allocations_and_backpropagates():
    model = TaskPerformanceNetwork(spatial_channels=6, context_features=18)
    prediction = model(
        torch.rand(4, 6, 20, 20),
        torch.rand(4, 18),
    )

    assert prediction["normalized_g"].shape == (4, 12)

    loss = task_performance_loss(
        prediction,
        normalized_g_target=torch.randn(4, 12) * 10,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_multiobjective_network_exposes_all_candidate_heads():
    model = MultiObjectiveNetwork(spatial_channels=6, context_features=16)
    prediction = model(torch.rand(3, 6, 20, 20), torch.rand(3, 16))

    assert set(prediction) == set(MultiObjectiveNetwork.OUTPUTS)
    assert all(values.shape == (3, 12) for values in prediction.values())


def test_task_utility_network_exposes_three_candidate_heads():
    model = TaskUtilityNetwork(spatial_channels=6, context_features=16)
    prediction = model(torch.rand(3, 6, 20, 20), torch.rand(3, 16))

    assert set(prediction) == {
        "success_logits",
        "operational_cost_scaled",
        "normalized_g_scaled",
    }
    assert all(values.shape == (3, 12) for values in prediction.values())


def test_decomposed_task_utility_network_exposes_four_candidate_heads():
    model = DecomposedTaskUtilityNetwork()
    prediction = model(torch.rand(3, 5, 20, 20), torch.rand(3, 14))

    assert set(prediction) == {
        "success_logits",
        "operational_cost_scaled",
        "preference_per_depth_scaled",
        "epistemic_per_depth_scaled",
    }
    assert all(values.shape == (3, 12) for values in prediction.values())


def test_inference_value_network_exposes_only_two_candidate_heads():
    model = InferenceValueNetwork()
    prediction = model(torch.rand(3, 5, 20, 20), torch.rand(3, 14))

    assert set(prediction) == {
        "preference_per_depth_scaled",
        "epistemic_per_depth_scaled",
    }
    assert all(values.shape == (3, 12) for values in prediction.values())


def test_resolution_shared_four_term_network_shares_state_terms_across_depth():
    model = ResolutionSharedFourTermValueNetwork()
    prediction = model(torch.rand(3, 5, 20, 20), torch.rand(3, 14))

    assert all(values.shape == (3, 12) for values in prediction.values())
    for name in ("state_accuracy_scaled", "state_complexity_scaled"):
        values = prediction[name].reshape(3, 4, 3)
        assert torch.equal(values[:, :, 0], values[:, :, 1])
        assert torch.equal(values[:, :, 1], values[:, :, 2])

    # Prospective policy-value heads remain independently parameterized for all
    # twelve joint (resolution, depth) choices.
    assert model.preference_head.out_features == 12
    assert model.epistemic_value_head.out_features == 12
    assert model.state_accuracy_head.out_features == 4
    assert model.state_complexity_head.out_features == 4


def test_immediate_improvement_network_exposes_six_candidate_heads():
    model = ImmediateImprovementValueNetwork()
    prediction = model(torch.rand(3, 5, 20, 20), torch.rand(3, 14))

    assert set(prediction) == {
        "state_accuracy_scaled",
        "state_complexity_scaled",
        "immediate_preference_scaled",
        "preference_improvement_scaled",
        "immediate_epistemic_scaled",
        "epistemic_improvement_scaled",
    }
    assert all(values.shape == (3, 12) for values in prediction.values())
