import pytest

torch = pytest.importorskip("torch")

from active_inference_neural_metacontrol.models import (
    TaskPerformanceNetwork,
    task_performance_loss,
)


def test_task_performance_network_predicts_all_allocations_and_backpropagates():
    model = TaskPerformanceNetwork(spatial_channels=6, context_features=18)
    prediction = model(
        torch.rand(4, 6, 20, 20),
        torch.rand(4, 18),
    )

    assert prediction["success_logits"].shape == (4, 12)
    assert prediction["relative_cost"].shape == (4, 12)
    assert torch.all(prediction["relative_cost"] >= 0)

    loss = task_performance_loss(
        prediction,
        success_target=torch.randint(0, 2, (4, 12), dtype=torch.float32),
        task_cost_target=torch.rand(4, 12) * 100,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in model.parameters())
