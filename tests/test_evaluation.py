import numpy as np

from active_inference_neural_metacontrol.datasets import CounterfactualArrays
from active_inference_neural_metacontrol.training import evaluate_predictions


def test_evaluation_uses_realized_labels_and_reports_oracle_regret():
    samples = 2
    task_cost = np.full((samples, 12), 10.0, dtype=np.float32)
    task_cost[0, 0] = 1.0
    task_cost[1, 1] = 2.0
    arrays = CounterfactualArrays(
        spatial=np.zeros((samples, 6, 20, 20), dtype=np.float32),
        context=np.zeros((samples, 17), dtype=np.float32),
        success=np.ones((samples, 12), dtype=np.float32),
        task_cost=task_cost,
        compute_ms=np.ones((samples, 12), dtype=np.float32),
        switch_ms=np.zeros((samples, 12), dtype=np.float32),
        candidate_actions=np.zeros((samples, 12), dtype=np.int8),
        context_ids=np.asarray(["a", "b"]),
        instance_seeds=np.asarray([0, 1]),
        source_resolution=np.full(samples, 20),
        source_depth=np.ones(samples, dtype=int),
    )
    prediction = task_cost.copy()

    metrics = evaluate_predictions(
        arrays,
        success_probability=np.full((samples, 12), 0.9),
        relative_cost_prediction=prediction - prediction.min(axis=1, keepdims=True),
        compute_profile_ms=np.ones(12),
        success_threshold=0.5,
        compute_budget_ms=None,
    )

    assert metrics["selected_mean_task_cost"] == 1.5
    assert metrics["oracle_mean_task_cost"] == 1.5
    assert metrics["mean_regret_to_oracle"] == 0.0
    assert metrics["selection_counts"]["g2_T1"] == 1
    assert metrics["selection_counts"]["g2_T2"] == 1
