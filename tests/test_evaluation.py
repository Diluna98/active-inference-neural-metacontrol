import numpy as np

from active_inference_neural_metacontrol.datasets import CounterfactualArrays
from active_inference_neural_metacontrol.training import evaluate_predictions


def test_evaluation_uses_realized_labels_and_reports_oracle_regret():
    samples = 2
    normalized_g = np.zeros((samples, 12), dtype=np.float32)
    normalized_g[0, 0] = 10.0
    normalized_g[1, 1] = 8.0
    arrays = CounterfactualArrays(
        spatial=np.zeros((samples, 6, 20, 20), dtype=np.float32),
        context=np.zeros((samples, 16), dtype=np.float32),
        normalized_g=normalized_g,
        raw_g=normalized_g.copy(),
        risk=normalized_g.copy(),
        ambiguity=np.zeros((samples, 12), dtype=np.float32),
        information_gain=np.zeros((samples, 12), dtype=np.float32),
        compute_ms=np.ones((samples, 12), dtype=np.float32),
        switch_ms=np.zeros((samples, 12), dtype=np.float32),
        candidate_actions=np.zeros((samples, 12), dtype=np.int8),
        context_ids=np.asarray(["a", "b"]),
        instance_seeds=np.asarray([0, 1]),
        source_resolution=np.full(samples, 20),
        source_depth=np.ones(samples, dtype=int),
    )
    prediction = normalized_g.copy()

    metrics = evaluate_predictions(
        arrays,
        normalized_g_prediction=prediction,
        compute_profile_ms=np.ones(12),
        compute_budget_ms=None,
    )

    assert metrics["selected_mean_normalized_g"] == 9.0
    assert metrics["oracle_mean_normalized_g"] == 9.0
    assert metrics["mean_normalized_g_regret_to_oracle"] == 0.0
    assert metrics["selection_counts"]["g2_T1"] == 1
    assert metrics["selection_counts"]["g2_T2"] == 1
