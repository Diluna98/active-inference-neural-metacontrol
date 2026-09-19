from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from active_inference_neural_metacontrol.datasets import CounterfactualArrays, split_by_instance
from active_inference_neural_metacontrol.training import TrainingConfig, train_task_model


def test_training_pipeline_writes_checkpoint_and_metrics():
    rng = np.random.default_rng(4)
    samples = 12
    arrays = CounterfactualArrays(
        spatial=rng.random((samples, 6, 20, 20), dtype=np.float32),
        context=rng.random((samples, 17), dtype=np.float32),
        success=rng.integers(0, 2, (samples, 12)).astype(np.float32),
        task_cost=(1 + 20 * rng.random((samples, 12))).astype(np.float32),
        compute_ms=(1 + rng.random((samples, 12))).astype(np.float32),
        switch_ms=rng.random((samples, 12)).astype(np.float32),
        candidate_actions=rng.integers(0, 5, (samples, 12), dtype=np.int8),
        context_ids=np.asarray([f"context-{index}" for index in range(samples)]),
        instance_seeds=np.repeat(np.arange(6), 2),
        source_resolution=np.full(samples, 20),
        source_depth=np.ones(samples, dtype=int),
    )
    split = split_by_instance(arrays, seed=1)
    output = Path("data/generated/test-training")

    report = train_task_model(
        split,
        output_dir=output,
        config=TrainingConfig(epochs=2, batch_size=4, patience=2, seed=3, device="cpu"),
    )

    assert report["epochs_completed"] == 2
    assert report["evaluation"]["test"]["samples"] == len(split.test.context)
    assert (output / "best_model.pt").is_file()
    assert (output / "metrics.json").is_file()
    assert (output / "history.csv").is_file()
    assert (output / "predictions.npz").is_file()
