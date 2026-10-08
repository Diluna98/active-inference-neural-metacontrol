"""Validate the bundled controller against a frozen reference episode."""

from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("active_inference_navigation.mos")

from active_inference_neural_metacontrol.meta_q_runtime import (
    MetaQEvaluationConfig,
    run_meta_q_episode,
)

ROOT = Path(__file__).resolve().parents[1]


def test_pretrained_ensemble_reproduces_reference_episode():
    checkpoints = ROOT / "artifacts/checkpoints"
    config = MetaQEvaluationConfig(
        checkpoint=checkpoints / "seed0.pt",
        ensemble_checkpoints=(checkpoints / "seed1.pt", checkpoints / "seed2.pt"),
        selection_mode="joint",
        compute_price=0.001,
        switch_cost_ms=50,
        switching_weight=1,
        switch_penalty_mode="allocation",
        information_loss_weight=0.8,
        uncertainty_beta=0.5,
        torch_threads=1,
    )
    summary, trajectory = run_meta_q_episode(20088, config)
    assert summary["success"] is True
    assert summary["steps"] == 41
    assert summary["task_cost"] == 46
    assert len(trajectory) == 41
