import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("active_inference_navigation.mos")

from active_inference_neural_metacontrol import ALLOCATIONS, Allocation
from active_inference_neural_metacontrol.closed_loop import (
    ClosedLoopConfig,
    NeuralMetaController,
    run_adaptive_episode,
    run_fixed_episode,
)
from active_inference_neural_metacontrol.models import TaskPerformanceNetwork
from active_inference_neural_metacontrol.selector import AllocationDecision


def _checkpoint(path: Path) -> Path:
    model = TaskPerformanceNetwork(spatial_channels=6, context_features=16)
    torch.save(
        {
            "schema_version": 4,
            "prediction_target": "depth_normalized_selected_policy_g_at_t_plus_1",
            "model_state_dict": model.state_dict(),
            "spatial_channels": 6,
            "context_features": 16,
            "context_mean": np.zeros(16, dtype=np.float32),
            "context_scale": np.ones(16, dtype=np.float32),
            "compute_profile_ms": np.arange(1, len(ALLOCATIONS) + 1, dtype=np.float32),
            "switching_profile_ms": np.zeros(
                (len(ALLOCATIONS), len(ALLOCATIONS)), dtype=np.float32
            ),
        },
        path,
    )
    return path


def test_fixed_and_adaptive_episodes_execute_real_mos_steps():
    checkpoint = _checkpoint(Path("data/generated/test-closed-loop-model.pt"))
    config = ClosedLoopConfig(
        checkpoint=str(checkpoint),
        deadline_median_ms=100.0,
        initial_resolution=2,
        initial_depth=1,
        max_steps=2,
        message_passing_iterations=1,
        include_fixed=False,
    )

    fixed = run_fixed_episode(0, Allocation(2, 1), config)
    adaptive, trajectory = run_adaptive_episode(0, config)

    assert 1 <= fixed["steps"] <= 2
    assert 1 <= adaptive["steps"] <= 2
    assert len(trajectory) == adaptive["steps"]
    assert adaptive["total_compute_ms"] >= adaptive["task_inference_ms"]
    assert sum(json.loads(adaptive["allocation_counts"]).values()) == adaptive["steps"]
    assert json.loads(adaptive["allocation_sequence"]) == [
        f"g{row['source_resolution']}_T{row['source_depth']}" for row in trajectory
    ]


def test_depth_commitment_invokes_metacontroller_after_selected_physical_steps(monkeypatch):
    checkpoint = _checkpoint(Path("data/generated/test-depth-hold-model.pt"))
    selected = Allocation(2, 3)

    def choose(*args, **kwargs):
        return (
            AllocationDecision(
                allocation=selected,
                allocation_index=ALLOCATIONS.index(selected),
                predicted_normalized_g=0.0,
                predicted_compute_ms=1.0,
                switching_ms=0.0,
                mean_compute_ms=1.0 / selected.depth,
                    deadline_survival_probability=1.0,
                    compute_surprisal_nats=0.0,
                    compute_preference_cost_nats=0.0,
                    normalized_compute_cost=0.0,
                switching_penalty=0.0,
                information_loss_normalized=0.0,
                information_loss_nats=0.0,
                objective_score=0.0,
            ),
            0.0,
        )

    monkeypatch.setattr(NeuralMetaController, "choose", choose)
    config = ClosedLoopConfig(
        checkpoint=str(checkpoint),
        deadline_median_ms=100.0,
        initial_resolution=2,
        initial_depth=1,
        max_steps=5,
        message_passing_iterations=1,
        include_fixed=False,
        hold_allocation_for_depth=True,
    )

    adaptive, trajectory = run_adaptive_episode(0, config)

    expected_calls = (adaptive["steps"] + selected.depth - 1) // selected.depth
    assert adaptive["meta_decisions"] == expected_calls
    assert sum(row["metacontroller_invoked"] for row in trajectory) == expected_calls
    assert all(
        row["selection_reason"] == "held for selected planning depth"
        for row in trajectory
        if not row["metacontroller_invoked"]
    )
