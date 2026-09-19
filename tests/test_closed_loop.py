import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("active_inference_navigation.mos")

from active_inference_neural_metacontrol import ALLOCATIONS, Allocation
from active_inference_neural_metacontrol.closed_loop import (
    ClosedLoopConfig,
    run_adaptive_episode,
    run_fixed_episode,
)
from active_inference_neural_metacontrol.models import TaskPerformanceNetwork


def _checkpoint(path: Path) -> Path:
    model = TaskPerformanceNetwork(spatial_channels=6, context_features=17)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "spatial_channels": 6,
            "context_features": 17,
            "context_mean": np.zeros(17, dtype=np.float32),
            "context_scale": np.ones(17, dtype=np.float32),
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
        initial_resolution=2,
        initial_depth=1,
        max_steps=2,
        message_passing_iterations=1,
        compute_budget_ms=20.0,
        information_loss_limit=1.0,
        include_fixed=False,
    )

    fixed = run_fixed_episode(0, Allocation(2, 1), config)
    adaptive, trajectory = run_adaptive_episode(0, config)

    assert 1 <= fixed["steps"] <= 2
    assert 1 <= adaptive["steps"] <= 2
    assert len(trajectory) == adaptive["steps"]
    assert adaptive["total_compute_ms"] >= adaptive["task_inference_ms"]
    assert sum(json.loads(adaptive["allocation_counts"]).values()) == adaptive["steps"]
