import numpy as np
import pytest

from active_inference_neural_metacontrol.fleet_throughput import (
    EpisodeTrace,
    FleetSimulationConfig,
    balanced_fleet_samples,
    simulate_fleet,
)


def _trace(seed: int, compute_s: float, *, steps: int = 2, success: bool = True):
    return EpisodeTrace(
        instance_seed=seed,
        controller="controller",
        decision_compute_s=(compute_s,) * steps,
        success=success,
        success_step=steps if success else None,
    )


def test_single_agent_finishes_target_before_mission_deadline():
    result = simulate_fleet(
        [_trace(1, 0.1)],
        FleetSimulationConfig(mission_time_s=2.0, planning_workers=1),
    )

    assert result["targets_found"] == 1
    assert result["completed_actions"] == 2
    assert result["deadline_misses"] == 0
    assert result["mean_queue_wait_s_per_request"] == pytest.approx(0.0)


def test_shared_worker_queue_can_prevent_target_completion():
    result = simulate_fleet(
        [_trace(1, 0.4), _trace(2, 0.4)],
        FleetSimulationConfig(mission_time_s=1.7, planning_workers=1),
    )

    assert result["targets_found"] == 0
    assert result["deadline_misses"] > 0
    assert result["mean_queue_wait_s_per_request"] > 0


def test_extra_worker_removes_initial_queueing():
    one_worker = simulate_fleet(
        [_trace(1, 0.2, steps=1), _trace(2, 0.2, steps=1)],
        FleetSimulationConfig(mission_time_s=1.0, planning_workers=1),
    )
    two_workers = simulate_fleet(
        [_trace(1, 0.2, steps=1), _trace(2, 0.2, steps=1)],
        FleetSimulationConfig(mission_time_s=1.0, planning_workers=2),
    )

    assert one_worker["mean_queue_wait_s_per_request"] == pytest.approx(0.1)
    assert two_workers["mean_queue_wait_s_per_request"] == pytest.approx(0.0)
    assert two_workers["targets_found"] == 2


def test_balanced_fleet_samples_use_every_instance_equally():
    samples = balanced_fleet_samples(
        np.arange(10),
        fleet_size=4,
        repeats=10,
        rng=np.random.default_rng(7),
    )
    counts = np.bincount(np.asarray(samples).ravel(), minlength=10)

    assert len(samples) == 10
    assert all(len(set(sample)) == 4 for sample in samples)
    assert counts.tolist() == [4] * 10
