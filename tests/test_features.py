import numpy as np
import pytest

from active_inference_neural_metacontrol import (
    Allocation,
    build_context_vector,
    build_spatial_features,
)


def test_spatial_features_have_fixed_six_channel_shape():
    posterior = np.ones(25) / 25
    detection = np.tile(np.linspace(0.1, 0.9, 20), (20, 1))
    features = build_spatial_features(
        posterior=posterior,
        predicted_posterior=posterior,
        resolution=5,
        next_robot_position=(4, 7),
        obstacle_map=np.zeros((20, 20)),
        likelihood_at_next_position=np.stack((1 - detection, detection)),
    )

    assert features.tensor.shape == (6, 20, 20)
    assert features.tensor[0].sum() == pytest.approx(1.0)
    assert features.tensor[2, 7, 4] == 1.0


def test_context_vector_contains_action_allocation_found_and_confidence():
    context = build_context_vector(
        action_index=2,
        action_count=6,
        allocation=Allocation(5, 2),
        found_flags=np.asarray([0]),
        posterior=np.ones(25),
        policy_posterior=np.asarray([0.1, 0.2, 0.7]),
        expected_free_energy=np.asarray([2.0, 1.0, 3.0]),
    )

    assert context.shape == (6 + 4 + 3 + 1 + 4,)
    assert np.all(np.isfinite(context))
