import numpy as np
import pytest

from active_inference_neural_metacontrol import (
    Allocation,
    build_context_vector,
    build_spatial_features,
)
from active_inference_neural_metacontrol.features import (
    FOUR_TERM_ABLATION_FEATURE_SCHEMA,
    FOUR_TERM_NO_ALLOCATION_FEATURE_SCHEMA,
    project_decomposed_features,
    project_feature_schema,
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
    )

    assert context.shape == (6 + 4 + 3 + 1 + 3,)
    assert np.all(np.isfinite(context))


def test_decomposed_projection_removes_requested_inputs():
    spatial = np.arange(6 * 20 * 20).reshape(6, 20, 20)
    context = np.arange(16)

    reduced_spatial, reduced_context = project_decomposed_features(spatial, context)

    assert reduced_spatial.shape == (5, 20, 20)
    assert reduced_context.shape == (14,)
    assert np.array_equal(reduced_spatial[:, 0, 0], spatial[[0, 1, 2, 3, 5], 0, 0])
    assert np.array_equal(reduced_context, context[[*range(13), 14]])


def test_four_term_ablation_removes_current_posterior_action_and_policy_entropy():
    spatial = np.arange(6 * 20 * 20).reshape(6, 20, 20)
    context = np.arange(16)

    reduced_spatial, reduced_context = project_feature_schema(
        spatial, context, FOUR_TERM_ABLATION_FEATURE_SCHEMA
    )

    assert np.array_equal(reduced_spatial, spatial[[1, 2, 3, 5]])
    assert np.array_equal(reduced_context, context[5:13])


def test_four_term_no_allocation_ablation_keeps_only_found_context():
    spatial = np.arange(6 * 20 * 20).reshape(6, 20, 20)
    context = np.arange(16)

    reduced_spatial, reduced_context = project_feature_schema(
        spatial, context, FOUR_TERM_NO_ALLOCATION_FEATURE_SCHEMA
    )

    assert np.array_equal(reduced_spatial, spatial[[1, 2, 3, 5]])
    assert np.array_equal(reduced_context, context[[12]])


def test_predicted_observation_schema_adds_pre_observation_distribution():
    from active_inference_neural_metacontrol.features import (
        FOUR_TERM_PREDICTED_OBSERVATION_FEATURE_SCHEMA,
    )

    spatial = np.zeros((6, 20, 20), dtype=np.float32)
    spatial[1] = 1.0 / 400.0
    spatial[4] = 0.25
    context = np.arange(16, dtype=np.float32)

    reduced_spatial, reduced_context = project_feature_schema(
        spatial, context, FOUR_TERM_PREDICTED_OBSERVATION_FEATURE_SCHEMA
    )

    assert np.array_equal(reduced_spatial, spatial[[1, 2, 3, 5]])
    assert np.allclose(reduced_context, [12.0, 0.75, 0.25])
