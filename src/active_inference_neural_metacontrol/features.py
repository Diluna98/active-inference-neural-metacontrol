"""Feature assembly for the neural task-performance model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .allocations import DEPTHS, RESOLUTIONS, Allocation
from .beliefs import canonicalize_posterior, normalized_entropy
from .information import categorical_fisher_map, detection_probability_map

DECOMPOSED_FEATURE_SCHEMA = "mos-decomposed-v1"
DECOMPOSED_SPATIAL_INDICES = (0, 1, 2, 3, 5)
DECOMPOSED_CONTEXT_INDICES = (*range(13), 14)
FOUR_TERM_ABLATION_FEATURE_SCHEMA = "mos-four-term-no-current-action-policy-entropy-v1"
FOUR_TERM_ABLATION_SPATIAL_INDICES = (1, 2, 3, 5)
FOUR_TERM_ABLATION_CONTEXT_INDICES = (*range(5, 13),)
FOUR_TERM_NO_ALLOCATION_FEATURE_SCHEMA = "mos-four-term-predicted-context-only-v1"
FOUR_TERM_NO_ALLOCATION_SPATIAL_INDICES = FOUR_TERM_ABLATION_SPATIAL_INDICES
FOUR_TERM_NO_ALLOCATION_CONTEXT_INDICES = (12,)
FOUR_TERM_PREDICTED_OBSERVATION_FEATURE_SCHEMA = (
    "mos-four-term-predicted-context-observation-v1"
)


@dataclass(frozen=True)
class SpatialFeatures:
    """Fixed-size spatial tensor and named channel order."""

    tensor: np.ndarray
    channel_names: tuple[str, ...]

    def __post_init__(self) -> None:
        tensor = np.asarray(self.tensor, dtype=np.float32)
        if tensor.ndim != 3 or tensor.shape[0] != len(self.channel_names):
            raise ValueError("tensor must have shape (channels, height, width)")
        if tensor.shape[1] != tensor.shape[2]:
            raise ValueError("spatial features must use a square canonical grid")
        if not np.all(np.isfinite(tensor)):
            raise ValueError("spatial features must be finite")
        object.__setattr__(self, "tensor", tensor)


def build_spatial_features(
    *,
    posterior: np.ndarray,
    predicted_posterior: np.ndarray,
    resolution: int,
    next_robot_position: tuple[int, int],
    obstacle_map: np.ndarray,
    likelihood_at_next_position: np.ndarray,
    canonical_size: int = 20,
) -> SpatialFeatures:
    """Build canonical posterior, geometry, visibility, and Fisher channels."""

    current = canonicalize_posterior(posterior, resolution, canonical_size)
    predicted = canonicalize_posterior(predicted_posterior, resolution, canonical_size)
    obstacles = np.asarray(obstacle_map, dtype=float)
    if obstacles.shape != (canonical_size, canonical_size):
        raise ValueError("obstacle_map must match the canonical grid")
    if np.any((obstacles < 0) | (obstacles > 1)):
        raise ValueError("obstacle_map values must lie in [0, 1]")
    x, y = next_robot_position
    if not (0 <= x < canonical_size and 0 <= y < canonical_size):
        raise ValueError("next_robot_position is outside the canonical grid")
    robot = np.zeros((canonical_size, canonical_size), dtype=float)
    robot[y, x] = 1.0
    visibility = detection_probability_map(likelihood_at_next_position)
    fisher = categorical_fisher_map(likelihood_at_next_position)
    expected_shape = (canonical_size, canonical_size)
    if visibility.shape != expected_shape:
        raise ValueError("likelihood spatial dimensions must match the canonical grid")
    tensor = np.stack((current, predicted, robot, obstacles, visibility, fisher)).astype(np.float32)
    return SpatialFeatures(
        tensor=tensor,
        channel_names=(
            "posterior",
            "predicted_posterior",
            "next_robot_position",
            "obstacles",
            "visibility",
            "fisher_information",
        ),
    )


def _one_hot(value: int, choices: tuple[int, ...]) -> np.ndarray:
    if value not in choices:
        raise ValueError(f"value must be one of {choices}")
    result = np.zeros(len(choices), dtype=np.float32)
    result[choices.index(value)] = 1.0
    return result


def build_context_vector(
    *,
    action_index: int,
    action_count: int,
    allocation: Allocation,
    found_flags: np.ndarray,
    posterior: np.ndarray,
    policy_posterior: np.ndarray,
) -> np.ndarray:
    """Build nonspatial action, allocation, object, and policy-confidence features."""

    if not 0 <= action_index < action_count:
        raise ValueError("action_index is outside the action space")
    action = np.zeros(action_count, dtype=np.float32)
    action[action_index] = 1.0
    found = np.asarray(found_flags, dtype=np.float32).ravel()
    if found.size == 0 or np.any((found < 0) | (found > 1)):
        raise ValueError("found_flags must contain at least one value in [0, 1]")
    policies = np.asarray(policy_posterior, dtype=float).ravel()
    if policies.size < 1:
        raise ValueError("policy posterior must be nonempty")
    if np.any(policies < 0) or not np.all(np.isfinite(policies)) or policies.sum() <= 0:
        raise ValueError("policy posterior must be finite, nonnegative, and normalized")
    policies = policies / policies.sum()
    sorted_probabilities = np.sort(policies)
    probability_margin = float(
        sorted_probabilities[-1] - (sorted_probabilities[-2] if policies.size > 1 else 0.0)
    )
    positive = policies > 0
    policy_entropy = float(-np.sum(policies[positive] * np.log(policies[positive])))
    if policies.size > 1:
        policy_entropy /= float(np.log(policies.size))
    confidence = np.asarray(
        [
            normalized_entropy(posterior, allocation.resolution),
            policy_entropy,
            probability_margin,
        ],
        dtype=np.float32,
    )
    return np.concatenate(
        (
            action,
            _one_hot(allocation.resolution, RESOLUTIONS),
            _one_hot(allocation.depth, DEPTHS),
            found,
            confidence,
        )
    )


def project_decomposed_features(
    spatial: np.ndarray,
    context: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove visibility, posterior entropy, and policy-margin inputs.

    Supports both one context and batched dataset arrays. The retained policy
    entropy is useful evidence about ambiguity in the task-level decision.
    """

    spatial_values = np.asarray(spatial)
    context_values = np.asarray(context)
    if spatial_values.ndim not in {3, 4} or spatial_values.shape[-3] != 6:
        raise ValueError("spatial must contain the six canonical MOS channels")
    if context_values.ndim not in {1, 2} or context_values.shape[-1] != 16:
        raise ValueError("context must contain the sixteen canonical MOS features")
    return (
        np.take(spatial_values, DECOMPOSED_SPATIAL_INDICES, axis=-3),
        np.take(context_values, DECOMPOSED_CONTEXT_INDICES, axis=-1),
    )


def project_feature_schema(
    spatial: np.ndarray,
    context: np.ndarray,
    feature_schema: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Project canonical MOS inputs according to a checkpointed feature schema."""

    if feature_schema == DECOMPOSED_FEATURE_SCHEMA:
        return project_decomposed_features(spatial, context)
    if feature_schema not in {
        FOUR_TERM_ABLATION_FEATURE_SCHEMA,
        FOUR_TERM_NO_ALLOCATION_FEATURE_SCHEMA,
        FOUR_TERM_PREDICTED_OBSERVATION_FEATURE_SCHEMA,
    }:
        raise ValueError(f"unsupported feature schema: {feature_schema}")
    spatial_values = np.asarray(spatial)
    context_values = np.asarray(context)
    if spatial_values.ndim not in {3, 4} or spatial_values.shape[-3] != 6:
        raise ValueError("spatial must contain the six canonical MOS channels")
    if context_values.ndim not in {1, 2} or context_values.shape[-1] != 16:
        raise ValueError("context must contain the sixteen canonical MOS features")
    if feature_schema == FOUR_TERM_ABLATION_FEATURE_SCHEMA:
        spatial_indices = FOUR_TERM_ABLATION_SPATIAL_INDICES
        context_indices = FOUR_TERM_ABLATION_CONTEXT_INDICES
    else:
        spatial_indices = FOUR_TERM_NO_ALLOCATION_SPATIAL_INDICES
        context_indices = FOUR_TERM_NO_ALLOCATION_CONTEXT_INDICES
    projected_spatial = np.take(spatial_values, spatial_indices, axis=-3)
    projected_context = np.take(context_values, context_indices, axis=-1)
    if feature_schema == FOUR_TERM_PREDICTED_OBSERVATION_FEATURE_SCHEMA:
        # Both arrays are available before o_(t+1): channel 1 is Q(s_(t+1)|t)
        # and channel 4 is P(detection|s_(t+1), R_(t+1)).  Their contraction
        # is the binary predicted-observation distribution, not the realized
        # future observation.
        predicted_posterior = np.take(spatial_values, 1, axis=-3)
        detection_probability = np.take(spatial_values, 4, axis=-3)
        predicted_detection = np.sum(
            predicted_posterior * detection_probability,
            axis=(-2, -1),
        )
        predicted_detection = np.clip(predicted_detection, 0.0, 1.0)
        predicted_observation = np.stack(
            (1.0 - predicted_detection, predicted_detection), axis=-1
        ).astype(projected_context.dtype, copy=False)
        projected_context = np.concatenate(
            (projected_context, predicted_observation), axis=-1
        )
    return projected_spatial, projected_context
