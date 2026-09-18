"""Feature assembly for the neural task-performance model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .allocations import DEPTHS, RESOLUTIONS, Allocation
from .beliefs import canonicalize_posterior, normalized_entropy
from .information import categorical_fisher_map, detection_probability_map


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
    expected_free_energy: np.ndarray,
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
    efe = np.asarray(expected_free_energy, dtype=float).ravel()
    if policies.size < 1 or policies.size != efe.size:
        raise ValueError("policy posterior and EFE must have the same nonzero size")
    if np.any(policies < 0) or not np.all(np.isfinite(policies)) or policies.sum() <= 0:
        raise ValueError("policy posterior must be finite, nonnegative, and normalized")
    policies = policies / policies.sum()
    sorted_probabilities = np.sort(policies)
    probability_margin = float(
        sorted_probabilities[-1] - (sorted_probabilities[-2] if policies.size > 1 else 0.0)
    )
    sorted_efe = np.sort(efe)
    efe_gap = float(sorted_efe[1] - sorted_efe[0]) if efe.size > 1 else 0.0
    positive = policies > 0
    policy_entropy = float(-np.sum(policies[positive] * np.log(policies[positive])))
    if policies.size > 1:
        policy_entropy /= float(np.log(policies.size))
    confidence = np.asarray(
        [
            normalized_entropy(posterior, allocation.resolution),
            policy_entropy,
            probability_margin,
            efe_gap,
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
