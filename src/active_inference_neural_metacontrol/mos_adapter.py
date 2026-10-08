"""Adapter from the navigation benchmark's MOS state to neural features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .allocations import Allocation
from .features import SpatialFeatures, build_context_vector, build_spatial_features


@dataclass(frozen=True)
class MOSFeatureBatch:
    """One controller input and useful provenance from a MOS decision."""

    spatial: SpatialFeatures
    context: np.ndarray
    predicted_posterior_l1: float
    selected_policy: int


def obstacle_map(layout: Any) -> np.ndarray:
    """Return a row-major binary obstacle map for a MOS layout."""

    values = np.zeros((layout.size, layout.size), dtype=np.float32)
    for x, y in layout.blocked:
        values[y, x] = 1.0
    return values


def canonical_detection_likelihood(layout: Any, robot: tuple[int, int]) -> np.ndarray:
    """Evaluate the binary sensor likelihood on every canonical target cell."""

    try:
        from active_inference_navigation.mos import detection_distribution
    except ImportError as error:  # pragma: no cover - depends on an optional sibling package
        raise ImportError(
            "MOS support requires active-inference-navigation-agent to be installed"
        ) from error

    likelihood = np.empty((2, layout.size, layout.size), dtype=float)
    for y in range(layout.size):
        for x in range(layout.size):
            likelihood[:, y, x] = detection_distribution(robot, (x, y), layout)
    return likelihood


def selected_policy_index(agent: Any) -> int:
    """Return the deterministic winning policy index."""

    posterior = np.asarray(agent.posterior_pi, dtype=float).reshape(-1)
    if posterior.size == 0:
        raise ValueError("the MOS agent has no policy posterior")
    return int(np.argmax(posterior))


def selected_policy_next_target_posterior(agent: Any) -> np.ndarray:
    """Read the already-computed target prediction after the policy's first action."""

    policy = selected_policy_index(agent)
    trajectories = agent.policy_dep_posteriors
    if trajectories is None or trajectories.shape[1] < 2:
        return np.asarray(agent.filtered_posteriors[2], dtype=float).copy()
    return np.asarray(trajectories[policy, 1, 2], dtype=float).copy()


def build_mos_features(
    *,
    agent: Any,
    allocation: Allocation,
    layout: Any,
    selected_action: int,
    next_robot_position: tuple[int, int],
    found_flags: np.ndarray | None = None,
) -> MOSFeatureBatch:
    """Convert an inferred MOS decision to the fixed neural-controller input."""

    posterior = np.asarray(agent.filtered_posteriors[2], dtype=float).copy()
    predicted = selected_policy_next_target_posterior(agent)
    spatial = build_spatial_features(
        posterior=posterior,
        predicted_posterior=predicted,
        resolution=allocation.resolution,
        next_robot_position=next_robot_position,
        obstacle_map=obstacle_map(layout),
        likelihood_at_next_position=canonical_detection_likelihood(layout, next_robot_position),
        canonical_size=layout.size,
    )
    context = build_context_vector(
        action_index=int(selected_action),
        action_count=5,
        allocation=allocation,
        found_flags=np.asarray([0.0] if found_flags is None else found_flags),
        posterior=posterior,
        policy_posterior=np.asarray(agent.posterior_pi, dtype=float),
    )
    current_normalized = posterior / posterior.sum()
    predicted_normalized = predicted / predicted.sum()
    return MOSFeatureBatch(
        spatial=spatial,
        context=context.astype(np.float32),
        predicted_posterior_l1=float(np.sum(np.abs(current_normalized - predicted_normalized))),
        selected_policy=selected_policy_index(agent),
    )
