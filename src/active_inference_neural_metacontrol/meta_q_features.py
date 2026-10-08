"""Allocation-independent features for sequential neural metacontrol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .allocations import Allocation
from .features import encode_mos_observation
from .information import expected_information_gain
from .mos_adapter import build_mos_features

META_Q_FEATURE_SCHEMA = "mos-meta-q-full-prediction-realized-observation-v1"
META_Q_SPATIAL_CHANNELS = 6
META_Q_CONTEXT_FEATURES = 56


@dataclass(frozen=True)
class MetaQFeatures:
    """One state presented to the meta-level value function."""

    spatial: np.ndarray
    context: np.ndarray

    def __post_init__(self) -> None:
        spatial = np.asarray(self.spatial, dtype=np.float32)
        context = np.asarray(self.context, dtype=np.float32)
        if spatial.shape != (META_Q_SPATIAL_CHANNELS, 20, 20):
            raise ValueError("meta-Q spatial features must have shape (6, 20, 20)")
        if context.shape != (META_Q_CONTEXT_FEATURES,):
            raise ValueError("meta-Q context must contain 56 features")
        if not np.all(np.isfinite(spatial)) or not np.all(np.isfinite(context)):
            raise ValueError("meta-Q features must be finite")
        object.__setattr__(self, "spatial", spatial)
        object.__setattr__(self, "context", context)


def summarize_decision_context(features: MetaQFeatures) -> dict[str, float]:
    """Return resolution-independent diagnostics for a meta-level decision."""

    current = np.asarray(features.spatial[0], dtype=float)
    predicted = np.asarray(features.spatial[1], dtype=float)
    detection = np.asarray(features.spatial[4], dtype=float)
    fisher = np.asarray(features.spatial[5], dtype=float)

    def normalized_canonical_entropy(values: np.ndarray) -> float:
        probabilities = values.ravel()
        probabilities = probabilities / probabilities.sum()
        positive = probabilities > 0
        entropy = float(-np.sum(probabilities[positive] * np.log(probabilities[positive])))
        return entropy / float(np.log(probabilities.size))

    predicted_probabilities = predicted / predicted.sum()
    likelihood = np.stack((1.0 - detection, detection))
    # project_meta_q_context keeps found flag followed by the three confidence
    # statistics at indices 5:9: posterior entropy, policy entropy, and margin.
    return {
        "belief_entropy_native": float(features.context[6]),
        "belief_entropy_canonical": normalized_canonical_entropy(current),
        "predicted_entropy_canonical": normalized_canonical_entropy(predicted),
        "policy_entropy": float(features.context[7]),
        "policy_margin": float(features.context[8]),
        "predicted_detection_probability": float(np.sum(predicted_probabilities * detection)),
        "posterior_weighted_fisher": float(np.sum(predicted_probabilities * fisher)),
        "expected_information_gain_nats": expected_information_gain(
            predicted_probabilities,
            likelihood,
        ),
    }


def project_meta_q_context(
    context: np.ndarray,
    realized_observation: tuple[int, ...] | np.ndarray,
) -> np.ndarray:
    """Remove source allocation identity and append the observation just received.

    The canonical MOS context is ordered as physical action (5), resolution (4),
    depth (3), found flag (1), and confidence statistics (3).  Resolution and
    depth describe how the present decision was obtained; they are deliberately
    excluded from task value.  Switching cost receives the source allocation
    separately at selection time.
    """

    values = np.asarray(context, dtype=np.float32)
    if values.shape != (16,):
        raise ValueError("canonical MOS context must contain 16 features")
    allocation_independent = np.concatenate((values[:5], values[12:16]))
    observation = encode_mos_observation(np.asarray(realized_observation))
    result = np.concatenate((allocation_independent, observation)).astype(np.float32)
    if result.shape != (META_Q_CONTEXT_FEATURES,):
        raise RuntimeError("unexpected meta-Q context shape")
    return result


def build_meta_q_features(
    *,
    agent: Any,
    allocation: Allocation,
    layout: Any,
    selected_action: int,
    next_robot_position: tuple[int, int],
    realized_observation: tuple[int, ...] | np.ndarray,
) -> MetaQFeatures:
    """Build the state visible when choosing the following allocation."""

    features = build_mos_features(
        agent=agent,
        allocation=allocation,
        layout=layout,
        selected_action=selected_action,
        next_robot_position=next_robot_position,
        found_flags=np.asarray([0.0], dtype=np.float32),
    )
    return MetaQFeatures(
        spatial=features.spatial.tensor,
        context=project_meta_q_context(features.context, realized_observation),
    )
