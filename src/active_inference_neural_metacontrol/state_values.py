"""Correct state-inference accuracy and complexity diagnostics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from PyAIF.numerics import log_stable_probability


def corrected_state_terms(
    agent: Any,
    observation: Sequence[int],
    priors: Sequence[np.ndarray],
    posteriors: Sequence[np.ndarray],
) -> tuple[float, float]:
    """Return accuracy once per modality and KL complexity once per factor."""

    accuracy, complexity = corrected_state_term_components(
        agent, observation, priors, posteriors
    )
    return float(accuracy.sum()), float(complexity.sum())


def corrected_state_term_components(
    agent: Any,
    observation: Sequence[int],
    priors: Sequence[np.ndarray],
    posteriors: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Return accuracy by modality and KL complexity by hidden-state factor."""

    core = getattr(agent, "_agent", agent)
    beliefs = [np.asarray(values, dtype=float) for values in posteriors]
    complexity = np.asarray(
        [
            posterior.dot(
                log_stable_probability(posterior)
                - log_stable_probability(np.asarray(prior, dtype=float))
            )
            for posterior, prior in zip(beliefs, priors, strict=True)
        ],
        dtype=float,
    )

    accuracy = []
    for modality, dependencies in enumerate(core.mod_dep):
        likelihood = np.take(core.A[modality], observation[modality], axis=0)
        arguments: list[Any] = [log_stable_probability(likelihood), list(dependencies)]
        for dependency in dependencies:
            arguments.extend([beliefs[dependency], [dependency]])
        arguments.append([])
        accuracy.append(float(np.einsum(*arguments, optimize=True)))
    return np.asarray(accuracy, dtype=float), complexity
