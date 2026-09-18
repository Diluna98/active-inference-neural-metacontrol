"""Canonical sensing-information features for categorical MOS observations."""

from __future__ import annotations

import numpy as np


def _normalized_likelihood(likelihood: np.ndarray) -> np.ndarray:
    values = np.asarray(likelihood, dtype=float)
    if values.ndim != 3:
        raise ValueError("likelihood must have shape (outcomes, height, width)")
    if values.shape[1] != values.shape[2]:
        raise ValueError("likelihood spatial dimensions must be square")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("likelihood values must be finite and nonnegative")
    totals = values.sum(axis=0, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("every spatial likelihood column must have positive mass")
    return values / totals


def detection_probability_map(likelihood: np.ndarray, no_detection_index: int = 0) -> np.ndarray:
    """Return the probability of any detection at each hypothetical target cell."""

    values = _normalized_likelihood(likelihood)
    if not 0 <= no_detection_index < values.shape[0]:
        raise ValueError("no_detection_index is outside the outcome axis")
    return 1.0 - values[no_detection_index]


def categorical_fisher_map(likelihood: np.ndarray, epsilon: float = 1e-9) -> np.ndarray:
    """Approximate trace Fisher information over a two-dimensional categorical grid."""

    probabilities = _normalized_likelihood(likelihood)
    fisher = np.zeros(probabilities.shape[1:], dtype=float)
    for outcome_probability in probabilities:
        gradient_y, gradient_x = np.gradient(outcome_probability)
        fisher += (gradient_x**2 + gradient_y**2) / np.clip(
            outcome_probability,
            epsilon,
            None,
        )
    maximum = float(fisher.max())
    return fisher if maximum <= 0 else fisher / maximum


def expected_information_gain(posterior: np.ndarray, likelihood: np.ndarray) -> float:
    """Return expected categorical entropy reduction for one sensor position."""

    probabilities = _normalized_likelihood(likelihood)
    prior = np.asarray(posterior, dtype=float)
    if prior.size != probabilities.shape[1] * probabilities.shape[2]:
        raise ValueError("posterior and likelihood spatial dimensions do not match")
    prior = prior.reshape(probabilities.shape[1:])
    if not np.all(np.isfinite(prior)) or np.any(prior < 0) or prior.sum() <= 0:
        raise ValueError("posterior must be finite, nonnegative, and have positive mass")
    prior = prior / prior.sum()
    positive = prior > 0
    prior_entropy = float(-np.sum(prior[positive] * np.log(prior[positive])))
    outcome_probabilities = np.sum(probabilities * prior[None, :, :], axis=(1, 2))
    expected_posterior_entropy = 0.0
    for outcome, outcome_probability in enumerate(outcome_probabilities):
        if outcome_probability <= 0:
            continue
        updated = probabilities[outcome] * prior / outcome_probability
        updated_positive = updated > 0
        entropy = float(-np.sum(updated[updated_positive] * np.log(updated[updated_positive])))
        expected_posterior_entropy += float(outcome_probability) * entropy
    return max(0.0, prior_entropy - expected_posterior_entropy)
