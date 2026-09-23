"""Observation-marginalized next-state labels for MOS metacontrol."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np
from PyAIF.numerics import log_stable_probability

from .beliefs import project_canonical

EXPECTED_STATE_LABEL_SCHEMA = "expected-binary-detection-state-terms-v1"


def expected_categorical_state_terms(
    prior: np.ndarray,
    likelihood: np.ndarray,
) -> tuple[float, float]:
    """Return expected accuracy and KL complexity before observing an outcome.

    ``likelihood`` has shape ``(outcomes, states)``. The expectation is over
    the predictive observation distribution induced by ``prior``.
    """

    prior_values = np.asarray(prior, dtype=float).reshape(-1)
    likelihood_values = np.asarray(likelihood, dtype=float)
    if likelihood_values.ndim != 2 or likelihood_values.shape[1] != prior_values.size:
        raise ValueError("likelihood must have shape (outcomes, states)")
    if (
        not np.all(np.isfinite(prior_values))
        or np.any(prior_values < 0)
        or prior_values.sum() <= 0
    ):
        raise ValueError("prior must be finite, nonnegative, and have positive mass")
    if (
        not np.all(np.isfinite(likelihood_values))
        or np.any(likelihood_values < 0)
        or not np.allclose(likelihood_values.sum(axis=0), 1.0)
    ):
        raise ValueError("likelihood columns must be categorical distributions")

    prior_values = prior_values / prior_values.sum()
    joint = likelihood_values * prior_values[None, :]
    predictive = joint.sum(axis=1)
    accuracy = float(
        np.sum(joint * log_stable_probability(likelihood_values))
    )
    complexity = 0.0
    for outcome, probability in enumerate(predictive):
        if probability <= 0:
            continue
        posterior = joint[outcome] / probability
        complexity += float(
            probability
            * posterior.dot(
                log_stable_probability(posterior)
                - log_stable_probability(prior_values)
            )
        )
    return accuracy, complexity


def _instance_seed(context_id: str) -> int:
    match = re.fullmatch(r"instance-(\d+)-decision-\d+", context_id)
    if match is None:
        raise ValueError(f"unrecognized context id: {context_id}")
    return int(match.group(1))


def build_expected_state_dataset(source_dir: Path, output_dir: Path) -> dict[str, object]:
    """Create a compact derived dataset with expected t+1 state labels."""

    try:
        from active_inference_navigation.mos import (
            FindOutcome,
            detection_distribution,
            sample_mos_instance,
            target_state_position,
        )
    except ImportError as error:  # pragma: no cover - optional sibling dependency
        raise ImportError(
            "expected MOS labels require active-inference-navigation-agent"
        ) from error

    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    with np.load(source_dir / "training_data.npz", allow_pickle=False) as archive:
        training = {name: archive[name].copy() for name in archive.files}
    with np.load(source_dir / "research_archive.npz", allow_pickle=False) as archive:
        observation = archive["observation"].copy()
        canonical_prior = archive["canonical_target_prior"].copy()
        policy_posterior = archive["policy_posterior"].copy()
        policy_risk = archive["policy_risk"].copy()
        policy_ambiguity = archive["policy_ambiguity"].copy()
        policy_information_gain = archive["policy_information_gain"].copy()

    context_ids = np.asarray(training["context_ids"]).astype(str)
    resolutions = np.asarray(training["resolutions"], dtype=int)
    keep = observation[:, 3] == int(FindOutcome.SEARCHING)
    retained_indices = np.flatnonzero(keep)
    if retained_indices.size == 0:
        raise ValueError("source dataset contains no ordinary search contexts")

    layouts: dict[int, object] = {}
    likelihood_cache: dict[tuple[int, int, int, int], np.ndarray] = {}
    expected_accuracy = np.empty((retained_indices.size, resolutions.size), dtype=np.float32)
    expected_complexity = np.empty_like(expected_accuracy)
    for output_index, source_index in enumerate(retained_indices):
        context_id = context_ids[source_index]
        seed = _instance_seed(context_id)
        if seed not in layouts:
            layouts[seed] = sample_mos_instance(seed).layout
        layout = layouts[seed]
        robot = (int(observation[source_index, 0]), int(observation[source_index, 1]))
        for allocation_index, resolution in enumerate(resolutions):
            prior = project_canonical(
                canonical_prior[source_index, allocation_index], int(resolution)
            ).reshape(-1)
            cache_key = (seed, robot[0], robot[1], int(resolution))
            likelihood = likelihood_cache.get(cache_key)
            if likelihood is None:
                likelihood = np.stack(
                    [
                        detection_distribution(
                            robot,
                            target_state_position(state, int(resolution), layout.size),
                            layout,
                        )
                        for state in range(int(resolution) ** 2)
                    ],
                    axis=1,
                )
                likelihood_cache[cache_key] = likelihood
            accuracy, complexity = expected_categorical_state_terms(prior, likelihood)
            expected_accuracy[output_index, allocation_index] = accuracy
            expected_complexity[output_index, allocation_index] = complexity

    sample_count = len(context_ids)
    derived: dict[str, np.ndarray] = {}
    for name, values in training.items():
        array = np.asarray(values)
        derived[name] = array[keep] if array.ndim > 0 and array.shape[0] == sample_count else array
    derived["state_accuracy"] = expected_accuracy
    derived["state_complexity"] = expected_complexity
    posterior_sum = np.nansum(policy_posterior, axis=2)
    if not np.allclose(posterior_sum, 1.0, atol=1e-5):
        raise ValueError("policy posteriors are not normalized")
    derived["posterior_risk"] = np.nansum(
        policy_posterior[keep] * policy_risk[keep], axis=2
    ).astype(np.float32)
    derived["posterior_ambiguity"] = np.nansum(
        policy_posterior[keep] * policy_ambiguity[keep], axis=2
    ).astype(np.float32)
    derived["posterior_information_gain"] = np.nansum(
        policy_posterior[keep] * policy_information_gain[keep], axis=2
    ).astype(np.float32)
    derived["state_label_schema"] = np.asarray(EXPECTED_STATE_LABEL_SCHEMA)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "training_data.npz", **derived)

    retained_ids = set(context_ids[keep].tolist())
    with (source_dir / "contexts.csv").open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        rows = [row for row in reader if row["context_id"] in retained_ids]
        fieldnames = reader.fieldnames
    if fieldnames is None:
        raise ValueError("contexts.csv has no header")
    with (output_dir / "contexts.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    report: dict[str, object] = {
        "source_dataset": str(source_dir.resolve()),
        "state_label_schema": EXPECTED_STATE_LABEL_SCHEMA,
        "source_contexts": int(sample_count),
        "retained_contexts": int(retained_indices.size),
        "excluded_nonsearch_contexts": int(sample_count - retained_indices.size),
        "expected_accuracy_range": [
            float(expected_accuracy.min()),
            float(expected_accuracy.max()),
        ],
        "expected_complexity_range": [
            float(expected_complexity.min()),
            float(expected_complexity.max()),
        ],
    }
    (output_dir / "derivation_manifest.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report
