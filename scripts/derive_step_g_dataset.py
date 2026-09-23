"""Derive per-future-step policy values from an existing MOS research archive.

The stored policy values are cumulative.  Because the MOS policy sets contain
every action prefix, an individual step can be recovered without rerunning the
environment or policy inference::

    g_1(a_1) = G(a_1)
    g_2(a_1, a_2) = G(a_1, a_2) - G(a_1)
    g_3(a_1, a_2, a_3) = G(a_1, a_2, a_3) - G(a_1, a_2)

Both winning-policy and policy-posterior-weighted values are retained.  Unused
steps are NaN.  For candidate comparison, ``future_*`` uses g1 for T=1, g2 for
T=2, and mean(g2, g3) for T=3.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from active_inference_navigation.mos import _policies

STEP_COMPONENTS = {
    "g": "policy_g",
    "preference": "policy_risk",
    "epistemic": "policy_ambiguity",
    "information_gain": "policy_information_gain",
}


def _policy_key(policy: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(value) for value in row) for row in np.asarray(policy))


def _policy_metadata() -> tuple[
    dict[int, tuple[tuple[tuple[int, ...], ...], ...]],
    dict[int, dict[tuple[tuple[int, ...], ...], int]],
]:
    keys: dict[int, tuple[tuple[tuple[int, ...], ...], ...]] = {}
    indices: dict[int, dict[tuple[tuple[int, ...], ...], int]] = {}
    for depth in (1, 2, 3):
        depth_keys = tuple(_policy_key(policy) for policy in _policies(depth, 20))
        keys[depth] = depth_keys
        indices[depth] = {key: index for index, key in enumerate(depth_keys)}
    return keys, indices


def _allocation_lookup(
    resolutions: np.ndarray, depths: np.ndarray
) -> dict[tuple[int, int], int]:
    lookup = {
        (int(resolution), int(depth)): index
        for index, (resolution, depth) in enumerate(zip(resolutions, depths, strict=True))
    }
    expected = {(resolution, depth) for resolution in (2, 5, 10, 20) for depth in (1, 2, 3)}
    missing = expected.difference(lookup)
    if missing:
        raise ValueError(f"dataset is missing allocations: {sorted(missing)}")
    return lookup


def _derive_component(
    cumulative: np.ndarray,
    posterior: np.ndarray,
    selected_policy: np.ndarray,
    policy_count: np.ndarray,
    resolutions: np.ndarray,
    depths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    context_count, allocation_count, _ = cumulative.shape
    selected_steps = np.full((context_count, allocation_count, 3), np.nan, dtype=np.float32)
    posterior_steps = np.full_like(selected_steps, np.nan)
    selected_future = np.empty((context_count, allocation_count), dtype=np.float32)
    posterior_future = np.empty_like(selected_future)
    policy_keys, policy_indices = _policy_metadata()
    allocation_indices = _allocation_lookup(resolutions, depths)

    for allocation_index, (resolution_value, depth_value) in enumerate(
        zip(resolutions, depths, strict=True)
    ):
        resolution = int(resolution_value)
        depth = int(depth_value)
        count = int(policy_count[0, allocation_index])
        if np.any(policy_count[:, allocation_index] != count):
            raise ValueError("policy count must be constant for each allocation")
        if count != len(policy_keys[depth]):
            raise ValueError(
                f"archive has {count} policies for T={depth}, expected {len(policy_keys[depth])}"
            )

        step_values = np.empty((context_count, count, depth), dtype=np.float64)
        for policy_index, key in enumerate(policy_keys[depth]):
            previous = np.zeros(context_count, dtype=np.float64)
            for step in range(1, depth + 1):
                prefix = key[:step]
                prefix_allocation = allocation_indices[(resolution, step)]
                prefix_policy = policy_indices[step][prefix]
                current = cumulative[:, prefix_allocation, prefix_policy].astype(np.float64)
                step_values[:, policy_index, step - 1] = current - previous
                previous = current

        chosen = selected_policy[:, allocation_index].astype(int)
        if np.any(chosen < 0) or np.any(chosen >= count):
            raise ValueError("selected policy index is outside the stored policy set")
        selected = step_values[np.arange(context_count), chosen]
        selected_steps[:, allocation_index, :depth] = selected.astype(np.float32)

        weights = posterior[:, allocation_index, :count].astype(np.float64)
        totals = weights.sum(axis=1, keepdims=True)
        if np.any(~np.isfinite(totals)) or np.any(totals <= 0):
            raise ValueError("policy posterior must have positive finite mass")
        weights /= totals
        weighted = np.einsum("np,npt->nt", weights, step_values, optimize=True)
        posterior_steps[:, allocation_index, :depth] = weighted.astype(np.float32)

        selected_future[:, allocation_index] = (
            selected[:, 0] if depth == 1 else selected[:, 1:depth].mean(axis=1)
        )
        posterior_future[:, allocation_index] = (
            weighted[:, 0] if depth == 1 else weighted[:, 1:depth].mean(axis=1)
        )

        reconstructed = selected.sum(axis=1)
        stored = cumulative[np.arange(context_count), allocation_index, chosen]
        if not np.allclose(reconstructed, stored, rtol=1e-5, atol=1e-5):
            difference = float(np.max(np.abs(reconstructed - stored)))
            raise ValueError(f"per-step reconstruction failed; maximum error={difference}")

    return selected_steps, posterior_steps, selected_future, posterior_future


def derive(source: Path, output: Path) -> dict[str, object]:
    source = source.resolve()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    with np.load(source / "training_data.npz", allow_pickle=False) as archive:
        training = {name: archive[name].copy() for name in archive.files}
    with np.load(source / "research_archive.npz", allow_pickle=False) as archive:
        research = {name: archive[name].copy() for name in archive.files}

    required_training = {"resolutions", "depths"}
    required_research = {
        "selected_policy",
        "policy_count",
        "policy_posterior",
        *STEP_COMPONENTS.values(),
    }
    missing_training = required_training.difference(training)
    missing_research = required_research.difference(research)
    if missing_training or missing_research:
        raise ValueError(
            f"missing arrays: training={sorted(missing_training)}, "
            f"research={sorted(missing_research)}"
        )

    resolutions = np.asarray(training["resolutions"], dtype=int)
    depths = np.asarray(training["depths"], dtype=int)
    for label, archive_name in STEP_COMPONENTS.items():
        selected_steps, posterior_steps, selected_future, posterior_future = _derive_component(
            np.asarray(research[archive_name]),
            np.asarray(research["policy_posterior"]),
            np.asarray(research["selected_policy"]),
            np.asarray(research["policy_count"]),
            resolutions,
            depths,
        )
        training[f"selected_step_{label}"] = selected_steps
        training[f"posterior_step_{label}"] = posterior_steps
        training[f"selected_future_{label}"] = selected_future
        training[f"posterior_future_{label}"] = posterior_future
        training[f"selected_immediate_{label}"] = selected_steps[:, :, 0]
        training[f"posterior_immediate_{label}"] = posterior_steps[:, :, 0]
        training[f"selected_improvement_{label}"] = (
            selected_future - selected_steps[:, :, 0]
        )
        training[f"posterior_improvement_{label}"] = (
            posterior_future - posterior_steps[:, :, 0]
        )

    training["step_value_schema"] = np.asarray(
        "prefix-differenced-selected-and-posterior-future-mean-v1"
    )
    np.savez_compressed(output / "training_data.npz", **training)

    for name in ("contexts.csv", "branches.csv", "reference_trajectory.csv"):
        path = source / name
        if path.is_file():
            shutil.copy2(path, output / name)

    report: dict[str, object] = {
        "source_dataset": str(source),
        "contexts": int(np.asarray(training["context_ids"]).size),
        "allocations": int(resolutions.size),
        "step_value_schema": str(training["step_value_schema"]),
        "formula": {
            "T1": "g1",
            "T2": "g2",
            "T3": "mean(g2, g3)",
        },
        "components": list(STEP_COMPONENTS),
        "research_archive_copied": False,
    }
    (output / "step_value_manifest.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(derive(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
