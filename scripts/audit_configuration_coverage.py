"""Audit whether MOS contexts require resolution, depth, both, or neither.

The input must be a step-derived dataset produced by
``scripts/derive_step_g_dataset.py``.  The audit reconstructs the task-only
meta-score used by the immediate/improvement controller::

    A - C + E_1 + Delta E

where epistemic value is ambiguity minus parameter information gain.  A
configuration dimension is called critical only when the best candidate on
the more expensive side improves the score by ``--margin-nats`` *and* changes
the first physical action.  Score-only improvements are reported separately.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


def _best(scores: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return indices[np.argmax(scores[:, indices], axis=1)]


def _counts(values: np.ndarray) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(Counter(values.tolist()).items())}


def audit(dataset_dir: Path, output_dir: Path, margin_nats: float) -> dict[str, object]:
    if margin_nats < 0:
        raise ValueError("margin_nats must be nonnegative")
    with np.load(dataset_dir / "training_data.npz", allow_pickle=False) as archive:
        required = {
            "context_ids",
            "resolutions",
            "depths",
            "candidate_actions",
            "state_accuracy",
            "state_complexity",
            "posterior_immediate_epistemic",
            "posterior_improvement_epistemic",
            "posterior_immediate_information_gain",
            "posterior_improvement_information_gain",
        }
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(
                "dataset must first be processed by derive_step_g_dataset.py; "
                f"missing arrays: {sorted(missing)}"
            )
        arrays = {name: archive[name].copy() for name in archive.files}

    resolutions = np.asarray(arrays["resolutions"], dtype=int)
    depths = np.asarray(arrays["depths"], dtype=int)
    actions = np.asarray(arrays["candidate_actions"], dtype=int)
    context_ids = np.asarray(arrays["context_ids"]).astype(str)
    accuracy = np.asarray(arrays["state_accuracy"], dtype=float)
    complexity = np.asarray(arrays["state_complexity"], dtype=float)
    if accuracy.shape != actions.shape or complexity.shape != actions.shape:
        raise ValueError("state labels and candidate actions must have matching shapes")

    # Filtered state inference depends on resolution, not prospective depth.
    # Match training by removing duplicate-depth numerical noise.
    shared_accuracy = accuracy.copy()
    shared_complexity = complexity.copy()
    for resolution in np.unique(resolutions):
        indices = np.flatnonzero(resolutions == resolution)
        shared_accuracy[:, indices] = accuracy[:, indices].mean(axis=1, keepdims=True)
        shared_complexity[:, indices] = complexity[:, indices].mean(axis=1, keepdims=True)

    epistemic = (
        np.asarray(arrays["posterior_immediate_epistemic"], dtype=float)
        + np.asarray(arrays["posterior_improvement_epistemic"], dtype=float)
        - np.asarray(arrays["posterior_immediate_information_gain"], dtype=float)
        - np.asarray(arrays["posterior_improvement_information_gain"], dtype=float)
    )
    scores = shared_accuracy - shared_complexity + epistemic

    shallow = np.flatnonzero(depths == 1)
    deep = np.flatnonzero(depths > 1)
    coarse = np.flatnonzero(resolutions == int(resolutions.min()))
    fine = np.flatnonzero(resolutions > int(resolutions.min()))
    if not all(group.size for group in (shallow, deep, coarse, fine)):
        raise ValueError("dataset must contain shallow/deep and coarse/fine candidates")

    best_all = np.argmax(scores, axis=1)
    best_shallow = _best(scores, shallow)
    best_deep = _best(scores, deep)
    best_coarse = _best(scores, coarse)
    best_fine = _best(scores, fine)
    rows = np.arange(scores.shape[0])

    depth_gain = scores[rows, best_deep] - scores[rows, best_shallow]
    resolution_gain = scores[rows, best_fine] - scores[rows, best_coarse]
    depth_action_change = actions[rows, best_deep] != actions[rows, best_shallow]
    resolution_action_change = actions[rows, best_fine] != actions[rows, best_coarse]
    depth_score_matters = depth_gain > margin_nats
    resolution_score_matters = resolution_gain > margin_nats
    depth_critical = depth_score_matters & depth_action_change
    resolution_critical = resolution_score_matters & resolution_action_change

    regimes = np.full(scores.shape[0], "insensitive", dtype="<U16")
    regimes[depth_critical & ~resolution_critical] = "depth_only"
    regimes[resolution_critical & ~depth_critical] = "resolution_only"
    regimes[depth_critical & resolution_critical] = "joint"

    success = arrays.get("success")
    task_cost = arrays.get("task_cost")
    records = []
    for index, context_id in enumerate(context_ids):
        record: dict[str, object] = {
            "context_id": context_id,
            "regime": str(regimes[index]),
            "best_resolution": int(resolutions[best_all[index]]),
            "best_depth": int(depths[best_all[index]]),
            "best_action": int(actions[index, best_all[index]]),
            "best_score": float(scores[index, best_all[index]]),
            "depth_gain_nats": float(depth_gain[index]),
            "depth_action_changed": bool(depth_action_change[index]),
            "resolution_gain_nats": float(resolution_gain[index]),
            "resolution_action_changed": bool(resolution_action_change[index]),
            "unique_candidate_actions": int(np.unique(actions[index]).size),
        }
        if success is not None and task_cost is not None:
            success_values = np.asarray(success)
            cost_values = np.asarray(task_cost)
            record.update(
                depth_success_delta=float(
                    success_values[index, best_deep[index]]
                    - success_values[index, best_shallow[index]]
                ),
                depth_task_cost_delta=float(
                    cost_values[index, best_deep[index]]
                    - cost_values[index, best_shallow[index]]
                ),
                resolution_success_delta=float(
                    success_values[index, best_fine[index]]
                    - success_values[index, best_coarse[index]]
                ),
                resolution_task_cost_delta=float(
                    cost_values[index, best_fine[index]]
                    - cost_values[index, best_coarse[index]]
                ),
            )
        records.append(record)

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "context_coverage.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    regime_counts = Counter(regimes.tolist())
    context_count = int(scores.shape[0])
    instance_prefixes = np.asarray(
        [value.split("-decision-")[0] for value in context_ids], dtype=str
    )
    report: dict[str, object] = {
        "dataset": str(dataset_dir.resolve()),
        "contexts": context_count,
        "instances": len({value.split("-decision-")[0] for value in context_ids}),
        "margin_nats": margin_nats,
        "task_score": "A - C + E1 + delta_E",
        "critical_definition": "score gain exceeds margin and first action changes",
        "regimes": {
            name: {
                "count": int(regime_counts.get(name, 0)),
                "fraction": float(regime_counts.get(name, 0) / context_count),
                "instances": int(np.unique(instance_prefixes[regimes == name]).size),
            }
            for name in ("insensitive", "resolution_only", "depth_only", "joint")
        },
        "score_only": {
            "resolution_gain_without_action_change": int(
                np.sum(resolution_score_matters & ~resolution_action_change)
            ),
            "depth_gain_without_action_change": int(
                np.sum(depth_score_matters & ~depth_action_change)
            ),
        },
        "action_disagreement": {
            "any_candidate": int(np.sum(np.apply_along_axis(lambda x: np.unique(x).size, 1, actions) > 1)),
            "resolution_best": int(np.sum(resolution_action_change)),
            "depth_best": int(np.sum(depth_action_change)),
        },
        "best_resolution": _counts(resolutions[best_all]),
        "best_depth": _counts(depths[best_all]),
        "best_configuration": _counts(
            np.asarray(
                [f"g{resolutions[i]}_T{depths[i]}" for i in best_all], dtype=str
            )
        ),
        "gain_nats": {
            "resolution_median": float(np.median(resolution_gain)),
            "resolution_p95": float(np.quantile(resolution_gain, 0.95)),
            "depth_median": float(np.median(depth_gain)),
            "depth_p95": float(np.quantile(depth_gain, 0.95)),
        },
    }
    if success is not None and task_cost is not None:
        success_values = np.asarray(success, dtype=float)
        cost_values = np.asarray(task_cost, dtype=float)

        def outcome_summary(
            better: np.ndarray, baseline: np.ndarray, critical: np.ndarray
        ) -> dict[str, int]:
            success_delta = success_values[rows, better] - success_values[rows, baseline]
            cost_delta = cost_values[rows, better] - cost_values[rows, baseline]
            return {
                "critical_contexts": int(np.sum(critical)),
                "success_improved": int(np.sum(critical & (success_delta > 0))),
                "success_worsened": int(np.sum(critical & (success_delta < 0))),
                "task_cost_reduced": int(np.sum(critical & (cost_delta < 0))),
                "task_cost_increased": int(np.sum(critical & (cost_delta > 0))),
            }

        report["realized_outcomes"] = {
            "depth": outcome_summary(best_deep, best_shallow, depth_critical),
            "resolution": outcome_summary(best_fine, best_coarse, resolution_critical),
        }
    (output_dir / "coverage_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--margin-nats", type=float, default=0.001)
    args = parser.parse_args()
    print(json.dumps(audit(args.dataset_dir, args.output_dir, args.margin_nats), indent=2))


if __name__ == "__main__":
    main()
