"""Verify frozen trajectories, binning, and independently replay context values."""

import json
from collections import Counter
from pathlib import Path

import numpy as np
from plot_figure6_controller_behavior import (
    ADAPTIVE_DIR,
    CONTEXT_DIR,
    ROOT,
    _read_csv,
    _validate_context_replay,
)

from active_inference_neural_metacontrol import meta_q_runtime as runtime
from active_inference_neural_metacontrol.allocations import Allocation


def main():
    frozen = _read_csv(ADAPTIVE_DIR / "adaptive_trajectory.csv")
    replay = _read_csv(CONTEXT_DIR / "adaptive_trajectory.csv")
    _validate_context_replay(frozen, replay)
    valid = [r for r in replay if r["expected_information_gain_nats"]]
    fields = {
        "entropy": "belief_entropy_canonical",
        "confidence": "policy_margin",
        "fisher": "posterior_weighted_fisher",
    }
    edges = {
        k: np.quantile([float(r[f]) for r in valid], [0.25, 0.5, 0.75]) for k, f in fields.items()
    }
    cfg = json.loads((CONTEXT_DIR / "summary.json").read_text())["configuration"]
    cfg["checkpoint"] = Path(cfg["checkpoint"])
    cfg["ensemble_checkpoints"] = tuple(Path(p) for p in cfg["ensemble_checkpoints"])
    cfg["initial_allocation"] = Allocation(**cfg["initial_allocation"])
    independent = []
    original = runtime.build_meta_q_features

    def checked_features(**kwargs):
        features = original(**kwargs)
        p = np.asarray(kwargs["agent"].posterior_pi, dtype=float).ravel()
        p = p / p.sum()
        ordered = sorted(p)
        margin = ordered[-1] - ordered[-2]
        assert np.isclose(float(features.context[8]), margin, atol=1e-7, rtol=1e-6)
        q = np.asarray(features.spatial[0], dtype=float).ravel()
        q = q / q.sum()
        q = q[q > 0]
        h = -sum(float(v) * np.log(float(v)) for v in q) / np.log(400)
        independent.append({"confidence": float(margin), "entropy": float(h), "policies": len(p)})
        return features

    runtime.build_meta_q_features = checked_features
    _summary, new = runtime.run_meta_q_episode(20088, runtime.MetaQEvaluationConfig(**cfg))
    saved = [r for r in replay if int(r["instance_seed"]) == 20088]
    assert len(new) == len(saved) == 41
    geometry = [
        "step",
        "x",
        "y",
        "action",
        "next_x",
        "next_y",
        "source_allocation",
        "selected_allocation",
        "success",
    ]
    for a, b in zip(new, saved):
        for k in geometry:
            assert str(a[k]) == b[k], (k, a[k], b[k])
        for f in fields.values():
            if b[f]:
                assert np.isclose(float(a[f]), float(b[f]), atol=1e-7, rtol=1e-5), (f, a[f], b[f])
    assert len(independent) == 40
    for check, r in zip(independent, saved):
        assert np.isclose(check["entropy"], float(r["belief_entropy_canonical"]), atol=1e-7)
        assert np.isclose(check["confidence"], float(r["policy_margin"]), atol=1e-7)
    points = []
    for r in saved:
        if not r["policy_margin"]:
            continue
        bins = {
            k: int(np.digitize(float(r[f]), edges[k], right=True)) + 1 for k, f in fields.items()
        }
        points.append(
            dict(
                step=int(r["step"]),
                previous_position=[int(r["x"]), int(r["y"])],
                switch_position=[int(r["next_x"]), int(r["next_y"])],
                source=r["source_allocation"],
                next=r["selected_allocation"],
                **{k: float(r[f]) for k, f in fields.items()},
                quartiles=bins,
            )
        )
    report = {
        "all_frozen_replay_rows_match": True,
        "total_rows": len(replay),
        "total_valid_contexts": len(valid),
        "independent_replay_matches": True,
        "steps": 41,
        "nonterminal_contexts": 40,
        "independently_recomputed_confidence_and_entropy_match": True,
        "edges": {k: v.tolist() for k, v in edges.items()},
        "confidence_counts": dict(Counter(p["quartiles"]["confidence"] for p in points)),
        "policy_counts": sorted({p["policies"] for p in independent}),
        "points": points,
        "interpretation": [
            "Entropy and confidence refer to preceding inference/planning, before assimilation of the newly received observation.",
            "Position shown is after the executed action, where the next allocation is selected.",
            "Confidence is a full-policy probability margin, not first-action confidence; policy cardinality changes with depth.",
            "Canonical normalized entropy differs from native normalized entropy.",
            "A panel Fisher stratum labels selected contexts, not an entire episode.",
        ],
    }
    out = ROOT / "results/trajectory_context_audit"
    out.mkdir(parents=True, exist_ok=True)
    (out / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "points"}, indent=2))
    print(
        json.dumps(
            [p for p in points if 14 <= p["step"] <= 20 or p["quartiles"]["confidence"] == 1],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
