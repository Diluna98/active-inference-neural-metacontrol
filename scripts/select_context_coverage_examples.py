"""Choose episodes by input-context coverage, never by allocated resources."""

import json

import numpy as np
from plot_figure6_controller_behavior import ADAPTIVE_DIR, CONTEXT_DIR, ROOT, _allocation, _read_csv

OUT = ROOT / "results/context_coverage_trajectory_examples"
RULE = {
    "status": "Retrospective fixed context-coverage rule, not preregistered.",
    "bins": "Use exactly the global decision-level quartile thresholds and right-inclusive binning of Figure 4.",
    "fisher_stratum": "For each context variable, maximize the smaller of the pooled Q1 and Q4 decision counts across Fisher quartiles; ties by lowest Fisher quartile.",
    "episode": "Within that Fisher stratum, maximize the smaller of the episode Q1 and Q4 counts; ties by lowest seed. Require both counts positive.",
    "anchor": "Within each selected episode/bin, choose the decision nearest to the pooled bin median in context-variable and Fisher coordinates, scaled by their global IQRs; ties by earliest step.",
    "exclusions": "No filtering on success, resource allocation, trajectory shape, or conformity to aggregate trends.",
    "meaning": "Illustrates contexts encountered; not a causal test or guarantee of the aggregate trend within a single episode.",
}


def select(successful_only=False, require_depth_decrease=False):
    rule = dict(RULE)
    if successful_only:
        rule["population"] = (
            "Successful episodes only (84 of the 100 frozen benchmark episodes); global quartile thresholds remain those of all 100 episodes."
        )
        rule["exclusions"] = (
            "Exclude failed episodes solely to illustrate successful behaviour. No filtering on allocation choices, shape, or conformity to trends."
        )
        rule["fisher_stratum"] = (
            "Maximize the smaller pooled Q1/Q4 decision count among successful episodes, ties by lowest Fisher quartile."
        )
        rule["meaning"] = (
            "Success-conditioned illustrations; not representative of all outcomes and not a causal justification of switches."
        )
    if require_depth_decrease:
        if not successful_only:
            raise ValueError("Depth-pattern illustration requires successful-only population.")
        rule["confidence_example"] = (
            "Within the same automatically selected Fisher quartile, retain successful episodes with positive Q1/Q4 counts and mean depth(Q1)>mean depth(Q4). Maximize balanced count; ties by lowest seed."
        )
        rule["confidence_anchors"] = (
            "Select a Q1/Q4 decision pair with selected depth(Q1)>selected depth(Q4). Minimize summed squared standardized distance to the pooled context/Fisher medians of the two bins; ties by Q1 step then Q4 step. These anchors explicitly illustrate the depth pattern."
        )
        rule["exclusions"] = (
            "Success conditioned. Confidence-panel selection additionally conditions on the observed depth decrease; entropy selection is unchanged. No trajectory-appearance filtering."
        )
        rule["meaning"] = (
            "Deliberately selected illustrative depth-pattern example, not independent validation or a representative sample."
        )
    rows = [
        r
        for r in _read_csv(CONTEXT_DIR / "adaptive_trajectory.csv")
        if r["expected_information_gain_nats"]
    ]
    fields = {
        "entropy": "belief_entropy_canonical",
        "confidence": "policy_margin",
        "fisher": "posterior_weighted_fisher",
    }
    values = {k: np.array([float(r[field]) for r in rows]) for k, field in fields.items()}
    edges = {k: np.quantile(v, [0.25, 0.5, 0.75]) for k, v in values.items()}
    bins = {k: np.digitize(v, edges[k], right=True) for k, v in values.items()}
    seeds = np.array([int(r["instance_seed"]) for r in rows])
    if successful_only:
        eligible_seeds = {
            int(e["instance_seed"])
            for e in _read_csv(ADAPTIVE_DIR / "episodes.csv")
            if e["controller"] == "adaptive" and e["success"] == "True"
        }
    else:
        eligible_seeds = set(seeds)
    eligible = np.isin(seeds, list(eligible_seeds))
    selected = []
    for kind in ("entropy", "confidence"):
        support = []
        for f in range(4):
            counts = [
                int(np.sum(eligible & (bins["fisher"] == f) & (bins[kind] == b))) for b in (0, 3)
            ]
            support.append(counts)
        f = max(range(4), key=lambda q: (min(support[q]), -q))
        scores = []
        for seed in sorted(eligible_seeds):
            counts = [
                int(np.sum((seeds == seed) & (bins["fisher"] == f) & (bins[kind] == b)))
                for b in (0, 3)
            ]
            if require_depth_decrease and kind == "confidence":
                if min(counts) == 0:
                    continue
                means = []
                for b in (0, 3):
                    indices = np.flatnonzero(
                        (seeds == seed) & (bins["fisher"] == f) & (bins[kind] == b)
                    )
                    means.append(
                        float(
                            np.mean(
                                [_allocation(rows[i]["selected_allocation"])[1] for i in indices]
                            )
                        )
                    )
                if means[0] <= means[1]:
                    continue
            scores.append((min(counts), int(seed), counts))
        if not scores:
            raise RuntimeError(
                f"No successful depth-decrease example in fixed Fisher quartile {f + 1}; do not silently change stratum."
            )
        score, seed, counts = max(scores, key=lambda s: (s[0], -s[1]))
        if score == 0:
            raise RuntimeError(f"No episode covers both extremes for {kind}; do not relax rule.")
        groups = []
        anchors = []
        for b in (0, 3):
            pool = (bins["fisher"] == f) & (bins[kind] == b)
            indices = np.flatnonzero(pool & (seeds == seed))
            distances = np.zeros(indices.size)
            for key in (kind, "fisher"):
                scale = max(
                    float(np.quantile(values[key], 0.75) - np.quantile(values[key], 0.25)), 1e-12
                )
                distances += ((values[key][indices] - np.median(values[key][pool])) / scale) ** 2
            idx = min(
                zip(distances, indices), key=lambda item: (item[0], int(rows[item[1]]["step"]))
            )[1]
            row = rows[idx]
            anchors.append(dict(row, context_quartile=b + 1))
            allocation = np.array([_allocation(rows[i]["selected_allocation"]) for i in indices])
            groups.append(
                {
                    "quartile": b + 1,
                    "n": len(indices),
                    "mean_state_space": float(np.mean(allocation[:, 0] ** 2)),
                    "mean_depth": float(np.mean(allocation[:, 1])),
                    "decisions": [int(rows[i]["step"]) for i in indices],
                }
            )
        if require_depth_decrease and kind == "confidence":
            options = []
            for b in (0, 3):
                pool = (bins["fisher"] == f) & (bins[kind] == b)
                indices = np.flatnonzero(pool & (seeds == seed))
                distances = np.zeros(indices.size)
                for key in (kind, "fisher"):
                    scale = max(
                        float(np.quantile(values[key], 0.75) - np.quantile(values[key], 0.25)),
                        1e-12,
                    )
                    distances += (
                        (values[key][indices] - np.median(values[key][pool])) / scale
                    ) ** 2
                options.append(list(zip(distances, indices)))
            pairs = [
                (d1 + d4, int(rows[i1]["step"]), int(rows[i4]["step"]), i1, i4)
                for d1, i1 in options[0]
                for d4, i4 in options[1]
                if _allocation(rows[i1]["selected_allocation"])[1]
                > _allocation(rows[i4]["selected_allocation"])[1]
            ]
            _, _, _, i1, i4 = min(pairs)
            anchors = [dict(rows[i1], context_quartile=1), dict(rows[i4], context_quartile=4)]
        selected.append(
            {
                "kind": kind,
                "fisher_quartile": f + 1,
                "instance_seed": seed,
                "pooled_support": support,
                "episode_counts": counts,
                "anchors": anchors,
                "groups": groups,
            }
        )
    return {
        "rule": rule,
        "edges": {k: v.tolist() for k, v in edges.items()},
        "eligible_episodes": len(eligible_seeds),
        "selected": selected,
    }


def main():
    results = select()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "selection.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    for r in results["selected"]:
        print(
            r["kind"],
            r["instance_seed"],
            "Fisher Q",
            r["fisher_quartile"],
            "counts",
            r["episode_counts"],
            "groups",
            r["groups"],
        )
        print(
            "anchors",
            [(a["step"], a["next_x"], a["next_y"], a["selected_allocation"]) for a in r["anchors"]],
        )


if __name__ == "__main__":
    main()
