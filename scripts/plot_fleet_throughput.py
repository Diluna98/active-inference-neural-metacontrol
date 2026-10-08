"""Plot shared-compute fleet throughput from fleet_trials.csv."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LABELS = {
    "adaptive": "Ours",
    "fixed_g10_T2": r"Fixed $(\gamma=10,\tau=2)$",
}
COLORS = {"adaptive": "#D55E00", "fixed_g10_T2": "#7B8794"}
REQUIREMENT_COLOR = "#334E68"


def _bootstrap_mean(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    indices = rng.integers(0, len(values), size=(4000, len(values)))
    samples = values[indices].mean(axis=1)
    return float(values.mean()), *np.percentile(samples, [2.5, 97.5])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.trials.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    groups: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["controller"], int(row["fleet_size"]))].append(row)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.titlesize": 9.2,
            "axes.labelsize": 7.4,
            "axes.linewidth": 0.55,
            "legend.fontsize": 6.4,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    # Match the exported context-profile PDF width, not just its font sizes.
    # This gives both figures the same typography when scaled to columnwidth.
    figure, axes = plt.subplots(1, 2, figsize=(298.8440366379 / 72, 2.65))
    figure.subplots_adjust(left=0.11, right=0.985, top=0.80, bottom=0.28, wspace=0.42)
    rng = np.random.default_rng(20261001)
    for controller, label in LABELS.items():
        sizes = sorted(size for name, size in groups if name == controller)
        target_stats = []
        miss_stats = []
        for size in sizes:
            group = groups[(controller, size)]
            target_stats.append(
                _bootstrap_mean(np.asarray([float(row["targets_found"]) for row in group]), rng)
            )
            miss_stats.append(
                _bootstrap_mean(
                    np.asarray([float(row["deadline_miss_rate"]) for row in group]), rng
                )
            )
        for axis, stats in zip(axes, (target_stats, miss_stats), strict=True):
            mean = np.asarray([row[0] for row in stats])
            low = np.asarray([row[1] for row in stats])
            high = np.asarray([row[2] for row in stats])
            axis.plot(
                sizes,
                mean,
                color=COLORS[controller],
                marker="o",
                markersize=3.8,
                linewidth=1.7,
                label=label,
            )
            axis.fill_between(sizes, low, high, color=COLORS[controller], alpha=0.16)

    throughput_limits = axes[0].get_ylim()
    fleet_sizes = np.asarray(sorted({size for _, size in groups}))
    # An 80% discovery rate corresponds to 0.8*N targets for fleet size N.
    (_discovery_requirement,) = axes[0].plot(
        fleet_sizes,
        0.8 * fleet_sizes,
        color=REQUIREMENT_COLOR,
        linestyle="--",
        linewidth=1.0,
        label="80% discovery",
        zorder=2,
    )
    # Keep the original data-driven range; clip the reference line at its edge.
    axes[0].set_ylim(throughput_limits)
    miss_requirement = axes[1].axhline(
        0.05,
        color=REQUIREMENT_COLOR,
        linestyle="--",
        linewidth=1.0,
        label="5% misses",
        zorder=2,
    )

    axes[0].set_title("(a) Throughput", loc="left", fontweight="semibold")
    axes[0].set_ylabel("Targets found in 30 s")
    axes[1].set_title("(b) Deadline misses", loc="left", fontweight="semibold")
    axes[1].set_ylabel("Deadline-miss rate")
    axes[1].set_ylim(-0.03, 1.03)
    for axis in axes:
        axis.set_xlabel("Concurrent search agents")
        # Sparse labels keep the one-column layout readable; all points remain.
        axis.set_xticks((1, 16, 32, 48, 64))
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(miss_requirement)
    labels.append("5% misses")
    figure.legend(
        handles,
        labels,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=4,
        columnspacing=0.6,
        handlelength=1.5,
        handletextpad=0.4,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Do not crop the canvas: the native width controls final printed text size.
    figure.savefig(args.output, bbox_inches=None)
    figure.savefig(args.output.with_suffix(".png"), dpi=300, bbox_inches=None)
    plt.close(figure)


if __name__ == "__main__":
    main()
