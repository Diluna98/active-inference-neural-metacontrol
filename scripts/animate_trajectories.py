"""Animate saved evaluation trajectories without rerunning the controller."""

from __future__ import annotations

import argparse
from itertools import pairwise
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from active_inference_navigation.mos import sample_mos_instance
from inspect_trajectory_candidates import load
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from plot_figure6_controller_behavior import RESOLUTION_COLORS, ROOT, _allocation

DEPTH_MARKERS = {1: "o", 2: "s", 3: "^"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance-seeds", nargs="+", type=int, default=[20088, 20045])
    parser.add_argument("--fps", type=int, default=3)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/media/adaptive_search.gif")
    args = parser.parse_args()
    if not 1 <= len(args.instance_seeds) <= 3 or args.fps < 1:
        parser.error("Specify one to three instance seeds and a positive frame rate.")

    episodes, trajectories = load()
    indexed = {int(row["instance_seed"]): row for row in episodes}
    records = []
    for seed in args.instance_seeds:
        rows = sorted(trajectories[seed], key=lambda row: int(row["step"]))
        episode = indexed[seed]
        assert len(rows) == int(episode["steps"])
        assert (
            abs(sum(float(row["step_cost"]) for row in rows) - float(episode["task_cost"])) < 1e-6
        )
        for previous, current in pairwise(rows):
            assert (previous["next_x"], previous["next_y"]) == (current["x"], current["y"])
        records.append((seed, sample_mos_instance(seed), rows, episode))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, axes = plt.subplots(1, len(records), figsize=(4.5 * len(records), 5.5), squeeze=False)
    fig.subplots_adjust(left=0.035, right=0.965, bottom=0.23, top=0.87, wspace=0.12)
    fig.suptitle(
        "Context-dependent allocation during object search", fontsize=14, weight="semibold"
    )
    legend = [
        Line2D([], [], color=color, lw=3, label=rf"$\gamma={resolution}$")
        for resolution, color in RESOLUTION_COLORS.items()
    ] + [
        Line2D([], [], marker=marker, color="#172B4D", linestyle="none", label=rf"$\tau={depth}$")
        for depth, marker in DEPTH_MARKERS.items()
    ]
    fig.legend(
        handles=legend, loc="lower center", bbox_to_anchor=(0.5, 0.105), ncol=7, frameon=False
    )
    fig.legend(
        handles=[
            Line2D([], [], marker="*", color="#E69F00", linestyle="none", label="Target"),
            Patch(facecolor="#FBEED0", label="Find-success region"),
            Line2D(
                [],
                [],
                marker="o",
                markerfacecolor="white",
                color="#52606D",
                linestyle="none",
                label="Allocation change",
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.055),
        ncol=3,
        frameon=False,
    )
    fig.text(
        0.5,
        0.025,
        "Saved evaluation traces; playback is not real time. Target shown for visualization only.",
        ha="center",
        fontsize=8,
        color="#52606D",
    )

    def update(frame):
        for ax, (seed, instance, rows, episode) in zip(axes[0], records, strict=True):
            ax.clear()
            count = min(frame, len(rows))
            for x in range(20):
                for y in range(20):
                    if instance.layout.is_free((x, y)) and instance.layout.target_is_visible(
                        (x, y), instance.target
                    ):
                        ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, fc="#FBEED0", ec="none"))
            for x, y in instance.layout.blocked:
                ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, fc="#7B8794", ec="none"))
            for index, row in enumerate(rows[:count]):
                resolution, _ = _allocation(row["source_allocation"])
                x, y, nx, ny = (int(row[key]) for key in ("x", "y", "next_x", "next_y"))
                ax.plot([x, nx], [y, ny], color=RESOLUTION_COLORS[resolution], lw=2.7, zorder=3)
                if index and row["source_allocation"] != rows[index - 1]["source_allocation"]:
                    ax.scatter(x, y, s=35, facecolors="white", edgecolors="#52606D", zorder=4)
            row = rows[max(count - 1, 0)]
            resolution, depth = _allocation(row["source_allocation"])
            position = (
                instance.layout.start if count == 0 else (int(row["next_x"]), int(row["next_y"]))
            )
            ax.scatter(*instance.target, marker="*", s=135, fc="#E69F00", ec="#172B4D", zorder=5)
            ax.scatter(
                *position,
                marker=DEPTH_MARKERS[depth],
                s=80,
                fc=RESOLUTION_COLORS[resolution],
                ec="#172B4D",
                lw=1.2,
                zorder=6,
            )
            finished = count == len(rows)
            status = (
                ("Found" if episode["success"].lower() == "true" else "Time limit")
                if finished
                else (row["action"].title() if count else "Start")
            )
            ax.set_title(
                f"Instance {seed}  |  Step {count}/{len(rows)}\n"
                + rf"Executed allocation: $(\gamma,\tau)=({resolution},{depth})$"
                + f"  |  {status}",
                fontsize=10,
                pad=8,
            )
            cost = sum(float(item["step_cost"]) for item in rows[:count])
            ax.text(
                0.5,
                -0.07,
                f"Task cost so far: {cost:g}  |  Recorded episode compute: {float(episode['total_online_compute_ms']) / 1000:.3f} s",
                ha="center",
                transform=ax.transAxes,
                fontsize=8,
            )
            ax.set(xlim=(-0.5, 19.5), ylim=(-0.5, 19.5), aspect="equal", xticks=[], yticks=[])
            for spine in ax.spines.values():
                spine.set_color("#AAB3BD")
                spine.set_linewidth(0.7)

    frame_count = max(len(record[2]) for record in records)
    # Hold the final state briefly before the GIF loops.
    frames = list(range(frame_count + 1)) + [frame_count] * (args.fps * 2)
    animation = FuncAnimation(fig, update, frames=frames, interval=1000 / args.fps, repeat=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    animation.save(args.output, writer=PillowWriter(fps=args.fps), dpi=110)
    update(frame_count)
    fig.savefig(args.output.with_suffix(".png"), dpi=110)
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
