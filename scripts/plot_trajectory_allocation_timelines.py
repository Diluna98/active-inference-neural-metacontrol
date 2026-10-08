"""Frozen successful search trajectories with executed allocation timelines."""

import argparse
import ast
import json
from collections import Counter

import matplotlib as mpl
import matplotlib.pyplot as plt
from active_inference_navigation.mos import sample_mos_instance
from inspect_trajectory_candidates import load
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from plot_clean_trajectories import place_labels
from plot_figure6_controller_behavior import RESOLUTION_COLORS, ROOT, _allocation
from select_context_coverage_examples import select


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--maps-only", action="store_true")
    args = parser.parse_args()
    selection = select(successful_only=True)
    episodes, trajectories = load()
    indexed = {int(e["instance_seed"]): e for e in episodes}
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.labelsize": 7.4,
            "axes.linewidth": 0.55,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "pdf.fonttype": 42,
        }
    )
    # Match the fleet/context-profile native PDF width for columnwidth typography.
    fig = plt.figure(figsize=(298.8440366379 / 72, 2.65) if args.maps_only else (7.16, 4.15))
    if args.maps_only:
        grid = fig.add_gridspec(1, 2, left=0.02, right=0.98, top=0.87, bottom=0.12, wspace=0.075)
    else:
        grid = fig.add_gridspec(
            3,
            2,
            height_ratios=[3.0, 0.23, 0.55],
            left=0.075,
            right=0.985,
            top=0.93,
            bottom=0.17,
            hspace=0.35,
            wspace=0.26,
        )
    records = []
    for panel, s in enumerate(selection["selected"]):
        seed = s["instance_seed"]
        e = indexed[seed]
        instance = sample_mos_instance(seed)
        rows = sorted(trajectories[seed], key=lambda r: int(r["step"]))
        assert len(rows) == int(e["steps"])
        assert dict(Counter(r["source_allocation"] for r in rows)) == ast.literal_eval(
            e["allocation_steps"]
        )
        ax = fig.add_subplot(grid[0, panel])
        for x in range(20):
            for y in range(20):
                if instance.layout.is_free((x, y)) and instance.layout.target_is_visible(
                    (x, y), instance.target
                ):
                    ax.add_patch(
                        Rectangle(
                            (x - 0.5, y - 0.5), 1, 1, fc="#E69F00", alpha=0.18, ec="none", zorder=0
                        )
                    )
        for x, y in instance.layout.blocked:
            ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, fc="#7B8794", ec="none", zorder=1))
        gamma = []
        depth = []
        previous = None
        changes = []
        points = []
        for r in rows:
            g, t = _allocation(r["source_allocation"])
            gamma.append(g)
            depth.append(t)
            x, y, nx, ny = [int(r[k]) for k in ("x", "y", "next_x", "next_y")]
            points.append((x, y))
            if previous != r["source_allocation"]:
                changes.append((x, y, r["source_allocation"]))
            ax.plot([x, nx], [y, ny], color=RESOLUTION_COLORS[g], lw=1.9, zorder=3)
            if previous and previous != r["source_allocation"]:
                ax.scatter(x, y, s=14, fc="white", ec="#52606D", lw=0.6, zorder=4)
            previous = r["source_allocation"]
        ax.scatter(*instance.layout.start, s=24, color="#172B4D", zorder=5)
        ax.scatter(
            int(rows[-1]["next_x"]),
            int(rows[-1]["next_y"]),
            s=27,
            marker="X",
            color="#172B4D",
            zorder=5,
        )
        ax.scatter(*instance.target, s=66, marker="*", fc="#E69F00", ec="#172B4D", lw=0.6, zorder=5)
        if args.maps_only:
            place_labels(
                ax,
                changes[:-1],
                points,
                instance.layout.blocked,
                [
                    instance.layout.start,
                    instance.target,
                    (int(rows[-1]["next_x"]), int(rows[-1]["next_y"])),
                ],
                label_half_width=1.9,
            )
            allocation_annotations = list(ax.texts)
            if seed == 20088:
                allocation_annotations[0].set_position((2.8, 12))
                allocation_annotations[0].set_ha("left")
                allocation_annotations[1].set_position((2, 15.3))
            elif seed == 20045:
                # Keep the early cluster entirely to the left of the vertical path.
                for index, y in ((1, 16.2), (2, 14.3), (3, 12.9), (4, 11.5), (5, 8.7)):
                    allocation_annotations[index].set_position((2.4, y))
        ax.set(
            xlim=(-0.5, 19.5),
            ylim=(-0.5, 19.5),
            aspect="equal",
            xticks=[0, 5, 10, 15, 19],
            yticks=[0, 5, 10, 15, 19],
        )
        if args.maps_only:
            ax.set(xticks=[], yticks=[])
            ax.tick_params(left=False, bottom=False)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.55)
                spine.set_color("#AAB3BD")
        else:
            ax.set_xlabel("Grid x", labelpad=1)
            ax.set_ylabel("Grid y", labelpad=1)
        ax.set_title(
            f"({chr(97 + panel)}) Instance {seed}",
            loc="left",
            fontsize=8.5,
            fontweight="bold",
            pad=6,
        )
        if args.maps_only:
            online = float(e["total_online_compute_ms"])
            assert (
                abs(
                    online
                    - sum(
                        float(e[k]) for k in ("task_inference_ms", "meta_inference_ms", "switch_ms")
                    )
                )
                < 1e-6
            )
            ax.text(
                0.025,
                0.025,
                f"Steps: {len(rows)}\nTask cost: {float(e['task_cost']):g}\nOnline {online / 1000:.3f} s",
                transform=ax.transAxes,
                fontsize=6.0,
                linespacing=1.3,
                bbox={"fc": "white", "ec": "none", "alpha": 0.94, "pad": 1.5},
                zorder=12,
            )
            ax.text(
                0.025,
                0.985,
                r"Allocation label: $(\gamma,\tau)$",
                transform=ax.transAxes,
                fontsize=6.3,
                va="top",
                color="#172B4D",
                zorder=12,
            )
            records.append(
                {
                    "seed": seed,
                    "steps": len(rows),
                    "task_cost": float(e["task_cost"]),
                    "total_online_compute_ms": online,
                    "labelled_allocations": changes[:-1],
                    "unlabelled_last_switch": changes[-1],
                }
            )
            continue
        ax.text(
            0.02,
            0.02,
            f"{len(rows)} steps | Success",
            transform=ax.transAxes,
            fontsize=6.7,
            bbox={"fc": "white", "ec": "none", "alpha": 0.9, "pad": 1},
        )
        strip = fig.add_subplot(grid[1, panel])
        dep = fig.add_subplot(grid[2, panel], sharex=strip)
        for i, g in enumerate(gamma):
            strip.add_patch(Rectangle((i + 0.5, 0), 1, 1, fc=RESOLUTION_COLORS[g], ec="none"))
        strip.set(xlim=(0.5, 43.5), ylim=(0, 1), yticks=[], xticks=[])
        strip.set_ylabel(r"$\gamma$", rotation=0, labelpad=10, va="center")
        strip.tick_params(axis="x", bottom=False, labelbottom=False)
        for spine in strip.spines.values():
            spine.set_visible(False)
        dep.stairs(
            depth, [i + 0.5 for i in range(len(depth) + 1)], color="#6B4C9A", lw=1.3, baseline=None
        )
        dep.set(ylim=(0.75, 3.25), yticks=[1, 2, 3], xticks=[1, 10, 20, 30, 40])
        dep.set_ylabel(r"$\tau$", rotation=0, labelpad=10, va="center")
        dep.set_xlabel("Executed decision step", labelpad=1)
        dep.spines[["top", "right"]].set_visible(False)
        dep.grid(axis="y", color="#E5E9EF", lw=0.5)
        records.append(
            {
                "seed": seed,
                "steps": len(rows),
                "executed_resolutions": gamma,
                "executed_depths": depth,
            }
        )
    handles = [
        Line2D([], [], color=RESOLUTION_COLORS[g], lw=2, label=rf"$\gamma={g}$") for g in (2, 5, 10)
    ]
    handles += [
        Line2D([], [], marker="o", color="#172B4D", ls="none", label="Start"),
        Line2D([], [], marker="*", mfc="#E69F00", mec="#172B4D", ls="none", label="Target"),
        Line2D([], [], marker="X", color="#172B4D", ls="none", label="End"),
        Line2D([], [], marker="o", mfc="white", mec="#52606D", ls="none", label="Switch"),
        Patch(fc="#E69F00", alpha=0.18, label="Find-success region"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.002 if args.maps_only else 0.015),
        ncol=8,
        frameon=False,
        fontsize=6.4 if args.maps_only else 6.6,
        columnspacing=0.55 if args.maps_only else 0.75,
        handlelength=1.05 if args.maps_only else 1.2,
        handletextpad=0.3,
    )
    name = (
        "figure_trajectory_allocations_costs"
        if args.maps_only
        else "figure_trajectory_allocation_timelines"
    )
    out = ROOT / "output/pdf" / name
    bbox = None if args.maps_only else "tight"
    for suffix in (".pdf", ".svg"):
        fig.savefig(out.with_suffix(suffix), bbox_inches=bbox, pad_inches=0.04)
    fig.savefig(ROOT / "output/png" / f"{name}.png", dpi=240, bbox_inches=bbox, pad_inches=0.04)
    plt.close(fig)
    out.with_suffix(".json").write_text(
        json.dumps(
            {
                "rule": selection["rule"],
                "episodes": records,
                "timing": "Path segments and timelines use source_allocation, the configuration used for the executed decision. Timelines include stationary actions; spatial traversals may overlap.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(out)


if __name__ == "__main__":
    main()
