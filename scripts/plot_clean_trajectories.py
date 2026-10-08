"""Clean, explicitly readability-selected illustrations of saved test runs."""

import ast
import itertools
import json
import statistics
from collections import Counter

import matplotlib as mpl
import matplotlib.pyplot as plt
from active_inference_navigation.mos import sample_mos_instance
from inspect_trajectory_candidates import load, metrics
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from plot_figure6_controller_behavior import ADAPTIVE_DIR, RESOLUTION_COLORS, ROOT, _allocation

OUT = ROOT / "output/pdf/figure_adaptive_trajectories_clean"


def select(episodes, trajectories):
    successful = [e for e in episodes if e["success"] == "True"]
    failed = [e for e in episodes if e["success"] == "False"]
    median = statistics.median(float(e["task_cost"]) for e in successful)
    near = lambda e: (abs(float(e["task_cost"]) - median), int(e["instance_seed"]))
    info = {
        int(e["instance_seed"]): metrics(trajectories[int(e["instance_seed"])]) for e in episodes
    }
    simple = [
        e
        for e in successful
        if info[int(e["instance_seed"])]["retraced_edges"] == 0
        and 2 <= info[int(e["instance_seed"])]["switches"] <= 4
    ]

    def clustered(e):
        rows = trajectories[int(e["instance_seed"])]
        positions = [
            (int(r["x"]), int(r["y"]))
            for i, r in enumerate(rows)
            if i and r["source_allocation"] != rows[i - 1]["source_allocation"]
        ]
        distances = [abs(a[0] - b[0]) + abs(a[1] - b[1]) for a, b in itertools.pairwise(positions)]
        return any(a <= 1 and b <= 1 for a, b in itertools.pairwise(distances))

    varied = [
        e
        for e in successful
        if info[int(e["instance_seed"])]["retraced_edges"] == 0
        and 5 <= info[int(e["instance_seed"])]["switches"] <= 8
        and info[int(e["instance_seed"])]["unique_switch_positions"]
        == info[int(e["instance_seed"])]["switches"]
        and not clustered(e)
    ]
    failure_median = statistics.median(float(e["task_cost"]) for e in failed)
    failure = min(
        [e for e in failed if info[int(e["instance_seed"])]["switches"] <= 8],
        key=lambda e: (
            info[int(e["instance_seed"])]["retraced_edges"],
            abs(float(e["task_cost"]) - failure_median),
            int(e["instance_seed"]),
        ),
    )
    return [min(simple, key=near), min(varied, key=near), failure], info


def place_labels(ax, changes, points, obstacles, landmarks, label_half_width=1.35):
    """Place compact allocation labels in free space with short leader lines."""
    placed = []
    offsets = [
        (dx, dy)
        for distance in (2.0, 3.5, 5.0)
        for dx, dy in [
            (distance, 1.3),
            (-distance, 1.3),
            (distance, -1.3),
            (-distance, -1.3),
            (0, distance),
            (0, -distance),
        ]
    ]
    for index, (x, y, allocation) in enumerate(changes):
        gamma, tau = _allocation(allocation)
        candidates = []
        for dx, dy in offsets:
            cx, cy = x + dx, y + dy
            if not (
                max(1.2, label_half_width + 0.3) <= cx <= min(17.8, 19 - label_half_width)
                and 1.6 <= cy <= 18.3
            ):
                continue
            box = (cx - label_half_width, cx + label_half_width, cy - 0.6, cy + 0.6)
            inside = lambda p, box=box: (
                box[0] - 0.3 <= p[0] <= box[1] + 0.3 and box[2] - 0.3 <= p[1] <= box[3] + 0.3
            )
            collision = sum(
                1
                for b in placed
                if box[0] < b[1] + 0.5
                and box[1] > b[0] - 0.5
                and box[2] < b[3] + 0.4
                and box[3] > b[2] - 0.4
            )
            score = (
                200 * collision
                + 20 * sum(inside(p) for p in landmarks)
                + 5 * sum(inside(p) for p in obstacles)
                + 3 * sum(inside(p) for p in points)
                + 0.3 * (dx * dx + dy * dy)
            )
            candidates.append((score, cx, cy, box))
        _, cx, cy, box = min(candidates, key=lambda c: c[0])
        placed.append(box)
        ax.annotate(
            rf"$({gamma},{tau})$",
            xy=(x, y),
            xytext=(cx, cy),
            ha="center",
            va="center",
            fontsize=7,
            color="#172B4D",
            bbox={"boxstyle": "round,pad=.12", "fc": "white", "ec": "none", "alpha": 0.95},
            arrowprops={
                "arrowstyle": "-",
                "color": "#7B8794",
                "lw": 0.6,
                "shrinkA": 2,
                "shrinkB": 3,
            },
            zorder=10,
        )


def draw(ax, episode, rows, title):
    instance = sample_mos_instance(int(episode["instance_seed"]))
    start = instance.layout.start
    end = (int(rows[-1]["next_x"]), int(rows[-1]["next_y"]))
    assert tuple(ast.literal_eval(episode["target"])) == instance.target
    assert len(rows) == int(episode["steps"])
    assert start == (int(rows[0]["x"]), int(rows[0]["y"]))
    assert dict(Counter(r["source_allocation"] for r in rows)) == ast.literal_eval(
        episode["allocation_steps"]
    )
    for x, y in instance.layout.blocked:
        ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, fc="#7B8794", ec="none", zorder=1))
    changes = []
    points = []
    previous = None
    for r in rows:
        gamma, _tau = _allocation(r["source_allocation"])
        x, y, nx, ny = [int(r[k]) for k in ("x", "y", "next_x", "next_y")]
        points.append((x, y))
        ax.plot(
            [x, nx],
            [y, ny],
            color=RESOLUTION_COLORS[gamma],
            lw=2.15,
            solid_capstyle="round",
            zorder=3,
        )
        if previous != r["source_allocation"]:
            changes.append((x, y, r["source_allocation"]))
            if previous is not None:
                ax.scatter(x, y, s=20, fc="white", ec="#172B4D", lw=0.7, zorder=6)
        previous = r["source_allocation"]
    ax.scatter(*start, s=27, fc="#172B4D", ec="white", lw=0.4, zorder=8)
    ax.scatter(*end, s=29, marker="X", color="#172B4D", ec="white", lw=0.4, zorder=8)
    ax.scatter(*instance.target, s=70, marker="*", fc="#E69F00", ec="#172B4D", lw=0.6, zorder=9)
    place_labels(ax, changes, points, instance.layout.blocked, [start, end, instance.target])
    ax.set(
        xlim=(-0.5, 19.5),
        ylim=(-0.5, 19.5),
        aspect="equal",
        xticks=[0, 5, 10, 15, 19],
        yticks=[0, 5, 10, 15, 19],
    )
    ax.set_xlabel("Grid x", labelpad=1)
    ax.set_ylabel("Grid y", labelpad=1)
    ax.set_title(title, loc="left", fontsize=8.5, fontweight="semibold", pad=6)
    ax.text(
        0.025,
        0.025,
        f"Seed {episode['instance_seed']} | {episode['steps']} steps",
        transform=ax.transAxes,
        fontsize=6.5,
        bbox={"fc": "white", "ec": "none", "alpha": 0.92, "pad": 1.5},
        zorder=12,
    )
    return changes


def main():
    episodes, trajectories = load()
    assert len(episodes) == 100 and sum(e["success"] == "True" for e in episodes) == 84
    selected, info = select(episodes, trajectories)
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.labelsize": 7.4,
            "axes.linewidth": 0.55,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.85))
    fig.subplots_adjust(left=0.05, right=0.995, bottom=0.20, top=0.91, wspace=0.24)
    titles = ["(a) Few allocation changes", "(b) Variable allocation", "(c) Unsuccessful search"]
    records = []
    for ax, e, title in zip(axes, selected, titles):
        seed = int(e["instance_seed"])
        rows = sorted(trajectories[seed], key=lambda r: int(r["step"]))
        changes = draw(ax, e, rows, title)
        records.append(
            {
                "instance_seed": seed,
                "success": e["success"] == "True",
                "steps": int(e["steps"]),
                "task_cost": float(e["task_cost"]),
                "metrics": info[seed],
                "allocation_changes": changes,
            }
        )
    handles = [
        Line2D([], [], color=c, lw=2, label=rf"$\gamma={g}$")
        for g, c in RESOLUTION_COLORS.items()
        if g != 20
    ]
    handles += [
        Line2D([], [], marker="o", mfc="#172B4D", mec="white", ls="none", label="Start"),
        Line2D(
            [],
            [],
            marker="*",
            mfc="#E69F00",
            mec="#172B4D",
            ls="none",
            markersize=7,
            label="Target",
        ),
        Line2D([], [], marker="X", color="#172B4D", ls="none", label="End"),
        Line2D(
            [], [], marker="o", mfc="white", mec="#172B4D", ls="none", label="Allocation change"
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        ncol=7,
        frameon=False,
        fontsize=7,
        columnspacing=1,
        handlelength=1.35,
        handletextpad=0.35,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".svg"):
        fig.savefig(OUT.with_suffix(suffix), bbox_inches="tight", pad_inches=0.035)
    png = ROOT / "output/png/figure_adaptive_trajectories_clean.png"
    fig.savefig(png, dpi=240, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)
    metadata = {
        "source": str(ADAPTIVE_DIR),
        "selection": "Illustrative examples selected for readability, not unbiased representative samples.",
        "rules": [
            "Success without retraced edges and 2-4 switches; closest to median successful task cost, ties by lowest seed.",
            "Success without retraced edges and 5-8 switches, all at distinct positions and no three consecutive switches separated by at most one cell each; closest to median successful task cost, ties by lowest seed.",
            "Failure with at most 8 switches; minimum number of retraced edges, ties by proximity to median failed task cost then lowest seed.",
        ],
        "notes": "Every executed allocation change and the initial allocation are labeled as (gamma,tau). Retraced segments can overlap. True target is shown only for evaluation visualization.",
        "episodes": records,
    }
    OUT.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
