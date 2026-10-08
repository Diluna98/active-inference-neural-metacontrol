"""Create Figure 6: aggregate and representative metacontroller behavior."""

from __future__ import annotations

import csv
import statistics
from collections import Counter
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from active_inference_navigation.mos import GRID_SIZE, sample_mos_instance
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
ADAPTIVE_DIR = ROOT / "artifacts" / "results" / "final_profiling_single_worker_20000_100"
CONTEXT_DIR = ROOT / "artifacts" / "results" / "figure6_context_replay_20000_100"
POMCP_DIR = ROOT / "artifacts" / "results" / "pomcp_test_20000_100_s250_d30"
OUTPUT = ROOT / "output" / "pdf" / "figure6_controller_behavior.pdf"

RESOLUTIONS = (2, 5, 10, 20)
DEPTHS = (1, 2, 3)
RESOLUTION_COLORS = {
    2: "#0072B2",
    5: "#009E73",
    10: "#E69F00",
    20: "#CC79A7",
}
DEPTH_WIDTHS = {1: 1.25, 2: 2.25, 3: 3.35}
FISHER_COLORS = tuple(mpl.colormaps["viridis"](value) for value in (0.12, 0.38, 0.64, 0.9))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _as_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def _allocation(value: str) -> tuple[int, int]:
    resolution, depth = value.removeprefix("g").split("_T")
    return int(resolution), int(depth)


def _select_examples(
    adaptive_episodes: dict[int, dict[str, str]],
    pomcp_episodes: dict[int, dict[str, str]],
) -> tuple[int, int, int]:
    learned_only = [
        seed
        for seed, row in adaptive_episodes.items()
        if _as_bool(row["success"]) and not _as_bool(pomcp_episodes[seed]["success"])
    ]
    largest_improvement = max(
        learned_only,
        key=lambda seed: (
            float(pomcp_episodes[seed]["task_cost"]) - float(adaptive_episodes[seed]["task_cost"]),
            -seed,
        ),
    )

    successful = [
        seed
        for seed, row in adaptive_episodes.items()
        if _as_bool(row["success"]) and seed != largest_improvement
    ]
    median_steps = statistics.median(float(adaptive_episodes[seed]["steps"]) for seed in successful)
    median_task_cost = statistics.median(
        float(adaptive_episodes[seed]["task_cost"]) for seed in successful
    )
    ordinary_success = min(
        successful,
        key=lambda seed: (
            abs(float(adaptive_episodes[seed]["steps"]) - median_steps),
            abs(float(adaptive_episodes[seed]["task_cost"]) - median_task_cost),
            seed,
        ),
    )

    failures = [seed for seed, row in adaptive_episodes.items() if not _as_bool(row["success"])]
    median_compute = statistics.median(
        float(adaptive_episodes[seed]["total_online_compute_ms"]) for seed in failures
    )
    representative_failure = min(
        failures,
        key=lambda seed: (
            abs(float(adaptive_episodes[seed]["total_online_compute_ms"]) - median_compute),
            seed,
        ),
    )
    return largest_improvement, ordinary_success, representative_failure


def _annotation_color(rgba: tuple[float, float, float, float]) -> str:
    red, green, blue, _ = rgba
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "#102A43" if luminance > 0.61 else "#F7FAFC"


def _plot_frequency(ax: mpl.axes.Axes, rows: list[dict[str, str]]) -> None:
    counts = Counter(_allocation(row["source_allocation"]) for row in rows)
    total = sum(counts.values())
    values = np.asarray(
        [
            [100.0 * counts[(resolution, depth)] / total for depth in DEPTHS]
            for resolution in RESOLUTIONS
        ]
    )
    palette = mpl.colormaps["viridis"](np.linspace(0.12, 0.88, 256))
    palette[:, :3] = 0.78 * palette[:, :3] + 0.22
    color_map = mpl.colors.ListedColormap(palette)
    norm = Normalize(vmin=0.0, vmax=float(values.max()) * 1.08)
    image = ax.imshow(
        values,
        cmap=color_map,
        norm=norm,
        aspect="auto",
        interpolation="nearest",
    )
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            ax.text(
                column,
                row,
                f"{value:.1f}%",
                ha="center",
                va="center",
                fontsize=6.8,
                fontweight="normal",
                color=_annotation_color(image.cmap(image.norm(value))),
                zorder=4,
            )
    ax.set_xticks(range(len(DEPTHS)), [str(value) for value in DEPTHS])
    ax.set_yticks(range(len(RESOLUTIONS)), [str(value) for value in RESOLUTIONS])
    ax.set_xlabel(r"Planning depth $\tau$")
    ax.set_ylabel(r"Resolution $\gamma$")
    ax.set_xticks(np.arange(-0.5, len(DEPTHS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(RESOLUTIONS), 1), minor=True)
    ax.grid(which="minor", color="#F7FAFC", linewidth=0.75)
    ax.tick_params(which="minor", bottom=False, left=False)


def _validate_context_replay(frozen: list[dict[str, str]], replay: list[dict[str, str]]) -> None:
    columns = (
        "instance_seed",
        "step",
        "x",
        "y",
        "action",
        "next_x",
        "next_y",
        "source_allocation",
        "selected_allocation",
        "success",
    )
    frozen_decisions = [tuple(row[column] for column in columns) for row in frozen]
    replay_decisions = [tuple(row[column] for column in columns) for row in replay]
    if frozen_decisions != replay_decisions:
        raise RuntimeError("context replay does not reproduce the frozen trajectory")


def _context_profiles(
    rows: list[dict[str, str]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    valid = [row for row in rows if row["expected_information_gain_nats"]]
    entropy = np.asarray([float(row["belief_entropy_canonical"]) for row in valid])
    information = np.asarray([float(row["posterior_weighted_fisher"]) for row in valid])
    entropy_edges = np.quantile(entropy, (0.25, 0.5, 0.75))
    information_edges = np.quantile(information, (0.25, 0.5, 0.75))
    confidence = np.asarray([float(row["policy_margin"]) for row in valid])
    confidence_edges = np.quantile(confidence, (0.25, 0.5, 0.75))
    entropy_bins = np.digitize(entropy, entropy_edges, right=True)
    information_bins = np.digitize(information, information_edges, right=True)
    confidence_bins = np.digitize(confidence, confidence_edges, right=True)
    selected = np.asarray([_allocation(row["selected_allocation"]) for row in valid])
    selected_belief_space_size = np.square(selected[:, 0].astype(float))
    episode_seeds = np.asarray([int(row["instance_seed"]) for row in valid])
    unique_seeds, seed_indices = np.unique(episode_seeds, return_inverse=True)

    resolution_sums = np.zeros((unique_seeds.size, 4, 4))
    resolution_counts = np.zeros((unique_seeds.size, 4, 4), dtype=int)
    depth_sums = np.zeros((unique_seeds.size, 4, 4))
    depth_counts = np.zeros((unique_seeds.size, 4, 4), dtype=int)
    np.add.at(
        resolution_sums,
        (seed_indices, information_bins, entropy_bins),
        selected_belief_space_size,
    )
    np.add.at(
        resolution_counts,
        (seed_indices, information_bins, entropy_bins),
        1,
    )
    np.add.at(
        depth_sums,
        (seed_indices, information_bins, confidence_bins),
        selected[:, 1],
    )
    np.add.at(
        depth_counts,
        (seed_indices, information_bins, confidence_bins),
        1,
    )

    resolution = resolution_sums.sum(axis=0) / resolution_counts.sum(axis=0)
    depth = depth_sums.sum(axis=0) / depth_counts.sum(axis=0)
    generator = np.random.default_rng(20260929)
    resolution_bootstrap = np.empty((5000, 4, 4))
    depth_bootstrap = np.empty((5000, 4, 4))
    for sample in range(5000):
        drawn = generator.integers(0, unique_seeds.size, size=unique_seeds.size)
        resolution_bootstrap[sample] = resolution_sums[drawn].sum(axis=0) / (
            resolution_counts[drawn].sum(axis=0)
        )
        depth_bootstrap[sample] = depth_sums[drawn].sum(axis=0) / (depth_counts[drawn].sum(axis=0))
    resolution_low, resolution_high = np.quantile(resolution_bootstrap, (0.025, 0.975), axis=0)
    depth_low, depth_high = np.quantile(depth_bootstrap, (0.025, 0.975), axis=0)
    return resolution, resolution_low, resolution_high, depth, depth_low, depth_high


def _plot_profile(
    ax: mpl.axes.Axes,
    means: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    kind: str,
) -> None:
    x_values = np.arange(4)
    for fisher_quartile, color in enumerate(FISHER_COLORS):
        ax.plot(
            x_values,
            means[fisher_quartile],
            color=color,
            marker="o",
            markersize=3.5,
            linewidth=1.35,
            zorder=3,
        )
        ax.fill_between(
            x_values,
            lower[fisher_quartile],
            upper[fisher_quartile],
            color=color,
            alpha=0.13,
            linewidth=0,
            zorder=2,
        )
    quartiles = ("Q1", "Q2", "Q3", "Q4")
    ax.set_xticks(x_values, quartiles)
    ax.set_xlabel(
        "Belief entropy (low to high)"
        if kind == "resolution"
        else "Policy confidence (low to high)"
    )
    if kind == "resolution":
        ax.set_ylabel("Mean belief-space size (states)")
        ax.set_ylim(20, 110)
        ax.set_yticks((25, 50, 75, 100))
    else:
        ax.set_ylabel(r"Mean selected $\tau_{t+1}$")
        ax.set_ylim(1.4, 2.02)
        ax.set_yticks((1.4, 1.6, 1.8, 2.0))
    ax.set_xlim(-0.12, 3.12)
    ax.grid(True, axis="y", color="#D9E2EC", linewidth=0.55)
    ax.set_axisbelow(True)


def _plot_trajectory(
    ax: mpl.axes.Axes,
    rows: list[dict[str, str]],
    episode: dict[str, str],
    *,
    seed: int,
    title: str,
    subtitle: str,
) -> None:
    instance = sample_mos_instance(seed)
    layout = instance.layout
    informative = [
        (x, y)
        for x in range(GRID_SIZE)
        for y in range(GRID_SIZE)
        if layout.is_free((x, y)) and layout.target_is_visible((x, y), instance.target)
    ]
    for x, y in informative:
        ax.add_patch(
            Rectangle(
                (x - 0.5, y - 0.5),
                1,
                1,
                facecolor="#56B4E9",
                edgecolor="none",
                alpha=0.17,
                zorder=0,
            )
        )
    for x, y in layout.blocked:
        ax.add_patch(
            Rectangle(
                (x - 0.5, y - 0.5),
                1,
                1,
                facecolor="#7B8794",
                edgecolor="#52616B",
                linewidth=0.25,
                zorder=1,
            )
        )

    previous_allocation: str | None = None
    for row in rows:
        resolution, depth = _allocation(row["source_allocation"])
        x, y = int(row["x"]), int(row["y"])
        next_x, next_y = int(row["next_x"]), int(row["next_y"])
        ax.plot(
            (x, next_x),
            (y, next_y),
            color=RESOLUTION_COLORS[resolution],
            linewidth=DEPTH_WIDTHS[depth],
            solid_capstyle="round",
            alpha=0.95,
            zorder=4,
        )
        if previous_allocation is not None and row["source_allocation"] != previous_allocation:
            ax.scatter(
                x,
                y,
                s=18,
                marker="D",
                facecolor="#FFFFFF",
                edgecolor="#172B4D",
                linewidth=0.9,
                zorder=6,
            )
        previous_allocation = row["source_allocation"]

    start = (int(rows[0]["x"]), int(rows[0]["y"]))
    end = (int(rows[-1]["next_x"]), int(rows[-1]["next_y"]))
    ax.scatter(
        *start,
        s=33,
        marker="o",
        facecolor="#FFFFFF",
        edgecolor="#172B4D",
        linewidth=1.0,
        zorder=8,
    )
    ax.scatter(
        *end,
        s=38,
        marker="X",
        facecolor="#D55E00" if not _as_bool(episode["success"]) else "#009E73",
        edgecolor="#172B4D",
        linewidth=0.65,
        zorder=8,
    )
    ax.scatter(
        *instance.target,
        s=70,
        marker="*",
        facecolor="#F0E442",
        edgecolor="#172B4D",
        linewidth=0.8,
        zorder=9,
    )
    ax.set_xlim(-0.5, GRID_SIZE - 0.5)
    ax.set_ylim(-0.5, GRID_SIZE - 0.5)
    ax.set_aspect("equal")
    ax.set_xticks((0, 5, 10, 15, 19))
    ax.set_yticks((0, 5, 10, 15, 19))
    ax.set_xlabel("Grid x")
    ax.set_ylabel("Grid y")
    ax.set_title(title, loc="left", pad=6, fontsize=9.2, fontweight="semibold")
    ax.text(
        0.02,
        0.02,
        subtitle,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.7,
        color="#334E68",
        bbox={"facecolor": "#FFFFFF", "edgecolor": "none", "alpha": 0.82, "pad": 1.8},
        zorder=10,
    )
    ax.grid(True, color="#D9E2EC", linewidth=0.28, alpha=0.55)
    ax.set_axisbelow(True)


def main() -> None:
    trajectory = _read_csv(ADAPTIVE_DIR / "adaptive_trajectory.csv")
    context_replay = _read_csv(CONTEXT_DIR / "adaptive_trajectory.csv")
    _validate_context_replay(trajectory, context_replay)
    (
        resolution_profile,
        resolution_low,
        resolution_high,
        depth_profile,
        depth_low,
        depth_high,
    ) = _context_profiles(context_replay)

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.labelsize": 7.4,
            "axes.titlesize": 9.2,
            "axes.linewidth": 0.55,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )

    figure = plt.figure(figsize=(7.16, 2.95))
    grid = figure.add_gridspec(
        1,
        3,
        left=0.075,
        right=0.985,
        top=0.89,
        bottom=0.36,
        wspace=0.46,
    )
    axes = np.asarray([figure.add_subplot(grid[0, index]) for index in range(3)])

    _plot_frequency(axes[0], trajectory)
    axes[0].set_title("(a) Allocation frequency", loc="left", pad=7, fontweight="semibold")

    _plot_profile(
        axes[1],
        resolution_profile,
        resolution_low,
        resolution_high,
        kind="resolution",
    )
    axes[1].set_title("(b) Belief-space size by context", loc="left", pad=7, fontweight="semibold")

    _plot_profile(axes[2], depth_profile, depth_low, depth_high, kind="depth")
    axes[2].set_title("(c) Depth by context", loc="left", pad=7, fontweight="semibold")
    fisher_handles = [
        Line2D(
            [0],
            [0],
            color=FISHER_COLORS[index],
            marker="o",
            markersize=3.5,
            linewidth=1.35,
            label=f"Fisher Q{index + 1}",
        )
        for index in range(4)
    ]
    figure.legend(
        handles=fisher_handles,
        loc="lower center",
        bbox_to_anchor=(0.68, 0.105),
        ncol=4,
        frameon=False,
        fontsize=6.4,
        columnspacing=1.0,
        handlelength=1.5,
        handletextpad=0.35,
    )

    figure.text(
        0.5,
        0.012,
        r"Color in (a) encodes allocation frequency. Panel (b) averages $\gamma^2$ states; "
        r"bands are episode-bootstrap 95% CIs and Q1-Q4 are Fisher quartiles.",
        ha="center",
        va="bottom",
        fontsize=6.7,
        color="#486581",
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        OUTPUT,
        format="pdf",
        metadata={"Title": "Learned metacontroller behavior"},
    )
    plt.close(figure)
    print(OUTPUT)


if __name__ == "__main__":
    main()
