"""Create the publication-ready task-computation Pareto comparison (Figure 4)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "artifacts" / "results"
OUTPUT = ROOT / "output" / "pdf" / "figure4_task_compute_pareto.pdf"


@dataclass(frozen=True)
class Result:
    key: str
    label: str
    success: float
    task_cost: float
    compute_seconds: float
    family: str


COLORS = {
    "fixed": "#7B8794",
    "learned": "#D55E00",
    "entropy": "#56B4E9",
    "fisher": "#009E73",
    "pomcp": "#0072B2",
    "mr_pomcp": "#CC79A7",
}

MARKERS = {
    "fixed": "o",
    "learned": "*",
    "entropy": "s",
    "fisher": "X",
    "pomcp": "D",
    "mr_pomcp": "v",
}


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _result(key: str, label: str, row: dict, family: str) -> Result:
    return Result(
        key=key,
        label=label,
        success=100.0 * float(row["success_rate"]),
        task_cost=float(row["mean_task_cost"]),
        compute_seconds=float(row["mean_total_compute_ms"]) / 1000.0,
        family=family,
    )


def _load_results() -> list[Result]:
    fixed_payload = _read(RESULTS / "final_profiling_single_worker_20000_100" / "summary.json")
    fixed_controllers = fixed_payload["controllers"]
    values: list[Result] = []
    for resolution in (2, 5, 10, 20):
        for depth in (1, 2, 3):
            key = f"fixed_g{resolution}_T{depth}"
            values.append(
                _result(
                    key,
                    rf"$\gamma_{{{resolution}}},\tau_{{{depth}}}$",
                    fixed_controllers[key],
                    "fixed",
                )
            )
    values.append(
        _result(
            "learned",
            "Ours",
            fixed_controllers["adaptive"],
            "learned",
        )
    )
    values.append(
        _result(
            "entropy",
            "Entropy heuristic",
            _read(RESULTS / "heuristic_entropy_bounded_test_20000_100" / "summary.json")[
                "controllers"
            ]["adaptive"],
            "entropy",
        )
    )
    values.append(
        _result(
            "fisher",
            "Fisher/surprise heuristic",
            _read(RESULTS / "heuristic_fisher_bounded_test_20000_100" / "summary.json")[
                "controllers"
            ]["adaptive"],
            "fisher",
        )
    )
    values.append(
        _result(
            "pomcp",
            "Standard POMCP",
            _read(RESULTS / "pomcp_test_20000_100_s250_d30" / "summary.json")["controller"],
            "pomcp",
        )
    )
    values.append(
        _result(
            "mr_pomcp",
            "Multi-resolution POMCP",
            _read(RESULTS / "mr_pomcp_test_20000_100_s250_d30" / "summary.json")["controller"],
            "mr_pomcp",
        )
    )
    return values


def _pareto(results: list[Result], metric: str, maximize: bool) -> list[Result]:
    ordered = sorted(results, key=lambda result: result.compute_seconds)
    frontier: list[Result] = []
    best = -np.inf if maximize else np.inf
    for result in ordered:
        value = float(getattr(result, metric))
        improves = value > best if maximize else value < best
        if improves:
            frontier.append(result)
            best = value
    return frontier


def _plot_panel(
    ax: mpl.axes.Axes,
    results: list[Result],
    *,
    metric: str,
    ylabel: str,
    panel: str,
    maximize: bool,
) -> None:
    fixed = [result for result in results if result.family == "fixed"]
    comparators = [result for result in results if result.family != "fixed"]

    ax.scatter(
        [result.compute_seconds for result in fixed],
        [getattr(result, metric) for result in fixed],
        s=24,
        marker=MARKERS["fixed"],
        facecolor="#FFFFFF",
        edgecolor=COLORS["fixed"],
        linewidth=1.15,
        alpha=0.9,
        zorder=3,
    )
    for result in comparators:
        size = 80 if result.family == "learned" else 42
        linewidth = 1.2 if result.family == "learned" else 0.85
        ax.scatter(
            result.compute_seconds,
            getattr(result, metric),
            s=size,
            marker=MARKERS[result.family],
            facecolor=COLORS[result.family],
            edgecolor="#1F2933",
            linewidth=linewidth,
            zorder=6 if result.family == "learned" else 5,
        )

    frontier = _pareto(results, metric, maximize=maximize)
    ax.plot(
        [result.compute_seconds for result in frontier],
        [getattr(result, metric) for result in frontier],
        color="#334E68",
        linestyle=(0, (4, 2.5)),
        linewidth=1.25,
        zorder=2,
        label="Empirical Pareto frontier",
    )

    fixed_labels = {"fixed_g2_T1", "fixed_g5_T2", "fixed_g10_T2"}
    fixed_offsets = {
        "fixed_g2_T1": (3, -16 if metric == "task_cost" else 6),
        "fixed_g5_T2": (-24, -5 if metric == "task_cost" else 2),
        "fixed_g10_T2": (4, 6 if metric == "task_cost" else -10),
    }
    for result in fixed:
        if result.key not in fixed_labels:
            continue
        ax.annotate(
            result.label,
            (result.compute_seconds, getattr(result, metric)),
            xytext=fixed_offsets[result.key],
            textcoords="offset points",
            fontsize=7.4,
            color="#243746",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 0.7},
            zorder=7,
        )
    """
    learned = next(result for result in comparators if result.family == "learned")
    ax.annotate(
        learned.label,
        (learned.compute_seconds, getattr(learned, metric)),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=7.7,
        fontweight="bold",
        color=COLORS[learned.family],
        zorder=8,
    )
    """

    ax.set_xscale("log")
    ax.set_xlim(0.105, 22.5)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.set_ylim((38, 157) if metric == "task_cost" else (-1, 91))
    ax.grid(True, which="major", color="#D9E2EC", linewidth=0.65)
    ax.grid(True, which="minor", axis="x", color="#EDF2F7", linewidth=0.45)
    ax.set_axisbelow(True)
    ax.set_title(panel, loc="left", pad=7, fontweight="semibold")

    arrow_style = {
        "arrowstyle": "-|>",
        "color": "#627D98",
        "linewidth": 1.15,
        "mutation_scale": 9,
    }
    if maximize:
        origin = (0.94, 0.08)
        horizontal_end = (0.79, 0.08)
        vertical_end = (0.94, 0.23)
    else:
        origin = (0.94, 0.92)
        horizontal_end = (0.79, 0.92)
        vertical_end = (0.94, 0.77)
    ax.annotate(
        "",
        xy=horizontal_end,
        xytext=origin,
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=arrow_style,
    )
    ax.annotate(
        "",
        xy=vertical_end,
        xytext=origin,
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=arrow_style,
    )


def main() -> None:
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
            "savefig.bbox": None,
            # "savefig.pad_inches": 0.04,
        }
    )
    results = _load_results()
    # Same exported width as the context-profile and fleet figures.
    figure, axes = plt.subplots(1, 2, figsize=(298.8440366379 / 72, 2.75))
    _plot_panel(
        axes[0],
        results,
        metric="task_cost",
        ylabel="Mean task cost",
        panel="(a) Task cost",
        maximize=False,
    )
    _plot_panel(
        axes[1],
        results,
        metric="success",
        ylabel="Success rate (%)",
        panel="(b) Task success",
        maximize=True,
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=MARKERS[family],
            linestyle="none",
            markerfacecolor="#FFFFFF" if family == "fixed" else COLORS[family],
            markeredgecolor="#1F2933" if family != "fixed" else COLORS[family],
            markeredgewidth=0.9,
            markersize=7.8 if family == "learned" else 6.5,
            label=label,
        )
        for family, label in (
            ("learned", "Ours"),
            ("entropy", "Entropy"),
            ("fisher", "Fisher/surprise"),
            ("pomcp", "POMCP"),
            ("mr_pomcp", "MR-POMCP"),
            ("fixed", "Fixed AIF"),
        )
    ]
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color="#334E68",
            linestyle=(0, (4, 2.5)),
            linewidth=1.25,
            label="Pareto frontier",
        )
    )
    legend_handles.append(
        Line2D(
            [0],
            [0],
            marker=r"$\rightarrow$",
            linestyle="none",
            color="#627D98",
            markersize=9,
            label="Desired direction",
        )
    )
    # A shared legend keeps the narrow panels free of occluded data.
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=4,
        frameon=False,
        fontsize=6.4,
        labelspacing=0.35,
        handlelength=1.55,
        handletextpad=0.5,
        columnspacing=0.8,
    )
    """
    figure.text(
        0.5,
        0.012,
        "Means over 100 held-out, randomly generated test instances; online computation excludes one-time setup.",
        ha="center",
        va="top",
        fontsize=7.3,
        color="#486581",
    )
    """
    figure.supxlabel("Mean online computation (s; log scale)", y=0.16, fontsize=7.4)
    figure.subplots_adjust(left=0.125, right=0.985, top=0.88, bottom=0.28, wspace=0.34)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, format="pdf", metadata={"Title": "Task-computation Pareto comparison"})
    figure.savefig(OUTPUT.with_suffix(".png"), dpi=300)
    plt.close(figure)
    print(OUTPUT)


if __name__ == "__main__":
    main()
