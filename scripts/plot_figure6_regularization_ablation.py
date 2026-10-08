"""Create the publication-ready switching and information-loss ablation (Figure 6)."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "pdf" / "figure6_regularization_ablation.pdf"

# Rows: information-loss term off/on. Columns: fixed switching term off/on.
SOURCES = {
    (False, False): ROOT
    / "artifacts"
    / "results"
    / "meta_regret_allocation50_info0_swi0_unseen_20000_100"
    / "summary.json",
    (False, True): ROOT
    / "artifacts"
    / "results"
    / "meta_regret_allocation50_info0_unseen_20000_100"
    / "summary.json",
    (True, False): ROOT
    / "artifacts"
    / "results"
    / "meta_regret_info0p8_no_fixed_switch_unseen_20000_100"
    / "summary.json",
    (True, True): ROOT
    / "artifacts"
    / "results"
    / "meta_regret_allocation50_info0p8_no_fixed_unseen_20000_100"
    / "summary.json",
}


def _load_ablation() -> dict[tuple[bool, bool], dict[str, float]]:
    results: dict[tuple[bool, bool], dict[str, float]] = {}
    for key, source in SOURCES.items():
        payload = json.loads(source.read_text(encoding="utf-8"))
        configuration = payload["configuration"]
        information_enabled, switching_enabled = key
        expected_mode = "allocation" if switching_enabled else "none"
        if switching_enabled:
            assert configuration["switching_weight"] > 0
            assert configuration["switch_penalty_mode"] == expected_mode
        else:
            assert (
                configuration["switching_weight"] == 0
                or configuration["switch_penalty_mode"] == expected_mode
            )
        if information_enabled:
            assert configuration["information_loss_weight"] > 0
        else:
            assert configuration["information_loss_weight"] == 0
        adaptive = payload["controllers"]["adaptive"]
        assert adaptive["episodes"] == 100
        results[key] = adaptive
    return results


def _matrix(results: dict[tuple[bool, bool], dict[str, float]], field: str) -> np.ndarray:
    return np.asarray(
        [
            [results[(information, switching)][field] for switching in (False, True)]
            for information in (False, True)
        ],
        dtype=float,
    )


def _annotation_color(rgba: tuple[float, float, float, float]) -> str:
    red, green, blue, _ = rgba
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "#102A43" if luminance > 0.61 else "#F7FAFC"


def _draw_panel(
    ax: mpl.axes.Axes,
    values: np.ndarray,
    *,
    title: str,
    panel: str,
    cmap: str,
    colorbar_label: str,
    formatter: Callable[[float], str],
) -> None:
    span = float(np.ptp(values))
    padding = max(0.04 * span, 1e-9)
    image = ax.imshow(
        values,
        cmap=cmap,
        norm=Normalize(vmin=float(values.min() - padding), vmax=float(values.max() + padding)),
        aspect="equal",
        interpolation="nearest",
    )
    ax.set_xticks((0, 1), ("Off", "On"))
    ax.set_yticks((0, 1), ("Off", "On"))
    ax.set_xlabel("Fixed switching term")
    ax.set_ylabel("Information-loss term")
    ax.set_title(title, loc="left", pad=8, fontsize=10.2, fontweight="semibold")
    ax.text(
        -0.19,
        1.12,
        panel,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
    )

    for row in range(2):
        for column in range(2):
            value = values[row, column]
            ax.text(
                column,
                row,
                formatter(value),
                ha="center",
                va="center",
                fontsize=9.2,
                fontweight="semibold",
                color=_annotation_color(image.cmap(image.norm(value))),
            )

    ax.add_patch(
        Rectangle(
            (0.53, 0.53),
            0.94,
            0.94,
            fill=False,
            edgecolor="#E76F51",
            linewidth=2.2,
            joinstyle="round",
        )
    )
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.047, pad=0.045)
    colorbar.set_label(colorbar_label, labelpad=7)
    colorbar.ax.tick_params(labelsize=7.3, width=0.6, length=2.5)
    colorbar.outline.set_linewidth(0.6)
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def main() -> None:
    results = _load_ablation()
    success = 100.0 * _matrix(results, "success_rate")
    task_cost = _matrix(results, "mean_task_cost")
    switches = _matrix(results, "mean_switches")
    compute = _matrix(results, "mean_total_compute_ms")

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 10.2,
            "axes.linewidth": 0.6,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )

    figure, axes = plt.subplots(2, 2, figsize=(7.16, 5.3), constrained_layout=True)
    figure.set_constrained_layout_pads(w_pad=0.045, h_pad=0.055, wspace=0.05, hspace=0.075)

    _draw_panel(
        axes[0, 0],
        success,
        title="Task success (higher is better)",
        panel="(a)",
        cmap="cividis",
        colorbar_label="Success rate (%)",
        formatter=lambda value: f"{value:.0f}%",
    )
    _draw_panel(
        axes[0, 1],
        task_cost,
        title="Operational task cost (lower is better)",
        panel="(b)",
        cmap="cividis_r",
        colorbar_label="Mean task cost",
        formatter=lambda value: f"{value:.1f}",
    )
    _draw_panel(
        axes[1, 0],
        switches,
        title="Allocation switching (lower is better)",
        panel="(c)",
        cmap="cividis_r",
        colorbar_label="Mean switches per episode",
        formatter=lambda value: f"{value:.2f}",
    )
    _draw_panel(
        axes[1, 1],
        compute,
        title="Online computation (lower is better)",
        panel="(d)",
        cmap="cividis_r",
        colorbar_label="Mean online compute (ms)",
        formatter=lambda value: f"{value:.0f}",
    )

    figure.text(
        0.5,
        -0.012,
        "Orange outlines mark the full controller with both regularizers enabled. "
        "Means over 100 held-out, randomly generated test instances.",
        ha="center",
        va="top",
        fontsize=7.4,
        color="#486581",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        OUTPUT,
        format="pdf",
        metadata={"Title": "Switching and information-loss ablation"},
    )
    plt.close(figure)
    print(OUTPUT)


if __name__ == "__main__":
    main()
