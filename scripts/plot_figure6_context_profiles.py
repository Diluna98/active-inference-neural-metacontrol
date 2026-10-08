"""Render the context-allocation profiles as a standalone two-panel figure."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from plot_figure6_controller_behavior import (
    CONTEXT_DIR,
    FISHER_COLORS,
    ROOT,
    _context_profiles,
    _plot_profile,
    _read_csv,
)

OUTPUT = ROOT / "output" / "pdf" / "figure6_context_profiles.pdf"


def main() -> None:
    context_rows = _read_csv(CONTEXT_DIR / "adaptive_trajectory.csv")
    (
        state_space_profile,
        state_space_low,
        state_space_high,
        depth_profile,
        depth_low,
        depth_high,
    ) = _context_profiles(context_rows)

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

    figure, axes = plt.subplots(1, 2, figsize=(4.2, 2.85))
    figure.subplots_adjust(
        left=0.105,
        right=0.985,
        top=0.88,
        bottom=0.34,
        wspace=0.32,
    )

    _plot_profile(
        axes[0],
        state_space_profile,
        state_space_low,
        state_space_high,
        kind="resolution",
    )
    axes[0].set_ylabel("Mean state space", labelpad=-1)
    axes[0].set_title(
        "(a) Resolution by context", loc="left", x=-0.06, pad=7, fontweight="semibold"
    )

    _plot_profile(axes[1], depth_profile, depth_low, depth_high, kind="depth")
    axes[1].set_ylabel("Mean planning depth")
    axes[1].set_title("(c) Depth by context", loc="left", pad=7, fontweight="semibold")

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
        bbox_to_anchor=(0.5, 0.115),
        ncol=4,
        frameon=False,
        fontsize=6.4,
        columnspacing=1.0,
        handlelength=1.5,
        handletextpad=0.35,
    )
    """
    figure.text(
        0.5,
        0.018,
        r"Shading shows episode-bootstrap 95% CIs; Q1-Q4 denote "
        r"low-to-high Fisher-information quartiles.",
        ha="center",
        va="bottom",
        fontsize=6.5,
        color="#486581",
    )
    """

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        OUTPUT,
        format="pdf",
        metadata={"Title": "Context-dependent resource allocation"},
    )
    plt.close(figure)
    print(OUTPUT)


if __name__ == "__main__":
    main()
