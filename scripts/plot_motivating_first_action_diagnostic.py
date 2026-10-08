"""Plot matched MOS contexts explaining action sensitivity to inference resources."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from active_inference_navigation.mos import (
    MOSAction,
    MOSAgentConfig,
    build_mos_agent,
    mos_action_controls,
    sample_mos_instance,
)
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "results" / "mos_research_archive_2000"
PDF_OUTPUT = ROOT / "output" / "pdf" / "figure_motivating_first_action_diagnostic.pdf"
PNG_OUTPUT = ROOT / "output" / "png" / "figure_motivating_first_action_diagnostic.png"
SVG_OUTPUT = ROOT / "output" / "svg" / "figure_motivating_first_action_diagnostic.svg"

# Same color-blind-safe design language used by the paper's results figures.
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#8E6CBB"
TARGET_RED = "#CC3311"
GREY = "#7B8794"
WALL = "#52606D"
PALE = "#F7F9FB"

BELIEF_CMAP = LinearSegmentedColormap.from_list(
    "belief", [(1.0, 1.0, 1.0, 0.0), (0.90, 0.62, 0.0, 0.90)]
)
WALL_CMAP = LinearSegmentedColormap.from_list("walls", [WALL, WALL])

ACTION_SYMBOL = {
    int(MOSAction.UP): r"$\uparrow$",
    int(MOSAction.DOWN): r"$\downarrow$",
    int(MOSAction.LEFT): r"$\leftarrow$",
    int(MOSAction.RIGHT): r"$\rightarrow$",
    int(MOSAction.FIND): "F",
}
ACTION_COLOR = {
    int(MOSAction.UP): BLUE,
    int(MOSAction.DOWN): ORANGE,
    int(MOSAction.LEFT): PURPLE,
    int(MOSAction.RIGHT): GREEN,
    int(MOSAction.FIND): "#C43C39",
}
POLICY_ARROW = {
    MOSAction.UP: "↑",
    MOSAction.DOWN: "↓",
    MOSAction.LEFT: "←",
    MOSAction.RIGHT: "→",
}


@dataclass(frozen=True)
class Example:
    context_id: str
    heading: str
    first: tuple[int, int]
    second: tuple[int, int]
    matrix_title: str
    distance_note: str | None = None
    show_deep_policy: bool = False
    single_resource: tuple[tuple[int, int], ...] = ()


EXAMPLES = (
    Example(
        "instance-10043-decision-4",
        "(a) Allocation-invariant",
        (5, 1),
        (20, 3),
        "Different resolution; same first action",
    ),
    Example(
        "instance-10368-decision-1",
        "(b) Resolution-sensitive",
        (2, 1),
        (10, 1),
        "Different resolution; different first action",
        # distance_note="Opening: 1 step",
    ),
    Example(
        "instance-11017-decision-4",
        "(c) Depth-sensitive",
        (5, 1),
        (5, 3),
        "Different depth; different first action",
        # distance_note="Opening: 3 steps",
        show_deep_policy=True,
    ),
    Example(
        "instance-10006-decision-17",
        "(d) Joint-sensitive",
        (2, 1),
        (10, 2),
        "Different allocation; different first action",
        show_deep_policy=True,
        single_resource=((10, 1), (2, 2)),
    ),
)


def _load_context_rows() -> dict[str, dict[str, str]]:
    with (DATASET / "contexts.csv").open(newline="", encoding="utf-8") as stream:
        return {row["context_id"]: row for row in csv.DictReader(stream)}


def _allocation_index(
    resolutions: np.ndarray, depths: np.ndarray, allocation: tuple[int, int]
) -> int:
    matches = np.flatnonzero((resolutions == allocation[0]) & (depths == allocation[1]))
    if matches.size != 1:
        raise RuntimeError(f"allocation {allocation} was not uniquely identified")
    return int(matches[0])


def _selected_policy_actions(
    *, layout, allocation: tuple[int, int], selected_policy: int
) -> tuple[MOSAction, ...]:
    agent = build_mos_agent(
        MOSAgentConfig(
            target_resolution=allocation[0],
            action_depth=allocation[1],
            message_passing_iterations=10,
            policy_workers=1,
        ),
        layout=layout,
    )
    policy = np.asarray(agent.policies[selected_policy], dtype=int)
    result: list[MOSAction] = []
    for controls in policy:
        matches = [
            action for action in MOSAction if np.array_equal(controls, mos_action_controls(action))
        ]
        if len(matches) != 1:
            raise RuntimeError(f"could not map policy controls {controls} to an action")
        result.append(matches[0])
    return tuple(result)


def _draw_policy(
    ax: plt.Axes,
    *,
    layout,
    robot: tuple[int, int],
    policy: tuple[MOSAction, ...],
    color: str,
    future: bool,
) -> None:
    position = robot
    for step, action in enumerate(policy):
        if action is MOSAction.FIND:
            ax.scatter(
                *position,
                s=82,
                facecolors="none",
                edgecolors=color,
                linewidth=1.25,
                alpha=1.0 if step == 0 else 0.66,
                zorder=9,
            )
            continue
        next_position = layout.move(position, action)
        unit_delta = {
            MOSAction.UP: (0.0, 1.0),
            MOSAction.DOWN: (0.0, -1.0),
            MOSAction.LEFT: (-1.0, 0.0),
            MOSAction.RIGHT: (1.0, 0.0),
        }[action]
        glyph_position = (
            position[0] + 0.92 * unit_delta[0],
            position[1] + 0.92 * unit_delta[1],
        )
        glyph = ax.text(
            *glyph_position,
            POLICY_ARROW[action],
            ha="center",
            va="center",
            fontsize=8.8,
            fontweight="bold",
            color=color,
            alpha=1.0 if step == 0 else (0.82 if future else 0.0),
            zorder=13,
        )
        glyph.set_path_effects([path_effects.withStroke(linewidth=1.1, foreground="white")])
        position = next_position


def _draw_posterior_map(
    ax: plt.Axes,
    *,
    posterior: np.ndarray,
    vmax: float,
    obstacles: np.ndarray,
    layout,
    robot: tuple[int, int],
    target: tuple[int, int],
    allocation: tuple[int, int],
    policy: tuple[MOSAction, ...],
    color: str,
    distance_note: str | None,
    show_future: bool,
) -> None:
    ax.set_facecolor(PALE)
    ax.imshow(
        posterior,
        origin="lower",
        cmap=BELIEF_CMAP,
        vmin=0.0,
        vmax=vmax,
        interpolation="nearest",
    )
    blocked = np.ma.masked_where(obstacles < 0.5, obstacles)
    ax.imshow(blocked, origin="lower", cmap=WALL_CMAP, vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.scatter(
        *target,
        s=46,
        marker="*",
        facecolor=TARGET_RED,
        edgecolor="white",
        linewidth=0.55,
        zorder=10,
        clip_on=False,
    )
    ax.scatter(*robot, s=13, facecolor=BLUE, edgecolor="white", linewidth=0.55, zorder=12)
    _draw_policy(
        ax,
        layout=layout,
        robot=robot,
        policy=policy if show_future else policy[:1],
        color=color,
        future=show_future,
    )

    gamma, tau = allocation
    ax.set_title(
        rf"$\gamma={gamma},\tau={tau}$",  # + f"\n{action_name}",
        fontsize=5.1,
        pad=2.0,
        color="#243B53",
    )
    if distance_note:
        ax.text(
            0.04,
            0.96,
            distance_note,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=4.4,
            color="#486581",
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": "#BCCCDC",
                "alpha": 0.90,
                "lw": 0.45,
            },
            zorder=12,
        )
    # Match the axes frame to the exact outer edges of the 20 x 20 cells.
    ax.set_xlim(-0.5, 19.5)
    ax.set_ylim(-0.5, 19.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#9FB3C8")
        spine.set_linewidth(0.55)


def _draw_action_matrix(
    ax: plt.Axes,
    *,
    actions: np.ndarray,
    resolutions: np.ndarray,
    depths: np.ndarray,
    first: tuple[int, int],
    second: tuple[int, int],
    title: str,
    single_resource: tuple[tuple[int, int], ...] = (),
) -> None:
    resolution_order = (2, 5, 10, 20)
    depth_order = (1, 2, 3)
    ax.set_xlim(0, 3)
    ax.set_ylim(4, 0)
    ax.set_aspect(0.3)
    ax.set_facecolor("white")
    for row, resolution in enumerate(resolution_order):
        for col, depth in enumerate(depth_order):
            index = _allocation_index(resolutions, depths, (resolution, depth))
            action = int(actions[index])
            ax.add_patch(
                Rectangle((col, row), 1, 1, facecolor="#F5F7FA", edgecolor="#D9E2EC", lw=0.48)
            )
            up_arrow_offset = 0.08 if action == int(MOSAction.UP) else 0.0
            ax.text(
                col + 0.5,
                row + 0.5 + up_arrow_offset,
                ACTION_SYMBOL[action],
                ha="center",
                va="center",
                fontsize=8.5,
                color=ACTION_COLOR[action],
                fontweight="bold",
            )

    for allocation in single_resource:
        row = resolution_order.index(allocation[0])
        col = depth_order.index(allocation[1])
        """
        ax.add_patch(
            Rectangle(
                (col + 0.07, row + 0.07),
                0.86,
                0.86,
                fill=False,
                edgecolor=GREY,
                linewidth=1.0,
                linestyle=":",
            )
        )
        """

    for allocation, color, linestyle in (
        (first, GREY, "--"),
        (second, BLUE, "-"),
    ):
        row = resolution_order.index(allocation[0])
        col = depth_order.index(allocation[1])
        ax.add_patch(
            Rectangle(
                (col + 0.035, row + 0.035),
                0.93,
                0.93,
                fill=False,
                edgecolor=color,
                linewidth=0.5,
                linestyle=linestyle,
            )
        )

    ax.set_xticks(np.arange(3) + 0.5, [r"$\tau=1$", r"$\tau=2$", r"$\tau=3$"])
    ax.set_yticks(
        np.arange(4) + 0.5,
        [r"$\gamma=2$", r"$\gamma=5$", r"$\gamma=10$", r"$\gamma=20$"],
    )
    ax.tick_params(length=0, pad=1.2)
    is_resolution_panel = title.startswith("Different")

    ax.set_title(
        title,
        fontsize=4.7 if is_resolution_panel else 5.25,
        x=0.43 if is_resolution_panel else 0.5,
        pad=2.0,
        color="#486581",
    )
    for spine in ax.spines.values():
        spine.set_visible(False)


def main() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.1,
            "axes.titlesize": 8.0,
            "xtick.labelsize": 5.8,
            "ytick.labelsize": 5.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.035,
        }
    )

    context_rows = _load_context_rows()
    with np.load(DATASET / "training_data.npz", allow_pickle=False) as training:
        context_ids = training["context_ids"].astype(str)
        id_to_index = {value: index for index, value in enumerate(context_ids)}
        spatial = training["spatial"]
        actions = training["candidate_actions"]
        resolutions = training["resolutions"].astype(int)
        depths = training["depths"].astype(int)

        with np.load(DATASET / "research_archive.npz", allow_pickle=False) as research:
            posteriors = research["canonical_target_posterior"]
            selected_policies = research["selected_policy"]

            figure = plt.figure(figsize=(7.15, 2.3))
            outer = figure.add_gridspec(
                2,
                len(EXAMPLES),
                height_ratios=(0.95, 1.15),
                left=0.045,
                right=0.995,
                top=0.81,
                bottom=0.19,
                wspace=-0.06,
                hspace=-0.35,
            )

            for column, example in enumerate(EXAMPLES):
                index = id_to_index[example.context_id]
                row = context_rows[example.context_id]
                instance = sample_mos_instance(int(row["instance_seed"]))
                obstacles = spatial[index, 3]
                robot = (int(row["next_x"]), int(row["next_y"]))
                pair = figure.add_subfigure(outer[0, column])
                pair_axes = pair.subplots(1, 2, gridspec_kw={"wspace": 0.07})
                for pair_ax in pair_axes:
                    pair_ax.set_anchor("S")

                pair_indices = [
                    _allocation_index(resolutions, depths, example.first),
                    _allocation_index(resolutions, depths, example.second),
                ]
                pair_posteriors = [posteriors[index, value] for value in pair_indices]
                pair_vmax = max(float(np.max(value)) for value in pair_posteriors)
                for side, (allocation, allocation_index, posterior) in enumerate(
                    zip((example.first, example.second), pair_indices, pair_posteriors)
                ):
                    policy = _selected_policy_actions(
                        layout=instance.layout,
                        allocation=allocation,
                        selected_policy=int(selected_policies[index, allocation_index]),
                    )
                    _draw_posterior_map(
                        pair_axes[side],
                        posterior=posterior,
                        vmax=pair_vmax,
                        obstacles=obstacles,
                        layout=instance.layout,
                        robot=robot,
                        target=instance.target,
                        allocation=allocation,
                        policy=policy,
                        color=GREY if side == 0 else BLUE,
                        distance_note=example.distance_note if side == 0 else None,
                        show_future=example.show_deep_policy and side == 1,
                    )

                matrix_ax = figure.add_subplot(outer[1, column])
                _draw_action_matrix(
                    matrix_ax,
                    actions=actions[index],
                    resolutions=resolutions,
                    depths=depths,
                    first=example.first,
                    second=example.second,
                    title=example.matrix_title,
                    single_resource=example.single_resource,
                )
                # Reduce only the action-matrix height; preserve its horizontal width.
                position = matrix_ax.get_position()
                height_scale = 0.70

                new_height = position.height * height_scale

                matrix_ax.set_position(
                    [
                        position.x0,  # unchanged horizontal position
                        position.y1 - new_height,  # preserve the top alignment
                        position.width,  # unchanged width
                        new_height,  # reduced height
                    ]
                )

            for column, example in enumerate(EXAMPLES):
                x = (column + 0.5) / len(EXAMPLES)
                figure.text(
                    x,
                    0.982,
                    example.heading,
                    ha="center",
                    va="top",
                    fontsize=8.0,
                    fontweight="bold",
                )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=BLUE,
            markeredgecolor="white",
            markersize=4.5,
            label="Robot",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            color="none",
            markerfacecolor=TARGET_RED,
            markeredgecolor="white",
            markersize=6.8,
            label="Target",
        ),
        Rectangle(
            (0, 0),
            1,
            1,
            facecolor="#E69F00",
            edgecolor="none",
            alpha=0.65,
            label="Target posterior",
        ),
        Rectangle((0, 0), 1, 1, facecolor=WALL, edgecolor="none", label="Obstacle"),
        Line2D(
            [0], [0], color=GREY, linestyle="--", linewidth=1.4, label="Lower-resource allocation"
        ),
        # Line2D([0], [0], color=ORANGE, linestyle=":", linewidth=1.3,
        # label="Single-resource increase"),
        Line2D([0], [0], color=BLUE, linewidth=1.4, label="Additional-resource allocation"),
    ]
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=7,
        frameon=False,
        fontsize=5.9,
        columnspacing=1.05,
        handlelength=1.45,
        handletextpad=0.38,
    )

    PDF_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    PNG_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    SVG_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        PDF_OUTPUT,
        format="pdf",
        metadata={"Title": "Matched-context first-action diagnostic"},
    )
    figure.savefig(PNG_OUTPUT, format="png", dpi=300)
    figure.savefig(
        SVG_OUTPUT,
        format="svg",
        metadata={"Title": "Matched-context first-action diagnostic"},
    )
    plt.close(figure)
    print(PDF_OUTPUT)
    print(PNG_OUTPUT)
    print(SVG_OUTPUT)


if __name__ == "__main__":
    main()
