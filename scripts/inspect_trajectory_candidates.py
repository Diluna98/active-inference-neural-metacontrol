"""Read-only ranking of saved trajectory readability."""

from collections import defaultdict

from plot_figure6_controller_behavior import ADAPTIVE_DIR, _read_csv


def metrics(rows):
    moves = [
        tuple(sorted(((int(r["x"]), int(r["y"])), (int(r["next_x"]), int(r["next_y"])))))
        for r in rows
        if (r["x"], r["y"]) != (r["next_x"], r["next_y"])
    ]
    switches = [
        r
        for i, r in enumerate(rows)
        if i and r["source_allocation"] != rows[i - 1]["source_allocation"]
    ]
    return {
        "retraced_edges": len(moves) - len(set(moves)),
        "unique_switch_positions": len({(r["x"], r["y"]) for r in switches}),
        "switches": len(switches),
        "movement_steps": len(moves),
    }


def load():
    trajectories = defaultdict(list)
    for r in _read_csv(ADAPTIVE_DIR / "adaptive_trajectory.csv"):
        trajectories[int(r["instance_seed"])].append(r)
    episodes = [
        r for r in _read_csv(ADAPTIVE_DIR / "episodes.csv") if r["controller"] == "adaptive"
    ]
    return episodes, trajectories


if __name__ == "__main__":
    episodes, trajectories = load()
    for e in episodes:
        m = metrics(trajectories[int(e["instance_seed"])])
        if m["switches"] <= 8:
            print(
                e["instance_seed"],
                e["success"],
                e["steps"],
                e["task_cost"],
                m,
                e["allocation_steps"],
            )
