"""Multi-resolution POUCT-style baseline for the 2D MOS benchmark.

The implementation follows the central construction of MR-POUCT: maintain a
ground-resolution belief, derive abstract POMDPs at several spatial levels,
solve one online search tree per level, and execute the action with the largest
root value across levels.  Robot states and physical actions remain at the
ground resolution; only the static target state is abstracted.
"""

from __future__ import annotations

import csv
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from .closed_loop import _episode_quantiles
from .counterfactuals import _step_cost, _timed
from .pomcp import (
    ActionNode,
    PlanResult,
    POMCPConfig,
    POMCPPlanner,
    TabularTargetBelief,
    _pomcp_mos_imports,
)


@dataclass(frozen=True)
class MultiResolutionPOMCPConfig:
    """Settings for a matched-budget multi-resolution POUCT evaluation."""

    resolutions: tuple[int, ...] = (2, 5, 10, 20)
    simulations: int = 250
    planning_budget_ms: float | None = None
    search_depth: int = 30
    exploration_constant: float = 2.0**0.5
    discount: float = 0.98
    rollout_random_probability: float = 0.10
    reward_mode: str = "external"
    step_penalty: float = 1.0
    found_reward: float = 9.0
    false_find_penalty: float = 12.0
    collision_penalty: float = 12.0
    max_steps: int = 50
    planner_seed_offset: int = 2_000_003

    def __post_init__(self) -> None:
        if not self.resolutions or len(set(self.resolutions)) != len(self.resolutions):
            raise ValueError("resolutions must be nonempty and unique")
        if any(resolution < 1 for resolution in self.resolutions):
            raise ValueError("resolutions must be positive")
        if self.simulations < len(self.resolutions):
            raise ValueError("simulations must provide at least one search per resolution")
        # Reuse the standard configuration's remaining validation.
        self.as_pomcp_config(simulations=1)

    def as_pomcp_config(self, *, simulations: int) -> POMCPConfig:
        return POMCPConfig(
            simulations=simulations,
            planning_budget_ms=None,
            search_depth=self.search_depth,
            exploration_constant=self.exploration_constant,
            discount=self.discount,
            rollout_random_probability=self.rollout_random_probability,
            reward_mode=self.reward_mode,
            step_penalty=self.step_penalty,
            found_reward=self.found_reward,
            false_find_penalty=self.false_find_penalty,
            collision_penalty=self.collision_penalty,
            max_steps=self.max_steps,
            planner_seed_offset=self.planner_seed_offset,
        )


@dataclass(frozen=True)
class MultiResolutionPlanResult:
    action: int
    resolution: int
    simulations: int
    planning_ms: float
    level_actions: tuple[int, ...]
    level_values: tuple[float, ...]
    level_visits: tuple[int, ...]


class AbstractTargetBelief:
    """Projection of an exact canonical belief onto an r-by-r target grid."""

    def __init__(self, canonical: TabularTargetBelief, resolution: int) -> None:
        size = int(canonical.layout.size)
        if size % resolution:
            raise ValueError(f"resolution {resolution} must divide layout size {size}")
        self.canonical = canonical
        self.resolution = int(resolution)
        self.cell_width = size // resolution
        mapping = np.empty(size**2, dtype=np.int32)
        for state in range(size**2):
            x, y = state % size, state // size
            mapping[state] = (y // self.cell_width) * resolution + (x // self.cell_width)
        self.ground_to_abstract = mapping
        self._sampling_probabilities = self.probabilities

    @property
    def probabilities(self) -> np.ndarray:
        values = np.bincount(
            self.ground_to_abstract,
            weights=self.canonical.probabilities,
            minlength=self.resolution**2,
        )
        total = float(values.sum())
        if total <= 0:
            raise ValueError("abstract belief has no probability mass")
        return values / total

    def sample(self, rng: np.random.Generator) -> int:
        return int(rng.choice(self._sampling_probabilities.size, p=self._sampling_probabilities))

    def refresh(self) -> None:
        """Cache the current projection once for the next online search."""

        self._sampling_probabilities = self.probabilities


class AbstractPOMCPPlanner(POMCPPlanner):
    """POUCT over an abstract target grid with ground-level robot actions."""

    def __init__(
        self,
        *,
        layout: Any,
        canonical_belief: TabularTargetBelief,
        resolution: int,
        config: POMCPConfig,
        seed: int,
    ) -> None:
        self.abstract_belief = AbstractTargetBelief(canonical_belief, resolution)
        self.resolution = int(resolution)
        super().__init__(layout=layout, belief=canonical_belief, config=config, seed=seed)

    def _build_static_model(self) -> None:
        # Build the exact ground model once, then apply a fixed uniform
        # state-abstraction operator to its target dimension.
        super()._build_static_model()
        canonical_detection = self._detection_probability
        canonical_visible = self._visible
        canonical_rollout = self._rollout_action
        self._canonical_detection_probability = canonical_detection
        self._canonical_visible = canonical_visible

        mapping = self.abstract_belief.ground_to_abstract
        abstract_states = self.resolution**2
        ground_free = np.asarray(
            [
                self.layout.is_free((state % self.layout.size, state // self.layout.size))
                for state in range(self.layout.size**2)
            ],
            dtype=bool,
        )
        self._abstract_detection_probability = np.zeros(
            (self.layout.size**2, abstract_states), dtype=np.float32
        )
        self._abstract_visible_probability = np.zeros(
            (self.layout.size**2, abstract_states), dtype=np.float32
        )
        self._abstract_rollout_action = np.full(
            (self.layout.size**2, abstract_states), self._find_action, dtype=np.int8
        )
        for cell in range(abstract_states):
            members = np.flatnonzero((mapping == cell) & ground_free)
            if members.size == 0:
                continue
            self._abstract_detection_probability[:, cell] = canonical_detection[:, members].mean(
                axis=1
            )
            self._abstract_visible_probability[:, cell] = canonical_visible[:, members].mean(axis=1)
            actions = canonical_rollout[:, members]
            for robot_state in range(actions.shape[0]):
                counts = np.bincount(actions[robot_state], minlength=len(self._actions))
                self._abstract_rollout_action[robot_state, cell] = int(np.argmax(counts))

    def sensor_model(self, robot: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        """Return the exact canonical model for the shared Bayesian update."""

        state = robot[1] * self.layout.size + robot[0]
        return self._canonical_detection_probability[state], self._canonical_visible[state]

    def _generate(
        self,
        robot: tuple[int, int],
        target: int,
        action: int,
    ) -> tuple[tuple[int, int], tuple[int, int], float, bool]:
        robot_state = robot[1] * self.layout.size + robot[0]
        next_state = int(self._next_state[robot_state, action])
        next_robot = (next_state % self.layout.size, next_state // self.layout.size)
        collision = bool(self._collision[robot_state, action])
        detection = int(
            self.rng.random() < self._abstract_detection_probability[next_state, target]
        )
        if action == self._find_action:
            success = bool(
                self.rng.random() < self._abstract_visible_probability[next_state, target]
            )
            find = int(
                self._mos["FindOutcome"].FOUND if success else self._mos["FindOutcome"].FALSE_FIND
            )
        else:
            success = False
            find = int(self._mos["FindOutcome"].SEARCHING)
        false_find = find == int(self._mos["FindOutcome"].FALSE_FIND)
        if self.config.reward_mode == "external":
            reward = -_step_cost(false_find=false_find, collision=collision)
        else:
            reward = -self.config.step_penalty
            if success:
                reward += self.config.found_reward
            elif false_find:
                reward -= self.config.false_find_penalty
            if collision:
                reward -= self.config.collision_penalty
        return next_robot, (detection, find), float(reward), success

    def _shortest_path_action(self, robot: tuple[int, int], target: int) -> int:
        state = robot[1] * self.layout.size + robot[0]
        return int(self._abstract_rollout_action[state, target])

    def plan(self, robot: tuple[int, int]) -> PlanResult:
        started = time.perf_counter_ns()
        self.abstract_belief.refresh()
        completed = 0
        while completed < self.config.simulations:
            target = self.abstract_belief.sample(self.rng)
            self._simulate(robot, target, self.root, self.config.search_depth)
            completed += 1
        planning_ms = (time.perf_counter_ns() - started) / 1e6
        for action in self._actions:
            self.root.actions.setdefault(action, ActionNode())
        visits = tuple(self.root.actions[action].visits for action in self._actions)
        values = tuple(self.root.actions[action].value for action in self._actions)
        most_visited = max(visits)
        candidates = [
            action
            for action, visits_for_action in zip(self._actions, visits, strict=True)
            if visits_for_action == most_visited
        ]
        action = max(candidates, key=lambda candidate: self.root.actions[candidate].value)
        return PlanResult(
            action=int(action),
            simulations=completed,
            planning_ms=float(planning_ms),
            action_values=values,
            action_visits=visits,
        )


class MultiResolutionPOMCPPlanner:
    """Coordinate one abstract POUCT tree per target resolution."""

    def __init__(
        self,
        *,
        layout: Any,
        belief: TabularTargetBelief,
        config: MultiResolutionPOMCPConfig,
        seed: int,
    ) -> None:
        self.layout = layout
        self.belief = belief
        self.config = config
        if any(layout.size % resolution for resolution in config.resolutions):
            raise ValueError("every resolution must divide the canonical layout size")
        quotient, remainder = divmod(config.simulations, len(config.resolutions))
        budgets = tuple(
            quotient + int(index < remainder) for index in range(len(config.resolutions))
        )
        self.planners = tuple(
            AbstractPOMCPPlanner(
                layout=layout,
                canonical_belief=belief,
                resolution=resolution,
                config=config.as_pomcp_config(simulations=budget),
                seed=seed + 104_729 * index,
            )
            for index, (resolution, budget) in enumerate(
                zip(config.resolutions, budgets, strict=True)
            )
        )

    def sensor_model(self, robot: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        return self.planners[-1].sensor_model(robot)

    def plan(self, robot: tuple[int, int]) -> MultiResolutionPlanResult:
        started = time.perf_counter_ns()
        plans: list[PlanResult] = []
        for planner in self.planners:
            plans.append(planner.plan(robot))
            if self.config.planning_budget_ms is not None:
                elapsed_ms = (time.perf_counter_ns() - started) / 1e6
                if elapsed_ms >= self.config.planning_budget_ms:
                    break
        # MR-POUCT selects the resolution/action pair with the highest root Q,
        # rather than first choosing a resolution and then comparing visit counts.
        level_actions = tuple(int(np.argmax(plan.action_values)) for plan in plans)
        level_values = tuple(
            float(plan.action_values[action])
            for plan, action in zip(plans, level_actions, strict=True)
        )
        best_index = int(np.argmax(level_values))
        return MultiResolutionPlanResult(
            action=level_actions[best_index],
            resolution=self.planners[best_index].resolution,
            simulations=sum(plan.simulations for plan in plans),
            planning_ms=float((time.perf_counter_ns() - started) / 1e6),
            level_actions=level_actions,
            level_values=level_values,
            level_visits=tuple(
                plan.action_visits[action]
                for plan, action in zip(plans, level_actions, strict=True)
            ),
        )

    def advance_root(self, action: int, observation: tuple[int, int, int, int, int]) -> None:
        for planner in self.planners:
            planner.advance_root(action, observation)


def run_multi_resolution_pomcp_episode(
    instance_seed: int,
    config: MultiResolutionPOMCPConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run the multi-resolution planner on one matched MOS instance."""

    mos = _pomcp_mos_imports()
    instance = mos["sample_mos_instance"](int(instance_seed))
    environment = mos["MOSEnvironment"](
        layout=instance.layout,
        target=instance.target,
        observation_seed=instance.observation_seed,
    )
    quantiles = _episode_quantiles(instance, config.max_steps)
    observation = environment.reset(noise_quantile=float(quantiles[0]))
    belief = TabularTargetBelief(instance.layout)
    planner, planner_setup_ms = _timed(
        lambda: MultiResolutionPOMCPPlanner(
            layout=instance.layout,
            belief=belief,
            config=config,
            seed=int(instance_seed + config.planner_seed_offset),
        )
    )
    detection_probabilities, visible_targets = planner.sensor_model(environment.position)
    _, initial_update_ms = _timed(
        lambda: belief.update(
            robot=environment.position,
            observation=observation,
            action=None,
            mos=mos,
            detection_probabilities=detection_probabilities,
            visible_targets=visible_targets,
        )
    )
    planning_ms = 0.0
    belief_update_ms = initial_update_ms
    task_cost = 0.0
    success = False
    total_simulations = 0
    selected_resolutions = {resolution: 0 for resolution in config.resolutions}
    trajectory: list[dict[str, Any]] = []

    for decision_index in range(config.max_steps):
        position = environment.position
        plan = planner.plan(position)
        planning_ms += plan.planning_ms
        total_simulations += plan.simulations
        selected_resolutions[plan.resolution] += 1
        action = mos["MOSAction"](plan.action)
        observation, success = environment.step(
            action,
            noise_quantile=float(quantiles[decision_index + 1]),
        )
        false_find = observation[3] == mos["FindOutcome"].FALSE_FIND
        step_cost = _step_cost(false_find=false_find, collision=environment.last_collision)
        task_cost += step_cost
        update_ms = 0.0
        if not success:
            detection_probabilities, visible_targets = planner.sensor_model(environment.position)
            _, update_ms = _timed(
                partial(
                    belief.update,
                    robot=environment.position,
                    observation=observation,
                    action=int(action),
                    mos=mos,
                    detection_probabilities=detection_probabilities,
                    visible_targets=visible_targets,
                )
            )
            belief_update_ms += update_ms
            planner.advance_root(int(action), observation)
        trajectory.append(
            {
                "step": decision_index + 1,
                "x": int(position[0]),
                "y": int(position[1]),
                "action": action.name,
                "next_x": int(environment.position[0]),
                "next_y": int(environment.position[1]),
                "detection": int(observation[2]),
                "find_outcome": int(observation[3]),
                "selected_resolution": int(plan.resolution),
                "level_actions": json.dumps(plan.level_actions),
                "level_values": json.dumps(plan.level_values),
                "level_visits": json.dumps(plan.level_visits),
                "belief_normalized_entropy": belief.normalized_entropy,
                "planning_ms": plan.planning_ms,
                "belief_update_ms": update_ms,
                "simulations": plan.simulations,
                "step_cost": float(step_cost),
                "success": bool(success),
            }
        )
        if success:
            break

    if not success:
        task_cost += 2.0 * config.max_steps
    total_compute_ms = planning_ms + belief_update_ms
    summary = {
        "controller": "multi_resolution_pomcp",
        "instance_seed": int(instance_seed),
        "target": list(instance.target),
        "success": bool(success),
        "steps": len(trajectory),
        "task_cost": float(task_cost),
        "planning_ms": float(planning_ms),
        "belief_update_ms": float(belief_update_ms),
        "planner_setup_ms": float(planner_setup_ms),
        "total_compute_ms": float(total_compute_ms),
        "total_simulations": int(total_simulations),
        "mean_simulations_per_decision": float(total_simulations / len(trajectory)),
        "selected_resolution_steps": {
            f"g{resolution}": count for resolution, count in selected_resolutions.items()
        },
    }
    return summary, trajectory


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(dict.fromkeys(name for row in rows for name in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _run_instance(instance_seed: int, config: MultiResolutionPOMCPConfig) -> dict[str, Any]:
    summary, trajectory = run_multi_resolution_pomcp_episode(instance_seed, config)
    for row in trajectory:
        row["instance_seed"] = int(instance_seed)
    return {"summary": summary, "trajectory": trajectory}


def evaluate_multi_resolution_pomcp(
    *,
    instance_seeds: list[int] | tuple[int, ...],
    output_dir: Path,
    config: MultiResolutionPOMCPConfig,
    instance_workers: int = 1,
    resume: bool = True,
) -> dict[str, Any]:
    """Run resumable matched multi-resolution POUCT evaluation."""

    seeds = tuple(int(seed) for seed in instance_seeds)
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("instance_seeds must be nonempty and unique")
    if instance_workers < 1:
        raise ValueError("instance_workers must be positive")
    output_dir = Path(output_dir)
    shard_dir = output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    configuration = json.loads(json.dumps(asdict(config)))
    signature = {
        "configuration": configuration,
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    pending: list[int] = []
    for seed in seeds:
        path = shard_dir / f"instance-{seed}.json"
        compatible = False
        if resume and path.is_file():
            try:
                compatible = (
                    json.loads(path.read_text(encoding="utf-8")).get("signature") == signature
                )
            except (OSError, json.JSONDecodeError):
                compatible = False
        if compatible:
            print(f"[reuse] instance={seed}", flush=True)
        else:
            pending.append(seed)

    def save(seed: int, payload: dict[str, Any]) -> None:
        payload["signature"] = signature
        temporary = shard_dir / f"instance-{seed}.json.tmp"
        temporary.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        temporary.replace(shard_dir / f"instance-{seed}.json")

    if instance_workers == 1:
        for completed, seed in enumerate(pending, start=1):
            save(seed, _run_instance(seed, config))
            print(f"[evaluate {completed}/{len(pending)}] instance={seed}", flush=True)
    elif pending:
        with ProcessPoolExecutor(max_workers=instance_workers) as executor:
            futures = {executor.submit(_run_instance, seed, config): seed for seed in pending}
            for completed, future in enumerate(as_completed(futures), start=1):
                seed = futures[future]
                save(seed, future.result())
                print(f"[evaluate {completed}/{len(pending)}] instance={seed}", flush=True)

    payloads = [
        json.loads((shard_dir / f"instance-{seed}.json").read_text(encoding="utf-8"))
        for seed in seeds
    ]
    episodes = [payload["summary"] for payload in payloads]
    trajectory = [row for payload in payloads for row in payload["trajectory"]]
    _write_rows(output_dir / "episodes.csv", episodes)
    _write_rows(output_dir / "trajectory.csv", trajectory)
    compute = np.asarray([row["total_compute_ms"] for row in episodes], dtype=float)
    resolution_steps = {
        f"g{resolution}": int(
            sum(row["selected_resolution_steps"][f"g{resolution}"] for row in episodes)
        )
        for resolution in config.resolutions
    }
    report = {
        "instance_seeds": list(seeds),
        "instance_workers": int(instance_workers),
        "configuration": configuration,
        "controller": {
            "episodes": len(episodes),
            "success_rate": float(np.mean([row["success"] for row in episodes])),
            "mean_task_cost": float(np.mean([row["task_cost"] for row in episodes])),
            "mean_steps": float(np.mean([row["steps"] for row in episodes])),
            "mean_total_compute_ms": float(compute.mean()),
            "median_total_compute_ms": float(np.median(compute)),
            "mean_planning_ms": float(np.mean([row["planning_ms"] for row in episodes])),
            "mean_belief_update_ms": float(np.mean([row["belief_update_ms"] for row in episodes])),
            "mean_planner_setup_ms": float(np.mean([row["planner_setup_ms"] for row in episodes])),
            "mean_simulations_per_decision": float(
                np.mean([row["mean_simulations_per_decision"] for row in episodes])
            ),
            "selected_resolution_steps": resolution_steps,
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def save_multi_resolution_pomcp_episode(
    summary: dict[str, Any],
    trajectory: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_rows(output_dir / "trajectory.csv", trajectory)
