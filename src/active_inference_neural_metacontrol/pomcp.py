"""Fixed-resolution POMCP baseline for the 20x20 MOS benchmark."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from .closed_loop import _episode_quantiles
from .counterfactuals import _mos_imports, _step_cost, _timed


@dataclass
class ActionNode:
    visits: int = 0
    value: float = 0.0
    children: dict[tuple[int, int], HistoryNode] = field(default_factory=dict)


@dataclass
class HistoryNode:
    visits: int = 0
    actions: dict[int, ActionNode] = field(default_factory=dict)


def _pomcp_mos_imports() -> dict[str, Any]:
    mos = _mos_imports()
    try:
        from active_inference_navigation.mos import MOSAction, detection_distribution
    except ImportError as error:  # pragma: no cover - optional integration dependency
        raise ImportError("POMCP evaluation requires active-inference-navigation-agent") from error
    return {**mos, "MOSAction": MOSAction, "detection_distribution": detection_distribution}


@dataclass(frozen=True)
class POMCPConfig:
    """Search and evaluation settings for standard fixed-resolution POMCP."""

    simulations: int = 500
    planning_budget_ms: float | None = None
    search_depth: int = 10
    exploration_constant: float = math.sqrt(2.0)
    discount: float = 0.98
    rollout_random_probability: float = 0.10
    reward_mode: str = "external"
    step_penalty: float = 1.0
    found_reward: float = 9.0
    false_find_penalty: float = 12.0
    collision_penalty: float = 12.0
    max_steps: int = 50
    planner_seed_offset: int = 1_000_003

    def __post_init__(self) -> None:
        if self.simulations < 1:
            raise ValueError("simulations must be positive")
        if self.planning_budget_ms is not None and self.planning_budget_ms <= 0:
            raise ValueError("planning_budget_ms must be positive")
        if self.search_depth < 1 or self.max_steps < 1:
            raise ValueError("search_depth and max_steps must be positive")
        if self.exploration_constant < 0:
            raise ValueError("exploration_constant cannot be negative")
        if not 0 < self.discount <= 1:
            raise ValueError("discount must lie in (0, 1]")
        if not 0 <= self.rollout_random_probability <= 1:
            raise ValueError("rollout_random_probability must lie in [0, 1]")
        if self.reward_mode not in {"external", "preference"}:
            raise ValueError("reward_mode must be 'external' or 'preference'")
        if (
            min(
                self.step_penalty,
                self.found_reward,
                self.false_find_penalty,
                self.collision_penalty,
            )
            < 0
        ):
            raise ValueError("reward magnitudes must be nonnegative")


@dataclass(frozen=True)
class PlanResult:
    action: int
    simulations: int
    planning_ms: float
    action_values: tuple[float, ...]
    action_visits: tuple[int, ...]


class TabularTargetBelief:
    """Exact target belief used as the root distribution for POMCP sampling."""

    def __init__(self, layout: Any) -> None:
        self.layout = layout
        probabilities = np.zeros(layout.size**2, dtype=float)
        for y in range(layout.size):
            for x in range(layout.size):
                if layout.is_free((x, y)):
                    probabilities[y * layout.size + x] = 1.0
        self.probabilities = probabilities / probabilities.sum()

    def sample(self, rng: np.random.Generator) -> tuple[int, int]:
        state = int(rng.choice(self.probabilities.size, p=self.probabilities))
        return state % self.layout.size, state // self.layout.size

    def update(
        self,
        *,
        robot: tuple[int, int],
        observation: tuple[int, int, int, int, int],
        action: int | None,
        mos: dict[str, Any],
        detection_probabilities: np.ndarray | None = None,
        visible_targets: np.ndarray | None = None,
    ) -> None:
        detection = int(observation[2])
        find = int(observation[3])
        if detection_probabilities is None:
            detection_probabilities = np.asarray(
                [
                    mos["detection_distribution"](
                        robot,
                        (state % self.layout.size, state // self.layout.size),
                        self.layout,
                    )[1]
                    for state in range(self.probabilities.size)
                ]
            )
        detection_probabilities = np.asarray(detection_probabilities, dtype=float)
        likelihood = detection_probabilities if detection == 1 else 1.0 - detection_probabilities
        if action is not None and int(action) == int(mos["MOSAction"].FIND):
            if visible_targets is None:
                visible_targets = np.asarray(
                    [
                        self.layout.target_is_visible(
                            robot,
                            (state % self.layout.size, state // self.layout.size),
                        )
                        for state in range(self.probabilities.size)
                    ]
                )
            expected_found = np.asarray(visible_targets, dtype=bool)
            observed_found = find == int(mos["FindOutcome"].FOUND)
            likelihood = likelihood * (expected_found == observed_found)
        posterior = self.probabilities * likelihood
        evidence = float(posterior.sum())
        if evidence <= 0 or not np.isfinite(evidence):
            raise ValueError("POMCP belief update received an impossible observation")
        self.probabilities = posterior / evidence

    @property
    def normalized_entropy(self) -> float:
        positive = self.probabilities[self.probabilities > 0]
        entropy = float(-np.sum(positive * np.log(positive)))
        maximum = float(np.log(self.probabilities.size))
        return 0.0 if maximum <= 0 else entropy / maximum


class POMCPPlanner:
    """Root-sampling Monte Carlo tree search with tree reuse across decisions."""

    def __init__(self, *, layout: Any, belief: TabularTargetBelief, config: POMCPConfig, seed: int):
        self.layout = layout
        self.belief = belief
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.root = HistoryNode()
        self._mos = _pomcp_mos_imports()
        self._actions = tuple(int(action) for action in self._mos["MOSAction"])
        self._find_action = int(self._mos["MOSAction"].FIND)
        self._build_static_model()

    def _build_static_model(self) -> None:
        """Cache the deterministic map and sensor model outside online planning."""

        size = self.layout.size
        states = size**2
        self._next_state = np.zeros((states, len(self._actions)), dtype=np.int32)
        self._collision = np.zeros((states, len(self._actions)), dtype=bool)
        self._detection_probability = np.zeros((states, states), dtype=np.float32)
        self._visible = np.zeros((states, states), dtype=bool)
        self._rollout_action = np.full((states, states), self._find_action, dtype=np.int8)
        free_states = [
            y * size + x for y in range(size) for x in range(size) if self.layout.is_free((x, y))
        ]
        movement_actions = tuple(action for action in self._actions if action != self._find_action)
        for robot_state in free_states:
            robot = (robot_state % size, robot_state // size)
            for action in self._actions:
                next_robot = self.layout.move(robot, self._mos["MOSAction"](action))
                next_state = next_robot[1] * size + next_robot[0]
                self._next_state[robot_state, action] = next_state
                self._collision[robot_state, action] = (
                    action != self._find_action and next_state == robot_state
                )
            for target_state in free_states:
                target = (target_state % size, target_state // size)
                distribution = self._mos["detection_distribution"](robot, target, self.layout)
                self._detection_probability[robot_state, target_state] = distribution[1]
                self._visible[robot_state, target_state] = self.layout.target_is_visible(
                    robot, target
                )

        neighbors: dict[int, list[tuple[int, int]]] = {}
        for state in free_states:
            options = []
            for action in movement_actions:
                next_state = int(self._next_state[state, action])
                if next_state != state:
                    options.append((next_state, action))
            neighbors[state] = options

        for target_state in free_states:
            distance = np.full(states, np.inf)
            queue: deque[int] = deque()
            visible_states = [state for state in free_states if self._visible[state, target_state]]
            for state in visible_states:
                distance[state] = 0.0
                queue.append(state)
            while queue:
                state = queue.popleft()
                for neighbor, _ in neighbors[state]:
                    if distance[neighbor] > distance[state] + 1.0:
                        distance[neighbor] = distance[state] + 1.0
                        queue.append(neighbor)
            for state in free_states:
                if self._visible[state, target_state]:
                    self._rollout_action[state, target_state] = self._find_action
                    continue
                options = neighbors[state]
                if options:
                    _, action = min(options, key=lambda item: distance[item[0]])
                    self._rollout_action[state, target_state] = action

    def sensor_model(self, robot: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        state = robot[1] * self.layout.size + robot[0]
        return self._detection_probability[state], self._visible[state]

    def _generate(
        self,
        robot: tuple[int, int],
        target: tuple[int, int],
        action: int,
    ) -> tuple[tuple[int, int], tuple[int, int], float, bool]:
        size = self.layout.size
        robot_state = robot[1] * size + robot[0]
        target_state = target[1] * size + target[0]
        next_state = int(self._next_state[robot_state, action])
        next_robot = (next_state % size, next_state // size)
        collision = bool(self._collision[robot_state, action])
        detection = int(self.rng.random() < self._detection_probability[next_state, target_state])
        if action == self._find_action:
            success = bool(self._visible[next_state, target_state])
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
        return next_robot, (detection, find), float(reward), bool(success)

    def _shortest_path_action(
        self,
        robot: tuple[int, int],
        target: tuple[int, int],
    ) -> int:
        size = self.layout.size
        robot_state = robot[1] * size + robot[0]
        target_state = target[1] * size + target[0]
        return int(self._rollout_action[robot_state, target_state])

    def _rollout(
        self,
        robot: tuple[int, int],
        target: tuple[int, int],
        depth: int,
    ) -> float:
        if depth <= 0:
            return 0.0
        if self.rng.random() < self.config.rollout_random_probability:
            action = int(self.rng.choice(self._actions))
        else:
            action = self._shortest_path_action(robot, target)
        next_robot, _, reward, terminal = self._generate(robot, target, action)
        if terminal:
            return reward
        return reward + self.config.discount * self._rollout(next_robot, target, depth - 1)

    def _tree_action(self, node: HistoryNode) -> int:
        for action in self._actions:
            node.actions.setdefault(action, ActionNode())
        unvisited = [action for action, branch in node.actions.items() if branch.visits == 0]
        if unvisited:
            return int(self.rng.choice(unvisited))
        log_visits = math.log(max(1, node.visits))
        scores = {
            action: branch.value
            + self.config.exploration_constant * math.sqrt(log_visits / branch.visits)
            for action, branch in node.actions.items()
        }
        maximum = max(scores.values())
        return int(
            self.rng.choice([action for action, score in scores.items() if score == maximum])
        )

    def _simulate(
        self,
        robot: tuple[int, int],
        target: tuple[int, int],
        node: HistoryNode,
        depth: int,
    ) -> float:
        if depth <= 0:
            return 0.0
        action = self._tree_action(node)
        branch = node.actions[action]
        next_robot, observation, reward, terminal = self._generate(robot, target, action)
        if terminal:
            total = reward
        elif observation not in branch.children:
            child = HistoryNode()
            branch.children[observation] = child
            total = reward + self.config.discount * self._rollout(next_robot, target, depth - 1)
        else:
            total = reward + self.config.discount * self._simulate(
                next_robot,
                target,
                branch.children[observation],
                depth - 1,
            )
        node.visits += 1
        branch.visits += 1
        branch.value += (total - branch.value) / branch.visits
        return float(total)

    def plan(self, robot: tuple[int, int]) -> PlanResult:
        started = time.perf_counter_ns()
        completed = 0
        while completed < self.config.simulations:
            if self.config.planning_budget_ms is not None and completed > 0:
                elapsed_ms = (time.perf_counter_ns() - started) / 1e6
                if elapsed_ms >= self.config.planning_budget_ms:
                    break
            target = self.belief.sample(self.rng)
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
            for action, visit in zip(self._actions, visits, strict=True)
            if visit == most_visited
        ]
        action = max(candidates, key=lambda candidate: self.root.actions[candidate].value)
        return PlanResult(
            action=int(action),
            simulations=completed,
            planning_ms=float(planning_ms),
            action_values=values,
            action_visits=visits,
        )

    def advance_root(self, action: int, observation: tuple[int, int, int, int, int]) -> None:
        key = (int(observation[2]), int(observation[3]))
        branch = self.root.actions.get(int(action))
        self.root = HistoryNode() if branch is None else branch.children.get(key, HistoryNode())


def run_pomcp_episode(
    instance_seed: int,
    config: POMCPConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run standard POMCP on one matched MOS instance."""

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
        lambda: POMCPPlanner(
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
    trajectory: list[dict[str, Any]] = []

    for decision_index in range(config.max_steps):
        position = environment.position
        plan = planner.plan(position)
        planning_ms += plan.planning_ms
        total_simulations += plan.simulations
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
                "belief_normalized_entropy": belief.normalized_entropy,
                "planning_ms": plan.planning_ms,
                "belief_update_ms": update_ms,
                "simulations": plan.simulations,
                "action_values": json.dumps(plan.action_values),
                "action_visits": json.dumps(plan.action_visits),
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
        "controller": "pomcp",
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
    }
    return summary, trajectory


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    known: set[str] = set()
    for row in rows:
        for name in row:
            if name not in known:
                known.add(name)
                fieldnames.append(name)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _run_instance(instance_seed: int, config: POMCPConfig) -> dict[str, Any]:
    summary, trajectory = run_pomcp_episode(instance_seed, config)
    for row in trajectory:
        row["instance_seed"] = int(instance_seed)
    return {"summary": summary, "trajectory": trajectory}


def evaluate_pomcp(
    *,
    instance_seeds: list[int] | tuple[int, ...],
    output_dir: Path,
    config: POMCPConfig,
    instance_workers: int = 1,
    resume: bool = True,
) -> dict[str, Any]:
    """Run resumable POMCP evaluation and aggregate episode statistics."""

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
    pending = []
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
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def save_pomcp_episode(
    summary: dict[str, Any],
    trajectory: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_rows(output_dir / "trajectory.csv", trajectory)
