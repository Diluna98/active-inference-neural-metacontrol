import numpy as np
import pytest

from active_inference_neural_metacontrol.multi_resolution_pomcp import (
    AbstractTargetBelief,
    MultiResolutionPOMCPConfig,
    MultiResolutionPOMCPPlanner,
)
from active_inference_neural_metacontrol.pomcp import (
    HistoryNode,
    POMCPConfig,
    POMCPPlanner,
    TabularTargetBelief,
)


class OpenLayout:
    size = 4
    start = (0, 0)
    blocked = frozenset()

    @staticmethod
    def is_free(position):
        return 0 <= position[0] < 4 and 0 <= position[1] < 4

    def move(self, position, action):
        if int(action) == 4:
            return position
        dx, dy = {0: (0, 1), 1: (0, -1), 2: (-1, 0), 3: (1, 0)}[int(action)]
        candidate = position[0] + dx, position[1] + dy
        return candidate if self.is_free(candidate) else position

    @staticmethod
    def target_is_visible(robot, target):
        return robot == target

    @staticmethod
    def visibility(robot, target):
        return float(robot == target)


def test_config_rejects_invalid_budget():
    with pytest.raises(ValueError, match="planning_budget_ms"):
        POMCPConfig(planning_budget_ms=0.0)


def test_config_rejects_unknown_reward_mode():
    with pytest.raises(ValueError, match="reward_mode"):
        POMCPConfig(reward_mode="unknown")


def test_tabular_belief_is_normalized_over_free_cells():
    belief = TabularTargetBelief(OpenLayout())

    assert belief.probabilities.sum() == pytest.approx(1.0)
    assert np.count_nonzero(belief.probabilities) == 16


def test_tree_action_visits_every_action_before_ucb():
    belief = TabularTargetBelief(OpenLayout())
    planner = POMCPPlanner(
        layout=OpenLayout(),
        belief=belief,
        config=POMCPConfig(simulations=5, search_depth=1),
        seed=0,
    )
    node = HistoryNode()

    selected = set()
    for _ in range(5):
        action = planner._tree_action(node)
        selected.add(action)
        node.actions[action].visits += 1
        node.visits += 1

    assert selected == {0, 1, 2, 3, 4}


def test_first_action_path_moves_toward_sampled_target():
    belief = TabularTargetBelief(OpenLayout())
    planner = POMCPPlanner(
        layout=OpenLayout(),
        belief=belief,
        config=POMCPConfig(simulations=5),
        seed=0,
    )

    action = planner._shortest_path_action((0, 0), (3, 0))

    assert action == 3


def test_abstract_belief_preserves_probability_mass():
    belief = TabularTargetBelief(OpenLayout())
    belief.probabilities[:] = 0.0
    belief.probabilities[0] = 0.25
    belief.probabilities[15] = 0.75

    abstract = AbstractTargetBelief(belief, resolution=2)

    assert abstract.probabilities.tolist() == pytest.approx([0.25, 0.0, 0.0, 0.75])


def test_multi_resolution_budget_is_split_without_inflation():
    belief = TabularTargetBelief(OpenLayout())
    planner = MultiResolutionPOMCPPlanner(
        layout=OpenLayout(),
        belief=belief,
        config=MultiResolutionPOMCPConfig(resolutions=(2, 4), simulations=10, search_depth=1),
        seed=0,
    )

    result = planner.plan((0, 0))

    assert result.simulations == 10
    assert result.resolution in {2, 4}
    assert len(result.level_values) == 2
