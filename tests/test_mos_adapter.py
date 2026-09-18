import numpy as np
import pytest

from active_inference_neural_metacontrol import Allocation
from active_inference_neural_metacontrol.mos_adapter import (
    build_mos_features,
    canonical_detection_likelihood,
    obstacle_map,
)

pytest.importorskip("active_inference_navigation.mos")


def test_mos_feature_adapter_builds_canonical_channels():
    from active_inference_navigation.mos import (
        MOSAgentConfig,
        MOSLayout,
        build_mos_agent,
        selected_mos_action,
    )

    layout = MOSLayout(map_seed=0)
    agent = build_mos_agent(
        MOSAgentConfig(
            target_resolution=2,
            action_depth=1,
            message_passing_iterations=1,
        ),
        layout=layout,
    )
    agent.reset()
    agent.observe((layout.start[0], layout.start[1], 0, 0, 0), time_step=0)
    agent.infer_states()
    agent.infer_policies()
    action = selected_mos_action(agent)
    next_position = layout.move(layout.start, action)
    features = build_mos_features(
        agent=agent,
        allocation=Allocation(2, 1),
        layout=layout,
        selected_action=int(action),
        next_robot_position=next_position,
    )

    assert features.spatial.tensor.shape == (6, 20, 20)
    assert features.context.shape == (17,)
    assert np.isclose(features.spatial.tensor[0].sum(), 1.0)
    assert np.isclose(features.spatial.tensor[1].sum(), 1.0)
    assert np.isclose(features.spatial.tensor[2].sum(), 1.0)
    assert features.predicted_posterior_l1 >= 0.0


def test_mos_geometry_and_likelihood_use_row_major_maps():
    from active_inference_navigation.mos import MOSLayout

    layout = MOSLayout(map_seed=3)
    obstacles = obstacle_map(layout)
    likelihood = canonical_detection_likelihood(layout, layout.start)

    assert obstacles.shape == (20, 20)
    assert all(obstacles[y, x] == 1.0 for x, y in layout.blocked)
    assert likelihood.shape == (2, 20, 20)
    assert np.allclose(likelihood.sum(axis=0), 1.0)
