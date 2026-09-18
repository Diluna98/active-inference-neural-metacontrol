import json
from pathlib import Path

import numpy as np
import pytest

from active_inference_neural_metacontrol import Allocation
from active_inference_neural_metacontrol.counterfactuals import (
    generate_mos_counterfactuals,
    save_counterfactual_dataset,
)

pytest.importorskip("active_inference_navigation.mos")


def test_small_counterfactual_dataset_is_aligned_and_serializable():
    allocations = (Allocation(2, 1), Allocation(5, 1))
    dataset = generate_mos_counterfactuals(
        instance_seeds=[0],
        reference_allocation=allocations[0],
        allocations=allocations,
        max_steps=2,
        message_passing_iterations=1,
    )

    assert dataset.spatial.shape == (1, 6, 20, 20)
    assert dataset.context.shape == (1, 17)
    assert dataset.success.shape == (1, 2)
    assert dataset.task_cost.shape == (1, 2)
    assert dataset.compute_ms.shape == (1, 2)
    assert dataset.candidate_actions.shape == (1, 2)
    assert np.all(dataset.compute_ms >= 0.0)
    assert dataset.contexts[0]["candidate_decision"] == 1
    assert [row["decision"] for row in dataset.trajectory] == [0, 1]

    output_dir = Path("data/generated/test-counterfactuals")
    save_counterfactual_dataset(dataset, output_dir)
    archive = np.load(output_dir / "training_data.npz")
    assert archive["spatial"].shape == dataset.spatial.shape
    assert archive["resolutions"].tolist() == [2, 5]
    assert json.loads((output_dir / "summary.json").read_text())["contexts"] == 1
