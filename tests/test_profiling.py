import csv
import json
from pathlib import Path

import numpy as np

from active_inference_neural_metacontrol import ALLOCATIONS
from active_inference_neural_metacontrol.profiling import (
    build_timing_profile,
    load_timing_profile,
    save_timing_profile,
)


def test_profile_separates_inference_and_source_target_switching():
    output = Path("data/generated/test-profile")
    output.mkdir(parents=True, exist_ok=True)
    samples = len(ALLOCATIONS)
    context_ids = np.asarray([f"context-{index}" for index in range(samples)])
    np.savez_compressed(
        output / "training_data.npz",
        spatial=np.zeros((samples, 6, 20, 20), dtype=np.float32),
        context=np.zeros((samples, 16), dtype=np.float32),
        normalized_g=np.ones((samples, samples), dtype=np.float32),
        raw_g=np.ones((samples, samples), dtype=np.float32),
        risk=np.ones((samples, samples), dtype=np.float32),
        ambiguity=np.ones((samples, samples), dtype=np.float32),
        information_gain=np.zeros((samples, samples), dtype=np.float32),
        compute_ms=np.tile(np.arange(1, samples + 1), (samples, 1)).astype(np.float32),
        switch_ms=np.ones((samples, samples), dtype=np.float32),
        candidate_actions=np.zeros((samples, samples), dtype=np.int8),
        context_ids=context_ids,
        resolutions=np.asarray([item.resolution for item in ALLOCATIONS]),
        depths=np.asarray([item.depth for item in ALLOCATIONS]),
    )
    with (output / "contexts.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "context_id",
                "instance_seed",
                "reference_resolution",
                "reference_depth",
            ],
        )
        writer.writeheader()
        for index, allocation in enumerate(ALLOCATIONS):
            writer.writerow(
                {
                    "context_id": context_ids[index],
                    "instance_seed": index,
                    "reference_resolution": allocation.resolution,
                    "reference_depth": allocation.depth,
                }
            )
    (output / "generation_manifest.json").write_text(
        json.dumps({"instance_workers": 1}), encoding="utf-8"
    )

    profile = build_timing_profile(output)

    assert profile["controlled_single_process"]
    assert profile["all_sources_observed"]
    assert len(profile["inference"]) == 12
    assert len(profile["switching"]) == 144
    profile_path = output / "profile.json"
    save_timing_profile(profile, profile_path)
    inference_ms, switching_ms = load_timing_profile(profile_path)
    assert inference_ms.shape == (12,)
    assert switching_ms.shape == (12, 12)
