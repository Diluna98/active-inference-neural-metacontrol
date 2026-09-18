"""Parallel, resumable orchestration for MOS counterfactual generation."""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

from .allocations import ALLOCATIONS, Allocation
from .counterfactuals import (
    CounterfactualDataset,
    concatenate_counterfactual_datasets,
    generate_mos_counterfactuals,
    load_generated_counterfactuals,
    save_counterfactual_dataset,
)


@dataclass(frozen=True)
class GenerationConfig:
    reference_resolution: int = 20
    reference_depth: int = 1
    max_steps: int = 50
    branch_stride: int = 1
    message_passing_iterations: int = 10
    policy_workers: int = 1
    allocations: tuple[tuple[int, int], ...] = tuple(
        (item.resolution, item.depth) for item in ALLOCATIONS
    )

    def __post_init__(self) -> None:
        Allocation(self.reference_resolution, self.reference_depth)
        if self.max_steps < 2 or self.branch_stride < 1:
            raise ValueError("max_steps must be at least 2 and branch_stride must be positive")
        if self.message_passing_iterations < 1 or self.policy_workers < 1:
            raise ValueError("inference worker and iteration counts must be positive")
        parsed = tuple(Allocation(*values) for values in self.allocations)
        if not parsed or len(set(parsed)) != len(parsed):
            raise ValueError("allocations must be nonempty and unique")
        if Allocation(self.reference_resolution, self.reference_depth) not in parsed:
            raise ValueError("reference allocation must be included in allocations")

    @property
    def allocation_objects(self) -> tuple[Allocation, ...]:
        return tuple(Allocation(*values) for values in self.allocations)


def _marker_payload(instance_seed: int, config: GenerationConfig) -> dict:
    configuration = json.loads(json.dumps(asdict(config)))
    return {"instance_seed": int(instance_seed), "configuration": configuration}


def _is_complete(shard_dir: Path, instance_seed: int, config: GenerationConfig) -> bool:
    marker = shard_dir / "complete.json"
    if not marker.is_file() or not (shard_dir / "training_data.npz").is_file():
        return False
    try:
        return json.loads(marker.read_text(encoding="utf-8")) == _marker_payload(
            instance_seed, config
        )
    except (OSError, json.JSONDecodeError):
        return False


def _generate_shard(
    instance_seed: int,
    config: GenerationConfig,
    shard_dir: Path,
) -> tuple[int, int, int]:
    (shard_dir / "complete.json").unlink(missing_ok=True)
    dataset = generate_mos_counterfactuals(
        instance_seeds=[instance_seed],
        reference_allocation=Allocation(config.reference_resolution, config.reference_depth),
        max_steps=config.max_steps,
        branch_stride=config.branch_stride,
        message_passing_iterations=config.message_passing_iterations,
        policy_workers=config.policy_workers,
        allocations=config.allocation_objects,
    )
    save_counterfactual_dataset(dataset, shard_dir)
    temporary_marker = shard_dir / "complete.json.tmp"
    temporary_marker.write_text(
        json.dumps(_marker_payload(instance_seed, config), indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_marker.replace(shard_dir / "complete.json")
    return instance_seed, len(dataset.context_ids), len(dataset.branches)


def generate_resumable_mos_counterfactuals(
    *,
    instance_seeds: list[int] | tuple[int, ...],
    output_dir: Path,
    config: GenerationConfig | None = None,
    instance_workers: int = 1,
    resume: bool = True,
) -> CounterfactualDataset:
    """Generate per-instance shards in parallel and assemble them deterministically."""

    config = config or GenerationConfig()
    seeds = tuple(int(value) for value in instance_seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("instance_seeds must be nonempty and unique")
    if instance_workers < 1:
        raise ValueError("instance_workers must be positive")
    output_dir = Path(output_dir)
    shard_root = output_dir / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    pending = []
    for seed in seeds:
        shard_dir = shard_root / f"instance-{seed}"
        if resume and _is_complete(shard_dir, seed, config):
            print(f"[reuse] instance={seed}", flush=True)
        else:
            pending.append((seed, shard_dir))

    if instance_workers == 1:
        for completed, (seed, shard_dir) in enumerate(pending, start=1):
            _, contexts, branches = _generate_shard(seed, config, shard_dir)
            print(
                f"[generate {completed}/{len(pending)}] instance={seed} "
                f"contexts={contexts} branches={branches}",
                flush=True,
            )
    elif pending:
        with ProcessPoolExecutor(max_workers=instance_workers) as executor:
            futures = {
                executor.submit(_generate_shard, seed, config, shard_dir): seed
                for seed, shard_dir in pending
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                seed, contexts, branches = future.result()
                print(
                    f"[generate {completed}/{len(pending)}] instance={seed} "
                    f"contexts={contexts} branches={branches}",
                    flush=True,
                )

    missing = [
        seed for seed in seeds if not _is_complete(shard_root / f"instance-{seed}", seed, config)
    ]
    if missing:
        raise RuntimeError(f"generation did not complete for instances: {missing}")
    shards = [load_generated_counterfactuals(shard_root / f"instance-{seed}") for seed in seeds]
    combined = concatenate_counterfactual_datasets(shards)
    save_counterfactual_dataset(combined, output_dir)
    manifest = {
        "instance_seeds": list(seeds),
        "instance_workers": instance_workers,
        "resume": resume,
        "configuration": asdict(config),
        "contexts": len(combined.context_ids),
        "branches": len(combined.branches),
    }
    (output_dir / "generation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return combined
