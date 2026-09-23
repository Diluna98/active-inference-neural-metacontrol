"""Multi-head training for task, Active-Inference, and resource targets."""

from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .allocations import ALLOCATIONS
from .datasets import CounterfactualArrays, DatasetSplit
from .models import MultiObjectiveNetwork
from .profiling import load_timing_profile

try:
    import torch
    from torch.nn import functional
    from torch.utils.data import DataLoader, Dataset
except ImportError:  # pragma: no cover
    torch = None
    functional = None
    DataLoader = None
    Dataset = object


REGRESSION_KEYS = (
    "task_cost",
    "normalized_g",
    "preference",
    "epistemic",
    "compute_log",
    "switch_log",
)


@dataclass(frozen=True)
class MultiObjectiveTrainingConfig:
    epochs: int = 200
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 25
    ranking_temperature: float = 0.1
    success_weight: float = 1.0
    regression_weight: float = 1.0
    ranking_weight: float = 1.0
    seed: int = 0
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1 or self.patience < 1:
            raise ValueError("epochs, batch_size, and patience must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid optimizer settings")
        if self.ranking_temperature <= 0:
            raise ValueError("ranking_temperature must be positive")
        if min(self.success_weight, self.regression_weight, self.ranking_weight) < 0:
            raise ValueError("loss weights must be nonnegative")


def _raw_targets(arrays: CounterfactualArrays) -> dict[str, np.ndarray]:
    if arrays.success is None or arrays.task_cost is None:
        raise ValueError("multi-objective training requires success and task_cost labels")
    depths = np.asarray([item.depth for item in ALLOCATIONS], dtype=np.float32)[None, :]
    return {
        "success": np.asarray(arrays.success, dtype=np.float32),
        "task_cost": np.log1p(np.asarray(arrays.task_cost, dtype=np.float32)),
        "normalized_g": np.asarray(arrays.normalized_g, dtype=np.float32),
        "preference": np.asarray(arrays.risk, dtype=np.float32) / depths,
        "epistemic": (
            np.asarray(arrays.ambiguity, dtype=np.float32)
            - np.asarray(arrays.information_gain, dtype=np.float32)
        )
        / depths,
        "compute_log": np.log1p(np.asarray(arrays.compute_ms, dtype=np.float32)),
        "switch_log": np.log1p(np.asarray(arrays.switch_ms, dtype=np.float32)),
    }


def _target_stats(arrays: CounterfactualArrays) -> dict[str, dict[str, float]]:
    targets = _raw_targets(arrays)
    stats = {}
    for key in REGRESSION_KEYS:
        mean = float(targets[key].mean())
        scale = float(targets[key].std())
        stats[key] = {"mean": mean, "scale": max(scale, 1e-6)}
    return stats


class _MultiObjectiveDataset(Dataset):
    def __init__(
        self,
        arrays: CounterfactualArrays,
        context_mean: np.ndarray,
        context_scale: np.ndarray,
        target_stats: dict[str, dict[str, float]],
    ) -> None:
        self.spatial = torch.from_numpy(arrays.spatial)
        context = (arrays.context - context_mean) / context_scale
        self.context = torch.from_numpy(context.astype(np.float32))
        targets = _raw_targets(arrays)
        self.targets = {"success": torch.from_numpy(targets["success"])}
        for key in REGRESSION_KEYS:
            values = (targets[key] - target_stats[key]["mean"]) / target_stats[key]["scale"]
            self.targets[key] = torch.from_numpy(values.astype(np.float32))

    def __len__(self) -> int:
        return len(self.spatial)

    def __getitem__(self, index: int) -> tuple[Any, ...]:
        return self.spatial[index], self.context[index], {
            key: value[index] for key, value in self.targets.items()
        }


def _device(name: str) -> Any:
    if torch is None:
        raise ImportError("multi-objective training requires PyTorch")
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = torch.device(name)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return result


def _losses(
    prediction: dict[str, Any],
    target: dict[str, Any],
    *,
    stats: dict[str, dict[str, float]],
    config: MultiObjectiveTrainingConfig,
) -> dict[str, Any]:
    values = {
        "success": functional.binary_cross_entropy_with_logits(
            prediction["success_logits"], target["success"]
        )
    }
    head_names = {
        "task_cost": "task_cost_scaled",
        "normalized_g": "normalized_g_scaled",
        "preference": "preference_scaled",
        "epistemic": "epistemic_scaled",
        "compute_log": "compute_log_scaled",
        "switch_log": "switch_log_scaled",
    }
    for key, head in head_names.items():
        values[key] = functional.smooth_l1_loss(prediction[head], target[key])
    g_mean = stats["normalized_g"]["mean"]
    g_scale = stats["normalized_g"]["scale"]
    predicted_g = prediction["normalized_g_scaled"] * g_scale + g_mean
    target_g = target["normalized_g"] * g_scale + g_mean
    target_probability = functional.softmax(target_g / config.ranking_temperature, dim=1)
    predicted_log_probability = functional.log_softmax(
        predicted_g / config.ranking_temperature, dim=1
    )
    values["ranking"] = -(target_probability * predicted_log_probability).sum(dim=1).mean()
    regression = sum(values[key] for key in REGRESSION_KEYS) / len(REGRESSION_KEYS)
    values["loss"] = (
        config.success_weight * values["success"]
        + config.regression_weight * regression
        + config.ranking_weight * values["ranking"]
    )
    return values


def _run_epoch(
    model: Any,
    loader: Any,
    *,
    device: Any,
    stats: dict[str, dict[str, float]],
    config: MultiObjectiveTrainingConfig,
    optimizer: Any | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    keys = ("loss", "success", *REGRESSION_KEYS, "ranking")
    totals = {key: 0.0 for key in keys}
    samples = 0
    for spatial, context, target in loader:
        spatial = spatial.to(device)
        context = context.to(device)
        target = {key: value.to(device) for key, value in target.items()}
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            values = _losses(model(spatial, context), target, stats=stats, config=config)
            if training:
                values["loss"].backward()
                optimizer.step()
        count = spatial.shape[0]
        samples += count
        for key in keys:
            totals[key] += count * float(values[key].item())
    return {key: value / samples for key, value in totals.items()}


def _predict(
    model: Any,
    arrays: CounterfactualArrays,
    *,
    context_mean: np.ndarray,
    context_scale: np.ndarray,
    stats: dict[str, dict[str, float]],
    device: Any,
    batch_size: int,
) -> dict[str, np.ndarray]:
    loader = DataLoader(
        _MultiObjectiveDataset(arrays, context_mean, context_scale, stats),
        batch_size=batch_size,
        shuffle=False,
    )
    values: dict[str, list[np.ndarray]] = {
        "success_probability": [],
        **{key: [] for key in REGRESSION_KEYS},
    }
    model.eval()
    head_names = {
        "task_cost": "task_cost_scaled",
        "normalized_g": "normalized_g_scaled",
        "preference": "preference_scaled",
        "epistemic": "epistemic_scaled",
        "compute_log": "compute_log_scaled",
        "switch_log": "switch_log_scaled",
    }
    with torch.no_grad():
        for spatial, context, _ in loader:
            output = model(spatial.to(device), context.to(device))
            values["success_probability"].append(
                torch.sigmoid(output["success_logits"]).cpu().numpy()
            )
            for key, head in head_names.items():
                scaled = output[head].cpu().numpy()
                values[key].append(scaled * stats[key]["scale"] + stats[key]["mean"])
    result = {key: np.concatenate(parts) for key, parts in values.items()}
    result["task_cost"] = np.maximum(0.0, np.expm1(result["task_cost"]))
    result["compute_ms"] = np.maximum(0.0, np.expm1(result.pop("compute_log")))
    result["switch_ms"] = np.maximum(0.0, np.expm1(result.pop("switch_log")))
    return result


def _evaluate(arrays: CounterfactualArrays, prediction: dict[str, np.ndarray]) -> dict[str, float]:
    target = _raw_targets(arrays)
    success_probability = np.clip(prediction["success_probability"], 0.0, 1.0)
    task_cost = np.asarray(arrays.task_cost, dtype=float)
    depths = np.asarray([item.depth for item in ALLOCATIONS], dtype=float)[None, :]
    preference = arrays.risk / depths
    epistemic = (arrays.ambiguity - arrays.information_gain) / depths
    return {
        "samples": len(arrays.context),
        "success_brier": float(np.mean((success_probability - target["success"]) ** 2)),
        "success_accuracy": float(
            np.mean((success_probability >= 0.5) == (target["success"] >= 0.5))
        ),
        "task_cost_mae": float(np.mean(np.abs(prediction["task_cost"] - task_cost))),
        "normalized_g_mae": float(
            np.mean(np.abs(prediction["normalized_g"] - arrays.normalized_g))
        ),
        "preference_mae": float(np.mean(np.abs(prediction["preference"] - preference))),
        "epistemic_mae": float(np.mean(np.abs(prediction["epistemic"] - epistemic))),
        "compute_ms_mae": float(np.mean(np.abs(prediction["compute_ms"] - arrays.compute_ms))),
        "switch_ms_mae": float(np.mean(np.abs(prediction["switch_ms"] - arrays.switch_ms))),
    }


def train_multiobjective_model(
    split: DatasetSplit,
    *,
    output_dir: Path,
    config: MultiObjectiveTrainingConfig | None = None,
    timing_profile: Path | None = None,
) -> dict[str, Any]:
    """Train and evaluate all task, inference-value, and resource heads."""

    config = config or MultiObjectiveTrainingConfig()
    if split.train.success is None or split.train.task_cost is None:
        raise ValueError("the dataset does not contain task-outcome labels")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    device = _device(config.device)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    context_mean = split.train.context.mean(axis=0).astype(np.float32)
    context_scale = split.train.context.std(axis=0).astype(np.float32)
    context_scale[context_scale < 1e-6] = 1.0
    stats = _target_stats(split.train)

    def loader(arrays: CounterfactualArrays, *, shuffle: bool) -> Any:
        return DataLoader(
            _MultiObjectiveDataset(arrays, context_mean, context_scale, stats),
            batch_size=config.batch_size,
            shuffle=shuffle,
            generator=torch.Generator().manual_seed(config.seed) if shuffle else None,
        )

    train_loader = loader(split.train, shuffle=True)
    validation_loader = loader(split.validation, shuffle=False)
    model = MultiObjectiveNetwork(
        spatial_channels=split.train.spatial.shape[1],
        context_features=split.train.context.shape[1],
        allocations=len(ALLOCATIONS),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    best_validation = np.inf
    best_state = None
    stale = 0
    history = []
    for epoch in range(1, config.epochs + 1):
        train_metrics = _run_epoch(
            model,
            train_loader,
            device=device,
            stats=stats,
            config=config,
            optimizer=optimizer,
        )
        validation_metrics = _run_epoch(
            model,
            validation_loader,
            device=device,
            stats=stats,
            config=config,
            optimizer=None,
        )
        history.append(
            {
                "epoch": epoch,
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"validation_{key}": value for key, value in validation_metrics.items()},
            }
        )
        if validation_metrics["loss"] < best_validation - 1e-7:
            best_validation = validation_metrics["loss"]
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)

    predictions = {}
    evaluations = {}
    for name, arrays in (
        ("train", split.train),
        ("validation", split.validation),
        ("test", split.test),
    ):
        predicted = _predict(
            model,
            arrays,
            context_mean=context_mean,
            context_scale=context_scale,
            stats=stats,
            device=device,
            batch_size=config.batch_size,
        )
        for key, values in predicted.items():
            predictions[f"{name}_{key}"] = values
        predictions[f"{name}_context_ids"] = arrays.context_ids
        evaluations[name] = _evaluate(arrays, predicted)

    compute_profile = None
    switching_profile = None
    if timing_profile is not None:
        compute_profile, switching_profile = load_timing_profile(timing_profile)
    checkpoint = {
        "schema_version": 6,
        "model_type": "multiobjective",
        "prediction_targets": [
            "success",
            "remaining_task_cost",
            "normalized_g",
            "preference_per_depth",
            "epistemic_per_depth",
            "compute_ms",
            "switch_ms",
        ],
        "model_state_dict": best_state,
        "spatial_channels": split.train.spatial.shape[1],
        "context_features": split.train.context.shape[1],
        "allocations": [(item.resolution, item.depth) for item in ALLOCATIONS],
        "context_mean": context_mean,
        "context_scale": context_scale,
        "target_stats": stats,
        "compute_profile_ms": compute_profile,
        "switching_profile_ms": switching_profile,
        "timing_profile": None if timing_profile is None else str(timing_profile),
        "training_config": asdict(config),
        "train_instances": split.train_instances,
        "validation_instances": split.validation_instances,
        "test_instances": split.test_instances,
    }
    torch.save(checkpoint, output_dir / "best_model.pt")
    np.savez_compressed(output_dir / "predictions.npz", **predictions)
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    report = {
        "device": str(device),
        "schema_version": 6,
        "model_type": "multiobjective",
        "epochs_completed": len(history),
        "best_validation_loss": float(best_validation),
        "split": {
            "train_instances": list(split.train_instances),
            "validation_instances": list(split.validation_instances),
            "test_instances": list(split.test_instances),
            "train_samples": len(split.train.context),
            "validation_samples": len(split.validation.context),
            "test_samples": len(split.test.context),
        },
        "evaluation": evaluations,
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
