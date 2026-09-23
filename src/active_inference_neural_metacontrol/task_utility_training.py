"""Training for the minimal success, operational-cost, and normalized-G model."""

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
from .models import TaskUtilityNetwork
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


@dataclass(frozen=True)
class TaskUtilityTrainingConfig:
    epochs: int = 200
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 25
    ranking_temperature: float = 0.1
    success_weight: float = 1.0
    cost_weight: float = 1.0
    normalized_g_weight: float = 1.0
    ranking_weight: float = 1.0
    failure_penalty: float = 100.0
    seed: int = 0
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1 or self.patience < 1:
            raise ValueError("epochs, batch_size, and patience must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid optimizer settings")
        if self.ranking_temperature <= 0 or self.failure_penalty < 0:
            raise ValueError("ranking temperature must be positive and failure penalty nonnegative")


def _targets(
    arrays: CounterfactualArrays, failure_penalty: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if arrays.success is None or arrays.task_cost is None:
        raise ValueError("task-utility training requires success and task_cost labels")
    success = np.asarray(arrays.success, dtype=np.float32)
    operational = np.asarray(arrays.task_cost, dtype=np.float32) - failure_penalty * (
        1.0 - success
    )
    if np.any(operational < -1e-5):
        raise ValueError("failure penalty exceeds recorded task cost")
    return success, np.maximum(operational, 0.0), np.asarray(arrays.normalized_g, dtype=np.float32)


def _stats(arrays: CounterfactualArrays, failure_penalty: float) -> dict[str, dict[str, float]]:
    _, operational, normalized_g = _targets(arrays, failure_penalty)
    values = {
        "operational_cost_log": np.log1p(operational),
        "normalized_g": normalized_g,
    }
    return {
        key: {
            "mean": float(value.mean()),
            "scale": max(float(value.std()), 1e-6),
        }
        for key, value in values.items()
    }


class _TaskUtilityDataset(Dataset):
    def __init__(
        self,
        arrays: CounterfactualArrays,
        *,
        context_mean: np.ndarray,
        context_scale: np.ndarray,
        stats: dict[str, dict[str, float]],
        failure_penalty: float,
    ) -> None:
        self.spatial = torch.from_numpy(arrays.spatial)
        context = (arrays.context - context_mean) / context_scale
        self.context = torch.from_numpy(context.astype(np.float32))
        success, operational, normalized_g = _targets(arrays, failure_penalty)
        operational_log = np.log1p(operational)
        self.success = torch.from_numpy(success)
        self.operational_cost = torch.from_numpy(
            (
                (operational_log - stats["operational_cost_log"]["mean"])
                / stats["operational_cost_log"]["scale"]
            ).astype(np.float32)
        )
        self.normalized_g = torch.from_numpy(
            (
                (normalized_g - stats["normalized_g"]["mean"])
                / stats["normalized_g"]["scale"]
            ).astype(np.float32)
        )

    def __len__(self) -> int:
        return len(self.spatial)

    def __getitem__(self, index: int) -> tuple[Any, ...]:
        return (
            self.spatial[index],
            self.context[index],
            self.success[index],
            self.operational_cost[index],
            self.normalized_g[index],
        )


def _run_epoch(
    model: Any,
    loader: Any,
    *,
    device: Any,
    stats: dict[str, dict[str, float]],
    config: TaskUtilityTrainingConfig,
    optimizer: Any | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = np.zeros(5, dtype=float)
    samples = 0
    for spatial, context, success, operational, normalized_g in loader:
        spatial = spatial.to(device)
        context = context.to(device)
        success = success.to(device)
        operational = operational.to(device)
        normalized_g = normalized_g.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            prediction = model(spatial, context)
            success_loss = functional.binary_cross_entropy_with_logits(
                prediction["success_logits"], success
            )
            cost_loss = functional.smooth_l1_loss(
                prediction["operational_cost_scaled"], operational
            )
            g_loss = functional.smooth_l1_loss(
                prediction["normalized_g_scaled"], normalized_g
            )
            g_mean = stats["normalized_g"]["mean"]
            g_scale = stats["normalized_g"]["scale"]
            predicted_g = prediction["normalized_g_scaled"] * g_scale + g_mean
            target_g = normalized_g * g_scale + g_mean
            target_probability = functional.softmax(
                target_g / config.ranking_temperature, dim=1
            )
            predicted_log_probability = functional.log_softmax(
                predicted_g / config.ranking_temperature, dim=1
            )
            ranking_loss = -(
                target_probability * predicted_log_probability
            ).sum(dim=1).mean()
            loss = (
                config.success_weight * success_loss
                + config.cost_weight * cost_loss
                + config.normalized_g_weight * g_loss
                + config.ranking_weight * ranking_loss
            )
            if training:
                loss.backward()
                optimizer.step()
        count = spatial.shape[0]
        samples += count
        totals += count * np.asarray(
            [loss.item(), success_loss.item(), cost_loss.item(), g_loss.item(), ranking_loss.item()]
        )
    names = ("loss", "success_loss", "cost_loss", "normalized_g_loss", "ranking_loss")
    return {name: float(value / samples) for name, value in zip(names, totals, strict=True)}


def _predict(
    model: Any,
    arrays: CounterfactualArrays,
    *,
    context_mean: np.ndarray,
    context_scale: np.ndarray,
    stats: dict[str, dict[str, float]],
    failure_penalty: float,
    device: Any,
    batch_size: int,
) -> dict[str, np.ndarray]:
    loader = DataLoader(
        _TaskUtilityDataset(
            arrays,
            context_mean=context_mean,
            context_scale=context_scale,
            stats=stats,
            failure_penalty=failure_penalty,
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    success_values = []
    cost_values = []
    g_values = []
    model.eval()
    with torch.no_grad():
        for spatial, context, *_ in loader:
            prediction = model(spatial.to(device), context.to(device))
            success_values.append(torch.sigmoid(prediction["success_logits"]).cpu().numpy())
            cost_values.append(prediction["operational_cost_scaled"].cpu().numpy())
            g_values.append(prediction["normalized_g_scaled"].cpu().numpy())
    cost_scaled = np.concatenate(cost_values)
    g_scaled = np.concatenate(g_values)
    cost_log = (
        cost_scaled * stats["operational_cost_log"]["scale"]
        + stats["operational_cost_log"]["mean"]
    )
    return {
        "success_probability": np.concatenate(success_values),
        "operational_cost": np.maximum(0.0, np.expm1(cost_log)),
        "normalized_g": (
            g_scaled * stats["normalized_g"]["scale"] + stats["normalized_g"]["mean"]
        ),
    }


def _evaluate(
    arrays: CounterfactualArrays,
    prediction: dict[str, np.ndarray],
    failure_penalty: float,
) -> dict[str, float]:
    success, operational, normalized_g = _targets(arrays, failure_penalty)
    probability = np.clip(prediction["success_probability"], 0.0, 1.0)
    return {
        "samples": len(arrays.context),
        "success_brier": float(np.mean((probability - success) ** 2)),
        "success_accuracy": float(np.mean((probability >= 0.5) == (success >= 0.5))),
        "operational_cost_mae": float(
            np.mean(np.abs(prediction["operational_cost"] - operational))
        ),
        "normalized_g_mae": float(np.mean(np.abs(prediction["normalized_g"] - normalized_g))),
    }


def train_task_utility_model(
    split: DatasetSplit,
    *,
    output_dir: Path,
    config: TaskUtilityTrainingConfig | None = None,
    timing_profile: Path | None = None,
) -> dict[str, Any]:
    """Train and checkpoint the minimal three-head task-utility model."""

    if torch is None:
        raise ImportError("task-utility training requires PyTorch")
    config = config or TaskUtilityTrainingConfig()
    _targets(split.train, config.failure_penalty)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    device = torch.device(
        "cuda" if config.device == "auto" and torch.cuda.is_available() else (
            "cpu" if config.device == "auto" else config.device
        )
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    context_mean = split.train.context.mean(axis=0).astype(np.float32)
    context_scale = split.train.context.std(axis=0).astype(np.float32)
    context_scale[context_scale < 1e-6] = 1.0
    stats = _stats(split.train, config.failure_penalty)

    def loader(arrays: CounterfactualArrays, shuffle: bool) -> Any:
        return DataLoader(
            _TaskUtilityDataset(
                arrays,
                context_mean=context_mean,
                context_scale=context_scale,
                stats=stats,
                failure_penalty=config.failure_penalty,
            ),
            batch_size=config.batch_size,
            shuffle=shuffle,
            generator=torch.Generator().manual_seed(config.seed) if shuffle else None,
        )

    train_loader = loader(split.train, True)
    validation_loader = loader(split.validation, False)
    model = TaskUtilityNetwork(
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
            model, train_loader, device=device, stats=stats, config=config, optimizer=optimizer
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
            failure_penalty=config.failure_penalty,
            device=device,
            batch_size=config.batch_size,
        )
        for key, values in predicted.items():
            predictions[f"{name}_{key}"] = values
        predictions[f"{name}_context_ids"] = arrays.context_ids
        evaluations[name] = _evaluate(arrays, predicted, config.failure_penalty)

    compute_profile = None
    switching_profile = None
    if timing_profile is not None:
        compute_profile, switching_profile = load_timing_profile(timing_profile)
    checkpoint = {
        "schema_version": 7,
        "model_type": "task_utility",
        "prediction_targets": ["success", "operational_remaining_cost", "normalized_g"],
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
        "schema_version": 7,
        "model_type": "task_utility",
        "failure_penalty_removed_from_cost_target": config.failure_penalty,
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
