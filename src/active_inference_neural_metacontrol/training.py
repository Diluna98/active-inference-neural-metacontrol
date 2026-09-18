"""PyTorch training and allocation-level evaluation for neural metacontrol."""

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
from .models import TaskPerformanceNetwork

try:
    import torch
    from torch.nn import functional
    from torch.utils.data import DataLoader, Dataset
except ImportError:  # pragma: no cover - optional dependency guard
    torch = None
    functional = None
    DataLoader = None
    Dataset = object


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 200
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 25
    success_weight: float = 1.0
    task_cost_weight: float = 1.0
    success_threshold: float = 0.5
    compute_budget_ms: float | None = None
    seed: int = 0
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1 or self.patience < 1:
            raise ValueError("epochs, batch_size, and patience must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay nonnegative")
        if self.success_weight < 0 or self.task_cost_weight < 0:
            raise ValueError("loss weights must be nonnegative")
        if self.success_weight + self.task_cost_weight == 0:
            raise ValueError("at least one loss weight must be positive")
        if not 0 <= self.success_threshold <= 1:
            raise ValueError("success_threshold must lie in [0, 1]")
        if self.compute_budget_ms is not None and self.compute_budget_ms <= 0:
            raise ValueError("compute_budget_ms must be positive")


class _TorchDataset(Dataset):
    def __init__(
        self,
        arrays: CounterfactualArrays,
        context_mean: np.ndarray,
        context_scale: np.ndarray,
    ) -> None:
        self.spatial = torch.from_numpy(arrays.spatial)
        normalized = (arrays.context - context_mean) / context_scale
        self.context = torch.from_numpy(normalized.astype(np.float32))
        self.success = torch.from_numpy(arrays.success)
        self.task_cost = torch.from_numpy(arrays.task_cost)

    def __len__(self) -> int:
        return len(self.spatial)

    def __getitem__(self, index: int) -> tuple[Any, ...]:
        return (
            self.spatial[index],
            self.context[index],
            self.success[index],
            self.task_cost[index],
        )


def _require_torch() -> None:
    if torch is None:
        raise ImportError(
            "training requires PyTorch; install active-inference-neural-metacontrol[neural]"
        )


def _resolve_device(requested: str) -> Any:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return device


def _loss_components(prediction: dict[str, Any], success: Any, task_cost: Any) -> tuple[Any, Any]:
    success_loss = functional.binary_cross_entropy_with_logits(
        prediction["success_logits"], success
    )
    task_loss = functional.smooth_l1_loss(
        torch.log1p(prediction["task_cost"]), torch.log1p(task_cost)
    )
    return success_loss, task_loss


def _run_epoch(
    model: Any,
    loader: Any,
    *,
    device: Any,
    config: TrainingConfig,
    optimizer: Any | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = np.zeros(3, dtype=float)
    samples = 0
    for spatial, context, success, task_cost in loader:
        spatial = spatial.to(device)
        context = context.to(device)
        success = success.to(device)
        task_cost = task_cost.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            prediction = model(spatial, context)
            success_loss, task_loss = _loss_components(prediction, success, task_cost)
            loss = config.success_weight * success_loss + config.task_cost_weight * task_loss
            if training:
                loss.backward()
                optimizer.step()
        count = spatial.shape[0]
        totals += count * np.asarray(
            [loss.item(), success_loss.item(), task_loss.item()], dtype=float
        )
        samples += count
    return {
        "loss": float(totals[0] / samples),
        "success_loss": float(totals[1] / samples),
        "task_cost_loss": float(totals[2] / samples),
    }


def _predict(
    model: Any,
    arrays: CounterfactualArrays,
    *,
    context_mean: np.ndarray,
    context_scale: np.ndarray,
    device: Any,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(
        _TorchDataset(arrays, context_mean, context_scale),
        batch_size=batch_size,
        shuffle=False,
    )
    success_values = []
    cost_values = []
    model.eval()
    with torch.no_grad():
        for spatial, context, _, _ in loader:
            prediction = model(spatial.to(device), context.to(device))
            success_values.append(torch.sigmoid(prediction["success_logits"]).cpu().numpy())
            cost_values.append(prediction["task_cost"].cpu().numpy())
    return np.concatenate(success_values), np.concatenate(cost_values)


def evaluate_predictions(
    arrays: CounterfactualArrays,
    *,
    success_probability: np.ndarray,
    task_cost_prediction: np.ndarray,
    compute_profile_ms: np.ndarray,
    success_threshold: float,
    compute_budget_ms: float | None,
) -> dict[str, Any]:
    """Measure predictive accuracy and realized allocation-selection quality."""

    success_probability = np.asarray(success_probability, dtype=float)
    task_cost_prediction = np.asarray(task_cost_prediction, dtype=float)
    expected_shape = arrays.success.shape
    if success_probability.shape != expected_shape or task_cost_prediction.shape != expected_shape:
        raise ValueError(f"predictions must have shape {expected_shape}")
    eligible_compute = np.ones(len(ALLOCATIONS), dtype=bool)
    if compute_budget_ms is not None:
        eligible_compute = np.asarray(compute_profile_ms) <= compute_budget_ms
    selected = []
    for row in range(len(arrays.success)):
        feasible = np.flatnonzero(
            (success_probability[row] >= success_threshold) & eligible_compute
        )
        if feasible.size:
            index = int(feasible[np.argmin(task_cost_prediction[row, feasible])])
        else:
            candidates = np.flatnonzero(eligible_compute)
            if not candidates.size:
                candidates = np.arange(len(ALLOCATIONS))
            best_success = success_probability[row, candidates].max()
            ties = candidates[np.isclose(success_probability[row, candidates], best_success)]
            index = int(ties[np.argmin(task_cost_prediction[row, ties])])
        selected.append(index)
    selected = np.asarray(selected, dtype=int)
    rows = np.arange(len(selected))
    realized_cost = arrays.task_cost[rows, selected]
    realized_success = arrays.success[rows, selected]
    oracle_cost = arrays.task_cost.min(axis=1)
    fixed = []
    for index, allocation in enumerate(ALLOCATIONS):
        fixed.append(
            {
                "resolution": allocation.resolution,
                "depth": allocation.depth,
                "mean_task_cost": float(arrays.task_cost[:, index].mean()),
                "success_rate": float(arrays.success[:, index].mean()),
                "median_compute_ms": float(compute_profile_ms[index]),
            }
        )
    return {
        "samples": len(selected),
        "success_brier": float(np.mean((success_probability - arrays.success) ** 2)),
        "success_accuracy": float(np.mean((success_probability >= 0.5) == (arrays.success >= 0.5))),
        "task_cost_mae": float(np.mean(np.abs(task_cost_prediction - arrays.task_cost))),
        "task_cost_rmse": float(np.sqrt(np.mean((task_cost_prediction - arrays.task_cost) ** 2))),
        "selected_mean_task_cost": float(realized_cost.mean()),
        "selected_success_rate": float(realized_success.mean()),
        "mean_regret_to_oracle": float(np.mean(realized_cost - oracle_cost)),
        "oracle_mean_task_cost": float(oracle_cost.mean()),
        "selection_counts": {
            f"g{allocation.resolution}_T{allocation.depth}": int(np.sum(selected == index))
            for index, allocation in enumerate(ALLOCATIONS)
        },
        "fixed_allocations": fixed,
    }


def train_task_model(
    split: DatasetSplit,
    *,
    output_dir: Path,
    config: TrainingConfig | None = None,
) -> dict[str, Any]:
    """Train, checkpoint, and evaluate the neural task-performance model."""

    _require_torch()
    config = config or TrainingConfig()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    device = _resolve_device(config.device)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    context_mean = split.train.context.mean(axis=0).astype(np.float32)
    context_scale = split.train.context.std(axis=0).astype(np.float32)
    context_scale[context_scale < 1e-6] = 1.0
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        _TorchDataset(split.train, context_mean, context_scale),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        _TorchDataset(split.validation, context_mean, context_scale),
        batch_size=config.batch_size,
        shuffle=False,
    )
    model = TaskPerformanceNetwork(
        spatial_channels=split.train.spatial.shape[1],
        context_features=split.train.context.shape[1],
        allocations=len(ALLOCATIONS),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    best_validation = np.inf
    best_state = None
    history = []
    stale_epochs = 0
    for epoch in range(1, config.epochs + 1):
        train_metrics = _run_epoch(
            model, train_loader, device=device, config=config, optimizer=optimizer
        )
        validation_metrics = _run_epoch(
            model, validation_loader, device=device, config=config, optimizer=None
        )
        row = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
        }
        history.append(row)
        if validation_metrics["loss"] < best_validation - 1e-7:
            best_validation = validation_metrics["loss"]
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)

    compute_profile = np.median(split.train.compute_ms, axis=0)
    evaluations = {}
    predictions = {}
    for name, arrays in (
        ("train", split.train),
        ("validation", split.validation),
        ("test", split.test),
    ):
        success_probability, task_cost_prediction = _predict(
            model,
            arrays,
            context_mean=context_mean,
            context_scale=context_scale,
            device=device,
            batch_size=config.batch_size,
        )
        predictions[f"{name}_success_probability"] = success_probability
        predictions[f"{name}_task_cost"] = task_cost_prediction
        predictions[f"{name}_context_ids"] = arrays.context_ids
        evaluations[name] = evaluate_predictions(
            arrays,
            success_probability=success_probability,
            task_cost_prediction=task_cost_prediction,
            compute_profile_ms=compute_profile,
            success_threshold=config.success_threshold,
            compute_budget_ms=config.compute_budget_ms,
        )

    checkpoint = {
        "model_state_dict": best_state,
        "spatial_channels": split.train.spatial.shape[1],
        "context_features": split.train.context.shape[1],
        "allocations": [(item.resolution, item.depth) for item in ALLOCATIONS],
        "context_mean": context_mean,
        "context_scale": context_scale,
        "compute_profile_ms": compute_profile,
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
