"""Training for reduced-input, decomposed task-utility metacontrol."""

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
from .features import DECOMPOSED_FEATURE_SCHEMA, project_decomposed_features
from .models import DecomposedTaskUtilityNetwork
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
class DecomposedTrainingConfig:
    epochs: int = 200
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 25
    ranking_temperature: float = 0.1
    success_weight: float = 1.0
    cost_weight: float = 1.0
    preference_weight: float = 1.0
    epistemic_weight: float = 1.0
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
        weights = (
            self.success_weight,
            self.cost_weight,
            self.preference_weight,
            self.epistemic_weight,
            self.ranking_weight,
        )
        if min(weights) < 0:
            raise ValueError("loss weights must be nonnegative")


def _targets(arrays: CounterfactualArrays, failure_penalty: float) -> dict[str, np.ndarray]:
    if arrays.success is None or arrays.task_cost is None:
        raise ValueError("decomposed training requires success and task_cost labels")
    success = np.asarray(arrays.success, dtype=np.float32)
    operational = np.asarray(arrays.task_cost, dtype=np.float32) - failure_penalty * (1.0 - success)
    if np.any(operational < -1e-5):
        raise ValueError("failure penalty exceeds recorded task cost")
    depths = np.asarray([item.depth for item in ALLOCATIONS], dtype=np.float32)[None, :]
    return {
        "success": success,
        "operational_cost": np.maximum(operational, 0.0),
        "preference_per_depth": np.asarray(arrays.risk, dtype=np.float32) / depths,
        "epistemic_per_depth": (
            np.asarray(arrays.ambiguity, dtype=np.float32)
            - np.asarray(arrays.information_gain, dtype=np.float32)
        )
        / depths,
    }


def _stats(arrays: CounterfactualArrays, failure_penalty: float) -> dict[str, dict[str, float]]:
    targets = _targets(arrays, failure_penalty)
    values = {
        "operational_cost_log": np.log1p(targets["operational_cost"]),
        "preference_per_depth": targets["preference_per_depth"],
        "epistemic_per_depth": targets["epistemic_per_depth"],
    }
    return {
        name: {"mean": float(value.mean()), "scale": max(float(value.std()), 1e-6)}
        for name, value in values.items()
    }


class _DecomposedDataset(Dataset):
    def __init__(
        self,
        arrays: CounterfactualArrays,
        *,
        context_mean: np.ndarray,
        context_scale: np.ndarray,
        stats: dict[str, dict[str, float]],
        failure_penalty: float,
    ) -> None:
        spatial, context = project_decomposed_features(arrays.spatial, arrays.context)
        self.spatial = torch.from_numpy(spatial.astype(np.float32))
        self.context = torch.from_numpy(
            ((context - context_mean) / context_scale).astype(np.float32)
        )
        targets = _targets(arrays, failure_penalty)
        self.success = torch.from_numpy(targets["success"])
        operational_log = np.log1p(targets["operational_cost"])
        self.operational = torch.from_numpy(
            (
                (operational_log - stats["operational_cost_log"]["mean"])
                / stats["operational_cost_log"]["scale"]
            ).astype(np.float32)
        )
        self.preference = torch.from_numpy(
            (
                (targets["preference_per_depth"] - stats["preference_per_depth"]["mean"])
                / stats["preference_per_depth"]["scale"]
            ).astype(np.float32)
        )
        self.epistemic = torch.from_numpy(
            (
                (targets["epistemic_per_depth"] - stats["epistemic_per_depth"]["mean"])
                / stats["epistemic_per_depth"]["scale"]
            ).astype(np.float32)
        )

    def __len__(self) -> int:
        return len(self.spatial)

    def __getitem__(self, index: int) -> tuple[Any, ...]:
        return (
            self.spatial[index],
            self.context[index],
            self.success[index],
            self.operational[index],
            self.preference[index],
            self.epistemic[index],
        )


def _unscale(value: Any, stats: dict[str, dict[str, float]], name: str) -> Any:
    return value * stats[name]["scale"] + stats[name]["mean"]


def _run_epoch(
    model: Any,
    loader: Any,
    *,
    device: Any,
    stats: dict[str, dict[str, float]],
    config: DecomposedTrainingConfig,
    optimizer: Any | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    names = ("loss", "success", "cost", "preference", "epistemic", "ranking")
    totals = {name: 0.0 for name in names}
    samples = 0
    for spatial, context, success, operational, preference, epistemic in loader:
        values = [spatial, context, success, operational, preference, epistemic]
        spatial, context, success, operational, preference, epistemic = (
            value.to(device) for value in values
        )
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            prediction = model(spatial, context)
            losses = {
                "success": functional.binary_cross_entropy_with_logits(
                    prediction["success_logits"], success
                ),
                "cost": functional.smooth_l1_loss(
                    prediction["operational_cost_scaled"], operational
                ),
                "preference": functional.smooth_l1_loss(
                    prediction["preference_per_depth_scaled"], preference
                ),
                "epistemic": functional.smooth_l1_loss(
                    prediction["epistemic_per_depth_scaled"], epistemic
                ),
            }
            predicted_g = _unscale(
                prediction["preference_per_depth_scaled"], stats, "preference_per_depth"
            ) + _unscale(prediction["epistemic_per_depth_scaled"], stats, "epistemic_per_depth")
            target_g = _unscale(preference, stats, "preference_per_depth") + _unscale(
                epistemic, stats, "epistemic_per_depth"
            )
            target_probability = functional.softmax(target_g / config.ranking_temperature, dim=1)
            predicted_log_probability = functional.log_softmax(
                predicted_g / config.ranking_temperature, dim=1
            )
            losses["ranking"] = -(target_probability * predicted_log_probability).sum(dim=1).mean()
            losses["loss"] = (
                config.success_weight * losses["success"]
                + config.cost_weight * losses["cost"]
                + config.preference_weight * losses["preference"]
                + config.epistemic_weight * losses["epistemic"]
                + config.ranking_weight * losses["ranking"]
            )
            if training:
                losses["loss"].backward()
                optimizer.step()
        count = spatial.shape[0]
        samples += count
        for name in names:
            totals[name] += count * float(losses[name].item())
    return {name: value / samples for name, value in totals.items()}


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
        _DecomposedDataset(
            arrays,
            context_mean=context_mean,
            context_scale=context_scale,
            stats=stats,
            failure_penalty=failure_penalty,
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    collected = {name: [] for name in ("success", "cost", "preference", "epistemic")}
    model.eval()
    with torch.no_grad():
        for spatial, context, *_ in loader:
            prediction = model(spatial.to(device), context.to(device))
            collected["success"].append(torch.sigmoid(prediction["success_logits"]).cpu().numpy())
            collected["cost"].append(prediction["operational_cost_scaled"].cpu().numpy())
            collected["preference"].append(prediction["preference_per_depth_scaled"].cpu().numpy())
            collected["epistemic"].append(prediction["epistemic_per_depth_scaled"].cpu().numpy())
    success = np.concatenate(collected["success"])
    cost_log = _unscale(np.concatenate(collected["cost"]), stats, "operational_cost_log")
    preference = _unscale(np.concatenate(collected["preference"]), stats, "preference_per_depth")
    epistemic = _unscale(np.concatenate(collected["epistemic"]), stats, "epistemic_per_depth")
    return {
        "success_probability": success,
        "operational_cost": np.maximum(0.0, np.expm1(cost_log)),
        "preference_per_depth": preference,
        "epistemic_per_depth": epistemic,
        "normalized_g": preference + epistemic,
    }


def _evaluate(
    arrays: CounterfactualArrays,
    prediction: dict[str, np.ndarray],
    failure_penalty: float,
) -> dict[str, float]:
    target = _targets(arrays, failure_penalty)
    probability = np.clip(prediction["success_probability"], 0.0, 1.0)
    normalized_g = target["preference_per_depth"] + target["epistemic_per_depth"]
    return {
        "samples": len(arrays.context),
        "success_brier": float(np.mean((probability - target["success"]) ** 2)),
        "success_accuracy": float(np.mean((probability >= 0.5) == (target["success"] >= 0.5))),
        "operational_cost_mae": float(
            np.mean(np.abs(prediction["operational_cost"] - target["operational_cost"]))
        ),
        "preference_per_depth_mae": float(
            np.mean(np.abs(prediction["preference_per_depth"] - target["preference_per_depth"]))
        ),
        "epistemic_per_depth_mae": float(
            np.mean(np.abs(prediction["epistemic_per_depth"] - target["epistemic_per_depth"]))
        ),
        "reconstructed_normalized_g_mae": float(
            np.mean(np.abs(prediction["normalized_g"] - normalized_g))
        ),
    }


def train_decomposed_model(
    split: DatasetSplit,
    *,
    output_dir: Path,
    config: DecomposedTrainingConfig | None = None,
    timing_profile: Path | None = None,
) -> dict[str, Any]:
    """Train the four-head model on the reduced feature schema."""

    if torch is None:
        raise ImportError("decomposed training requires PyTorch")
    config = config or DecomposedTrainingConfig()
    _targets(split.train, config.failure_penalty)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    device = torch.device(
        "cuda"
        if config.device == "auto" and torch.cuda.is_available()
        else ("cpu" if config.device == "auto" else config.device)
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_spatial, train_context = project_decomposed_features(
        split.train.spatial, split.train.context
    )
    context_mean = train_context.mean(axis=0).astype(np.float32)
    context_scale = train_context.std(axis=0).astype(np.float32)
    context_scale[context_scale < 1e-6] = 1.0
    stats = _stats(split.train, config.failure_penalty)

    def loader(arrays: CounterfactualArrays, shuffle: bool) -> Any:
        return DataLoader(
            _DecomposedDataset(
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
    model = DecomposedTaskUtilityNetwork(
        spatial_channels=train_spatial.shape[1],
        context_features=train_context.shape[1],
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
        "schema_version": 8,
        "model_type": "decomposed_task_utility",
        "feature_schema": DECOMPOSED_FEATURE_SCHEMA,
        "removed_inputs": [
            "visibility",
            "posterior_entropy",
            "policy_probability_margin",
        ],
        "prediction_targets": [
            "success",
            "operational_remaining_cost",
            "preference_per_depth",
            "epistemic_per_depth",
        ],
        "model_state_dict": best_state,
        "spatial_channels": train_spatial.shape[1],
        "context_features": train_context.shape[1],
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
        "schema_version": 8,
        "model_type": "decomposed_task_utility",
        "feature_schema": DECOMPOSED_FEATURE_SCHEMA,
        "removed_inputs": checkpoint["removed_inputs"],
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
