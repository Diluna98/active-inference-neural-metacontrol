"""Train a four-head next-step Active-Inference value model."""

from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .allocations import ALLOCATIONS, DEPTHS, RESOLUTIONS
from .datasets import CounterfactualArrays, DatasetSplit
from .features import (
    DECOMPOSED_FEATURE_SCHEMA,
    FOUR_TERM_ABLATION_FEATURE_SCHEMA,
    FOUR_TERM_NO_ALLOCATION_FEATURE_SCHEMA,
    FOUR_TERM_PREDICTED_OBSERVATION_FEATURE_SCHEMA,
    project_feature_schema,
)
from .models import ImmediateImprovementValueNetwork, ResolutionSharedFourTermValueNetwork

try:
    import torch
    from torch.nn import functional
    from torch.utils.data import DataLoader, Dataset
except ImportError:  # pragma: no cover
    torch = None
    functional = None
    DataLoader = None
    Dataset = object

TARGET_NAMES = (
    "state_accuracy",
    "state_complexity",
    "preference_per_depth",
    "epistemic_value_per_depth",
)
IMMEDIATE_IMPROVEMENT_TARGET_NAMES = (
    "state_accuracy",
    "state_complexity",
    "immediate_preference",
    "preference_improvement",
    "immediate_epistemic",
    "epistemic_improvement",
)

CHECKPOINT_SCHEMA_VERSION = 14
FUTURE_ONLY_CHECKPOINT_SCHEMA_VERSION = 15
IMMEDIATE_IMPROVEMENT_CHECKPOINT_SCHEMA_VERSION = 16


@dataclass(frozen=True)
class FourTermTrainingConfig:
    epochs: int = 150
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 20
    ranking_temperature: float = 0.1
    ranking_weight: float = 1.0
    detach_state_terms_in_ranking: bool = True
    seed: int = 0
    device: str = "auto"
    policy_aggregation: str = "selected"
    target_horizon: str = "all_steps_mean"
    feature_schema: str = DECOMPOSED_FEATURE_SCHEMA

    def __post_init__(self) -> None:
        if self.ranking_temperature <= 0 or self.ranking_weight < 0:
            raise ValueError("ranking temperature must be positive and weight nonnegative")
        if self.policy_aggregation not in {"selected", "uniform", "posterior"}:
            raise ValueError(
                "policy_aggregation must be 'selected', 'uniform', or 'posterior'"
            )
        if self.target_horizon not in {
            "all_steps_mean",
            "future_only_mean",
            "immediate_improvement",
        }:
            raise ValueError("unsupported target_horizon")
        if self.target_horizon == "future_only_mean" and self.policy_aggregation == "uniform":
            raise ValueError("future-only labels support selected or posterior aggregation")
        if self.target_horizon == "immediate_improvement" and self.policy_aggregation != "posterior":
            raise ValueError("immediate-improvement labels require posterior aggregation")
        if self.feature_schema not in {
            DECOMPOSED_FEATURE_SCHEMA,
            FOUR_TERM_ABLATION_FEATURE_SCHEMA,
            FOUR_TERM_NO_ALLOCATION_FEATURE_SCHEMA,
            FOUR_TERM_PREDICTED_OBSERVATION_FEATURE_SCHEMA,
        }:
            raise ValueError("unsupported four-term feature schema")


def _targets(
    arrays: CounterfactualArrays,
    policy_aggregation: str = "selected",
    target_horizon: str = "all_steps_mean",
) -> dict[str, np.ndarray]:
    if arrays.state_accuracy is None or arrays.state_complexity is None:
        raise ValueError("dataset does not contain state accuracy and complexity labels")

    def share_across_depth(values: np.ndarray) -> np.ndarray:
        shaped = np.asarray(values, dtype=np.float32).reshape(
            -1, len(RESOLUTIONS), len(DEPTHS)
        )
        return np.repeat(shaped.mean(axis=2, keepdims=True), len(DEPTHS), axis=2).reshape(
            -1, len(ALLOCATIONS)
        )

    def center_candidates(values: np.ndarray) -> np.ndarray:
        """Remove context-wide offsets while preserving allocation ordering."""

        return values - values.mean(axis=1, keepdims=True)

    if target_horizon == "immediate_improvement":
        names = (
            "posterior_immediate_preference",
            "posterior_immediate_epistemic",
            "posterior_immediate_information_gain",
            "posterior_improvement_preference",
            "posterior_improvement_epistemic",
            "posterior_improvement_information_gain",
        )
        values = {name: getattr(arrays, name) for name in names}
        if any(value is None for value in values.values()):
            raise ValueError("dataset does not contain posterior immediate/improvement labels")
        return {
            "state_accuracy": share_across_depth(arrays.state_accuracy),
            "state_complexity": share_across_depth(arrays.state_complexity),
            "immediate_preference": center_candidates(
                np.asarray(values["posterior_immediate_preference"], dtype=np.float32)
            ),
            # Keep improvement on its natural zero baseline: T=1 has no
            # continuation and must remain exactly zero. Centering this term
            # would manufacture a nonzero "improvement" for shallow planning.
            "preference_improvement": np.asarray(
                values["posterior_improvement_preference"], dtype=np.float32
            ),
            "immediate_epistemic": center_candidates(
                np.asarray(values["posterior_immediate_epistemic"], dtype=np.float32)
                - np.asarray(
                    values["posterior_immediate_information_gain"], dtype=np.float32
                )
            ),
            "epistemic_improvement": (
                np.asarray(values["posterior_improvement_epistemic"], dtype=np.float32)
                - np.asarray(
                    values["posterior_improvement_information_gain"], dtype=np.float32
                )
            ),
        }
    depths = np.asarray([item.depth for item in ALLOCATIONS], dtype=np.float32)[None, :]
    if target_horizon == "future_only_mean":
        prefix = "selected" if policy_aggregation == "selected" else "posterior"
        risk = getattr(arrays, f"{prefix}_future_preference")
        ambiguity = getattr(arrays, f"{prefix}_future_epistemic")
        information_gain = getattr(arrays, f"{prefix}_future_information_gain")
        if risk is None or ambiguity is None or information_gain is None:
            raise ValueError(f"dataset does not contain {prefix} future-only labels")
        divide_by_depth = False
    elif policy_aggregation == "selected":
        risk = arrays.risk
        ambiguity = arrays.ambiguity
        information_gain = arrays.information_gain
        divide_by_depth = True
    elif policy_aggregation == "uniform":
        if (
            arrays.uniform_risk is None
            or arrays.uniform_ambiguity is None
            or arrays.uniform_information_gain is None
        ):
            raise ValueError("dataset does not contain uniform-policy labels")
        risk = arrays.uniform_risk
        ambiguity = arrays.uniform_ambiguity
        information_gain = arrays.uniform_information_gain
        divide_by_depth = True
    elif policy_aggregation == "posterior":
        if (
            arrays.posterior_risk is None
            or arrays.posterior_ambiguity is None
            or arrays.posterior_information_gain is None
        ):
            raise ValueError("dataset does not contain posterior-policy labels")
        risk = arrays.posterior_risk
        ambiguity = arrays.posterior_ambiguity
        information_gain = arrays.posterior_information_gain
        divide_by_depth = True
    else:
        raise ValueError("unsupported policy aggregation")
    # Filtered state inference at t+1 is determined by resolution, not by the
    # prospective policy depth. Average any numerical noise between duplicate
    # depth labels, then repeat the shared target in allocation order.
    denominator = depths if divide_by_depth else 1.0
    preference_per_depth = np.asarray(risk, dtype=np.float32) / denominator
    epistemic_value_per_depth = (
        np.asarray(ambiguity, dtype=np.float32)
        - np.asarray(information_gain, dtype=np.float32)
    ) / denominator

    return {
        "state_accuracy": share_across_depth(arrays.state_accuracy),
        "state_complexity": share_across_depth(arrays.state_complexity),
        "preference_per_depth": center_candidates(preference_per_depth),
        "epistemic_value_per_depth": center_candidates(epistemic_value_per_depth),
    }


def _stats(
    arrays: CounterfactualArrays,
    policy_aggregation: str,
    target_horizon: str,
) -> dict[str, dict[str, float]]:
    return {
        name: {"mean": float(values.mean()), "scale": max(float(values.std()), 1e-6)}
        for name, values in _targets(arrays, policy_aggregation, target_horizon).items()
    }


class _FourTermDataset(Dataset):
    def __init__(
        self,
        arrays,
        context_mean,
        context_scale,
        stats,
        policy_aggregation,
        target_horizon,
        feature_schema,
    ):
        spatial, context = project_feature_schema(
            arrays.spatial, arrays.context, feature_schema
        )
        self.spatial = torch.from_numpy(spatial.astype(np.float32))
        self.context = torch.from_numpy(
            ((context - context_mean) / context_scale).astype(np.float32)
        )
        self.targets = tuple(
            torch.from_numpy(
                ((values - stats[name]["mean"]) / stats[name]["scale"]).astype(np.float32)
            )
            for name, values in _targets(
                arrays, policy_aggregation, target_horizon
            ).items()
        )

    def __len__(self):
        return len(self.spatial)

    def __getitem__(self, index):
        return (self.spatial[index], self.context[index]) + tuple(
            values[index] for values in self.targets
        )


def _unscale(value, stats, name):
    return value * stats[name]["scale"] + stats[name]["mean"]


def _target_names(target_horizon: str) -> tuple[str, ...]:
    return (
        IMMEDIATE_IMPROVEMENT_TARGET_NAMES
        if target_horizon == "immediate_improvement"
        else TARGET_NAMES
    )


def _output_keys(target_horizon: str) -> tuple[str, ...]:
    if target_horizon == "immediate_improvement":
        return (
            "state_accuracy_scaled",
            "state_complexity_scaled",
            "immediate_preference_scaled",
            "preference_improvement_scaled",
            "immediate_epistemic_scaled",
            "epistemic_improvement_scaled",
        )
    return (
        "state_accuracy_scaled",
        "state_complexity_scaled",
        "preference_per_depth_scaled",
        "epistemic_value_per_depth_scaled",
    )


def _utility(values: dict[str, Any]) -> Any:
    if "immediate_preference" in values:
        return (
            values["state_accuracy"]
            - values["state_complexity"]
            + values["immediate_preference"]
            + values["preference_improvement"]
            + values["immediate_epistemic"]
            + values["epistemic_improvement"]
        )
    return (
        values["state_accuracy"]
        - values["state_complexity"]
        + values["preference_per_depth"]
        + values["epistemic_value_per_depth"]
    )


def _ranking_utility(values: dict[str, Any], *, detach_state_terms: bool) -> Any:
    accuracy = values["state_accuracy"]
    complexity = values["state_complexity"]
    if detach_state_terms:
        accuracy = accuracy.detach()
        complexity = complexity.detach()
    if "immediate_preference" in values:
        return (
            accuracy
            - complexity
            + values["immediate_preference"]
            + values["preference_improvement"]
            + values["immediate_epistemic"]
            + values["epistemic_improvement"]
        )
    return accuracy - complexity + values["preference_per_depth"] + values[
        "epistemic_value_per_depth"
    ]


def _run_epoch(model, loader, *, device, stats, config, optimizer):
    training = optimizer is not None
    model.train(training)
    target_names = _target_names(config.target_horizon)
    output_keys = _output_keys(config.target_horizon)
    totals = np.zeros(len(target_names) + 2, dtype=float)
    samples = 0
    for batch in loader:
        spatial, context, *target_values = [value.to(device) for value in batch]
        targets = dict(zip(target_names, target_values, strict=True))
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            output = model(spatial, context)
            component_losses = [
                functional.smooth_l1_loss(output[key], targets[name])
                for name, key in zip(target_names, output_keys, strict=True)
            ]
            predicted_raw = {
                name: _unscale(output[key], stats, name)
                for name, key in zip(target_names, output_keys, strict=True)
            }
            target_raw = {
                name: _unscale(targets[name], stats, name) for name in target_names
            }
            predicted_utility = _ranking_utility(
                predicted_raw,
                detach_state_terms=config.detach_state_terms_in_ranking,
            )
            target_utility = _utility(target_raw)
            target_probability = functional.softmax(
                target_utility / config.ranking_temperature, dim=1
            )
            ranking_loss = -(
                target_probability
                * functional.log_softmax(
                    predicted_utility / config.ranking_temperature, dim=1
                )
            ).sum(dim=1).mean()
            loss = sum(component_losses) + config.ranking_weight * ranking_loss
            if training:
                loss.backward()
                optimizer.step()
        count = spatial.shape[0]
        samples += count
        totals += count * np.asarray(
            [loss.item(), *(item.item() for item in component_losses), ranking_loss.item()]
        )
    names = ("loss", *[f"{name}_loss" for name in target_names], "ranking_loss")
    return {name: float(value / samples) for name, value in zip(names, totals, strict=True)}


def _predict(
    model,
    arrays,
    *,
    context_mean,
    context_scale,
    stats,
    policy_aggregation,
    target_horizon,
    feature_schema,
    device,
    batch_size,
):
    loader = DataLoader(
        _FourTermDataset(
            arrays,
            context_mean,
            context_scale,
            stats,
            policy_aggregation,
            target_horizon,
            feature_schema,
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    target_names = _target_names(target_horizon)
    output_keys = _output_keys(target_horizon)
    collected = {name: [] for name in target_names}
    model.eval()
    with torch.no_grad():
        for spatial, context, *_ in loader:
            output = model(spatial.to(device), context.to(device))
            for name, key in zip(target_names, output_keys, strict=True):
                collected[name].append(output[key].cpu().numpy())
    result = {
        name: _unscale(np.concatenate(values), stats, name)
        for name, values in collected.items()
    }
    result["utility"] = _utility(result)
    return result


def _evaluate(arrays, prediction, policy_aggregation, target_horizon):
    target = _targets(arrays, policy_aggregation, target_horizon)
    target_names = _target_names(target_horizon)
    target_utility = _utility(target)
    return {
        "samples": len(arrays.context),
        **{
            f"{name}_mae": float(np.mean(np.abs(prediction[name] - target[name])))
            for name in target_names
        },
        "utility_mae": float(np.mean(np.abs(prediction["utility"] - target_utility))),
        "top1_allocation_accuracy": float(
            np.mean(np.argmax(prediction["utility"], axis=1) == np.argmax(target_utility, axis=1))
        ),
    }


def train_four_term_model(
    split: DatasetSplit,
    *,
    output_dir: Path,
    config: FourTermTrainingConfig | None = None,
) -> dict[str, Any]:
    if torch is None:
        raise ImportError("four-term training requires PyTorch")
    config = config or FourTermTrainingConfig()
    checkpoint_schema_version = {
        "all_steps_mean": CHECKPOINT_SCHEMA_VERSION,
        "future_only_mean": FUTURE_ONLY_CHECKPOINT_SCHEMA_VERSION,
        "immediate_improvement": IMMEDIATE_IMPROVEMENT_CHECKPOINT_SCHEMA_VERSION,
    }[config.target_horizon]
    objective_convention = {
        "all_steps_mean": "accuracy-complexity+relative-preference+relative-epistemic-v2",
        "future_only_mean": (
            "accuracy-complexity+relative-future-only-preference+"
            "relative-future-only-epistemic-v1"
        ),
        "immediate_improvement": (
            "accuracy-complexity+immediate-and-future-improvement-preference-epistemic-v1"
        ),
    }[config.target_horizon]
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device(
        "cuda"
        if config.device == "auto" and torch.cuda.is_available()
        else ("cpu" if config.device == "auto" else config.device)
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_spatial, train_context = project_feature_schema(
        split.train.spatial, split.train.context, config.feature_schema
    )
    context_mean = train_context.mean(axis=0).astype(np.float32)
    context_scale = train_context.std(axis=0).astype(np.float32)
    context_scale[context_scale < 1e-6] = 1.0
    stats = _stats(split.train, config.policy_aggregation, config.target_horizon)

    def make_loader(arrays, shuffle):
        return DataLoader(
            _FourTermDataset(
                arrays,
                context_mean,
                context_scale,
                stats,
                config.policy_aggregation,
                config.target_horizon,
                config.feature_schema,
            ),
            batch_size=config.batch_size,
            shuffle=shuffle,
            generator=torch.Generator().manual_seed(config.seed) if shuffle else None,
        )

    train_loader = make_loader(split.train, True)
    validation_loader = make_loader(split.validation, False)
    model_class = (
        ImmediateImprovementValueNetwork
        if config.target_horizon == "immediate_improvement"
        else ResolutionSharedFourTermValueNetwork
    )
    model = model_class(
        spatial_channels=train_spatial.shape[1],
        context_features=train_context.shape[1],
        allocations=len(ALLOCATIONS),
        resolutions=len(RESOLUTIONS),
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
            policy_aggregation=config.policy_aggregation,
            target_horizon=config.target_horizon,
            feature_schema=config.feature_schema,
            device=device,
            batch_size=config.batch_size,
        )
        for key, values in predicted.items():
            predictions[f"{name}_{key}"] = values
        evaluations[name] = _evaluate(
            arrays, predicted, config.policy_aggregation, config.target_horizon
        )

    checkpoint = {
        "schema_version": checkpoint_schema_version,
        "model_type": "state_detached_candidate_relative_four_term_value",
        "objective_convention": objective_convention,
        "feature_schema": config.feature_schema,
        "prediction_targets": list(_target_names(config.target_horizon)),
        "policy_aggregation": config.policy_aggregation,
        "target_horizon": config.target_horizon,
        "model_state_dict": best_state,
        "spatial_channels": train_spatial.shape[1],
        "context_features": train_context.shape[1],
        "allocations": [(item.resolution, item.depth) for item in ALLOCATIONS],
        "context_mean": context_mean,
        "context_scale": context_scale,
        "target_stats": stats,
        "compute_profile_ms": np.median(split.train.compute_ms, axis=0),
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
        "schema_version": checkpoint_schema_version,
        "model_type": "state_detached_candidate_relative_four_term_value",
        "objective_convention": objective_convention,
        "policy_aggregation": config.policy_aggregation,
        "target_horizon": config.target_horizon,
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
