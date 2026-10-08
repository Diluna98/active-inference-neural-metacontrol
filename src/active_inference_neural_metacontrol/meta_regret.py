"""Reference-relative fitted-Q learning for MOS resource allocation."""

from __future__ import annotations

import copy
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .allocations import ALLOCATIONS, Allocation
from .meta_q import _split_instances
from .meta_q_data import MetaQTransitionDataset
from .meta_q_features import META_Q_CONTEXT_FEATURES, META_Q_FEATURE_SCHEMA

META_REGRET_CHECKPOINT_SCHEMA = 1
META_REGRET_MODEL_TYPE = "meta_regret"
META_REGRET_REFERENCE = Allocation(5, 2)
ALLOCATION_FEATURES = 7
META_REGRET_CONTEXT_FEATURES = META_Q_CONTEXT_FEATURES + ALLOCATION_FEATURES

try:
    import torch
    from torch import nn
    from torch.nn import functional
    from torch.utils.data import DataLoader, Dataset
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    functional = None
    DataLoader = None
    Dataset = object


def allocation_features(allocation: Allocation) -> np.ndarray:
    """One-hot source resolution and planning depth."""

    result = np.zeros(ALLOCATION_FEATURES, dtype=np.float32)
    result[(2, 5, 10, 20).index(allocation.resolution)] = 1.0
    result[4 + allocation.depth - 1] = 1.0
    return result


def augment_regret_context(context: np.ndarray, allocation: Allocation) -> np.ndarray:
    values = np.asarray(context, dtype=np.float32)
    if values.shape != (META_Q_CONTEXT_FEATURES,):
        raise ValueError("meta-Q context has an unexpected shape")
    return np.concatenate((values, allocation_features(allocation))).astype(np.float32)


if nn is not None:

    class MetaRegretNetwork(nn.Module):
        """Dueling Q network whose value head is the reference configuration Q."""

        def __init__(
            self,
            *,
            spatial_channels: int = 6,
            context_features: int = META_REGRET_CONTEXT_FEATURES,
            allocations: int = len(ALLOCATIONS),
            reference_index: int = ALLOCATIONS.index(META_REGRET_REFERENCE),
        ) -> None:
            super().__init__()
            self.reference_index = int(reference_index)
            self.spatial_encoder = nn.Sequential(
                nn.Conv2d(spatial_channels, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((5, 5)),
            )
            self.fusion = nn.Sequential(
                nn.Linear(32 * 5 * 5 + context_features, 96),
                nn.ReLU(),
                nn.Linear(96, 64),
                nn.ReLU(),
            )
            self.value_head = nn.Linear(64, 1)
            self.advantage_head = nn.Linear(64, allocations)

        def forward(self, spatial: Any, context: Any) -> Any:
            encoded = self.spatial_encoder(spatial).flatten(start_dim=1)
            hidden = self.fusion(torch.cat((encoded, context), dim=1))
            value = self.value_head(hidden)
            advantage = self.advantage_head(hidden)
            return value + advantage - advantage[:, self.reference_index : self.reference_index + 1]

else:

    class MetaRegretNetwork:  # pragma: no cover
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            raise ImportError("meta-regret learning requires PyTorch")


@dataclass(frozen=True)
class MetaRegretTrainingConfig:
    epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 3e-4
    discount: float = 0.98
    target_update_interval: int = 100
    absolute_weight: float = 0.25
    regret_weight: float = 1.0
    ranking_weight: float = 0.25
    ranking_margin: float = 0.01
    hidden_seed: int = 0
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    device: str = "cpu"
    torch_threads: int = 4

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1 or self.learning_rate <= 0:
            raise ValueError("invalid training hyperparameters")
        if not 0 <= self.discount <= 1:
            raise ValueError("discount must lie in [0, 1]")
        if min(self.absolute_weight, self.regret_weight, self.ranking_weight) < 0:
            raise ValueError("loss weights cannot be negative")
        if self.ranking_margin < 0:
            raise ValueError("ranking_margin cannot be negative")


class _RegretRows(Dataset):
    def __init__(
        self,
        dataset: MetaQTransitionDataset,
        indices: np.ndarray,
        context_mean: np.ndarray,
        context_scale: np.ndarray,
    ) -> None:
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=int)
        self.context_mean = context_mean
        self.context_scale = context_scale
        self.next_allocation_features = np.stack(
            [allocation_features(allocation) for allocation in dataset.allocations]
        )

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, item: int) -> tuple[Any, ...]:
        index = int(self.indices[item])
        source = Allocation(
            int(self.dataset.source_resolution[index]),
            int(self.dataset.source_depth[index]),
        )
        context = augment_regret_context(self.dataset.context[index], source)
        next_context = np.concatenate(
            (
                self.dataset.next_context[index],
                self.next_allocation_features,
            ),
            axis=1,
        ).astype(np.float32)
        context = (context - self.context_mean) / self.context_scale
        next_context = (next_context - self.context_mean[None, :]) / self.context_scale[None, :]
        return (
            torch.from_numpy(self.dataset.spatial[index].astype(np.float32)),
            torch.from_numpy(context.astype(np.float32)),
            torch.from_numpy(self.dataset.next_spatial[index].astype(np.float32)),
            torch.from_numpy(next_context.astype(np.float32)),
            torch.from_numpy(self.dataset.reward[index]),
            torch.from_numpy(self.dataset.terminal[index]),
        )


def _bellman_targets(
    target_model: Any,
    next_spatial: Any,
    next_context: Any,
    reward: Any,
    terminal: Any,
    discount: float,
) -> Any:
    batch, candidates = reward.shape
    flat_spatial = next_spatial.reshape(batch * candidates, *next_spatial.shape[2:])
    flat_context = next_context.reshape(batch * candidates, next_context.shape[-1])
    next_best = target_model(flat_spatial, flat_context).max(dim=1).values
    return reward + discount * (~terminal).float() * next_best.reshape(batch, candidates)


def relative_regret(values: Any, reference_index: int) -> Any:
    """Return reference Q minus every candidate Q; lower is better."""

    return values[:, reference_index : reference_index + 1] - values


def pairwise_ranking_loss(prediction: Any, target: Any, margin: float) -> Any:
    """Logistic loss on candidate orderings with nontrivial target gaps."""

    target_difference = target[:, :, None] - target[:, None, :]
    prediction_difference = prediction[:, :, None] - prediction[:, None, :]
    mask = torch.abs(target_difference) > margin
    if not bool(mask.any()):
        return prediction.sum() * 0.0
    direction = torch.sign(target_difference[mask])
    return functional.softplus(-direction * prediction_difference[mask]).mean()


def _metrics(
    model: Any,
    target_model: Any,
    loader: Any,
    *,
    device: str,
    discount: float,
    reference_index: int,
    ranking_margin: float,
) -> dict[str, float] | None:
    if len(loader.dataset) == 0:
        return None
    absolute_errors = []
    regret_errors = []
    correct = 0
    compared = 0
    model.eval()
    target_model.eval()
    with torch.no_grad():
        for spatial, context, next_spatial, next_context, reward, terminal in loader:
            spatial, context = spatial.to(device), context.to(device)
            next_spatial, next_context = next_spatial.to(device), next_context.to(device)
            reward, terminal = reward.to(device), terminal.to(device)
            prediction = model(spatial, context)
            target = _bellman_targets(
                target_model, next_spatial, next_context, reward, terminal, discount
            )
            predicted_regret = relative_regret(prediction, reference_index)
            target_regret = relative_regret(target, reference_index)
            absolute_errors.append(torch.abs(prediction - target).cpu().numpy())
            regret_errors.append(torch.abs(predicted_regret - target_regret).cpu().numpy())
            target_difference = target[:, :, None] - target[:, None, :]
            prediction_difference = prediction[:, :, None] - prediction[:, None, :]
            mask = torch.abs(target_difference) > ranking_margin
            correct += int(
                ((target_difference[mask] * prediction_difference[mask]) > 0).sum().cpu()
            )
            compared += int(mask.sum().cpu())
    return {
        "absolute_bellman_mae": float(np.mean(np.concatenate(absolute_errors, axis=0))),
        "reference_regret_mae": float(np.mean(np.concatenate(regret_errors, axis=0))),
        "pairwise_ranking_accuracy": float(correct / compared) if compared else 1.0,
    }


def train_meta_regret(
    dataset: MetaQTransitionDataset,
    *,
    output_dir: Path,
    config: MetaRegretTrainingConfig | None = None,
) -> dict[str, Any]:
    """Train a dueling Q model with reference-regret and ranking supervision."""

    config = config or MetaRegretTrainingConfig()
    if torch is None:
        raise ImportError("meta-regret learning requires PyTorch")
    torch.set_num_threads(config.torch_threads)
    torch.manual_seed(config.hidden_seed)
    np.random.seed(config.hidden_seed)
    random.seed(config.hidden_seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_index = dataset.allocations.index(META_REGRET_REFERENCE)
    split = _split_instances(
        dataset.instance_seeds,
        seed=config.hidden_seed,
        validation_fraction=config.validation_fraction,
        test_fraction=config.test_fraction,
    )
    train_context = np.stack(
        [
            augment_regret_context(
                dataset.context[index],
                Allocation(
                    int(dataset.source_resolution[index]),
                    int(dataset.source_depth[index]),
                ),
            )
            for index in split["train"]
        ]
    )
    context_mean = train_context.mean(axis=0).astype(np.float32)
    context_scale = train_context.std(axis=0).astype(np.float32)
    context_scale[context_scale < 1e-6] = 1.0
    rows = {
        name: _RegretRows(dataset, indices, context_mean, context_scale)
        for name, indices in split.items()
    }
    loaders = {
        name: DataLoader(values, batch_size=config.batch_size, shuffle=name == "train")
        for name, values in rows.items()
    }
    model = MetaRegretNetwork(
        allocations=len(dataset.allocations), reference_index=reference_index
    ).to(config.device)
    target_model = copy.deepcopy(model).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history = []
    updates = 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        losses = []
        for spatial, context, next_spatial, next_context, reward, terminal in loaders["train"]:
            spatial, context = spatial.to(config.device), context.to(config.device)
            next_spatial, next_context = (
                next_spatial.to(config.device),
                next_context.to(config.device),
            )
            reward, terminal = reward.to(config.device), terminal.to(config.device)
            with torch.no_grad():
                target = _bellman_targets(
                    target_model,
                    next_spatial,
                    next_context,
                    reward,
                    terminal,
                    config.discount,
                )
            prediction = model(spatial, context)
            absolute_loss = functional.smooth_l1_loss(prediction, target)
            regret_loss = functional.smooth_l1_loss(
                relative_regret(prediction, reference_index),
                relative_regret(target, reference_index),
            )
            ranking_loss = pairwise_ranking_loss(prediction, target, config.ranking_margin)
            loss = (
                config.absolute_weight * absolute_loss
                + config.regret_weight * regret_loss
                + config.ranking_weight * ranking_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            updates += 1
            if updates % config.target_update_interval == 0:
                target_model.load_state_dict(model.state_dict())
            losses.append(float(loss.detach().cpu()))
        target_model.load_state_dict(model.state_dict())
        validation = _metrics(
            model,
            target_model,
            loaders["validation"],
            device=config.device,
            discount=config.discount,
            reference_index=reference_index,
            ranking_margin=config.ranking_margin,
        )
        history.append(
            {"epoch": epoch, "training_loss": float(np.mean(losses)), "validation": validation}
        )
    compute_profile = np.median(dataset.compute_ms[split["train"]], axis=0).astype(np.float32)
    checkpoint = {
        "schema_version": META_REGRET_CHECKPOINT_SCHEMA,
        "model_type": META_REGRET_MODEL_TYPE,
        "feature_schema": META_Q_FEATURE_SCHEMA,
        "model_state": model.state_dict(),
        "context_mean": context_mean,
        "context_scale": context_scale,
        "compute_profile_ms": compute_profile,
        "allocations": [
            {"resolution": a.resolution, "depth": a.depth} for a in dataset.allocations
        ],
        "reference_allocation": {
            "resolution": META_REGRET_REFERENCE.resolution,
            "depth": META_REGRET_REFERENCE.depth,
        },
        "discount": config.discount,
        "training_config": asdict(config),
    }
    torch.save(checkpoint, output_dir / "best_model.pt")
    report = {
        "schema_version": META_REGRET_CHECKPOINT_SCHEMA,
        "model_type": META_REGRET_MODEL_TYPE,
        "transitions": int(dataset.spatial.shape[0]),
        "split_rows": {name: int(indices.size) for name, indices in split.items()},
        "split_instances": {
            name: int(np.unique(dataset.instance_seeds[indices]).size)
            for name, indices in split.items()
        },
        "checkpoint_epoch": config.epochs,
        "validation": _metrics(
            model,
            target_model,
            loaders["validation"],
            device=config.device,
            discount=config.discount,
            reference_index=reference_index,
            ranking_margin=config.ranking_margin,
        ),
        "test": _metrics(
            model,
            target_model,
            loaders["test"],
            device=config.device,
            discount=config.discount,
            reference_index=reference_index,
            ranking_margin=config.ranking_margin,
        ),
        "compute_profile_ms": compute_profile.tolist(),
        "history": history,
    }
    (output_dir / "training_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report
