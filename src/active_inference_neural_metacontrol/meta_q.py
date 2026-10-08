"""Fitted-Q learning and cheapest-sufficient allocation selection."""

from __future__ import annotations

import copy
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .allocations import ALLOCATIONS, Allocation
from .meta_q_data import MetaQTransitionDataset
from .meta_q_features import META_Q_CONTEXT_FEATURES, META_Q_FEATURE_SCHEMA

META_Q_CHECKPOINT_SCHEMA = 1

try:
    import torch
    from torch import nn
    from torch.nn import functional
    from torch.utils.data import DataLoader, Dataset
except ImportError:  # pragma: no cover - optional neural dependency
    torch = None
    nn = None
    functional = None
    DataLoader = None
    Dataset = object


if nn is not None:

    class MetaQNetwork(nn.Module):
        """Predict task return for every joint resolution/depth allocation."""

        def __init__(
            self,
            *,
            spatial_channels: int = 6,
            context_features: int = META_Q_CONTEXT_FEATURES,
            allocations: int = len(ALLOCATIONS),
        ) -> None:
            super().__init__()
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
            self.q_head = nn.Linear(64, allocations)

        def forward(self, spatial: Any, context: Any) -> Any:
            encoded = self.spatial_encoder(spatial).flatten(start_dim=1)
            return self.q_head(self.fusion(torch.cat((encoded, context), dim=1)))

else:

    class MetaQNetwork:  # pragma: no cover - dependency guard
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            raise ImportError("meta-Q learning requires PyTorch")


@dataclass(frozen=True)
class MetaQTrainingConfig:
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 3e-4
    discount: float = 0.98
    target_update_interval: int = 100
    hidden_seed: int = 0
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    device: str = "cpu"
    torch_threads: int = 1

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1 or self.learning_rate <= 0:
            raise ValueError("invalid training hyperparameters")
        if not 0 <= self.discount <= 1:
            raise ValueError("discount must lie in [0, 1]")
        if self.target_update_interval < 1:
            raise ValueError("target_update_interval must be positive")
        if self.validation_fraction < 0 or self.test_fraction < 0:
            raise ValueError("split fractions cannot be negative")
        if self.validation_fraction + self.test_fraction >= 1:
            raise ValueError("validation and test fractions must sum to less than one")


@dataclass(frozen=True)
class MetaQSelection:
    allocation: Allocation
    index: int
    best_task_index: int
    q_value: float
    best_q_value: float
    task_gap: float
    resource_score: float
    sufficient_indices: tuple[int, ...]


def select_cheapest_sufficient(
    q_values: np.ndarray,
    *,
    allocations: tuple[Allocation, ...] = ALLOCATIONS,
    compute_ms: np.ndarray,
    current_allocation: Allocation,
    task_tolerance: float,
    compute_weight: float = 1.0,
    switch_cost_ms: float = 0.0,
    switching_weight: float = 1.0,
) -> MetaQSelection:
    """Choose minimum resource cost among candidates close to best task value."""

    q = np.asarray(q_values, dtype=float).reshape(-1)
    compute = np.asarray(compute_ms, dtype=float).reshape(-1)
    if q.shape != (len(allocations),) or compute.shape != q.shape:
        raise ValueError("q_values and compute_ms must match allocations")
    if not np.all(np.isfinite(q)) or not np.all(np.isfinite(compute)):
        raise ValueError("selection inputs must be finite")
    if min(task_tolerance, compute_weight, switch_cost_ms, switching_weight) < 0:
        raise ValueError("selection costs and tolerance must be nonnegative")
    best = int(np.argmax(q))
    sufficient = np.flatnonzero(q >= q[best] - task_tolerance)
    switching = np.asarray(
        [allocation != current_allocation for allocation in allocations], dtype=float
    )
    resource = compute_weight * compute + switching_weight * switch_cost_ms * switching
    selected = int(sufficient[np.argmin(resource[sufficient])])
    return MetaQSelection(
        allocation=allocations[selected],
        index=selected,
        best_task_index=best,
        q_value=float(q[selected]),
        best_q_value=float(q[best]),
        task_gap=float(q[best] - q[selected]),
        resource_score=float(resource[selected]),
        sufficient_indices=tuple(int(value) for value in sufficient),
    )


def select_joint_task_compute(
    q_values: np.ndarray,
    *,
    allocations: tuple[Allocation, ...] = ALLOCATIONS,
    compute_ms: np.ndarray,
    current_allocation: Allocation,
    compute_price: float,
    switch_cost_ms: float = 0.0,
    switching_weight: float = 1.0,
    switch_penalty_mode: str = "allocation",
    information_loss: np.ndarray | None = None,
    information_loss_weight: float = 0.0,
) -> MetaQSelection:
    """Minimize task regret, priced computation, and destructive belief loss."""

    q = np.asarray(q_values, dtype=float).reshape(-1)
    compute = np.asarray(compute_ms, dtype=float).reshape(-1)
    if q.shape != (len(allocations),) or compute.shape != q.shape:
        raise ValueError("q_values and compute_ms must match allocations")
    if not np.all(np.isfinite(q)) or not np.all(np.isfinite(compute)):
        raise ValueError("selection inputs must be finite")
    if (
        min(
            compute_price,
            switch_cost_ms,
            switching_weight,
            information_loss_weight,
        )
        < 0
    ):
        raise ValueError("joint-objective costs must be nonnegative")
    information = (
        np.zeros_like(q)
        if information_loss is None
        else np.asarray(information_loss, dtype=float).reshape(-1)
    )
    if information.shape != q.shape or not np.all(np.isfinite(information)):
        raise ValueError("information_loss must match allocations and be finite")
    if np.any((information < 0) | (information > 1)):
        raise ValueError("information_loss values must lie in [0, 1]")
    if switch_penalty_mode not in {"allocation", "resolution", "none"}:
        raise ValueError("switch_penalty_mode must be allocation, resolution, or none")
    best = int(np.argmax(q))
    regret = q[best] - q
    if switch_penalty_mode == "allocation":
        switching = np.asarray(
            [allocation != current_allocation for allocation in allocations], dtype=float
        )
    elif switch_penalty_mode == "resolution":
        switching = np.asarray(
            [allocation.resolution != current_allocation.resolution for allocation in allocations],
            dtype=float,
        )
    else:
        switching = np.zeros(len(allocations), dtype=float)
    priced_compute = compute_price * (compute + switching_weight * switch_cost_ms * switching)
    objective = regret + priced_compute + information_loss_weight * information
    selected = int(np.argmin(objective))
    return MetaQSelection(
        allocation=allocations[selected],
        index=selected,
        best_task_index=best,
        q_value=float(q[selected]),
        best_q_value=float(q[best]),
        task_gap=float(regret[selected]),
        resource_score=float(objective[selected]),
        sufficient_indices=(selected,),
    )


def aggregate_reference_relative_q(
    q_samples: np.ndarray,
    *,
    reference_index: int,
    uncertainty_beta: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate an ensemble after removing each member's absolute Q offset.

    The returned robust score is a lower confidence bound: ensemble mean minus
    ``uncertainty_beta`` times ensemble standard deviation.  All three arrays
    remain exactly zero at the reference allocation.
    """

    samples = np.asarray(q_samples, dtype=float)
    if samples.ndim != 2 or samples.shape[1] < 1:
        raise ValueError("q_samples must have shape (members, allocations)")
    if not 0 <= reference_index < samples.shape[1]:
        raise ValueError("reference_index is outside q_samples")
    if uncertainty_beta < 0:
        raise ValueError("uncertainty_beta cannot be negative")
    if not np.all(np.isfinite(samples)):
        raise ValueError("q_samples must be finite")
    relative = samples - samples[:, reference_index : reference_index + 1]
    mean = relative.mean(axis=0)
    standard_deviation = relative.std(axis=0, ddof=0)
    robust = mean - uncertainty_beta * standard_deviation
    return robust, mean, standard_deviation


class _TransitionRows(Dataset):
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

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, item: int) -> tuple[Any, ...]:
        index = int(self.indices[item])
        context = (self.dataset.context[index] - self.context_mean) / self.context_scale
        next_context = (
            self.dataset.next_context[index] - self.context_mean[None, :]
        ) / self.context_scale[None, :]
        return (
            torch.from_numpy(self.dataset.spatial[index].astype(np.float32)),
            torch.from_numpy(context.astype(np.float32)),
            torch.from_numpy(self.dataset.next_spatial[index].astype(np.float32)),
            torch.from_numpy(next_context.astype(np.float32)),
            torch.from_numpy(self.dataset.reward[index]),
            torch.from_numpy(self.dataset.terminal[index]),
        )


def _split_instances(
    instance_seeds: np.ndarray,
    *,
    seed: int,
    validation_fraction: float,
    test_fraction: float,
) -> dict[str, np.ndarray]:
    unique = np.unique(instance_seeds)
    rng = np.random.default_rng(seed)
    unique = rng.permutation(unique)
    if unique.size == 1:
        return {
            "train": np.arange(instance_seeds.size),
            "validation": np.empty(0, dtype=int),
            "test": np.empty(0, dtype=int),
        }
    validation_count = round(unique.size * validation_fraction)
    test_count = round(unique.size * test_fraction)
    if validation_fraction > 0:
        validation_count = max(1, validation_count)
    if test_fraction > 0 and unique.size - validation_count > 1:
        test_count = max(1, test_count)
    while validation_count + test_count >= unique.size:
        if test_count > 0:
            test_count -= 1
        elif validation_count > 0:
            validation_count -= 1
    validation_seeds = unique[:validation_count]
    test_seeds = unique[validation_count : validation_count + test_count]
    train_seeds = unique[validation_count + test_count :]
    return {
        "train": np.flatnonzero(np.isin(instance_seeds, train_seeds)),
        "validation": np.flatnonzero(np.isin(instance_seeds, validation_seeds)),
        "test": np.flatnonzero(np.isin(instance_seeds, test_seeds)),
    }


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
    next_best = next_best.reshape(batch, candidates)
    return reward + discount * (~terminal).float() * next_best


def _evaluate_bellman(
    model: Any,
    target_model: Any,
    loader: Any,
    *,
    device: str,
    discount: float,
) -> dict[str, float] | None:
    if len(loader.dataset) == 0:
        return None
    absolute = []
    losses = []
    model.eval()
    target_model.eval()
    with torch.no_grad():
        for spatial, context, next_spatial, next_context, reward, terminal in loader:
            spatial = spatial.to(device)
            context = context.to(device)
            next_spatial = next_spatial.to(device)
            next_context = next_context.to(device)
            reward = reward.to(device)
            terminal = terminal.to(device)
            prediction = model(spatial, context)
            target = _bellman_targets(
                target_model,
                next_spatial,
                next_context,
                reward,
                terminal,
                discount,
            )
            losses.append(float(functional.smooth_l1_loss(prediction, target).cpu()))
            absolute.append(torch.abs(prediction - target).cpu().numpy())
    return {
        "bellman_huber": float(np.mean(losses)),
        "bellman_mae": float(np.mean(np.concatenate(absolute, axis=0))),
    }


def train_meta_q(
    dataset: MetaQTransitionDataset,
    *,
    output_dir: Path,
    config: MetaQTrainingConfig | None = None,
) -> dict[str, Any]:
    """Train a target-network fitted-Q model and write the best checkpoint."""

    if config is None:
        config = MetaQTrainingConfig()
    if torch is None:
        raise ImportError("meta-Q learning requires PyTorch")
    if dataset.spatial.shape[0] < 1:
        raise ValueError("cannot train on an empty transition dataset")
    torch.set_num_threads(config.torch_threads)
    torch.manual_seed(config.hidden_seed)
    np.random.seed(config.hidden_seed)
    random.seed(config.hidden_seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    split = _split_instances(
        dataset.instance_seeds,
        seed=config.hidden_seed,
        validation_fraction=config.validation_fraction,
        test_fraction=config.test_fraction,
    )
    train_context = dataset.context[split["train"]]
    context_mean = train_context.mean(axis=0).astype(np.float32)
    context_scale = train_context.std(axis=0).astype(np.float32)
    context_scale[context_scale < 1e-6] = 1.0
    rows = {
        name: _TransitionRows(dataset, indices, context_mean, context_scale)
        for name, indices in split.items()
    }
    loaders = {
        name: DataLoader(
            values,
            batch_size=config.batch_size,
            shuffle=name == "train",
        )
        for name, values in rows.items()
    }

    model = MetaQNetwork(allocations=len(dataset.allocations)).to(config.device)
    target_model = copy.deepcopy(model).to(config.device)
    target_model.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history = []
    updates = 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        training_losses = []
        for spatial, context, next_spatial, next_context, reward, terminal in loaders["train"]:
            spatial = spatial.to(config.device)
            context = context.to(config.device)
            next_spatial = next_spatial.to(config.device)
            next_context = next_context.to(config.device)
            reward = reward.to(config.device)
            terminal = terminal.to(config.device)
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
            loss = functional.smooth_l1_loss(prediction, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            updates += 1
            if updates % config.target_update_interval == 0:
                target_model.load_state_dict(model.state_dict())
            training_losses.append(float(loss.detach().cpu()))
        target_model.load_state_dict(model.state_dict())
        validation = _evaluate_bellman(
            model,
            target_model,
            loaders["validation"],
            device=config.device,
            discount=config.discount,
        )
        history.append(
            {
                "epoch": epoch,
                "training_bellman_huber": float(np.mean(training_losses)),
                "validation": validation,
            }
        )
    # A fitted-Q target changes as delayed return propagates backwards. Early
    # near-zero networks can therefore have deceptively small one-step Bellman
    # residuals. Retain the final fixed-point iterate instead of early-stopping
    # against targets produced by a different value function.
    target_model.load_state_dict(model.state_dict())
    compute_profile = np.median(dataset.compute_ms[split["train"]], axis=0).astype(np.float32)
    checkpoint = {
        "schema_version": META_Q_CHECKPOINT_SCHEMA,
        "model_type": "meta_q",
        "feature_schema": META_Q_FEATURE_SCHEMA,
        "model_state": model.state_dict(),
        "context_mean": context_mean,
        "context_scale": context_scale,
        "compute_profile_ms": compute_profile,
        "allocations": [
            {"resolution": a.resolution, "depth": a.depth} for a in dataset.allocations
        ],
        "discount": config.discount,
        "training_config": asdict(config),
    }
    torch.save(checkpoint, output_dir / "best_model.pt")
    report = {
        "schema_version": META_Q_CHECKPOINT_SCHEMA,
        "transitions": int(dataset.spatial.shape[0]),
        "split_rows": {name: int(indices.size) for name, indices in split.items()},
        "split_instances": {
            name: int(np.unique(dataset.instance_seeds[indices]).size)
            for name, indices in split.items()
        },
        "checkpoint_epoch": config.epochs,
        "validation": _evaluate_bellman(
            model,
            target_model,
            loaders["validation"],
            device=config.device,
            discount=config.discount,
        ),
        "test": _evaluate_bellman(
            model,
            target_model,
            loaders["test"],
            device=config.device,
            discount=config.discount,
        ),
        "compute_profile_ms": compute_profile.tolist(),
        "history": history,
    }
    (output_dir / "training_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report
