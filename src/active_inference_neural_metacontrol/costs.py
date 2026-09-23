"""Independent computation and switching cost models."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .allocations import ALLOCATION_INDEX, ALLOCATIONS, Allocation


@dataclass(frozen=True)
class ComputeProfile:
    """Measured steady-state inference latency for one allocation."""

    median_ms: float
    p95_ms: float

    def __post_init__(self) -> None:
        if self.median_ms < 0 or self.p95_ms < self.median_ms:
            raise ValueError("compute-profile latencies must satisfy 0 <= median <= p95")


class ProfiledComputeCostModel:
    """Predict latency from measured allocation profiles and a load multiplier."""

    def __init__(self, profiles: Mapping[Allocation, ComputeProfile]) -> None:
        missing = set(ALLOCATIONS).difference(profiles)
        if missing:
            raise ValueError(f"missing compute profiles for: {sorted(missing)}")
        self._profiles = dict(profiles)

    def predict(
        self,
        allocation: Allocation,
        *,
        load_multiplier: float = 1.0,
        conservative: bool = False,
    ) -> float:
        if not np.isfinite(load_multiplier) or load_multiplier <= 0:
            raise ValueError("load_multiplier must be finite and positive")
        profile = self._profiles[allocation]
        baseline = profile.p95_ms if conservative else profile.median_ms
        return float(baseline * load_multiplier)

    def predict_all(
        self,
        *,
        load_multiplier: float = 1.0,
        conservative: bool = False,
    ) -> np.ndarray:
        return np.asarray(
            [
                self.predict(
                    allocation,
                    load_multiplier=load_multiplier,
                    conservative=conservative,
                )
                for allocation in ALLOCATIONS
            ],
            dtype=float,
        )


@dataclass(frozen=True)
class LogNormalDeadlinePreference:
    """Preference over latency induced by a log-normal available deadline.

    The compute cost is the surprisal ``-log P(D >= latency)`` in nats, where
    ``D`` is the random time for which the pending decision remains useful.
    """

    median_ms: float
    log_sigma: float = 0.5
    probability_floor: float = 1e-12

    def __post_init__(self) -> None:
        if not np.isfinite(self.median_ms) or self.median_ms <= 0:
            raise ValueError("median_ms must be finite and positive")
        if not np.isfinite(self.log_sigma) or self.log_sigma <= 0:
            raise ValueError("log_sigma must be finite and positive")
        if not 0 < self.probability_floor < 1:
            raise ValueError("probability_floor must lie strictly between zero and one")

    def survival_probability(self, latency_ms: np.ndarray | float) -> np.ndarray:
        latency = np.asarray(latency_ms, dtype=float)
        if not np.all(np.isfinite(latency)) or np.any(latency < 0):
            raise ValueError("latency_ms must be finite and nonnegative")
        survival = np.ones_like(latency, dtype=float)
        positive = latency > 0
        z = np.log(latency[positive] / self.median_ms) / (self.log_sigma * math.sqrt(2.0))
        survival[positive] = np.asarray([0.5 * math.erfc(float(value)) for value in z])
        return np.clip(survival, self.probability_floor, 1.0)

    def surprisal_nats(self, latency_ms: np.ndarray | float) -> np.ndarray:
        """Return negative-log deadline survival in natural information units."""

        return -np.log(self.survival_probability(latency_ms))


@dataclass(frozen=True)
class LatencyScalePreference:
    """ICRA-style preference over the continuous inference-latency scale.

    The unnormalised log preference is the negative of a linear latency cost
    plus a quadratic excess-latency cost above ``comfort_ms``. Exponentiating
    and normalising this utility on any common latency grid gives the same
    ordering and pairwise log-preference differences, so ``cost_nats`` returns
    the preference surprisal relative to zero latency without introducing a
    grid-dependent normalising constant.
    """

    comfort_ms: float
    deadline_ms: float
    linear_weight: float = 1.0
    excess_weight: float = 2.0

    def __post_init__(self) -> None:
        values = np.asarray(
            [
                self.comfort_ms,
                self.deadline_ms,
                self.linear_weight,
                self.excess_weight,
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("latency-scale preference parameters must be finite")
        if self.comfort_ms < 0 or self.deadline_ms <= self.comfort_ms:
            raise ValueError("deadline_ms must exceed nonnegative comfort_ms")
        if self.linear_weight < 0 or self.excess_weight < 0:
            raise ValueError("latency preference weights must be nonnegative")

    def utility(self, latency_ms: np.ndarray | float) -> np.ndarray:
        """Return unnormalised log preference for one or more latencies."""

        return -self.cost_nats(latency_ms)

    def cost_nats(self, latency_ms: np.ndarray | float) -> np.ndarray:
        """Return latency preference surprisal relative to zero latency."""

        latency = np.asarray(latency_ms, dtype=float)
        if not np.all(np.isfinite(latency)) or np.any(latency < 0):
            raise ValueError("latency_ms must be finite and nonnegative")
        linear = self.linear_weight * latency / self.deadline_ms
        excess = np.clip(
            (latency - self.comfort_ms) / (self.deadline_ms - self.comfort_ms),
            0.0,
            None,
        )
        return linear + self.excess_weight * excess**2


class SwitchingCostMatrix:
    """Measured one-off transition latency between all allocation pairs."""

    def __init__(self, latency_ms: np.ndarray) -> None:
        values = np.asarray(latency_ms, dtype=float)
        expected = (len(ALLOCATIONS), len(ALLOCATIONS))
        if values.shape != expected:
            raise ValueError(f"switching matrix must have shape {expected}")
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("switching costs must be finite and nonnegative")
        if not np.allclose(np.diag(values), 0.0):
            raise ValueError("switching-cost diagonal must be zero")
        self._latency_ms = values.copy()

    @classmethod
    def zeros(cls) -> SwitchingCostMatrix:
        return cls(np.zeros((len(ALLOCATIONS), len(ALLOCATIONS)), dtype=float))

    def cost(self, current: Allocation, candidate: Allocation) -> float:
        return float(self._latency_ms[ALLOCATION_INDEX[current], ALLOCATION_INDEX[candidate]])

    def from_current(self, current: Allocation) -> np.ndarray:
        return self._latency_ms[ALLOCATION_INDEX[current]].copy()
