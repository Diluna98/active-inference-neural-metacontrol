"""Independent computation and switching cost models."""

from __future__ import annotations

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
