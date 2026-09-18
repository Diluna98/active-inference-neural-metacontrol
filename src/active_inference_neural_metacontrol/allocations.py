"""The discrete joint resolution/depth allocation space."""

from __future__ import annotations

from dataclasses import dataclass

RESOLUTIONS = (2, 5, 10, 20)
DEPTHS = (1, 2, 3)


@dataclass(frozen=True, order=True)
class Allocation:
    """One task-model allocation selected by the metacontroller."""

    resolution: int
    depth: int

    def __post_init__(self) -> None:
        if self.resolution not in RESOLUTIONS:
            raise ValueError(f"resolution must be one of {RESOLUTIONS}")
        if self.depth not in DEPTHS:
            raise ValueError(f"depth must be one of {DEPTHS}")


ALLOCATIONS = tuple(Allocation(resolution, depth) for resolution in RESOLUTIONS for depth in DEPTHS)
ALLOCATION_INDEX = {allocation: index for index, allocation in enumerate(ALLOCATIONS)}
