"""Neural metacontrol for joint Active-Inference resource allocation."""

from .allocations import ALLOCATIONS, DEPTHS, RESOLUTIONS, Allocation
from .beliefs import (
    canonicalize_posterior,
    information_loss,
    normalized_entropy,
    project_canonical,
    remap_posterior,
)
from .costs import ComputeProfile, ProfiledComputeCostModel, SwitchingCostMatrix
from .features import SpatialFeatures, build_context_vector, build_spatial_features
from .information import (
    categorical_fisher_map,
    detection_probability_map,
    expected_information_gain,
)
from .selector import AllocationDecision, SelectionConstraints, select_allocation

__all__ = [
    "ALLOCATIONS",
    "DEPTHS",
    "RESOLUTIONS",
    "Allocation",
    "AllocationDecision",
    "ComputeProfile",
    "ProfiledComputeCostModel",
    "SelectionConstraints",
    "SpatialFeatures",
    "SwitchingCostMatrix",
    "build_context_vector",
    "build_spatial_features",
    "canonicalize_posterior",
    "categorical_fisher_map",
    "detection_probability_map",
    "expected_information_gain",
    "information_loss",
    "normalized_entropy",
    "project_canonical",
    "remap_posterior",
    "select_allocation",
]
