"""Neural metacontrol for joint Active-Inference resource allocation."""

from .allocations import ALLOCATIONS, DEPTHS, RESOLUTIONS, Allocation
from .beliefs import (
    canonicalize_posterior,
    information_loss,
    normalized_entropy,
    project_canonical,
    remap_posterior,
)
from .closed_loop import ClosedLoopConfig, NeuralMetaController, evaluate_closed_loop
from .costs import (
    ComputeProfile,
    LatencyScalePreference,
    LogNormalDeadlinePreference,
    ProfiledComputeCostModel,
    SwitchingCostMatrix,
)
from .datasets import (
    CounterfactualArrays,
    DatasetSplit,
    load_counterfactual_dataset,
    split_by_instance,
)
from .features import SpatialFeatures, build_context_vector, build_spatial_features
from .generation import (
    GenerationConfig,
    generate_balanced_mos_counterfactuals,
    generate_resumable_mos_counterfactuals,
)
from .information import (
    categorical_fisher_map,
    detection_probability_map,
    expected_information_gain,
)
from .mos_adapter import MOSFeatureBatch, build_mos_features
from .rollout_diagnostic import (
    GOracleConfig,
    diagnose_accumulated_g,
    evaluate_g_oracles,
    run_g_oracle_episode,
)
from .selector import AllocationDecision, select_allocation

__all__ = [
    "ALLOCATIONS",
    "DEPTHS",
    "RESOLUTIONS",
    "Allocation",
    "AllocationDecision",
    "ClosedLoopConfig",
    "ComputeProfile",
    "CounterfactualArrays",
    "DatasetSplit",
    "GOracleConfig",
    "GenerationConfig",
    "LatencyScalePreference",
    "LogNormalDeadlinePreference",
    "MOSFeatureBatch",
    "NeuralMetaController",
    "ProfiledComputeCostModel",
    "SpatialFeatures",
    "SwitchingCostMatrix",
    "build_context_vector",
    "build_mos_features",
    "build_spatial_features",
    "canonicalize_posterior",
    "categorical_fisher_map",
    "detection_probability_map",
    "diagnose_accumulated_g",
    "evaluate_closed_loop",
    "evaluate_g_oracles",
    "expected_information_gain",
    "generate_balanced_mos_counterfactuals",
    "generate_resumable_mos_counterfactuals",
    "information_loss",
    "load_counterfactual_dataset",
    "normalized_entropy",
    "project_canonical",
    "remap_posterior",
    "run_g_oracle_episode",
    "select_allocation",
    "split_by_instance",
]
