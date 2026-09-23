"""Optional PyTorch task-performance model.

Install the ``neural`` extra to use this module. Core feature and cost utilities
remain NumPy-only so data generation and analysis do not require PyTorch.
"""

from __future__ import annotations

try:
    import torch
    from torch import nn
    from torch.nn import functional
except ImportError:  # pragma: no cover - exercised only without the optional extra
    torch = None
    nn = None
    functional = None


if nn is not None:

    class TaskPerformanceNetwork(nn.Module):
        """Small CNN/MLP predicting normalized selected-policy G per allocation."""

        def __init__(
            self,
            *,
            spatial_channels: int,
            context_features: int,
            allocations: int = 12,
        ) -> None:
            super().__init__()
            if spatial_channels < 1 or context_features < 1 or allocations < 1:
                raise ValueError("network dimensions must be positive")
            self.spatial_encoder = nn.Sequential(
                nn.Conv2d(spatial_channels, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                # Preserve coarse absolute and relational geometry. A global
                # 1x1 average would make distinct MOS layouts too similar.
                nn.AdaptiveAvgPool2d((5, 5)),
            )
            self.fusion = nn.Sequential(
                nn.Linear(32 * 5 * 5 + context_features, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
            )
            self.normalized_g_head = nn.Linear(64, allocations)

        def forward(self, spatial, context):
            encoded = self.spatial_encoder(spatial).flatten(start_dim=1)
            fused = self.fusion(torch.cat((encoded, context), dim=1))
            return {"normalized_g": self.normalized_g_head(fused)}

    class MultiObjectiveNetwork(nn.Module):
        """Shared MOS encoder with one candidate-wise head per supervised quantity."""

        OUTPUTS = (
            "success_logits",
            "task_cost_scaled",
            "normalized_g_scaled",
            "preference_scaled",
            "epistemic_scaled",
            "compute_log_scaled",
            "switch_log_scaled",
        )

        def __init__(
            self,
            *,
            spatial_channels: int,
            context_features: int,
            allocations: int = 12,
        ) -> None:
            super().__init__()
            if spatial_channels < 1 or context_features < 1 or allocations < 1:
                raise ValueError("network dimensions must be positive")
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
            self.heads = nn.ModuleDict({name: nn.Linear(64, allocations) for name in self.OUTPUTS})

        def forward(self, spatial, context):
            encoded = self.spatial_encoder(spatial).flatten(start_dim=1)
            fused = self.fusion(torch.cat((encoded, context), dim=1))
            return {name: head(fused) for name, head in self.heads.items()}

    class TaskUtilityNetwork(nn.Module):
        """Minimal task model for success, operational cost, and normalized G."""

        def __init__(
            self,
            *,
            spatial_channels: int,
            context_features: int,
            allocations: int = 12,
        ) -> None:
            super().__init__()
            if spatial_channels < 1 or context_features < 1 or allocations < 1:
                raise ValueError("network dimensions must be positive")
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
            self.success_head = nn.Linear(64, allocations)
            self.operational_cost_head = nn.Linear(64, allocations)
            self.normalized_g_head = nn.Linear(64, allocations)

        def forward(self, spatial, context):
            encoded = self.spatial_encoder(spatial).flatten(start_dim=1)
            fused = self.fusion(torch.cat((encoded, context), dim=1))
            return {
                "success_logits": self.success_head(fused),
                "operational_cost_scaled": self.operational_cost_head(fused),
                "normalized_g_scaled": self.normalized_g_head(fused),
            }

    class DecomposedTaskUtilityNetwork(nn.Module):
        """Reduced-input model with separate preference and epistemic heads."""

        def __init__(
            self,
            *,
            spatial_channels: int = 5,
            context_features: int = 14,
            allocations: int = 12,
        ) -> None:
            super().__init__()
            if spatial_channels < 1 or context_features < 1 or allocations < 1:
                raise ValueError("network dimensions must be positive")
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
            self.success_head = nn.Linear(64, allocations)
            self.operational_cost_head = nn.Linear(64, allocations)
            self.preference_head = nn.Linear(64, allocations)
            self.epistemic_head = nn.Linear(64, allocations)

        def forward(self, spatial, context):
            encoded = self.spatial_encoder(spatial).flatten(start_dim=1)
            fused = self.fusion(torch.cat((encoded, context), dim=1))
            return {
                "success_logits": self.success_head(fused),
                "operational_cost_scaled": self.operational_cost_head(fused),
                "preference_per_depth_scaled": self.preference_head(fused),
                "epistemic_per_depth_scaled": self.epistemic_head(fused),
            }

    class InferenceValueNetwork(nn.Module):
        """Reduced-input model for preference and epistemic value only."""

        def __init__(
            self,
            *,
            spatial_channels: int = 5,
            context_features: int = 14,
            allocations: int = 12,
        ) -> None:
            super().__init__()
            if spatial_channels < 1 or context_features < 1 or allocations < 1:
                raise ValueError("network dimensions must be positive")
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
            self.preference_head = nn.Linear(64, allocations)
            self.epistemic_head = nn.Linear(64, allocations)

        def forward(self, spatial, context):
            encoded = self.spatial_encoder(spatial).flatten(start_dim=1)
            fused = self.fusion(torch.cat((encoded, context), dim=1))
            return {
                "preference_per_depth_scaled": self.preference_head(fused),
                "epistemic_per_depth_scaled": self.epistemic_head(fused),
            }

    class FourTermValueNetwork(nn.Module):
        """Predict next-step state and prospective policy-value components."""

        def __init__(
            self,
            *,
            spatial_channels: int = 5,
            context_features: int = 14,
            allocations: int = 12,
        ) -> None:
            super().__init__()
            if spatial_channels < 1 or context_features < 1 or allocations < 1:
                raise ValueError("network dimensions must be positive")
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
            self.state_accuracy_head = nn.Linear(64, allocations)
            self.state_complexity_head = nn.Linear(64, allocations)
            self.preference_head = nn.Linear(64, allocations)
            self.epistemic_value_head = nn.Linear(64, allocations)

        def forward(self, spatial, context):
            encoded = self.spatial_encoder(spatial).flatten(start_dim=1)
            fused = self.fusion(torch.cat((encoded, context), dim=1))
            return {
                "state_accuracy_scaled": self.state_accuracy_head(fused),
                "state_complexity_scaled": self.state_complexity_head(fused),
                "preference_per_depth_scaled": self.preference_head(fused),
                "epistemic_value_per_depth_scaled": self.epistemic_value_head(fused),
            }

    class ResolutionSharedFourTermValueNetwork(nn.Module):
        """Four-term model whose filtered-state terms depend only on resolution.

        The allocation order is resolution-major with every planning depth for a
        resolution contiguous. Accuracy and complexity describe the filtered
        posterior at the next step, which is independent of prospective policy
        depth. Their four resolution-level predictions are therefore repeated
        across depth, while prospective preference and epistemic value retain
        one prediction per joint (resolution, depth) allocation.
        """

        def __init__(
            self,
            *,
            spatial_channels: int = 5,
            context_features: int = 14,
            allocations: int = 12,
            resolutions: int = 4,
        ) -> None:
            super().__init__()
            if spatial_channels < 1 or context_features < 1:
                raise ValueError("network dimensions must be positive")
            if allocations < 1 or resolutions < 1 or allocations % resolutions:
                raise ValueError("allocations must be divisible by resolutions")
            self.allocations = allocations
            self.resolutions = resolutions
            self.depths_per_resolution = allocations // resolutions
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
            self.state_accuracy_head = nn.Linear(64, resolutions)
            self.state_complexity_head = nn.Linear(64, resolutions)
            self.preference_head = nn.Linear(64, allocations)
            self.epistemic_value_head = nn.Linear(64, allocations)

        def _broadcast_depths(self, values):
            return values.repeat_interleave(self.depths_per_resolution, dim=1)

        def forward(self, spatial, context):
            encoded = self.spatial_encoder(spatial).flatten(start_dim=1)
            fused = self.fusion(torch.cat((encoded, context), dim=1))
            return {
                "state_accuracy_scaled": self._broadcast_depths(
                    self.state_accuracy_head(fused)
                ),
                "state_complexity_scaled": self._broadcast_depths(
                    self.state_complexity_head(fused)
                ),
                "preference_per_depth_scaled": self.preference_head(fused),
                "epistemic_value_per_depth_scaled": self.epistemic_value_head(fused),
            }

    class ImmediateImprovementValueNetwork(nn.Module):
        """Predict state evidence plus immediate and continuation improvements."""

        def __init__(
            self,
            *,
            spatial_channels: int = 5,
            context_features: int = 14,
            allocations: int = 12,
            resolutions: int = 4,
        ) -> None:
            super().__init__()
            if spatial_channels < 1 or context_features < 1:
                raise ValueError("network dimensions must be positive")
            if allocations < 1 or resolutions < 1 or allocations % resolutions:
                raise ValueError("allocations must be divisible by resolutions")
            self.depths_per_resolution = allocations // resolutions
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
            self.state_accuracy_head = nn.Linear(64, resolutions)
            self.state_complexity_head = nn.Linear(64, resolutions)
            self.immediate_preference_head = nn.Linear(64, allocations)
            self.preference_improvement_head = nn.Linear(64, allocations)
            self.immediate_epistemic_head = nn.Linear(64, allocations)
            self.epistemic_improvement_head = nn.Linear(64, allocations)

        def _broadcast_depths(self, values):
            return values.repeat_interleave(self.depths_per_resolution, dim=1)

        def forward(self, spatial, context):
            encoded = self.spatial_encoder(spatial).flatten(start_dim=1)
            fused = self.fusion(torch.cat((encoded, context), dim=1))
            return {
                "state_accuracy_scaled": self._broadcast_depths(
                    self.state_accuracy_head(fused)
                ),
                "state_complexity_scaled": self._broadcast_depths(
                    self.state_complexity_head(fused)
                ),
                "immediate_preference_scaled": self.immediate_preference_head(fused),
                "preference_improvement_scaled": self.preference_improvement_head(fused),
                "immediate_epistemic_scaled": self.immediate_epistemic_head(fused),
                "epistemic_improvement_scaled": self.epistemic_improvement_head(fused),
            }

    def task_performance_loss(
        prediction,
        *,
        normalized_g_target,
        normalized_g_weight: float = 1.0,
        ranking_weight: float = 1.0,
        ranking_temperature: float = 5.0,
    ):
        """Joint normalized-G regression and allocation ranking objective."""

        normalized_g_loss = functional.smooth_l1_loss(
            prediction["normalized_g"], normalized_g_target
        )
        target_probability = functional.softmax(normalized_g_target / ranking_temperature, dim=1)
        predicted_log_probability = functional.log_softmax(
            prediction["normalized_g"] / ranking_temperature, dim=1
        )
        ranking_loss = -(target_probability * predicted_log_probability).sum(dim=1).mean()
        return normalized_g_weight * normalized_g_loss + ranking_weight * ranking_loss

else:

    class TaskPerformanceNetwork:  # pragma: no cover - simple dependency guard
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError(
                "PyTorch is required for TaskPerformanceNetwork; "
                "install active-inference-neural-metacontrol[neural]"
            )

    class MultiObjectiveNetwork:  # pragma: no cover - simple dependency guard
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError(
                "PyTorch is required for MultiObjectiveNetwork; "
                "install active-inference-neural-metacontrol[neural]"
            )

    class TaskUtilityNetwork:  # pragma: no cover - simple dependency guard
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError(
                "PyTorch is required for TaskUtilityNetwork; "
                "install active-inference-neural-metacontrol[neural]"
            )

    class DecomposedTaskUtilityNetwork:  # pragma: no cover - simple dependency guard
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError(
                "PyTorch is required for DecomposedTaskUtilityNetwork; "
                "install active-inference-neural-metacontrol[neural]"
            )

    class InferenceValueNetwork:  # pragma: no cover - simple dependency guard
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError(
                "PyTorch is required for InferenceValueNetwork; "
                "install active-inference-neural-metacontrol[neural]"
            )

    class FourTermValueNetwork:  # pragma: no cover - simple dependency guard
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError(
                "PyTorch is required for FourTermValueNetwork; "
                "install active-inference-neural-metacontrol[neural]"
            )

    class ResolutionSharedFourTermValueNetwork:  # pragma: no cover
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError(
                "PyTorch is required for ResolutionSharedFourTermValueNetwork; "
                "install active-inference-neural-metacontrol[neural]"
            )

    class ImmediateImprovementValueNetwork:  # pragma: no cover
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError(
                "PyTorch is required for ImmediateImprovementValueNetwork; "
                "install active-inference-neural-metacontrol[neural]"
            )

    def task_performance_loss(*args, **kwargs):  # pragma: no cover
        raise ImportError(
            "PyTorch is required for task_performance_loss; "
            "install active-inference-neural-metacontrol[neural]"
        )
