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
        """Small CNN/MLP predicting success and relative cost for every allocation."""

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
            self.success_head = nn.Linear(64, allocations)
            self.relative_cost_head = nn.Linear(64, allocations)

        def forward(self, spatial, context):
            encoded = self.spatial_encoder(spatial).flatten(start_dim=1)
            fused = self.fusion(torch.cat((encoded, context), dim=1))
            return {
                "success_logits": self.success_head(fused),
                "relative_cost": functional.softplus(self.relative_cost_head(fused)),
            }

    def task_performance_loss(
        prediction,
        *,
        success_target,
        task_cost_target,
        success_weight: float = 1.0,
        relative_cost_weight: float = 1.0,
        ranking_weight: float = 1.0,
        ranking_temperature: float = 5.0,
    ):
        """Joint success, relative-regret regression, and listwise ranking objective."""

        success_loss = functional.binary_cross_entropy_with_logits(
            prediction["success_logits"],
            success_target,
        )
        relative_target = task_cost_target - task_cost_target.min(dim=1, keepdim=True).values
        relative_loss = functional.smooth_l1_loss(
            torch.log1p(prediction["relative_cost"]),
            torch.log1p(relative_target),
        )
        target_probability = functional.softmax(-relative_target / ranking_temperature, dim=1)
        predicted_log_probability = functional.log_softmax(
            -prediction["relative_cost"] / ranking_temperature, dim=1
        )
        ranking_loss = -(target_probability * predicted_log_probability).sum(dim=1).mean()
        return (
            success_weight * success_loss
            + relative_cost_weight * relative_loss
            + ranking_weight * ranking_loss
        )

else:

    class TaskPerformanceNetwork:  # pragma: no cover - simple dependency guard
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError(
                "PyTorch is required for TaskPerformanceNetwork; "
                "install active-inference-neural-metacontrol[neural]"
            )

    def task_performance_loss(*args, **kwargs):  # pragma: no cover
        raise ImportError(
            "PyTorch is required for task_performance_loss; "
            "install active-inference-neural-metacontrol[neural]"
        )
