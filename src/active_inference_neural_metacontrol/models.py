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
        """Small CNN/MLP predicting success and task cost for every allocation."""

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
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.fusion = nn.Sequential(
                nn.Linear(32 + context_features, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
            )
            self.success_head = nn.Linear(64, allocations)
            self.task_cost_head = nn.Linear(64, allocations)

        def forward(self, spatial, context):
            encoded = self.spatial_encoder(spatial).flatten(start_dim=1)
            fused = self.fusion(torch.cat((encoded, context), dim=1))
            return {
                "success_logits": self.success_head(fused),
                "task_cost": functional.softplus(self.task_cost_head(fused)),
            }

    def task_performance_loss(
        prediction,
        *,
        success_target,
        task_cost_target,
        success_weight: float = 1.0,
        task_cost_weight: float = 1.0,
    ):
        """Joint auxiliary success and positive cost-regression objective."""

        success_loss = functional.binary_cross_entropy_with_logits(
            prediction["success_logits"],
            success_target,
        )
        task_loss = functional.smooth_l1_loss(
            torch.log1p(prediction["task_cost"]),
            torch.log1p(task_cost_target),
        )
        return success_weight * success_loss + task_cost_weight * task_loss

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
