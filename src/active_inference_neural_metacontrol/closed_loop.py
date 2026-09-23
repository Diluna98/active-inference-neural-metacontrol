"""Closed-loop MOS evaluation for neural and fixed resource controllers."""

from __future__ import annotations

import csv
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from .allocations import ALLOCATION_INDEX, ALLOCATIONS, Allocation
from .beliefs import information_loss
from .costs import LatencyScalePreference, LogNormalDeadlinePreference
from .counterfactuals import (
    _infer_decision,
    _mos_imports,
    _step_cost,
    _timed,
    _transfer_receding_state,
)
from .features import (
    DECOMPOSED_FEATURE_SCHEMA,
    FOUR_TERM_ABLATION_FEATURE_SCHEMA,
    FOUR_TERM_NO_ALLOCATION_FEATURE_SCHEMA,
    FOUR_TERM_PREDICTED_OBSERVATION_FEATURE_SCHEMA,
    project_feature_schema,
)
from .models import (
    DecomposedTaskUtilityNetwork,
    FourTermValueNetwork,
    ImmediateImprovementValueNetwork,
    InferenceValueNetwork,
    ResolutionSharedFourTermValueNetwork,
    TaskPerformanceNetwork,
    TaskUtilityNetwork,
)
from .mos_adapter import build_mos_features
from .profiling import load_timing_profile
from .selector import (
    AllocationDecision,
    select_allocation,
    select_decomposed_task_utility_allocation,
    select_four_term_value_allocation,
    select_inference_value_allocation,
    select_task_utility_allocation,
)

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency guard
    torch = None


@dataclass(frozen=True)
class ClosedLoopConfig:
    checkpoint: str
    deadline_median_ms: float
    timing_profile: str | None = None
    deadline_log_sigma: float = 0.5
    initial_resolution: int = 5
    initial_depth: int = 2
    max_steps: int = 50
    message_passing_iterations: int = 10
    policy_workers: int = 1
    device: str = "cpu"
    torch_threads: int = 1
    include_fixed: bool = True
    hold_allocation_for_depth: bool = False
    utility_success_weight: float = 0.5
    operational_cost_weight: float = 1.0
    operational_cost_temperature: float = 5.0
    utility_g_weight: float = 1.0
    utility_preference_weight: float = 1.0
    utility_epistemic_weight: float = 1.0
    preference_improvement_weight: float = 1.0
    epistemic_improvement_weight: float = 1.0
    state_accuracy_weight: float = 1.0
    state_complexity_weight: float = 1.0
    compute_cost_weight: float = 1.0
    compute_preference_comfort_ms: float | None = None
    compute_preference_deadline_ms: float | None = None
    compute_preference_linear_weight: float = 1.0
    compute_preference_excess_weight: float = 2.0
    switching_cost_weight: float = 1.0
    fixed_switch_cost_ms: float = 1.0
    information_loss_weight: float = 1.0
    fisher_contextual_objective: bool = False
    fisher_weight_min: float = 0.1
    fisher_weight_max: float = 1.0
    fisher_weight_power: float = 1.0

    def __post_init__(self) -> None:
        Allocation(self.initial_resolution, self.initial_depth)
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if self.message_passing_iterations < 1 or self.policy_workers < 1:
            raise ValueError("inference settings must be positive")
        if self.torch_threads < 1:
            raise ValueError("torch_threads must be positive")
        weights = (
            self.utility_success_weight,
            self.operational_cost_weight,
            self.utility_g_weight,
            self.utility_preference_weight,
            self.utility_epistemic_weight,
            self.preference_improvement_weight,
            self.epistemic_improvement_weight,
            self.state_accuracy_weight,
            self.state_complexity_weight,
            self.compute_cost_weight,
            self.compute_preference_linear_weight,
            self.compute_preference_excess_weight,
            self.switching_cost_weight,
            self.information_loss_weight,
        )
        if min(weights) < 0:
            raise ValueError("utility weights must be nonnegative")
        if self.fixed_switch_cost_ms < 0:
            raise ValueError("fixed_switch_cost_ms must be nonnegative")
        if self.operational_cost_temperature <= 0:
            raise ValueError("operational_cost_temperature must be positive")
        if not 0 <= self.fisher_weight_min <= self.fisher_weight_max:
            raise ValueError("Fisher weights must satisfy 0 <= min <= max")
        if self.fisher_weight_power <= 0:
            raise ValueError("fisher_weight_power must be positive")
        LogNormalDeadlinePreference(self.deadline_median_ms, self.deadline_log_sigma)
        _ = self.compute_preference

    @property
    def initial_allocation(self) -> Allocation:
        return Allocation(self.initial_resolution, self.initial_depth)

    @property
    def deadline_preference(self) -> LogNormalDeadlinePreference:
        return LogNormalDeadlinePreference(
            median_ms=self.deadline_median_ms,
            log_sigma=self.deadline_log_sigma,
        )

    @property
    def compute_preference(self) -> LatencyScalePreference:
        deadline = (
            self.deadline_median_ms
            if self.compute_preference_deadline_ms is None
            else self.compute_preference_deadline_ms
        )
        comfort = (
            0.75 * deadline
            if self.compute_preference_comfort_ms is None
            else self.compute_preference_comfort_ms
        )
        return LatencyScalePreference(
            comfort_ms=comfort,
            deadline_ms=deadline,
            linear_weight=self.compute_preference_linear_weight,
            excess_weight=self.compute_preference_excess_weight,
        )


class NeuralMetaController:
    """Checkpoint-backed selector for the joint meta-objective."""

    def __init__(self, checkpoint: Path, *, device: str = "cpu", torch_threads: int = 1) -> None:
        if torch is None:
            raise ImportError("closed-loop neural evaluation requires PyTorch")
        torch.set_num_threads(torch_threads)
        self.device = torch.device(device)
        payload = torch.load(Path(checkpoint), map_location=self.device, weights_only=False)
        self.schema_version = int(payload.get("schema_version", -1))
        if self.schema_version not in {4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}:
            raise ValueError(
                "checkpoint must use schema version 4/5 normalized-G, version 7 task utility, "
                "version 8 decomposed task utility, version 9 inference value, "
                "version 10/11 four-term value, or version 12 resolution-shared "
                "four-term value, version 13 candidate-relative four-term value, "
                "version 14 state-detached candidate-relative four-term value, "
                "version 15 future-only four-term value, or version 16 "
                "immediate/improvement value"
            )
        model_class = {
            7: TaskUtilityNetwork,
            8: DecomposedTaskUtilityNetwork,
            9: InferenceValueNetwork,
            10: FourTermValueNetwork,
            11: FourTermValueNetwork,
            12: ResolutionSharedFourTermValueNetwork,
            13: ResolutionSharedFourTermValueNetwork,
            14: ResolutionSharedFourTermValueNetwork,
            15: ResolutionSharedFourTermValueNetwork,
            16: ImmediateImprovementValueNetwork,
        }.get(self.schema_version, TaskPerformanceNetwork)
        self.model = model_class(
            spatial_channels=int(payload["spatial_channels"]),
            context_features=int(payload["context_features"]),
            allocations=len(ALLOCATIONS),
        ).to(self.device)
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.eval()
        self.context_mean = np.asarray(payload["context_mean"], dtype=np.float32)
        self.context_scale = np.asarray(payload["context_scale"], dtype=np.float32)
        self.target_stats = payload.get("target_stats")
        self.feature_schema = payload.get("feature_schema")
        supported_feature_schemas = {DECOMPOSED_FEATURE_SCHEMA}
        if self.schema_version in {10, 11, 12, 13, 14, 15, 16}:
            supported_feature_schemas.add(FOUR_TERM_ABLATION_FEATURE_SCHEMA)
            supported_feature_schemas.add(FOUR_TERM_NO_ALLOCATION_FEATURE_SCHEMA)
            supported_feature_schemas.add(
                FOUR_TERM_PREDICTED_OBSERVATION_FEATURE_SCHEMA
            )
        if (
            self.schema_version in {8, 9, 10, 11, 12, 13, 14, 15, 16}
            and self.feature_schema not in supported_feature_schemas
        ):
            raise ValueError("decomposed checkpoint has an unsupported feature schema")
        self.compute_ms = np.asarray(payload["compute_profile_ms"], dtype=float)
        if self.compute_ms.shape != (len(ALLOCATIONS),):
            raise ValueError("checkpoint compute profile has the wrong shape")

    def choose(
        self,
        *,
        agent: Any,
        allocation: Allocation,
        layout: Any,
        selected_action: int,
        next_robot_position: tuple[int, int],
        deadline_preference: LogNormalDeadlinePreference,
        compute_preference: LatencyScalePreference,
        commit_for_depth: bool,
        utility_success_weight: float = 0.5,
        operational_cost_weight: float = 1.0,
        operational_cost_temperature: float = 5.0,
        utility_g_weight: float = 1.0,
        utility_preference_weight: float = 1.0,
        utility_epistemic_weight: float = 1.0,
        preference_improvement_weight: float = 1.0,
        epistemic_improvement_weight: float = 1.0,
        state_accuracy_weight: float = 1.0,
        state_complexity_weight: float = 1.0,
        compute_cost_weight: float = 1.0,
        switching_cost_weight: float = 1.0,
        fixed_switch_cost_ms: float = 1.0,
        information_loss_weight: float = 1.0,
        fisher_contextual_objective: bool = False,
        fisher_weight_min: float = 0.1,
        fisher_weight_max: float = 1.0,
        fisher_weight_power: float = 1.0,
    ) -> tuple[AllocationDecision, float]:
        features = build_mos_features(
            agent=agent,
            allocation=allocation,
            layout=layout,
            selected_action=selected_action,
            next_robot_position=next_robot_position,
            found_flags=np.asarray([0.0]),
        )
        spatial_values = features.spatial.tensor
        context_values = features.context
        predicted_posterior_map = np.asarray(spatial_values[1], dtype=float)
        fisher_map = np.asarray(spatial_values[5], dtype=float)
        posterior_mass = float(predicted_posterior_map.sum())
        fisher_context_score = float(
            np.sum(predicted_posterior_map * fisher_map) / posterior_mass
        )
        if self.schema_version in {8, 9, 10, 11, 12, 13, 14, 15, 16}:
            spatial_values, context_values = project_feature_schema(
                spatial_values, context_values, self.feature_schema
            )
        spatial = torch.from_numpy(spatial_values[None]).to(self.device)
        context = (context_values - self.context_mean) / self.context_scale
        context_tensor = torch.from_numpy(context[None].astype(np.float32)).to(self.device)
        started = time.perf_counter_ns()
        with torch.no_grad():
            prediction = self.model(spatial, context_tensor)
            if self.schema_version in {7, 8}:
                if self.target_stats is None:
                    raise ValueError("task-utility checkpoint has no target normalization")
                probability = torch.sigmoid(prediction["success_logits"])[0].cpu().numpy()
                cost_stats = self.target_stats["operational_cost_log"]
                cost_log = prediction["operational_cost_scaled"][0].cpu().numpy() * float(
                    cost_stats["scale"]
                ) + float(cost_stats["mean"])
                operational_cost = np.maximum(0.0, np.expm1(cost_log))
                if self.schema_version == 7:
                    g_stats = self.target_stats["normalized_g"]
                    normalized_g = prediction["normalized_g_scaled"][0].cpu().numpy() * float(
                        g_stats["scale"]
                    ) + float(g_stats["mean"])
                else:
                    preference_stats = self.target_stats["preference_per_depth"]
                    epistemic_stats = self.target_stats["epistemic_per_depth"]
                    preference_per_depth = prediction["preference_per_depth_scaled"][
                        0
                    ].cpu().numpy() * float(preference_stats["scale"]) + float(
                        preference_stats["mean"]
                    )
                    epistemic_per_depth = prediction["epistemic_per_depth_scaled"][
                        0
                    ].cpu().numpy() * float(epistemic_stats["scale"]) + float(
                        epistemic_stats["mean"]
                    )
                    normalized_g = preference_per_depth + epistemic_per_depth
            elif self.schema_version == 9:
                if self.target_stats is None:
                    raise ValueError("inference-value checkpoint has no target normalization")
                preference_stats = self.target_stats["preference_per_depth"]
                epistemic_stats = self.target_stats["epistemic_per_depth"]
                preference_per_depth = prediction["preference_per_depth_scaled"][
                    0
                ].cpu().numpy() * float(preference_stats["scale"]) + float(preference_stats["mean"])
                epistemic_per_depth = prediction["epistemic_per_depth_scaled"][
                    0
                ].cpu().numpy() * float(epistemic_stats["scale"]) + float(epistemic_stats["mean"])
                normalized_g = preference_per_depth + epistemic_per_depth
            elif self.schema_version in {10, 11, 12, 13, 14, 15, 16}:
                if self.target_stats is None:
                    raise ValueError("four-term checkpoint has no target normalization")

                def unscale(name: str, key: str) -> np.ndarray:
                    stats = self.target_stats[name]
                    return prediction[key][0].cpu().numpy() * float(stats["scale"]) + float(
                        stats["mean"]
                    )

                state_accuracy = unscale("state_accuracy", "state_accuracy_scaled")
                state_complexity = unscale("state_complexity", "state_complexity_scaled")
                selector_preference_weight = utility_preference_weight
                selector_epistemic_weight = utility_epistemic_weight
                if self.schema_version == 16:
                    immediate_preference = unscale(
                        "immediate_preference", "immediate_preference_scaled"
                    )
                    preference_improvement = unscale(
                        "preference_improvement", "preference_improvement_scaled"
                    )
                    immediate_epistemic = unscale(
                        "immediate_epistemic", "immediate_epistemic_scaled"
                    )
                    epistemic_improvement = unscale(
                        "epistemic_improvement", "epistemic_improvement_scaled"
                    )
                    preference_per_depth = (
                        utility_preference_weight * immediate_preference
                        + preference_improvement_weight * preference_improvement
                    )
                    epistemic_value_per_depth = (
                        utility_epistemic_weight * immediate_epistemic
                        + epistemic_improvement_weight * epistemic_improvement
                    )
                    selector_preference_weight = 1.0
                    selector_epistemic_weight = 1.0
                else:
                    preference_per_depth = unscale(
                        "preference_per_depth", "preference_per_depth_scaled"
                    )
                    epistemic_value_per_depth = unscale(
                        "epistemic_value_per_depth", "epistemic_value_per_depth_scaled"
                    )
                if self.schema_version == 10:
                    # Version 10 stored information_gain - ambiguity. Version
                    # 11 names and stores positive state epistemic value as
                    # ambiguity - information_gain.
                    epistemic_value_per_depth = -epistemic_value_per_depth
                self.last_component_predictions = {
                    "state_accuracy": state_accuracy.copy(),
                    "state_complexity": state_complexity.copy(),
                    "preference_per_depth": preference_per_depth.copy(),
                    "epistemic_value_per_depth": epistemic_value_per_depth.copy(),
                    "task_score": (
                        state_accuracy
                        - state_complexity
                        + preference_per_depth
                        + epistemic_value_per_depth
                    ).copy(),
                }
                if self.schema_version == 16:
                    self.last_component_predictions.update(
                        {
                            "immediate_preference": immediate_preference.copy(),
                            "preference_improvement": preference_improvement.copy(),
                            "immediate_epistemic": immediate_epistemic.copy(),
                            "epistemic_improvement": epistemic_improvement.copy(),
                        }
                    )
            else:
                normalized_g = prediction["normalized_g"][0].cpu().numpy()
        model_ms = (time.perf_counter_ns() - started) / 1e6
        posterior = np.asarray(agent.filtered_posteriors[2], dtype=float)
        losses = np.asarray(
            [
                information_loss(
                    posterior,
                    allocation.resolution,
                    candidate.resolution,
                    layout.size,
                )
                for candidate in ALLOCATIONS
            ]
        )
        source_index = ALLOCATION_INDEX[allocation]
        switching_ms = np.full(len(ALLOCATIONS), fixed_switch_cost_ms, dtype=float)
        switching_ms[source_index] = 0.0
        if self.schema_version in {10, 11, 12, 13, 14, 15, 16}:
            decision = select_four_term_value_allocation(
                state_accuracy=state_accuracy,
                state_complexity=state_complexity,
                preference_per_depth=preference_per_depth,
                epistemic_value_per_depth=epistemic_value_per_depth,
                compute_ms=self.compute_ms,
                switching_ms=switching_ms,
                information_loss=losses,
                deadline_preference=deadline_preference,
                compute_preference=compute_preference,
                state_accuracy_weight=state_accuracy_weight,
                state_complexity_weight=state_complexity_weight,
                preference_weight=selector_preference_weight,
                epistemic_weight=selector_epistemic_weight,
                commit_for_depth=commit_for_depth,
                compute_cost_weight=compute_cost_weight,
                switching_cost_weight=switching_cost_weight,
                information_loss_weight=information_loss_weight,
                fisher_context_score=(
                    fisher_context_score if fisher_contextual_objective else None
                ),
                fisher_weight_min=fisher_weight_min,
                fisher_weight_max=fisher_weight_max,
                fisher_weight_power=fisher_weight_power,
            )
        elif self.schema_version == 9:
            decision = select_inference_value_allocation(
                preference_per_depth=preference_per_depth,
                epistemic_per_depth=epistemic_per_depth,
                compute_ms=self.compute_ms,
                switching_ms=switching_ms,
                information_loss=losses,
                deadline_preference=deadline_preference,
                compute_preference=compute_preference,
                preference_weight=utility_preference_weight,
                epistemic_weight=utility_epistemic_weight,
                commit_for_depth=commit_for_depth,
                compute_cost_weight=compute_cost_weight,
                switching_cost_weight=switching_cost_weight,
                information_loss_weight=information_loss_weight,
            )
        elif self.schema_version == 8:
            decision = select_decomposed_task_utility_allocation(
                success_probability=probability,
                operational_cost=operational_cost,
                preference_per_depth=preference_per_depth,
                epistemic_per_depth=epistemic_per_depth,
                compute_ms=self.compute_ms,
                switching_ms=switching_ms,
                information_loss=losses,
                deadline_preference=deadline_preference,
                compute_preference=compute_preference,
                success_weight=utility_success_weight,
                operational_cost_weight=operational_cost_weight,
                operational_cost_temperature=operational_cost_temperature,
                preference_weight=utility_preference_weight,
                epistemic_weight=utility_epistemic_weight,
                commit_for_depth=commit_for_depth,
                compute_cost_weight=compute_cost_weight,
                switching_cost_weight=switching_cost_weight,
                information_loss_weight=information_loss_weight,
            )
        elif self.schema_version == 7:
            decision = select_task_utility_allocation(
                success_probability=probability,
                operational_cost=operational_cost,
                normalized_g=normalized_g,
                compute_ms=self.compute_ms,
                switching_ms=switching_ms,
                information_loss=losses,
                deadline_preference=deadline_preference,
                compute_preference=compute_preference,
                success_weight=utility_success_weight,
                operational_cost_weight=operational_cost_weight,
                operational_cost_temperature=operational_cost_temperature,
                normalized_g_weight=utility_g_weight,
                commit_for_depth=commit_for_depth,
                compute_cost_weight=compute_cost_weight,
                switching_cost_weight=switching_cost_weight,
                information_loss_weight=information_loss_weight,
            )
        else:
            decision = select_allocation(
                normalized_g=normalized_g,
                compute_ms=self.compute_ms,
                switching_ms=switching_ms,
                information_loss=losses,
                deadline_preference=deadline_preference,
                compute_preference=compute_preference,
                commit_for_depth=commit_for_depth,
            )
        return decision, model_ms


def _episode_quantiles(instance: Any, max_steps: int) -> np.ndarray:
    seed = instance.layout.map_seed * 1_000_003 + instance.target_seed * 1009
    seed += instance.observation_seed
    return np.random.default_rng(seed).random(max_steps + 1)


def _build_agent(allocation: Allocation, layout: Any, config: ClosedLoopConfig, mos: dict) -> Any:
    agent = mos["build_mos_agent"](
        mos["MOSAgentConfig"](
            target_resolution=allocation.resolution,
            action_depth=allocation.depth,
            message_passing_iterations=config.message_passing_iterations,
            policy_workers=config.policy_workers,
        ),
        layout=layout,
    )
    agent.reset()
    return agent


class _CachedMOSAgentPool:
    """Pre-initialized allocation agents for low-latency online switching.

    MOS model parameters are fixed within an episode and PyAIF learning is
    disabled, so each allocation shell can safely retain its normalized static
    model. Re-entering an allocation overwrites all recurrent belief state via
    ``_transfer_receding_state`` rather than rebuilding likelihood matrices.
    """

    def __init__(
        self,
        *,
        initial_allocation: Allocation,
        initial_agent: Any,
        layout: Any,
        config: ClosedLoopConfig,
        mos: dict,
    ) -> None:
        self._agents = {initial_allocation: initial_agent}
        for allocation in ALLOCATIONS:
            if allocation != initial_allocation:
                self._agents[allocation] = _build_agent(allocation, layout, config, mos)

    def switch(
        self,
        *,
        source_agent: Any,
        source_allocation: Allocation,
        target_allocation: Allocation,
        layout: Any,
        executed_action: Any,
        next_time_step: int,
        mos: dict,
    ) -> Any:
        return _transfer_receding_state(
            candidate=self._agents[target_allocation],
            source_agent=source_agent,
            source_allocation=source_allocation,
            target_allocation=target_allocation,
            layout=layout,
            executed_action=executed_action,
            next_time_step=next_time_step,
            mos=mos,
        )


def run_fixed_episode(instance_seed: int, allocation: Allocation, config: ClosedLoopConfig) -> dict:
    """Run one genuinely fixed-allocation MOS episode."""

    mos = _mos_imports()
    instance = mos["sample_mos_instance"](int(instance_seed))
    quantiles = _episode_quantiles(instance, config.max_steps)
    environment = mos["MOSEnvironment"](
        layout=instance.layout,
        target=instance.target,
        observation_seed=instance.observation_seed,
    )
    observation = environment.reset(noise_quantile=float(quantiles[0]))
    agent = _build_agent(allocation, instance.layout, config, mos)
    previous_action = None
    task_cost = 0.0
    inference_ms = 0.0
    false_finds = 0
    collisions = 0
    success = False
    steps = 0
    for decision_index in range(config.max_steps):
        decision = _infer_decision(agent, observation, previous_action, decision_index, mos)
        inference_ms += decision.total_ms
        observation, success = environment.step(
            decision.action,
            noise_quantile=float(quantiles[decision_index + 1]),
        )
        false_find = observation[3] == mos["FindOutcome"].FALSE_FIND
        collision = environment.last_collision
        task_cost += _step_cost(false_find=false_find, collision=collision)
        false_finds += int(false_find)
        collisions += int(collision)
        steps += 1
        previous_action = decision.action
        if success:
            break
    if not success:
        task_cost += 2.0 * config.max_steps
    return {
        "controller": f"fixed_g{allocation.resolution}_T{allocation.depth}",
        "instance_seed": int(instance_seed),
        "success": bool(success),
        "task_cost": float(task_cost),
        "steps": steps,
        "false_finds": false_finds,
        "collisions": collisions,
        "task_inference_ms": float(inference_ms),
        "meta_inference_ms": 0.0,
        "switch_ms": 0.0,
        "total_compute_ms": float(inference_ms),
        "switches": 0,
        "decisions": steps,
        "allocation_counts": json.dumps({f"g{allocation.resolution}_T{allocation.depth}": steps}),
    }


def run_adaptive_episode(
    instance_seed: int,
    config: ClosedLoopConfig,
) -> tuple[dict, list[dict]]:
    """Run one adaptive episode whose selections alter subsequent inference."""

    mos = _mos_imports()
    instance = mos["sample_mos_instance"](int(instance_seed))
    quantiles = _episode_quantiles(instance, config.max_steps)
    environment = mos["MOSEnvironment"](
        layout=instance.layout,
        target=instance.target,
        observation_seed=instance.observation_seed,
    )
    observation = environment.reset(noise_quantile=float(quantiles[0]))
    allocation = config.initial_allocation
    agent = _build_agent(allocation, instance.layout, config, mos)
    agent_pool, agent_pool_setup_ms = _timed(
        partial(
            _CachedMOSAgentPool,
            initial_allocation=allocation,
            initial_agent=agent,
            layout=instance.layout,
            config=config,
            mos=mos,
        )
    )
    controller = NeuralMetaController(
        Path(config.checkpoint),
        device=config.device,
        torch_threads=config.torch_threads,
    )
    if config.timing_profile is not None:
        controller.compute_ms = load_timing_profile(Path(config.timing_profile))[0].astype(float)
    previous_action = None
    task_cost = 0.0
    task_inference_ms = 0.0
    meta_inference_ms = 0.0
    switch_ms = 0.0
    false_finds = 0
    collisions = 0
    switches = 0
    meta_decisions = 0
    success = False
    steps = 0
    commitment_remaining = 0
    allocation_counts = {candidate: 0 for candidate in ALLOCATIONS}
    trajectory = []

    for decision_index in range(config.max_steps):
        source_allocation = allocation
        task_decision = _infer_decision(agent, observation, previous_action, decision_index, mos)
        task_inference_ms += task_decision.total_ms
        allocation_counts[allocation] += 1
        next_position = instance.layout.move(environment.position, task_decision.action)
        invoke_metacontroller = not (config.hold_allocation_for_depth and commitment_remaining > 1)
        if invoke_metacontroller:
            meta_decision, model_ms = controller.choose(
                agent=agent,
                allocation=allocation,
                layout=instance.layout,
                selected_action=int(task_decision.action),
                next_robot_position=next_position,
                deadline_preference=config.deadline_preference,
                compute_preference=config.compute_preference,
                commit_for_depth=config.hold_allocation_for_depth,
                utility_success_weight=config.utility_success_weight,
                operational_cost_weight=config.operational_cost_weight,
                operational_cost_temperature=config.operational_cost_temperature,
                utility_g_weight=config.utility_g_weight,
                utility_preference_weight=config.utility_preference_weight,
                utility_epistemic_weight=config.utility_epistemic_weight,
                preference_improvement_weight=config.preference_improvement_weight,
                epistemic_improvement_weight=config.epistemic_improvement_weight,
                state_accuracy_weight=config.state_accuracy_weight,
                state_complexity_weight=config.state_complexity_weight,
                compute_cost_weight=config.compute_cost_weight,
                switching_cost_weight=config.switching_cost_weight,
                fixed_switch_cost_ms=config.fixed_switch_cost_ms,
                information_loss_weight=config.information_loss_weight,
                fisher_contextual_objective=config.fisher_contextual_objective,
                fisher_weight_min=config.fisher_weight_min,
                fisher_weight_max=config.fisher_weight_max,
                fisher_weight_power=config.fisher_weight_power,
            )
            target = meta_decision.allocation
            meta_inference_ms += model_ms
            meta_decisions += 1
            if config.hold_allocation_for_depth:
                commitment_remaining = target.depth
        else:
            meta_decision = None
            model_ms = 0.0
            target = allocation
            commitment_remaining -= 1
        observation, success = environment.step(
            task_decision.action,
            noise_quantile=float(quantiles[decision_index + 1]),
        )
        false_find = observation[3] == mos["FindOutcome"].FALSE_FIND
        collision = environment.last_collision
        step_cost = _step_cost(false_find=false_find, collision=collision)
        task_cost += step_cost
        false_finds += int(false_find)
        collisions += int(collision)
        steps += 1
        actual_switch_ms = 0.0
        if not success and decision_index + 1 < config.max_steps:
            if target != allocation:
                agent, actual_switch_ms = _timed(
                    partial(
                        agent_pool.switch,
                        source_agent=agent,
                        source_allocation=allocation,
                        target_allocation=target,
                        layout=instance.layout,
                        executed_action=task_decision.action,
                        next_time_step=decision_index + 1,
                        mos=mos,
                    )
                )
                switch_ms += actual_switch_ms
                switches += 1
            allocation = target
        trajectory.append(
            {
                "instance_seed": int(instance_seed),
                "decision": decision_index,
                "source_resolution": source_allocation.resolution,
                "source_depth": source_allocation.depth,
                "action": task_decision.action.name,
                "metacontroller_invoked": invoke_metacontroller,
                "selected_resolution": target.resolution,
                "selected_depth": target.depth,
                "selection_reason": (
                    "held for selected planning depth"
                    if meta_decision is None
                    else "maximum joint meta-objective"
                ),
                "predicted_normalized_g": (
                    None if meta_decision is None else meta_decision.predicted_normalized_g
                ),
                "predicted_success_probability": (
                    None if meta_decision is None else meta_decision.predicted_success_probability
                ),
                "predicted_operational_cost": (
                    None if meta_decision is None else meta_decision.predicted_operational_cost
                ),
                "predicted_preference_per_depth": (
                    None if meta_decision is None else meta_decision.predicted_preference_per_depth
                ),
                "predicted_epistemic_per_depth": (
                    None if meta_decision is None else meta_decision.predicted_epistemic_per_depth
                ),
                "predicted_epistemic_value_per_depth": (
                    None
                    if meta_decision is None
                    else meta_decision.predicted_epistemic_value_per_depth
                ),
                "predicted_state_accuracy": (
                    None if meta_decision is None else meta_decision.predicted_state_accuracy
                ),
                "predicted_state_complexity": (
                    None if meta_decision is None else meta_decision.predicted_state_complexity
                ),
                "task_utility_score": (
                    None if meta_decision is None else meta_decision.task_utility_score
                ),
                "predicted_inference_ms": (
                    None if meta_decision is None else meta_decision.predicted_compute_ms
                ),
                "predicted_switch_ms": (
                    None if meta_decision is None else meta_decision.switching_ms
                ),
                "predicted_mean_compute_ms": (
                    None if meta_decision is None else meta_decision.mean_compute_ms
                ),
                "deadline_survival_probability": (
                    None if meta_decision is None else meta_decision.deadline_survival_probability
                ),
                "compute_surprisal_nats": (
                    None if meta_decision is None else meta_decision.compute_surprisal_nats
                ),
                "compute_preference_cost_nats": (
                    None
                    if meta_decision is None
                    else meta_decision.compute_preference_cost_nats
                ),
                "normalized_compute_cost": (
                    None if meta_decision is None else meta_decision.normalized_compute_cost
                ),
                "switching_penalty": (
                    None if meta_decision is None else meta_decision.switching_penalty
                ),
                "information_loss_normalized": (
                    None if meta_decision is None else meta_decision.information_loss_normalized
                ),
                "information_loss_nats": (
                    None if meta_decision is None else meta_decision.information_loss_nats
                ),
                "objective_score": (
                    None if meta_decision is None else meta_decision.objective_score
                ),
                "fisher_context_score": (
                    None if meta_decision is None else meta_decision.fisher_context_score
                ),
                "fisher_context_weight": (
                    None if meta_decision is None else meta_decision.fisher_context_weight
                ),
                "actual_task_inference_ms": task_decision.total_ms,
                "actual_meta_inference_ms": model_ms,
                "actual_switch_ms": actual_switch_ms,
                "step_cost": step_cost,
                "success": bool(success),
            }
        )
        previous_action = task_decision.action
        if success:
            break
    if not success:
        task_cost += 2.0 * config.max_steps
    total_compute = task_inference_ms + meta_inference_ms + switch_ms
    return (
        {
            "controller": "adaptive",
            "instance_seed": int(instance_seed),
            "success": bool(success),
            "task_cost": float(task_cost),
            "steps": steps,
            "false_finds": false_finds,
            "collisions": collisions,
            "task_inference_ms": float(task_inference_ms),
            "meta_inference_ms": float(meta_inference_ms),
            "switch_ms": float(switch_ms),
            "agent_pool_setup_ms": float(agent_pool_setup_ms),
            "total_compute_ms": float(total_compute),
            "switches": switches,
            "meta_decisions": meta_decisions,
            "decisions": steps,
            "allocation_counts": json.dumps(
                {
                    f"g{candidate.resolution}_T{candidate.depth}": count
                    for candidate, count in allocation_counts.items()
                    if count
                },
                sort_keys=True,
            ),
            "allocation_sequence": json.dumps(
                [f"g{row['source_resolution']}_T{row['source_depth']}" for row in trajectory]
            ),
        },
        trajectory,
    )


def _run_instance(instance_seed: int, config: ClosedLoopConfig) -> dict:
    adaptive, trajectory = run_adaptive_episode(instance_seed, config)
    episodes = [adaptive]
    if config.include_fixed:
        episodes.extend(
            run_fixed_episode(instance_seed, allocation, config) for allocation in ALLOCATIONS
        )
    return {"episodes": episodes, "trajectory": trajectory}


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summary(episodes: list[dict]) -> dict[str, Any]:
    controllers = sorted({row["controller"] for row in episodes})
    result = {}
    for controller in controllers:
        rows = [row for row in episodes if row["controller"] == controller]
        summary = {
            "episodes": len(rows),
            "success_rate": float(np.mean([row["success"] for row in rows])),
            "mean_task_cost": float(np.mean([row["task_cost"] for row in rows])),
            "mean_steps": float(np.mean([row["steps"] for row in rows])),
            "mean_total_compute_ms": float(np.mean([row["total_compute_ms"] for row in rows])),
            "median_total_compute_ms": float(np.median([row["total_compute_ms"] for row in rows])),
            "mean_switches": float(np.mean([row["switches"] for row in rows])),
        }
        if controller == "adaptive":
            allocation_steps: dict[str, int] = {}
            for row in rows:
                for name, count in json.loads(row["allocation_counts"]).items():
                    allocation_steps[name] = allocation_steps.get(name, 0) + int(count)
            meta_decisions = sum(int(row.get("meta_decisions", row["decisions"])) for row in rows)
            summary["mean_meta_decisions"] = float(meta_decisions / len(rows))
            summary["mean_agent_pool_setup_ms"] = float(
                np.mean([row.get("agent_pool_setup_ms", 0.0) for row in rows])
            )
            summary["allocation_steps"] = allocation_steps
            if len(rows) == 1 and "allocation_sequence" in rows[0]:
                summary["allocation_sequence"] = json.loads(rows[0]["allocation_sequence"])
        result[controller] = summary
    return result


def _paired_comparisons(episodes: list[dict]) -> dict[str, Any]:
    adaptive = {
        int(row["instance_seed"]): row for row in episodes if row["controller"] == "adaptive"
    }
    rng = np.random.default_rng(0)
    comparisons = {}
    for controller in sorted({row["controller"] for row in episodes} - {"adaptive"}):
        fixed = {
            int(row["instance_seed"]): row for row in episodes if row["controller"] == controller
        }
        seeds = sorted(set(adaptive) & set(fixed))
        metrics = {}
        for name in ("task_cost", "success", "steps", "total_compute_ms"):
            delta = np.asarray(
                [float(adaptive[seed][name]) - float(fixed[seed][name]) for seed in seeds]
            )
            if delta.size == 1:
                confidence = [float(delta[0]), float(delta[0])]
            else:
                bootstrap = np.asarray(
                    [rng.choice(delta, size=delta.size, replace=True).mean() for _ in range(5000)]
                )
                confidence = np.percentile(bootstrap, (2.5, 97.5)).tolist()
            metrics[f"adaptive_minus_fixed_{name}"] = {
                "mean": float(delta.mean()),
                "ci95": [float(value) for value in confidence],
            }
        comparisons[controller] = metrics
    return comparisons


def evaluate_closed_loop(
    *,
    instance_seeds: list[int] | tuple[int, ...],
    output_dir: Path,
    config: ClosedLoopConfig,
    instance_workers: int = 1,
    resume: bool = True,
) -> dict[str, Any]:
    """Run resumable per-instance adaptive and fixed closed-loop episodes."""

    seeds = tuple(int(value) for value in instance_seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("instance_seeds must be nonempty and unique")
    if instance_workers < 1:
        raise ValueError("instance_workers must be positive")
    output_dir = Path(output_dir)
    shard_dir = output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(config.checkpoint)
    signature = {
        "configuration": asdict(config),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
    }
    pending = []
    for seed in seeds:
        path = shard_dir / f"instance-{seed}.json"
        compatible = False
        if resume and path.is_file():
            try:
                compatible = (
                    json.loads(path.read_text(encoding="utf-8")).get("signature") == signature
                )
            except (OSError, json.JSONDecodeError):
                compatible = False
        if compatible:
            print(f"[reuse] instance={seed}", flush=True)
        else:
            pending.append(seed)

    def save(seed: int, payload: dict) -> None:
        payload["signature"] = signature
        temporary = shard_dir / f"instance-{seed}.json.tmp"
        temporary.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        temporary.replace(shard_dir / f"instance-{seed}.json")

    if instance_workers == 1:
        for completed, seed in enumerate(pending, start=1):
            save(seed, _run_instance(seed, config))
            print(f"[evaluate {completed}/{len(pending)}] instance={seed}", flush=True)
    elif pending:
        with ProcessPoolExecutor(max_workers=instance_workers) as executor:
            futures = {executor.submit(_run_instance, seed, config): seed for seed in pending}
            for completed, future in enumerate(as_completed(futures), start=1):
                seed = futures[future]
                save(seed, future.result())
                print(f"[evaluate {completed}/{len(pending)}] instance={seed}", flush=True)

    payloads = [
        json.loads((shard_dir / f"instance-{seed}.json").read_text(encoding="utf-8"))
        for seed in seeds
    ]
    episodes = [row for payload in payloads for row in payload["episodes"]]
    trajectory = [row for payload in payloads for row in payload["trajectory"]]
    _write_csv(output_dir / "episodes.csv", episodes)
    _write_csv(output_dir / "adaptive_trajectory.csv", trajectory)
    report = {
        "instance_seeds": list(seeds),
        "instance_workers": instance_workers,
        "configuration": asdict(config),
        "controllers": _summary(episodes),
        "paired_comparisons": _paired_comparisons(episodes),
    }
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
