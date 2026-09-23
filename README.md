# Active-Inference Neural Metacontrol

A research implementation of belief-conditioned neural resource allocation for
Active Inference. The metacontroller selects the task agent's spatial target
resolution `gamma` and receding-horizon action depth `T` in a 2D
Multi-Object Search domain.

This repository is deliberately separate from:

- [PyAIF](https://github.com/Diluna98/python_active_inference), which supplies
  generic state and policy inference;
- [Active-Inference Navigation Agent](https://github.com/Diluna98/active-inference-navigation-agent),
  which supplies the MOS task and fixed `(gamma, T)` benchmarks; and
- the earlier hierarchical Active-Inference meta-controller, which remains an
  interpretable experimental baseline.

## Design

The task-performance network and resource-cost model are separate:

```text
current and predicted task beliefs
geometry, visibility, Fisher information
policy confidence and current allocation
                    |
                    v
       small CNN/MLP task model
                    |
       normalized selected-policy G

candidate allocation + measured machine profile
                    |
                    v
          independent compute model
                    |
              predicted latency

predicted G/T + inference + switching/T
+ deadline survival + JS loss in nats
                    |
                    v
          joint-objective selector
```

The task model never receives CPU load merely to learn a mixed task/resource
score. It predicts the local policy score produced at the next inference. A
separately calibrated compute model predicts latency, while switching time and
resolution-induced information loss remain explicit and auditable.

## Allocation space

The initial benchmark uses 12 allocations:

```text
gamma in {2, 5, 10, 20}
T     in {1, 2, 3}
```

`T` is the number of future actions in the MOS diagnostic. The corresponding
PyAIF state horizon is `T + 1`.

## Fixed-size neural input

Every spatial input uses a canonical `20 x 20` grid:

1. current object posterior;
2. predicted next object posterior;
3. predicted next robot position;
4. known obstacle map;
5. detection/visibility probability from the predicted position; and
6. categorical Fisher-information map from the predicted position.

Variable-resolution posteriors are expanded while conserving probability mass.
The current resolution is also provided explicitly. A coarse-to-fine expansion
therefore represents unresolved probability uniformly; it does not claim to
recover missing spatial information.

The nonspatial context contains the selected physical action, one-hot current
`gamma` and `T`, found-object flags, normalized native belief entropy, policy
entropy, and the best-policy probability margin.

For a static target, the ordinary transition prediction may make the predicted
target posterior identical to the current posterior. This is intentionally
testable: the channel can be removed by ablation when it is redundant. It is
retained in the interface for moving-object or genuinely policy-conditioned
future-belief experiments.

## Fisher-information reference

The Fisher map is always evaluated on the canonical `20 x 20` grid from the
known sensor model, obstacle geometry, and predicted robot position. It does not
use the true target location or a future observation. It tells the controller
where the next sensor configuration would be locally sensitive to target
position. Maps can be precomputed once per environment and retrieved by robot
position at runtime.

## Joint meta-objective

The controller selects a resolution-depth pair by maximizing

```text
predicted task value
  - compute weight * ICRA-style latency-preference cost in nats
  - switching weight * indicator(allocation changed)
  - information-loss weight * Jensen-Shannon information loss in nats
```

PyAIF's convention places `+ gamma * G` in the policy softmax, so larger task
values are preferred. The task planner remains receding-horizon and repeats
inference at every physical step. Every allocation change receives the same
literal penalty, independently of its measured construction time. As in the
ICRA controller, latency preference has a linear cost over the full scale and
an additional quadratic cost above a configurable comfort point:

```text
C_compute(c) = w_linear * c / deadline
             + w_excess * max((c - comfort) / (deadline - comfort), 0)^2
```

This is the negative unnormalised log preference; normalising `exp(-C_compute)`
on a shared latency grid only adds the same constant to every candidate. The
default comfort point is 75% of `--deadline-median-ms`, and the scale can be
controlled independently with `--compute-preference-comfort-ms` and
`--compute-preference-deadline-ms`. Log-normal deadline survival and its
surprisal are still reported as diagnostics, but they no longer enter the
objective. There are no hard feasibility constraints, and neural
metacontroller latency is reported but excluded because it is already incurred
and is identical across candidate allocations.

## Installation

Install the NumPy-only feature and selection core:

```bash
python -m pip install -e .
```

Install the optional PyTorch model and development tools:

```bash
python -m pip install -e ".[neural,dev]"
```

The MOS adapter is intentionally an integration dependency rather than a core
package dependency. Install the benchmark repository alongside this one:

```bash
python -m pip install "active-inference-navigation-agent @ git+https://github.com/Diluna98/active-inference-navigation-agent.git@d749734fefcfcb0a9541633f2be0bd18a29a1d91"
```

## Matched-state MOS data

Generate labels for all 12 candidate allocations from the same decision states:

```bash
active-inference-mos-counterfactuals \
  --instance-seeds 0 1 2 3 \
  --reference-resolution 20 \
  --reference-depth 1 \
  --max-steps 50 \
  --instance-workers 4 \
  --output-dir results/mos_counterfactuals
```

For scientific training, vary the source allocation as well as evaluating all
12 candidates. On PowerShell, this command assigns the 200 instances
round-robin across all 12 source `(gamma, T)` configurations:

```powershell
active-inference-mos-counterfactuals `
  --instance-seeds (1000..1199) `
  --balanced-sources `
  --max-steps 50 `
  --instance-workers 4 `
  --output-dir results/mos_balanced_200
```

Thus the input belief is no longer always produced by `(gamma=20, T=1)`.
Source allocation is retained in every shard and context, and train/validation/
test splitting is stratified by source allocation. Each source requires at
least three MOS instances so all three splits can contain it.

For each reference action at step `t`, the generator:

1. maps the reference posterior into every candidate resolution without changing
   its probability mass;
2. gives every candidate the same executed action and observation at `t + 1`;
3. times state inference, policy evaluation, and action selection for every
   `(gamma, T)` candidate;
4. reads the selected policy's raw `G`, risk, ambiguity, and parameter-information
   gain from PyAIF's policy-inference decomposition;
5. divides the resulting selected-policy `G` by planning depth `T`; and
6. by default, stores that one-step value as the training target.

For MOS, the normalized target is

```text
(risk + ambiguity - information_gain) / T
```

PyAIF's discrete `ambiguity` diagnostic is the positive state-observation mutual
information term `H[Q(o)] - E[H[P(o|s)]]`; parameter-information gain is normally
zero in the fixed-model MOS benchmark. No `log(gamma^2)` division is applied:
the pilot showed that native ambiguity per step was already similar across
resolutions and state-space division artificially favoured coarse models.

Add `--rollout-horizon 5` for the accumulated target. Each candidate then
controls its own copied agent and environment for up to five physical steps.
The stored target is the discounted mean of its successive `G/T` values (set
`--rollout-discount`, default 1.0). This is schema v5; the default one-step
dataset remains schema v4. The immediate `G/T`, rollout actions, success,
steps, and realized task cost remain in `branches.csv` for auditing. No future
reference-controller outcome or failure penalty is attributed to the candidate.
Agent construction, posterior remapping, and other
switch overhead are excluded from `compute_ms` and separately recorded in
`switch_ms` for the explicit switching-cost model. The no-change diagonal is
zero: rebuilding an identical agent is needed only for counterfactual isolation,
not during deployment where the existing agent is carried forward.

The output directory contains:

- `training_data.npz`: tensors and 12-way labels used for training;
- `contexts.csv`: feature/label timing and state provenance;
- `branches.csv`: one auditable policy-diagnostic record per candidate allocation;
- `reference_trajectory.csv`: the trajectory that defined matched states; and
- `summary.json`: array shapes, counts, and allocation order.

`--branch-stride N` evaluates every Nth context when producing an inexpensive
pilot dataset. Use stride 1 for the final dataset.

Generation is resumable by default. Each MOS instance is written independently
under `OUTPUT_DIR/shards/instance-SEED`, and a completion marker records the
exact generation configuration. Repeating the same command reuses compatible
completed shards. Missing, interrupted, or configuration-mismatched shards are
generated again before the final NPZ and CSV files are assembled in seed order.
Use `--no-resume` to deliberately regenerate every requested instance.

`--instance-workers` evaluates independent MOS instances in separate processes.
Start with 2–4 workers, depending on available memory and physical CPU cores.
Keep `--policy-workers 1` during instance-level parallel runs to avoid nested
process oversubscription. If a run is interrupted, execute the identical command
again; completed instances will print `[reuse]` and will not be recomputed.

Before a long research-archive run, generate and audit a 100-instance pilot.
These seeds do not overlap with the original 8200--8399 archive. Balanced
sources vary the controller that creates the belief history while every saved
context still evaluates all 12 candidate allocations:

```powershell
cd "C:\path\to\active-inference-neural-metacontrol"
& ".\.venv\Scripts\python.exe" -m active_inference_neural_metacontrol.cli `
  --instance-seeds (10000..10099) `
  --balanced-sources `
  --max-steps 50 `
  --branch-stride 1 `
  --message-passing-iterations 10 `
  --policy-workers 1 `
  --instance-workers 4 `
  --collect-task-outcomes `
  --collect-research-archive `
  --output-dir results/mos_research_pilot_100
```

Derive the per-future-step policy values and audit whether the pilot contains
configuration-insensitive, resolution-only, depth-only, and joint-critical
contexts:

```powershell
& ".\.venv\Scripts\python.exe" scripts/derive_step_g_dataset.py `
  --source results/mos_research_pilot_100 `
  --output results/mos_research_pilot_100_step_g

& ".\.venv\Scripts\python.exe" scripts/audit_configuration_coverage.py `
  --dataset-dir results/mos_research_pilot_100_step_g `
  --output-dir results/mos_research_pilot_100_coverage `
  --margin-nats 0.001
```

The audit calls a dimension critical only when its task-score improvement
exceeds the declared margin and changes the first physical action. It reports
score-only differences separately, counts the number of distinct instances
supporting each regime, and compares eventual outcomes when those labels are
available. Do not start the 2,000-instance archive until the depth-only and
joint regimes recur across multiple independent instances.

### Visualize a planning-depth disagreement

Plot the posterior, posterior-weighted Fisher-information landscape, selected
policy paths, and PyAIF value decomposition for a matched context:

```powershell
python scripts/plot_depth_disagreement.py `
  --instance-seed 4008 --decision 6 --resolution 5 `
  --source-resolution 10 --source-depth 3 `
  --output results/mos_normalized_g_pilot_12/depth_disagreement_seed4008_d6_g5.png
```

The companion JSON records the complete selected action sequences, paths, raw
`G`, preference per step, ambiguity per step, and `G/T`. The true target is shown
only for retrospective validation and is never supplied to either agent.

### Diagnose accumulated policy value

Before replacing the one-step training target, compare it with a
candidate-controlled multi-step target. Every allocation receives the same
starting state and sensor quantiles, controls its own five-step receding-horizon
rollout, and accumulates its selected-policy `G/T`:

```powershell
active-inference-diagnose-g-rollouts `
  --instance-seeds (4000..4009) `
  --max-reference-steps 20 `
  --branch-stride 5 `
  --rollout-horizon 5 `
  --output-dir results/accumulated_g_diagnostic
```

The diagnostic reports how often the one-step and accumulated-`G/T` oracles
choose different allocations or first actions, plus their realized success and
task cost within the common rollout horizon.

Generate a deliberately small schema-v5 pipeline test before committing to a
large run:

```powershell
active-inference-mos-counterfactuals `
  --instance-seeds (5000..5011) `
  --balanced-sources `
  --max-steps 6 `
  --branch-stride 5 `
  --rollout-horizon 5 `
  --rollout-discount 1.0 `
  --message-passing-iterations 3 `
  --policy-workers 1 `
  --instance-workers 4 `
  --output-dir results/mos_accumulated_g_h5_pilot_12
```

This produces only one context per source allocation and is a software smoke
test, not a scientifically useful training set.

## Train the task-performance model

Install the neural extra and train from a generated dataset:

```bash
python -m pip install -e ".[neural,dev]"

active-inference-train-metacontroller \
  --dataset-dir results/mos_pilot \
  --output-dir results/mos_pilot_model \
  --timing-profile results/mos_timing_profile/timing_profile.json \
  --epochs 200 \
  --batch-size 32 \
  --patience 25 \
  --seed 0
```

For the 12-instance smoke dataset, add `--no-stratify-by-source`, because one
instance per source is insufficient for source-stratified train/validation/test
splits. Keep the default stratification for the large dataset, with at least
three instances per source allocation.

For repeated neural initializations on exactly the same data split, keep
`--split-seed` fixed while changing `--seed`. This separates model variance
from train/test-partition variance.

Splitting is performed by MOS instance, not by individual decision context.
Every state from one map/target/sensor trajectory therefore belongs entirely to
the training, validation, or test set. The saved split is recorded in both the
checkpoint and metrics report.

The trainer standardizes nonspatial context using training-set statistics. It
optimizes normalized selected-policy `G` regression and a listwise ranking loss
over the 12 candidates. Validation-based early stopping
and allocation-level evaluation are applied on all three splits. Test metrics
include:

- normalized-`G` MAE and RMSE;
- realized normalized `G` of the network-selected allocation;
- normalized-`G` regret relative to the matched-state oracle; and
- all 12 fixed-allocation baselines.

Use `--compute-budget-ms` to exclude allocations whose controlled median
inference time exceeds a deadline. With `--timing-profile`, the checkpoint also
stores the complete controlled source-to-target switching matrix. Without that
option, training falls back to the training-set median inference time and does
not claim a calibrated switching matrix. Computation remains independently
measured and is not learned by the task network.

Training writes `best_model.pt`, `history.csv`, `metrics.json`, and
`predictions.npz`. The 10-instance pilot is suitable only for an end-to-end
sanity check; scientific training requires a larger instance-disjoint corpus.

## Controlled timing profile

Do not use parallel data-generation timings as a machine profile. Generate a
small balanced corpus with one instance worker, then summarize both steady
inference latency and the full source-to-target switching matrix:

```powershell
active-inference-mos-counterfactuals `
  --instance-seeds (2000..2023) `
  --balanced-sources `
  --max-steps 20 `
  --branch-stride 5 `
  --instance-workers 1 `
  --output-dir results/mos_timing_profile

active-inference-profile-metacontrol `
  --dataset-dir results/mos_timing_profile `
  --output results/mos_timing_profile/timing_profile.json
```

The profiler refuses multi-worker measurements by default because concurrent
instances distort latency. Its JSON output reports median and p95 inference
time for every candidate and median/p95 switching time for every observed
source-to-target pair.

## Closed-loop evaluation

Matched-state labels are useful for training, but they do not establish that an
adaptive controller works when its choices alter later beliefs. Run the
closed-loop benchmark on held-out MOS seeds:

```powershell
active-inference-evaluate-closed-loop `
  --checkpoint results/mos_g_per_t_balanced_400_model_seed0_temp01_profiled/best_model.pt `
  --instance-seeds (3000..3029) `
  --deadline-median-ms 100 `
  --deadline-log-sigma 0.5 `
  --compute-preference-comfort-ms 600 `
  --compute-preference-deadline-ms 800 `
  --compute-preference-linear-weight 1 `
  --compute-preference-excess-weight 2 `
  --initial-resolution 5 `
  --initial-depth 2 `
  --hold-allocation-for-depth `
  --max-steps 50 `
  --instance-workers 4 `
  --output-dir results/mos_closed_loop_joint
```

At every decision, the current task agent performs state and policy inference
and chooses the physical action. The neural controller then selects the
allocation for the next decision from the predicted next belief. After the
observation arrives, the posterior is transferred into the selected
representation, making that allocation the source for the following step.
All 12 allocation shells are initialized before the timed control loop. Their
static likelihoods, transitions, policies, and normalized generative arrays
are retained, so an online switch transfers only the recurrent posterior,
previous action, clock, and receding-inference stage. `agent_pool_setup_ms` is
reported separately; `switch_ms` and `total_compute_ms` contain online work.
Staying at the current allocation carries the existing agent forward with no
switch cost. With `--hold-allocation-for-depth`, a selected `(gamma, T)` is
used for `T` physical decisions before the metacontroller selects again, which
matches the switching-cost amortization in the joint objective.

By default, the same instance and common random sensor sequence are also run
through all 12 fixed allocations. Evaluation is resumable per instance and
safe against configuration or checkpoint changes. Use `--adaptive-only` for a
quick controller smoke test. Keep `--policy-workers 1` when using multiple
instance workers. Keep `--torch-threads 1` as well; the network is small, and
multiple PyTorch thread pools otherwise oversubscribe the CPU and distort its
measured overhead.

The output contains:

- `episodes.csv`: task, computation, and switching outcomes for every controller;
- `adaptive_trajectory.csv`: every neural allocation decision and realized timing;
- `summary.json`: controller aggregates plus paired bootstrap confidence intervals; and
- `shards/`: resumable per-instance results.

## Current status

The repository currently implements:

- mass-preserving canonical belief transformations;
- normalized native-resolution entropy;
- Jensen-Shannon information-loss measurement;
- canonical visibility and categorical Fisher maps;
- spatial and nonspatial feature assembly;
- a small optional PyTorch normalized-`G` network with ranking loss;
- a separate profiled compute-cost model;
- a transition-specific switching matrix; and
- joint task-quality, compute, switching, and information-loss selection;
- a live MOS-to-neural feature adapter; and
- parallel, resumable matched-state generation for all 12 allocations;
- balanced source-allocation scheduling and stratified splitting; and
- controlled inference and switching-time profiling.

No trained controller or performance claim is included yet. The next stage is
to generate a sufficiently broad training/validation corpus, train the task
model, and compare it against fixed allocations and the interpretable
Active-Inference metacontroller baseline.

## Development

```bash
ruff check .
ruff format --check .
pytest -q
python -m build
```
