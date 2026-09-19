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
       success probability + relative task regret

candidate allocation + measured machine profile
                    |
                    v
          independent compute model
                    |
              predicted latency

task predictions + compute deadline
+ transition switching time + information loss
                    |
                    v
       constrained allocation selector
```

The task model never receives CPU load merely to learn a mixed task/resource
score. It predicts task consequences. A separately calibrated compute model
predicts latency, while switching time and resolution-induced information loss
remain explicit and auditable.

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
entropy, the best-policy probability margin, and the best-versus-second-best
expected-free-energy gap.

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

## Constrained selection

The selector does not add quantities with incompatible units. It first requires:

```text
predicted success >= reliability threshold
steady inference + switching time <= compute deadline
resolution-switch information loss <= information-loss limit
```

It then chooses the feasible allocation with the lowest predicted task cost. If
none is feasible, it selects the highest predicted-success allocation available
under the deadline.

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
4. executes each distinct candidate physical action once from the matched state;
5. hands control back to a clone of the reference controller; and
6. records success and remaining task cost for every allocation.

Candidates that select the same physical action share the same task outcome.
This avoids repeating identical environment rollouts while preserving separate
inference-time measurements. Agent construction, posterior remapping, and other
switch overhead are excluded from `compute_ms` and separately recorded in
`switch_ms` for the explicit switching-cost model. The no-change diagonal is
zero: rebuilding an identical agent is needed only for counterfactual isolation,
not during deployment where the existing agent is carried forward.

The output directory contains:

- `training_data.npz`: tensors and 12-way labels used for training;
- `contexts.csv`: feature/label timing and state provenance;
- `branches.csv`: one record per distinct physical-action intervention;
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

Splitting is performed by MOS instance, not by individual decision context.
Every state from one map/target/sensor trajectory therefore belongs entirely to
the training, validation, or test set. The saved split is recorded in both the
checkpoint and metrics report.

The trainer standardizes nonspatial context using training-set statistics. It
optimizes binary success prediction, log-scale relative regret
`J(gamma,T) - min J`, and a listwise ranking loss over the 12 candidates. This
targets the allocation decision directly instead of requiring the network to
learn instance-dependent absolute cost offsets. Validation-based early stopping
and allocation-level evaluation are applied on all three splits. Test metrics
include:

- success Brier score and classification accuracy;
- relative-cost MAE and RMSE;
- realized success and task cost of the network-selected allocation;
- regret relative to the matched counterfactual oracle; and
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

## Current status

The repository currently implements:

- mass-preserving canonical belief transformations;
- normalized native-resolution entropy;
- Jensen-Shannon information-loss measurement;
- canonical visibility and categorical Fisher maps;
- spatial and nonspatial feature assembly;
- a small optional PyTorch success/relative-regret network with ranking loss;
- a separate profiled compute-cost model;
- a transition-specific switching matrix; and
- constrained allocation selection;
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
