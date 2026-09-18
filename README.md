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
       success probability + task cost

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

## Current status

The initial scaffold implements:

- mass-preserving canonical belief transformations;
- normalized native-resolution entropy;
- Jensen-Shannon information-loss measurement;
- canonical visibility and categorical Fisher maps;
- spatial and nonspatial feature assembly;
- a small optional PyTorch success/task-cost network;
- a separate profiled compute-cost model;
- a transition-specific switching matrix; and
- constrained allocation selection.

The next stage is an adapter that extracts these inputs at matched MOS decision
states, followed by counterfactual data generation for all 12 allocations. No
trained controller or performance claim is included yet.

## Development

```bash
ruff check .
ruff format --check .
pytest -q
python -m build
```
