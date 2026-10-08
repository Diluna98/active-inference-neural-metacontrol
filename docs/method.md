# Implementation specification

## Task and timing

The benchmark contains a stationary hidden target, known obstacles and exact
robot position. Target detection is binary, noisy, distance-dependent and
attenuated by occlusion. A Find action succeeds at a viewpoint satisfying the
environment's visibility rule. Robot x/y, detection, Find outcome and collision
are categorical observation modalities.

PyAIF uses filtered current-state inference and receding-horizon policy
evaluation. Target resolution gamma changes the gamma-squared target factor;
robot factors remain at physical-grid resolution. Depth T evaluates T future
actions (the PyAIF state horizon is T+1). Only the first action is executed.

After that action and observation, the next allocation is chosen before the new
observation is assimilated. The context includes the preceding posterior and
policy statistics, the reached viewpoint and the realized observation. Hidden
target truth is not an input to the deployed controller.

## Network inputs and outputs

The released checkpoints expect **6 x 20 x 20 spatial maps**:

1. current target posterior on the canonical grid;
2. predicted target posterior on that grid;
3. reached robot position (one-hot);
4. known binary obstacle map;
5. detection probabilities from the reached viewpoint;
6. finite-difference Fisher map, normalized by its viewpoint maximum.

For stationary targets, channels 1 and 2 coincide. The duplicate is retained in
the trained checkpoint interface; it contributes no additional sensor evidence.
Coarse probabilities are projected uniformly by cell-overlap weights.

The meta-regret network uses **63 nonspatial inputs**: five action indicators,
one found flag, three belief/policy statistics, 47 observation indicators
(20 x-position, 20 y-position, 2 detection, 3 Find-outcome, 2 collision), and seven
source-allocation indicators (four resolutions, three depths). Context features
are standardized using training-split statistics. The underlying meta-Q feature
schema has 56 inputs; the regret model appends the seven allocation indicators.

Spatial encoder: Conv2d(6,16,3,padding=1), ReLU, MaxPool2d(2),
Conv2d(16,32,3,padding=1), ReLU, AdaptiveAvgPool2d(5,5).
Flattened spatial features and the context pass through MLP widths 96 and 64
with ReLU. The output heads are a scalar reference value V and 12 advantages A:

```text
Q(z,m) = V(z) + A(z,m) - A(z,m_ref),   m_ref = (5,2).
```

The two heads are not separate task and compute predictions. Together they
produce 12 operational task-return estimates.

## Offline learning

Each recorded context branches over all allocations. A branch transfers the
belief, assimilates the available observation, performs policy evaluation and
executes one physical action in an isolated environment copy. Branches share
an observation-noise quantile. Sticky exploration chooses the continuation
branch used to collect the next context; branching is one step, not an
exponential 12-way tree.

Reward is -(1 + 5*false_find + 0.5*collision), with a 100-unit timeout penalty.
Successful discovery terminates without that penalty. Bellman targets use the
maximum next allocation value with discount 0.98 and terminal masking.
Training combines smooth-L1 absolute and reference-relative losses with a
pairwise ranking loss. Weights are (0.25,1,0.25); near-tie threshold is 0.01.
Target weights are copied every 100 gradient updates and at epoch boundaries. Each of three seeds uses
100 epochs, batch size 128 and Adam learning rate 0.0003.

Instances are partitioned into training/validation/test splits (70/15/15);
transitions from one instance cannot cross splits. The reference corpus has
41,689 contexts and 500,268 candidate branches from 1,180 instances containing
transitions. Generation requested 1,200 seeds (14000-15199); 20 generated
instances contributed no nonterminal branch contexts.

## Runtime objective

Each network's outputs are centered on reference allocation (5,2). The
conservative score is mean relative Q minus beta times its population ensemble
standard deviation, with beta=0.5. Regret is the maximum conservative score
minus the candidate score.

```text
J(m) = regret(m) + 0.001*c_compute_ms(m)
     + 0.05*indicator(m != current_m) + 0.8*L_info(m).
```

The coefficient 0.001 converts the priced milliseconds to seconds. In seconds,
the computational coefficient is 1. A change of depth alone also receives the
fixed allocation-change penalty. L_info is canonical-grid Jensen-Shannon
divergence divided by log(2), in [0,1], after candidate aggregation and
reconstruction. Non-nested partitions may introduce reconstruction distortion
even when the candidate has more cells.

The cost profile is stored separately in each checkpoint and is not a learned
output head. Recalibrate for another machine; timings collected with concurrent
generation workers can be load-contaminated. Reported paper timings are from
single-worker profiling, not the checkpoint's candidate-profile calibration.

All 12 agent shells and their static models are built before the measured loop.
A switch transfers recurrent belief/action/inference state to a cached agent;
it does not construct a full MOS environment. Online time includes inference,
policy evaluation, metacontrol and measured transfer; setup is reported
separately. The fixed objective penalty is independent of measured transfer
time.

## Analysis and fleet assumptions

Context profiles use canonical-grid normalized belief entropy, the preceding
full-policy probability margin, and posterior-weighted viewpoint-normalized
Fisher information. Confidence margins depend on policy cardinality, which
varies with depth. These plots are descriptive associations, not causal tests.
Intervals resample whole episodes.

The fleet evaluator replays frozen static-environment traces through a
non-preemptive FCFS worker. Adaptive traces retain per-decision timings; fixed
traces use measured episode total divided evenly over decisions because fixed
per-decision timings were not retained. This approximates fixed service-time
variation. Every physical action is assigned 0.5 s. The planning deadline is
0.5 s including queue wait and service. A miss is recorded but does not discard
the plan; the agent waits and then executes it. Missions stop at 30 s.

Matched sampled instance IDs are used for both controllers. The reference
experiment uses one worker and 200 repeats with sampling seed 20261001.
Reliable fleet support means at least 80% discovery and at most 5% deadline
misses. Peak throughput is a different operating point and need not meet
those conditions. Agents neither coordinate nor share beliefs.
