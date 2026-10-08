# Training and evaluation

Run commands from the repository root after the installation in README.
PowerShell commands are single-line to avoid continuation-character problems.
Use Python module commands on any supported shell; range syntax below is
PowerShell-specific. Generated outputs go to the ignored results directory.

## Included artifacts

- artifacts/checkpoints/seed0.pt, seed1.pt, seed2.pt: frozen value ensemble.
- artifacts/training/: generation manifest, corpus summary and training reports.
- artifacts/results/: original aggregate results, episode tables, trajectory
  traces, context replay and fleet trials with machine-local paths removed.

Large transitions.npz tensors are not included. Regenerate them with the
collection command below. Sampling and numeric versions can affect exact
trajectory reproduction; wall-clock measurements always depend on hardware.
Pinned dependency revisions are in requirements-benchmark.txt.

## Data collection

```powershell
python -m active_inference_neural_metacontrol.meta_q_generate_cli --instance-seeds (14000..15199) --max-steps 50 --message-passing-iterations 10 --policy-workers 1 --instance-workers 4 --source-mode balanced --exploration-switch-probability 0.25 --failure-penalty 100 --output-dir results/meta_q_large_1200
```

For a setup test, replace the seed range with (13000..13002) and max steps with
3. Generation resumes compatible completed shards. Do not delete completed
shards to resume an interrupted run. Dataset timing is not a clean machine
profile when instance workers run concurrently.

## Ensemble training

```powershell
foreach ($seed in 0..2) { python -m active_inference_neural_metacontrol.meta_regret_train_cli --dataset-dir results/meta_q_large_1200 --output-dir "results/meta_regret_seed$seed" --epochs 100 --batch-size 128 --learning-rate 0.0003 --discount 0.98 --target-update-interval 100 --absolute-weight 0.25 --regret-weight 1 --ranking-weight 0.25 --ranking-margin 0.01 --validation-fraction 0.15 --test-fraction 0.15 --torch-threads 4 --seed $seed }
```

The commands below use the bundled frozen checkpoints, not newly trained
outputs. For a retraining experiment, replace the three checkpoint paths.

## Learned controller and fixed allocations

This runs the frozen objective on all test seeds and all 12 fixed allocations.
One instance worker is used for interpretable timing.

```powershell
python -m active_inference_neural_metacontrol.meta_q_evaluate_cli --checkpoint artifacts/checkpoints/seed0.pt --ensemble-checkpoints artifacts/checkpoints/seed1.pt artifacts/checkpoints/seed2.pt --instance-seeds (20000..20099) --initial-resolution 2 --initial-depth 1 --max-steps 50 --message-passing-iterations 10 --policy-workers 1 --selection-mode joint --compute-price 0.001 --compute-weight 1 --switch-cost-ms 50 --switching-weight 1 --switch-penalty-mode allocation --information-loss-weight 0.8 --uncertainty-beta 0.5 --torch-threads 1 --instance-workers 1 --include-fixed --output-dir results/benchmark_evaluation
```

The initial fixed allocations stay fixed; adaptive selection is reconsidered
after each nonterminal action. Pool construction is excluded from online
computation. Output includes episodes.csv, adaptive_trajectory.csv,
summary.json, paired bootstrap intervals, and resumable shards.

For switching ablation, set switching-weight to 0. For information-loss
ablation, set information-loss-weight to 0. Use a distinct output directory
for each combination. Keep all other arguments and seeds unchanged.

## Baselines

Thresholds below specify the evaluated heuristics, not CLI defaults.

```powershell
python -m active_inference_neural_metacontrol.heuristic_evaluate_cli --mode entropy --instance-seeds (20000..20099) --resolution-thresholds 0.05 0.15 1.0 --depth-thresholds 0.75 1.0 --minimum-hold-steps 2 --information-loss-limit 0.15 --message-passing-iterations 10 --policy-workers 1 --instance-workers 1 --output-dir results/entropy
python -m active_inference_neural_metacontrol.heuristic_evaluate_cli --mode fisher_surprise --instance-seeds (20000..20099) --resolution-thresholds 0.01 0.03 1.0 --depth-thresholds 0.45 1.0 --minimum-hold-steps 2 --information-loss-limit 0.15 --message-passing-iterations 10 --policy-workers 1 --instance-workers 1 --output-dir results/fisher
python -m active_inference_neural_metacontrol.pomcp_cli --instance-seeds (20000..20099) --simulations 250 --search-depth 30 --discount 0.98 --reward-mode external --max-steps 50 --instance-workers 1 --output-dir results/pomcp
python -m active_inference_neural_metacontrol.multi_resolution_pomcp_cli --instance-seeds (20000..20099) --resolutions 2 5 10 20 --simulations 250 --search-depth 30 --discount 0.98 --reward-mode external --max-steps 50 --instance-workers 1 --output-dir results/mr_pomcp
```

MR-POMCP splits 250 total simulations over resolutions. The exact categorical
root belief remains on the physical grid; this comparator is not an exact
reproduction of a 3D octree/macro-action planner. Validation used seeds
19000-19099; final reporting uses 20000-20099.

## Shared-compute fleet

Replay the included finalized profiling traces:

```powershell
python -m active_inference_neural_metacontrol.fleet_throughput_cli --results-dir artifacts/results/final_profiling_single_worker_20000_100 --fleet-sizes 1 2 4 8 12 16 20 24 32 40 48 64 --repeats 200 --sampling-seed 20261001 --mission-time-s 30 --cell-size-m 0.25 --robot-speed-mps 0.5 --planning-workers 1 --deadline-s 0.5 --output-dir results/fleet
python scripts/plot_fleet_throughput.py --trials results/fleet/fleet_trials.csv --output output/pdf/fleet_throughput.pdf
```

Cell size / speed specifies an assumed action duration, not validated robot
dynamics. Fixed service times are episode-average approximations. Details and
limitations are in method.md.

## Figures from saved reference results

These scripts use artifacts/results, so new long-running simulation is not
required to reproduce the numerical plots:

```powershell
python scripts/plot_figure4_task_compute_pareto.py
python scripts/plot_figure6_context_profiles.py
python scripts/plot_figure6_regularization_ablation.py
python scripts/plot_trajectory_allocation_timelines.py --maps-only
python scripts/animate_trajectories.py
python scripts/plot_fleet_throughput.py --trials artifacts/results/fleet_capacity_30s_1core/fleet_trials.csv --output output/pdf/fleet_throughput.pdf
```

Outputs are written to output/pdf. Trajectory illustrations are selected by a
retrospective fixed context-coverage rule among successful test episodes, with
ties resolved by seed. They are not preregistered selections or causal evidence.
The selection procedure is in scripts/select_context_coverage_examples.py.
