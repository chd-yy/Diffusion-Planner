# Hyper Diffusion Planner (NuPlan)

In this repo, we provide an implementation of our Hyper Diffusion Planner on NuPlan benchmark, based on [*Diffusion Planner*](https://github.com/ZhengYinan-AIR/Diffusion-Planner). One can follow the *Diffusion Planner* for data processing, model training and evaluation.

## Main Modification

### Diffusion Loss Space

**(See more details in [Section 4.1] of the paper)**

We add choices of in diffusion loss space. Specifically, we add diffusion sde transformation in `hdp_nuplan/model/diffusion_utils/sde.py` and choices of supervision in `hdp_nuplan/loss.py`. One can train HDP models with different combination of model prediction and loss function by modifying the `diffusion_model_type` and `diffusion_supervision_type` arguments in `train_predictor.py`. 
```
# HDP-nuplan/torch_run.sh
# The default configuration is x_start model prediction with x_start supervision
sudo -E $RUN_PYTHON_PATH -m torch.distributed.run --nnodes 1 --nproc-per-node 8 --standalone train_predictor.py \
--train_set  $TRAIN_SET_PATH \
--train_set_list  $TRAIN_SET_LIST_PATH \
--diffusion_model_type "x_start" \
--diffusion_supervision_type "x_start" \
--batch_size 2048

# (e.g.) to use noise model prediction with v supervision
sudo -E $RUN_PYTHON_PATH -m torch.distributed.run --nnodes 1 --nproc-per-node 8 --standalone train_predictor.py \
--train_set  $TRAIN_SET_PATH \
--train_set_list  $TRAIN_SET_LIST_PATH \
--diffusion_model_type "noise" \
--diffusion_supervision_type "v" \
--batch_size 2048
```
We currently support `x_start`($\tau_0$), `noise`($\epsilon$) and `velocity`($v_t$). The transformation can be found in Table III of the paper.

### Hybrid Loss

**(See more details in [Section 4.2] of the paper)**

We use hybrid loss with velocity prediction in `hdp_nuplan/loss.py`: $$\mathcal{L}_{hybrid} = \mathcal{L}_{velocity} + \omega \cdot \mathcal{L}_{waypoints}$$
where the hybrid loss weight $\omega$ is passed by `planning_hybrid_loss` argument in `train_predictor.py`. The detach integration can be found in `hdp_nuplan/utils/traj_kinematics.py`.
```
def detached_integral(u, detach_window_size):
    # u: (B, T=80, D)
    cum_detach = torch.cumsum(u.detach(), dim=-2)
    cum_normal = torch.cumsum(u, dim=-2)

    # number of gradient from previous timesteps contained in: 
    # shifted: [0, 1, 2, ..., window_size-1, window_size, ...., T] ->
    # shifted: [T-window_size+1, T-window_size+2, ...,T, 0, 1, 2, ...., T - window_size] ->
    # sum_recent: [0, 1, 2, ..., window_size-1, window_size, ...., window_size]
    shifted = torch.roll(cum_normal, shifts=detach_window_size, dims=-2)
    shifted[:, :, :detach_window_size] = 0
    sum_recent = cum_normal - shifted
        
    cum_detach_shifted = torch.roll(cum_detach, shifts=detach_window_size, dims=-2)
    cum_detach_shifted[:, :, :detach_window_size] = 0
        
    cumulative_sum = cum_detach_shifted + sum_recent
    return cumulative_sum
```
We also provide a default normalization compatible with the numerical scale of velocity, specified in `normalization.json`.

## Balanced Mini Data and Encoder Warm-start

For small-data experiments, global random sampling can omit entire NuPlan
logs. The preprocessing entry supports deterministic per-log quotas followed
by seeded random remainder filling:

```bash
python data_process.py \
  --data_path /absolute/path/to/nuplan-v1.1_mini/data/cache/mini \
  --map_path /absolute/path/to/maps \
  --save_path /absolute/path/to/train-cache \
  --log_names_json config/mini_splits/mini_train_logs.json \
  --output_list_path /absolute/path/to/train-manifest.json \
  --sampling_report_path /absolute/path/to/sampling-report.json \
  --total_scenarios 10000 \
  --sampling_strategy balanced_logs \
  --seed 3407
```

The generated sampling report records the available and selected count for
each requested log. The manifest contains only files selected by the current
run and preprocessing fails if any expected NPZ is missing.

Validate the completed cache before training:

```bash
python scripts/validate_processed_cache.py \
  --cache_dir /absolute/path/to/train-cache \
  --manifest /absolute/path/to/train-manifest.json \
  --sampling_report /absolute/path/to/sampling-report.json \
  --expected_count 10000 \
  --expected_log_count 44 \
  --output /absolute/path/to/cache-validation-report.json
```

This rejects duplicate or unsorted manifests, missing or stale NPZ files,
incorrect tensor shapes, non-finite values, and disagreements between actual
NPZ log counts and the sampling audit.

HDP can also initialize only its structurally compatible encoder from a
released Diffusion-Planner checkpoint while leaving the HDP decoder randomly
initialized:

```bash
python -m torch.distributed.run --standalone --nproc-per-node 1 \
  train_predictor.py \
  --train_set /absolute/path/to/train-cache \
  --train_set_list /absolute/path/to/train-manifest.json \
  --encoder_pretrained_model_path /absolute/path/to/diffusion-planner/model.pth \
  --freeze_encoder_epochs 3 \
  --warm_up_epoch 2 \
  --train_epochs 20 \
  --batch_size 8
```

Every run writes `encoder_warm_start_report.json`, including the source state,
loaded tensor and parameter counts, missing keys, and shape mismatches.
`--resume_model_path` and `--encoder_pretrained_model_path` are mutually
exclusive because they represent full-state resume and encoder-only
initialization, respectively.

Rank all saved supervised epochs on an independent cached validation split:

```bash
python evaluate_checkpoints.py \
  --args_file /absolute/path/to/training-run/args.json \
  --checkpoint_dir /absolute/path/to/training-run \
  --data_dir /absolute/path/to/val-cache \
  --data_list /absolute/path/to/val-manifest.json \
  --repeats 1 \
  --output /absolute/path/to/checkpoint-ranking.json
```

The ranking uses EMA weights and validation total loss. A practical protocol
is to scan every epoch once, then re-evaluate the top candidates with three
repeats before running the official closed-loop gate.

## NuPlan Diffusion RL Fine-tuning

This repository also provides an offline NuPlan migration of the reward-based
diffusion fine-tuning design. The training loop alternates between:

1. sampling a group of trajectories for every NuPlan scene;
2. scoring them with cached NuPlan neighbors, route lanes and kinematics;
3. storing `(scene, trajectories, rewards)` in a replay buffer;
4. optimizing a group-normalized reward-weighted diffusion objective.

The default tensor scorer is intentionally separated from the RL trainer. It
can therefore be replaced by a full NuPlan/PDM closed-loop scorer without
changing the replay buffer or diffusion loss.

```bash
cd ./HDP-nuplan

python -m torch.distributed.run --standalone --nproc-per-node 1 \
  train_predictor_rl.py \
  --train_set /path/to/nuplan/cache \
  --train_set_list /path/to/diffusion_planner_training.json \
  --pretrained_model_path /path/to/pretrained/latest.pth \
  --batch_size 32 \
  --rl_group_size 8 \
  --rl_rollout_steps 5
```

Main implementation files:

- `hdp_nuplan/rl/reward.py`: NuPlan tensor reward and replaceable scorer API;
- `hdp_nuplan/rl/replay_buffer.py`: scene-level grouped replay buffer;
- `hdp_nuplan/rl/loss.py`: group advantage and reward-weighted diffusion loss;
- `hdp_nuplan/rl/train_epoch_rl.py`: alternating rollout/update epochs;
- `train_predictor_rl.py`: command-line training entry.

The default reward is an offline approximation based on progress, collision
distance, route deviation, comfort and backward motion. Final model quality
must still be measured with the official NuPlan closed-loop simulation and
metrics.

## NuPlan Mini Closed-loop Evaluation

The repository includes a reproducible three-scenario mini-val smoke protocol
for the official `closed_loop_nonreactive_agents` challenge. The selected
tokens cover high speed, stationary traffic, and traffic-light intersection
scenarios and all have valid mission goals.

Run an HDP checkpoint:

```bash
bash scripts/run_mini_closed_loop.sh \
  my-hdp-eval \
  /absolute/path/to/args.json \
  /absolute/path/to/latest.pth
```

Run the released Diffusion-Planner checkpoint on the same scenarios:

```bash
bash scripts/run_mini_closed_loop.sh \
  my-diffusion-planner-eval \
  /absolute/path/to/diffusion-planner/args.json \
  /absolute/path/to/diffusion-planner/model.pth \
  mini-val-closed-loop-3 \
  diffusion
```

The output directory contains the official metric parquet files, weighted
aggregator result, runner report, NuBoard descriptor, and summary PDF. Multiple
runs can be converted to one comparison JSON with:

```bash
python scripts/summarize_closed_loop_metrics.py \
  --run supervised=/absolute/path/to/supervised-run \
  --run rl=/absolute/path/to/rl-run \
  --output /absolute/path/to/comparison.json
```

The complete local pilot protocol, failures, fixes, hashes, and measured
results are recorded in `docs/mini_supervised_rl_operation_log.md`. The
three-scenario result is an engineering smoke test, not a benchmark claim.

The repository also contains a deterministic 20-scenario mini-val gate. Pass
its scenario-filter name as the fourth argument:

```bash
bash scripts/run_mini_closed_loop.sh \
  my-hdp-eval-20 \
  /absolute/path/to/args.json \
  /absolute/path/to/model.pth \
  mini-val-closed-loop-20
```

The local train10k experiment selected supervised epoch 10 by independent
val1k loss. On the exact same 20 closed-loop scenarios, the released
Diffusion-Planner and selected HDP checkpoint produced:

| metric | Diffusion-Planner | HDP supervised epoch 10 |
|---|---:|---:|
| overall score | 0.881446 | 0.787373 |
| no collision | 0.900000 | 0.950000 |
| TTC within bound | 0.900000 | 0.900000 |
| expert-route progress | 0.960774 | 0.668771 |
| drivable-area compliance | 0.950000 | 1.000000 |
| comfort | 0.950000 | 1.000000 |

The NuPlan RL pipeline also completed a full 10,000-scene fine-tuning run.
Its best checkpoint was deliberately rejected: val1k loss was 28.879% worse
than the supervised source, and the fixed three-scenario score decreased from
0.947107 to 0.939915. The accepted artifact is therefore the supervised epoch
10 checkpoint. See the operation log for exact paths, hashes, failed attempts,
and the acceptance-gate rationale.

## Getting Started

- Setup conda environment
```
conda create -n hdp_nuplan python=3.9
conda activate hdp_nuplan

# setup hyper_diffusion_planner
# pwd: */Hyper-Diffusion-Planner/
cd ./HDP-nuplan/
pip install -e .
pip install -r requirements_torch.txt
```
- Setup the nuPlan dependency, prepare the training data, and launch training and evaluation following the guidance in [*Diffusion Planner*](https://github.com/ZhengYinan-AIR/Diffusion-Planner).
