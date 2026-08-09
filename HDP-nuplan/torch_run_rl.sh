#!/usr/bin/env bash
set -e

# 复用原 Diffusion-Planner 的 Conda 环境。
RUN_PYTHON_PATH=/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python
TRAIN_SET_PATH=
TRAIN_SET_LIST_PATH=
PRETRAINED_MODEL_PATH=

"${RUN_PYTHON_PATH}" -m torch.distributed.run \
  --nnodes 1 \
  --nproc-per-node "${NPROC_PER_NODE:-1}" \
  --standalone \
  train_predictor_rl.py \
  --train_set "${TRAIN_SET_PATH}" \
  --train_set_list "${TRAIN_SET_LIST_PATH}" \
  --pretrained_model_path "${PRETRAINED_MODEL_PATH}" \
  --batch_size 32 \
  --rl_group_size 8 \
  --rl_rollout_steps 5
