# Test14 三模型参数与配置完整清单

> 范围：2026-09-02 完成的 Test14-hard（272）与 Test14-random（261）正式单 worker 对比。
> 本文区分 checkpoint 的训练形成参数、闭环推理实际参数和 Hydra 最终解析配置。

## 1. 三个被评测模型与权重

三个 planner 均使用构造函数默认值 `enable_ema=true`，因此闭环推理实际加载 checkpoint 的
`ema_state_dict`，不是瞬时 `model` 权重。

| 标识 | 模型 | 参数量 | checkpoint epoch/loss | checkpoint SHA256 |
|---|---|---:|---|---|
| `aligned_original_dp` | 对齐训练的原始 Diffusion Planner Epoch10 | 6,042,628 | 10 / 0.0618437193334 | `37e024e88160a5766ec64984bf4a1a3e385d47cf8d4408ac73fd2c6b8e5b9786` |
| `hdp_b_epoch10` | HDP B Epoch10 | 5,092,996 | 10 / 0.00906237028539 | `22ec0cf6be7cc89a3cf8414cddd0c7446ce2737ddc4e44b23e50dbeeeb0b29ce` |
| `hdp_rl_epoch2` | HDP RL Epoch2（seed 2026） | 5,092,996 | 2 / 0.000309404665179 | `8cd630d0780521268a6a425c63dab3bb03e7f5897a5199f2ef7c1c3633fa2c3c` |

### 1.1 关键参数横向对照

下表按被评测 checkpoint 的实际形成过程填写；它不是简单照抄后来可能被覆盖的 args.json。

| 参数 | 对齐原始 DP Epoch10 | HDP B Epoch10 | HDP RL Epoch2 |
|---|---|---|---|
| Planner/模型 | `DiffusionPlanner` | `HyperDiffusionPlanner` | `HyperDiffusionPlanner` |
| 参数量 | 6,042,628 | 5,092,996 | 5,092,996 |
| 训练起点 | 原始 DP Encoder EMA warm-start | 原始 DP Encoder EMA warm-start | HDP B Epoch10 |
| 训练数据 | 完整 mini-train，306,801 NPZ | 完整 mini-train，306,801 NPZ | 固定 10,000 NPZ |
| checkpoint epoch | 10 | 10 | 2（后训练） |
| seed | 3407 | 3407 | 2026 |
| batch size | 8 | 8 | 2 |
| Epoch1～4 学习率 | `5e-4` | 启动基础值 `5e-4`；因断点恢复，checkpoint 实际均为 `5e-5` | 不适用 |
| Epoch5～10/后训练学习率 | `5e-5` | `5e-5` | `4e-7` |
| Encoder | Epoch1～3 冻结，之后解冻 | Epoch1～3 冻结，之后解冻 | 全程冻结 |
| 数据增强 | 开启，概率 `0.5` | 开启，概率 `0.5` | 关闭，概率 `0` |
| diffusion model type | `x_start` | `x_start` | `x_start` |
| 监督/规划损失 | `alpha_planning_loss=1.0` | `planning_hybrid_loss=0.01` | RL 加权回归 + `expert_anchor=0.1` |
| detach window | 不适用 | `0` | `0` |
| EMA | 开启 | 开启 | 开启 |

### 1.2 文件索引

#### 对齐训练的原始 Diffusion Planner Epoch10

- 模型类：`diffusion_planner.planner.planner.DiffusionPlanner`
- 训练入口：`train_predictor.py`
- checkpoint：`HDP-nuplan/tmp/original_diffusion_retrain_aligned_b10_306801_seed3407_epoch10/phase1_pretrain/training_log/original-diffusion-aligned-b10-phase1/2026-08-27-14:31:09/model_epoch_10_trainloss_0.0618.pth`
- args：`HDP-nuplan/tmp/original_diffusion_retrain_aligned_b10_306801_seed3407_epoch10/phase1_pretrain/training_log/original-diffusion-aligned-b10-phase1/2026-08-27-14:31:09/args.json`
- planner YAML：`diffusion_planner/config/planner/diffusion_planner.yaml`
- checkpoint keys：`['epoch', 'model', 'ema_state_dict', 'optimizer', 'schedule', 'loss', 'wandb_id']`
- `model`/`ema_state_dict` 张量数：276 / 276
- 说明：该 args.json 是最后一次从 Epoch9 恢复并完成 Epoch10 时写入的实际推理配置；初始 Encoder warm-start 与两阶段学习率需结合训练脚本查看。
- 训练/恢复脚本：
  - `HDP-nuplan/scripts/train_original_diffusion_same_data.sh`
  - `HDP-nuplan/scripts/resume_original_diffusion_aligned_epoch10_and_eval.sh`

#### HDP B Epoch10

- 模型类：`hdp_nuplan.planner.planner.HyperDiffusionPlanner`
- 训练入口：`HDP-nuplan/train_predictor.py`
- checkpoint：`HDP-nuplan/tmp/mini_train_full_306801_seed3407_v1/experiment_b_constant5e5_from_epoch4/model_epoch_10_trainloss_0.0091.pth`
- args：`HDP-nuplan/tmp/mini_train_full_306801_seed3407_v1/experiment_b_constant5e5_from_epoch4/args.json`
- planner YAML：`HDP-nuplan/hdp_nuplan/config/planner/hyper_diffusion_planner.yaml`
- checkpoint keys：`['epoch', 'model', 'ema_state_dict', 'optimizer', 'schedule', 'loss', 'wandb_id']`
- `model`/`ema_state_dict` 张量数：260 / 260
- 说明：该目录后来从 Epoch10 续训到 Epoch20，args.json 于 2026-08-28 再次写入，所以其中 train_epochs=20/name=...epoch20 是后续值；本次评测 checkpoint 仍严格为 Epoch10。
- 训练/恢复脚本：
  - `HDP-nuplan/scripts/switch_epoch9_a_to_constant5e5_b.sh`
  - `HDP-nuplan/scripts/resume_experiment_b_epoch10_to20.sh`

#### HDP RL Epoch2（seed 2026）

- 模型类：`hdp_nuplan.planner.planner.HyperDiffusionPlanner`
- 训练入口：`HDP-nuplan/train_predictor_rl.py`
- checkpoint：`HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/rl_safetygate_ttc1_anchor01_10k_seed2026_from_b10/training_log/hdp-rl-safetygate_ttc1_anchor01_10k_seed2026-from-full-mini-b10/2026-08-22-12:22:45/model_epoch_2_trainloss_0.0003.pth`
- args：`HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/rl_safetygate_ttc1_anchor01_10k_seed2026_from_b10/training_log/hdp-rl-safetygate_ttc1_anchor01_10k_seed2026-from-full-mini-b10/2026-08-22-12:22:45/args.json`
- planner YAML：`HDP-nuplan/hdp_nuplan/config/planner/hyper_diffusion_planner.yaml`
- checkpoint keys：`['epoch', 'model', 'ema_state_dict', 'optimizer', 'schedule', 'loss', 'wandb_id']`
- `model`/`ema_state_dict` 张量数：260 / 260
- 说明：该 args.json 为本次 RL Epoch2 checkpoint 同一次训练产生，未被后续训练覆盖。
- 训练/恢复脚本：
  - `HDP-nuplan/scripts/run_rl_v2_safety_gate59_from_b10_pilot1000.sh`

## 2. 训练形成过程

### 2.1 对齐原始 DP Epoch10

- 数据：完整 mini-train，306,801 NPZ；seed 3407；batch size 8；EMA 开启。
- Epoch1～4：Encoder 从 `checkpoints/model.pth` 的 EMA 权重 warm-start；加载 151/151 个
  Encoder 张量（1,799,040 参数），Decoder 不加载；Encoder 在 Epoch1～3 冻结；
  `learning_rate=5e-4`，`warm_up_epoch=2`。
- Epoch5～10：从 Epoch4/后续完整 checkpoint 恢复，Encoder 已解冻；
  `learning_rate=5e-5`，`warm_up_epoch=1`，`reset_lr_schedule_on_resume=true`。
- 原始损失：`alpha_planning_loss=1.0`；模型类型 `x_start`。

### 2.2 HDP B Epoch10

- 数据：与原始 DP 相同的完整 mini-train 306,801 NPZ；seed 3407；batch size 8；EMA 开启。
- 共同 Epoch1～4 起点：Encoder warm-start，Epoch1～3 冻结；启动参数为
  `learning_rate=5e-4`、`warm_up_epoch=2`。但训练在 Epoch1～4 间多次从保存于
  `scheduler.step()` 之前的 checkpoint 恢复，导致这四个完整 checkpoint 的实际
  optimizer LR 均为 warm-up 起始值 `5e-5`。
- Experiment B 从固定 Epoch4 checkpoint 分叉，Epoch5～10 使用恒定基础学习率 `5e-5`，
  `warm_up_epoch=1`，`reset_lr_schedule_on_resume=true`。
- HDP 损失：`planning_hybrid_loss=0.01`，`planning_detach_window_size=0`，
  `diffusion_model_type=x_start`，`diffusion_supervision_type=x_start`。
- 当前同目录 args 的 Epoch20 字段不改变 Epoch10 checkpoint 内已保存的权重。

### 2.3 HDP RL Epoch2

- 起点：B Epoch10；数据：固定 10,000 NPZ；seed 2026；batch size 2。
- 后训练 2 epoch；学习率 `4e-7`；Encoder 冻结；EMA 开启。
- 完整 RL、reward 和 safety-gate 参数见第 3.3 节的 97 项 args。

## 3. 三个 args.json 的全部字段

表中值是对应文件当前实际内容；嵌套 normalizer 在每节末完整展开。

### 3.1 对齐训练的原始 Diffusion Planner Epoch10（48 项）

文件：`HDP-nuplan/tmp/original_diffusion_retrain_aligned_b10_306801_seed3407_epoch10/phase1_pretrain/training_log/original-diffusion-aligned-b10-phase1/2026-08-27-14:31:09/args.json`
SHA256：`5908f185cf61fdbd96f8384690da7a61ba2e55c1217009471c49eaa406850eec`

| 参数 | 值 |
|---|---|
| `name` | `"original-diffusion-aligned-b10-phase2-resume-epoch9"` |
| `save_dir` | `"/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/original_diffusion_retrain_aligned_b10_306801_seed3407_epoch10/phase2_finetune"` |
| `train_set` | `"/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_train_full_306801_seed3407_v1/cache"` |
| `train_set_list` | `"/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_train_full_306801_seed3407_v1/diffusion_planner_training.json"` |
| `future_len` | `80` |
| `time_len` | `21` |
| `agent_state_dim` | `11` |
| `agent_num` | `32` |
| `static_objects_state_dim` | `10` |
| `static_objects_num` | `5` |
| `lane_len` | `20` |
| `lane_state_dim` | `12` |
| `lane_num` | `70` |
| `route_len` | `20` |
| `route_state_dim` | `12` |
| `route_num` | `25` |
| `augment_prob` | `0.5` |
| `normalization_file_path` | `"/home/yanjun/NewDisk/Diffusion-Planner/normalization.json"` |
| `use_data_augment` | `true` |
| `num_workers` | `0` |
| `pin_mem` | `true` |
| `seed` | `3407` |
| `train_epochs` | `10` |
| `save_utd` | `1` |
| `batch_size` | `8` |
| `learning_rate` | `5e-05` |
| `warm_up_epoch` | `1` |
| `encoder_drop_path_rate` | `0.1` |
| `decoder_drop_path_rate` | `0.1` |
| `alpha_planning_loss` | `1.0` |
| `device` | `"cuda"` |
| `use_ema` | `true` |
| `encoder_depth` | `3` |
| `decoder_depth` | `3` |
| `num_heads` | `6` |
| `hidden_dim` | `192` |
| `diffusion_model_type` | `"x_start"` |
| `predicted_neighbor_num` | `10` |
| `resume_model_path` | `"/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/original_diffusion_retrain_aligned_b10_306801_seed3407_epoch10/phase1_pretrain/training_log/original-diffusion-aligned-b10-phase1/2026-08-27-14:31:09"` |
| `encoder_pretrained_model_path` | `null` |
| `freeze_encoder_epochs` | `3` |
| `reset_lr_schedule_on_resume` | `true` |
| `use_wandb` | `false` |
| `notes` | `""` |
| `ddp` | `true` |
| `port` | `"22323"` |

#### `state_normalizer`

```json
{
  "mean": [
    [
      [
        10,
        0,
        0,
        0
      ]
    ],
    [
      [
        10,
        0,
        0,
        0
      ]
    ],
    [
      [
        10,
        0,
        0,
        0
      ]
    ],
    [
      [
        10,
        0,
        0,
        0
      ]
    ],
    [
      [
        10,
        0,
        0,
        0
      ]
    ],
    [
      [
        10,
        0,
        0,
        0
      ]
    ],
    [
      [
        10,
        0,
        0,
        0
      ]
    ],
    [
      [
        10,
        0,
        0,
        0
      ]
    ],
    [
      [
        10,
        0,
        0,
        0
      ]
    ],
    [
      [
        10,
        0,
        0,
        0
      ]
    ],
    [
      [
        10,
        0,
        0,
        0
      ]
    ]
  ],
  "std": [
    [
      [
        20,
        20,
        1,
        1
      ]
    ],
    [
      [
        20,
        20,
        1,
        1
      ]
    ],
    [
      [
        20,
        20,
        1,
        1
      ]
    ],
    [
      [
        20,
        20,
        1,
        1
      ]
    ],
    [
      [
        20,
        20,
        1,
        1
      ]
    ],
    [
      [
        20,
        20,
        1,
        1
      ]
    ],
    [
      [
        20,
        20,
        1,
        1
      ]
    ],
    [
      [
        20,
        20,
        1,
        1
      ]
    ],
    [
      [
        20,
        20,
        1,
        1
      ]
    ],
    [
      [
        20,
        20,
        1,
        1
      ]
    ],
    [
      [
        20,
        20,
        1,
        1
      ]
    ]
  ]
}
```

#### `observation_normalizer`

```json
{
  "ego_current_state": {
    "mean": [
      10.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ],
    "std": [
      20.0,
      20.0,
      1.0,
      1.0,
      20.0,
      20.0,
      20.0,
      20.0,
      1.0,
      1.0
    ]
  },
  "neighbor_agents_past": {
    "mean": [
      10.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ],
    "std": [
      20.0,
      20.0,
      1.0,
      1.0,
      20.0,
      20.0,
      20.0,
      20.0,
      1.0,
      1.0,
      1.0
    ]
  },
  "static_objects": {
    "mean": [
      10.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ],
    "std": [
      20.0,
      20.0,
      1.0,
      1.0,
      20.0,
      20.0,
      1.0,
      1.0,
      1.0,
      1.0
    ]
  },
  "lanes": {
    "mean": [
      10.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ],
    "std": [
      20.0,
      20.0,
      20.0,
      20.0,
      20.0,
      20.0,
      20.0,
      20.0,
      1.0,
      1.0,
      1.0,
      1.0
    ]
  },
  "lanes_speed_limit": {
    "mean": [
      0.0
    ],
    "std": [
      20.0
    ]
  },
  "route_lanes": {
    "mean": [
      10.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ],
    "std": [
      20.0,
      20.0,
      20.0,
      20.0,
      20.0,
      20.0,
      20.0,
      20.0,
      1.0,
      1.0,
      1.0,
      1.0
    ]
  },
  "route_lanes_speed_limit": {
    "mean": [
      0.0
    ],
    "std": [
      20.0
    ]
  }
}
```

### 3.2 HDP B Epoch10（51 项）

文件：`HDP-nuplan/tmp/mini_train_full_306801_seed3407_v1/experiment_b_constant5e5_from_epoch4/args.json`
SHA256：`ba6848d19bf5ea486fcc9ff31299cadccd8778c927fdf7e961e65381b2caa39f`

| 参数 | 值 |
|---|---|
| `name` | `"hdp-full-mini-experiment-b-epoch20-constant5e5"` |
| `save_dir` | `"/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_train_full_306801_seed3407_v1/experiment_b_constant5e5_from_epoch4"` |
| `train_set` | `"/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_train_full_306801_seed3407_v1/cache"` |
| `train_set_list` | `"/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_train_full_306801_seed3407_v1/diffusion_planner_training.json"` |
| `future_len` | `80` |
| `time_len` | `21` |
| `agent_state_dim` | `11` |
| `agent_num` | `32` |
| `static_objects_state_dim` | `10` |
| `static_objects_num` | `5` |
| `lane_len` | `20` |
| `lane_state_dim` | `12` |
| `lane_num` | `70` |
| `route_len` | `20` |
| `route_state_dim` | `12` |
| `route_num` | `25` |
| `augment_prob` | `0.5` |
| `normalization_file_path` | `"/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/normalization.json"` |
| `use_data_augment` | `true` |
| `num_workers` | `0` |
| `log_unused_parameters` | `false` |
| `pin_mem` | `true` |
| `seed` | `3407` |
| `train_epochs` | `20` |
| `save_utd` | `1` |
| `batch_size` | `8` |
| `learning_rate` | `5e-05` |
| `warm_up_epoch` | `1` |
| `reset_lr_schedule_on_resume` | `true` |
| `encoder_drop_path_rate` | `0.1` |
| `decoder_drop_path_rate` | `0.1` |
| `planning_hybrid_loss` | `0.01` |
| `planning_detach_window_size` | `0` |
| `device` | `"cuda"` |
| `use_ema` | `true` |
| `encoder_depth` | `3` |
| `decoder_depth` | `3` |
| `num_heads` | `6` |
| `hidden_dim` | `192` |
| `diffusion_model_type` | `"x_start"` |
| `diffusion_supervision_type` | `"x_start"` |
| `predicted_neighbor_num` | `10` |
| `resume_model_path` | `"/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_train_full_306801_seed3407_v1/experiment_b_constant5e5_from_epoch4"` |
| `encoder_pretrained_model_path` | `null` |
| `freeze_encoder_epochs` | `3` |
| `use_wandb` | `false` |
| `notes` | `""` |
| `ddp` | `true` |
| `port` | `"22323"` |

#### `state_normalizer`

```json
{
  "mean": [
    [
      0,
      0,
      0,
      0
    ]
  ],
  "std": [
    [
      0.5,
      0.5,
      1.0,
      1.0
    ]
  ]
}
```

#### `observation_normalizer`

```json
{
  "ego_current_state": {
    "mean": [
      10.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ],
    "std": [
      20.0,
      20.0,
      1.0,
      1.0,
      20.0,
      20.0,
      20.0,
      20.0,
      1.0,
      1.0
    ]
  },
  "neighbor_agents_past": {
    "mean": [
      10.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ],
    "std": [
      20.0,
      20.0,
      1.0,
      1.0,
      20.0,
      20.0,
      20.0,
      20.0,
      1.0,
      1.0,
      1.0
    ]
  },
  "static_objects": {
    "mean": [
      10.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ],
    "std": [
      20.0,
      20.0,
      1.0,
      1.0,
      20.0,
      20.0,
      1.0,
      1.0,
      1.0,
      1.0
    ]
  },
  "lanes": {
    "mean": [
      10.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ],
    "std": [
      20.0,
      20.0,
      20.0,
      20.0,
      20.0,
      20.0,
      20.0,
      20.0,
      1.0,
      1.0,
      1.0,
      1.0
    ]
  },
  "lanes_speed_limit": {
    "mean": [
      0.0
    ],
    "std": [
      20.0
    ]
  },
  "route_lanes": {
    "mean": [
      10.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ],
    "std": [
      20.0,
      20.0,
      20.0,
      20.0,
      20.0,
      20.0,
      20.0,
      20.0,
      1.0,
      1.0,
      1.0,
      1.0
    ]
  },
  "route_lanes_speed_limit": {
    "mean": [
      0.0
    ],
    "std": [
      20.0
    ]
  }
}
```

### 3.3 HDP RL Epoch2（seed 2026）（97 项）

文件：`HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/rl_safetygate_ttc1_anchor01_10k_seed2026_from_b10/training_log/hdp-rl-safetygate_ttc1_anchor01_10k_seed2026-from-full-mini-b10/2026-08-22-12:22:45/args.json`
SHA256：`1c0736ad108ff79fe03711b14886c0a97e77e284b2d5ab252422fec0dcdeda9f`

| 参数 | 值 |
|---|---|
| `name` | `"hdp-rl-safetygate_ttc1_anchor01_10k_seed2026-from-full-mini-b10"` |
| `save_dir` | `"/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/rl_safetygate_ttc1_anchor01_10k_seed2026_from_b10"` |
| `train_set` | `"/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/cache"` |
| `train_set_list` | `"/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/diffusion_planner_training.json"` |
| `pretrained_model_path` | `"/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_train_full_306801_seed3407_v1/experiment_b_constant5e5_from_epoch4/model_epoch_10_trainloss_0.0091.pth"` |
| `normalization_file_path` | `"/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/normalization.json"` |
| `future_len` | `80` |
| `time_len` | `21` |
| `agent_state_dim` | `11` |
| `agent_num` | `32` |
| `predicted_neighbor_num` | `10` |
| `static_objects_state_dim` | `10` |
| `static_objects_num` | `5` |
| `lane_len` | `20` |
| `lane_state_dim` | `12` |
| `lane_num` | `70` |
| `route_len` | `20` |
| `route_state_dim` | `12` |
| `route_num` | `25` |
| `encoder_depth` | `3` |
| `decoder_depth` | `3` |
| `num_heads` | `6` |
| `hidden_dim` | `192` |
| `diffusion_model_type` | `"x_start"` |
| `diffusion_supervision_type` | `"x_start"` |
| `encoder_drop_path_rate` | `0.1` |
| `decoder_drop_path_rate` | `0.1` |
| `planning_hybrid_loss` | `0.01` |
| `train_epochs` | `2` |
| `batch_size` | `2` |
| `learning_rate` | `4e-07` |
| `warm_up_epoch` | `1` |
| `save_utd` | `1` |
| `num_workers` | `2` |
| `pin_mem` | `true` |
| `device` | `"cuda"` |
| `seed` | `2026` |
| `use_wandb` | `false` |
| `notes` | `""` |
| `rl_group_size` | `32` |
| `rl_rollout_steps` | `6` |
| `rl_sampling_noise_scale` | `0.1` |
| `rl_trajectory_augmentation_std` | `0.0` |
| `rl_trajectory_augmentation_epochs` | `0` |
| `rl_buffer_update_epoch` | `2` |
| `rl_buffer_size` | `1024` |
| `rl_ema_update_rate` | `0.05` |
| `rl_reward_temperature` | `1.0` |
| `rl_advantage_clip` | `5.0` |
| `rl_min_reward_std` | `1e-06` |
| `rl_normalize_weights` | `true` |
| `rl_center_reward_weights` | `true` |
| `rl_rollout_loss_weight` | `1.0` |
| `rl_expert_anchor_weight` | `0.1` |
| `rl_max_update_steps_per_epoch` | `500` |
| `rl_detach_window_size` | `0` |
| `rl_grad_clip` | `5.0` |
| `rl_freeze_encoder` | `true` |
| `rl_deterministic_update` | `true` |
| `reward_progress_weight` | `1.0` |
| `reward_collision_weight` | `10.0` |
| `reward_route_weight` | `1.0` |
| `reward_comfort_weight` | `0.01` |
| `reward_backward_weight` | `1.0` |
| `reward_imitation_weight` | `0.0` |
| `reward_collision_distance` | `0.5` |
| `reward_risk_weight` | `1.0` |
| `reward_follow_weight` | `3.0` |
| `reward_lane_weight` | `2.5` |
| `reward_progress_guard_weight` | `5.0` |
| `reward_progress_guard_stop_tolerance` | `0.2` |
| `reward_safety_gate_threshold` | `0.3` |
| `reward_safety_gate_margin` | `1.0` |
| `reward_safety_gate_min_ttc_seconds` | `1.0` |
| `reward_risk_speed_reference` | `15.0` |
| `reward_ttc_safe_low_speed` | `2.0` |
| `reward_ttc_safe_high_speed` | `4.0` |
| `reward_thw_critical` | `0.5` |
| `reward_thw_safe_low_speed` | `1.0` |
| `reward_thw_safe_high_speed` | `2.0` |
| `reward_occupancy_safe_min` | `0.5` |
| `reward_occupancy_safe_max` | `3.0` |
| `reward_occupancy_time_headway` | `0.2` |
| `reward_rear_end_collision_penalty` | `0.3` |
| `reward_follow_time_gap_low_speed` | `1.0` |
| `reward_follow_time_gap_high_speed` | `2.0` |
| `reward_follow_min_spacing` | `2.0` |
| `reward_follow_speed_tolerance` | `2.0` |
| `reward_follow_comfort_acceleration` | `2.0` |
| `reward_follow_comfort_deceleration` | `3.0` |
| `reward_leader_lateral_margin` | `0.5` |
| `reward_lane_half_width_fallback` | `1.75` |
| `reward_lane_change_ratio` | `0.5` |
| `ddp` | `true` |
| `port` | `"22324"` |

#### `state_normalizer`

```json
{
  "mean": [
    [
      0,
      0,
      0,
      0
    ]
  ],
  "std": [
    [
      0.5,
      0.5,
      1.0,
      1.0
    ]
  ]
}
```

#### `observation_normalizer`

```json
{
  "ego_current_state": {
    "mean": [
      10.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ],
    "std": [
      20.0,
      20.0,
      1.0,
      1.0,
      20.0,
      20.0,
      20.0,
      20.0,
      1.0,
      1.0
    ]
  },
  "neighbor_agents_past": {
    "mean": [
      10.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ],
    "std": [
      20.0,
      20.0,
      1.0,
      1.0,
      20.0,
      20.0,
      20.0,
      20.0,
      1.0,
      1.0,
      1.0
    ]
  },
  "static_objects": {
    "mean": [
      10.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ],
    "std": [
      20.0,
      20.0,
      1.0,
      1.0,
      20.0,
      20.0,
      1.0,
      1.0,
      1.0,
      1.0
    ]
  },
  "lanes": {
    "mean": [
      10.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ],
    "std": [
      20.0,
      20.0,
      20.0,
      20.0,
      20.0,
      20.0,
      20.0,
      20.0,
      1.0,
      1.0,
      1.0,
      1.0
    ]
  },
  "lanes_speed_limit": {
    "mean": [
      0.0
    ],
    "std": [
      20.0
    ]
  },
  "route_lanes": {
    "mean": [
      10.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ],
    "std": [
      20.0,
      20.0,
      20.0,
      20.0,
      20.0,
      20.0,
      20.0,
      20.0,
      1.0,
      1.0,
      1.0,
      1.0
    ]
  },
  "route_lanes_speed_limit": {
    "mean": [
      0.0
    ],
    "std": [
      20.0
    ]
  }
}
```

## 4. Planner YAML 原文

原始 DP 与两个 HDP checkpoint 分别使用以下两个 planner YAML；B 与 RL 共用 HDP YAML，
但 `args_file` 和 `ckpt_path` 在运行时被覆盖。

### 4.1 原始 Diffusion Planner

来源：`diffusion_planner/config/planner/diffusion_planner.yaml`
SHA256：`8f01d3c51adaa00e9d43fa8731e781a4f797700b908768179188ea9d6b4e3cba`

```yaml
diffusion_planner:
  _target_: diffusion_planner.planner.planner.DiffusionPlanner
  _convert_: "all"

  config:
    _target_: diffusion_planner.utils.config.Config
    _convert_: "all"

    args_file: ???

    guidance_fn: null
  ckpt_path: ???

  past_trajectory_sampling:
    _target_: nuplan.planning.simulation.trajectory.trajectory_sampling.TrajectorySampling
    _convert_: "all"

    num_poses: 20
    time_horizon: 2

  future_trajectory_sampling:
    _target_: nuplan.planning.simulation.trajectory.trajectory_sampling.TrajectorySampling
    _convert_: "all"

    num_poses: 80
    time_horizon: 8

  device: cuda
```

### 4.2 HDP B / RL

来源：`HDP-nuplan/hdp_nuplan/config/planner/hyper_diffusion_planner.yaml`
SHA256：`6517e15cf1bd76d336fe455fec0ccbe387d73052a142b14e192a2c48c22151f6`

```yaml
hyper_diffusion_planner:
  # HDP 使用独立的 planner 类和配置对象，不再加载 diffusion_planner 的 guidance 配置。
  _target_: hdp_nuplan.planner.planner.HyperDiffusionPlanner
  _convert_: "all"

  config:
    _target_: hdp_nuplan.utils.config.Config
    _convert_: "all"

    args_file: ???
  ckpt_path: ???

  past_trajectory_sampling:
    _target_: nuplan.planning.simulation.trajectory.trajectory_sampling.TrajectorySampling
    _convert_: "all"

    num_poses: 20
    time_horizon: 2

  future_trajectory_sampling:
    _target_: nuplan.planning.simulation.trajectory.trajectory_sampling.TrajectorySampling
    _convert_: "all"

    num_poses: 80
    time_horizon: 8

  device: cuda
```

## 5. Test14 闭环评测实际覆盖参数

六组正式运行共享：

```text
+simulation=closed_loop_nonreactive_agents
scenario_builder=nuplan
worker=single_machine_thread_pool
worker.max_workers=1
worker.use_process_pool=false
number_of_gpus_allocated_per_simulation=1
number_of_cpus_allocated_per_simulation=1
max_callback_workers=1
disable_callback_parallelization=true
enable_simulation_progress_bar=true
verbose=true
~callback.simulation_log_callback
past_trajectory_sampling: num_poses=20, time_horizon=2s
future_trajectory_sampling: num_poses=80, time_horizon=8s
device=cuda
```

不同模型仅覆盖 planner 类型、args、checkpoint 和 experiment UID；不同 benchmark 仅覆盖
scenario filter。完整执行入口：

`HDP-nuplan/scripts/evaluate_full_test14_three_models_chunked.sh`

### 5.1 每个正式运行保存的 Hydra 文件

`config.yaml` 是 NuPlan/Hydra 最终解析后的完整配置（约 1.6 万行）；`overrides.yaml`
是本次命令行覆盖项；`hydra.yaml` 是 Hydra 自身运行配置。由于同一 UID 分片续跑，
这些快照对应各 UID 最后执行的分片，所有分片 token 则由第 6 节 YAML 完整记录。

| 模型/Benchmark | 完整配置 | 覆盖项 | Hydra 配置 |
|---|---|---|---|
| `aligned_original_dp` / test14-hard | `HDP-nuplan/tmp/test14_full_remote_subset/three_model_eval/exp/simulation/closed_loop_nonreactive_agents/aligned-original-dp-test14-hard/code/hydra/config.yaml` | `HDP-nuplan/tmp/test14_full_remote_subset/three_model_eval/exp/simulation/closed_loop_nonreactive_agents/aligned-original-dp-test14-hard/code/hydra/overrides.yaml` | `HDP-nuplan/tmp/test14_full_remote_subset/three_model_eval/exp/simulation/closed_loop_nonreactive_agents/aligned-original-dp-test14-hard/code/hydra/hydra.yaml` |
| `aligned_original_dp` / test14-random | `HDP-nuplan/tmp/test14_full_remote_subset/three_model_eval/exp/simulation/closed_loop_nonreactive_agents/aligned-original-dp-test14-random/code/hydra/config.yaml` | `HDP-nuplan/tmp/test14_full_remote_subset/three_model_eval/exp/simulation/closed_loop_nonreactive_agents/aligned-original-dp-test14-random/code/hydra/overrides.yaml` | `HDP-nuplan/tmp/test14_full_remote_subset/three_model_eval/exp/simulation/closed_loop_nonreactive_agents/aligned-original-dp-test14-random/code/hydra/hydra.yaml` |
| `hdp_b_epoch10` / test14-hard | `HDP-nuplan/tmp/test14_full_remote_subset/three_model_eval/exp/simulation/closed_loop_nonreactive_agents/hdp-b-epoch10-test14-hard/code/hydra/config.yaml` | `HDP-nuplan/tmp/test14_full_remote_subset/three_model_eval/exp/simulation/closed_loop_nonreactive_agents/hdp-b-epoch10-test14-hard/code/hydra/overrides.yaml` | `HDP-nuplan/tmp/test14_full_remote_subset/three_model_eval/exp/simulation/closed_loop_nonreactive_agents/hdp-b-epoch10-test14-hard/code/hydra/hydra.yaml` |
| `hdp_b_epoch10` / test14-random | `HDP-nuplan/tmp/test14_full_remote_subset/three_model_eval/exp/simulation/closed_loop_nonreactive_agents/hdp-b-epoch10-test14-random/code/hydra/config.yaml` | `HDP-nuplan/tmp/test14_full_remote_subset/three_model_eval/exp/simulation/closed_loop_nonreactive_agents/hdp-b-epoch10-test14-random/code/hydra/overrides.yaml` | `HDP-nuplan/tmp/test14_full_remote_subset/three_model_eval/exp/simulation/closed_loop_nonreactive_agents/hdp-b-epoch10-test14-random/code/hydra/hydra.yaml` |
| `hdp_rl_epoch2` / test14-hard | `HDP-nuplan/tmp/test14_full_remote_subset/three_model_eval/exp/simulation/closed_loop_nonreactive_agents/hdp-rl-epoch2-test14-hard/code/hydra/config.yaml` | `HDP-nuplan/tmp/test14_full_remote_subset/three_model_eval/exp/simulation/closed_loop_nonreactive_agents/hdp-rl-epoch2-test14-hard/code/hydra/overrides.yaml` | `HDP-nuplan/tmp/test14_full_remote_subset/three_model_eval/exp/simulation/closed_loop_nonreactive_agents/hdp-rl-epoch2-test14-hard/code/hydra/hydra.yaml` |
| `hdp_rl_epoch2` / test14-random | `HDP-nuplan/tmp/test14_full_remote_subset/three_model_eval/exp/simulation/closed_loop_nonreactive_agents/hdp-rl-epoch2-test14-random/code/hydra/config.yaml` | `HDP-nuplan/tmp/test14_full_remote_subset/three_model_eval/exp/simulation/closed_loop_nonreactive_agents/hdp-rl-epoch2-test14-random/code/hydra/overrides.yaml` | `HDP-nuplan/tmp/test14_full_remote_subset/three_model_eval/exp/simulation/closed_loop_nonreactive_agents/hdp-rl-epoch2-test14-random/code/hydra/hydra.yaml` |

### 5.2 三模型 Test14-hard 最终分片 overrides.yaml 原文

#### 对齐训练的原始 Diffusion Planner Epoch10

来源：`HDP-nuplan/tmp/test14_full_remote_subset/three_model_eval/exp/simulation/closed_loop_nonreactive_agents/aligned-original-dp-test14-hard/code/hydra/overrides.yaml`
SHA256：`b67b0397ba8543effd98fee6897bb40ff24f674e8857b5180942aa1c90524764`

```yaml
- +simulation=closed_loop_nonreactive_agents
- planner=diffusion_planner
- planner.diffusion_planner.config.args_file=/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/original_diffusion_retrain_aligned_b10_306801_seed3407_epoch10/phase1_pretrain/training_log/original-diffusion-aligned-b10-phase1/2026-08-27-14:31:09/args.json
- planner.diffusion_planner.ckpt_path=/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/original_diffusion_retrain_aligned_b10_306801_seed3407_epoch10/phase1_pretrain/training_log/original-diffusion-aligned-b10-phase1/2026-08-27-14:31:09/model_epoch_10_trainloss_0.0618.pth
- scenario_builder=nuplan
- scenario_builder.db_files=/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/test14_full_remote_subset/data/cache/test14
- scenario_filter=test14-hard-aligned-original-recovery-chunk-006
- experiment_uid=aligned-original-dp-test14-hard
- worker=single_machine_thread_pool
- worker.max_workers=1
- worker.use_process_pool=false
- number_of_gpus_allocated_per_simulation=1
- number_of_cpus_allocated_per_simulation=1
- max_callback_workers=1
- disable_callback_parallelization=true
- enable_simulation_progress_bar=true
- verbose=true
- ~callback.simulation_log_callback
```

#### HDP B Epoch10

来源：`HDP-nuplan/tmp/test14_full_remote_subset/three_model_eval/exp/simulation/closed_loop_nonreactive_agents/hdp-b-epoch10-test14-hard/code/hydra/overrides.yaml`
SHA256：`eaa35041e410f85253644a9ed2892ee7819e33323fbb31eb2ba5a4b1755512e1`

```yaml
- +simulation=closed_loop_nonreactive_agents
- planner=hyper_diffusion_planner
- planner.hyper_diffusion_planner.config.args_file=/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_train_full_306801_seed3407_v1/experiment_b_constant5e5_from_epoch4/args.json
- planner.hyper_diffusion_planner.ckpt_path=/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_train_full_306801_seed3407_v1/experiment_b_constant5e5_from_epoch4/model_epoch_10_trainloss_0.0091.pth
- scenario_builder=nuplan
- scenario_builder.db_files=/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/test14_full_remote_subset/data/cache/test14
- scenario_filter=test14-hard-chunk-006
- experiment_uid=hdp-b-epoch10-test14-hard
- worker=single_machine_thread_pool
- worker.max_workers=1
- worker.use_process_pool=false
- number_of_gpus_allocated_per_simulation=1
- number_of_cpus_allocated_per_simulation=1
- max_callback_workers=1
- disable_callback_parallelization=true
- enable_simulation_progress_bar=true
- verbose=true
- ~callback.simulation_log_callback
```

#### HDP RL Epoch2（seed 2026）

来源：`HDP-nuplan/tmp/test14_full_remote_subset/three_model_eval/exp/simulation/closed_loop_nonreactive_agents/hdp-rl-epoch2-test14-hard/code/hydra/overrides.yaml`
SHA256：`5cb9c664f8f30d6e411c667a12aea92a51ab5478b93343df7d6822f7cea7a602`

```yaml
- +simulation=closed_loop_nonreactive_agents
- planner=hyper_diffusion_planner
- planner.hyper_diffusion_planner.config.args_file=/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/rl_safetygate_ttc1_anchor01_10k_seed2026_from_b10/training_log/hdp-rl-safetygate_ttc1_anchor01_10k_seed2026-from-full-mini-b10/2026-08-22-12:22:45/args.json
- planner.hyper_diffusion_planner.ckpt_path=/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/rl_safetygate_ttc1_anchor01_10k_seed2026_from_b10/training_log/hdp-rl-safetygate_ttc1_anchor01_10k_seed2026-from-full-mini-b10/2026-08-22-12:22:45/model_epoch_2_trainloss_0.0003.pth
- scenario_builder=nuplan
- scenario_builder.db_files=/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/test14_full_remote_subset/data/cache/test14
- scenario_filter=test14-hard-chunk-006
- experiment_uid=hdp-rl-epoch2-test14-hard
- worker=single_machine_thread_pool
- worker.max_workers=1
- worker.use_process_pool=false
- number_of_gpus_allocated_per_simulation=1
- number_of_cpus_allocated_per_simulation=1
- max_callback_workers=1
- disable_callback_parallelization=true
- enable_simulation_progress_bar=true
- verbose=true
- ~callback.simulation_log_callback
```

## 6. 场景分片 YAML 与 manifest

- Test14-hard：272 场景，7 片（40×6+32）。
- Test14-random：261 场景，7 片（40×6+21）。
- B 与 RL 共用普通 hard/random YAML；原始 DP hard 使用 token 集合相同的 recovery YAML。
- 每个 YAML 内完整列出固定 16 位 scenario token，并关闭 shuffle。

| YAML | token 数 | SHA256 |
|---|---:|---|
| `HDP-nuplan/tmp/test14_full_remote_subset/config/scenario_filter/test14-hard-aligned-original-recovery-chunk-000.yaml` | 40 | `8a1e54fe15fa41505539366da09db2fb58940bbd536743711ee26ebca115abbd` |
| `HDP-nuplan/tmp/test14_full_remote_subset/config/scenario_filter/test14-hard-aligned-original-recovery-chunk-001.yaml` | 40 | `9b414ce2523efd0a69391aa3ba4019ded7149e99199c02d36eeac1d2027a2839` |
| `HDP-nuplan/tmp/test14_full_remote_subset/config/scenario_filter/test14-hard-aligned-original-recovery-chunk-002.yaml` | 40 | `dd4490e3d1ddd1f9ee285017813a05e24f4707869c368350851eb50fe47da764` |
| `HDP-nuplan/tmp/test14_full_remote_subset/config/scenario_filter/test14-hard-aligned-original-recovery-chunk-003.yaml` | 40 | `3bd49ade57380933536c04ffca9f8179cf0a4f7f1d80c1cb9f0ad10c3f6564c5` |
| `HDP-nuplan/tmp/test14_full_remote_subset/config/scenario_filter/test14-hard-aligned-original-recovery-chunk-004.yaml` | 40 | `ab73021ba83a5500d84827ed44e7a07caef5e81fe20e5f9f3a1cd0fdf3efec62` |
| `HDP-nuplan/tmp/test14_full_remote_subset/config/scenario_filter/test14-hard-aligned-original-recovery-chunk-005.yaml` | 40 | `ee0f1e690d97429f789e55300c930af38aa9377532179a79a4268f39c94bfc93` |
| `HDP-nuplan/tmp/test14_full_remote_subset/config/scenario_filter/test14-hard-aligned-original-recovery-chunk-006.yaml` | 32 | `056234fc2478f0f6892a10b01de5c428c0612ca5ef42a25a9acf8cfe18db2e00` |
| `HDP-nuplan/tmp/test14_full_remote_subset/config/scenario_filter/test14-hard-chunk-000.yaml` | 40 | `fd4540ee9f48f09f683f369577bd45d22efd80258e546b0230c653c057a7103b` |
| `HDP-nuplan/tmp/test14_full_remote_subset/config/scenario_filter/test14-hard-chunk-001.yaml` | 40 | `5d110f5920a089980375ecf42b2c813856413de4c278f3e75b17f91fc7b15149` |
| `HDP-nuplan/tmp/test14_full_remote_subset/config/scenario_filter/test14-hard-chunk-002.yaml` | 40 | `9a1f0c979e42c44dcfec39629835d74b4cdfabe6ac7418135d1dd0b6b440998e` |
| `HDP-nuplan/tmp/test14_full_remote_subset/config/scenario_filter/test14-hard-chunk-003.yaml` | 40 | `7db59540ebf6caaa85b3a42ea9a5882905bcc08a089258b651af75028e1e7002` |
| `HDP-nuplan/tmp/test14_full_remote_subset/config/scenario_filter/test14-hard-chunk-004.yaml` | 40 | `3676bc29ca1c362ed9733307e10094d5719a06ca5d1a73dc19194549ed03b9e8` |
| `HDP-nuplan/tmp/test14_full_remote_subset/config/scenario_filter/test14-hard-chunk-005.yaml` | 40 | `32d690ccd600e129ddfd25ee80fd9b1bb82dcc43c5c8f49018fe1af84020f43d` |
| `HDP-nuplan/tmp/test14_full_remote_subset/config/scenario_filter/test14-hard-chunk-006.yaml` | 32 | `b122f74ab7cdd5f0aa6808d6191a719095f8a87f0931d3916ad7fadaa01e83b0` |
| `HDP-nuplan/tmp/test14_full_remote_subset/config/scenario_filter/test14-random-chunk-000.yaml` | 40 | `1a5149747bdd0f80f04a791559e29559a26e232f13fb06abeba2685556e89e10` |
| `HDP-nuplan/tmp/test14_full_remote_subset/config/scenario_filter/test14-random-chunk-001.yaml` | 40 | `43a37db4bd83129db95f81b7fbfb2bd610bb8c30f8468ab4c47dee4ac642cc3a` |
| `HDP-nuplan/tmp/test14_full_remote_subset/config/scenario_filter/test14-random-chunk-002.yaml` | 40 | `c2970cd5dcdee376f30aef3ad6937ccbfb2a580acbfa789c75570362826bf058` |
| `HDP-nuplan/tmp/test14_full_remote_subset/config/scenario_filter/test14-random-chunk-003.yaml` | 40 | `beb3928d7de0edab968f9b9b241737215ae3a573b2aede445b74f50f02189f8b` |
| `HDP-nuplan/tmp/test14_full_remote_subset/config/scenario_filter/test14-random-chunk-004.yaml` | 40 | `f5c90677555d99b3be3dcfc045b9d5bd42d5590db652d5bac3cf63c90293f8e1` |
| `HDP-nuplan/tmp/test14_full_remote_subset/config/scenario_filter/test14-random-chunk-005.yaml` | 40 | `66bc75a43c203c9b5afeab5f358599b57e8741a216ff957a1954c4e61cec7f8f` |
| `HDP-nuplan/tmp/test14_full_remote_subset/config/scenario_filter/test14-random-chunk-006.yaml` | 21 | `bef632129bbb1a1c66fc58c470bfb48cbfcf54cecee4566748a27e8cb5336a76` |

Manifest：

- `HDP-nuplan/tmp/test14_full_remote_subset/eval_chunk_manifest.json`；SHA256 `a6b495893945d719055d9841a91e1e637063da7f3d6b439c5e82077d5964230d`
- `HDP-nuplan/tmp/test14_full_remote_subset/aligned_original_hard_recovery_manifest.json`；SHA256 `e3abfe03c8ccd766ba2a100fa592761ca767b1fa2e711eb62342546840e4faa5`

## 7. 数据与归一化文件

| 文件 | 用途/条目数 | SHA256 |
|---|---|---|
| `normalization.json` | 原始 DP normalization | `2772eb0bacb57164aeb7ebb2c1a5e6c8410b9a6c3ee64acd78e767c3415f39c6` |
| `HDP-nuplan/normalization.json` | HDP B/RL normalization | `c36ccb9807a64fe75ea3f43c1b169a076e6824f194512e09d46788a8a0158a5a` |
| `HDP-nuplan/tmp/mini_train_full_306801_seed3407_v1/diffusion_planner_training.json` | 306,801 条训练样本 | `a8a5c4a3ebba0f127e197d86c3f899940e8bdc3a7dbbc361b9a51bb2b8f70ab0` |
| `HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/diffusion_planner_training.json` | 10,000 条训练样本 | `1597c2f63bbcba7bdc7ed7e5e357cac059e283c84aab6f418ff153c182bdc514` |

## 8. 关键辨析

1. checkpoint 文件内没有 args；它只保存 epoch、model、EMA、optimizer、schedule、loss 和 wandb_id。
2. 闭环推理读取外部 args.json 来构造网络，再从 checkpoint 加载 EMA 权重。
3. `train_epochs`、`batch_size`、RL reward 等训练字段在推理时大多不参与 forward，但仍保留在 args 中。
4. B Epoch10 当前 args 的 `train_epochs=20` 不能解释成被评测模型训练了 20 epoch；checkpoint
   元数据和文件名均确认被评测权重是 Epoch10。
5. Test14-hard/random 的三模型对比使用相同 token 集合、闭环模式和指标配置；模型相关差异
   仅为模型类、args 和 checkpoint。
