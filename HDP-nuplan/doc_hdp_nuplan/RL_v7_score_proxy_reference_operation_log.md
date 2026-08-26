# RL v7：NuPlan score proxy v2 与 reference anchor 操作日志

## 1. 实验目的

在不改变训练数据的前提下，修正 RL reward 与 NuPlan 闭环指标之间的目标错位，并限制 RL 模型偏离 B Epoch 10 监督模型的幅度。

## 2. 固定条件

- 训练数据：`HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/cache`
- 数据清单：`HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/diffusion_planner_training.json`
- NPZ 数量：10,000
- RL 起点：`model_epoch_10_trainloss_0.0091.pth`
- seed：2026
- batch size：2
- RL epoch：2
- 学习率：`4e-7`
- `rl_buffer_size=10000`
- `rl_detach_window_size=0`
- 编码器冻结：开启

启动脚本会在训练前检查 NPZ、manifest 和 checkpoint；数量或 checkpoint 结构不匹配时直接退出。

## 3. 代码改动

### 3.1 新增 `nuplan_score_proxy_v2`

保留原有 `legacy` 和 `nuplan_aligned`，新增独立 reward 模式，不影响旧实验复现。

新模式显式使用以下候选质量：

- 风险与无碰撞质量；
- 相对专家路线进度；
- 真实 `route_cost` 转换得到的路线质量；
- `comfort_cost` 转换得到的舒适性质量；
- 跟车质量。

这些质量项经过归一化后使用加权几何平均，避免高进度候选抵消安全或路线退化。

### 3.2 新增 B Epoch 10 reference anchor

新增参数：

```text
rl_reference_anchor_weight=0.05
```

训练开始时从 B Epoch 10 checkpoint 复制一份冻结 reference model。在相同的扩散状态、扩散时间和监督空间下，约束当前模型输出不要过度偏离起点模型。

### 3.3 关闭候选硬筛选作为控制变量

本轮设置：

```text
rl_filter_safety_eligible_candidates=false
rl_filter_progress_guard_candidates=false
reward_safety_gate_min_ttc_seconds=1.0
```

保留 reward safety gate 对不安全候选的降权，但不再因为候选未通过硬筛选而直接删除，从而检验 v6 中约 51.4% 无有效候选导致的训练样本稀疏问题。

## 4. 启动命令

```bash
bash HDP-nuplan/scripts/run_rl_v7_score_proxy_reference_10k_seed2026.sh
```

训练日志：

```text
HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/rl_score_proxy_v2_reference_10k_seed2026_from_b10/rl_train.log
```

## 5. 验证记录

### 5.1 代码检查

```bash
python -m py_compile \
  HDP-nuplan/hdp_nuplan/rl/reward.py \
  HDP-nuplan/hdp_nuplan/rl/loss.py \
  HDP-nuplan/hdp_nuplan/rl/train_epoch_rl.py \
  HDP-nuplan/train_predictor_rl.py
```

结果：通过。

### 5.2 单元测试

```bash
/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python \
  -m pytest -q HDP-nuplan/tests
```

结果：`55 passed`。

### 5.3 reward sanity check

使用当前 10,000 NPZ 清单中的前 20 个样本验证新 reward：

```bash
/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python \
  HDP-nuplan/scripts/validate_reward_v2.py \
  --cache-dir HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/cache \
  --manifest HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/diffusion_planner_training.json \
  --output HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/reward_proxy_v2_sanity.json \
  --objective-mode nuplan_score_proxy_v2 \
  --max-scenes 20
```

结果：`accepted=true`；路线偏离、舒适性、倒退、停止、碰撞等检查均通过。

## 6. 待记录结果

### 6.1 训练结果

训练于 2026-08-25 01:33:25 CST 完成，checkpoint 为：

```text
HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/rl_score_proxy_v2_reference_10k_seed2026_from_b10/training_log/hdp-rl-score_proxy_v2_reference_10k_seed2026-from-full-mini-b10/2026-08-25-01:22:17/model_epoch_2_trainloss_0.0019.pth
```

关键结果：

| 指标 | 数值 |
|---|---:|
| update steps | 500 |
| total loss | 0.0018521 |
| reward-weighted diffusion loss | 0.0008917 |
| reward-weighted waypoint loss | 0.0610811 |
| reward mean | 0.5203765 |
| reward std p10 / p50 / p90 | 0.0006287 / 0.0022155 / 0.0038023 |
| active group fraction | 0.973 |
| eligible candidate fraction | 1.0 |
| no-eligible-group fraction | 0.0 |
| reference anchor loss | 7.0537e-6 |
| weighted reference anchor loss | 3.5268e-7 |
| weighted expert anchor loss | 3.4921e-4 |

训练过程无异常退出；训练集清单和 NPZ 数量未改变。

### 6.2 fixed200 评测

使用与此前 B Epoch 10 / RL 对照相同的 `mini-val-fixed-200-rl-v4` 过滤器，启动命令为：

```bash
bash HDP-nuplan/scripts/run_fixed_200_eval_rl_v7.sh
```

评测日志：

```text
HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/rl_score_proxy_v2_reference_10k_seed2026_from_b10/fixed200_eval/fixed200_eval.log
```

启动时 NuPlan 成功构建 200 个场景，先评测 B Epoch 10，再评测 v7。最终指标待闭环运行结束后补充。

### 6.3 评测流程修正

启动后复核发现，历史文件

```text
HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/rl_v4_fixed200_eval_retry/b10_vs_rl_fixed200.json
```

已经包含 B Epoch 10 的完整 `200/200` 结果，且其中的 200 个 scenario token 与
`mini-val-fixed-200-rl-v4.yaml` 完全一致。因此停止了重复的 B Epoch 10 评测，后续直接复用该基线，只运行 v7。

为缩短闭环时间，v7 改用已在 test14-random 评测中验证过的单机双线程调度：

```text
worker=single_machine_thread_pool
worker.max_workers=2
worker.use_process_pool=false
number_of_gpus_allocated_per_simulation=0.5
```

新的启动入口：

```bash
bash HDP-nuplan/scripts/evaluate_rl_v7_fixed200_parallel2.sh
```

该脚本只改变场景执行调度，不改变场景 token、planner、checkpoint、NuPlan metric 或训练集；评测完成后把历史 B Epoch 10 汇总与 v7 汇总合并，再做逐场景配对分析。

### 6.4 fixed200 最终结果

并行闭环于 2026-08-25 04:29:04 CST 完成：`200/200` 成功，`0` 失败。NuPlan 的直方图渲染收尾没有执行完，导致首次后台脚本未继续生成汇总；`runner_report.parquet` 和 aggregator parquet 均已完整保存，因此重新运行评测脚本。脚本检测到现有 report 后跳过仿真，只执行汇总和配对分析，于 04:38:40 CST 完成。

| 指标 | B Epoch 10 | RL v7 | v7 - B10 |
|---|---:|---:|---:|
| score | 0.90456672 | 0.90168001 | -0.00288672 |
| no ego at-fault collision | 0.95750000 | 0.95750000 | 0.00000000 |
| drivable-area compliance | 0.95000000 | 0.95000000 | 0.00000000 |
| making progress | 0.99000000 | 0.99000000 | 0.00000000 |
| driving-direction compliance | 1.00000000 | 1.00000000 | 0.00000000 |
| route progress | 0.91206773 | 0.90178298 | -0.01028475 |
| TTC | 0.93500000 | 0.93500000 | 0.00000000 |
| speed-limit compliance | 0.99543114 | 0.99595502 | +0.00052388 |
| comfort | 0.99500000 | 0.99500000 | 0.00000000 |

score 的逐场景胜/负/平为 `4/93/103`。本轮虽然略微改善限速指标，并保持碰撞、可行驶区域、TTC 和舒适性不变，但路线进度下降约 0.01028，最终总 score 为负收益，因此 v7 不应作为后续扩展评测的候选模型。

完整配对结果：

```text
HDP-nuplan/tmp/mini_train_balanced_10000_seed3407_v1/rl_score_proxy_v2_reference_10k_seed2026_from_b10/fixed200_eval_parallel2/b10_vs_rl_v7_fixed200_analysis.md
```
