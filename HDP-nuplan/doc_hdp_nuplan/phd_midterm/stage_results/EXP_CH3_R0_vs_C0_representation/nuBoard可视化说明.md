# R0、C0与Expert的nuBoard可视化说明

> 生成时间：10月03日 20:32（北京时间）

## 可视化对象

- 场景token：`b5454dc163675784`
- 场景类型：`high_magnitude_speed`
- 日志：`2021.06.08.12.54.54_veh-26_04262_04732`
- 地图：`us-nv-las-vegas-strip`
- R0/C0来源：validation-1000三次配对推理中的第1次推理（`repeat_index=0`）
- Expert来源：同一nuPlan场景数据库；数据库前8秒相对轨迹、预处理缓存`[x,y,yaw]`和配对评测目标`[x,y,cos(yaw),sin(yaw)]`已完成逐点一致性核验

## 展示语义

本可视化展示同一初始场景下保存的精确8秒开环预测，而不是闭环仿真后的车辆实际执行轨迹。nuBoard中包含两个并排tile：

1. `R0_absolute`：绝对航点表示及原始Diffusion Planner目标归一化；
2. `C0_increment`：增量表示及HDP目标归一化。

每个tile均由nuBoard叠加同一场景的Expert轨迹、地图、交通灯和动态交通参与者。这样可以在保持nuPlan原生场景上下文的同时，对比R0与C0的保存轨迹。

## 产物

- nuBoard入口：`../../../../tmp/chapter3_representation_ablation_v1/nuboard_r0_c0_expert_b5454dc163675784/R0_C0_Expert_b5454dc163675784.nuboard`
- 生成清单：`../../../../tmp/chapter3_representation_ablation_v1/nuboard_r0_c0_expert_b5454dc163675784/manifest.json`
- 可复现导出脚本：`../../../../scripts/export_open_loop_pair_to_nuboard.py`

两个simulation log均通过nuBoard索引和反序列化校验，每个planner轨迹包含当前状态及80个未来状态，共81个状态。数据库Expert与预处理缓存逐元素完全相同，最大绝对差为0。

## 当前启动方式

```bash
env \
  PYTHONPATH=/home/yanjun/NewDisk/nuplan-devkit:/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan:/home/yanjun/NewDisk/Diffusion-Planner \
  NUPLAN_DATA_ROOT=/home/yanjun/NewDisk/nuplan/dataset \
  NUPLAN_MAPS_ROOT=/home/yanjun/NewDisk/nuplan/dataset/maps \
  NUPLAN_EXP_ROOT=/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/chapter3_representation_ablation_v1/nuboard_r0_c0_expert_b5454dc163675784 \
  /home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python \
  /home/yanjun/NewDisk/nuplan-devkit/nuplan/planning/script/run_nuboard.py \
  scenario_builder=nuplan \
  'scenario_builder.db_files=[/home/yanjun/NewDisk/nuplan/dataset/nuplan-v1.1_mini/data/cache/mini/2021.06.08.12.54.54_veh-26_04262_04732.db]' \
  'simulation_path=[/home/yanjun/NewDisk/Diffusion-Planner/HDP-nuplan/tmp/chapter3_representation_ablation_v1/nuboard_r0_c0_expert_b5454dc163675784/R0_C0_Expert_b5454dc163675784.nuboard]' \
  port_number=6599 \
  async_scenario_rendering=false
```

启动后访问 `http://localhost:6599/`，进入 `Scenarios`，选择：

1. Scenario Type：`high_magnitude_speed`；
2. Log Name：`2021.06.08.12.54.54_veh-26_04262_04732`；
3. Scenario Name：`b5454dc163675784`；
4. 点击 `Query Scenario`。

页面会同时显示`R0_absolute`和`C0_increment`；右侧`Trajectory`选项中保留`Expert Trajectory`和`Ego Trajectory`即可完成对比。
