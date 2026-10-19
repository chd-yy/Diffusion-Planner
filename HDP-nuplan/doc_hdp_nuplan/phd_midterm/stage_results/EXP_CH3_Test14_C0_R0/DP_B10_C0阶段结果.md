# DP、B10、C0完整Test14阶段结果

> 结果固定时间：10月06日 09:26（北京时间）  
> 评测模式：`closed_loop_nonreactive_agents`，推理seed=0，单simulation worker。  
> 正式范围：DP、B10、C0；R0不完整评测不纳入结论。

## 1. 数据完整性

| 集合 | 场景记录数 | DP | B10 | C0 |
|---|---:|---:|---:|---:|
| Test14-hard | 272 | 272/272 | 272/272 | 272/272 |
| Test14-random | 261 | 261/261 | 261/261 | 261/261 |

三个模型在两个集合上均无仿真执行失败。这里的“无仿真失败”只说明pipeline成功完成，不等于无碰撞、无越界或所有场景得分非零。hard与random合计533条评测记录，但因两个集合存在11个重复token，只能作为分集合结果或按记录数加权的描述量，不能写成533个独立场景。

## 2. Test14-hard

| 模型 | Score | NC | DA | Making progress | Route progress | TTC | Comfort |
|---|---:|---:|---:|---:|---:|---:|---:|
| DP | 0.602691 | 0.779412 | 0.886029 | 0.915441 | 0.809441 | 0.713235 | 0.886029 |
| B10 | 0.705304 | **0.867647** | 0.944853 | **0.948529** | 0.819141 | 0.761029 | **0.977941** |
| C0 | **0.709170** | 0.852941 | **0.963235** | 0.941176 | **0.829728** | **0.768382** | 0.963235 |

## 3. Test14-random

| 模型 | Score | NC | DA | Making progress | Route progress | TTC | Comfort |
|---|---:|---:|---:|---:|---:|---:|---:|
| DP | 0.689908 | 0.800766 | 0.892720 | **0.992337** | 0.900808 | 0.766284 | 0.869732 |
| B10 | 0.824270 | 0.908046 | 0.965517 | 0.980843 | 0.910270 | **0.869732** | **0.977011** |
| C0 | **0.837819** | **0.925287** | **0.977011** | 0.977011 | **0.911319** | 0.865900 | 0.973180 |

## 4. 当前可陈述的阶段结论

1. B10与C0在hard和random上的总分都明显高于本地重训DP，说明HDP路线的系统级闭环性能优势在两个集合上方向一致。
2. C0总分在两个集合上都略高于B10：hard为`+0.003866`，random为`+0.013549`；按533条记录数加权的描述性总分为DP `0.645400`、B10 `0.763559`、C0 `0.772167`。
3. C0并非所有安全与驾驶质量分项都优于B10。hard上B10的NC、making progress和comfort更好；random上B10的TTC、making progress和comfort略好。这说明总体分数接近时仍存在安全、进度与舒适性的结构性权衡。
4. C0与历史B10不是严格单因素训练实验，因此不能把二者差异全部归因于Hybrid Loss。严格的Hybrid因果结论仍应以C0/C2等受控实验为主；本表只提供系统级外部有效性证据。
5. 当前未获得完整R0 Test14数据，因此本结果不能回答absolute与increment轨迹表示谁在系统闭环上更优。

## 5. 原始证据

- [Test14-hard三模型逐场景结果](./test14_hard_DP_B10_C0.json)
- [Test14-random三模型逐场景结果](./test14_random_DP_B10_C0.json)
- [实验协议与运行状态](./实验协议与运行状态.md)
- C0 hard合并校验：`HDP-nuplan/tmp/chapter3_test14_c0_r0_v1/exp/simulation/closed_loop_nonreactive_agents/C0-test14-hard/chunk_merge_validation.json`
- C0 random合并校验：`HDP-nuplan/tmp/chapter3_test14_c0_r0_v1/exp/simulation/closed_loop_nonreactive_agents/C0-test14-random/chunk_merge_validation.json`
