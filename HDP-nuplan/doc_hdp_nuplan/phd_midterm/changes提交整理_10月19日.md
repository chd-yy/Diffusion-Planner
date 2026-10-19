# 当前changes提交整理

> 整理日期：10月19日（北京时间）  
> 分支：`codex/phd-thesis-research`  
> 操作范围：整理ignore、归档图表、修正图像引用并验证；按用户后续要求拆为四批commit，逐批推送至`origin/codex/phd-thesis-research`。

## 1. 本次提交内容

| 类别 | 主要内容 | 路径 |
|---|---|---|
| 轨迹表示与监督训练 | absolute/increment编码与解码、配套归一化入口、相同初始化指纹检查、恢复训练参数留档 | `HDP-nuplan/hdp_nuplan/utils/traj_kinematics.py`、`loss.py`、`train_predictor.py` |
| 安全约束后训练 | 多噪声候选池、88→32 guided safe-set、候选安全/任务掩码、参考策略约束、优势与有效目标门禁 | `HDP-nuplan/hdp_nuplan/rl/`、`model/guidance/`、`train_predictor_rl.py` |
| 全量Replay | 磁盘分片Replay、目标感知采样、目标覆盖、实际更新步数checkpoint | `HDP-nuplan/hdp_nuplan/rl/replay_buffer.py`、`train_epoch_rl.py`、`utils/train_utils.py` |
| 实验工具 | 第三章表示/Hybrid梯度审计、R0训练与Test14补跑、第四章候选/奖励/门控审计及训练、外置存储路径校验 | `HDP-nuplan/scripts/` |
| 回归测试 | 表示变换、多候选采样、安全集合压缩、奖励、安全更新与Replay测试 | `HDP-nuplan/tests/` |
| 实验依据 | 第三章R0/C0/B10结果、Test14逐场JSON与汇总、第四章全量后训练及负结果、PNG/PDF与drawio图 | `HDP-nuplan/doc_hdp_nuplan/` |

结果文档中的“实现完成”“初步实验”“完整评测”和“尚未满足安全接受条件”保持原有证据边界。B10继续作为既定后训练起点与参考策略。根据B10零分场景筛选的子集保留其选择偏差说明，不替代完整Test14结果。

## 2. 不进入Git的本地文件

| 路径 | 原因 | 处理 |
|---|---|---|
| `/P1_reproduce_20260925.zip` | 约120.2 MiB的导入复现包，包含权重等二进制产物 | 精确加入根目录ignore；保留本地文件 |
| `/P1_reproduce_20260925/` | 约137.5 MiB的解压副本，含两份约64.3 MiB的checkpoint、日志及历史脚本/指标 | 精确加入根目录ignore；保留全部文件，不视为已整合进当前源码 |
| `/tmp/` | 仓库根目录下的本地临时文件，本次包含PDF抽取文本 | 加入根目录ignore；保留文件 |
| `/HDP-nuplan/tmp_external` | 指向大体积外置实验存储的本地链接 | 保留当前ignore规则 |
| `HDP-nuplan/tmp/`下新实验缓存与权重 | 已有ignore覆盖；少量历史已跟踪依据不受影响 | 沿用原规则，不删除、不取消历史文件跟踪 |

不使用全局`*.json`、`*.png`、`*.pdf`或`*.log`规则，避免隐藏研究指标、论文图和少量精简日志。本次归档两个仅数KiB的C2审计日志；不归档checkpoint、数据集或完整运行目录。

## 3. 图表归档整理

- R0/C0两份分析文档的14处图像引用原先依赖ignored的`HDP-nuplan/tmp/`，现归档12张唯一PNG到同实验的`figures/`并改为相对链接。
- 原版图与宋体/Abs/Delta标记图均保留；复制前后SHA256一致，没有修改图片内容。
- `EXP_CH4_advantage_boundary_audit`结果文档引用的一张PNG与PDF在当前本地未找到，已将失效链接改为明确的“图文件尚未归档”说明，并保留数值结果、原图说明及制图脚本，未补造图像。
- Markdown中的双空格硬换行保留，使用`.gitattributes`配置对应空白检查；另清理三份新脚本末尾的多余空行。

## 4. 验证结果

| 检查 | 结果 |
|---|---|
| 完整CPU回归：`PYTHONPATH=HDP-nuplan OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' python -m pytest HDP-nuplan/tests -q --disable-warnings --maxfail=5` | 184 passed、1 skipped；跳过项为CUDA mask存储测试 |
| 单独CUDA回归：`python -m pytest HDP-nuplan/tests/test_rl_safety_update.py::test_replay_accepts_cuda_masks_and_stores_them_on_cpu -q --disable-warnings`，同样设置`PYTHONPATH` | 1 passed；补齐上述CUDA测试，合计185个不同测试通过 |
| Python语法检查 | 87个本次变更Python文件均可解析 |
| JSON检查 | 23个本次新增/修改JSON均可解析 |
| Shell检查 | 6个新Shell脚本通过`bash -n` |
| 文档图像检查 | 本次改动Markdown的51处本地图像引用均指向存在且未被ignore的文件 |
| 凭据模式检查 | 本次源码及文本中未匹配到所检查的私钥/GitHub/OpenAI/AWS凭据模式；不等于全面安全审计 |

上述验证不等于重新执行训练或Test14仿真。实验数值沿用已保存的真实结果。

## 5. 分批commit方案

共260个文件，按功能与依赖拆分；不将代码与其新增依赖或测试拆开。运行期使用的定性回放场景YAML随第二批代码提交。

| 批次 | 文件数 | 内容 | Commit标题 |
|---|---:|---|---|
| 1 | 2 | `.gitignore`和Markdown空白规则 | `chore(git): ignore local experiment bundles and scratch files` |
| 2 | 80 | 表示与后训练源码、入口、实验脚本、回放配置和测试 | `feat(hdp): add safe-set post-training and controlled experiment tooling` |
| 3 | 60 | 第三章表示/Hybrid实验、Test14逐场数据、汇总分析与图 | `docs(ch3): archive representation ablations and complete Test14 results` |
| 4 | 118 | 第四章实验与图表、Reward/后训练说明、成果台账与本记录 | `docs(ch4): archive safety post-training experiments and research evidence` |

每批采用普通commit和fast-forward push，不使用force push，不改写历史commit。提交前已确认本地与远程起点均为`b56bdf3`。若某批推送失败，保留该批本地commit及未提交改动，先处理失败原因，再继续后续批次。
