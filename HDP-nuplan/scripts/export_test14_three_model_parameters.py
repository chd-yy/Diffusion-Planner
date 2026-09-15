#!/usr/bin/env python3
"""Export the complete parameter/config inventory for the final Test14 comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HDP_ROOT = PROJECT_ROOT / "HDP-nuplan"
EVAL_ROOT = HDP_ROOT / "tmp/test14_full_remote_subset/three_model_eval"
RUN_ROOT = EVAL_ROOT / "exp/simulation/closed_loop_nonreactive_agents"

MODELS: Dict[str, Dict[str, Any]] = {
    "aligned_original_dp": {
        "title": "对齐训练的原始 Diffusion Planner Epoch10",
        "planner": "diffusion_planner",
        "planner_class": "diffusion_planner.planner.planner.DiffusionPlanner",
        "args": HDP_ROOT
        / "tmp/original_diffusion_retrain_aligned_b10_306801_seed3407_epoch10"
        / "phase1_pretrain/training_log/original-diffusion-aligned-b10-phase1"
        / "2026-08-27-14:31:09/args.json",
        "checkpoint": HDP_ROOT
        / "tmp/original_diffusion_retrain_aligned_b10_306801_seed3407_epoch10"
        / "phase1_pretrain/training_log/original-diffusion-aligned-b10-phase1"
        / "2026-08-27-14:31:09/model_epoch_10_trainloss_0.0618.pth",
        "planner_yaml": PROJECT_ROOT / "diffusion_planner/config/planner/diffusion_planner.yaml",
        "train_entry": PROJECT_ROOT / "train_predictor.py",
        "train_scripts": [
            HDP_ROOT / "scripts/train_original_diffusion_same_data.sh",
            HDP_ROOT / "scripts/resume_original_diffusion_aligned_epoch10_and_eval.sh",
        ],
        "run_uids": {
            "test14-hard": "aligned-original-dp-test14-hard",
            "test14-random": "aligned-original-dp-test14-random",
        },
        "parameter_count": 6_042_628,
        "note": (
            "该 args.json 是最后一次从 Epoch9 恢复并完成 Epoch10 时写入的实际推理配置；"
            "初始 Encoder warm-start 与两阶段学习率需结合训练脚本查看。"
        ),
    },
    "hdp_b_epoch10": {
        "title": "HDP B Epoch10",
        "planner": "hyper_diffusion_planner",
        "planner_class": "hdp_nuplan.planner.planner.HyperDiffusionPlanner",
        "args": HDP_ROOT / "tmp/mini_train_full_306801_seed3407_v1/experiment_b_constant5e5_from_epoch4/args.json",
        "checkpoint": HDP_ROOT
        / "tmp/mini_train_full_306801_seed3407_v1/experiment_b_constant5e5_from_epoch4"
        / "model_epoch_10_trainloss_0.0091.pth",
        "planner_yaml": HDP_ROOT / "hdp_nuplan/config/planner/hyper_diffusion_planner.yaml",
        "train_entry": HDP_ROOT / "train_predictor.py",
        "train_scripts": [
            HDP_ROOT / "scripts/switch_epoch9_a_to_constant5e5_b.sh",
            HDP_ROOT / "scripts/resume_experiment_b_epoch10_to20.sh",
        ],
        "run_uids": {
            "test14-hard": "hdp-b-epoch10-test14-hard",
            "test14-random": "hdp-b-epoch10-test14-random",
        },
        "parameter_count": 5_092_996,
        "note": (
            "该目录后来从 Epoch10 续训到 Epoch20，args.json 于 2026-08-28 再次写入，"
            "所以其中 train_epochs=20/name=...epoch20 是后续值；本次评测 checkpoint 仍严格为 Epoch10。"
        ),
    },
    "hdp_rl_epoch2": {
        "title": "HDP RL Epoch2（seed 2026）",
        "planner": "hyper_diffusion_planner",
        "planner_class": "hdp_nuplan.planner.planner.HyperDiffusionPlanner",
        "args": HDP_ROOT
        / "tmp/mini_train_balanced_10000_seed3407_v1/rl_safetygate_ttc1_anchor01_10k_seed2026_from_b10"
        / "training_log/hdp-rl-safetygate_ttc1_anchor01_10k_seed2026-from-full-mini-b10"
        / "2026-08-22-12:22:45/args.json",
        "checkpoint": HDP_ROOT
        / "tmp/mini_train_balanced_10000_seed3407_v1/rl_safetygate_ttc1_anchor01_10k_seed2026_from_b10"
        / "training_log/hdp-rl-safetygate_ttc1_anchor01_10k_seed2026-from-full-mini-b10"
        / "2026-08-22-12:22:45/model_epoch_2_trainloss_0.0003.pth",
        "planner_yaml": HDP_ROOT / "hdp_nuplan/config/planner/hyper_diffusion_planner.yaml",
        "train_entry": HDP_ROOT / "train_predictor_rl.py",
        "train_scripts": [HDP_ROOT / "scripts/run_rl_v2_safety_gate59_from_b10_pilot1000.sh"],
        "run_uids": {
            "test14-hard": "hdp-rl-epoch2-test14-hard",
            "test14-random": "hdp-rl-epoch2-test14-random",
        },
        "parameter_count": 5_092_996,
        "note": "该 args.json 为本次 RL Epoch2 checkpoint 同一次训练产生，未被后续训练覆盖。",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def code(value: Any) -> str:
    rendered = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return f"`{rendered.replace('|', '&#124;')}`"


def append_file_block(lines: List[str], path: Path, language: str) -> None:
    lines.extend(
        [
            f"来源：`{relative(path)}`  ",
            f"SHA256：`{sha256(path)}`",
            "",
            f"```{language}",
            path.read_text(encoding="utf-8").rstrip(),
            "```",
            "",
        ]
    )


def scalar_args(args: Dict[str, Any]) -> Iterable[tuple[str, Any]]:
    for key, value in args.items():
        if key not in {"state_normalizer", "observation_normalizer"}:
            yield key, value


def checkpoint_metadata(path: Path) -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu")
    ema = checkpoint.get("ema_state_dict", {})
    model = checkpoint.get("model", {})
    return {
        "epoch": checkpoint.get("epoch"),
        "loss": checkpoint.get("loss"),
        "top_level_keys": list(checkpoint.keys()),
        "model_tensor_count": len(model),
        "ema_tensor_count": len(ema),
        "ema_parameter_count": sum(value.numel() for value in ema.values()),
    }


def scenario_yaml_paths() -> List[Path]:
    root = HDP_ROOT / "tmp/test14_full_remote_subset/config/scenario_filter"
    paths: List[Path] = []
    for pattern in (
        "test14-hard-aligned-original-recovery-chunk-*.yaml",
        "test14-hard-chunk-*.yaml",
        "test14-random-chunk-*.yaml",
    ):
        paths.extend(sorted(root.glob(pattern)))
    return paths


def build_document() -> str:
    args_by_model = {
        name: json.loads(spec["args"].read_text(encoding="utf-8")) for name, spec in MODELS.items()
    }
    metadata = {name: checkpoint_metadata(spec["checkpoint"]) for name, spec in MODELS.items()}
    lines: List[str] = [
        "# Test14 三模型参数与配置完整清单",
        "",
        "> 范围：2026-09-02 完成的 Test14-hard（272）与 Test14-random（261）正式单 worker 对比。",
        "> 本文区分 checkpoint 的训练形成参数、闭环推理实际参数和 Hydra 最终解析配置。",
        "",
        "## 1. 三个被评测模型与权重",
        "",
        "三个 planner 均使用构造函数默认值 `enable_ema=true`，因此闭环推理实际加载 checkpoint 的",
        "`ema_state_dict`，不是瞬时 `model` 权重。",
        "",
        "| 标识 | 模型 | 参数量 | checkpoint epoch/loss | checkpoint SHA256 |",
        "|---|---|---:|---|---|",
    ]
    for name, spec in MODELS.items():
        meta = metadata[name]
        lines.append(
            f"| `{name}` | {spec['title']} | {spec['parameter_count']:,} | "
            f"{meta['epoch']} / {meta['loss']:.12g} | `{sha256(spec['checkpoint'])}` |"
        )
    lines.extend(
        [
            "",
            "### 1.1 关键参数横向对照",
            "",
            "下表按被评测 checkpoint 的实际形成过程填写；它不是简单照抄后来可能被覆盖的 args.json。",
            "",
            "| 参数 | 对齐原始 DP Epoch10 | HDP B Epoch10 | HDP RL Epoch2 |",
            "|---|---|---|---|",
            "| Planner/模型 | `DiffusionPlanner` | `HyperDiffusionPlanner` | `HyperDiffusionPlanner` |",
            "| 参数量 | 6,042,628 | 5,092,996 | 5,092,996 |",
            "| 训练起点 | 原始 DP Encoder EMA warm-start | 原始 DP Encoder EMA warm-start | HDP B Epoch10 |",
            "| 训练数据 | 完整 mini-train，306,801 NPZ | 完整 mini-train，306,801 NPZ | 固定 10,000 NPZ |",
            "| checkpoint epoch | 10 | 10 | 2（后训练） |",
            "| seed | 3407 | 3407 | 2026 |",
            "| batch size | 8 | 8 | 2 |",
            "| Epoch1～4 学习率 | `5e-4` | 启动基础值 `5e-4`；因断点恢复，checkpoint 实际均为 `5e-5` | 不适用 |",
            "| Epoch5～10/后训练学习率 | `5e-5` | `5e-5` | `4e-7` |",
            "| Encoder | Epoch1～3 冻结，之后解冻 | Epoch1～3 冻结，之后解冻 | 全程冻结 |",
            "| 数据增强 | 开启，概率 `0.5` | 开启，概率 `0.5` | 关闭，概率 `0` |",
            "| diffusion model type | `x_start` | `x_start` | `x_start` |",
            "| 监督/规划损失 | `alpha_planning_loss=1.0` | `planning_hybrid_loss=0.01` | RL 加权回归 + `expert_anchor=0.1` |",
            "| detach window | 不适用 | `0` | `0` |",
            "| EMA | 开启 | 开启 | 开启 |",
            "",
            "### 1.2 文件索引",
            "",
        ]
    )
    for name, spec in MODELS.items():
        lines.extend(
            [
                f"#### {spec['title']}",
                "",
                f"- 模型类：`{spec['planner_class']}`",
                f"- 训练入口：`{relative(spec['train_entry'])}`",
                f"- checkpoint：`{relative(spec['checkpoint'])}`",
                f"- args：`{relative(spec['args'])}`",
                f"- planner YAML：`{relative(spec['planner_yaml'])}`",
                f"- checkpoint keys：`{metadata[name]['top_level_keys']}`",
                f"- `model`/`ema_state_dict` 张量数：{metadata[name]['model_tensor_count']} / "
                f"{metadata[name]['ema_tensor_count']}",
                f"- 说明：{spec['note']}",
                "- 训练/恢复脚本：",
            ]
        )
        lines.extend(f"  - `{relative(path)}`" for path in spec["train_scripts"])
        lines.append("")

    lines.extend(
        [
            "## 2. 训练形成过程",
            "",
            "### 2.1 对齐原始 DP Epoch10",
            "",
            "- 数据：完整 mini-train，306,801 NPZ；seed 3407；batch size 8；EMA 开启。",
            "- Epoch1～4：Encoder 从 `checkpoints/model.pth` 的 EMA 权重 warm-start；加载 151/151 个",
            "  Encoder 张量（1,799,040 参数），Decoder 不加载；Encoder 在 Epoch1～3 冻结；",
            "  `learning_rate=5e-4`，`warm_up_epoch=2`。",
            "- Epoch5～10：从 Epoch4/后续完整 checkpoint 恢复，Encoder 已解冻；",
            "  `learning_rate=5e-5`，`warm_up_epoch=1`，`reset_lr_schedule_on_resume=true`。",
            "- 原始损失：`alpha_planning_loss=1.0`；模型类型 `x_start`。",
            "",
            "### 2.2 HDP B Epoch10",
            "",
            "- 数据：与原始 DP 相同的完整 mini-train 306,801 NPZ；seed 3407；batch size 8；EMA 开启。",
            "- 共同 Epoch1～4 起点：Encoder warm-start，Epoch1～3 冻结；启动参数为",
            "  `learning_rate=5e-4`、`warm_up_epoch=2`。但训练在 Epoch1～4 间多次从保存于",
            "  `scheduler.step()` 之前的 checkpoint 恢复，导致这四个完整 checkpoint 的实际",
            "  optimizer LR 均为 warm-up 起始值 `5e-5`。",
            "- Experiment B 从固定 Epoch4 checkpoint 分叉，Epoch5～10 使用恒定基础学习率 `5e-5`，",
            "  `warm_up_epoch=1`，`reset_lr_schedule_on_resume=true`。",
            "- HDP 损失：`planning_hybrid_loss=0.01`，`planning_detach_window_size=0`，",
            "  `diffusion_model_type=x_start`，`diffusion_supervision_type=x_start`。",
            "- 当前同目录 args 的 Epoch20 字段不改变 Epoch10 checkpoint 内已保存的权重。",
            "",
            "### 2.3 HDP RL Epoch2",
            "",
            "- 起点：B Epoch10；数据：固定 10,000 NPZ；seed 2026；batch size 2。",
            "- 后训练 2 epoch；学习率 `4e-7`；Encoder 冻结；EMA 开启。",
            "- 完整 RL、reward 和 safety-gate 参数见第 3.3 节的 97 项 args。",
            "",
            "## 3. 三个 args.json 的全部字段",
            "",
            "表中值是对应文件当前实际内容；嵌套 normalizer 在每节末完整展开。",
            "",
        ]
    )
    for index, (name, spec) in enumerate(MODELS.items(), 1):
        args = args_by_model[name]
        lines.extend(
            [
                f"### 3.{index} {spec['title']}（{len(args)} 项）",
                "",
                f"文件：`{relative(spec['args'])}`  ",
                f"SHA256：`{sha256(spec['args'])}`",
                "",
                "| 参数 | 值 |",
                "|---|---|",
            ]
        )
        lines.extend(f"| `{key}` | {code(value)} |" for key, value in scalar_args(args))
        for normalizer in ("state_normalizer", "observation_normalizer"):
            if normalizer in args:
                lines.extend(
                    [
                        "",
                        f"#### `{normalizer}`",
                        "",
                        "```json",
                        json.dumps(args[normalizer], ensure_ascii=False, indent=2),
                        "```",
                    ]
                )
        lines.append("")

    lines.extend(
        [
            "## 4. Planner YAML 原文",
            "",
            "原始 DP 与两个 HDP checkpoint 分别使用以下两个 planner YAML；B 与 RL 共用 HDP YAML，",
            "但 `args_file` 和 `ckpt_path` 在运行时被覆盖。",
            "",
            "### 4.1 原始 Diffusion Planner",
            "",
        ]
    )
    append_file_block(lines, MODELS["aligned_original_dp"]["planner_yaml"], "yaml")
    lines.extend(["### 4.2 HDP B / RL", ""])
    append_file_block(lines, MODELS["hdp_b_epoch10"]["planner_yaml"], "yaml")

    lines.extend(
        [
            "## 5. Test14 闭环评测实际覆盖参数",
            "",
            "六组正式运行共享：",
            "",
            "```text",
            "+simulation=closed_loop_nonreactive_agents",
            "scenario_builder=nuplan",
            "worker=single_machine_thread_pool",
            "worker.max_workers=1",
            "worker.use_process_pool=false",
            "number_of_gpus_allocated_per_simulation=1",
            "number_of_cpus_allocated_per_simulation=1",
            "max_callback_workers=1",
            "disable_callback_parallelization=true",
            "enable_simulation_progress_bar=true",
            "verbose=true",
            "~callback.simulation_log_callback",
            "past_trajectory_sampling: num_poses=20, time_horizon=2s",
            "future_trajectory_sampling: num_poses=80, time_horizon=8s",
            "device=cuda",
            "```",
            "",
            "不同模型仅覆盖 planner 类型、args、checkpoint 和 experiment UID；不同 benchmark 仅覆盖",
            "scenario filter。完整执行入口：",
            "",
            f"`{relative(HDP_ROOT / 'scripts/evaluate_full_test14_three_models_chunked.sh')}`",
            "",
            "### 5.1 每个正式运行保存的 Hydra 文件",
            "",
            "`config.yaml` 是 NuPlan/Hydra 最终解析后的完整配置（约 1.6 万行）；`overrides.yaml`",
            "是本次命令行覆盖项；`hydra.yaml` 是 Hydra 自身运行配置。由于同一 UID 分片续跑，",
            "这些快照对应各 UID 最后执行的分片，所有分片 token 则由第 6 节 YAML 完整记录。",
            "",
            "| 模型/Benchmark | 完整配置 | 覆盖项 | Hydra 配置 |",
            "|---|---|---|---|",
        ]
    )
    for name, spec in MODELS.items():
        for benchmark, uid in spec["run_uids"].items():
            hydra_root = RUN_ROOT / uid / "code/hydra"
            lines.append(
                f"| `{name}` / {benchmark} | `{relative(hydra_root / 'config.yaml')}` | "
                f"`{relative(hydra_root / 'overrides.yaml')}` | `{relative(hydra_root / 'hydra.yaml')}` |"
            )

    lines.extend(
        [
            "",
            "### 5.2 三模型 Test14-hard 最终分片 overrides.yaml 原文",
            "",
        ]
    )
    for name, spec in MODELS.items():
        uid = spec["run_uids"]["test14-hard"]
        path = RUN_ROOT / uid / "code/hydra/overrides.yaml"
        lines.extend([f"#### {spec['title']}", ""])
        append_file_block(lines, path, "yaml")

    lines.extend(
        [
            "## 6. 场景分片 YAML 与 manifest",
            "",
            "- Test14-hard：272 场景，7 片（40×6+32）。",
            "- Test14-random：261 场景，7 片（40×6+21）。",
            "- B 与 RL 共用普通 hard/random YAML；原始 DP hard 使用 token 集合相同的 recovery YAML。",
            "- 每个 YAML 内完整列出固定 16 位 scenario token，并关闭 shuffle。",
            "",
            "| YAML | token 数 | SHA256 |",
            "|---|---:|---|",
        ]
    )
    for path in scenario_yaml_paths():
        content = yaml.safe_load(path.read_text(encoding="utf-8"))
        lines.append(
            f"| `{relative(path)}` | {len(content['scenario_tokens'])} | `{sha256(path)}` |"
        )

    manifests = [
        HDP_ROOT / "tmp/test14_full_remote_subset/eval_chunk_manifest.json",
        HDP_ROOT / "tmp/test14_full_remote_subset/aligned_original_hard_recovery_manifest.json",
    ]
    lines.extend(["", "Manifest：", ""])
    for path in manifests:
        lines.append(f"- `{relative(path)}`；SHA256 `{sha256(path)}`")

    normalizations = [PROJECT_ROOT / "normalization.json", HDP_ROOT / "normalization.json"]
    train_manifests = [
        HDP_ROOT / "tmp/mini_train_full_306801_seed3407_v1/diffusion_planner_training.json",
        HDP_ROOT / "tmp/mini_train_balanced_10000_seed3407_v1/diffusion_planner_training.json",
    ]
    lines.extend(
        [
            "",
            "## 7. 数据与归一化文件",
            "",
            "| 文件 | 用途/条目数 | SHA256 |",
            "|---|---|---|",
        ]
    )
    lines.append(
        f"| `{relative(normalizations[0])}` | 原始 DP normalization | `{sha256(normalizations[0])}` |"
    )
    lines.append(
        f"| `{relative(normalizations[1])}` | HDP B/RL normalization | `{sha256(normalizations[1])}` |"
    )
    for path in train_manifests:
        count = len(json.loads(path.read_text(encoding="utf-8")))
        lines.append(f"| `{relative(path)}` | {count:,} 条训练样本 | `{sha256(path)}` |")

    lines.extend(
        [
            "",
            "## 8. 关键辨析",
            "",
            "1. checkpoint 文件内没有 args；它只保存 epoch、model、EMA、optimizer、schedule、loss 和 wandb_id。",
            "2. 闭环推理读取外部 args.json 来构造网络，再从 checkpoint 加载 EMA 权重。",
            "3. `train_epochs`、`batch_size`、RL reward 等训练字段在推理时大多不参与 forward，但仍保留在 args 中。",
            "4. B Epoch10 当前 args 的 `train_epochs=20` 不能解释成被评测模型训练了 20 epoch；checkpoint",
            "   元数据和文件名均确认被评测权重是 Epoch10。",
            "5. Test14-hard/random 的三模型对比使用相同 token 集合、闭环模式和指标配置；模型相关差异",
            "   仅为模型类、args 和 checkpoint。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=HDP_ROOT / "doc_hdp_nuplan/Test14三模型参数与配置完整清单.md",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_document(), encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
