#!/usr/bin/env python3
"""Wait for C0, audit Epoch10, and run paired open-loop evaluation automatically."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

import torch


HDP_ROOT = Path(__file__).resolve().parents[1]
PYTHON_BIN = Path("/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python")
C0_ROOT = HDP_ROOT / "tmp/chapter3_hybrid_ablation_v1/C0"
OUTPUT_ROOT = HDP_ROOT / "tmp/chapter3_hybrid_ablation_v1/post_c0"
HISTORICAL_B10 = (
    HDP_ROOT
    / "tmp/mini_train_full_306801_seed3407_v1"
    / "experiment_b_constant5e5_from_epoch4/model_epoch_10_trainloss_0.0091.pth"
)
VAL_ROOT = HDP_ROOT / "tmp/mini_val_balanced_1000_seed3407_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-pid", required=True, type=int)
    parser.add_argument("--launcher-pid", required=True, type=int)
    parser.add_argument("--poll-seconds", default=30, type=int)
    return parser.parse_args()


def now() -> str:
    return datetime.now().astimezone().isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def process_exists(pid: int) -> bool:
    stat_path = Path(f"/proc/{pid}/stat")
    try:
        stat_text = stat_path.read_text(encoding="utf-8")
        # The command name is parenthesized and may contain spaces.  The state
        # is the first field after the final ``)``.  A zombie cannot make more
        # progress and should be treated as finished even before its parent
        # reaps it.
        state = stat_text.rsplit(")", 1)[1].strip().split()[0]
        if state == "Z":
            return False
        os.kill(pid, 0)
    except (FileNotFoundError, ProcessLookupError):
        return False
    return True


def stop_staged_launcher(pid: int) -> None:
    if process_exists(pid):
        os.kill(pid, signal.SIGKILL)
        for _ in range(50):
            if not process_exists(pid):
                return
            time.sleep(0.1)
        raise RuntimeError(f"launcher PID {pid} did not exit after SIGKILL")


def exactly_one(pattern: str, label: str) -> Path:
    matches = sorted(C0_ROOT.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {label}, found {len(matches)}: {matches}")
    return matches[0]


def audit_checkpoint(checkpoint_path: Path) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    required = {"epoch", "model", "ema_state_dict", "optimizer", "schedule", "loss"}
    missing = sorted(required - set(checkpoint))
    if missing:
        raise RuntimeError(f"C0 Epoch10 checkpoint missing keys: {missing}")
    if checkpoint["epoch"] != 10:
        raise RuntimeError(f"expected C0 epoch=10, got {checkpoint['epoch']!r}")
    if not math.isfinite(float(checkpoint["loss"])):
        raise RuntimeError(f"C0 Epoch10 has non-finite loss: {checkpoint['loss']!r}")
    if len(checkpoint["model"]) != 260 or len(checkpoint["ema_state_dict"]) != 260:
        raise RuntimeError(
            "unexpected state-dict size: "
            f"model={len(checkpoint['model'])}, ema={len(checkpoint['ema_state_dict'])}"
        )
    return {
        "status": "passed",
        "checkpoint": str(checkpoint_path.resolve()),
        "sha256": sha256(checkpoint_path),
        "epoch": checkpoint["epoch"],
        "train_loss": float(checkpoint["loss"]),
        "model_tensor_count": len(checkpoint["model"]),
        "ema_tensor_count": len(checkpoint["ema_state_dict"]),
        "historical_b10": str(HISTORICAL_B10.resolve()),
        "historical_b10_sha256": sha256(HISTORICAL_B10),
        "audited_at": now(),
    }


def render_summary(result: dict, checkpoint_audit: dict) -> str:
    reference = result["protocol"]["paired_reference"]
    candidate = "historical_B10"
    models = result["models"]
    paired = result["paired_delta_candidate_minus_reference"][candidate]
    primary_metrics = [
        "ade_m",
        "fde_m",
        "ade_1s_m",
        "fde_1s_m",
        "ade_3s_m",
        "fde_3s_m",
        "ade_5s_m",
        "fde_5s_m",
        "ade_8s_m",
        "fde_8s_m",
        "heading_mae_rad",
        "heading_fde_rad",
    ]
    descriptor_metrics = [
        "mean_speed_mps",
        "max_speed_mps",
        "mean_acceleration_mps2",
        "max_acceleration_mps2",
        "mean_jerk_mps3",
        "max_jerk_mps3",
        "mean_abs_curvature_1pm",
        "max_abs_curvature_1pm",
    ]
    lower_count = sum(paired[name]["mean"] < 0 for name in primary_metrics)
    lines = [
        "# 第三章 C0 与历史B10配对开环结果",
        "",
        f"> 自动生成时间：{now()}  ",
        "> 证据等级：历史对照准消融；不是严格单变量因果实验。",
        "",
        "## 1. 协议与身份",
        "",
        f"- 开发验证集：{result['protocol']['dataset_samples']}个场景；训练集token交集为0。",
        f"- 每模型观测数：{result['protocol']['observations_per_model']}（3次重复）。",
        f"- 公共随机数：{result['protocol']['common_random_numbers']}，seed={result['protocol']['seed']}。",
        f"- 扩散采样：{result['protocol']['diffusion_steps']}步，noise scale={result['protocol']['sampling_noise_scale']}。",
        f"- C0 Epoch10 SHA256：`{checkpoint_audit['sha256']}`。",
        f"- 历史B10 SHA256：`{checkpoint_audit['historical_b10_sha256']}`。",
        "- 两模型均从checkpoint的EMA权重评价。",
        "",
        "## 2. 主要误差指标",
        "",
        "下表差值为历史B10−C0；对误差指标而言，负值表示历史B10更低。",
        "",
        "| 指标 | C0均值 | 历史B10均值 | B10−C0 | B10改善样本比例 |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric in primary_metrics:
        delta = paired[metric]
        lines.append(
            f"| {metric} | {models[reference][metric]['mean']:.6f} | "
            f"{models[candidate][metric]['mean']:.6f} | {delta['mean']:+.6f} | "
            f"{delta['candidate_improved_fraction']:.2%} |"
        )
    lines += [
        "",
        "## 3. 运动学描述量",
        "",
        "这些是有限差分描述量，不是安全或动力学可行性保证；数值更高不自动代表更好。",
        "",
        "| 指标 | C0均值 | 历史B10均值 | B10−C0 |",
        "|---|---:|---:|---:|",
    ]
    for metric in descriptor_metrics:
        delta = paired[metric]
        lines.append(
            f"| {metric} | {models[reference][metric]['mean']:.6f} | "
            f"{models[candidate][metric]['mean']:.6f} | {delta['mean']:+.6f} |"
        )
    lines += [
        "",
        "## 4. 自动事实摘要与结论边界",
        "",
        f"历史B10在{len(primary_metrics)}个预先列出的主要误差指标中，有{lower_count}个均值低于C0。",
        "该计数只是描述性结果，不提供显著性检验，也不能证明差异由hybrid loss单独造成。",
        "历史B10缺少与C0对应的训练前完整初始化hash，并经历过分叉和恢复训练。",
        "因此即使所有误差方向一致，也只能表述为‘结果与位置重建辅助监督有效的假设一致’。",
        "是否进入固定40场景闭环评价，应结合长时域ADE/FDE、改善样本比例和运动学描述量共同决定。",
        "",
        "## 5. 原始证据",
        "",
        f"- 评价JSON：`{(OUTPUT_ROOT / 'open_loop_results.json').resolve()}`",
        f"- checkpoint审计：`{(OUTPUT_ROOT / 'checkpoint_audit.json').resolve()}`",
        f"- 执行日志：`{(OUTPUT_ROOT / 'evaluation.log').resolve()}`",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.poll_seconds < 1 or args.wait_pid <= 0 or args.launcher_pid <= 0:
        raise ValueError("PIDs and poll-seconds must be positive")
    status_path = OUTPUT_ROOT / "status.json"
    status = {
        "state": "waiting_for_c0",
        "wait_pid": args.wait_pid,
        "launcher_pid": args.launcher_pid,
        "started_at": now(),
    }
    atomic_write_json(status_path, status)
    try:
        while process_exists(args.wait_pid):
            time.sleep(args.poll_seconds)
        time.sleep(5)
        stop_staged_launcher(args.launcher_pid)

        status.update({"state": "auditing_c0", "c0_process_finished_at": now()})
        atomic_write_json(status_path, status)
        checkpoint_path = exactly_one("training_log/*/*/model_epoch_10_*.pth", "C0 Epoch10 checkpoint")
        args_file = exactly_one("training_log/*/*/args.json", "C0 args.json")
        if not HISTORICAL_B10.is_file():
            raise FileNotFoundError(HISTORICAL_B10)
        for required in (
            VAL_ROOT / "cache",
            VAL_ROOT / "diffusion_planner_validation.json",
            VAL_ROOT / "cache_validation_report.json",
            PYTHON_BIN,
        ):
            if not required.exists():
                raise FileNotFoundError(required)
        checkpoint_audit = audit_checkpoint(checkpoint_path)
        atomic_write_json(OUTPUT_ROOT / "checkpoint_audit.json", checkpoint_audit)

        output_json = OUTPUT_ROOT / "open_loop_results.json"
        command = [
            str(PYTHON_BIN),
            str(HDP_ROOT / "scripts/evaluate_supervised_open_loop.py"),
            "--args-file",
            str(args_file),
            "--model",
            f"C0={checkpoint_path}",
            "--model",
            f"historical_B10={HISTORICAL_B10}",
            "--data-dir",
            str(VAL_ROOT / "cache"),
            "--data-list",
            str(VAL_ROOT / "diffusion_planner_validation.json"),
            "--batch-size",
            "16",
            "--num-workers",
            "2",
            "--repeats",
            "3",
            "--seed",
            "3407",
            "--num-samples",
            "1",
            "--diffusion-steps",
            "10",
            "--sampling-noise-scale",
            "0.1",
            "--device",
            "cuda",
            "--output",
            str(output_json),
        ]
        status.update({"state": "evaluating_open_loop", "command": command, "evaluation_started_at": now()})
        atomic_write_json(status_path, status)
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(HDP_ROOT)
        with (OUTPUT_ROOT / "evaluation.log").open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                cwd=HDP_ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                text=True,
            )
        if completed.returncode != 0:
            raise RuntimeError(f"open-loop evaluation exited with {completed.returncode}")
        result = json.loads(output_json.read_text(encoding="utf-8"))
        atomic_write_text(
            OUTPUT_ROOT / "open_loop_summary.md",
            render_summary(result, checkpoint_audit),
        )
        status.update(
            {
                "state": "complete",
                "finished_at": now(),
                "open_loop_results_sha256": sha256(output_json),
                "summary_sha256": sha256(OUTPUT_ROOT / "open_loop_summary.md"),
            }
        )
        atomic_write_json(status_path, status)
    except Exception as error:
        status.update(
            {
                "state": "failed",
                "failed_at": now(),
                "error": repr(error),
                "traceback": traceback.format_exc(),
            }
        )
        atomic_write_json(status_path, status)
        raise


if __name__ == "__main__":
    main()
