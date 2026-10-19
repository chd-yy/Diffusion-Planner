#!/usr/bin/env python3
"""Wait for R0 training, audit it, then compare R0/C0/B10 on validation-1000."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import subprocess
import time
import traceback

import torch


HDP_ROOT = Path(__file__).resolve().parents[1]
PYTHON_BIN = Path("/home/yanjun/NewDisk/conda_envs/diffusion_planner/bin/python")
R0_ROOT = HDP_ROOT / "tmp/chapter3_representation_ablation_v1/R0_absolute"
OUTPUT_ROOT = HDP_ROOT / "tmp/chapter3_representation_ablation_v1/post_r0"
VAL_ROOT = HDP_ROOT / "tmp/mini_val_balanced_1000_seed3407_v1"
C0_RUN = (
    HDP_ROOT
    / "tmp/chapter3_hybrid_ablation_v1/C0/training_log/"
    "chapter3-C0-hybrid0/2026-09-17-22:47:24"
)
B10_ROOT = HDP_ROOT / "tmp/mini_train_full_306801_seed3407_v1"
B10_ARGS = B10_ROOT / "experiment_b_constant5e5_from_epoch4/args.json"
B10_CHECKPOINT = (
    B10_ROOT
    / "experiment_b_constant5e5_from_epoch4/model_epoch_10_trainloss_0.0091.pth"
)
C0_ARGS = C0_RUN / "args.json"
C0_CHECKPOINT = C0_RUN / "model_epoch_10_trainloss_0.0031.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-pid", required=True, type=int)
    parser.add_argument("--poll-seconds", default=30, type=int)
    parser.add_argument("--timeout-hours", default=12.0, type=float)
    return parser.parse_args()


def now() -> str:
    return datetime.now().astimezone().isoformat()


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def exactly_one(pattern: str, label: str) -> Path:
    matches = sorted(R0_ROOT.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {label}, found {len(matches)}: {matches}")
    return matches[0]


def audit_r0(checkpoint_path: Path, args_path: Path, report_path: Path) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("epoch") != 10:
        raise RuntimeError(f"expected R0 epoch=10, got {checkpoint.get('epoch')!r}")
    if not math.isfinite(float(checkpoint.get("loss"))):
        raise RuntimeError(f"non-finite R0 loss: {checkpoint.get('loss')!r}")
    args = json.loads(args_path.read_text(encoding="utf-8"))
    if args.get("trajectory_representation") != "absolute":
        raise RuntimeError("R0 is not configured for absolute representation")
    if args.get("planning_hybrid_loss") != 0.0:
        raise RuntimeError("R0 hybrid weight is not zero")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected = {
        "model": "0df19230ccd06ae4a25cf37013a662bcc1aa716f05a9150183b1224705262586",
        "encoder": "359289e3b0d7d4bdd8fd3e9316ebc064a038b80a3a00d6cdadf602f60d95c94e",
        "decoder": "5af7b545094a903f473d8c12da06210f93368c288a5c51753b330826824252e3",
    }
    actual = {
        name: report["fingerprints"][name]["sha256"]
        for name in expected
    }
    if actual != expected:
        raise RuntimeError(f"R0 initialization mismatch: {actual}")
    return {
        "status": "passed",
        "checkpoint": str(checkpoint_path),
        "epoch": checkpoint["epoch"],
        "train_loss": float(checkpoint["loss"]),
        "args": str(args_path),
        "initialization_fingerprints": actual,
    }


def render_summary(result: dict) -> str:
    primary = (
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
        "mean_acceleration_mps2",
        "mean_jerk_mps3",
    )
    models = result["models"]
    names = list(models)
    lines = [
        "# 第三章 R0/C0/历史B10 配对开环结果",
        "",
        f"> 完成时间：{datetime.now().astimezone().strftime('%m月%d日 %H:%M')}（北京时间）  ",
        "> R0为绝对waypoint且Hybrid=0；C0为增量且Hybrid=0；历史B10为增量且Hybrid=0.01。  ",
        "> 三者使用各自训练时的目标normalizer，采样后统一恢复成物理空间绝对waypoint再计算指标。",
        "",
        f"- validation场景数：{result['protocol']['dataset_samples']}；",
        f"- 每模型观测数：{result['protocol']['observations_per_model']}；",
        f"- 重复次数：{result['protocol']['repeats']}；",
        "- 相同场景顺序和公共随机数；误差指标均为越低越好。",
        "",
        "| 指标 | " + " | ".join(names) + " |",
        "|---|" + "---:|" * len(names),
    ]
    for metric in primary:
        values = " | ".join(f"{models[name][metric]['mean']:.6f}" for name in names)
        lines.append(f"| {metric} | {values} |")
    lines += [
        "",
        "## 解释边界",
        "",
        "- R0与C0的主要设计差异是XY轨迹表示及其对应目标normalizer，可用于表示方案比较；",
        "- C0与历史B10仍是历史准消融，不能把差异严格归因于Hybrid权重；",
        "- 开环结果不能替代闭环安全评价；是否进入safety40须在本结果生成后再决定；",
        "- 本文件先固定数值事实，不自动把预期假设写成结论。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    status_path = OUTPUT_ROOT / "status.json"
    status = {
        "state": "waiting_for_r0",
        "wait_pid": args.wait_pid,
        "started_at": now(),
    }
    write_json(status_path, status)
    deadline = time.monotonic() + args.timeout_hours * 3600
    try:
        while process_exists(args.wait_pid):
            if time.monotonic() >= deadline:
                raise TimeoutError("timed out waiting for R0 training")
            time.sleep(args.poll_seconds)
        time.sleep(5)

        status.update({"state": "auditing_r0", "training_finished_at": now()})
        write_json(status_path, status)
        r0_checkpoint = exactly_one("training_log/*/*/model_epoch_10_*.pth", "R0 checkpoint")
        r0_args = exactly_one("training_log/*/*/args.json", "R0 args")
        r0_report = exactly_one(
            "training_log/*/*/initialization_report.json",
            "R0 initialization report",
        )
        for required in (
            C0_ARGS,
            C0_CHECKPOINT,
            B10_ARGS,
            B10_CHECKPOINT,
            VAL_ROOT / "cache",
            VAL_ROOT / "diffusion_planner_validation.json",
        ):
            if not required.exists():
                raise FileNotFoundError(required)
        audit = audit_r0(r0_checkpoint, r0_args, r0_report)
        write_json(OUTPUT_ROOT / "r0_checkpoint_audit.json", audit)

        output_json = OUTPUT_ROOT / "open_loop_results.json"
        command = [
            str(PYTHON_BIN),
            str(HDP_ROOT / "scripts/evaluate_supervised_open_loop.py"),
            "--args-file", str(C0_ARGS),
            "--model", f"R0_absolute={r0_checkpoint}",
            "--model", f"C0_increment={C0_CHECKPOINT}",
            "--model", f"historical_B10={B10_CHECKPOINT}",
            "--model-args", f"R0_absolute={r0_args}",
            "--model-args", f"C0_increment={C0_ARGS}",
            "--model-args", f"historical_B10={B10_ARGS}",
            "--data-dir", str(VAL_ROOT / "cache"),
            "--data-list", str(VAL_ROOT / "diffusion_planner_validation.json"),
            "--batch-size", "16",
            "--num-workers", "2",
            "--repeats", "3",
            "--seed", "3407",
            "--num-samples", "1",
            "--diffusion-steps", "10",
            "--sampling-noise-scale", "0.1",
            "--device", "cuda",
            "--output", str(output_json),
        ]
        status.update({
            "state": "evaluating_open_loop",
            "command": command,
            "evaluation_started_at": now(),
        })
        write_json(status_path, status)
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
            raise RuntimeError(f"open-loop evaluation failed: {completed.returncode}")
        result = json.loads(output_json.read_text(encoding="utf-8"))
        (OUTPUT_ROOT / "open_loop_summary.md").write_text(
            render_summary(result),
            encoding="utf-8",
        )
        status.update({"state": "completed", "completed_at": now()})
        write_json(status_path, status)
    except Exception as error:
        status.update({
            "state": "failed",
            "failed_at": now(),
            "error": repr(error),
            "traceback": traceback.format_exc(),
        })
        write_json(status_path, status)
        raise


if __name__ == "__main__":
    main()
