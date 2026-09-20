#!/usr/bin/env python3
"""Run the frozen Chapter-3 C0/B10 safety40 paired closed-loop evaluation."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import yaml

from analyze_midterm_paired import compare, validate_run
from analyze_paired_closed_loop import METRICS
from summarize_closed_loop_metrics import summarize_run


ROOT = Path(__file__).resolve().parents[2]
HDP = ROOT / "HDP-nuplan"
DEVKIT = ROOT.parent / "nuplan-devkit"
DATA = ROOT.parent / "nuplan/dataset"
MINI = DATA / "nuplan-v1.1_mini/data/cache/mini"
BENCHMARK = HDP / "benchmarks/midterm_safety40_v1"
C0_RUN = HDP / "tmp/chapter3_hybrid_ablation_v1/C0/training_log/chapter3-C0-hybrid0/2026-09-17-22:47:24"
B10_RUN = HDP / "tmp/mini_train_full_306801_seed3407_v1/experiment_b_constant5e5_from_epoch4"
MODELS = {
    "B10": {
        "args": B10_RUN / "args.json",
        "checkpoint": B10_RUN / "model_epoch_10_trainloss_0.0091.pth",
        "checkpoint_sha256": "22ec0cf6be7cc89a3cf8414cddd0c7446ce2737ddc4e44b23e50dbeeeb0b29ce",
    },
    "C0": {
        "args": C0_RUN / "args.json",
        "checkpoint": C0_RUN / "model_epoch_10_trainloss_0.0031.pth",
        "checkpoint_sha256": "afb07593c2c989372ad5d11e39b13b532f65a457121b302f11c4b00e5c02aeb0",
    },
}
GROUP_ORDER = (
    "intersection",
    "vehicle_interaction",
    "vru_rules",
    "dynamics_environment",
    "basic_preservation",
)


def now() -> str:
    return datetime.now().astimezone().isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def frozen_write(path: Path, value: dict) -> None:
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise RuntimeError(f"frozen artifact drift: {path}")
    else:
        write_json(path, value)


def validate_benchmark() -> tuple[dict, dict]:
    manifest_path = BENCHMARK / "manifest.json"
    filter_path = BENCHMARK / "midterm-safety40-v1.yaml"
    hashes = json.loads((BENCHMARK / "SHA256.json").read_text(encoding="utf-8"))
    for name, expected in hashes.items():
        if sha256(BENCHMARK / name) != expected:
            raise RuntimeError(f"benchmark hash mismatch: {name}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = yaml.safe_load(filter_path.read_text(encoding="utf-8"))
    tokens = manifest["scenario_tokens"]
    rows = manifest["scenarios"]
    if (
        manifest["scenario_count"] != 40
        or len(tokens) != 40
        or len(set(tokens)) != 40
        or len(rows) != 40
        or [row["scenario"] for row in rows] != tokens
        or config["scenario_tokens"] != tokens
    ):
        raise RuntimeError("invalid benchmark scenario identity/order")
    if manifest["counts_by_group"] != {
        "intersection": 12,
        "vehicle_interaction": 10,
        "vru_rules": 8,
        "dynamics_environment": 6,
        "basic_preservation": 4,
    }:
        raise RuntimeError("major group quota drift")
    log_counts = Counter(row["log_name"] for row in rows)
    if len(log_counts) != 8 or min(log_counts.values()) < 1 or max(log_counts.values()) > 8:
        raise RuntimeError("log coverage/concentration drift")
    for index, row in enumerate(rows):
        for previous in rows[:index]:
            if (
                row["log_name"] == previous["log_name"]
                and abs(row["timestamp_us"] - previous["timestamp_us"]) < 15_000_000
            ):
                raise RuntimeError("same-log anchors violate 15-second separation")
    if manifest["training_token_overlap"] != 0 or manifest["training_log_overlap"] != 0:
        raise RuntimeError("training/development overlap")
    if any(
        row["group"] == "intersection"
        and row["scenario_type"] in {"stationary", "stationary_in_traffic"}
        for row in rows
    ):
        raise RuntimeError("stationary anchor leaked into intersection group")
    if any(not (MINI / f"{row['log_name']}.db").is_file() for row in rows):
        raise FileNotFoundError("one or more safety40 database files are missing")
    return manifest, hashes


def source_identity(output: Path) -> dict[str, str]:
    paths = {
        Path(__file__).resolve(),
        BENCHMARK / "manifest.json",
        BENCHMARK / "midterm-safety40-v1.yaml",
        BENCHMARK / "SHA256.json",
        HDP / "scripts/analyze_midterm_paired.py",
        HDP / "scripts/analyze_paired_closed_loop.py",
        HDP / "scripts/summarize_closed_loop_metrics.py",
    }
    for root in (HDP / "hdp_nuplan", ROOT / "diffusion_planner"):
        paths.update(root.rglob("*.py"))
        paths.update(root.rglob("*.yaml"))
    for spec in MODELS.values():
        paths.update((spec["args"], spec["checkpoint"]))
    paths.update((output / "config/scenario_filter").glob("*.yaml"))
    return {str(path): sha256(path) for path in sorted(paths)}


def check_identity(identity: dict[str, str]) -> None:
    for path, expected in identity.items():
        if sha256(Path(path)) != expected:
            raise RuntimeError(f"input changed during evaluation: {path}")


def command_for(label: str, output: Path) -> list[str]:
    spec = MODELS[label]
    uid = f"{label}-chapter3-safety40-seed0"
    return [
        sys.executable,
        str(DEVKIT / "nuplan/planning/script/run_simulation.py"),
        "+simulation=closed_loop_nonreactive_agents",
        "planner=hyper_diffusion_planner",
        f"planner.hyper_diffusion_planner.config.args_file={spec['args']}",
        f"planner.hyper_diffusion_planner.ckpt_path={spec['checkpoint']}",
        "scenario_builder=nuplan",
        f"scenario_builder.db_files={MINI}",
        "scenario_filter=midterm-safety40-v1",
        f"experiment_uid={uid}",
        "seed=0",
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
        f"hydra.searchpath=[file://{output / 'config'},pkg://hdp_nuplan.config.scenario_filter,pkg://hdp_nuplan.config,pkg://diffusion_planner.config,pkg://nuplan.planning.script.config.common,pkg://nuplan.planning.script.experiments]",
    ]


def resource_check() -> None:
    if shutil.disk_usage(HDP).free < 15 * 1024**3:
        raise RuntimeError("less than 15 GiB free on experiment filesystem")
    used = subprocess.check_output(
        [
            "nvidia-smi",
            "--id=0",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    if int(used.strip()) > 1024:
        raise RuntimeError("GPU already occupied")


def validate_summary(summary: dict, manifest: dict) -> None:
    indexed = validate_run(summary)
    expected = {(row["log_name"], row["scenario"]) for row in manifest["scenarios"]}
    if set(indexed) != expected:
        raise RuntimeError("closed-loop result coverage differs from safety40")


def group_analysis(manifest: dict, baseline: dict, candidate: dict) -> dict:
    group_by_token = {row["scenario"]: row["group"] for row in manifest["scenarios"]}
    base = {row["scenario"]: row for row in baseline["scenarios"]}
    cand = {row["scenario"]: row for row in candidate["scenarios"]}
    result = {}
    for group in GROUP_ORDER:
        tokens = [token for token, value in group_by_token.items() if value == group]
        metrics = {}
        for metric in METRICS:
            b = np.asarray([base[token][metric] for token in tokens], dtype=np.float64)
            c = np.asarray([cand[token][metric] for token in tokens], dtype=np.float64)
            delta = c - b
            metrics[metric] = {
                "baseline_mean": float(b.mean()),
                "candidate_mean": float(c.mean()),
                "mean_delta": float(delta.mean()),
                "wins": int((delta > 1e-6).sum()),
                "losses": int((delta < -1e-6).sum()),
                "ties": int((np.abs(delta) <= 1e-6).sum()),
            }
        result[group] = {"count": len(tokens), "metrics": metrics}
    return result


def render(result: dict) -> str:
    comparison = result["comparison_C0_minus_B10"]
    lines = [
        "# 第三章 C0 与历史B10固定40场景闭环结果",
        "",
        f"> 生成时间：{result['created_at']}  ",
        "> 差值定义：C0−B10；所有官方指标均为越大越好。  ",
        "> 证据边界：分层开发集、单训练/推理seed、历史B10准消融，不是正式安全保证。",
        "",
        "## 总体结果",
        "",
        "| 指标 | B10 | C0 | C0−B10 | 日志聚类95%区间 | 胜/负/平 |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for metric, values in comparison["metrics"].items():
        interval = values["log_cluster_percentile_interval_95"]
        ci = "不可估计" if interval is None else f"[{interval[0]:+.6f}, {interval[1]:+.6f}]"
        lines.append(
            f"| {metric} | {values['baseline_mean']:.6f} | {values['candidate_mean']:.6f} | "
            f"{values['mean_delta']:+.6f} | {ci} | "
            f"{values['wins']}/{values['losses']}/{values['ties']} |"
        )
    lines += ["", "## 五个预注册研究分组", ""]
    for group in GROUP_ORDER:
        item = result["group_results"][group]
        lines += [f"### {group}（N={item['count']}）", "", "|指标|B10|C0|差值|胜/负/平|", "|---|---:|---:|---:|---:|"]
        for metric in (
            "score",
            "ego_progress_along_expert_route",
            "no_ego_at_fault_collisions",
            "drivable_area_compliance",
            "time_to_collision_within_bound",
            "ego_is_comfortable",
        ):
            values = item["metrics"][metric]
            lines.append(
                f"|{metric}|{values['baseline_mean']:.6f}|{values['candidate_mean']:.6f}|"
                f"{values['mean_delta']:+.6f}|{values['wins']}/{values['losses']}/{values['ties']}|"
            )
        lines.append("")
    lines += [
        "## 结论边界",
        "",
        "开环ADE/FDE与闭环结果必须分层解释。区间按8个日志整组重采样，只描述本开发集日志间变异；不覆盖训练seed、推理seed或集合选择不确定性。",
        "历史B10缺少与C0相同的初始化身份，因此任何差值均不能完全归因于hybrid loss。",
        "逐场新增碰撞、DA、TTC和comfort退化以及最大score下降案例见同名JSON。",
        "",
    ]
    return "\n".join(lines)


def prepare(output: Path) -> tuple[dict, dict[str, str]]:
    manifest, benchmark_hashes = validate_benchmark()
    for label, spec in MODELS.items():
        if not spec["args"].is_file() or not spec["checkpoint"].is_file():
            raise FileNotFoundError(f"missing {label} model inputs")
        if sha256(spec["checkpoint"]) != spec["checkpoint_sha256"]:
            raise RuntimeError(f"{label} checkpoint identity mismatch")
    config_root = output / "config/scenario_filter"
    config_root.mkdir(parents=True, exist_ok=True)
    destination = config_root / "midterm-safety40-v1.yaml"
    source = BENCHMARK / "midterm-safety40-v1.yaml"
    if destination.exists() and destination.read_bytes() != source.read_bytes():
        raise RuntimeError("output scenario filter drift")
    if not destination.exists():
        shutil.copy2(source, destination)
    identity = source_identity(output)
    protocol = {
        "version": "chapter3-safety40-v1",
        "created_at": now(),
        "models": {
            label: {
                "args": str(spec["args"]),
                "args_sha256": sha256(spec["args"]),
                "checkpoint": str(spec["checkpoint"]),
                "checkpoint_sha256": spec["checkpoint_sha256"],
            }
            for label, spec in MODELS.items()
        },
        "benchmark": str(BENCHMARK / "manifest.json"),
        "benchmark_sha256": benchmark_hashes,
        "scenario_count": 40,
        "inference_seed": 0,
        "simulation": "closed_loop_nonreactive_agents",
        "worker": "single_machine_thread_pool/max_workers=1/use_process_pool=false",
        "order": ["B10", "C0"],
        "source_identity": identity,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "evaluation_role": "challenge-stratified development set; not blind testing",
        "promotion_authorized": False,
    }
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        previous = json.loads(protocol_path.read_text(encoding="utf-8"))
        # Creation time is provenance, not a resume-sensitive input.
        left, right = dict(previous), dict(protocol)
        left.pop("created_at", None)
        right.pop("created_at", None)
        if left != right:
            raise RuntimeError("protocol/input drift; use a new output directory")
        protocol = previous
    elif (output / "exp").exists():
        raise RuntimeError("untracked experiment outputs")
    else:
        write_json(protocol_path, protocol)
    return manifest, identity


def execute(output: Path, manifest: dict, identity: dict[str, str]) -> None:
    state_path = output / "state.json"
    state = {"status": "running", "started_at": now(), "stage": "initializing"}
    write_json(state_path, state)
    environment = dict(
        os.environ,
        NUPLAN_DATA_ROOT=str(DATA),
        NUPLAN_MAPS_ROOT=str(DATA / "maps"),
        NUPLAN_DEVKIT_ROOT=str(DEVKIT),
        NUPLAN_EXP_ROOT=str(output),
        CUDA_VISIBLE_DEVICES="0",
        HYDRA_FULL_ERROR="1",
        PYTHONUNBUFFERED="1",
    )
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(HDP), str(ROOT), str(DEVKIT), environment.get("PYTHONPATH", "")]
    )
    summaries = {}
    try:
        for label in ("B10", "C0"):
            check_identity(identity)
            uid = f"{label}-chapter3-safety40-seed0"
            run_dir = output / "exp/simulation/closed_loop_nonreactive_agents" / uid
            if run_dir.exists():
                summary = summarize_run(run_dir)
                validate_summary(summary, manifest)
            else:
                resource_check()
                command = command_for(label, output)
                state.update(stage="simulation", model=label, command=command, updated_at=now())
                write_json(state_path, state)
                with (output / f"{uid}.log").open("a", encoding="utf-8") as log:
                    completed = subprocess.run(
                        command,
                        cwd=ROOT,
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=False,
                        text=True,
                    )
                if completed.returncode != 0:
                    raise RuntimeError(f"{label} simulation exited with {completed.returncode}")
                check_identity(identity)
                summary = summarize_run(run_dir)
                validate_summary(summary, manifest)
            summaries[label] = summary
            frozen_write(output / f"{label}.summary.json", summary)
            state.update(last_completed=label, updated_at=now())
            write_json(state_path, state)
        comparison = compare(summaries["B10"], summaries["C0"], repeats=10000, seed=3407)
        result = {
            "created_at": now(),
            "scenario_count": 40,
            "log_count": 8,
            "baseline": "B10",
            "candidate": "C0",
            "delta_definition": "C0 minus B10",
            "comparison_C0_minus_B10": comparison,
            "group_results": group_analysis(manifest, summaries["B10"], summaries["C0"]),
            "inputs": {
                "protocol": str(output / "protocol.json"),
                "benchmark": str(BENCHMARK / "manifest.json"),
            },
            "promotion_authorized": False,
        }
        write_json(output / "paired_results.json", result)
        (output / "paired_results.md").write_text(render(result) + "\n", encoding="utf-8")
        state.update(status="complete", stage="complete", finished_at=now())
        state.pop("command", None)
        write_json(state_path, state)
    except Exception as error:
        state.update(status="failed", error=repr(error), failed_at=now())
        write_json(state_path, state)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    allowed_root = (HDP / "tmp/chapter3_closed_loop").resolve()
    if allowed_root not in output.parents:
        raise ValueError(f"output must be a child of {allowed_root}")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest, identity = prepare(output)
        if args.execute:
            execute(output, manifest, identity)
        else:
            print(f"Prepared frozen C0/B10 safety40 protocol: {output}")


if __name__ == "__main__":
    main()
