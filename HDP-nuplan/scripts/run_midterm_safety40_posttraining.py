#!/usr/bin/env python3
"""Evaluate the frozen M0--M4 post-training ablation on safety40.

M0 reuses the already completed B10 safety40 run only after identity and
coverage checks. M1--M4 are evaluated sequentially with the same scenario
order, inference seed, simulation mode, and single-worker execution.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from analyze_midterm_paired import compare
from run_chapter3_safety40_closed_loop import (
    BENCHMARK,
    DATA,
    DEVKIT,
    HDP,
    MINI,
    ROOT,
    check_identity,
    frozen_write,
    group_analysis,
    resource_check,
    sha256,
    validate_benchmark,
    validate_summary,
    write_json,
)
from summarize_closed_loop_metrics import summarize_run


B10_RUN = HDP / "tmp/mini_train_full_306801_seed3407_v1/experiment_b_constant5e5_from_epoch4"
SOURCE_M0 = HDP / "tmp/chapter3_closed_loop/EXP_CH3_C0_B10_safety40_0918"
RUNS = {
    "M1": HDP / "tmp/midterm_posttraining/EXP002_expert_only_seed2026_0917/run.json",
    "M2": HDP / "tmp/midterm_posttraining/EXP004_unfiltered_seed2026_0917/run.json",
    "M3": HDP / "tmp/midterm_posttraining/EXP005_fixed200_train2026_0917_safety_seed2026/run.json",
    "M4": HDP / "tmp/midterm_posttraining/EXP005_fixed200_train2026_0917_safety_reference_seed2026/run.json",
}
EXPECTED_VARIANTS = {
    "M1": "expert_only",
    "M2": "unfiltered",
    "M3": "safety",
    "M4": "safety_reference",
}
EXPECTED_EFFECTIVE_DIFFS = {
    ("M1", "M2"): {
        "rl_filter_collision_candidates": (True, False),
        "rl_filter_safety_eligible_candidates": (True, False),
        "rl_rollout_loss_weight": (0.0, 1.0),
    },
    ("M2", "M3"): {
        "rl_filter_collision_candidates": (False, True),
        "rl_filter_safety_eligible_candidates": (False, True),
    },
    ("M3", "M4"): {"rl_reference_anchor_weight": (0.0, 0.1)},
}
PAIR_ORDER = (
    ("M0", "M1"),
    ("M0", "M2"),
    ("M0", "M3"),
    ("M0", "M4"),
    ("M1", "M2"),
    ("M2", "M3"),
    ("M3", "M4"),
)
DISPLAY_METRICS = (
    "score",
    "ego_progress_along_expert_route",
    "no_ego_at_fault_collisions",
    "drivable_area_compliance",
    "time_to_collision_within_bound",
    "ego_is_comfortable",
)


def now() -> str:
    return datetime.now().astimezone().isoformat()


def display_time() -> str:
    return datetime.now().strftime("%m月%d日 %H:%M")


def load_and_audit_models() -> tuple[dict, dict]:
    run_records = {label: json.loads(path.read_text(encoding="utf-8")) for label, path in RUNS.items()}
    models = {
        "M0": {
            "variant": "B10_frozen",
            "args": B10_RUN / "args.json",
            "checkpoint": B10_RUN / "model_epoch_10_trainloss_0.0091.pth",
            "checkpoint_sha256": "22ec0cf6be7cc89a3cf8414cddd0c7446ce2737ddc4e44b23e50dbeeeb0b29ce",
        }
    }
    common_source_checkpoint = models["M0"]["checkpoint_sha256"]
    common_manifest = None
    for label, record in run_records.items():
        if record.get("status") != "complete" or record.get("returncode") != 0:
            raise RuntimeError(f"{label} training is not complete")
        if record.get("variant") != EXPECTED_VARIANTS[label]:
            raise RuntimeError(f"{label} variant drift")
        if record.get("seed") != 2026:
            raise RuntimeError(f"{label} training seed drift")
        if record.get("source_changed_during_run"):
            raise RuntimeError(f"{label} source changed during training")
        if record.get("training_test14_token_overlap") != 0:
            raise RuntimeError(f"{label} training/Test14 token overlap")
        if record.get("source_checkpoint_sha256") != common_source_checkpoint:
            raise RuntimeError(f"{label} did not start from frozen B10")
        manifest_hash = record.get("training_manifest_sha256")
        common_manifest = common_manifest or manifest_hash
        if manifest_hash != common_manifest:
            raise RuntimeError("post-training manifest differs across models")
        checkpoints = record.get("checkpoints", {})
        if len(checkpoints) != 1:
            raise RuntimeError(f"{label} must have exactly one frozen checkpoint")
        checkpoint_text, recorded_hash = next(iter(checkpoints.items()))
        checkpoint = Path(checkpoint_text)
        args = checkpoint.parent / "args.json"
        if not checkpoint.is_file() or not args.is_file():
            raise FileNotFoundError(f"missing {label} checkpoint or args")
        if sha256(checkpoint) != recorded_hash:
            raise RuntimeError(f"{label} checkpoint hash mismatch")
        models[label] = {
            "variant": record["variant"],
            "args": args,
            "checkpoint": checkpoint,
            "checkpoint_sha256": recorded_hash,
            "run_json": RUNS[label],
            "effective_args": record["effective_args"],
        }

    ignored = {"name", "save_dir", "resume_path"}
    for pair, expected in EXPECTED_EFFECTIVE_DIFFS.items():
        left, right = (models[label]["effective_args"] for label in pair)
        differences = {
            key: (left.get(key), right.get(key))
            for key in sorted(set(left) | set(right))
            if key not in ignored and left.get(key) != right.get(key)
        }
        if differences != expected:
            raise RuntimeError(f"unexpected effective-argument drift for {pair}: {differences}")

    relevant_hashes = {}
    for label, record in run_records.items():
        hashes = {
            path: value
            for path, value in record["source_sha256"].items()
            if "/hdp_nuplan/" in path or path.endswith("/train_predictor_rl.py")
        }
        relevant_hashes[label] = hashes
    common_paths = set.intersection(*(set(item) for item in relevant_hashes.values()))
    if not common_paths:
        raise RuntimeError("no common training source paths recorded")
    for path in common_paths:
        if len({relevant_hashes[label][path] for label in relevant_hashes}) != 1:
            raise RuntimeError(f"training source differs across M1--M4: {path}")
    return models, {
        "source_checkpoint_sha256": common_source_checkpoint,
        "training_manifest_sha256": common_manifest,
        "controlled_effective_argument_differences": {
            f"{left}_to_{right}": {
                key: list(value) for key, value in values.items()
            }
            for (left, right), values in EXPECTED_EFFECTIVE_DIFFS.items()
        },
        "common_training_source_file_count": len(common_paths),
    }


def validate_reused_m0(manifest: dict, models: dict) -> dict:
    source_protocol = json.loads((SOURCE_M0 / "protocol.json").read_text(encoding="utf-8"))
    source_state = json.loads((SOURCE_M0 / "state.json").read_text(encoding="utf-8"))
    if source_state.get("status") != "complete":
        raise RuntimeError("source M0 evaluation is not complete")
    source_model = source_protocol["models"]["B10"]
    if (
        source_model["checkpoint_sha256"] != models["M0"]["checkpoint_sha256"]
        or source_model["args_sha256"] != sha256(models["M0"]["args"])
        or source_protocol.get("scenario_count") != 40
        or source_protocol.get("inference_seed") != 0
        or source_protocol.get("simulation") != "closed_loop_nonreactive_agents"
    ):
        raise RuntimeError("source M0 protocol differs from the new evaluation contract")
    summary = json.loads((SOURCE_M0 / "B10.summary.json").read_text(encoding="utf-8"))
    validate_summary(summary, manifest)
    return summary


def source_identity(output: Path, models: dict) -> dict[str, str]:
    paths = {
        Path(__file__).resolve(),
        HDP / "scripts/run_chapter3_safety40_closed_loop.py",
        HDP / "scripts/analyze_midterm_paired.py",
        HDP / "scripts/analyze_paired_closed_loop.py",
        HDP / "scripts/summarize_closed_loop_metrics.py",
        BENCHMARK / "manifest.json",
        BENCHMARK / "midterm-safety40-v1.yaml",
        BENCHMARK / "SHA256.json",
        SOURCE_M0 / "protocol.json",
        SOURCE_M0 / "state.json",
        SOURCE_M0 / "B10.summary.json",
    }
    for root in (HDP / "hdp_nuplan", ROOT / "diffusion_planner"):
        paths.update(root.rglob("*.py"))
        paths.update(root.rglob("*.yaml"))
    for spec in models.values():
        paths.update((spec["args"], spec["checkpoint"]))
        if spec.get("run_json"):
            paths.add(spec["run_json"])
    paths.update((output / "config/scenario_filter").glob("*.yaml"))
    return {str(path): sha256(path) for path in sorted(paths)}


def command_for(label: str, output: Path, models: dict) -> list[str]:
    spec = models[label]
    uid = f"{label}-posttraining-safety40-seed0"
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


def prepare(output: Path) -> tuple[dict, dict, dict[str, str]]:
    manifest, benchmark_hashes = validate_benchmark()
    models, audit = load_and_audit_models()
    reused_m0 = validate_reused_m0(manifest, models)
    config_root = output / "config/scenario_filter"
    config_root.mkdir(parents=True, exist_ok=True)
    source_filter = BENCHMARK / "midterm-safety40-v1.yaml"
    destination = config_root / source_filter.name
    if destination.exists() and destination.read_bytes() != source_filter.read_bytes():
        raise RuntimeError("output scenario filter drift")
    if not destination.exists():
        shutil.copy2(source_filter, destination)
    identity = source_identity(output, models)
    protocol = {
        "version": "posttraining-safety40-v1",
        "created_at": now(),
        "models": {
            label: {
                "variant": spec["variant"],
                "args": str(spec["args"]),
                "args_sha256": sha256(spec["args"]),
                "checkpoint": str(spec["checkpoint"]),
                "checkpoint_sha256": spec["checkpoint_sha256"],
                "training_seed": None if label == "M0" else 2026,
            }
            for label, spec in models.items()
        },
        "model_identity_audit": audit,
        "M0_reuse": {
            "source": str(SOURCE_M0 / "B10.summary.json"),
            "source_sha256": sha256(SOURCE_M0 / "B10.summary.json"),
            "reason": "same checkpoint, args, benchmark, simulation mode, inference seed, and complete 40/40 coverage",
        },
        "benchmark": str(BENCHMARK / "manifest.json"),
        "benchmark_sha256": benchmark_hashes,
        "scenario_count": 40,
        "inference_seed": 0,
        "simulation": "closed_loop_nonreactive_agents",
        "worker": "single_machine_thread_pool/max_workers=1/use_process_pool=false",
        "run_order": ["M1", "M2", "M3", "M4"],
        "comparison_order": [list(pair) for pair in PAIR_ORDER],
        "source_identity": identity,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "evaluation_role": "challenge-stratified development set; not blind testing",
        "promotion_authorized": False,
    }
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        previous = json.loads(protocol_path.read_text(encoding="utf-8"))
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
    frozen_write(output / "M0.summary.json", reused_m0)
    return manifest, models, identity


def render(result: dict) -> str:
    lines = [
        "# M0～M4固定safety40后训练机制对照",
        "",
        f"> 生成时间：{display_time()}（北京时间）  ",
        "> 所有差值均为后者减前者；官方指标越大越好。  ",
        "> 证据边界：分层开发集、单训练seed、单推理seed，不是安全保证或最终盲测。",
        "",
        "## 五模型总体结果",
        "",
        "|模型|机制|score|progress|NC|DA|TTC|comfort|成功/失败|",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in ("M0", "M1", "M2", "M3", "M4"):
        summary = result["summaries"][label]
        means = summary["means"]
        lines.append(
            f"|{label}|{result['model_variants'][label]}|{means['score']:.6f}|"
            f"{means['ego_progress_along_expert_route']:.6f}|{means['no_ego_at_fault_collisions']:.6f}|"
            f"{means['drivable_area_compliance']:.6f}|{means['time_to_collision_within_bound']:.6f}|"
            f"{means['ego_is_comfortable']:.6f}|{summary['successful_simulations']}/{summary['failed_simulations']}|"
        )
    lines += ["", "## 预注册边际机制比较", ""]
    for left, right in (("M0", "M1"), ("M1", "M2"), ("M2", "M3"), ("M3", "M4")):
        comparison = result["comparisons"][f"{right}_minus_{left}"]
        lines += [
            f"### {right}−{left}",
            "",
            "|指标|均值差|日志聚类95%区间|胜/负/平|",
            "|---|---:|---|---:|",
        ]
        for metric in DISPLAY_METRICS:
            values = comparison["metrics"][metric]
            interval = values["log_cluster_percentile_interval_95"]
            ci = "不可估计" if interval is None else f"[{interval[0]:+.6f}, {interval[1]:+.6f}]"
            lines.append(
                f"|{metric}|{values['mean_delta']:+.6f}|{ci}|"
                f"{values['wins']}/{values['losses']}/{values['ties']}|"
            )
        lines.append("")
    lines += [
        "## 解释约束",
        "",
        "M1−M0用于控制额外专家训练；M2−M1用于观察reward-weighted rollout相对专家续训的增量；M3−M2隔离候选安全/碰撞资格过滤；M4−M3隔离reference输出MSE锚定。",
        "任何区间跨0的差异不得写成稳定提升；任何新增碰撞、DA或TTC退化都必须逐场披露。M4中的reference项是输出MSE，不是策略KL。",
        "",
    ]
    return "\n".join(lines)


def execute(output: Path, manifest: dict, models: dict, identity: dict[str, str]) -> None:
    state_path = output / "state.json"
    state = {"status": "running", "started_at": now(), "stage": "initializing", "last_completed": "M0_reused"}
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
    summaries = {"M0": json.loads((output / "M0.summary.json").read_text(encoding="utf-8"))}
    try:
        for label in ("M1", "M2", "M3", "M4"):
            check_identity(identity)
            uid = f"{label}-posttraining-safety40-seed0"
            run_dir = output / "exp/simulation/closed_loop_nonreactive_agents" / uid
            if run_dir.exists():
                summary = summarize_run(run_dir)
                validate_summary(summary, manifest)
            else:
                resource_check()
                command = command_for(label, output, models)
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

        comparisons = {}
        group_results = {}
        for left, right in PAIR_ORDER:
            key = f"{right}_minus_{left}"
            comparisons[key] = compare(summaries[left], summaries[right], repeats=10000, seed=3407)
            group_results[key] = group_analysis(manifest, summaries[left], summaries[right])
        result = {
            "created_at": now(),
            "scenario_count": 40,
            "log_count": 8,
            "model_variants": {label: spec["variant"] for label, spec in models.items()},
            "summaries": summaries,
            "comparisons": comparisons,
            "group_results": group_results,
            "inputs": {
                "protocol": str(output / "protocol.json"),
                "benchmark": str(BENCHMARK / "manifest.json"),
            },
            "promotion_authorized": False,
        }
        write_json(output / "results.json", result)
        (output / "results.md").write_text(render(result) + "\n", encoding="utf-8")
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
    allowed_root = (HDP / "tmp/midterm_closed_loop").resolve()
    if allowed_root not in output.parents:
        raise ValueError(f"output must be a child of {allowed_root}")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest, models, identity = prepare(output)
        if args.execute:
            execute(output, manifest, models, identity)
        else:
            print(f"Prepared frozen M0--M4 safety40 protocol: {output}")


if __name__ == "__main__":
    main()
