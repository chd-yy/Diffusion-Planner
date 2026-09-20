"""Frozen M0--M4 training/evaluation queue, one GPU subprocess at a time.

Prepare-only by default. Resume accepts complete training and archived chunks;
partial/damaged archives and changed identities fail closed. No model promotion.
"""

import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import yaml

from analyze_midterm_paired import compare, validate_run as validate_summary
from evaluate_rl_safety_geometry_multiseed import artifact_state, check_identity, write_json
from manage_test14_chunk_results import archive_chunk, merge, validate_run
from run_midterm_posttraining import B10_SHA, HDP, MANIFEST_SHA, ROOT, VARIANTS, sha256, variant_args
from summarize_closed_loop_metrics import METRIC_COLUMNS, summarize_run

DEVKIT = ROOT.parent / "nuplan-devkit"
DATA = ROOT.parent / "nuplan/dataset"
MINI = DATA / "nuplan-v1.1_mini/data/cache/mini"
FIXED = HDP / "tmp/mini_train_balanced_10000_seed3407_v1/mini_val_fixed_200_rl_v4_manifest.json"
FILTER = HDP / "hdp_nuplan/config/scenario_filter/mini-val-fixed-200-rl-v4.yaml"
REPLAY = HDP / "doc_hdp_nuplan/phd_midterm/stage_results/EXP003_replay_coverage/replay_coverage.json"
HISTORICAL = HDP / "tmp/mini_train_balanced_10000_seed3407_v1/rl_safetygate_ttc1_anchor01_10k_seed2026_from_b10/training_log/hdp-rl-safetygate_ttc1_anchor01_10k_seed2026-from-full-mini-b10/2026-08-22-12:22:45/args.json"
LABELS = ("M0", "M1", "M2", "M3", "M4")
# Historical CLI args omit later-added defaults. Freeze their parsed values too;
# they are inactive in legacy mode / disabled progress and drivable-area gates.
EXTRA_DEFAULTS = dict(reward_aligned_comfort_exponent=0.1, reward_aligned_follow_exponent=0.05,
                      reward_safety_gate_drivable_area_margin=0.1,
                      reward_aligned_progress_exponent=0.25, reward_aligned_route_exponent=0.15,
                      rl_min_progress_guard_reward=0.9, reward_aligned_safety_exponent=0.45)


def now():
    return datetime.now().strftime("%m月%d日 %H:%M")


def validate_fixed(fixed, train_tokens, train_logs):
    tokens = fixed["scenario_tokens"]
    rows = fixed["scenarios"]
    if (fixed["scenario_count"] != 200 or len(tokens) != 200 or len(set(tokens)) != 200
            or len(rows) != 200 or {r["scenario"] for r in rows} != set(tokens)):
        raise ValueError("Fixed200 coverage is invalid")
    if any(not isinstance(t, str) or len(t) != 16 or any(c not in "0123456789abcdef" for c in t)
           for t in tokens):
        raise ValueError("Scenario tokens must remain 16-character hex strings")
    if set(tokens) & set(train_tokens):
        raise ValueError("Training/development token overlap")
    if {r["log_name"] for r in rows} & set(train_logs):
        raise ValueError("Training/development log overlap")


def frozen_write(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise RuntimeError(f"Protocol/input drift: {path}")
    else:
        write_json(path, value)


def prepare(output, train_seed, infer_seeds):
    source = json.loads(HISTORICAL.read_text())
    fixed = json.loads(FIXED.read_text())
    replay = json.loads(REPLAY.read_text())
    manifest = Path(source["train_set_list"])
    if sha256(manifest) != MANIFEST_SHA or replay["manifest_sha256"] != MANIFEST_SHA:
        raise ValueError("Training manifest identity mismatch")
    names = json.loads(manifest.read_text())
    train_tokens = {Path(n).stem.rsplit("_", 1)[-1] for n in names}
    validate_fixed(fixed, train_tokens, replay["full"]["logs"])
    if not MINI.is_dir() or any(not (MINI / (r["log_name"] + ".db")).is_file() for r in fixed["scenarios"]):
        raise FileNotFoundError("Missing development DBs")
    base_filter = yaml.safe_load(FILTER.read_text())
    if base_filter["scenario_tokens"] != fixed["scenario_tokens"]:
        raise ValueError("Filter/manifest token ordering or token types disagree")
    config_root = output / "config/scenario_filter"
    config_root.mkdir(parents=True, exist_ok=True)
    chunks = []
    for index, start in enumerate(range(0, 200, 40)):
        tokens = fixed["scenario_tokens"][start:start + 40]
        name = f"midterm-dev200-chunk-{index:03d}"
        path = config_root / (name + ".yaml")
        config = dict(base_filter, scenario_tokens=tokens)
        # JSON is valid YAML and forces token strings (including digit/e-only tokens).
        frozen_write(path, config)
        chunks.append(dict(index=index, count=len(tokens), tokens=tokens, filter=name, config=str(path)))
    chunk_manifest = dict(chunk_size=40, benchmarks={"dev200": dict(
        count=200, tokens=fixed["scenario_tokens"], chunks=chunks)})
    frozen_write(output / "eval_chunk_manifest.json", chunk_manifest)
    runs = {}
    for label, variant in zip(LABELS[1:], VARIANTS):
        if train_seed == 2026 and variant == "expert_only":
            path = HDP / "tmp/midterm_posttraining/EXP002_expert_only_seed2026_0917"
        elif train_seed == 2026 and variant == "unfiltered":
            path = HDP / "tmp/midterm_posttraining/EXP004_unfiltered_seed2026_0917"
        else:
            path = HDP / f"tmp/midterm_posttraining/{output.name}_{variant}_seed{train_seed}"
        runs[label] = dict(variant=variant, output=str(path))
    checkpoint = Path(source["pretrained_model_path"])
    if sha256(checkpoint) != B10_SHA:
        raise ValueError("B10 identity mismatch")
    paths = {FIXED, FILTER, REPLAY, HISTORICAL, manifest, checkpoint, checkpoint.parent / "args.json",
             HDP / "normalization.json", Path(__file__).resolve(), output / "eval_chunk_manifest.json"}
    for root in (HDP / "hdp_nuplan", ROOT / "diffusion_planner", DEVKIT / "nuplan"):
        paths.update(root.rglob("*.py"))
        paths.update(root.rglob("*.yaml"))
    for name in ("run_midterm_posttraining.py", "analyze_midterm_paired.py", "analyze_paired_closed_loop.py",
                 "evaluate_rl_safety_geometry_multiseed.py", "manage_test14_chunk_results.py",
                 "summarize_closed_loop_metrics.py", "diagnose_rl_safety_regressions.py"):
        paths.add(HDP / "scripts" / name)
    paths.update(config_root.glob("*.yaml"))
    identities = {str(p): sha256(p) for p in sorted(paths)}
    protocol = dict(version="midterm-fixed200-v1", train_seed=train_seed, inference_seeds=infer_seeds,
                    training_runs=runs, b10=dict(args=str(checkpoint.parent / "args.json"),
                    checkpoint=str(checkpoint), checkpoint_sha256=B10_SHA),
                    file_sha256=identities, fixed200_log_count=len(fixed["counts_by_log"]),
                    training_dev_token_overlap=0, training_dev_log_overlap=0,
                    log_overlap_evidence="EXP003 metadata audit matched to training manifest SHA",
                    max_gpu_processes=1, worker_per_process=1, chunk_size=40,
                    order="inference seed, chunk, M0/M1/M2/M3/M4", promotion_authorized=False,
                    evaluation_role="historically used development set, NOT blind testing",
                    stop_new_work_at="2026-09-21T12:00:00+08:00",
                    git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    if not (output / "protocol.json").exists() and (output / "exp").exists():
        raise RuntimeError("Untracked previous evaluation outputs")
    frozen_write(output / "protocol.json", protocol)
    return protocol, chunk_manifest


def resource_check():
    if shutil.disk_usage(HDP).free < 15 * 1024**3 or shutil.disk_usage("/").free < 5 * 1024**3:
        raise RuntimeError("Disk reserve reached (experiment 15 GiB / root 5 GiB)")
    used = subprocess.check_output(["nvidia-smi", "--id=0", "--query-gpu=memory.used",
                                    "--format=csv,noheader,nounits"], text=True)
    if int(used.strip()) > 1024:
        raise RuntimeError("GPU already occupied; refusing a competing process")


def check_deadline(protocol):
    # The deadline only prevents NEW work; never time-kills an active simulation/training run.
    if datetime.now().astimezone() >= datetime.fromisoformat(protocol["stop_new_work_at"]):
        raise RuntimeError("Material freeze reached; no new GPU work")


def model_from_run(path, variant, seed):
    record = json.loads((path / "run.json").read_text())
    expected = variant_args(json.loads(HISTORICAL.read_text()), variant, seed, path)
    if (record["status"] != "complete" or record.get("source_changed_during_run") != []
            or record["effective_args"] != expected or record["source_checkpoint_sha256"] != B10_SHA
            or record["training_manifest_sha256"] != MANIFEST_SHA or len(record["checkpoints"]) != 1):
        raise RuntimeError(f"Training incomplete or protocol mismatch: {path}")
    for file, digest in record["source_sha256"].items():
        # M1 predates the timeout-only launcher change; core code must still match exactly.
        if Path(file).name != "run_midterm_posttraining.py" and sha256(file) != digest:
            raise RuntimeError(f"Training source drift: {file}")
    checkpoint, digest = next(iter(record["checkpoints"].items()))
    args_path = Path(checkpoint).parent / "args.json"
    actual = json.loads(args_path.read_text())
    actual = {k: v for k, v in actual.items() if k not in ("state_normalizer", "observation_normalizer")}
    if actual != dict(EXTRA_DEFAULTS, **expected) or sha256(checkpoint) != digest:
        raise RuntimeError(f"Checkpoint/parsed args mismatch: {path}")
    return dict(args=str(args_path), checkpoint=checkpoint, checkpoint_sha256=digest,
                args_sha256=sha256(args_path), run_record=str(path / "run.json"),
                run_record_sha256=sha256(path / "run.json"))


def command_for(spec, seed, chunk, uid, output):
    return [sys.executable, str(DEVKIT / "nuplan/planning/script/run_simulation.py"),
            "+simulation=closed_loop_nonreactive_agents", "planner=hyper_diffusion_planner",
            f"planner.hyper_diffusion_planner.config.args_file={spec['args']}",
            f"planner.hyper_diffusion_planner.ckpt_path={spec['checkpoint']}",
            "scenario_builder=nuplan", f"scenario_builder.db_files={MINI}",
            f"scenario_filter={chunk['filter']}", f"experiment_uid={uid}", f"seed={seed}",
            "worker=single_machine_thread_pool", "worker.max_workers=1", "worker.use_process_pool=false",
            "number_of_gpus_allocated_per_simulation=1", "number_of_cpus_allocated_per_simulation=1",
            "max_callback_workers=1", "disable_callback_parallelization=true", "verbose=true",
            "~callback.simulation_log_callback",
            f"hydra.searchpath=[file://{output / 'config'},pkg://hdp_nuplan.config.scenario_filter,pkg://hdp_nuplan.config,pkg://diffusion_planner.config,pkg://nuplan.planning.script.config.common,pkg://nuplan.planning.script.experiments]"]


def execute(output, protocol, manifest):
    state_path = output / "state.json"
    state = dict(status="running", updated_at=now(), promotion_authorized=False)
    identity = dict(protocol["file_sha256"])

    def save(**values):
        state.update(values, updated_at=now())
        write_json(state_path, state)

    try:
        models = {"M0": protocol["b10"]}
        for label, run in protocol["training_runs"].items():
            check_identity(identity)
            path = Path(run["output"])
            if not path.exists():
                check_deadline(protocol)
                resource_check()
                command = [sys.executable, str(HDP / "scripts/run_midterm_posttraining.py"),
                           "--variant", run["variant"], "--seed", str(protocol["train_seed"]),
                           "--output", str(path), "--execute", "--timeout-seconds", "0"]
                save(stage="training", model=label, command=command)
                print(f"[{now()}] Train {label} {run['variant']}", flush=True)
                subprocess.run(command, cwd=ROOT, check=True)
            models[label] = model_from_run(path, run["variant"], protocol["train_seed"])
        frozen_write(output / "models.json", models)
        for spec in models.values():
            for field in ("args", "checkpoint", "run_record"):
                if field in spec:
                    identity[spec[field]] = sha256(spec[field])
        env = dict(os.environ, NUPLAN_DATA_ROOT=str(DATA), NUPLAN_MAPS_ROOT=str(DATA / "maps"),
                   NUPLAN_DEVKIT_ROOT=str(DEVKIT), NUPLAN_EXP_ROOT=str(output), CUDA_VISIBLE_DEVICES="0",
                   HYDRA_FULL_ERROR="1", PYTHONUNBUFFERED="1")
        env["PYTHONPATH"] = os.pathsep.join([str(HDP), str(ROOT), str(DEVKIT), env.get("PYTHONPATH", "")])
        benchmark = manifest["benchmarks"]["dev200"]
        for seed in protocol["inference_seeds"]:
            for chunk in benchmark["chunks"]:
                for label, spec in models.items():
                    check_identity(identity)
                    uid = f"{label}-dev200-train{protocol['train_seed']}-infer{seed}"
                    run_dir = output / "exp/simulation/closed_loop_nonreactive_agents" / uid
                    if artifact_state(run_dir, benchmark, chunk) == "complete":
                        continue
                    check_deadline(protocol)
                    resource_check()
                    command = command_for(spec, seed, chunk, uid, output)
                    save(stage="simulation", model=label, inference_seed=seed, chunk=chunk["index"], command=command)
                    print(f"[{now()}] {uid} chunk {chunk['index']}", flush=True)
                    with (output / f"{uid}.log").open("a") as log:
                        subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                    check_identity(identity)
                    archive_chunk(SimpleNamespace(manifest=output / "eval_chunk_manifest.json", benchmark="dev200",
                                  chunk_index=chunk["index"], run_dir=run_dir, report=run_dir / "runner_report.parquet"))
                    save(last_completed=dict(model=label, inference_seed=seed, chunk=chunk["index"]))
            summaries = {}
            for label in models:
                uid = f"{label}-dev200-train{protocol['train_seed']}-infer{seed}"
                run_dir = output / "exp/simulation/closed_loop_nonreactive_agents" / uid
                args = SimpleNamespace(manifest=output / "eval_chunk_manifest.json", benchmark="dev200", run_dir=run_dir)
                summary_path = output / f"{uid}.summary.json"
                if not summary_path.exists():
                    merge(args)
                validate_run(args)
                summary = summarize_run(run_dir)
                indexed = validate_summary(summary)
                expected = {(r["log_name"], r["scenario"]): r["scenario_type"] for r in json.loads(FIXED.read_text())["scenarios"]}
                if set(indexed) != set(expected) or any(row["scenario_type"] != expected[key] for key, row in indexed.items()):
                    raise ValueError("Observed log/token/type identity differs from frozen Fixed200")
                summaries[label] = summary
                frozen_write(summary_path, summary)
            comparisons = {f"{c}_minus_{b}": compare(summaries[b], summaries[c])
                           for b, c in (("M0", "M1"), ("M0", "M2"), ("M0", "M3"), ("M0", "M4"),
                                        ("M1", "M2"), ("M2", "M3"), ("M3", "M4"))}
            frozen_write(output / f"paired_infer{seed}.json", dict(
                train_seed=protocol["train_seed"], inference_seed=seed, comparisons=comparisons,
                evaluation_role=protocol["evaluation_role"], promotion_authorized=False))
            report = [f"# 固定200开发集：训练seed={protocol['train_seed']}，推理seed={seed}", "",
                      f"生成时间：{now()}。五个模型各200场；非盲测；无自动模型升级。", "",
                      "| 模型 | " + " | ".join(METRIC_COLUMNS) + " |", "|---|" + "---|" * len(METRIC_COLUMNS)]
            for label, summary in summaries.items():
                report.append(f"| {label} | " + " | ".join(f"{summary['means'][m]:.6f}" for m in METRIC_COLUMNS) + " |")
            report.extend(["", "训练完成不等于有效；配对JSON包含逐指标差值、日志聚类区间及新增失败。",
                           "仅8个开发集日志，区间不代表训练seed不确定性或正式安全保证。"])
            report_path = output / f"results_infer{seed}.md"
            if not report_path.exists():
                report_path.write_text("\n".join(report) + "\n")
            save(stage="paired_results", inference_seed=seed)
        check_identity(identity)
        save(status="complete", stage="complete")
    except Exception as error:
        save(status="failed", error=str(error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-seed", type=int, default=2026)
    parser.add_argument("--inference-seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--execute", action="store_true")
    cli = parser.parse_args()
    if cli.train_seed < 0 or any(s < 0 for s in cli.inference_seeds) or len(set(cli.inference_seeds)) != len(cli.inference_seeds):
        parser.error("Seeds must be distinct nonnegative integers")
    output = cli.output.resolve()
    if (HDP / "tmp/midterm_closed_loop").resolve() not in output.parents:
        raise ValueError("Use a child of HDP-nuplan/tmp/midterm_closed_loop")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol, manifest = prepare(output, cli.train_seed, cli.inference_seeds)
        if cli.execute:
            execute(output, protocol, manifest)
        else:
            print(f"Prepared frozen queue only: {output}")


if __name__ == "__main__":
    main()
