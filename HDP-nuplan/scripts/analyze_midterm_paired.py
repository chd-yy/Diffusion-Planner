"""Paired, log-clustered descriptive analysis of archived Test14 results.

Never trains, changes source summaries, authorizes promotion, or pools overlapping
benchmarks. Percentile intervals condition on the observed policies/inference
runs; they do not measure training-seed variance or establish safety guarantees.
"""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

from analyze_paired_closed_loop import METRICS


def validate_run(run):
    rows = run["scenarios"]
    n = len(rows)
    if (not n or run.get("scenario_count") != n
            or run.get("failed_simulations") != 0
            or run.get("successful_simulations") != n
            or run.get("runner_report_count") != n
            or run.get("runner_report_complete") is not True):
        raise ValueError("Incomplete simulation/runner coverage")
    indexed = {}
    for row in rows:
        key = (row["log_name"], row["scenario"])
        if not all(isinstance(x, str) and x for x in key) or key in indexed:
            raise ValueError("Missing or duplicate log/scenario identity")
        if not isinstance(row.get("scenario_type"), str) or not row["scenario_type"]:
            raise ValueError("Missing scenario type")
        for metric in METRICS:
            value = row.get(metric)
            if value is None or not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Invalid bounded metric: {key} {metric}")
        indexed[key] = row
    for metric in METRICS:
        mean = np.mean([row[metric] for row in rows])
        reported = run.get("means", {}).get(metric)
        if reported is None or not np.isfinite(reported) or abs(mean - reported) > 1e-8:
            raise ValueError(f"Reported mean disagrees with scenario rows: {metric}")
    return indexed


def clustered_interval(deltas, log_names, repeats=10000, seed=3407):
    """Resample logs; retain all paired rows per sampled log, using ratio means.

    The point estimate is scenario-weighted, not a mean of per-log means.
    With fewer than two observed logs there is no useful cluster interval.
    """
    if repeats < 100 or seed < 0:
        raise ValueError("Require >=100 resamples and a nonnegative seed")
    deltas = np.asarray(deltas, dtype=np.float64)
    if deltas.ndim != 2 or len(log_names) != len(deltas) or not np.isfinite(deltas).all():
        raise ValueError("Invalid paired delta matrix")
    logs = sorted(set(log_names))
    if len(logs) < 2:
        return None
    ids = np.array([logs.index(name) for name in log_names])
    sums = np.stack([deltas[ids == i].sum(axis=0) for i in range(len(logs))])
    counts = np.bincount(ids, minlength=len(logs))
    rng = np.random.default_rng(seed)
    values = []
    for start in range(0, repeats, 256):
        draws = rng.integers(len(logs), size=(min(256, repeats - start), len(logs)))
        values.append(sums[draws].sum(axis=1) / counts[draws].sum(axis=1)[:, None])
    quantiles = np.quantile(np.concatenate(values), [0.025, 0.975], axis=0)
    return quantiles.T.tolist()


def compare(baseline, candidate, repeats=10000, seed=3407):
    base, cand = validate_run(baseline), validate_run(candidate)
    if base.keys() != cand.keys():
        raise ValueError("Paired scenario coverage mismatch")
    keys = sorted(base)
    if any(base[k]["scenario_type"] != cand[k]["scenario_type"] for k in keys):
        raise ValueError("Scenario type mismatch")
    b = np.array([[base[k][m] for m in METRICS] for k in keys])
    c = np.array([[cand[k][m] for m in METRICS] for k in keys])
    delta = c - b
    intervals = clustered_interval(delta, [k[0] for k in keys], repeats, seed)
    metrics = {}
    tolerance = 1e-6
    for i, metric in enumerate(METRICS):
        d = delta[:, i]
        metrics[metric] = dict(
            baseline_mean=float(b[:, i].mean()), candidate_mean=float(c[:, i].mean()),
            mean_delta=float(d.mean()), median_delta=float(np.median(d)),
            wins=int((d > tolerance).sum()), losses=int((d < -tolerance).sum()),
            ties=int((np.abs(d) <= tolerance).sum()),
            log_cluster_percentile_interval_95=None if intervals is None else intervals[i],
        )
    score = delta[:, METRICS.index("score")]
    negative = np.maximum(-score, 0)
    categories = {}
    for category in sorted({base[k]["scenario_type"] for k in keys}):
        mask = np.array([base[k]["scenario_type"] == category for k in keys])
        categories[category] = dict(count=int(mask.sum()), mean_deltas={
            m: float(delta[mask, i].mean()) for i, m in enumerate(METRICS)})
    regressions = {}
    for metric in ["no_ego_at_fault_collisions", "drivable_area_compliance",
                   "time_to_collision_within_bound", "ego_is_comfortable"]:
        i = METRICS.index(metric)
        regressions[metric] = [dict(log_name=k[0], token=k[1],
                                   baseline=float(b[j, i]), candidate=float(c[j, i]))
                               for j, k in enumerate(keys) if delta[j, i] < -tolerance]
    return dict(
        scenario_count=len(keys), log_count=len({k[0] for k in keys}),
        metrics=metrics, category_results=categories, new_metric_regressions=regressions,
        top4_share_of_total_score_decrease=(float(np.sort(negative)[-4:].sum() / negative.sum())
                                           if negative.sum() > 0 else None),
        largest_score_decreases=[dict(log_name=keys[i][0], token=keys[i][1],
                                      delta=float(score[i]))
                                 for i in np.argsort(score)[:10] if score[i] < -tolerance],
    )


def render(result):
    lines = ["# EXP001｜归档 Test14 成对统计与退化分析", "",
             f"生成时间：{result['created_at']}（北京时间）。", "",
             "这是已有实验的二次分析，不是新训练或新闭环实验。差值均为候选减基线。",
             "按 log 整组有放回重采样，保留组内配对，报告 scenario-weighted 均值差的95%百分位区间。",
             "区间仅描述观测日志间的变异；未覆盖训练/推理随机性、测试集选择偏差或多重比较。",
             "不构成高置信安全保证，不授权替换B10。hard/random分别报告，不合并重复token。", "",
             f"重采样次数：{result['resamples']}；随机种子：{result['seed']}。", ""]
    for name, comparisons in result["benchmarks"].items():
        lines += [f"## {name}", ""]
        for label, value in comparisons.items():
            lines += [f"### {label}", "", f"{value['scenario_count']} 场景，{value['log_count']} 日志。", "",
                      "|指标|基线|候选|差值|日志聚类95%区间|胜/负/平|",
                      "|---|---:|---:|---:|---|---|"]
            for metric, d in value["metrics"].items():
                ci = d["log_cluster_percentile_interval_95"]
                interval = "不可估计" if ci is None else f"[{ci[0]:+.6f}, {ci[1]:+.6f}]"
                lines.append(f"|{metric}|{d['baseline_mean']:.6f}|{d['candidate_mean']:.6f}|"
                             f"{d['mean_delta']:+.6f}|{interval}|{d['wins']}/{d['losses']}/{d['ties']}|")
            share = value["top4_share_of_total_score_decrease"]
            lines += ["", f"最差4场占全部score下降量：{'无下降' if share is None else f'{share:.2%}'}。", "",
                      "|场景类型|N|Δscore|Δprogress|Δ无责任碰撞|", "|---|---:|---:|---:|---:|"]
            for category, d in value["category_results"].items():
                m = d["mean_deltas"]
                lines.append(f"|{category}|{d['count']}|{m['score']:+.6f}|"
                             f"{m['ego_progress_along_expert_route']:+.6f}|{m['no_ego_at_fault_collisions']:+.6f}|")
            lines += ["", "新退化的逐场景记录及最大下降案例见同名JSON；分类型结果是探索性结果，不用于挑选有利类别。", ""]
    lines += ["## 来源与重现", "", "输入摘要见同名JSON（含SHA256、脚本hash和版本）。", ""]
    for path, digest in result["inputs_sha256"].items():
        lines.append(f"- [{Path(path).name}]({path})：{digest}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[1] / "tmp")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    root = args.data_root.resolve()
    paths = [root / f"test14_full_remote_subset/three_model_eval/test14-{b}_three_models.json"
             for b in ("hard", "random")]
    repair_path = root / "rl_safety_geometry_test14_seed0/full_comparison.json"
    repaired = json.loads(repair_path.read_text())
    result = dict(created_at=datetime.now().strftime("%m月%d日 %H:%M"),
                  resamples=args.resamples, seed=args.seed, numpy_version=np.__version__,
                  analysis_version=1, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  promotion_authorized=False, new_simulations=0, benchmarks={},
                  inputs_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in paths + [repair_path]})
    benchmark_keys = []
    for name, path in zip(("test14-hard", "test14-random"), paths):
        source = json.loads(path.read_text())
        base, old = source["hdp_b_epoch10"], source["hdp_rl_epoch2"]
        repaired_base = repaired["benchmarks"][name]["baseline"]
        # The new aggregate must reuse the same B10 observations, not just means.
        left, right = validate_run(base), validate_run(repaired_base)
        if left.keys() != right.keys() or any(left[k][m] != right[k][m]
                                              for k in left for m in METRICS):
            raise ValueError("Repair baseline does not match archived B10")
        fixed = repaired["benchmarks"][name]["candidate"]
        result["benchmarks"][name] = {
            label: compare(b, c, args.resamples, args.seed)
            for label, b, c in [("旧RL减B10", base, old), ("修复RL减B10", base, fixed),
                                ("修复RL减旧RL", old, fixed)]}
        benchmark_keys.append(set(left))
    result["between_benchmark_overlap"] = len(benchmark_keys[0] & benchmark_keys[1])
    # Refuse existing directories so historical analyses cannot be overwritten.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "paired_statistics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    (args.output_dir / "paired_statistics.md").write_text(render(result))
    print(f"Wrote {args.output_dir.resolve()}; no new simulations or promotion")


if __name__ == "__main__":
    main()
