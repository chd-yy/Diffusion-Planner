"""汇总多个 NuPlan closed-loop 运行目录，输出可直接比较的 JSON。"""

import argparse
import json
from pathlib import Path

import pandas as pd


METRIC_COLUMNS = [
    "score",
    "no_ego_at_fault_collisions",
    "drivable_area_compliance",
    "ego_is_making_progress",
    "driving_direction_compliance",
    "ego_progress_along_expert_route",
    "time_to_collision_within_bound",
    "speed_limit_compliance",
    "ego_is_comfortable",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=DIR",
        help="可重复指定，例如 supervised=/abs/run/path",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def parse_run(value):
    if "=" not in value:
        raise ValueError(f"--run 必须使用 LABEL=DIR 格式: {value}")
    label, path = value.split("=", 1)
    if not label:
        raise ValueError(f"--run label 不能为空: {value}")
    return label, Path(path)


def native_number(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    return value


def summarize_run(root):
    aggregate_files = sorted((root / "aggregator_metric").glob("*.parquet"))
    if len(aggregate_files) != 1:
        raise RuntimeError(
            f"{root} 应有且仅有一个 aggregator parquet，实际为 {len(aggregate_files)}"
        )
    runner_path = root / "runner_report.parquet"
    if not runner_path.is_file():
        raise FileNotFoundError(runner_path)

    metrics = pd.read_parquet(aggregate_files[0])
    # 聚合文件还包含 scenario type 汇总行和 final_score 行；只有真实场景行具有 log_name。
    scenarios = metrics[metrics["log_name"].notna()].copy()
    runner = pd.read_parquet(runner_path)

    # 中文注释：正常运行时 runner 与指标文件一一对应；断点恢复时，旧进程已经
    # 写出的指标文件仍可精确用于规划得分，但中断前的 runner 耗时记录无法恢复。
    # 因此得分按全部指标场景统计，耗时均值只在 runner 覆盖完整时才对外报告。
    runner_succeeded = int(runner["succeeded"].sum())
    runner_failed = int((~runner["succeeded"]).sum())
    metric_succeeded = len(scenarios)
    runner_report_complete = len(runner) == len(scenarios)

    scenario_records = []
    for _, row in scenarios.sort_values("scenario").iterrows():
        record = {
            "scenario": str(row["scenario"]),
            "scenario_type": str(row["scenario_type"]),
            "log_name": str(row["log_name"]),
        }
        for column in METRIC_COLUMNS:
            record[column] = native_number(row[column])
        runtime = runner[runner["scenario_name"] == row["scenario"]]
        if len(runtime) == 1:
            record["trajectory_runtime_mean_s"] = native_number(
                runtime.iloc[0]["compute_trajectory_runtimes_mean"]
            )
            record["simulation_duration_s"] = native_number(runtime.iloc[0]["duration"])
        scenario_records.append(record)

    means = {}
    for column in METRIC_COLUMNS:
        means[column] = native_number(pd.to_numeric(scenarios[column]).mean())
    means["trajectory_runtime_mean_s"] = (
        native_number(pd.to_numeric(runner["compute_trajectory_runtimes_mean"]).mean())
        if runner_report_complete
        else None
    )
    means["simulation_duration_s"] = (
        native_number(pd.to_numeric(runner["duration"]).mean())
        if runner_report_complete
        else None
    )

    return {
        "run_dir": str(root),
        "scenario_count": len(scenarios),
        "successful_simulations": max(metric_succeeded, runner_succeeded),
        "failed_simulations": runner_failed,
        "runner_report_count": len(runner),
        "runner_report_complete": runner_report_complete,
        "metric_only_recovered_scenarios": max(0, metric_succeeded - len(runner)),
        "means": means,
        "scenarios": scenario_records,
    }


def main():
    args = parse_args()
    result = {label: summarize_run(path) for label, path in map(parse_run, args.run)}
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
