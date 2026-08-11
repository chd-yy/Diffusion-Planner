"""NuPlan 场景抽样工具。"""

import random
from collections import defaultdict


def scenario_output_name(scenario):
    """返回 DataProcessor 实际使用的唯一 NPZ 文件名。"""
    return f"{scenario._map_name}_{scenario.token}.npz"


def deduplicate_scenarios(scenarios):
    """按最终输出文件名稳定去重，返回唯一场景和删除数量。"""
    unique = {}
    for scenario in sorted(
        scenarios,
        key=lambda item: (
            scenario_output_name(item),
            str(item.scenario_type),
        ),
    ):
        unique.setdefault(scenario_output_name(scenario), scenario)
    return list(unique.values()), len(scenarios) - len(unique)


def select_scenarios_balanced_by_log(
    scenarios,
    log_names,
    total_scenarios,
    seed,
    allow_empty_logs=False,
):
    """为每个日志提供基础配额，再从剩余唯一场景中随机补足总量。"""
    expected_logs = list(dict.fromkeys(log_names))

    # 同一 lidar token 可能带多个 scenario tag；按最终 NPZ 文件名去重，避免静默覆盖。
    unique_scenarios, duplicate_count = deduplicate_scenarios(scenarios)

    grouped = defaultdict(list)
    for scenario in unique_scenarios:
        grouped[scenario.log_name].append(scenario)

    missing_logs = sorted(set(expected_logs) - set(grouped))
    if missing_logs and not allow_empty_logs:
        raise RuntimeError(f"No scenarios found for requested logs: {missing_logs}")
    eligible_logs = [log_name for log_name in expected_logs if log_name in grouped]
    if not eligible_logs:
        raise RuntimeError("No scenarios found for any requested log")
    if total_scenarios < len(eligible_logs):
        raise ValueError(
            f"total_scenarios={total_scenarios} is smaller than the "
            f"number of eligible logs={len(eligible_logs)}"
        )
    if len(unique_scenarios) < total_scenarios:
        raise RuntimeError(
            f"Only {len(unique_scenarios)} unique scenarios are available, "
            f"but {total_scenarios} were requested"
        )

    rng = random.Random(seed)
    for items in grouped.values():
        rng.shuffle(items)

    # 官方 train 清单可能含少于 5 个 scene 的短 DB。NuPlan Builder 对这类 DB 返回
    # 0 Scenario；显式允许时，只在实际有候选的日志之间重新均衡总目标。
    base_quota = total_scenarios // len(eligible_logs)
    selected = []
    remaining = []
    for log_name in eligible_logs:
        items = grouped[log_name]
        take = min(base_quota, len(items))
        selected.extend(items[:take])
        remaining.extend(items[take:])

    needed = total_scenarios - len(selected)
    rng.shuffle(remaining)
    if len(remaining) < needed:
        raise RuntimeError(
            f"Unable to fill balanced selection: need {needed}, "
            f"only {len(remaining)} remain"
        )
    selected.extend(remaining[:needed])
    selected.sort(key=scenario_output_name)

    selected_counts = defaultdict(int)
    for scenario in selected:
        selected_counts[scenario.log_name] += 1

    report = {
        "strategy": "balanced_logs",
        "seed": seed,
        "requested_scenarios": total_scenarios,
        "raw_scenarios": len(scenarios),
        "unique_scenarios": len(unique_scenarios),
        "duplicate_output_names_removed": duplicate_count,
        "requested_log_count": len(expected_logs),
        "eligible_log_count": len(eligible_logs),
        "empty_log_count": len(missing_logs),
        "empty_logs": missing_logs,
        "base_quota_per_log": base_quota,
        "available_per_log": {
            log_name: len(grouped[log_name]) for log_name in eligible_logs
        },
        # 这里只记录真正产生 NPZ 的日志，确保缓存验证器能与 NPZ 元数据精确比较。
        "selected_per_log": {
            log_name: selected_counts[log_name] for log_name in eligible_logs
        },
    }
    return selected, report
