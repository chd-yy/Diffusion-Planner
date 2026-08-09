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


def select_scenarios_balanced_by_log(scenarios, log_names, total_scenarios, seed):
    """为每个日志提供基础配额，再从剩余唯一场景中随机补足总量。"""
    expected_logs = list(dict.fromkeys(log_names))
    if total_scenarios < len(expected_logs):
        raise ValueError(
            f"total_scenarios={total_scenarios} is smaller than the "
            f"number of requested logs={len(expected_logs)}"
        )

    # 同一 lidar token 可能带多个 scenario tag；按最终 NPZ 文件名去重，避免静默覆盖。
    unique_scenarios, duplicate_count = deduplicate_scenarios(scenarios)

    grouped = defaultdict(list)
    for scenario in unique_scenarios:
        grouped[scenario.log_name].append(scenario)

    missing_logs = sorted(set(expected_logs) - set(grouped))
    if missing_logs:
        raise RuntimeError(f"No scenarios found for requested logs: {missing_logs}")
    if len(unique_scenarios) < total_scenarios:
        raise RuntimeError(
            f"Only {len(unique_scenarios)} unique scenarios are available, "
            f"but {total_scenarios} were requested"
        )

    rng = random.Random(seed)
    for items in grouped.values():
        rng.shuffle(items)

    base_quota = total_scenarios // len(expected_logs)
    selected = []
    remaining = []
    for log_name in expected_logs:
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
        "base_quota_per_log": base_quota,
        "available_per_log": {
            log_name: len(grouped[log_name]) for log_name in expected_logs
        },
        "selected_per_log": {
            log_name: selected_counts[log_name] for log_name in expected_logs
        },
    }
    return selected, report
