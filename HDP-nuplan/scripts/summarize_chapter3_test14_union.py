"""Summarize the frozen Test14 hard/random union with equal weight per token."""

import hashlib
import json
import math
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "doc_hdp_nuplan/phd_midterm/stage_results/EXP_CH3_Test14_C0_R0"
MODELS = {"hdp_b_epoch10": "B10（Delta-Recon）", "C0": "C0（Delta）", "R0": "R0（Abs）"}
METRICS = {
    "score": "Score",
    "no_ego_at_fault_collisions": "NC",
    "drivable_area_compliance": "DA",
    "ego_is_making_progress": "Making progress",
    "driving_direction_compliance": "Direction",
    "ego_progress_along_expert_route": "Route progress",
    "time_to_collision_within_bound": "TTC",
    "speed_limit_compliance": "Speed compliance",
    "ego_is_comfortable": "Comfort",
}


def main():
    sources = {split: RESULT / f"test14_{split}_four_models.json" for split in ("hard", "random")}
    data = {split: json.loads(path.read_text(encoding="utf-8")) for split, path in sources.items()}
    output = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "scope": "frozen local Test14-hard union Test14-random",
        "duplicate_rule": "average each metric across two runs of the same token; then equal weight per unique token",
        "sources": {split: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for split, path in sources.items()},
        "models": {},
    }
    common_tokens = None
    for model, label in MODELS.items():
        records = {}
        for split, expected in (("hard", 272), ("random", 261)):
            run = data[split][model]
            rows = run["scenarios"]
            assert run["scenario_count"] == expected == len(rows)
            assert run["successful_simulations"] == expected and run["failed_simulations"] == 0
            assert run["runner_report_complete"] and run["runner_report_count"] == expected
            records[split] = {row["scenario"]: row for row in rows}
            assert len(records[split]) == expected
            for key in METRICS:
                assert all(isinstance(row[key], (int, float)) and math.isfinite(row[key]) for row in rows)
                assert math.isclose(sum(row[key] for row in rows) / expected, run["means"][key], abs_tol=1e-12)
        overlap = set(records["hard"]) & set(records["random"])
        tokens = set(records["hard"]) | set(records["random"])
        assert len(overlap) == 11 and len(tokens) == 522
        if common_tokens is None:
            common_tokens = tokens
        assert tokens == common_tokens
        unique_rows = []
        for token in sorted(tokens):
            occurrences = [records[split][token] for split in records if token in records[split]]
            assert len({(row["log_name"], row["scenario_type"]) for row in occurrences}) == 1
            unique_rows.append({
                "scenario": token,
                "log_name": occurrences[0]["log_name"],
                "scenario_type": occurrences[0]["scenario_type"],
                "occurrence_count": len(occurrences),
                **{key: sum(row[key] for row in occurrences) / len(occurrences) for key in METRICS},
            })
        output["models"][model] = {
            "label": label, "scenario_count": 522, "record_count": 533,
            "overlap_count": 11, "overlap_tokens": sorted(overlap),
            "means": {key: sum(row[key] for row in unique_rows) / 522 for key in METRICS},
            "scenarios": unique_rows,
        }
    json_path = RESULT / "test14_union_B10_C0_R0.json"
    json_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    best = {key: max(model["means"][key] for model in output["models"].values()) for key in METRICS}
    lines = [
        "# Test14全集合：B10、C0、R0对比表", "",
        f"> 整理时间：{datetime.now():%m月%d日 %H:%M}（北京时间）  ",
        "> 范围：当前冻结的Test14-hard与Test14-random并集；522个唯一场景，533条评测记录。  ",
        "> 评测模式：`closed_loop_nonreactive_agents`，推理seed=0，均使用Epoch10 EMA权重。", "",
        "## 1. 全集合对比", "",
        "| 模型 | " + " | ".join(METRICS.values()) + " |",
        "|---|" + "---:|" * len(METRICS),
    ]
    for model in output["models"].values():
        values = []
        for key in METRICS:
            value = model["means"][key]
            text = f"{value:.6f}"
            values.append(f"**{text}**" if math.isclose(value, best[key], abs_tol=1e-12) else text)
        lines.append("| " + model["label"] + " | " + " | ".join(values) + " |")
    lines.extend([
        "", "表中指标均为越大越好；加粗表示三个模型中的最高观测均值。NC为无自车责任碰撞指标，DA为可行驶区域合规，Direction为行驶方向合规，TTC为碰撞时间阈值合规，Speed compliance为限速合规。NC可能包含0.5分，不能直接等同于无碰撞场景比例。", "",
        "## 2. 汇总口径", "",
        "Test14-hard包含272个场景，Test14-random包含261个场景，两者重叠11个token，故并集为522个唯一场景。重叠场景在两次独立运行中存在少量指标差异：先对其两次指标取平均，再对522个唯一场景等权求均值；所有模型使用相同的场景集合和处理规则。", "",
        "该表描述当前冻结评测场景并集，不能替代hard/random分集结果，也不表示已经覆盖nuPlan全部可用Test14场景。522个唯一token仍可能来自同一日志，不等于522个统计独立样本。当前未对本表计算置信区间，也没有重复训练seed，不作统计显著性判断。", "",
        "## 3. 精炼结果分析", "",
        "C0的综合分数、可行驶区域和TTC均值最高；R0的任务推进率和路线进度最高；B10的行驶方向合规、限速合规及舒适性最高。无自车责任碰撞指标的最高均值为C0，但与B10的差距较小，现有结果不能证明该差异具有统计显著性。三个模型体现出不同的任务推进与驾驶质量权衡。", "",
        "后续安全约束后训练继续使用既定的B10作为初始化模型和冻结参考策略。C0与R0用于第三章的表示及训练目标分析；综合得分排序本身不能证明应该更换后训练起点，也不能将C0与历史B10的差异全部归因于Hybrid Loss。", "",
        "## 4. 原始证据与复现", "",
        "- [Test14-hard逐场结果](./test14_hard_four_models.json)",
        "- [Test14-random逐场结果](./test14_random_four_models.json)",
        "- [去重逐场指标与总体均值](./test14_union_B10_C0_R0.json)",
        "- 复现脚本：`HDP-nuplan/scripts/summarize_chapter3_test14_union.py`；运行后生成本表和去重JSON。", "",
    ])
    markdown_path = RESULT / "Test14全集合_B10_C0_R0对比表.md"
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    print(markdown_path)
    print("\n".join(lines[8:14]))


if __name__ == "__main__":
    main()
