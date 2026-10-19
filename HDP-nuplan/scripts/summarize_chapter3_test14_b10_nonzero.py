"""Post-hoc matched comparison excluding tokens whose B10 union score is zero."""

import hashlib
import json
import math
from datetime import datetime

from summarize_chapter3_test14_union import METRICS, MODELS, RESULT


def markdown_table(models):
    best = {key: max(model["means"][key] for model in models.values()) for key in METRICS}
    lines = ["| 模型 | " + " | ".join(METRICS.values()) + " |", "|---|" + "---:|" * len(METRICS)]
    for label, model in models.items():
        cells = []
        for key in METRICS:
            value = model["means"][key]
            rendered = f"{value:.6f}"
            cells.append(f"**{rendered}**" if math.isclose(value, best[key], abs_tol=1e-12) else rendered)
        lines.append("| " + MODELS[label] + " | " + " | ".join(cells) + " |")
    return lines


def main():
    source = RESULT / "test14_union_B10_C0_R0.json"
    data = json.loads(source.read_text(encoding="utf-8"))
    indexed = {model: {row["scenario"]: row for row in data["models"][model]["scenarios"]} for model in MODELS}
    full = set(indexed["hdp_b_epoch10"])
    assert len(full) == 522 and all(set(rows) == full for rows in indexed.values())
    excluded = {token for token, row in indexed["hdp_b_epoch10"].items() if row["score"] == 0.0}
    retained = full - excluded
    assert retained and excluded
    output = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": {"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()},
        "selection_rule": "post-hoc: exclude tokens with B10 deduplicated union score == 0.0; apply identical token mask to all three models",
        "selection_limitation": "B10-outcome-conditioned subset; not a model-independent test set or an estimate of full Test14 performance",
        "full_count": len(full), "excluded_count": len(excluded), "retained_count": len(retained),
        "excluded_fraction": len(excluded) / len(full),
        "excluded_tokens": sorted(excluded), "retained_tokens": sorted(retained),
        "models": {},
        "excluded_scenarios": [
            {"scenario": token, "log_name": indexed["hdp_b_epoch10"][token]["log_name"],
             "scenario_type": indexed["hdp_b_epoch10"][token]["scenario_type"],
             "metrics_by_model": {model: {key: rows[token][key] for key in METRICS} for model, rows in indexed.items()}}
            for token in sorted(excluded)
        ],
    }
    for model, rows in indexed.items():
        kept = [rows[token] for token in sorted(retained)]
        dropped = [rows[token] for token in sorted(excluded)]
        assert all(math.isfinite(row[key]) for row in kept + dropped for key in METRICS)
        output["models"][model] = {
            "label": MODELS[model], "scenario_count": len(kept),
            "means": {key: sum(row[key] for row in kept) / len(kept) for key in METRICS},
            "full_means": data["models"][model]["means"],
            "excluded_means": {key: sum(row[key] for row in dropped) / len(dropped) for key in METRICS},
            "excluded_nonzero_score_count": sum(row["score"] > 0 for row in dropped),
            "scenarios": kept,
        }
    json_path = RESULT / "test14_B10_nonzero_B10_C0_R0.json"
    json_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Test14：剔除B10零分场景后的三模型对比", "",
        f"> 整理时间：{datetime.now():%m月%d日 %H:%M}（北京时间）  ",
        "> 分析性质：根据B10测试结果事后筛选的条件子集，作为补充机制分析。  ",
        "> 评测设置：`closed_loop_nonreactive_agents`，推理seed=0，Epoch10 EMA权重。", "",
        "## 1. 场景筛选规则", "",
        f"从当前冻结Test14-hard/random并集的{len(full)}个唯一场景出发，将B10去重后的综合分数为0的场景定义为本次剔除的最差场景，共{len(excluded)}个（{len(excluded)/len(full):.2%}）；剩余{len(retained)}个场景。三个模型同步剔除同一批token，再在相同的保留集合上逐场等权求均值。", "",
        "重叠于hard/random的11个场景沿用全集合口径：先对同一token的两次指标取平均，再判断B10分数是否为0。不根据新表中的模型排名调整剔除数量，也不对C0/R0各自另外筛选。零分表示综合指标失败，不一定只由责任碰撞引起；原始数据与全集合统计全部保留。", "",
        f"## 2. 保留{len(retained)}个场景的对比表", "",
        *markdown_table(output["models"]), "",
        "表中指标均为越大越好，加粗表示当前条件子集中的最高观测均值。NC为无自车责任碰撞指标，DA为可行驶区域合规，Direction为行驶方向合规，TTC为碰撞时间阈值合规，Speed compliance为限速合规；NC可能包含0.5分，不能直接等同于无碰撞场景比例。", "",
        "## 3. 被剔除场景不能忽略", "",
        f"| 模型 | 全集{len(full)}场景Score | 保留{len(retained)}场景Score | 剔除{len(excluded)}场景Score | 剔除集中的非零分场景数 |",
        "|---|---:|---:|---:|---:|",
    ]
    for model, stats in output["models"].items():
        lines.append(f"| {MODELS[model]} | {stats['full_means']['score']:.6f} | {stats['means']['score']:.6f} | {stats['excluded_means']['score']:.6f} | {stats['excluded_nonzero_score_count']}/{len(excluded)} |")
    lines.extend([
        "", "剔除后B10综合分数排名最高，说明在B10原本取得非零综合分的场景上，C0/R0没有完全保持其已有表现。另一方面，被剔除的B10零分场景中，C0与R0分别恢复了部分非零得分，构成它们在全集合上的收益来源。这些是不同模型的场景成功与失败交换，不能只保留其中一侧作为全局性能结论。", "",
        "## 4. 论文中的正确使用方式", "",
        "建议将本表命名为“B10非零分场景子集上的能力保持分析”，并与522场景全集合表共同展示。它回答的是：在基础模型B10已有非零得分的场景上，更改表示或训练配方后能否保持原有能力。它不能证明B10在全集合上表现最好，也不能证明Hybrid Loss存在严格因果优势。", "",
        "场景筛选使用了B10自身的测试结果，故本表对B10存在选择偏差；不作为模型无偏排名或正式后训练接受门禁。当前没有计算置信区间，不作统计显著性陈述。B10作为既定后训练基础模型的身份保持不变。", "",
        "## 5. 数据与复现", "",
        "- [522场景全集合对比表](./Test14全集合_B10_C0_R0对比表.md)",
        "- [剔除token清单、保留逐场数据及统计结果](./test14_B10_nonzero_B10_C0_R0.json)",
        "- 复现脚本：`HDP-nuplan/scripts/summarize_chapter3_test14_b10_nonzero.py`。", "",
    ])
    md_path = RESULT / "Test14_B10零分场景剔除后对比表.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(md_path)
    print(f"Excluded {len(excluded)}/{len(full)}; retained {len(retained)} identical tokens across all models.")


if __name__ == "__main__":
    main()
