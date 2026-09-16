"""Pair full Test14 B10 records with the safety-geometry candidate."""

import argparse
from datetime import datetime
import json
from pathlib import Path

from diagnose_rl_safety_regressions import compare_safety


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-root", required=True, type=Path)
    parser.add_argument("--candidate-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--markdown", required=True, type=Path)
    args = parser.parse_args()
    result = {"created_at": datetime.now().strftime("%m月%d日 %H:%M"), "benchmarks": {}}
    lines = ["# RL安全几何修复版完整Test14 seed0成对评测", "",
             f"记录时间：{result['created_at']}（北京时间）。", ""]
    for benchmark in ("test14-hard", "test14-random"):
        historical = json.loads((args.historical_root / f"{benchmark}_three_models.json").read_text())
        candidate_file = json.loads((args.candidate_root / f"{benchmark}_candidate.json").read_text())
        baseline = historical["hdp_b_epoch10"]
        candidate = candidate_file["hdp_rl_safety_geometry"]
        gate = compare_safety(baseline, candidate)
        result["benchmarks"][benchmark] = {
            "baseline": baseline, "candidate": candidate, "gate": gate,
        }
        lines += [f"## {benchmark}", "",
                  f"- 场景数：{baseline['scenario_count']}",
                  f"- B10 score：{baseline['means']['score']:.6f}",
                  f"- 修复版 score：{candidate['means']['score']:.6f}",
                  f"- score差：{gate['mean_deltas']['score']:+.6f}",
                  f"- 路线进度差：{gate['mean_deltas']['ego_progress_along_expert_route']:+.6f}",
                  f"- 诊断gate：{'通过' if gate['diagnostic_gate_passed'] else '未通过'}", ""]
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    args.markdown.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
