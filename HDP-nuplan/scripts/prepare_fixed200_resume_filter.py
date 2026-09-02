#!/usr/bin/env python3
"""根据已有 NuPlan metric 文件生成 fixed200 缺失场景过滤配置。"""

import argparse
import re
from pathlib import Path

import yaml


TOKEN_PATTERN = re.compile(
    r"_([0-9a-f]{16})_hyper_diffusion_planner\.pickle\.temp$"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-filter", type=Path, required=True)
    parser.add_argument("--metrics-dir", type=Path, required=True)
    parser.add_argument("--output-filter", type=Path, required=True)
    parser.add_argument("--expected-total", type=int, default=200)
    parser.add_argument(
        "--metric-suffix",
        default="hyper_diffusion_planner",
        help="指标文件名中的 planner 后缀，默认用于 HDP；原始模型使用 diffusion_planner。",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = yaml.safe_load(args.base_filter.read_text(encoding="utf-8"))
    expected_tokens = list(config["scenario_tokens"])
    if len(expected_tokens) != args.expected_total:
        raise RuntimeError(
            f"Base filter contains {len(expected_tokens)} tokens, "
            f"expected {args.expected_total}"
        )

    completed_tokens = set()
    token_pattern = re.compile(
        rf"_([0-9a-f]{{16}})_{re.escape(args.metric_suffix)}\.pickle\.temp$"
    )
    for path in args.metrics_dir.glob("*.pickle.temp"):
        match = token_pattern.search(path.name)
        if match:
            completed_tokens.add(match.group(1))

    unexpected = completed_tokens.difference(expected_tokens)
    if unexpected:
        raise RuntimeError(f"Metric directory contains unexpected tokens: {unexpected}")

    missing_tokens = [
        token for token in expected_tokens if token not in completed_tokens
    ]
    config["scenario_tokens"] = missing_tokens
    args.output_filter.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# 自动生成的 fixed200 断点恢复配置。\n"
        f"# 已完成 {len(completed_tokens)}，缺失 {len(missing_tokens)}。\n"
    )
    args.output_filter.write_text(
        header + yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    print(
        f"completed={len(completed_tokens)} "
        f"missing={len(missing_tokens)} output={args.output_filter}"
    )


if __name__ == "__main__":
    main()
