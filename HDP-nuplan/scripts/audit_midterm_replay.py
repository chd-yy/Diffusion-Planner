"""Read-only reconstruction of FIFO replay coverage, not observed update draws."""

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
from torch.utils.data import DistributedSampler


def retained_indices(size, capacity, sampler_seed=0, epoch=0):
    if size < 1 or not 1 <= capacity <= size:
        raise ValueError("Require 1 <= capacity <= dataset size")
    sampler = DistributedSampler(range(size), num_replicas=1, rank=0,
                                 shuffle=True, seed=sampler_seed, drop_last=False)
    sampler.set_epoch(epoch)
    return list(sampler)[-capacity:]


def summarize(metadata, indices):
    rows = [metadata[i] for i in indices]
    return dict(count=len(rows), logs=dict(sorted(Counter(row["log_name"] for row in rows).items())),
                types=dict(sorted(Counter(row["scenario_type"] for row in rows).items())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    data = root / "tmp/mini_train_balanced_10000_seed3407_v1"
    manifest = data / "diffusion_planner_training.json"
    names = json.loads(manifest.read_text())
    if len(names) != 10000 or len(set(names)) != 10000:
        raise ValueError("Expected unique historical 10k manifest")
    if any(Path(name).name != name for name in names):
        raise ValueError("Flat cache filenames required")
    metadata = []
    for name in names:
        with np.load(data / "cache" / name, allow_pickle=False) as sample:
            metadata.append(dict(log_name=str(sample["log_name"].item()),
                                 scenario_type=str(sample["scenario_type"].item()),
                                 token=str(sample["token"].item())))
    cov = json.loads((root / "tmp/test14_full_remote_subset/coverage_validation.json").read_text())
    test_tokens = set(cov["test14_hard"]["tokens"]) | set(cov["test14_random"]["tokens"])
    overlap = sorted({row["token"] for row in metadata} & test_tokens)
    if overlap:
        raise ValueError("Training/Test14 token overlap")
    retained = retained_indices(len(names), 1024)
    result = dict(
        created_at=datetime.now().strftime("%m月%d日 %H:%M"),
        status="deterministic_reconstruction_not_observed_update_draws",
        assumptions=dict(world_size=1, rank=0, sampler_seed=0, rollout_epoch=0,
                         sampler_shuffle=True, full_rollout_completed=True, fifo_capacity=1024),
        manifest=str(manifest.resolve()), manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        full=summarize(metadata, range(len(names))), retained=summarize(metadata, retained),
        retained_names=[names[i] for i in retained], training_test14_token_overlap=0,
        expected_unique_uniform_draws_1000=1024 * (1 - (1 - 1 / 1024) ** 1000),
        actual_unique_update_draws=None,
        limitation="No update-draw tracing was recorded. No claim that retention caused driving degradation.",
    )
    full, kept = result["full"], result["retained"]
    types = sorted(set(full["types"]) | set(kept["types"]))
    result["type_total_variation"] = 0.5 * sum(
        abs(full["types"].get(t, 0) / full["count"] - kept["types"].get(t, 0) / kept["count"])
        for t in types)
    lines = ["# EXP003｜Replay覆盖重建", "", f"时间：{result['created_at']}（北京时间）。", "",
             "依据当前单卡DistributedSampler默认seed=0、epoch=0和FIFO1024重建；不是从训练日志观测到的抽样序列。",
             "本轮逐文件读取10k NPZ的log_name、scenario_type、token，未修改缓存。", "",
             f"全部日志数：{len(full['logs'])}；保留日志数：{len(kept['logs'])}。",
             f"保留每log最少/最多：{min(kept['logs'].values())}/{max(kept['logs'].values())}。",
             f"全部类型数：{len(full['types'])}；保留类型数：{len(kept['types'])}。",
             f"类别分布总变差距离：{result['type_total_variation']:.6f}（描述性，不是显著性检验）。",
             f"均匀有放回抽1000次的期望唯一组数：{result['expected_unique_uniform_draws_1000']:.3f}；实际未记录。", "",
             "|类型|完整10k|保留1024|完整比例|保留比例|", "|---|---:|---:|---:|---:|"]
    for t in types:
        f, k = full["types"].get(t, 0), kept["types"].get(t, 0)
        lines.append(f"|{t}|{f}|{k}|{f/10000:.4%}|{k/1024:.4%}|")
    lines += ["", "结论边界：保留覆盖不是实际唯一更新覆盖；balanced_logs配额在FIFO后不保证保留。",
              "训练seed42/2026不改变默认sampler seed0，因此在同manifest和epoch下保留集合相同；",
              "候选噪声、回放抽样与扩散加噪仍受训练seed影响。不能把类型分布变化直接归因为闭环退化。", ""]
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "replay_coverage.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    (args.output / "replay_coverage.md").write_text("\n".join(lines))
    print({k: result[k] for k in ["created_at", "type_total_variation", "expected_unique_uniform_draws_1000"]})
    print("full", len(full["logs"]), len(full["types"]), "retained", len(kept["logs"]), len(kept["types"]))


if __name__ == "__main__":
    main()
