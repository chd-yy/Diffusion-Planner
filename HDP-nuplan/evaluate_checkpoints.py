"""Evaluate and rank every supervised checkpoint in one HDP training run."""

import argparse
import json
from pathlib import Path
import re

import torch
from torch.utils.data import DataLoader

from evaluate_predictor import evaluate_once
from hdp_nuplan.model.hyper_diffusion_planner import Hyper_Diffusion_Planner
from hdp_nuplan.utils.config import Config
from hdp_nuplan.utils.dataset import DiffusionPlannerData
from hdp_nuplan.utils.train_utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--args_file", required=True, type=Path)
    parser.add_argument("--checkpoint_dir", required=True, type=Path)
    parser.add_argument("--pattern", default="model_epoch_*.pth")
    parser.add_argument("--data_dir", required=True, type=Path)
    parser.add_argument("--data_list", required=True, type=Path)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--repeats", default=1, type=int)
    parser.add_argument("--seed", default=3407, type=int)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _load_checkpoint(model, checkpoint_path: Path) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("ema_state_dict", checkpoint.get("model", checkpoint))
    state_dict = {
        (key[len("module."):] if key.startswith("module.") else key): value
        for key, value in state_dict.items()
    }
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return {
        "epoch": checkpoint.get("epoch"),
        "train_loss": checkpoint.get("loss"),
    }


def rank_records(records: list[dict]) -> list[dict]:
    """按验证 total loss 升序排名，不改变调用方传入的原始记录。"""
    ranked = sorted(records, key=lambda record: record["metrics"]["loss"])
    return [dict(record, rank=index) for index, record in enumerate(ranked, start=1)]


def checkpoint_sort_key(path: Path):
    """按 epoch 数值排序，避免 epoch 10 被排在 epoch 2 前面。"""
    match = re.search(r"model_epoch_(\d+)", path.name)
    if match is None:
        return (float("inf"), path.name)
    return (int(match.group(1)), path.name)


def main():
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested but CUDA is unavailable")

    checkpoints = sorted(args.checkpoint_dir.glob(args.pattern), key=checkpoint_sort_key)
    if not checkpoints:
        raise FileNotFoundError(
            f"no checkpoints matched {args.pattern!r} in {args.checkpoint_dir}"
        )

    device = torch.device(args.device)
    config = Config(args.args_file)
    dataset = DiffusionPlannerData(
        str(args.data_dir),
        str(args.data_list),
        config.agent_num,
        config.predicted_neighbor_num,
        config.future_len,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device == "cuda",
        drop_last=False,
    )
    model = Hyper_Diffusion_Planner(config).to(device)

    records = []
    for checkpoint_path in checkpoints:
        checkpoint_metadata = _load_checkpoint(model, checkpoint_path)
        per_repeat = []
        for repeat in range(args.repeats):
            set_seed(args.seed + repeat)
            per_repeat.append(evaluate_once(model, data_loader, config, device))
        metrics = {
            key: sum(item[key] for item in per_repeat) / len(per_repeat)
            for key in per_repeat[0]
        }
        record = {
            "checkpoint": str(checkpoint_path.resolve()),
            **checkpoint_metadata,
            "metrics": metrics,
            "per_repeat": per_repeat,
        }
        records.append(record)
        print(
            f"epoch={record['epoch']} val_loss={metrics['loss']:.6f} "
            f"checkpoint={checkpoint_path.name}"
        )

    ranked = rank_records(records)
    result = {
        "args_file": str(args.args_file.resolve()),
        "checkpoint_dir": str(args.checkpoint_dir.resolve()),
        "pattern": args.pattern,
        "data_list": str(args.data_list.resolve()),
        "samples": len(dataset),
        "repeats": args.repeats,
        "seed": args.seed,
        "checkpoint_count": len(checkpoints),
        "best_checkpoint": ranked[0]["checkpoint"],
        "best_metrics": ranked[0]["metrics"],
        "ranked": ranked,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=4) + "\n", encoding="utf-8")
    print(f"Saved checkpoint ranking to {args.output}")


if __name__ == "__main__":
    main()
