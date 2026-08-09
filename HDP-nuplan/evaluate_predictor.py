"""Evaluate a supervised HDP-nuPlan checkpoint on a cached NPZ split."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from hdp_nuplan.loss import diffusion_loss_func
from hdp_nuplan.model.hyper_diffusion_planner import Hyper_Diffusion_Planner
from hdp_nuplan.rl.train_epoch_rl import prepare_nuplan_batch
from hdp_nuplan.utils.config import Config
from hdp_nuplan.utils.dataset import DiffusionPlannerData
from hdp_nuplan.utils.train_utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--args_file", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data_dir", required=True, type=Path)
    parser.add_argument("--data_list", required=True, type=Path)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--repeats", default=3, type=int)
    parser.add_argument("--seed", default=3407, type=int)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_ema_model(config, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("ema_state_dict", checkpoint.get("model", checkpoint))
    state_dict = {
        (key[len("module."):] if key.startswith("module.") else key): value
        for key, value in state_dict.items()
    }
    model = Hyper_Diffusion_Planner(config).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


@torch.no_grad()
def evaluate_once(model, data_loader, config, device):
    totals = {
        "ego_planning_loss": 0.0,
        "ego_planning_hybrid_loss": 0.0,
        "loss": 0.0,
    }
    sample_count = 0
    for batch in data_loader:
        model_inputs, _, ego_future, neighbors_future, neighbor_mask, _ = prepare_nuplan_batch(
            batch,
            device,
            config.observation_normalizer,
            with_metadata=False,
        )
        loss, _ = diffusion_loss_func(
            model,
            model_inputs,
            model.sde,
            (ego_future, neighbors_future, neighbor_mask),
            config.state_normalizer,
            {},
            config.diffusion_model_type,
            config.diffusion_supervision_type,
        )
        loss["loss"] = (
            loss["ego_planning_loss"]
            + config.planning_hybrid_loss * loss["ego_planning_hybrid_loss"]
        )
        batch_size = ego_future.shape[0]
        sample_count += batch_size
        for key in totals:
            totals[key] += loss[key].item() * batch_size
    return {key: value / sample_count for key, value in totals.items()}


def main():
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested but CUDA is unavailable")

    device = torch.device(args.device)
    config = Config(args.args_file)
    model = load_ema_model(config, args.checkpoint, device)
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

    repeat_metrics = []
    for repeat in range(args.repeats):
        set_seed(args.seed + repeat)
        repeat_metrics.append(evaluate_once(model, data_loader, config, device))
    metrics = {
        key: sum(record[key] for record in repeat_metrics) / len(repeat_metrics)
        for key in repeat_metrics[0]
    }
    result = {
        "checkpoint": str(args.checkpoint),
        "data_list": str(args.data_list),
        "samples": len(dataset),
        "repeats": args.repeats,
        "seed": args.seed,
        "metrics": metrics,
        "per_repeat": repeat_metrics,
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
