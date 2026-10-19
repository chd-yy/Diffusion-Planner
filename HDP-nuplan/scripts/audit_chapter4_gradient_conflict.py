#!/usr/bin/env python3
"""Audit decoder-gradient conflict among rollout, expert and reference losses.

This script never calls optimizer.step().  It rebuilds the exact constrained
rollout targets from B10, evaluates gradients at a candidate checkpoint, and
reports loss-component norms and pairwise cosines on fixed training scenes.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from hdp_nuplan.loss import diffusion_loss_func
from hdp_nuplan.rl.loss import reward_weighted_diffusion_loss
from hdp_nuplan.rl.replay_buffer import NuPlanReplayBuffer
from hdp_nuplan.rl.reward import NuPlanTensorRewardScorer
from hdp_nuplan.rl.train_epoch_rl import (
    _load_replay_batch,
    prepare_nuplan_batch,
    rollout_epoch,
)
from hdp_nuplan.utils.config import Config
from hdp_nuplan.utils.dataset import DiffusionPlannerData
from hdp_nuplan.utils.train_utils import set_seed
from scripts.compare_checkpoint_behavior import (
    load_checkpoint_model,
    reward_config_from_args,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("args-file", "base-checkpoint", "candidate-checkpoint",
                 "data-dir", "data-list", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--scene-count", default=64, type=int)
    parser.add_argument("--batch-size", default=2, type=int)
    parser.add_argument("--num-workers", default=2, type=int)
    parser.add_argument("--seed", default=2026, type=int)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def grad_norm(grads) -> float:
    total = torch.zeros((), dtype=torch.float64)
    for grad in grads:
        if grad is not None:
            total += grad.detach().double().square().sum().cpu()
    return math.sqrt(float(total))


def grad_cosine(left, right) -> float:
    dot = torch.zeros((), dtype=torch.float64)
    left_sq = torch.zeros((), dtype=torch.float64)
    right_sq = torch.zeros((), dtype=torch.float64)
    for lhs, rhs in zip(left, right):
        if lhs is None or rhs is None:
            continue
        lhs = lhs.detach().double()
        rhs = rhs.detach().double()
        dot += (lhs * rhs).sum().cpu()
        left_sq += lhs.square().sum().cpu()
        right_sq += rhs.square().sum().cpu()
    denominator = math.sqrt(float(left_sq) * float(right_sq))
    return float(dot) / denominator if denominator > 0 else None


def add_grads(*weighted_grads):
    result = []
    for entries in zip(*(grads for _, grads in weighted_grads)):
        combined = None
        for (weight, _), grad in zip(weighted_grads, entries):
            if grad is None:
                continue
            term = grad.detach() * weight
            combined = term if combined is None else combined + term
        result.append(combined)
    return tuple(result)


def summarize(values):
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if not finite.size:
        return {"count": int(array.size), "finite_count": 0}
    return {
        "count": int(array.size),
        "finite_count": int(finite.size),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "p10": float(np.quantile(finite, 0.10)),
        "p50": float(np.quantile(finite, 0.50)),
        "p90": float(np.quantile(finite, 0.90)),
        "negative_fraction": float(np.mean(finite < 0)),
    }


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    config = Config(args.args_file)
    config.ddp = False
    if config.rl_weighting_mode != "constraint_uniform_positive":
        raise ValueError("args do not describe the constrained target protocol")
    device = torch.device("cuda")
    base, base_meta = load_checkpoint_model(
        config, args.base_checkpoint, device, requested_source="ema_state_dict"
    )
    candidate, candidate_meta = load_checkpoint_model(
        config, args.candidate_checkpoint, device, requested_source="ema_state_dict"
    )
    reference, _ = load_checkpoint_model(
        config, args.base_checkpoint, device, requested_source="ema_state_dict"
    )
    for name, parameter in candidate.named_parameters():
        parameter.requires_grad_("decoder" in name)
    parameters = [parameter for parameter in candidate.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("no decoder parameters selected")

    dataset = DiffusionPlannerData(
        str(args.data_dir), str(args.data_list), config.agent_num,
        config.predicted_neighbor_num, config.future_len, return_metadata=True,
    )
    scene_count = min(args.scene_count, len(dataset))
    subset = Subset(dataset, list(range(scene_count)))
    loader = DataLoader(
        subset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
    )
    scorer = NuPlanTensorRewardScorer(reward_config_from_args(config))
    replay = NuPlanReplayBuffer(max_size=scene_count)
    set_seed(args.seed)
    rollout_metrics = rollout_epoch(
        loader, base, replay, scorer, config, device, epoch=0,
        reference_model=reference,
    )
    items = list(replay._items)
    records = []
    storage = defaultdict(list)
    for batch_index in range(0, len(items), args.batch_size):
        replay_items = items[batch_index:batch_index + args.batch_size]
        (
            replay_batch, trajectories, rewards, candidate_masks,
            _, update_masks, active_masks,
        ) = _load_replay_batch(
            dataset, replay_items, require_candidate_mask=True,
            require_update_contract=True,
        )
        model_inputs, _, ego_future, neighbors_future, neighbor_mask, _ = (
            prepare_nuplan_batch(
                replay_batch, device, config.observation_normalizer,
                with_metadata=False,
            )
        )
        trajectories = trajectories.to(device)
        rewards = rewards.to(device)
        candidate_masks = candidate_masks.to(device)
        update_masks = update_masks.to(device)
        active_masks = active_masks.to(device)
        set_seed(args.seed + 10000 + batch_index)
        rl_loss, metrics = reward_weighted_diffusion_loss(
            model=candidate,
            inputs=model_inputs,
            trajectories=trajectories,
            rewards=rewards,
            state_normalizer=config.state_normalizer,
            sde=candidate.sde,
            model_type=config.diffusion_model_type,
            supervision_type=config.diffusion_supervision_type,
            hybrid_loss_weight=config.planning_hybrid_loss,
            detach_window_size=config.rl_detach_window_size,
            reward_temperature=config.rl_reward_temperature,
            advantage_clip=config.rl_advantage_clip,
            min_reward_std=config.rl_min_reward_std,
            normalize_weights=config.rl_normalize_weights,
            center_reward_weights=config.rl_center_reward_weights,
            weighting_mode=config.rl_weighting_mode,
            candidate_mask=candidate_masks,
            update_mask=update_masks,
            active_group_mask=active_masks,
            positive_margin=config.rl_positive_advantage_margin,
            reference_model=reference,
        )
        anchor_dict, _ = diffusion_loss_func(
            candidate, model_inputs, candidate.sde,
            (ego_future, neighbors_future, neighbor_mask),
            config.state_normalizer, {}, config.diffusion_model_type,
            config.diffusion_supervision_type,
            detach_window_size=config.rl_detach_window_size,
            trajectory_representation=config.trajectory_representation,
        )
        expert_loss = (
            anchor_dict["ego_planning_loss"]
            + config.planning_hybrid_loss
            * anchor_dict["ego_planning_hybrid_loss"]
        )
        reference_loss = metrics["reference_anchor_loss"]
        rl_grads = torch.autograd.grad(
            rl_loss, parameters, retain_graph=True, allow_unused=True
        )
        reference_grads = torch.autograd.grad(
            reference_loss, parameters, retain_graph=False, allow_unused=True
        )
        expert_grads = torch.autograd.grad(
            expert_loss, parameters, retain_graph=False, allow_unused=True
        )
        combined = add_grads(
            (config.rl_rollout_loss_weight, rl_grads),
            (config.rl_expert_anchor_weight, expert_grads),
            (config.rl_reference_anchor_weight, reference_grads),
        )
        record = {
            "scene_names": [item.scene_name for item in replay_items],
            "has_rollout_target": bool(metrics["has_regression_targets"]),
            "target_count": int((metrics["regression_targets_per_group"] * len(replay_items)).item()),
            "rl_loss": float(rl_loss.detach()),
            "expert_loss": float(expert_loss.detach()),
            "reference_loss": float(reference_loss.detach()),
            "rl_grad_norm": grad_norm(rl_grads),
            "weighted_expert_grad_norm": config.rl_expert_anchor_weight * grad_norm(expert_grads),
            "weighted_reference_grad_norm": config.rl_reference_anchor_weight * grad_norm(reference_grads),
            "combined_grad_norm": grad_norm(combined),
            "cos_rl_expert": grad_cosine(rl_grads, expert_grads),
            "cos_rl_reference": grad_cosine(rl_grads, reference_grads),
            "cos_expert_reference": grad_cosine(expert_grads, reference_grads),
        }
        records.append(record)
        if record["has_rollout_target"]:
            for key, value in record.items():
                if isinstance(value, (float, int)) and key != "target_count":
                    storage[key].append(float(value))
        print(
            f"{datetime.now().isoformat(timespec='seconds')} "
            f"batch={batch_index // args.batch_size + 1}/{math.ceil(len(items)/args.batch_size)} "
            f"target={int(record['has_rollout_target'])}", flush=True,
        )

    output = {
        "schema_version": 1,
        "created_at": datetime.now().strftime("%m月%d日 %H:%M"),
        "weights_updated": False,
        "optimizer_step_called": False,
        "protocol": {
            "scene_count": scene_count,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "parameter_scope": "trainable decoder parameters",
            "rollout_source": "B10 EMA",
            "gradient_evaluation_source": "20-step candidate EMA",
            "loss_weights": {
                "rollout": config.rl_rollout_loss_weight,
                "expert": config.rl_expert_anchor_weight,
                "reference": config.rl_reference_anchor_weight,
            },
        },
        "identity": {
            "args_file": str(args.args_file.resolve()),
            "args_sha256": sha256(args.args_file),
            "data_list": str(args.data_list.resolve()),
            "data_list_sha256": sha256(args.data_list),
            "base": base_meta,
            "candidate": candidate_meta,
        },
        "rollout_metrics": rollout_metrics,
        "target_batch_count": int(sum(record["has_rollout_target"] for record in records)),
        "batch_count": len(records),
        "summary_on_target_batches": {
            key: summarize(values) for key, values in storage.items()
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = json_safe(output)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "sha256": sha256(args.output),
        "target_batches": output["target_batch_count"],
        "batch_count": output["batch_count"],
        "summary": output["summary_on_target_batches"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
