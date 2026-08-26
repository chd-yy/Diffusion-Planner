from collections import deque
from dataclasses import dataclass
import random
from typing import Deque, List, Optional

import torch


@dataclass
class NuPlanReplayItem:
    """一个 NuPlan 场景对应的候选组、奖励和可选参考策略奖励。"""

    scene_name: str
    trajectories: torch.Tensor  # [G, T, 4]，保存在 CPU
    rewards: torch.Tensor  # [G]，保存在 CPU
    # 冻结参考策略（通常为 B Epoch10）在同一场景上的 reward，保存为标量。
    # None 用于兼容旧 replay 条目；旧条目仍退化为组内 mean baseline。
    reference_reward: Optional[torch.Tensor] = None
    # 候选是否通过 rollout 阶段的安全门。None 表示旧格式条目，读取时按全 True
    # 兼容；新保守更新会把未通过门的候选从 rollout 回归目标中完全排除。
    candidate_mask: Optional[torch.Tensor] = None  # [G] bool，保存在 CPU


class NuPlanReplayBuffer:
    """按场景保存分组 rollout，训练时进行有放回采样。"""

    def __init__(self, max_size: Optional[int] = None):
        self._items: Deque[NuPlanReplayItem] = deque(maxlen=max_size)

    def put(
        self,
        scene_name,
        trajectories,
        rewards,
        candidate_mask=None,
        reference_reward=None,
    ):
        if trajectories.ndim != 3:
            raise ValueError("trajectories must have shape [G, T, D]")
        if rewards.ndim != 1 or rewards.shape[0] != trajectories.shape[0]:
            raise ValueError("rewards must have shape [G] and match trajectories")
        if candidate_mask is not None:
            if candidate_mask.ndim != 1 or candidate_mask.shape != rewards.shape:
                raise ValueError(
                    "candidate_mask must have shape [G] and match rewards"
                )
            candidate_mask = candidate_mask.detach().to(dtype=torch.bool).cpu()
        if reference_reward is not None:
            if reference_reward.numel() != 1:
                raise ValueError("reference_reward must be a scalar tensor")
            reference_reward = reference_reward.detach().reshape(()).cpu()
        self._items.append(
            NuPlanReplayItem(
                scene_name=str(scene_name),
                trajectories=trajectories.detach().cpu(),
                rewards=rewards.detach().cpu(),
                reference_reward=reference_reward,
                candidate_mask=candidate_mask,
            )
        )

    def sample(self, batch_size: int) -> List[NuPlanReplayItem]:
        if not self._items:
            raise ValueError("cannot sample from an empty replay buffer")
        return random.choices(tuple(self._items), k=batch_size)

    def clear(self):
        self._items.clear()

    def __len__(self):
        return len(self._items)
