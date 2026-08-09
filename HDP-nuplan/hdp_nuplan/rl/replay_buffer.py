from collections import deque
from dataclasses import dataclass
import random
from typing import Deque, List, Optional

import torch


@dataclass
class NuPlanReplayItem:
    """一个 NuPlan 场景对应的一组候选轨迹和绝对奖励。"""

    scene_name: str
    trajectories: torch.Tensor  # [G, T, 4]，保存在 CPU
    rewards: torch.Tensor  # [G]，保存在 CPU


class NuPlanReplayBuffer:
    """按场景保存分组 rollout，训练时进行有放回采样。"""

    def __init__(self, max_size: Optional[int] = None):
        self._items: Deque[NuPlanReplayItem] = deque(maxlen=max_size)

    def put(self, scene_name, trajectories, rewards):
        if trajectories.ndim != 3:
            raise ValueError("trajectories must have shape [G, T, D]")
        if rewards.ndim != 1 or rewards.shape[0] != trajectories.shape[0]:
            raise ValueError("rewards must have shape [G] and match trajectories")
        self._items.append(
            NuPlanReplayItem(
                scene_name=str(scene_name),
                trajectories=trajectories.detach().cpu(),
                rewards=rewards.detach().cpu(),
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
