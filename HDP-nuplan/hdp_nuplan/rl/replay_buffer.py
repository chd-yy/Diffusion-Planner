from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import Deque, List, Optional, Sequence

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
    # None 仅兼容未开启过滤的历史实验；开启过滤后必须存在显式 bool mask。
    candidate_mask: Optional[torch.Tensor] = None  # [G] bool，保存在 CPU
    # 更新掩码只决定哪些安全候选能成为回归目标；不能用于重算组内优势。
    update_mask: Optional[torch.Tensor] = None  # [G] bool，保存在 CPU
    # 外部质量空间门控。None 保持历史的 scalar-reward-std 行为。
    active_group: Optional[torch.Tensor] = None  # scalar bool，保存在 CPU
    # Candidate source is retained for rollout diagnostics only.
    source_ids: Optional[torch.Tensor] = None  # [G] int64


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
        update_mask=None,
        active_group=None,
        reference_reward=None,
        source_ids=None,
    ):
        if trajectories.ndim != 3:
            raise ValueError("trajectories must have shape [G, T, D]")
        if rewards.ndim != 1 or rewards.shape[0] != trajectories.shape[0]:
            raise ValueError("rewards must have shape [G] and match trajectories")
        if not torch.isfinite(trajectories).all() or not torch.isfinite(rewards).all():
            raise ValueError("Replay trajectories and rewards must be finite")
        if candidate_mask is not None:
            if candidate_mask.ndim != 1 or candidate_mask.shape != rewards.shape:
                raise ValueError(
                    "candidate_mask must have shape [G] and match rewards"
                )
            if candidate_mask.dtype != torch.bool:
                raise ValueError("candidate_mask must be boolean, not an implicit numeric qualification")
            candidate_mask = candidate_mask.detach().cpu()
        if update_mask is not None:
            if update_mask.ndim != 1 or update_mask.shape != rewards.shape:
                raise ValueError("update_mask must have shape [G] and match rewards")
            if update_mask.dtype != torch.bool:
                raise ValueError("update_mask must be boolean")
            if candidate_mask is not None and torch.any(
                update_mask & ~candidate_mask.to(update_mask.device)
            ):
                raise ValueError("update_mask cannot enable an ineligible candidate")
            update_mask = update_mask.detach().cpu()
        if active_group is not None:
            if active_group.numel() != 1 or active_group.dtype != torch.bool:
                raise ValueError("active_group must be a scalar boolean tensor")
            active_group = active_group.detach().reshape(()).cpu()
        if source_ids is not None:
            if source_ids.ndim != 1 or source_ids.shape != rewards.shape:
                raise ValueError("source_ids must have shape [G]")
            if source_ids.dtype != torch.long:
                raise ValueError("source_ids must use torch.long")
            source_ids = source_ids.detach().cpu()
        if reference_reward is not None:
            if reference_reward.numel() != 1:
                raise ValueError("reference_reward must be a scalar tensor")
            if not torch.isfinite(reference_reward).all():
                raise ValueError("reference_reward must be finite")
            reference_reward = reference_reward.detach().reshape(()).cpu()
        self._items.append(
            NuPlanReplayItem(
                scene_name=str(scene_name),
                trajectories=trajectories.detach().cpu(),
                rewards=rewards.detach().cpu(),
                reference_reward=reference_reward,
                candidate_mask=candidate_mask,
                update_mask=update_mask,
                active_group=active_group,
                source_ids=source_ids,
            )
        )

    def sample(self, batch_size: int) -> List[NuPlanReplayItem]:
        if not self._items:
            raise ValueError("cannot sample from an empty replay buffer")
        return random.choices(tuple(self._items), k=batch_size)

    def get_items(self, indices: Sequence[int]) -> List[NuPlanReplayItem]:
        items = tuple(self._items)
        return [items[int(index)] for index in indices]

    def ordered_indices(self, indices: Sequence[int], rng) -> List[int]:
        ordered = [int(index) for index in indices]
        rng.shuffle(ordered)
        return ordered

    def sample_target_aware(
        self,
        batch_size: int,
        target_flags: Sequence[bool],
        target_fraction: float,
    ) -> List[NuPlanReplayItem]:
        """Sample a controlled minimum fraction from target-bearing groups.

        The remaining slots are sampled from the complete replay, so expert and
        reference anchors retain broad scene coverage.  This changes only the
        update sampler; rollout coverage and replay contents are untouched.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not 0 < target_fraction <= 1:
            raise ValueError("target_fraction must be in (0, 1]")
        items = tuple(self._items)
        if len(target_flags) != len(items):
            raise ValueError("target_flags must align with replay items")
        target_items = [
            item for item, is_target in zip(items, target_flags) if bool(is_target)
        ]
        if not target_items:
            raise RuntimeError(
                "target-aware replay sampling requires at least one target-bearing group"
            )
        target_count = min(
            batch_size,
            max(1, int(batch_size * target_fraction + 0.999999)),
        )
        sampled = random.choices(target_items, k=target_count)
        sampled.extend(random.choices(items, k=batch_size - target_count))
        random.shuffle(sampled)
        return sampled

    def clear(self):
        self._items.clear()

    def finalize(self):
        """Memory replay needs no explicit persistence barrier."""

    def iter_item_batches(self, batch_size: int = 1024):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        items = tuple(self._items)
        for start in range(0, len(items), batch_size):
            yield items[start : start + batch_size]

    def items(self):
        """Return an immutable snapshot for read-only replay audits."""
        return tuple(self._items)

    def __len__(self):
        return len(self._items)


class ShardedNuPlanReplayBuffer:
    """Disk-backed replay that never materializes the complete dataset in RAM."""

    def __init__(self, directory, max_size=None, shard_size=1000, cache_shards=2):
        if shard_size < 1 or cache_shards < 1:
            raise ValueError("shard_size and cache_shards must be positive")
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.max_size = max_size
        self.shard_size = int(shard_size)
        self.cache_shards = int(cache_shards)
        self._staging = []
        self._shard_lengths = []
        self._count = 0
        self._cache = OrderedDict()
        self._validator = NuPlanReplayBuffer(max_size=1)
        self._finalized = False

    def put(self, *args, **kwargs):
        if self._finalized:
            raise RuntimeError("cannot append to a finalized disk replay")
        if self.max_size is not None and self._count >= self.max_size:
            raise RuntimeError(
                "disk replay capacity exhausted; refusing silent FIFO truncation"
            )
        self._validator.clear()
        self._validator.put(*args, **kwargs)
        self._staging.append(self._validator.items()[0])
        self._count += 1
        if len(self._staging) >= self.shard_size:
            self._flush()

    @staticmethod
    def _stack_optional(items, field):
        values = [getattr(item, field) for item in items]
        if all(value is None for value in values):
            return None
        if any(value is None for value in values):
            raise ValueError(f"mixed None/non-None replay field: {field}")
        return torch.stack(values)

    def _flush(self):
        if not self._staging:
            return
        shard_id = len(self._shard_lengths)
        payload = {
            "scene_names": [item.scene_name for item in self._staging],
            "trajectories": torch.stack(
                [item.trajectories for item in self._staging]
            ),
            "rewards": torch.stack([item.rewards for item in self._staging]),
            "reference_rewards": self._stack_optional(
                self._staging, "reference_reward"
            ),
            "candidate_masks": self._stack_optional(
                self._staging, "candidate_mask"
            ),
            "update_masks": self._stack_optional(self._staging, "update_mask"),
            "active_groups": self._stack_optional(
                self._staging, "active_group"
            ),
            "source_ids": self._stack_optional(self._staging, "source_ids"),
        }
        target = self.directory / f"shard_{shard_id:06d}.pt"
        temporary = target.with_suffix(".pt.tmp")
        torch.save(payload, temporary)
        temporary.replace(target)
        self._shard_lengths.append(len(self._staging))
        self._staging.clear()

    def finalize(self):
        if self._finalized:
            return
        self._flush()
        metadata = {
            "schema_version": 1,
            "count": self._count,
            "shard_size": self.shard_size,
            "shard_lengths": self._shard_lengths,
            "storage": "torch_stacked_cpu_tensors",
        }
        target = self.directory / "index.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)
        self._finalized = True

    def _load_shard(self, shard_id):
        shard_id = int(shard_id)
        if shard_id in self._cache:
            payload = self._cache.pop(shard_id)
            self._cache[shard_id] = payload
            return payload
        path = self.directory / f"shard_{shard_id:06d}.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self._cache[shard_id] = payload
        while len(self._cache) > self.cache_shards:
            self._cache.popitem(last=False)
        return payload

    @staticmethod
    def _item_from_payload(payload, offset):
        def optional(name):
            value = payload[name]
            return None if value is None else value[offset]

        return NuPlanReplayItem(
            scene_name=payload["scene_names"][offset],
            trajectories=payload["trajectories"][offset],
            rewards=payload["rewards"][offset],
            reference_reward=optional("reference_rewards"),
            candidate_mask=optional("candidate_masks"),
            update_mask=optional("update_masks"),
            active_group=optional("active_groups"),
            source_ids=optional("source_ids"),
        )

    def get_items(self, indices: Sequence[int]) -> List[NuPlanReplayItem]:
        self.finalize()
        grouped = defaultdict(list)
        for position, raw_index in enumerate(indices):
            index = int(raw_index)
            if index < 0 or index >= self._count:
                raise IndexError(index)
            grouped[index // self.shard_size].append(
                (position, index % self.shard_size)
            )
        result = [None] * len(indices)
        for shard_id, requests in grouped.items():
            payload = self._load_shard(shard_id)
            for position, offset in requests:
                result[position] = self._item_from_payload(payload, offset)
        return result

    def sample(self, batch_size):
        if self._count == 0:
            raise ValueError("cannot sample from an empty replay buffer")
        return self.get_items(random.choices(range(self._count), k=batch_size))

    def sample_target_aware(self, batch_size, target_flags, target_fraction):
        if len(target_flags) != self._count:
            raise ValueError("target_flags must align with replay items")
        target_indices = [
            index for index, flag in enumerate(target_flags) if bool(flag)
        ]
        if not target_indices:
            raise RuntimeError(
                "target-aware replay sampling requires at least one target-bearing group"
            )
        target_count = min(
            batch_size,
            max(1, int(math.ceil(batch_size * target_fraction))),
        )
        indices = random.choices(target_indices, k=target_count)
        indices.extend(
            random.choices(range(self._count), k=batch_size - target_count)
        )
        random.shuffle(indices)
        return self.get_items(indices)

    def ordered_indices(self, indices: Sequence[int], rng) -> List[int]:
        """Shuffle shard order and in-shard order to keep disk access local."""
        grouped = defaultdict(list)
        for raw_index in indices:
            index = int(raw_index)
            grouped[index // self.shard_size].append(index)
        shard_ids = list(grouped)
        rng.shuffle(shard_ids)
        ordered = []
        for shard_id in shard_ids:
            rng.shuffle(grouped[shard_id])
            ordered.extend(grouped[shard_id])
        return ordered

    def iter_item_batches(self, batch_size=1024):
        self.finalize()
        for shard_id, shard_length in enumerate(self._shard_lengths):
            payload = self._load_shard(shard_id)
            for start in range(0, shard_length, batch_size):
                yield tuple(
                    self._item_from_payload(payload, offset)
                    for offset in range(start, min(start + batch_size, shard_length))
                )

    def items(self):
        raise RuntimeError(
            "disk replay cannot materialize all items; use iter_item_batches"
        )

    def clear(self):
        self._staging.clear()
        self._cache.clear()
        self._shard_lengths.clear()
        self._count = 0
        self._finalized = False
        for path in self.directory.glob("shard_*.pt"):
            path.unlink()
        for name in ("index.json", "target_index.json"):
            path = self.directory / name
            if path.exists():
                path.unlink()

    def __len__(self):
        return self._count


class ReplayTargetBatchSampler:
    """No-replacement target coverage with a separate general-scene stream."""

    def __init__(self, replay_buffer, target_flags, batch_size, target_fraction, seed):
        if batch_size < 1 or not 0 < target_fraction <= 1:
            raise ValueError("invalid target batch sampler configuration")
        if len(target_flags) != len(replay_buffer):
            raise ValueError("target flags must align with replay")
        self.replay_buffer = replay_buffer
        self.batch_size = int(batch_size)
        self.target_slots = min(
            self.batch_size,
            max(1, int(math.ceil(self.batch_size * target_fraction))),
        )
        target_indices = [
            index for index, flag in enumerate(target_flags) if bool(flag)
        ]
        if not target_indices:
            raise RuntimeError("no target-bearing groups for no-replacement update")
        general_indices = [
            index for index, flag in enumerate(target_flags) if not bool(flag)
        ]
        rng = random.Random(seed)
        self.targets = replay_buffer.ordered_indices(target_indices, rng)
        self.general = replay_buffer.ordered_indices(general_indices, rng)
        self._rng = rng
        self._target_cursor = 0
        self._general_cursor = 0
        self.steps_for_target_coverage = int(
            math.ceil(len(self.targets) / self.target_slots)
        )

    def _take_general(self, count):
        if count <= 0:
            return []
        if not self.general:
            return self._rng.choices(self.targets, k=count)
        result = []
        while len(result) < count:
            if self._general_cursor >= len(self.general):
                self._general_cursor = 0
                self.general = self.replay_buffer.ordered_indices(
                    self.general, self._rng
                )
            take = min(count - len(result), len(self.general) - self._general_cursor)
            result.extend(
                self.general[self._general_cursor : self._general_cursor + take]
            )
            self._general_cursor += take
        return result

    def next_batch(self):
        if self._target_cursor >= len(self.targets):
            raise StopIteration
        target_end = min(
            self._target_cursor + self.target_slots, len(self.targets)
        )
        indices = self.targets[self._target_cursor : target_end]
        self._target_cursor = target_end
        indices.extend(self._take_general(self.batch_size - len(indices)))
        self._rng.shuffle(indices)
        return self.replay_buffer.get_items(indices)
