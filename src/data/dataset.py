"""
PyTorch Dataset - 广告预估训练数据加载

支持从 .npz 文件加载预处理后的数据，提供：
- 用户特征
- 广告特征（含语义ID）
- 上下文特征
- 用户行为序列
- CTR / CVR 标签
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class DSPAdDataset(Dataset):
    """DSP 广告多任务预估数据集。"""

    def __init__(self, npz_path: str, feature_config: dict[str, Any],
                 ad_semantic_ids: np.ndarray | None = None):
        """
        Args:
            npz_path: 预处理数据 .npz 文件路径。
            feature_config: 配置中的 data 部分。
            ad_semantic_ids: 广告语义ID映射表 {ad编码ID -> semantic_id向量}，
                             shape (num_ads, sid_length)。若为 None 则不使用语义ID。
        """
        data = np.load(npz_path, allow_pickle=True)

        self.user_feature_names = feature_config["user_features"]
        self.ad_feature_names = feature_config["ad_features"]
        self.context_feature_names = feature_config["context_features"]
        self.stat_feature_names = feature_config.get("stat_features", [])
        self.label_names = feature_config["labels"]

        # 加载各类特征
        self.user_features = {
            name: torch.from_numpy(data[name]) for name in self.user_feature_names
            if name in data
        }
        self.ad_features = {
            name: torch.from_numpy(data[name]) for name in self.ad_feature_names
            if name in data
        }
        self.context_features = {
            name: torch.from_numpy(data[name]) for name in self.context_feature_names
            if name in data
        }
        self.stat_features = {
            name: torch.from_numpy(data[name]) for name in self.stat_feature_names
            if name in data
        }

        # 行为序列
        self.behavior_seq = (
            torch.from_numpy(data["behavior_seq"])
            if "behavior_seq" in data else None
        )

        # 标签
        self.labels = {
            name: torch.from_numpy(data[name]) for name in self.label_names
            if name in data
        }

        # 广告语义ID（如果有的话）
        self.ad_semantic_ids = None
        if ad_semantic_ids is not None:
            self.ad_semantic_ids = torch.from_numpy(ad_semantic_ids)

        # 样本数
        self._length = len(next(iter(self.labels.values())))

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = {}

        # 用户特征
        for name, tensor in self.user_features.items():
            sample[f"user_{name}"] = tensor[idx]

        # 广告特征
        for name, tensor in self.ad_features.items():
            sample[f"ad_{name}"] = tensor[idx]

        # 上下文特征
        for name, tensor in self.context_features.items():
            sample[f"ctx_{name}"] = tensor[idx]

        # 统计特征
        for name, tensor in self.stat_features.items():
            sample[f"stat_{name}"] = tensor[idx]

        # 行为序列
        if self.behavior_seq is not None:
            sample["behavior_seq"] = self.behavior_seq[idx]

        # 广告语义ID (通过 ad_id 查找)
        if self.ad_semantic_ids is not None and "ad_ad_id" in sample:
            ad_idx = sample["ad_ad_id"].long()
            if ad_idx < len(self.ad_semantic_ids):
                sample["ad_semantic_id"] = self.ad_semantic_ids[ad_idx]
            else:
                sample["ad_semantic_id"] = torch.zeros(
                    self.ad_semantic_ids.shape[1], dtype=torch.long
                )

        # 标签
        for name, tensor in self.labels.items():
            sample[f"label_{name}"] = tensor[idx]

        return sample


def create_dataloader(npz_path: str, feature_config: dict[str, Any],
                      batch_size: int, shuffle: bool = True,
                      num_workers: int = 4,
                      ad_semantic_ids: np.ndarray | None = None) -> DataLoader:
    """创建 DataLoader 的便捷函数。"""
    dataset = DSPAdDataset(npz_path, feature_config, ad_semantic_ids)
    use_workers = num_workers > 0
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=shuffle,
        persistent_workers=use_workers,
        prefetch_factor=4 if use_workers else None,
    )
