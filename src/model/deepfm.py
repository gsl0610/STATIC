"""
DeepFM 模型 — 传统机器学习 Baseline (AB 对比测试用)

DeepFM = FM (二阶特征交互) + DNN (高阶非线性)
参考:
  - 原始论文: Guo et al., DeepFM: A Factorization-Machine based Neural Network
    for CTR Prediction, IJCAI 2017
  - 工业实践: 美团/快手/字节等大厂广告系统中广泛使用的经典CTR预估模型

与混元 LLM 方案的核心差异:
  - 特征交互: FM 显式二阶 + DNN 隐式高阶 vs LLM 全局注意力
  - 参数量: ~2-10M vs ~1.8B (轻量 200x+)
  - 行为建模: 简单 pooling vs Transformer 序列编码
  - 推理延迟: <1ms vs ~13ms (GPU)
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class FMLayer(nn.Module):
    """Factorization Machine 二阶交互层。

    FM 的二阶交互可以通过以下公式在 O(kn) 时间计算:
      0.5 * (sum_i(v_i*x_i))^2 - sum_i((v_i*x_i)^2)
    其中 v_i 是第 i 个特征的 embedding 向量。
    """

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (B, num_fields, embed_dim)
        Returns:
            fm_output: (B, 1)
        """
        # sum of square vs square of sum
        sum_square = torch.sum(embeddings, dim=1) ** 2       # (B, embed_dim)
        square_sum = torch.sum(embeddings ** 2, dim=1)       # (B, embed_dim)
        fm_output = 0.5 * torch.sum(sum_square - square_sum, dim=1, keepdim=True)
        return fm_output  # (B, 1)


class DeepFMModel(nn.Module):
    """DeepFM 多任务模型 (CTR + CVR 联合预估)。

    架构:
        所有特征 → Embedding → ┬─ FM 二阶交互 ─┬─→ CTR 头
                               └─ DNN 高阶非线性 ┘
                               ┬─ FM 二阶交互 ─┬─→ CVR 头
                               └─ DNN 高阶非线性 ┘

    特征处理:
        - 单值类别特征: Embedding(vocab_size, embed_dim)
        - 多值类别特征: Embedding + mean pooling
        - 行为序列: Embedding + mean pooling (简化版, 不使用 attention)
        - 统计特征: 与类别特征统一处理 (已分桶)
    """

    def __init__(self, config: dict[str, Any],
                 user_feature_vocab_sizes: dict[str, int],
                 ad_feature_vocab_sizes: dict[str, int],
                 context_feature_vocab_sizes: dict[str, int],
                 stat_feature_vocab_sizes: dict[str, int] | None = None,
                 behavior_vocab_size: int = 10000):
        super().__init__()

        self.embed_dim = 16  # 每个特征的 embedding 维度
        data_cfg = config["data"]

        # ---------- 特征 Embedding ----------
        self.user_feature_names = sorted(user_feature_vocab_sizes.keys())
        self.ad_feature_names = sorted(ad_feature_vocab_sizes.keys())
        self.ctx_feature_names = sorted(context_feature_vocab_sizes.keys())
        self.stat_feature_names = sorted((stat_feature_vocab_sizes or {}).keys())

        self.multi_value_feats = {"interest_tags", "interest_tags_l2", "creative_label"}

        # 所有特征的 embedding
        self.embeddings = nn.ModuleDict()
        all_vocab = {}
        for name, vsize in user_feature_vocab_sizes.items():
            all_vocab[f"user_{name}"] = vsize
        for name, vsize in ad_feature_vocab_sizes.items():
            all_vocab[f"ad_{name}"] = vsize
        for name, vsize in context_feature_vocab_sizes.items():
            all_vocab[f"ctx_{name}"] = vsize
        for name, vsize in (stat_feature_vocab_sizes or {}).items():
            all_vocab[f"stat_{name}"] = vsize

        for key, vsize in all_vocab.items():
            self.embeddings[key] = nn.Embedding(vsize, self.embed_dim, padding_idx=0)

        # 行为序列 embedding
        self.behavior_embedding = nn.Embedding(
            behavior_vocab_size + 1, self.embed_dim, padding_idx=0
        )

        # 总特征域数 = 所有特征 + 行为序列 (pooled 为 1 个域)
        self.num_fields = len(all_vocab) + 1  # +1 for behavior

        # ---------- FM 层 ----------
        self.fm_layer = FMLayer()

        # ---------- 一阶偏置 ----------
        self.linear_weights = nn.ModuleDict()
        for key, vsize in all_vocab.items():
            self.linear_weights[key] = nn.Embedding(vsize, 1, padding_idx=0)
        self.behavior_linear = nn.Embedding(behavior_vocab_size + 1, 1, padding_idx=0)
        self.global_bias = nn.Parameter(torch.zeros(1))

        # ---------- DNN 层 ----------
        dnn_input_dim = self.num_fields * self.embed_dim
        dropout = config["model"].get("dropout", 0.3)

        # 共享 DNN 底层
        self.shared_dnn = nn.Sequential(
            nn.Linear(dnn_input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # CTR Tower
        self.ctr_tower = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

        # CVR Tower (ESMM 思路: 共享底层, 独立塔)
        self.cvr_tower = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

        # 保存配置
        self.vocab_size = config["model"].get("vocab_size", 1024)
        self._all_vocab_keys = sorted(all_vocab.keys())

        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"DeepFM 模型创建: {total_params / 1e6:.2f}M 参数, "
                    f"{self.num_fields} 个特征域, embed_dim={self.embed_dim}")

    def _get_embeddings(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """提取所有特征的 embedding 并组装为 (B, num_fields, embed_dim)。"""
        field_embeds = []

        for key in self._all_vocab_keys:
            if key in batch:
                emb = self.embeddings[key](batch[key].long())
                if emb.dim() == 3:
                    emb = emb.mean(dim=1)  # 多值特征 mean pooling
                field_embeds.append(emb)
            else:
                # 缺失特征用零向量
                B = next(iter(batch.values())).size(0)
                device = next(iter(batch.values())).device
                field_embeds.append(torch.zeros(B, self.embed_dim, device=device))

        # 行为序列
        behavior_seq = batch.get(
            "behavior_seq",
            torch.zeros(field_embeds[0].size(0), 1, dtype=torch.long,
                        device=field_embeds[0].device)
        )
        beh_emb = self.behavior_embedding(behavior_seq.long())  # (B, seq_len, embed_dim)
        mask = (behavior_seq != 0).unsqueeze(-1).float()
        beh_pooled = (beh_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        field_embeds.append(beh_pooled)

        return torch.stack(field_embeds, dim=1)  # (B, num_fields, embed_dim)

    def _get_linear_output(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """一阶线性部分。"""
        linear_out = self.global_bias.expand(next(iter(batch.values())).size(0), 1)

        for key in self._all_vocab_keys:
            if key in batch:
                w = self.linear_weights[key](batch[key].long())
                if w.dim() == 3:
                    w = w.mean(dim=1)
                linear_out = linear_out + w

        behavior_seq = batch.get(
            "behavior_seq",
            torch.zeros(linear_out.size(0), 1, dtype=torch.long,
                        device=linear_out.device)
        )
        beh_w = self.behavior_linear(behavior_seq.long())
        mask = (behavior_seq != 0).unsqueeze(-1).float()
        beh_linear = (beh_w * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        linear_out = linear_out + beh_linear

        return linear_out  # (B, 1)

    def forward(self, batch: dict[str, torch.Tensor],
                target_sids: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """
        Args:
            batch: 特征字典 (与 HunyuanDSPModel 相同的 key 命名)
            target_sids: 忽略 (DeepFM 不支持生成式检索, 仅兼容接口)

        Returns:
            包含 ctr_logit, ctr_prob, cvr_logit, cvr_prob, user_repr 的字典
        """
        # 获取所有特征 embedding
        embeddings = self._get_embeddings(batch)  # (B, num_fields, embed_dim)

        # FM 二阶交互
        fm_out = self.fm_layer(embeddings)  # (B, 1)

        # 一阶线性
        linear_out = self._get_linear_output(batch)  # (B, 1)

        # DNN
        dnn_input = embeddings.view(embeddings.size(0), -1)  # (B, num_fields * embed_dim)
        shared_repr = self.shared_dnn(dnn_input)  # (B, 256)

        # CTR
        ctr_dnn = self.ctr_tower(shared_repr)  # (B, 1)
        ctr_logit = (linear_out + fm_out + ctr_dnn).squeeze(-1)  # (B,)

        # CVR
        cvr_dnn = self.cvr_tower(shared_repr)  # (B, 1)
        cvr_logit = (linear_out + fm_out + cvr_dnn).squeeze(-1)  # (B,)

        result = {
            "ctr_logit": ctr_logit,
            "ctr_prob": torch.sigmoid(ctr_logit),
            "cvr_logit": cvr_logit,
            "cvr_prob": torch.sigmoid(cvr_logit),
            "user_repr": shared_repr,  # (B, 256) — 供兼容接口
        }

        # 生成式检索占位 (DeepFM 不支持, 返回随机 logits 兼容训练器)
        if target_sids is not None:
            B, L = target_sids.shape
            result["retrieval_logits"] = torch.zeros(
                B, L, self.vocab_size, device=ctr_logit.device
            )

        return result

    def get_user_representation(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """兼容接口: 获取用户表示。"""
        embeddings = self._get_embeddings(batch)
        dnn_input = embeddings.view(embeddings.size(0), -1)
        return self.shared_dnn(dnn_input)
