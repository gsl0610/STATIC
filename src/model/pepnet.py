"""
PEPNet 模型 — Parameter and Embedding Personalized Network (CTR/CVR 多任务预估)

参考论文:
  PEPNet: Parameter and Embedding Personalized Network for Infusing with
  Personalized Prior Information, KDD 2023
  https://arxiv.org/pdf/2302.01115

参考实现:
  https://github.com/QunBB/RecSys/blob/main/recsys/multidomain/pepnet.py

核心组件:
  - GateNU: 门控神经单元 (ReLU → Sigmoid × gamma)
  - EPNet: Embedding Personalized Network (域特征个性化 embedding)
  - PPNet: Parameter Personalized Network (参数个性化多任务塔)

与混元 LLM / DeepFM 的核心差异:
  - 个性化先验: 通过 EPNet/PPNet 门控实现 embedding/参数级别的个性化
  - 多任务: 天然支持多域多任务 (CTR/CVR 各自独立参数化)
  - 参数量: ~5-15M (介于 DeepFM 和混元之间)
  - 推理延迟: ~1-3ms (接近 DeepFM)
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ================================================================
# 核心组件
# ================================================================
class GateNU(nn.Module):
    """Gate Neural Unit — PEPNet 的基础门控单元。

    结构: input → Dense(ReLU) → Dense(Sigmoid) → × gamma
    用于根据先验信息动态生成门控权重。
    """

    def __init__(self, input_dim: int, output_dim: int, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.fc2 = nn.Linear(output_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.fc1(x))
        return self.gamma * torch.sigmoid(self.fc2(h))


class EPNet(nn.Module):
    """Embedding Personalized Network — 嵌入个性化网络。

    通过域特征 (domain) 和上下文特征 (context) 生成门控,
    对原始 embedding 进行个性化缩放:
        output = GateNU(concat(domain, stop_grad(emb))) * emb
    """

    def __init__(self, domain_dim: int, emb_dim: int, gamma: float = 2.0):
        super().__init__()
        self.gate_nu = GateNU(domain_dim + emb_dim, emb_dim, gamma=gamma)

    def forward(self, domain: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        gate_input = torch.cat([domain, emb.detach()], dim=-1)
        return self.gate_nu(gate_input) * emb


class PPNet(nn.Module):
    """Parameter Personalized Network — 参数个性化网络。

    包含多个并行任务塔, 每个塔共享 DNN 结构但独立权重,
    通过 GateNU 实现层级的参数个性化:
        每层输出 = GateNU(persona) * Dense(input)

    Args:
        num_tasks: 并行任务数 (CTR + CVR = 2)
        input_dim: DNN 输入维度
        hidden_units: 每层隐藏单元数
        persona_dim: 个性化先验特征维度
        dropout: dropout 比率
        gamma: GateNU 的 gamma 系数
    """

    def __init__(self, num_tasks: int, input_dim: int,
                 hidden_units: list[int], persona_dim: int,
                 dropout: float = 0.3, gamma: float = 2.0):
        super().__init__()
        self.num_tasks = num_tasks
        self.hidden_units = hidden_units

        # 每个任务一套独立的 Dense + Dropout
        self.task_layers = nn.ModuleList()
        self.task_dropouts = nn.ModuleList()
        for _ in range(num_tasks):
            layers = nn.ModuleList()
            drops = nn.ModuleList()
            in_dim = input_dim
            for out_dim in hidden_units:
                layers.append(nn.Linear(in_dim, out_dim))
                drops.append(nn.Dropout(dropout))
                in_dim = out_dim
            self.task_layers.append(layers)
            self.task_dropouts.append(drops)

        # 每层一个 GateNU: 输入是 persona + stop_grad(layer_input)
        # 输出维度 = hidden_unit × num_tasks (split 给每个任务)
        self.layer_gates = nn.ModuleList()
        gate_in_dim = persona_dim + input_dim  # 第一层
        for i, h_dim in enumerate(hidden_units):
            gate_out = h_dim * num_tasks
            self.layer_gates.append(GateNU(gate_in_dim, gate_out, gamma=gamma))
            gate_in_dim = persona_dim + h_dim  # 后续层

    def forward(self, x: torch.Tensor,
                persona: torch.Tensor) -> list[torch.Tensor]:
        """
        Args:
            x: DNN 输入 (B, input_dim)
            persona: 个性化先验特征 (B, persona_dim)
        Returns:
            list of (B, hidden_units[-1]) — 每个任务的输出
        """
        # 计算每层的门控权重
        gate_outputs = []
        layer_input = x
        for i, gate in enumerate(self.layer_gates):
            gate_in = torch.cat([persona, layer_input.detach()], dim=-1)
            gate_out = gate(gate_in)  # (B, h_dim * num_tasks)
            gate_split = torch.chunk(gate_out, self.num_tasks, dim=-1)
            gate_outputs.append(gate_split)
            # 更新 layer_input 为当前层的平均输出 (用于下一层 gate)
            if i < len(self.layer_gates) - 1:
                avg_out = torch.zeros(
                    x.size(0), self.hidden_units[i], device=x.device
                )
                for t in range(self.num_tasks):
                    fc_out = torch.relu(self.task_layers[t][i](layer_input))
                    avg_out = avg_out + gate_split[t] * fc_out
                layer_input = avg_out / self.num_tasks

        # 各任务独立前向
        task_outputs = []
        for t in range(self.num_tasks):
            h = x
            for i in range(len(self.hidden_units)):
                h = self.task_layers[t][i](h)
                h = torch.relu(h)
                h = gate_outputs[i][t] * h  # 门控调节
                h = self.task_dropouts[t][i](h)
            task_outputs.append(h)

        return task_outputs


# ================================================================
# PEPNet 多任务模型
# ================================================================
class PEPNetModel(nn.Module):
    """PEPNet 多任务模型 (CTR + CVR 联合预估)。

    架构:
        所有特征 → Embedding → ┬─ 域/上下文 → EPNet → ep_emb
                               ├─ 用户/广告 → concat → PPNet 输入
                               └─ 个性化先验 (ep_emb) → PPNet persona
                                    ↓
                               PPNet → [CTR Tower, CVR Tower]

    特征处理 (与 DeepFM 一致):
        - 单值类别特征: Embedding(vocab_size, embed_dim)
        - 多值类别特征: Embedding + mean pooling
        - 行为序列: Embedding + mean pooling
    """

    def __init__(self, config: dict[str, Any],
                 user_feature_vocab_sizes: dict[str, int],
                 ad_feature_vocab_sizes: dict[str, int],
                 context_feature_vocab_sizes: dict[str, int],
                 stat_feature_vocab_sizes: dict[str, int] | None = None,
                 behavior_vocab_size: int = 10000):
        super().__init__()

        pepnet_cfg = config.get("pepnet", {})
        self.embed_dim = pepnet_cfg.get("embed_dim", 16)
        self.hidden_units = pepnet_cfg.get("hidden_units", [256, 128])
        self.gamma = pepnet_cfg.get("gamma", 2.0)
        dropout = config["model"].get("dropout", 0.3)

        data_cfg = config["data"]

        # ---------- 特征 Embedding ----------
        self.user_feature_names = sorted(user_feature_vocab_sizes.keys())
        self.ad_feature_names = sorted(ad_feature_vocab_sizes.keys())
        self.ctx_feature_names = sorted(context_feature_vocab_sizes.keys())
        self.stat_feature_names = sorted((stat_feature_vocab_sizes or {}).keys())

        self.multi_value_feats = {"interest_tags", "interest_tags_l2", "creative_label"}

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

        # ---------- 特征分域统计 ----------
        num_user_fields = len(self.user_feature_names) + 1  # +1 for behavior
        num_ad_fields = len(self.ad_feature_names)
        num_ctx_fields = len(self.ctx_feature_names)
        num_stat_fields = len(self.stat_feature_names)

        # domain 特征维度 = (上下文 + 统计) 特征 embedding concat
        self.domain_dim = (num_ctx_fields + num_stat_fields) * self.embed_dim
        # user+ad 特征维度
        self.user_ad_dim = (num_user_fields + num_ad_fields) * self.embed_dim

        # EPNet: 域特征个性化
        self.epnet = EPNet(
            domain_dim=self.domain_dim,
            emb_dim=self.domain_dim,
            gamma=self.gamma,
        )

        # PPNet 输入 = ep_emb + user_ad_concat
        ppnet_input_dim = self.domain_dim + self.user_ad_dim
        # PPNet persona = ep_emb (域个性化后的特征)
        persona_dim = self.domain_dim

        self.ppnet = PPNet(
            num_tasks=2,  # CTR + CVR
            input_dim=ppnet_input_dim,
            hidden_units=self.hidden_units,
            persona_dim=persona_dim,
            dropout=dropout,
            gamma=self.gamma,
        )

        # 预测头
        final_dim = self.hidden_units[-1]
        self.ctr_head = nn.Linear(final_dim, 1)
        self.cvr_head = nn.Linear(final_dim, 1)

        # 保存配置
        self.vocab_size = config["model"].get("vocab_size", 1024)
        self._all_vocab_keys = sorted(all_vocab.keys())
        self._user_keys = [f"user_{n}" for n in self.user_feature_names]
        self._ad_keys = [f"ad_{n}" for n in self.ad_feature_names]
        self._ctx_keys = [f"ctx_{n}" for n in self.ctx_feature_names]
        self._stat_keys = [f"stat_{n}" for n in self.stat_feature_names]

        total_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"PEPNet 模型创建: {total_params / 1e6:.2f}M 参数, "
            f"embed_dim={self.embed_dim}, hidden={self.hidden_units}, "
            f"gamma={self.gamma}, domain_dim={self.domain_dim}, "
            f"user_ad_dim={self.user_ad_dim}"
        )

    def _get_field_embedding(self, batch: dict[str, torch.Tensor],
                             key: str) -> torch.Tensor:
        """获取单个特征域的 embedding (B, embed_dim)。"""
        if key in batch:
            emb = self.embeddings[key](batch[key].long())
            if emb.dim() == 3:
                emb = emb.mean(dim=1)
            return emb
        B = next(iter(batch.values())).size(0)
        device = next(iter(batch.values())).device
        return torch.zeros(B, self.embed_dim, device=device)

    def _get_behavior_embedding(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """行为序列 embedding (B, embed_dim)。"""
        behavior_seq = batch.get(
            "behavior_seq",
            torch.zeros(
                next(iter(batch.values())).size(0), 1,
                dtype=torch.long,
                device=next(iter(batch.values())).device,
            ),
        )
        beh_emb = self.behavior_embedding(behavior_seq.long())
        mask = (behavior_seq != 0).unsqueeze(-1).float()
        return (beh_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

    def forward(self, batch: dict[str, torch.Tensor],
                target_sids: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """
        Args:
            batch: 特征字典 (与 HunyuanDSPModel/DeepFMModel 相同的 key 命名)
            target_sids: 忽略 (PEPNet 不支持生成式检索)

        Returns:
            包含 ctr_logit, ctr_prob, cvr_logit, cvr_prob, user_repr 的字典
        """
        # ── 1) 提取各域 embedding ──

        # 用户特征 + 行为序列
        user_embeds = [self._get_field_embedding(batch, k) for k in self._user_keys]
        user_embeds.append(self._get_behavior_embedding(batch))
        user_concat = torch.cat(user_embeds, dim=-1)  # (B, user_dim)

        # 广告特征
        ad_embeds = [self._get_field_embedding(batch, k) for k in self._ad_keys]
        ad_concat = torch.cat(ad_embeds, dim=-1)  # (B, ad_dim)

        # 上下文特征 (域特征)
        ctx_embeds = [self._get_field_embedding(batch, k) for k in self._ctx_keys]
        ctx_concat = torch.cat(ctx_embeds, dim=-1)  # (B, ctx_dim)

        # 统计特征 (也作为域特征)
        if self._stat_keys:
            stat_embeds = [self._get_field_embedding(batch, k) for k in self._stat_keys]
            stat_concat = torch.cat(stat_embeds, dim=-1)
            domain_concat = torch.cat([ctx_concat, stat_concat], dim=-1)  # (B, domain_dim)
        else:
            domain_concat = ctx_concat

        # ── 2) EPNet: 域特征个性化 ──
        ep_emb = self.epnet(domain_concat, domain_concat)  # (B, domain_dim)

        # ── 3) PPNet: 参数个性化多任务 ──
        user_ad_concat = torch.cat([user_concat, ad_concat], dim=-1)  # (B, user_ad_dim)
        ppnet_input = torch.cat([ep_emb, user_ad_concat], dim=-1)  # (B, ppnet_input_dim)
        persona = ep_emb  # 用域个性化特征作为 persona

        task_outputs = self.ppnet(ppnet_input, persona)  # [ctr_repr, cvr_repr]

        # ── 4) 预测头 ──
        ctr_logit = self.ctr_head(task_outputs[0]).squeeze(-1)  # (B,)
        cvr_logit = self.cvr_head(task_outputs[1]).squeeze(-1)  # (B,)

        result = {
            "ctr_logit": ctr_logit,
            "ctr_prob": torch.sigmoid(ctr_logit),
            "cvr_logit": cvr_logit,
            "cvr_prob": torch.sigmoid(cvr_logit),
            "user_repr": task_outputs[0],  # CTR tower 输出作为用户表示
        }

        # 生成式检索占位
        if target_sids is not None:
            B, L = target_sids.shape
            result["retrieval_logits"] = torch.zeros(
                B, L, self.vocab_size, device=ctr_logit.device
            )

        return result

    def get_user_representation(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """兼容接口: 获取用户表示。"""
        output = self.forward(batch)
        return output["user_repr"]
