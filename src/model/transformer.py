"""
生成式检索模型 - 多任务 Transformer

模型结构：
  用户特征 + 上下文特征 + 行为序列 → Transformer Encoder → 多任务头
    ├─ CTR 头 (二分类)
    ├─ CVR 头 (二分类)
    └─ 生成式检索头 (自回归生成广告语义ID)

生成式检索头在推理时配合 STATIC 约束解码，确保生成的语义ID对应有效广告。
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """正弦位置编码。"""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class FeatureEmbedding(nn.Module):
    """统一的特征 embedding 层。

    将多个类别特征各自映射为固定维度的 embedding，然后拼接。
    """

    def __init__(self, feature_vocab_sizes: dict[str, int], embed_dim: int = 16):
        super().__init__()
        self.embeddings = nn.ModuleDict()
        self.feature_names = sorted(feature_vocab_sizes.keys())
        for name in self.feature_names:
            vocab_size = feature_vocab_sizes[name]
            self.embeddings[name] = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.output_dim = len(self.feature_names) * embed_dim

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: {feature_name: tensor of shape (B,) or (B, multi_len)}
        Returns:
            拼接后的 embedding, shape (B, output_dim)
        """
        embeds = []
        for name in self.feature_names:
            if name not in features:
                continue
            x = features[name]
            emb = self.embeddings[name](x.long())
            if emb.dim() == 3:
                # 多值特征: 取均值
                emb = emb.mean(dim=1)
            embeds.append(emb)
        return torch.cat(embeds, dim=-1)


class BehaviorEncoder(nn.Module):
    """用户行为序列编码器。

    使用 Transformer Encoder 编码用户历史点击广告序列。
    """

    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int,
                 num_layers: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        self.pos_encoding = PositionalEncoding(embed_dim, max_seq_len)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_dim = embed_dim

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            seq: 行为序列, shape (B, seq_len)
        Returns:
            序列表示, shape (B, embed_dim)
        """
        padding_mask = (seq == 0)
        x = self.token_embedding(seq.long())
        x = self.pos_encoding(x)
        x = self.dropout(x)
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        # 取非 padding 位置的平均
        mask = (~padding_mask).unsqueeze(-1).float()
        x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return x


class GenerativeRetrievalHead(nn.Module):
    """生成式检索头 - 自回归生成广告语义ID。

    给定用户表示，逐步生成长度为 L 的语义ID序列。
    训练时使用 teacher forcing，推理时配合 STATIC beam search。
    """

    def __init__(self, input_dim: int, embed_dim: int, vocab_size: int,
                 sid_length: int, num_heads: int = 4, num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.sid_length = sid_length

        # 输入投影
        self.input_proj = nn.Linear(input_dim, embed_dim)

        # Token embedding for SID tokens
        self.token_embedding = nn.Embedding(vocab_size + 1, embed_dim)  # +1 for BOS
        self.pos_encoding = PositionalEncoding(embed_dim, sid_length + 1)

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # 输出头
        self.output_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, user_repr: torch.Tensor,
                target_sids: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            user_repr: 用户综合表示, shape (B, input_dim)
            target_sids: 目标语义ID (teacher forcing), shape (B, L), 仅训练时提供

        Returns:
            logits: shape (B, L, vocab_size)
        """
        B = user_repr.size(0)
        device = user_repr.device

        # 用户表示作为 memory
        memory = self.input_proj(user_repr).unsqueeze(1)  # (B, 1, embed_dim)

        if target_sids is not None:
            # Teacher forcing: 用 BOS + target[:-1] 作为输入
            bos = torch.full((B, 1), self.vocab_size, dtype=torch.long, device=device)
            decoder_input = torch.cat([bos, target_sids[:, :-1]], dim=1)  # (B, L)
        else:
            # 推理时只需 BOS (实际由 STATIC beam search 接管)
            decoder_input = torch.full((B, 1), self.vocab_size, dtype=torch.long, device=device)

        tgt = self.token_embedding(decoder_input)
        tgt = self.pos_encoding(tgt)

        # Causal mask
        seq_len = tgt.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=device)

        output = self.decoder(tgt, memory, tgt_mask=causal_mask)
        logits = self.output_head(output)  # (B, L, vocab_size)

        return logits

    def get_next_token_logits(self, user_repr: torch.Tensor,
                              partial_sids: torch.Tensor) -> torch.Tensor:
        """推理时获取下一个 token 的 logits。

        Args:
            user_repr: shape (B, input_dim)
            partial_sids: 已生成的部分序列, shape (B, t), t = 当前步数

        Returns:
            logits: shape (B, 1, vocab_size)
        """
        B = user_repr.size(0)
        device = user_repr.device

        memory = self.input_proj(user_repr).unsqueeze(1)

        bos = torch.full((B, 1), self.vocab_size, dtype=torch.long, device=device)
        if partial_sids.size(1) > 0:
            decoder_input = torch.cat([bos, partial_sids], dim=1)
        else:
            decoder_input = bos

        tgt = self.token_embedding(decoder_input)
        tgt = self.pos_encoding(tgt)

        seq_len = tgt.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=device)

        output = self.decoder(tgt, memory, tgt_mask=causal_mask)
        logits = self.output_head(output[:, -1:, :])  # 只取最后一步

        return logits


class DSPMultiTaskModel(nn.Module):
    """DSP 广告多任务模型。

    联合训练三个任务：
    1. CTR 预估 (点击率)
    2. CVR 预估 (转化率)
    3. 生成式检索 (自回归生成广告语义ID)
    """

    def __init__(self, config: dict[str, Any],
                 user_feature_vocab_sizes: dict[str, int],
                 ad_feature_vocab_sizes: dict[str, int],
                 context_feature_vocab_sizes: dict[str, int],
                 behavior_vocab_size: int = 10000):
        super().__init__()

        model_cfg = config["model"]
        embed_dim = model_cfg["embed_dim"]
        vocab_size = model_cfg["vocab_size"]

        # 特征 Embedding
        self.user_embedding = FeatureEmbedding(user_feature_vocab_sizes, embed_dim=16)
        self.ad_embedding = FeatureEmbedding(ad_feature_vocab_sizes, embed_dim=16)
        self.context_embedding = FeatureEmbedding(context_feature_vocab_sizes, embed_dim=16)

        # 行为序列编码
        self.behavior_encoder = BehaviorEncoder(
            vocab_size=behavior_vocab_size,
            embed_dim=embed_dim // 2,
            num_heads=4,
            num_layers=2,
            max_seq_len=config["data"]["behavior"]["max_seq_len"],
            dropout=model_cfg["dropout"],
        )

        # 总特征维度
        total_dim = (
            self.user_embedding.output_dim
            + self.ad_embedding.output_dim
            + self.context_embedding.output_dim
            + self.behavior_encoder.output_dim
        )

        # 共享特征融合层
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(model_cfg["dropout"]),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

        # --- 任务头 ---
        # CTR 头
        self.ctr_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(model_cfg["dropout"]),
            nn.Linear(embed_dim // 2, 1),
        )

        # CVR 头
        self.cvr_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(model_cfg["dropout"]),
            nn.Linear(embed_dim // 2, 1),
        )

        # 生成式检索头
        sid_length = config["static_index"]["sid_length"]
        self.retrieval_head = GenerativeRetrievalHead(
            input_dim=embed_dim,
            embed_dim=embed_dim,
            vocab_size=vocab_size,
            sid_length=sid_length,
            num_heads=model_cfg["num_heads"],
            num_layers=2,
            dropout=model_cfg["dropout"],
        )

        # 保存配置
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size

    def forward(self, batch: dict[str, torch.Tensor],
                target_sids: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """
        Args:
            batch: 包含各类特征的字典
            target_sids: 目标广告语义ID, shape (B, L), 训练时提供

        Returns:
            包含各任务输出的字典
        """
        # 提取各类特征 (使用 removeprefix 避免替换中间出现的同名子串)
        user_feats = {k.removeprefix("user_"): v for k, v in batch.items() if k.startswith("user_")}
        ad_feats = {k.removeprefix("ad_"): v for k, v in batch.items() if k.startswith("ad_")}
        ctx_feats = {k.removeprefix("ctx_"): v for k, v in batch.items() if k.startswith("ctx_")}

        # Embedding
        user_emb = self.user_embedding(user_feats)
        ad_emb = self.ad_embedding(ad_feats)
        ctx_emb = self.context_embedding(ctx_feats)

        # 行为序列编码
        behavior_emb = self.behavior_encoder(batch.get("behavior_seq", torch.zeros(user_emb.size(0), 1).long().to(user_emb.device)))

        # 特征融合
        fused = self.fusion(torch.cat([user_emb, ad_emb, ctx_emb, behavior_emb], dim=-1))

        # 多任务输出
        ctr_logit = self.ctr_head(fused).squeeze(-1)
        cvr_logit = self.cvr_head(fused).squeeze(-1)

        result = {
            "ctr_logit": ctr_logit,       # shape (B,)
            "ctr_prob": torch.sigmoid(ctr_logit),
            "cvr_logit": cvr_logit,        # shape (B,)
            "cvr_prob": torch.sigmoid(cvr_logit),
            "user_repr": fused,             # shape (B, embed_dim), 供检索头使用
        }

        # 生成式检索
        if target_sids is not None:
            retrieval_logits = self.retrieval_head(fused, target_sids)
            result["retrieval_logits"] = retrieval_logits  # shape (B, L, V)

        return result

    def get_user_representation(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """仅计算用户表示，用于推理时的检索。"""
        user_feats = {k.removeprefix("user_"): v for k, v in batch.items() if k.startswith("user_")}
        ctx_feats = {k.removeprefix("ctx_"): v for k, v in batch.items() if k.startswith("ctx_")}

        user_emb = self.user_embedding(user_feats)
        ctx_emb = self.context_embedding(ctx_feats)
        behavior_emb = self.behavior_encoder(batch.get("behavior_seq", torch.zeros(user_emb.size(0), 1).long().to(user_emb.device)))

        # 广告侧使用零向量（检索时不知道目标广告）
        ad_zero = torch.zeros(user_emb.size(0), self.ad_embedding.output_dim, device=user_emb.device)

        fused = self.fusion(torch.cat([user_emb, ad_zero, ctx_emb, behavior_emb], dim=-1))
        return fused
