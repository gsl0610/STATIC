"""
混元大模型适配层 - 替换自定义 Transformer

使用腾讯混元开源 LLM (≈1B 参数) 作为生成式检索的 backbone，
同时保留 CTR/CVR 多任务预估头和 STATIC 约束解码能力。

架构变更：
  旧: 自定义 Transformer Encoder/Decoder (百万级参数)
  新: Hunyuan LLM Decoder-Only (十亿级参数) + LoRA 微调

核心设计：
  1. 特征序列化: 将用户/广告/上下文特征编码为 token 序列输入 LLM
  2. LLM backbone: 混元 Decoder-Only 模型提供深层语义表示
  3. 多任务头: 基于 LLM hidden states 的 CTR/CVR/检索头
  4. STATIC 兼容: 检索头输出 logits → STATIC 约束掩码 → beam search

支持的模型 (通过 config 指定 model_name_or_path):
  - tencent/Hunyuan-A1.5B-Instruct  (推荐，≈1.8B 参数)
  - 任何 HuggingFace AutoModelForCausalLM 兼容的模型
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

logger = logging.getLogger(__name__)


# ================================================================
# 特征投影层: 将离散特征 embedding 投影到 LLM 的 hidden 空间
# ================================================================
class FeatureProjector(nn.Module):
    """将多组类别特征投影为 LLM 能理解的隐层向量序列。

    每个特征字段 → Embedding(16d) → 拼接 → 线性投影到 LLM hidden_size。
    这样每组特征（用户/广告/上下文）产生 1 个 token 级别的表示。
    """

    def __init__(self, feature_vocab_sizes: dict[str, int],
                 embed_dim: int, hidden_size: int):
        super().__init__()
        self.feature_names = sorted(feature_vocab_sizes.keys())
        self.embed_dim = embed_dim
        self.embeddings = nn.ModuleDict()
        for name in self.feature_names:
            self.embeddings[name] = nn.Embedding(
                feature_vocab_sizes[name], embed_dim, padding_idx=0
            )
        total_dim = len(self.feature_names) * embed_dim
        self.projection = nn.Sequential(
            nn.Linear(total_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
        )

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: {feature_name: tensor (B,) or (B, multi_len)}
        Returns:
            projected: shape (B, hidden_size) — 1 个 token 表示
        """
        embeds = []
        # 确定 batch size
        B = next(iter(features.values())).size(0) if features else 1
        device = next(iter(features.values())).device if features else "cpu"
        for name in self.feature_names:
            if name not in features:
                # 缺失特征用零向量填充，保证维度一致
                embeds.append(torch.zeros(B, self.embed_dim, device=device))
                continue
            emb = self.embeddings[name](features[name].long())
            if emb.dim() == 3:
                emb = emb.mean(dim=1)
            embeds.append(emb)
        concat = torch.cat(embeds, dim=-1)
        return self.projection(concat)


class BehaviorProjector(nn.Module):
    """将用户行为序列投影到 LLM hidden 空间。

    行为序列中每个广告 ID → embedding → 序列 token。
    """

    def __init__(self, behavior_vocab_size: int, hidden_size: int,
                 max_seq_len: int = 50):
        super().__init__()
        self.token_embedding = nn.Embedding(
            behavior_vocab_size + 1, hidden_size, padding_idx=0
        )
        self.max_seq_len = max_seq_len

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            seq: shape (B, seq_len)
        Returns:
            shape (B, seq_len, hidden_size)
        """
        return self.token_embedding(seq.long())


# ================================================================
# 多任务头
# ================================================================
class CTRHead(nn.Module):
    """CTR 预估头: LLM hidden → P(click)"""

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.head(hidden).squeeze(-1)


class CVRHead(nn.Module):
    """CVR 预估头: LLM hidden → P(convert)"""

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.head(hidden).squeeze(-1)


class RetrievalHead(nn.Module):
    """生成式检索头: 基于 LLM hidden state 自回归生成语义ID。

    不再使用独立的 Transformer Decoder，而是复用 LLM 自身的解码能力，
    只需一个 SID token embedding + 输出投影层。
    """

    def __init__(self, hidden_size: int, sid_vocab_size: int,
                 sid_length: int, dropout: float = 0.1):
        super().__init__()
        self.sid_vocab_size = sid_vocab_size
        self.sid_length = sid_length

        # SID token embedding (包含 BOS token, index = sid_vocab_size)
        self.sid_embedding = nn.Embedding(sid_vocab_size + 1, hidden_size)

        # 输出投影: hidden → SID logits
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, sid_vocab_size),
        )

    def forward(self, hidden_states: torch.Tensor,
                target_sids: torch.Tensor | None = None) -> torch.Tensor:
        """训练时用 teacher forcing。

        Args:
            hidden_states: LLM 输出的 hidden, shape (B, seq_len, H)
                           取最后一个有效位置作为 user representation
            target_sids: shape (B, L)

        Returns:
            logits: shape (B, L, sid_vocab_size)
        """
        # 使用序列最后一个位置的 hidden state 作为 memory
        # 这里 hidden_states 已经是 user_repr, shape (B, H)
        if hidden_states.dim() == 3:
            user_repr = hidden_states[:, -1, :]  # (B, H)
        else:
            user_repr = hidden_states  # (B, H)

        B = user_repr.size(0)
        device = user_repr.device

        if target_sids is not None:
            # Teacher forcing: [BOS, sid_0, ..., sid_{L-2}] → predict [sid_0, ..., sid_{L-1}]
            bos = torch.full((B, 1), self.sid_vocab_size, dtype=torch.long, device=device)
            decoder_input = torch.cat([bos, target_sids[:, :-1]], dim=1)  # (B, L)
            input_emb = self.sid_embedding(decoder_input)  # (B, L, H)

            # 加上 user context
            input_emb = input_emb + user_repr.unsqueeze(1)

            logits = self.output_proj(input_emb)  # (B, L, V)
            return logits
        else:
            # 推理模式: 返回初始 BOS 的 logits
            bos = torch.full((B, 1), self.sid_vocab_size, dtype=torch.long, device=device)
            input_emb = self.sid_embedding(bos) + user_repr.unsqueeze(1)
            logits = self.output_proj(input_emb)
            return logits

    def get_next_token_logits(self, user_repr: torch.Tensor,
                              partial_sids: torch.Tensor) -> torch.Tensor:
        """推理时获取下一个 SID token 的 logits。

        Args:
            user_repr: shape (B, H)
            partial_sids: 已生成的部分序列, shape (B, t)

        Returns:
            logits: shape (B, 1, sid_vocab_size)
        """
        B = user_repr.size(0)
        device = user_repr.device

        bos = torch.full((B, 1), self.sid_vocab_size, dtype=torch.long, device=device)
        if partial_sids.size(1) > 0:
            decoder_input = torch.cat([bos, partial_sids], dim=1)
        else:
            decoder_input = bos

        input_emb = self.sid_embedding(decoder_input) + user_repr.unsqueeze(1)
        all_logits = self.output_proj(input_emb)
        return all_logits[:, -1:, :]  # 只取最后一步


# ================================================================
# 主模型: Hunyuan LLM + 多任务头
# ================================================================
class HunyuanDSPModel(nn.Module):
    """基于混元 LLM 的 DSP 广告多任务模型。

    架构:
        [用户特征token | 广告特征token | 上下文特征token | 行为序列tokens]
          → Hunyuan LLM (LoRA 微调)
          → 最后一层 hidden states
          → CTR头 / CVR头 / 生成式检索头

    与原始 DSPMultiTaskModel 相比：
    - backbone 从 ~1M 参数升级到 ~1.8B 参数
    - 特征交互能力和语义理解能力大幅提升
    - 支持 LoRA 低秩微调，训练效率可控
    """

    def __init__(self, config: dict[str, Any],
                 user_feature_vocab_sizes: dict[str, int],
                 ad_feature_vocab_sizes: dict[str, int],
                 context_feature_vocab_sizes: dict[str, int],
                 stat_feature_vocab_sizes: dict[str, int] | None = None,
                 behavior_vocab_size: int = 10000):
        super().__init__()

        llm_cfg = config["llm"]
        model_cfg = config["model"]
        self.model_name_or_path = llm_cfg["model_name_or_path"]
        self.use_lora = llm_cfg.get("use_lora", True)
        self.freeze_backbone = llm_cfg.get("freeze_backbone", True)

        # ---------- 加载 LLM backbone ----------
        logger.info(f"加载混元 LLM: {self.model_name_or_path}")

        # 先获取模型配置确定 hidden_size
        try:
            llm_config = AutoConfig.from_pretrained(
                self.model_name_or_path,
                trust_remote_code=True,
            )
            self.hidden_size = llm_config.hidden_size
            logger.info(f"LLM hidden_size: {self.hidden_size}")
        except Exception:
            try:
                llm_config = AutoConfig.from_pretrained(
                    self.model_name_or_path,
                    trust_remote_code=True,
                    local_files_only=True,
                )
                self.hidden_size = llm_config.hidden_size
                logger.info(f"LLM hidden_size (本地缓存): {self.hidden_size}")
            except Exception as e:
                logger.warning(f"无法加载配置，使用默认 hidden_size: {e}")
                self.hidden_size = llm_cfg.get("hidden_size", 2048)

        # 加载预训练模型（优先在线，失败则尝试本地缓存，最后 fallback）
        try:
            self.llm = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if llm_cfg.get("bf16", True) else torch.float32,
                attn_implementation=llm_cfg.get("attn_implementation", "sdpa"),
            )
            logger.info(f"LLM 加载成功: {sum(p.numel() for p in self.llm.parameters()) / 1e9:.2f}B 参数")
            self._is_fallback = False
        except Exception as e:
            logger.warning(f"在线加载失败 ({e})，尝试从本地缓存加载...")
            try:
                self.llm = AutoModelForCausalLM.from_pretrained(
                    self.model_name_or_path,
                    trust_remote_code=True,
                    local_files_only=True,
                    torch_dtype=torch.bfloat16 if llm_cfg.get("bf16", True) else torch.float32,
                    attn_implementation=llm_cfg.get("attn_implementation", "sdpa"),
                )
                logger.info(f"LLM 从本地缓存加载成功: {sum(p.numel() for p in self.llm.parameters()) / 1e9:.2f}B 参数")
                self._is_fallback = False
            except Exception as e2:
                logger.warning(f"本地缓存也失败，创建 fallback 模型: {e2}")
                self.llm = self._create_fallback_llm(self.hidden_size, model_cfg)
                self._is_fallback = True

        # 冻结 backbone
        if self.freeze_backbone:
            for param in self.llm.parameters():
                param.requires_grad = False
            logger.info("LLM backbone 已冻结")

        # LoRA 适配（如果安装了 peft）
        if self.use_lora and self.freeze_backbone:
            self._apply_lora(llm_cfg)

        # ---------- 特征投影层 ----------
        feat_embed_dim = 16
        self.user_projector = FeatureProjector(
            user_feature_vocab_sizes, feat_embed_dim, self.hidden_size
        )
        self.ad_projector = FeatureProjector(
            ad_feature_vocab_sizes, feat_embed_dim, self.hidden_size
        )
        self.context_projector = FeatureProjector(
            context_feature_vocab_sizes, feat_embed_dim, self.hidden_size
        )
        self.stat_feature_vocab_sizes = stat_feature_vocab_sizes or {}
        if self.stat_feature_vocab_sizes:
            self.stat_projector = FeatureProjector(
                self.stat_feature_vocab_sizes, feat_embed_dim, self.hidden_size
            )
        else:
            self.stat_projector = None
        self.behavior_projector = BehaviorProjector(
            behavior_vocab_size, self.hidden_size,
            max_seq_len=config["data"]["behavior"]["max_seq_len"],
        )

        # ---------- 多任务头 ----------
        dropout = model_cfg.get("dropout", 0.1)
        sid_vocab_size = config["static_index"]["vocab_size"]
        sid_length = config["static_index"]["sid_length"]

        self.ctr_head = CTRHead(self.hidden_size, dropout)
        self.cvr_head = CVRHead(self.hidden_size, dropout)
        self.retrieval_head = RetrievalHead(
            self.hidden_size, sid_vocab_size, sid_length, dropout
        )

        # ---------- 序列类型标记 ----------
        # [USER_TOKEN, AD_TOKEN, CTX_TOKEN, BEH_START, BEH_END, SEP, CLS, STAT_TOKEN]
        self.special_tokens = nn.Embedding(8, self.hidden_size)

        # 保存配置
        self.vocab_size = sid_vocab_size

    def _create_fallback_llm(self, hidden_size: int, model_cfg: dict) -> nn.Module:
        """当无法加载预训练模型时，创建一个轻量级替代模型。"""
        logger.info(f"创建 fallback Transformer 模型: hidden={hidden_size}")

        class FallbackLLM(nn.Module):
            def __init__(self, hidden_size: int, num_layers: int = 6,
                         num_heads: int = 8, dropout: float = 0.1):
                super().__init__()
                self.config = type("Config", (), {"hidden_size": hidden_size})()
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=hidden_size, nhead=num_heads,
                    dim_feedforward=hidden_size * 4, dropout=dropout,
                    activation="gelu", batch_first=True, norm_first=True,
                )
                self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                self.final_norm = nn.LayerNorm(hidden_size)

            def forward(self, inputs_embeds=None, attention_mask=None,
                        output_hidden_states=False, **kwargs):
                h = self.encoder(inputs_embeds)
                h = self.final_norm(h)
                return type("Output", (), {
                    "last_hidden_state": h,
                    "hidden_states": (h,) if output_hidden_states else None,
                })()

        num_layers = model_cfg.get("num_layers", 6)
        num_heads = model_cfg.get("num_heads", 8)
        return FallbackLLM(hidden_size, num_layers, num_heads)

    def _apply_lora(self, llm_cfg: dict):
        """应用 LoRA 低秩适配器。"""
        if getattr(self, "_is_fallback", False):
            logger.info("Fallback 模型不支持 LoRA，跳过 LoRA 应用，直接解冻所有参数")
            for param in self.llm.parameters():
                param.requires_grad = True
            return

        try:
            from peft import LoraConfig, get_peft_model, TaskType

            lora_rank = llm_cfg.get("lora_rank", 16)
            lora_alpha = llm_cfg.get("lora_alpha", 32)
            lora_dropout = llm_cfg.get("lora_dropout", 0.05)
            target_modules = llm_cfg.get("lora_target_modules", [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ])

            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                bias="none",
            )
            self.llm = get_peft_model(self.llm, lora_config)
            trainable = sum(p.numel() for p in self.llm.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.llm.parameters())
            logger.info(
                f"LoRA 已应用: rank={lora_rank}, alpha={lora_alpha}, "
                f"可训练参数: {trainable / 1e6:.1f}M / {total / 1e6:.1f}M "
                f"({100 * trainable / total:.2f}%)"
            )
        except ImportError:
            logger.warning(
                "peft 未安装，跳过 LoRA。安装方式: pip install peft\n"
                "将直接微调所有参数（不推荐用于大模型）"
            )
            for param in self.llm.parameters():
                param.requires_grad = True

    def _build_input_sequence(self, batch: dict[str, torch.Tensor],
                              for_retrieval: bool = False) -> torch.Tensor:
        """将多组特征编码为 LLM 输入的 embedding 序列。

        序列结构:
            [USER_TOKEN] user_feat [AD_TOKEN] ad_feat [CTX_TOKEN] ctx_feat
            [BEH_START] beh_1 ... beh_n [BEH_END] [CLS]

        训练时启用特征随机 Mask (数据增强，防止过拟合):
            - 随机将部分特征 embedding 置零
            - 随机 mask 部分行为序列 token

        Args:
            batch: 特征字典
            for_retrieval: 是否为检索模式（广告侧用零向量）

        Returns:
            input_embeds: shape (B, seq_len, hidden_size)
        """
        device = next(self.parameters()).device

        # 提取各类特征 (使用 removeprefix 避免替换中间出现的同名子串)
        user_feats = {k.removeprefix("user_"): v for k, v in batch.items()
                      if k.startswith("user_")}
        ad_feats = {k.removeprefix("ad_"): v for k, v in batch.items()
                    if k.startswith("ad_")}
        ctx_feats = {k.removeprefix("ctx_"): v for k, v in batch.items()
                     if k.startswith("ctx_")}
        stat_feats = {k.removeprefix("stat_"): v for k, v in batch.items()
                      if k.startswith("stat_")}

        B = next(iter(user_feats.values())).size(0)

        # 投影特征
        user_emb = self.user_projector(user_feats)  # (B, H)
        if for_retrieval:
            ad_emb = torch.zeros(B, self.hidden_size, device=device)
        else:
            ad_emb = self.ad_projector(ad_feats)  # (B, H)
        ctx_emb = self.context_projector(ctx_feats)  # (B, H)

        # 统计特征投影
        if self.stat_projector is not None and stat_feats:
            stat_emb = self.stat_projector(stat_feats)  # (B, H)
        else:
            stat_emb = None

        # 训练时特征随机 Mask (数据增强)
        if self.training:
            feat_mask_rate = 0.15  # 15% 概率 mask 掉整个特征组
            if torch.rand(1).item() < feat_mask_rate:
                user_emb = torch.zeros_like(user_emb)
            if torch.rand(1).item() < feat_mask_rate:
                ad_emb = torch.zeros_like(ad_emb)
            if torch.rand(1).item() < feat_mask_rate:
                ctx_emb = torch.zeros_like(ctx_emb)
            if stat_emb is not None and torch.rand(1).item() < feat_mask_rate:
                stat_emb = torch.zeros_like(stat_emb)

        # 行为序列
        behavior_seq = batch.get(
            "behavior_seq",
            torch.zeros(B, 1, dtype=torch.long, device=device)
        )
        # 训练时随机 mask 部分行为 token (20% 概率置零)
        if self.training:
            beh_mask = torch.rand_like(behavior_seq.float()) < 0.2
            behavior_seq = behavior_seq.clone()
            behavior_seq[beh_mask] = 0
        beh_emb = self.behavior_projector(behavior_seq)  # (B, seq_len, H)

        # 特殊 token embeddings
        token_ids = torch.arange(8, device=device)
        special_embs = self.special_tokens(token_ids)  # (8, H)
        USER_TOK = special_embs[0].unsqueeze(0).expand(B, -1)  # (B, H)
        AD_TOK = special_embs[1].unsqueeze(0).expand(B, -1)
        CTX_TOK = special_embs[2].unsqueeze(0).expand(B, -1)
        BEH_START = special_embs[3].unsqueeze(0).expand(B, -1)
        BEH_END = special_embs[4].unsqueeze(0).expand(B, -1)
        CLS_TOK = special_embs[6].unsqueeze(0).expand(B, -1)
        STAT_TOK = special_embs[7].unsqueeze(0).expand(B, -1)

        # 拼接序列: [USER] user [AD] ad [CTX] ctx [STAT] stat [BEH_S] beh... [BEH_E] [CLS]
        prefix_parts = [USER_TOK, user_emb, AD_TOK, ad_emb, CTX_TOK, ctx_emb]
        if stat_emb is not None:
            prefix_parts.extend([STAT_TOK, stat_emb])
        sequence = torch.stack(prefix_parts, dim=1)  # (B, 6 or 8, H)

        beh_start = BEH_START.unsqueeze(1)  # (B, 1, H)
        beh_end = BEH_END.unsqueeze(1)
        cls = CLS_TOK.unsqueeze(1)

        input_embeds = torch.cat([
            sequence, beh_start, beh_emb, beh_end, cls
        ], dim=1)  # (B, 6+1+seq_len+1+1, H)

        return input_embeds

    def _run_llm(self, input_embeds: torch.Tensor) -> torch.Tensor:
        """通过 LLM backbone 获取 hidden states。

        Args:
            input_embeds: shape (B, seq_len, hidden_size)

        Returns:
            hidden_states: shape (B, seq_len, hidden_size)
        """
        # 转换为 LLM 期望的 dtype
        if hasattr(self.llm, 'dtype'):
            input_embeds = input_embeds.to(self.llm.dtype)
        elif hasattr(self.llm, 'config') and hasattr(self.llm.config, 'torch_dtype'):
            pass  # 保持当前 dtype

        outputs = self.llm(
            inputs_embeds=input_embeds,
            output_hidden_states=True,
            use_cache=False,
        )

        if hasattr(outputs, 'last_hidden_state'):
            hidden = outputs.last_hidden_state
        elif hasattr(outputs, 'hidden_states') and outputs.hidden_states:
            hidden = outputs.hidden_states[-1]
        else:
            hidden = outputs[0] if isinstance(outputs, tuple) else outputs

        # 转 FP32 以保证 CTR/CVR Head 中 LayerNorm 类型一致
        return hidden.float()

    def forward(self, batch: dict[str, torch.Tensor],
                target_sids: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """
        Args:
            batch: 包含各类特征的字典
            target_sids: 目标广告语义ID, shape (B, L)

        Returns:
            包含各任务输出的字典
        """
        batch = {k: v.to(next(self.parameters()).device)
                 if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        # 构建输入序列
        input_embeds = self._build_input_sequence(batch, for_retrieval=False)

        # 通过 LLM
        hidden_states = self._run_llm(input_embeds)  # (B, seq_len, H)

        # 取 [CLS] 位置的 hidden state（最后一个位置）
        cls_hidden = hidden_states[:, -1, :]  # (B, H)

        # 多任务输出
        ctr_logit = self.ctr_head(cls_hidden)
        cvr_logit = self.cvr_head(cls_hidden)

        result = {
            "ctr_logit": ctr_logit,
            "ctr_prob": torch.sigmoid(ctr_logit),
            "cvr_logit": cvr_logit,
            "cvr_prob": torch.sigmoid(cvr_logit),
            "user_repr": cls_hidden,
        }

        # 生成式检索
        if target_sids is not None:
            retrieval_logits = self.retrieval_head(cls_hidden, target_sids)
            result["retrieval_logits"] = retrieval_logits

        return result

    def get_user_representation(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """仅计算用户表示，用于推理时的检索。"""
        batch = {k: v.to(next(self.parameters()).device)
                 if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        input_embeds = self._build_input_sequence(batch, for_retrieval=True)
        hidden_states = self._run_llm(input_embeds)
        return hidden_states[:, -1, :]  # (B, H)

    @torch.inference_mode()
    def merged_forward(self, batch: dict[str, torch.Tensor]
                       ) -> dict[str, torch.Tensor]:
        """合并前向推理 — 单次 LLM 调用同时产出 CTR/CVR/user_repr。

        消除 process_batch() 中 get_user_representation() + forward() 的重复计算,
        延迟降低约 50%。
        """
        batch = {k: v.to(next(self.parameters()).device)
                 if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        input_embeds = self._build_input_sequence(batch, for_retrieval=False)
        hidden_states = self._run_llm(input_embeds)
        cls_hidden = hidden_states[:, -1, :]

        ctr_logit = self.ctr_head(cls_hidden)
        cvr_logit = self.cvr_head(cls_hidden)

        return {
            "ctr_logit": ctr_logit,
            "ctr_prob": torch.sigmoid(ctr_logit),
            "cvr_logit": cvr_logit,
            "cvr_prob": torch.sigmoid(cvr_logit),
            "user_repr": cls_hidden,
        }


def create_hunyuan_model(config: dict,
                         user_feature_vocab_sizes: dict[str, int],
                         ad_feature_vocab_sizes: dict[str, int],
                         context_feature_vocab_sizes: dict[str, int],
                         stat_feature_vocab_sizes: dict[str, int] | None = None,
                         behavior_vocab_size: int = 10000) -> HunyuanDSPModel:
    """工厂函数: 创建混元 DSP 模型。"""
    model = HunyuanDSPModel(
        config=config,
        user_feature_vocab_sizes=user_feature_vocab_sizes,
        ad_feature_vocab_sizes=ad_feature_vocab_sizes,
        context_feature_vocab_sizes=context_feature_vocab_sizes,
        stat_feature_vocab_sizes=stat_feature_vocab_sizes,
        behavior_vocab_size=behavior_vocab_size,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"HunyuanDSPModel 创建完成: "
        f"总参数 {total_params / 1e9:.2f}B, "
        f"可训练 {trainable_params / 1e6:.1f}M"
    )
    return model
