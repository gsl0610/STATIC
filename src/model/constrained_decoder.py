"""
STATIC 约束解码器 - PyTorch 实现

将论文中的 STATIC 约束 beam search 与 DSP 多任务模型集成，
实现在推理时只生成有效广告的语义ID。

核心流程：
  用户表示 → Retrieval Head + STATIC 约束掩码 → 有效广告语义ID → CTR/CVR 评分

兼容两种 backbone:
  - 原始 DSPMultiTaskModel (自定义小型 Transformer)
  - HunyuanDSPModel (混元 LLM backbone)

参考实现：https://github.com/youtube/static-constraint-decoding
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _gather_beams(x: torch.Tensor, beam_indices: torch.Tensor) -> torch.Tensor:
    """高效的 beam 数据收集。

    Args:
        x: shape (batch_size, old_beam_size, ...)
        beam_indices: shape (batch_size, new_beam_size)

    Returns:
        shape (batch_size, new_beam_size, ...)
    """
    batch_size, new_beam_size = beam_indices.shape
    view_shape = [batch_size, new_beam_size] + [1] * (x.dim() - 2)
    expand_shape = [batch_size, new_beam_size] + list(x.shape[2:])
    indices = beam_indices.view(view_shape).expand(expand_shape)
    return x.gather(1, indices)


@torch.inference_mode()
def generate_and_apply_logprobs_mask(
    flat_logprobs: torch.Tensor,
    flat_states: torch.Tensor,
    packed_csr: torch.Tensor,
    csr_indptr: torch.Tensor,
    limit: int,
    vocab_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """向量化稀疏候选提取 —— STATIC 的核心内核。

    将不规则的 Trie 指针遍历替换为单次向量化突发读取，
    实现相对于约束集大小 O(1) 的延迟。

    Args:
        flat_logprobs: 模型输出的 log 概率, shape (B*M, V)
        flat_states: 当前 Trie 状态ID, shape (B*M,)
        packed_csr: CSR 转移表 [token_id, next_state], shape (T+V, 2)
        csr_indptr: CSR 行指针, shape (S+2,)
        limit: 当前层最大分支因子 K
        vocab_size: 词表大小 V
        device: 计算设备

    Returns:
        candidate_logprobs: shape (B*M, K)
        candidate_token_ids: shape (B*M, K)
        candidate_next_states: shape (B*M, K)
    """
    # 1. 突发读取 CSR 行
    starts = csr_indptr[flat_states.long()]
    actual_lens = csr_indptr[flat_states.long() + 1] - starts

    # 构造偏移网格
    offsets = torch.arange(limit, device=device)
    gather_indices = starts.unsqueeze(1) + offsets.unsqueeze(0)

    # 安全索引
    max_idx = packed_csr.size(0) - 1
    safe_gather_indices = gather_indices.clamp(max=max_idx)

    # 从 HBM 读取 [Token, NextState] 对
    gathered_vals = packed_csr[safe_gather_indices]
    candidate_token_ids = gathered_vals[..., 0]
    candidate_next_states = gathered_vals[..., 1]

    # 2. 有效性掩码
    valid_mask = offsets.unsqueeze(0) < actual_lens.unsqueeze(1)

    # 3. 收集 log 概率
    safe_token_ids = candidate_token_ids.long().clamp(max=vocab_size - 1)
    candidate_logprobs = flat_logprobs.gather(1, safe_token_ids)

    # 无效路径置为 -inf
    candidate_logprobs = torch.where(
        valid_mask, candidate_logprobs, torch.tensor(-float("inf"), device=device)
    )

    return candidate_logprobs, candidate_token_ids, candidate_next_states


class STATICConstrainedDecoder:
    """STATIC 约束 Beam Search 解码器。

    集成 DSP 多任务模型的生成式检索头和 STATIC 索引，
    执行约束解码只生成有效广告的语义ID。

    兼容:
      - DSPMultiTaskModel (model.retrieval_head)
      - HunyuanDSPModel (model.retrieval_head)
    """

    def __init__(self, model: nn.Module, static_index_tensors: dict[str, torch.Tensor],
                 config: dict[str, Any], device: torch.device):
        """
        Args:
            model: DSPMultiTaskModel 或 HunyuanDSPModel 实例
            static_index_tensors: STATIC 索引张量字典，包含:
                - packed_csr, csr_indptr, start_mask, dense_mask, dense_states
            config: 推理配置
            device: 计算设备
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device

        infer_cfg = config["inference"]
        self.beam_size = infer_cfg["beam_size"]
        self.tokens_per_beam = infer_cfg["tokens_per_beam"]
        self.d_dense = infer_cfg["d_dense"]

        idx_cfg = config["static_index"]
        self.vocab_size = idx_cfg["vocab_size"]
        self.sid_length = idx_cfg["sid_length"]

        # 将索引张量移到设备上
        self.packed_csr = static_index_tensors["packed_csr"].to(device)
        self.csr_indptr = static_index_tensors["csr_indptr"].to(device)
        self.start_mask = static_index_tensors["start_mask"].to(device)
        self.dense_mask = static_index_tensors["dense_mask"].to(device)
        self.dense_states = static_index_tensors["dense_states"].to(device)
        self.max_branch_factors = static_index_tensors["max_branch_factors"]

    def _get_retrieval_head(self):
        """统一获取检索头，兼容两种模型及 MergedForwardWrapper。"""
        if hasattr(self.model, 'retrieval_head'):
            return self.model.retrieval_head
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'retrieval_head'):
            return self.model.model.retrieval_head
        raise AttributeError("模型缺少 retrieval_head")

    @torch.inference_mode()
    def decode(self, user_repr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """执行 STATIC 约束 Beam Search。

        Args:
            user_repr: 用户综合表示, shape (B, hidden_size)

        Returns:
            decoded_sids: 生成的语义ID, shape (B, beam_size, L)
            beam_scores: 各 beam 的得分, shape (B, beam_size)
        """
        B = user_repr.size(0)
        retrieval_head = self._get_retrieval_head()
        beam_size = self.beam_size
        tokens_per_beam = self.tokens_per_beam
        V = self.vocab_size
        L = self.sid_length

        # --- 步骤1: 初始步 (第1个 codeword) ---
        # 使用检索头获取初始 logits
        initial_logits = retrieval_head.get_next_token_logits(
            user_repr, torch.empty(B, 0, dtype=torch.long, device=self.device)
        )  # (B, 1, V)
        raw_logprobs = F.log_softmax(initial_logits[:, 0, :], dim=-1)  # (B, V)

        # 应用根掩码
        initial_logprobs = torch.where(
            self.start_mask, raw_logprobs, torch.tensor(-float("inf"), device=self.device)
        )

        top_logprobs, top_tokens = torch.topk(initial_logprobs, beam_size, dim=-1)

        # 初始化缓冲区
        token_buffer = torch.full(
            (B, beam_size, L), 0, dtype=top_tokens.dtype, device=self.device
        )
        token_buffer[:, :, 0] = top_tokens

        # Level-0 token → state ID (token T → ID T+1)
        current_states = top_tokens + 1
        current_scores = top_logprobs

        # --- 步骤2: 自回归循环 (codeword 2 到 L) ---
        for step in range(L - 1):
            # 准备输入: 已生成的部分序列
            partial_sids = token_buffer[:, :, :step + 1]  # (B, beam_size, step+1)
            flat_partial = partial_sids.reshape(B * beam_size, step + 1)

            # 扩展 user_repr 到 beam 维度
            flat_user_repr = user_repr.unsqueeze(1).expand(
                -1, beam_size, -1
            ).reshape(B * beam_size, -1)

            # 获取下一 token logits
            flat_logits = retrieval_head.get_next_token_logits(
                flat_user_repr, flat_partial
            )
            flat_logprobs = F.log_softmax(flat_logits[:, 0, :], dim=-1)  # (B*M, V)
            flat_states = current_states.reshape(B * beam_size)

            # 混合密集/稀疏掩码
            if step < self.d_dense - 1:
                # --- 密集特化 ---
                parent_tokens = (flat_states - 1).long()
                masks = self.dense_mask[parent_tokens]
                flat_logprobs = torch.where(
                    masks, flat_logprobs, torch.tensor(-float("inf"), device=self.device)
                )
                topk_logprobs, topk_indices = torch.topk(
                    flat_logprobs, tokens_per_beam, dim=-1
                )
                next_state_candidates = self.dense_states[
                    parent_tokens.unsqueeze(1), topk_indices.long()
                ]
                limit = tokens_per_beam
                cand_logprobs = topk_logprobs
                cand_indices = topk_indices
                cand_states = next_state_candidates
            else:
                # --- 稀疏 CSR 查找 ---
                limit = self.max_branch_factors[step + 1]
                cand_logprobs, cand_indices, cand_states = generate_and_apply_logprobs_mask(
                    flat_logprobs, flat_states,
                    self.packed_csr, self.csr_indptr,
                    limit, V, self.device,
                )

            # --- 得分更新 & Beam 选择 ---
            scores = current_scores.unsqueeze(2) + cand_logprobs.view(B, beam_size, limit)
            flat_scores = scores.view(B, beam_size * limit)

            top_scores, flat_top_indices = torch.topk(flat_scores, beam_size, dim=-1)

            top_beam_indices = flat_top_indices // limit
            flat_tokens = cand_indices.view(B, beam_size * limit)
            flat_next_states = cand_states.view(B, beam_size * limit)

            top_tokens = _gather_beams(flat_tokens, flat_top_indices)
            current_states = _gather_beams(flat_next_states, flat_top_indices)

            token_buffer = _gather_beams(token_buffer, top_beam_indices)
            token_buffer[:, :, step + 1] = top_tokens
            current_scores = top_scores

        return token_buffer, current_scores

    @torch.inference_mode()
    def retrieve_and_score(self, batch: dict[str, torch.Tensor],
                           sid_to_ad_map: dict[tuple, int]) -> list[list[dict]]:
        """端到端检索+评分流程。

        Args:
            batch: 用户请求 batch
            sid_to_ad_map: 语义ID → 广告ID 的映射

        Returns:
            每个请求的候选广告列表，每个广告含 {ad_id, ctr, cvr, score, semantic_id}
        """
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        # 获取用户表示
        user_repr = self.model.get_user_representation(batch)

        # STATIC 约束解码
        decoded_sids, beam_scores = self.decode(user_repr)

        B = decoded_sids.size(0)
        results = []

        for i in range(B):
            candidates = []
            for j in range(self.beam_size):
                sid = tuple(decoded_sids[i, j].cpu().numpy().tolist())
                ad_id = sid_to_ad_map.get(sid, -1)
                if ad_id < 0:
                    continue

                candidates.append({
                    "ad_id": ad_id,
                    "semantic_id": sid,
                    "retrieval_score": beam_scores[i, j].item(),
                })

            results.append(candidates)

        return results
