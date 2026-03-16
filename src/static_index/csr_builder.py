"""
STATIC 索引构建模块 - Trie → CSR 稀疏矩阵

将广告语义ID集合转换为 STATIC 索引结构，用于高效约束解码。

核心数据结构：
  - start_mask: 根节点有效token掩码, shape (V,)
  - dense_mask: 前d层密集掩码, shape (V,)*d
  - dense_states: 前d层状态表, shape (V,)*d
  - packed_csr: 扁平化CSR转移表 [token_id, next_state], shape (num_transitions+V, 2)
  - indptr: CSR行指针, shape (num_states+2,)
  - layer_max_branches: 各层最大分支因子, length L

实现参考：https://github.com/youtube/static-constraint-decoding
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class STATICIndex:
    """STATIC 索引数据结构。"""
    packed_csr: np.ndarray          # shape (num_transitions + V, 2)
    indptr: np.ndarray              # shape (num_states + 2,)
    layer_max_branches: tuple[int, ...]  # length L
    start_mask: np.ndarray          # shape (V,)
    dense_mask: np.ndarray          # shape (V,)*d
    dense_states: np.ndarray        # shape (V,)*d
    vocab_size: int
    sid_length: int
    num_constraints: int

    def save(self, path: str) -> None:
        """持久化索引到 .npz 文件。"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            packed_csr=self.packed_csr,
            indptr=self.indptr,
            layer_max_branches=np.array(self.layer_max_branches),
            start_mask=self.start_mask,
            dense_mask=self.dense_mask,
            dense_states=self.dense_states,
            meta=np.array([self.vocab_size, self.sid_length, self.num_constraints]),
        )
        logger.info(f"STATIC 索引已保存: {path}")

    @classmethod
    def load(cls, path: str) -> STATICIndex:
        """从 .npz 文件加载索引。"""
        data = np.load(path, allow_pickle=True)
        meta = data["meta"]
        index = cls(
            packed_csr=data["packed_csr"],
            indptr=data["indptr"],
            layer_max_branches=tuple(data["layer_max_branches"].tolist()),
            start_mask=data["start_mask"],
            dense_mask=data["dense_mask"],
            dense_states=data["dense_states"],
            vocab_size=int(meta[0]),
            sid_length=int(meta[1]),
            num_constraints=int(meta[2]),
        )
        logger.info(
            f"STATIC 索引已加载: {path}, "
            f"约束数={index.num_constraints}, V={index.vocab_size}, L={index.sid_length}"
        )
        return index


def build_static_index(
    fresh_sids: np.ndarray,
    vocab_size: int = 1024,
    dense_lookup_layers: int = 2,
) -> STATICIndex:
    """构建 STATIC 索引。

    将语义ID集合（前缀树）扁平化为静态 CSR 矩阵 + 密集查找表混合结构。

    Args:
        fresh_sids: 排序后的语义ID数组, shape (N, L)
                    N=约束集大小, L=语义ID长度
        vocab_size: 词表大小 V（每层码本大小）
        dense_lookup_layers: 用密集查找表处理的前几层数

    Returns:
        STATICIndex 数据结构
    """
    N, L = fresh_sids.shape
    logger.info(f"开始构建 STATIC 索引: N={N}, L={L}, V={vocab_size}, dense_layers={dense_lookup_layers}")

    if dense_lookup_layers >= L:
        raise ValueError(
            f"dense_lookup_layers ({dense_lookup_layers}) 必须小于语义ID长度 L ({L})"
        )

    # --- 1. 根节点掩码 ---
    start_mask = np.zeros(vocab_size, dtype=bool)
    start_mask[np.unique(fresh_sids[:, 0])] = True

    # --- 2. 向量化Trie节点识别 ---
    diff = fresh_sids[1:] != fresh_sids[:-1]
    first_diff = np.full(N - 1, L, dtype=np.int8)
    has_diff = diff.any(axis=1)
    first_diff[has_diff] = diff[has_diff].argmax(axis=1)

    is_new = np.zeros((N, L), dtype=bool)
    is_new[0, :] = True
    for depth in range(L):
        is_new[1:, depth] = (first_diff <= depth)

    # --- 3. 状态ID分配 ---
    state_ids = np.zeros((N, L - 1), dtype=np.int32)
    state_ids[:, 0] = fresh_sids[:, 0].astype(np.int32) + 1

    depth_id_ranges = []
    current_offset = vocab_size + 1

    for depth in range(1, L - 1):
        mask = is_new[:, depth]
        num_new = np.sum(mask)
        start_id = current_offset
        end_id = current_offset + num_new
        depth_id_ranges.append((start_id, end_id))

        state_ids[mask, depth] = np.arange(start_id, end_id, dtype=np.int32)
        state_ids[:, depth] = np.maximum.accumulate(state_ids[:, depth])
        current_offset += num_new

    num_states = current_offset

    # --- 4. 边收集 ---
    all_parents, all_tokens, all_children = [], [], []
    for depth in range(1, L):
        mask = is_new[:, depth]
        parent_ids = state_ids[mask, depth - 1]
        token_ids = fresh_sids[mask, depth].astype(np.int32)
        child_ids = (
            state_ids[mask, depth] if depth < L - 1
            else np.zeros_like(parent_ids, dtype=np.int32)
        )
        all_parents.append(parent_ids)
        all_tokens.append(token_ids)
        all_children.append(child_ids)

    # --- 5. 密集特化 ---
    dense_shape = tuple([vocab_size] * dense_lookup_layers)
    dense_mask = np.zeros(dense_shape, dtype=bool)
    dense_states = np.zeros(dense_shape, dtype=np.int32)

    indices = tuple(
        fresh_sids[:, i].astype(np.int32) for i in range(dense_lookup_layers)
    )
    final_dense_ids = state_ids[:, dense_lookup_layers - 1]
    dense_mask[indices] = True
    dense_states[indices] = final_dense_ids

    # --- 6. CSR 构建 ---
    parents = np.concatenate(all_parents)
    tokens = np.concatenate(all_tokens)
    children = np.concatenate(all_children)
    del state_ids, is_new
    gc.collect()

    counts = np.bincount(parents, minlength=num_states)
    indptr = np.zeros(num_states + 1, dtype=np.int32)
    indptr[1:] = np.cumsum(counts)

    # --- 7. 各层最大分支因子 ---
    layer_max_branches = [int(np.sum(start_mask))]
    l0_counts = counts[1:vocab_size + 1]
    layer_max_branches.append(int(l0_counts.max()) if len(l0_counts) > 0 else 0)
    for (start_id, end_id) in depth_id_ranges:
        if start_id < len(counts):
            layer_counts = counts[start_id:end_id]
            layer_max_branches.append(int(layer_counts.max()) if len(layer_counts) > 0 else 0)
        else:
            layer_max_branches.append(0)
    while len(layer_max_branches) < L:
        layer_max_branches.append(1)

    # --- 8. 打包 ---
    raw_indices = np.concatenate([tokens, np.full(vocab_size, vocab_size, dtype=np.int32)])
    raw_data = np.concatenate([children, np.zeros(vocab_size, dtype=np.int32)])
    indptr = np.append(indptr, indptr[-1] + vocab_size)
    packed_csr = np.ascontiguousarray(np.vstack([raw_indices, raw_data]).T)

    logger.info(
        f"STATIC 索引构建完成: states={num_states}, transitions={len(tokens)}, "
        f"max_branches={layer_max_branches}"
    )

    return STATICIndex(
        packed_csr=packed_csr,
        indptr=indptr,
        layer_max_branches=tuple(layer_max_branches),
        start_mask=start_mask,
        dense_mask=dense_mask,
        dense_states=dense_states,
        vocab_size=vocab_size,
        sid_length=L,
        num_constraints=N,
    )


def build_index_with_business_rules(
    all_semantic_ids: np.ndarray,
    ad_metadata: dict[int, dict],
    rules: list[dict],
    vocab_size: int = 1024,
    dense_lookup_layers: int = 2,
) -> STATICIndex:
    """带业务规则过滤的索引构建。

    在构建STATIC索引前，先根据业务规则过滤广告。

    Args:
        all_semantic_ids: 所有广告的语义ID, shape (num_ads, L)
        ad_metadata: 广告元数据, {ad_index: {"is_active": True, "budget_remaining": 100, ...}}
        rules: 业务规则列表
        vocab_size: 词表大小
        dense_lookup_layers: 密集查找层数

    Returns:
        STATICIndex
    """
    # 过滤有效广告
    valid_indices = []
    for idx in range(len(all_semantic_ids)):
        meta = ad_metadata.get(idx, {})
        is_valid = True
        for rule in rules:
            field = rule.get("field", "")
            if field == "is_active" and not meta.get("is_active", False):
                is_valid = False
                break
            if "min_value" in rule:
                if meta.get(field, 0) < rule["min_value"]:
                    is_valid = False
                    break
            if "allowed" in rule and rule["allowed"]:
                if meta.get(field, "") not in rule["allowed"]:
                    is_valid = False
                    break
        if is_valid:
            valid_indices.append(idx)

    filtered_sids = all_semantic_ids[valid_indices]
    logger.info(f"业务规则过滤: {len(all_semantic_ids)} → {len(filtered_sids)} 个广告")

    # 去重并排序
    unique_sids = np.unique(filtered_sids, axis=0)
    sorted_indices = np.lexsort(unique_sids[:, ::-1].T)
    sorted_sids = unique_sids[sorted_indices]

    return build_static_index(sorted_sids, vocab_size, dense_lookup_layers)
