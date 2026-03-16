"""
DSP 竞价引擎 - CTR/CVR 预估驱动的智能出价

将 STATIC 生成式检索与 CTR/CVR 预估结果整合为竞价决策，支持：
- oCPC (优化点击出价)
- oCPM (优化千次展示出价)
- CPC / CPM 基础出价

竞价流程：
  广告请求 → 用户特征提取 → STATIC 约束检索候选广告
       → CTR/CVR 精排 → 出价策略 → 竞价响应
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class BidRequest:
    """广告竞价请求。"""
    request_id: str
    user_features: dict[str, Any]
    context_features: dict[str, Any]
    behavior_seq: list[int]
    slot_info: dict[str, Any]  # 广告位信息
    floor_price: float = 0.0   # 底价


@dataclass
class BidCandidate:
    """竞价候选广告。"""
    ad_id: int
    semantic_id: tuple[int, ...]
    ctr: float = 0.0
    cvr: float = 0.0
    ecpm: float = 0.0       # 预估千次展示收益
    bid_price: float = 0.0  # 出价
    retrieval_score: float = 0.0


@dataclass
class BidResponse:
    """竞价响应。"""
    request_id: str
    candidates: list[BidCandidate] = field(default_factory=list)
    winner: BidCandidate | None = None
    latency_ms: float = 0.0


class BiddingStrategy:
    """出价策略。"""

    def __init__(self, config: dict[str, Any]):
        bid_cfg = config["bidding"]
        self.strategy = bid_cfg["strategy"]
        self.base_cpm = bid_cfg["base_cpm"]
        self.ctr_threshold = bid_cfg["ctr_threshold"]
        self.cvr_threshold = bid_cfg["cvr_threshold"]

    def compute_bid(self, candidate: BidCandidate,
                    advertiser_bid: float = 0.0) -> float:
        """根据策略计算出价。

        Args:
            candidate: 候选广告（含CTR/CVR预估值）
            advertiser_bid: 广告主设定的出价

        Returns:
            最终出价 (CPM)
        """
        ctr = candidate.ctr
        cvr = candidate.cvr

        if self.strategy == "ocpc":
            # oCPC: bid = advertiser_cpc_bid * predicted_ctr * 1000
            bid = advertiser_bid * ctr * 1000
        elif self.strategy == "ocpm":
            # oCPM: bid = advertiser_target_cpa * predicted_ctr * predicted_cvr * 1000
            bid = advertiser_bid * ctr * cvr * 1000
        elif self.strategy == "cpc":
            # CPC: bid = fixed_cpc * predicted_ctr * 1000
            bid = advertiser_bid * ctr * 1000
        elif self.strategy == "cpm":
            # CPM: 直接使用广告主出价
            bid = advertiser_bid
        else:
            # 默认: eCPM = base_cpm * ctr_multiplier
            bid = self.base_cpm * (ctr / 0.05)  # 以5%为基准CTR

        return max(bid, 0.0)

    def compute_ecpm(self, candidate: BidCandidate) -> float:
        """计算 eCPM（预估千次展示收益）。"""
        return candidate.ctr * candidate.bid_price * 1000

    def should_bid(self, candidate: BidCandidate) -> bool:
        """判断是否应该出价（CTR/CVR 过滤）。"""
        if candidate.ctr < self.ctr_threshold:
            return False
        if candidate.cvr < self.cvr_threshold:
            return False
        return True


class DSPBidEngine:
    """DSP 竞价引擎。

    端到端处理流程：
    1. 接收 BidRequest
    2. STATIC 约束检索候选广告
    3. CTR/CVR 精排打分
    4. 出价策略计算
    5. 返回 BidResponse
    """

    def __init__(self, model: nn.Module, decoder: Any,
                 sid_to_ad_map: dict[tuple, int],
                 ad_info: dict[int, dict],
                 config: dict[str, Any],
                 device: torch.device):
        """
        Args:
            model: DSPMultiTaskModel
            decoder: STATICConstrainedDecoder
            sid_to_ad_map: 语义ID → 广告ID
            ad_info: 广告详情 {ad_id: {"advertiser_bid": 1.0, "budget": 1000, ...}}
            config: 全局配置
            device: 计算设备
        """
        self.model = model
        self.decoder = decoder
        self.sid_to_ad_map = sid_to_ad_map
        self.ad_info = ad_info
        self.config = config
        self.device = device

        self.strategy = BiddingStrategy(config)

    def process_request(self, request: BidRequest) -> BidResponse:
        """处理单个竞价请求。

        优化: 使用 merged_forward 单次 LLM 前向获取 CTR/CVR + user_repr。
        """
        start_time = time.time()

        response = BidResponse(request_id=request.request_id)

        try:
            # 1. 构造模型输入 batch
            batch = self._build_batch(request)

            # 2. 合并前向: 单次 LLM 调用同时出 CTR/CVR + user_repr
            with torch.inference_mode():
                batch_gpu = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                if hasattr(self.model, 'merged_forward'):
                    output = self.model.merged_forward(batch_gpu)
                elif hasattr(self.model, 'merged_inference'):
                    output = self.model.merged_inference(batch_gpu)
                else:
                    output = self.model(batch_gpu)
                    output["user_repr"] = self.model.get_user_representation(batch_gpu)

                user_repr = output["user_repr"]
                ctr_prob = output["ctr_prob"].cpu().item()
                cvr_prob = output["cvr_prob"].cpu().item()

                # 3. STATIC 约束检索
                decoded_sids, beam_scores = self.decoder.decode(user_repr)

            # 4. 构建候选
            candidates = []
            for j in range(decoded_sids.size(1)):
                sid = tuple(decoded_sids[0, j].cpu().numpy().tolist())
                ad_id = self.sid_to_ad_map.get(sid, -1)
                if ad_id < 0:
                    continue
                cand = BidCandidate(
                    ad_id=ad_id, semantic_id=sid,
                    ctr=ctr_prob, cvr=cvr_prob,
                    retrieval_score=beam_scores[0, j].item(),
                )
                info = self.ad_info.get(cand.ad_id, {})
                advertiser_bid = info.get("advertiser_bid", self.strategy.base_cpm / 1000)
                cand.bid_price = self.strategy.compute_bid(cand, advertiser_bid)
                cand.ecpm = self.strategy.compute_ecpm(cand)
                candidates.append(cand)

            # 5. 过滤 + 排序
            valid_candidates = [c for c in candidates if self.strategy.should_bid(c)]
            valid_candidates.sort(key=lambda c: c.ecpm, reverse=True)
            valid_candidates = [
                c for c in valid_candidates
                if c.bid_price >= request.floor_price
            ]

            response.candidates = valid_candidates
            if valid_candidates:
                response.winner = valid_candidates[0]

        except Exception as e:
            logger.error(f"竞价请求处理失败 [{request.request_id}]: {e}")

        response.latency_ms = (time.time() - start_time) * 1000
        return response

    def process_batch(self, requests: list[BidRequest]) -> list[BidResponse]:
        """批量处理竞价请求（提升GPU利用率）。

        优化: 使用 merged_forward 消除重复 LLM 调用,
        一次前向同时产出 CTR/CVR/user_repr。
        """
        start_time = time.time()

        # 批量构造输入
        batches = [self._build_batch(req) for req in requests]
        merged_batch = self._merge_batches(batches)

        with torch.inference_mode():
            merged_batch_gpu = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in merged_batch.items()
            }

            # 合并前向: 单次 LLM 调用同时获取 CTR/CVR + user_repr
            if hasattr(self.model, 'merged_forward'):
                output = self.model.merged_forward(merged_batch_gpu)
            elif hasattr(self.model, 'merged_inference'):
                output = self.model.merged_inference(merged_batch_gpu)
            else:
                # 回退: 分别调用 (兼容旧模型)
                user_reprs = self.model.get_user_representation(merged_batch_gpu)
                output = self.model(merged_batch_gpu)
                output["user_repr"] = user_reprs

            user_reprs = output["user_repr"]
            ctr_probs = output["ctr_prob"].cpu().numpy()
            cvr_probs = output["cvr_prob"].cpu().numpy()

            # 批量约束解码
            decoded_sids, beam_scores = self.decoder.decode(user_reprs)

        # 逐请求构建响应
        responses = []
        B = len(requests)
        for i in range(B):
            response = BidResponse(request_id=requests[i].request_id)
            candidates = []

            for j in range(decoded_sids.size(1)):  # beam_size
                sid = tuple(decoded_sids[i, j].cpu().numpy().tolist())
                ad_id = self.sid_to_ad_map.get(sid, -1)
                if ad_id < 0:
                    continue

                cand = BidCandidate(
                    ad_id=ad_id,
                    semantic_id=sid,
                    ctr=float(ctr_probs[i]),
                    cvr=float(cvr_probs[i]),
                    retrieval_score=beam_scores[i, j].item(),
                )

                info = self.ad_info.get(ad_id, {})
                advertiser_bid = info.get("advertiser_bid", self.strategy.base_cpm / 1000)
                cand.bid_price = self.strategy.compute_bid(cand, advertiser_bid)
                cand.ecpm = self.strategy.compute_ecpm(cand)

                if self.strategy.should_bid(cand) and cand.bid_price >= requests[i].floor_price:
                    candidates.append(cand)

            candidates.sort(key=lambda c: c.ecpm, reverse=True)
            response.candidates = candidates
            response.winner = candidates[0] if candidates else None
            responses.append(response)

        total_ms = (time.time() - start_time) * 1000
        logger.info(f"批量竞价完成: {B} 个请求, 总耗时 {total_ms:.1f}ms, 平均 {total_ms / B:.1f}ms/req")

        return responses

    def _build_batch(self, request: BidRequest) -> dict[str, torch.Tensor]:
        """将单个请求转为模型输入张量。"""
        batch = {}

        for key, val in request.user_features.items():
            if isinstance(val, (list, np.ndarray)):
                batch[f"user_{key}"] = torch.tensor(val, dtype=torch.long).unsqueeze(0)
            else:
                batch[f"user_{key}"] = torch.tensor([val], dtype=torch.long)

        for key, val in request.context_features.items():
            if isinstance(val, (list, np.ndarray)):
                batch[f"ctx_{key}"] = torch.tensor(val, dtype=torch.long).unsqueeze(0)
            else:
                batch[f"ctx_{key}"] = torch.tensor([val], dtype=torch.long)

        # 行为序列
        seq = request.behavior_seq[-50:]
        padded = [0] * (50 - len(seq)) + seq
        batch["behavior_seq"] = torch.tensor([padded], dtype=torch.long)

        # 广告侧占位（检索模式）
        for key in self.config["data"]["ad_features"]:
            batch[f"ad_{key}"] = torch.zeros(1, dtype=torch.long)

        return batch

    def _merge_batches(self, batches: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """合并多个单样本 batch。"""
        merged = {}
        if not batches:
            return merged

        for key in batches[0]:
            tensors = [b[key] for b in batches if key in b]
            if tensors:
                merged[key] = torch.cat(tensors, dim=0)

        return merged
