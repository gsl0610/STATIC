"""
三方模型对比测试框架 — 混元 LLM (A) vs DeepFM (B) vs PEPNet (C)

使用相同的训练/测试数据, 从多个维度对比三种 CTR/CVR 预估方案:
  1. 预估效果: CTR AUC, CVR AUC, LogLoss, 校准度
  2. 推理延迟: P50/P90/P99, 吞吐量 (QPS)
  3. 竞价效果: eCPM, 出价分布, 胜出率, RPM
  4. 模型效率: 参数量, 显存占用, 训练耗时

用法:
  python run_pipeline.py --stage model_compare
  python run_pipeline.py --stage model_compare --device cuda
"""

from __future__ import annotations

import logging
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, log_loss

logger = logging.getLogger(__name__)


def _flush_logs():
    """强制刷新所有日志 handler 和 stdout/stderr。"""
    for handler in logging.root.handlers:
        handler.flush()
    sys.stdout.flush()
    sys.stderr.flush()


# ================================================================
# 数据类
# ================================================================
@dataclass
class ModelMetrics:
    """单个模型的评估指标。"""
    name: str
    tag: str = ""  # A/B/C

    # 模型信息
    total_params: float = 0.0
    trainable_params: float = 0.0
    model_size_mb: float = 0.0

    # 延迟指标
    latency_mean_ms: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p90_ms: float = 0.0
    latency_p99_ms: float = 0.0
    throughput_qps: float = 0.0

    # 动态批处理吞吐
    batch_throughput_qps: float = 0.0
    batch_size_tested: int = 0

    # 预估效果
    ctr_auc: float = 0.0
    cvr_auc: float = 0.0
    ctr_logloss: float = 0.0
    cvr_logloss: float = 0.0
    ctr_calibration: float = 0.0
    cvr_calibration: float = 0.0

    # 竞价效果
    avg_ecpm: float = 0.0
    avg_bid_price: float = 0.0
    win_rate: float = 0.0
    revenue_per_mille: float = 0.0

    # 训练信息
    train_time_sec: float = 0.0
    best_val_loss: float = 0.0


@dataclass
class CompareResult:
    """三方对比测试结果。"""
    models: list = field(default_factory=list)  # list[ModelMetrics]
    test_samples: int = 0
    test_date: str = ""


# ================================================================
# 三方对比测试引擎
# ================================================================
class ModelCompareEngine:
    """三方模型对比测试引擎。

    自动完成:
    1. 创建三个模型: 混元 LLM / DeepFM / PEPNet
    2. 使用相同数据训练
    3. 在相同测试集上评估 CTR/CVR 效果
    4. 延迟基准测试
    5. 模拟竞价对比
    """

    def __init__(self, config: dict[str, Any], device: torch.device):
        self.config = config
        self.device = device

    def run(self, vocab_info: dict,
            semantic_ids: np.ndarray | None = None) -> CompareResult:
        """运行完整三方对比测试。"""
        from src.data.feature_engineering import generate_synthetic_data, DSPDatasetBuilder
        from src.data.dataset import create_dataloader

        logger.info("=" * 80)
        logger.info("  三方模型对比测试")
        logger.info("  [A] 混元 LLM 1.8B + LoRA")
        logger.info("  [B] DeepFM")
        logger.info("  [C] PEPNet")
        logger.info("=" * 80)
        _flush_logs()

        result = CompareResult()

        # ──────────────────────────────────────────────
        # Step 1: 准备统一数据
        # ──────────────────────────────────────────────
        logger.info("\n[Step 1/6] 准备统一训练/测试数据...")
        _flush_logs()
        data_cfg = self.config["data"]
        processed_dir = data_cfg["processed_dir"]

        train_path = os.path.join(processed_dir, "train.npz")
        test_path = os.path.join(processed_dir, "test.npz")
        val_path = os.path.join(processed_dir, "val.npz")

        if not all(os.path.exists(p) for p in [train_path, test_path, val_path]):
            logger.info("  数据不存在, 生成模拟数据...")
            df = generate_synthetic_data(num_samples=200000)
            builder = DSPDatasetBuilder(self.config)
            builder.build_from_dataframe(df, processed_dir)

            if not vocab_info:
                encoder = builder.encoder
                vocab_info = {col: encoder.get_vocab_size(col)
                              for col in encoder.vocab}
                import yaml
                vocab_path = os.path.join(processed_dir, "vocab_sizes.yaml")
                with open(vocab_path, "w") as f:
                    yaml.dump(vocab_info, f)

        batch_size = self.config["training"]["batch_size"]
        train_loader = create_dataloader(
            train_path, data_cfg, batch_size=batch_size,
            shuffle=True, num_workers=0, ad_semantic_ids=semantic_ids,
        )
        val_loader = create_dataloader(
            val_path, data_cfg, batch_size=batch_size,
            shuffle=False, num_workers=0, ad_semantic_ids=semantic_ids,
        )
        test_loader = create_dataloader(
            test_path, data_cfg, batch_size=batch_size,
            shuffle=False, num_workers=0, ad_semantic_ids=semantic_ids,
        )

        result.test_samples = len(test_loader.dataset)
        logger.info(f"  训练集: {len(train_loader.dataset)}, "
                    f"验证集: {len(val_loader.dataset)}, "
                    f"测试集: {len(test_loader.dataset)}")

        # ──────────────────────────────────────────────
        # Step 2: 创建三个模型
        # ──────────────────────────────────────────────
        logger.info("\n[Step 2/6] 创建对比模型...")
        models = [
            ("A", "Hunyuan-1.8B + LoRA", self._create_hunyuan(vocab_info)),
            ("B", "DeepFM", self._create_deepfm(vocab_info)),
            ("C", "PEPNet", self._create_pepnet(vocab_info)),
        ]

        metrics_list = []
        for tag, name, model in models:
            m = ModelMetrics(name=name, tag=tag)
            m.total_params = sum(p.numel() for p in model.parameters()) / 1e6
            m.trainable_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ) / 1e6
            m.model_size_mb = sum(
                p.numel() * p.element_size() for p in model.parameters()
            ) / 1e6
            metrics_list.append(m)
            logger.info(f"  [{tag}] {name}: {m.total_params:.1f}M params "
                        f"({m.trainable_params:.1f}M trainable), "
                        f"{m.model_size_mb:.0f}MB")

        # ──────────────────────────────────────────────
        # Step 3: 训练
        # ──────────────────────────────────────────────
        logger.info("\n[Step 3/6] 训练模型...")
        _flush_logs()
        for i, (tag, name, model) in enumerate(models):
            try:
                model, metrics_list[i] = self._train_model(
                    model, metrics_list[i], train_loader, val_loader
                )
                models[i] = (tag, name, model)
            except Exception as e:
                logger.error(f"  [{tag}] {name} 训练失败: {e}")
                logger.error(traceback.format_exc())
                _flush_logs()
            _flush_logs()

        # ──────────────────────────────────────────────
        # Step 4: 评估效果
        # ──────────────────────────────────────────────
        logger.info("\n[Step 4/6] 评估预估效果...")
        _flush_logs()
        for i, (tag, name, model) in enumerate(models):
            try:
                metrics_list[i] = self._evaluate_prediction(
                    model, metrics_list[i], test_loader
                )
            except Exception as e:
                logger.error(f"  [{tag}] {name} 评估失败: {e}")
                logger.error(traceback.format_exc())
                _flush_logs()

        # ──────────────────────────────────────────────
        # Step 5: 延迟基准测试
        # ──────────────────────────────────────────────
        logger.info("\n[Step 5/6] 延迟基准测试...")
        _flush_logs()
        for i, (tag, name, model) in enumerate(models):
            try:
                metrics_list[i] = self._benchmark_latency(
                    model, metrics_list[i], vocab_info
                )
            except Exception as e:
                logger.error(f"  [{tag}] {name} 延迟测试失败: {e}")
                logger.error(traceback.format_exc())
                _flush_logs()

        # ──────────────────────────────────────────────
        # Step 6: 竞价模拟
        # ──────────────────────────────────────────────
        logger.info("\n[Step 6/6] 竞价模拟...")
        _flush_logs()
        try:
            metrics_list = self._simulate_bidding(
                [(tag, name, model) for tag, name, model in models],
                metrics_list, test_loader,
            )
        except Exception as e:
            logger.error(f"  竞价模拟失败: {e}")
            logger.error(traceback.format_exc())
            _flush_logs()

        result.models = metrics_list
        result.test_date = time.strftime("%Y-%m-%d %H:%M:%S")

        self._print_report(result)
        _flush_logs()
        return result

    # ────────────────────────────────────────────────────
    # 模型创建
    # ────────────────────────────────────────────────────
    def _get_vocabs(self, vocab_info: dict):
        data_cfg = self.config["data"]
        user_vocab = {n: vocab_info.get(n, 100) for n in data_cfg["user_features"]}
        ad_vocab = {n: vocab_info.get(n, 100) for n in data_cfg["ad_features"]}
        ctx_vocab = {n: vocab_info.get(n, 100) for n in data_cfg["context_features"]}
        stat_vocab = {n: vocab_info.get(n, 20)
                      for n in data_cfg.get("stat_features", [])}
        return user_vocab, ad_vocab, ctx_vocab, stat_vocab

    def _create_hunyuan(self, vocab_info: dict) -> nn.Module:
        logger.info("  创建 [A] Hunyuan-1.8B + LoRA...")
        user_vocab, ad_vocab, ctx_vocab, stat_vocab = self._get_vocabs(vocab_info)
        from src.model.hunyuan_model import create_hunyuan_model
        model = create_hunyuan_model(
            config=self.config,
            user_feature_vocab_sizes=user_vocab,
            ad_feature_vocab_sizes=ad_vocab,
            context_feature_vocab_sizes=ctx_vocab,
            stat_feature_vocab_sizes=stat_vocab,
            behavior_vocab_size=50000,
        )
        return model.to(self.device)

    def _create_deepfm(self, vocab_info: dict) -> nn.Module:
        logger.info("  创建 [B] DeepFM...")
        user_vocab, ad_vocab, ctx_vocab, stat_vocab = self._get_vocabs(vocab_info)
        from src.model.deepfm import DeepFMModel
        model = DeepFMModel(
            config=self.config,
            user_feature_vocab_sizes=user_vocab,
            ad_feature_vocab_sizes=ad_vocab,
            context_feature_vocab_sizes=ctx_vocab,
            stat_feature_vocab_sizes=stat_vocab,
            behavior_vocab_size=50000,
        )
        return model.to(self.device)

    def _create_pepnet(self, vocab_info: dict) -> nn.Module:
        logger.info("  创建 [C] PEPNet...")
        user_vocab, ad_vocab, ctx_vocab, stat_vocab = self._get_vocabs(vocab_info)
        from src.model.pepnet import PEPNetModel
        model = PEPNetModel(
            config=self.config,
            user_feature_vocab_sizes=user_vocab,
            ad_feature_vocab_sizes=ad_vocab,
            context_feature_vocab_sizes=ctx_vocab,
            stat_feature_vocab_sizes=stat_vocab,
            behavior_vocab_size=50000,
        )
        return model.to(self.device)

    # ────────────────────────────────────────────────────
    # 训练
    # ────────────────────────────────────────────────────
    def _train_model(self, model: nn.Module, metrics: ModelMetrics,
                     train_loader, val_loader) -> tuple[nn.Module, ModelMetrics]:
        logger.info(f"\n  训练 [{metrics.tag}] {metrics.name}...")

        from src.model.trainer import MultiTaskTrainer
        ab_config = {**self.config}
        ab_training = {**self.config["training"]}
        ab_training["max_steps"] = min(ab_training.get("max_steps", 2000), 2000)
        ab_training["eval_every"] = 500
        ab_training["save_every"] = 99999
        ab_training["early_stopping_patience"] = 5
        ab_config["training"] = ab_training

        trainer = MultiTaskTrainer(model, ab_config, self.device)

        start_time = time.time()
        history = trainer.train(train_loader, val_loader)
        metrics.train_time_sec = time.time() - start_time

        if history.get("val_loss"):
            metrics.best_val_loss = min(history["val_loss"])
        elif history.get("train_loss"):
            metrics.best_val_loss = history["train_loss"][-1]

        logger.info(f"  [{metrics.tag}] {metrics.name} 训练完成: "
                    f"{metrics.train_time_sec:.0f}s, "
                    f"best_val_loss={metrics.best_val_loss:.4f}")

        model.eval()
        return model, metrics

    # ────────────────────────────────────────────────────
    # 预估效果评估
    # ────────────────────────────────────────────────────
    @torch.inference_mode()
    def _evaluate_prediction(self, model: nn.Module, metrics: ModelMetrics,
                             test_loader) -> ModelMetrics:
        model.eval()

        all_ctr_preds, all_cvr_preds = [], []
        all_ctr_labels, all_cvr_labels = [], []

        for batch in test_loader:
            batch_gpu = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            ctr_label = batch_gpu.pop("label_click", None)
            cvr_label = batch_gpu.pop("label_conversion", None)
            batch_gpu.pop("ad_semantic_id", None)

            output = model(batch_gpu)

            if ctr_label is not None:
                all_ctr_preds.append(output["ctr_prob"].cpu().numpy())
                all_ctr_labels.append(ctr_label.cpu().numpy())
            if cvr_label is not None:
                all_cvr_preds.append(output["cvr_prob"].cpu().numpy())
                all_cvr_labels.append(cvr_label.cpu().numpy())

        ctr_preds = np.concatenate(all_ctr_preds)
        ctr_labels = np.concatenate(all_ctr_labels)
        cvr_preds = np.concatenate(all_cvr_preds)
        cvr_labels = np.concatenate(all_cvr_labels)

        # CTR 指标
        try:
            metrics.ctr_auc = roc_auc_score(ctr_labels, ctr_preds)
        except ValueError:
            metrics.ctr_auc = 0.5

        ctr_preds_clipped = np.clip(ctr_preds, 1e-7, 1 - 1e-7)
        metrics.ctr_logloss = log_loss(ctr_labels, ctr_preds_clipped)

        actual_ctr = ctr_labels.mean()
        pred_ctr = ctr_preds.mean()
        metrics.ctr_calibration = pred_ctr / max(actual_ctr, 1e-7)

        # CVR 指标
        try:
            metrics.cvr_auc = roc_auc_score(cvr_labels, cvr_preds)
        except ValueError:
            metrics.cvr_auc = 0.5

        cvr_preds_clipped = np.clip(cvr_preds, 1e-7, 1 - 1e-7)
        metrics.cvr_logloss = log_loss(cvr_labels, cvr_preds_clipped)

        actual_cvr = cvr_labels.mean()
        pred_cvr = cvr_preds.mean()
        metrics.cvr_calibration = pred_cvr / max(actual_cvr, 1e-7)

        logger.info(f"  [{metrics.tag}] {metrics.name}: "
                    f"CTR AUC={metrics.ctr_auc:.4f}, LogLoss={metrics.ctr_logloss:.4f}, "
                    f"校准={metrics.ctr_calibration:.3f}")
        logger.info(f"  [{metrics.tag}] {metrics.name}: "
                    f"CVR AUC={metrics.cvr_auc:.4f}, LogLoss={metrics.cvr_logloss:.4f}, "
                    f"校准={metrics.cvr_calibration:.3f}")

        return metrics

    # ────────────────────────────────────────────────────
    # 延迟基准测试
    # ────────────────────────────────────────────────────
    def _measure_latency(self, model: nn.Module, sample: dict,
                         num_warmup: int = 30, num_runs: int = 200) -> dict:
        use_cuda = self.device.type == "cuda"

        for _ in range(num_warmup):
            with torch.no_grad():
                model(sample)
        if use_cuda:
            torch.cuda.synchronize()

        latencies = []
        for _ in range(num_runs):
            if use_cuda:
                se = torch.cuda.Event(enable_timing=True)
                ee = torch.cuda.Event(enable_timing=True)
                se.record()
                with torch.no_grad():
                    model(sample)
                ee.record()
                torch.cuda.synchronize()
                latencies.append(se.elapsed_time(ee))
            else:
                t0 = time.perf_counter()
                with torch.no_grad():
                    model(sample)
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000)

        lat = np.array(latencies)
        return {
            "mean_ms": float(np.mean(lat)),
            "p50_ms": float(np.percentile(lat, 50)),
            "p90_ms": float(np.percentile(lat, 90)),
            "p99_ms": float(np.percentile(lat, 99)),
            "qps": 1000.0 / float(np.mean(lat)),
        }

    @torch.inference_mode()
    def _benchmark_latency(self, model: nn.Module, metrics: ModelMetrics,
                           vocab_info: dict) -> ModelMetrics:
        model.eval()
        sample = self._make_sample_input(vocab_info)

        logger.info(f"  [{metrics.tag}] {metrics.name} 测量延迟...")
        raw = self._measure_latency(model, sample)
        metrics.latency_mean_ms = raw["mean_ms"]
        metrics.latency_p50_ms = raw["p50_ms"]
        metrics.latency_p90_ms = raw["p90_ms"]
        metrics.latency_p99_ms = raw["p99_ms"]
        metrics.throughput_qps = raw["qps"]

        logger.info(f"  [{metrics.tag}] {metrics.name}: "
                    f"mean={raw['mean_ms']:.2f}ms, P99={raw['p99_ms']:.2f}ms, "
                    f"QPS={raw['qps']:.0f}")

        # 动态批处理
        batch_sizes = [8, 16, 32]
        best_batch_qps = 0.0
        best_batch_size = 1

        for bs in batch_sizes:
            try:
                batch_sample = self._make_sample_input(vocab_info, batch_size=bs)
                bm = self._measure_latency(model, batch_sample,
                                           num_warmup=10, num_runs=50)
                batch_qps = bs * 1000.0 / bm["mean_ms"]
                logger.info(f"  [{metrics.tag}] batch={bs}: "
                            f"latency={bm['mean_ms']:.2f}ms, QPS={batch_qps:.0f}")
                if batch_qps > best_batch_qps:
                    best_batch_qps = batch_qps
                    best_batch_size = bs
            except Exception as e:
                logger.warning(f"  [{metrics.tag}] batch={bs} 测试失败: {e}")
                break

        metrics.batch_throughput_qps = best_batch_qps
        metrics.batch_size_tested = best_batch_size

        return metrics

    # ────────────────────────────────────────────────────
    # 竞价模拟
    # ────────────────────────────────────────────────────
    @torch.inference_mode()
    def _simulate_bidding(self, models: list, metrics_list: list[ModelMetrics],
                          test_loader) -> list[ModelMetrics]:
        for _, _, model in models:
            model.eval()

        bid_cfg = self.config["bidding"]
        strategy = bid_cfg["strategy"]
        ctr_threshold = bid_cfg["ctr_threshold"]
        cvr_threshold = bid_cfg["cvr_threshold"]

        rng = np.random.RandomState(42)
        max_bid_samples = 2000

        # 存储每个模型的竞价数据
        n = len(models)
        ecpms = [[] for _ in range(n)]
        bids = [[] for _ in range(n)]
        wins = [0] * n
        total_requests = 0

        sample_count = 0
        for batch in test_loader:
            if sample_count >= max_bid_samples:
                break

            batch_gpu = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            batch_gpu.pop("label_click", None)
            batch_gpu.pop("label_conversion", None)
            batch_gpu.pop("ad_semantic_id", None)

            outputs = [model(batch_gpu) for _, _, model in models]

            B = outputs[0]["ctr_prob"].size(0)
            for i in range(B):
                if sample_count >= max_bid_samples:
                    break
                sample_count += 1
                total_requests += 1

                advertiser_bid = 0.5 + rng.random() * 2.0

                model_ecpms = []
                model_valid = []
                for mi in range(n):
                    ca = float(outputs[mi]["ctr_prob"][i].cpu())
                    va = float(outputs[mi]["cvr_prob"][i].cpu())

                    if strategy == "ocpc":
                        bid = advertiser_bid * ca * 1000
                    elif strategy == "ocpm":
                        bid = advertiser_bid * ca * va * 1000
                    else:
                        bid = advertiser_bid * ca * 1000

                    ecpm = ca * bid * 1000
                    valid = ca >= ctr_threshold and va >= cvr_threshold

                    model_ecpms.append(ecpm)
                    model_valid.append(valid)

                    if valid:
                        ecpms[mi].append(ecpm)
                        bids[mi].append(bid)

                # 多方竞价: 最高 eCPM 胜出
                valid_indices = [j for j in range(n) if model_valid[j]]
                if valid_indices:
                    winner = max(valid_indices, key=lambda j: model_ecpms[j])
                    wins[winner] += 1

        for mi in range(n):
            m = metrics_list[mi]
            m.avg_ecpm = float(np.mean(ecpms[mi])) if ecpms[mi] else 0
            m.avg_bid_price = float(np.mean(bids[mi])) if bids[mi] else 0
            m.win_rate = wins[mi] / max(total_requests, 1)
            m.revenue_per_mille = m.avg_ecpm * m.win_rate

        logger.info(f"  竞价模拟完成: {total_requests} 个请求")
        for mi in range(n):
            m = metrics_list[mi]
            logger.info(f"  [{m.tag}] {m.name}: 胜出={wins[mi]}, "
                        f"胜率={m.win_rate:.1%}, avgECPM={m.avg_ecpm:.2f}")

        return metrics_list

    # ────────────────────────────────────────────────────
    # 工具方法
    # ────────────────────────────────────────────────────
    def _make_sample_input(self, vocab_info: dict,
                           batch_size: int = 1) -> dict[str, torch.Tensor]:
        data_cfg = self.config["data"]
        device = self.device
        bs = batch_size

        multi_value_feats = {"interest_tags", "interest_tags_l2", "creative_label"}
        batch = {}

        for feat in data_cfg["user_features"]:
            vsize = vocab_info.get(feat, 100)
            if feat in multi_value_feats:
                batch[f"user_{feat}"] = torch.randint(2, max(3, vsize), (bs, 5), device=device)
            else:
                batch[f"user_{feat}"] = torch.randint(2, max(3, vsize), (bs,), device=device)

        for feat in data_cfg["ad_features"]:
            vsize = vocab_info.get(feat, 100)
            if feat in multi_value_feats:
                batch[f"ad_{feat}"] = torch.randint(2, max(3, vsize), (bs, 4), device=device)
            else:
                batch[f"ad_{feat}"] = torch.randint(2, max(3, vsize), (bs,), device=device)

        for feat in data_cfg["context_features"]:
            vsize = vocab_info.get(feat, 50)
            batch[f"ctx_{feat}"] = torch.randint(2, max(3, vsize), (bs,), device=device)

        for feat in data_cfg.get("stat_features", []):
            vsize = vocab_info.get(feat, 20)
            batch[f"stat_{feat}"] = torch.randint(2, max(3, vsize), (bs,), device=device)

        max_seq = data_cfg["behavior"]["max_seq_len"]
        batch["behavior_seq"] = torch.randint(0, 50000, (bs, max_seq), device=device)

        return batch

    # ────────────────────────────────────────────────────
    # 对比报告
    # ────────────────────────────────────────────────────
    def _print_report(self, result: CompareResult):
        models = result.models

        logger.info("\n")
        logger.info("=" * 100)
        logger.info("  三方模型对比测试报告: 混元 LLM vs DeepFM vs PEPNet")
        logger.info("=" * 100)
        logger.info(f"  测试时间: {result.test_date}")
        logger.info(f"  测试样本: {result.test_samples}")
        logger.info(f"  出价策略: {self.config['bidding']['strategy']}")
        logger.info("")

        # 动态生成表头
        tags = [f"[{m.tag}] {m.name}" for m in models]
        header = f"  {'指标':<28}"
        for t in tags:
            header += f" {t:>18}"
        header += f" {'最优':>6}"
        logger.info(header)
        logger.info("  " + "-" * (28 + 18 * len(models) + 8))

        def _row(label, values, fmt=".4f", higher_better=True):
            best_idx = max(range(len(values)), key=lambda i: values[i]) if higher_better \
                else min(range(len(values)), key=lambda i: values[i])
            row = f"  {label:<28}"
            for v in values:
                row += f" {v:>18{fmt}}"
            row += f" {models[best_idx].tag:>6}"
            logger.info(row)

        def _row_calib(label, values):
            best_idx = min(range(len(values)), key=lambda i: abs(values[i] - 1.0))
            row = f"  {label:<28}"
            for v in values:
                row += f" {v:>18.3f}"
            row += f" {models[best_idx].tag:>6}"
            logger.info(row)

        # ── 模型信息 ──
        logger.info("  ── 模型信息 ──")
        _row("总参数量 (M)", [m.total_params for m in models], ".1f", False)
        _row("可训练参数 (M)", [m.trainable_params for m in models], ".1f", False)
        _row("模型大小 (MB)", [m.model_size_mb for m in models], ".0f", False)
        _row("训练耗时 (s)", [m.train_time_sec for m in models], ".0f", False)
        logger.info("")

        # ── 预估效果 ──
        logger.info("  ── 预估效果 ──")
        _row("CTR AUC ↑", [m.ctr_auc for m in models], ".4f", True)
        _row("CTR LogLoss ↓", [m.ctr_logloss for m in models], ".4f", False)
        _row_calib("CTR 校准度 (→1.0)", [m.ctr_calibration for m in models])
        _row("CVR AUC ↑", [m.cvr_auc for m in models], ".4f", True)
        _row("CVR LogLoss ↓", [m.cvr_logloss for m in models], ".4f", False)
        _row_calib("CVR 校准度 (→1.0)", [m.cvr_calibration for m in models])
        logger.info("")

        # ── 推理延迟 ──
        logger.info("  ── 推理延迟 (batch=1) ──")
        _row("Mean 延迟 (ms) ↓", [m.latency_mean_ms for m in models], ".2f", False)
        _row("P50 延迟 (ms) ↓", [m.latency_p50_ms for m in models], ".2f", False)
        _row("P90 延迟 (ms) ↓", [m.latency_p90_ms for m in models], ".2f", False)
        _row("P99 延迟 (ms) ↓", [m.latency_p99_ms for m in models], ".2f", False)
        _row("QPS (batch=1) ↑", [m.throughput_qps for m in models], ".0f", True)
        logger.info("")

        # ── 动态批处理 ──
        logger.info("  ── 动态批处理吞吐 ──")
        _row("最佳批处理 QPS ↑", [m.batch_throughput_qps for m in models], ".0f", True)
        row = f"  {'最佳 batch_size':<28}"
        for m in models:
            row += f" {m.batch_size_tested:>18}"
        logger.info(row)
        logger.info("")

        # ── 竞价效果 ──
        logger.info("  ── 竞价效果 ──")
        _row("平均 eCPM ↑", [m.avg_ecpm for m in models], ".2f", True)
        _row("平均出价 ↑", [m.avg_bid_price for m in models], ".4f", True)
        row = f"  {'竞价胜率 ↑':<28}"
        for m in models:
            row += f" {m.win_rate:>18.1%}"
        best_wr = max(range(len(models)), key=lambda i: models[i].win_rate)
        row += f" {models[best_wr].tag:>6}"
        logger.info(row)
        _row("RPM ↑", [m.revenue_per_mille for m in models], ".2f", True)
        logger.info("")

        # ── 综合评分 ──
        logger.info("  " + "=" * (28 + 18 * len(models) + 8))
        scores = [0] * len(models)

        comparisons = [
            ([m.ctr_auc for m in models], True),
            ([m.cvr_auc for m in models], True),
            ([m.ctr_logloss for m in models], False),
            ([m.cvr_logloss for m in models], False),
            ([m.latency_p99_ms for m in models], False),
            ([m.avg_ecpm for m in models], True),
            ([m.win_rate for m in models], True),
        ]

        for vals, higher in comparisons:
            if higher:
                best = max(range(len(vals)), key=lambda i: vals[i])
            else:
                best = min(range(len(vals)), key=lambda i: vals[i])
            scores[best] += 1

        score_str = f"  {'综合评分 (7项)':<28}"
        for i, m in enumerate(models):
            score_str += f" {scores[i]:>18}/7"
        logger.info(score_str)

        best_overall = max(range(len(scores)), key=lambda i: scores[i])
        logger.info(f"\n  结论: [{models[best_overall].tag}] {models[best_overall].name} "
                    f"综合表现最优 ({scores[best_overall]}/7)")

        # ── 方案对比总结 ──
        logger.info("")
        logger.info("  " + "=" * (28 + 18 * len(models) + 8))
        logger.info("  方案对比总结:")
        logger.info("  " + "-" * 80)
        logger.info("  [A] 混元 LLM 1.8B + LoRA:")
        logger.info("      优势: 全局注意力建模, 语义理解能力强, 支持生成式检索")
        logger.info("      劣势: 参数量大, 推理延迟高, 需要 GPU 推理优化")
        logger.info("      适用: 精排阶段, 对效果要求极高的场景")
        logger.info("")
        logger.info("  [B] DeepFM:")
        logger.info("      优势: 轻量高效, 推理延迟极低, 工业界成熟方案")
        logger.info("      劣势: 特征交互能力有限, 无序列建模, 无个性化机制")
        logger.info("      适用: 粗排/预排阶段, 对延迟要求极高的场景")
        logger.info("")
        logger.info("  [C] PEPNet:")
        logger.info("      优势: 通过 EPNet/PPNet 实现 embedding/参数级个性化,")
        logger.info("            多任务天然解耦, 延迟接近 DeepFM, 效果接近 LLM")
        logger.info("      劣势: 相比 DeepFM 略复杂, 相比 LLM 语义理解较弱")
        logger.info("      适用: 精排阶段, 效果与延迟的最佳平衡点")
        logger.info("")
        logger.info("  推荐组合方案:")
        logger.info("    召回 → 混元 LLM 语义召回 (STATIC 索引)")
        logger.info("    粗排 → DeepFM (低延迟, 大规模过滤)")
        logger.info("    精排 → PEPNet (个性化多任务, 效果/延迟平衡)")
        logger.info("    重排 → 混元 LLM (精细排序, 语义理解)")
        logger.info("=" * 100)
