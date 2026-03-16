"""
V100 GPU QPS 基准测试 — 混元 LLM 纯架构验证

在 V100 (16GB / 32GB) 上系统测试混元模型在不同优化组合下的 QPS 表现。
验证纯混元架构 (召回 + 排序均使用混元模型) 能否满足工业级 QPS 要求。

测试矩阵:
  ┌──────────────────────┬───────────────────────────────────────────┐
  │  优化级别             │ 组合                                      │
  │──────────────────────│───────────────────────────────────────────│
  │  L0: 原始 (baseline) │ FP32, 32层, 无优化                        │
  │  L1: FP16 半精度      │ + FP16 (V100 Tensor Core)                │
  │  L2: + 层裁剪         │ + 32→6层 (减少 75% 计算)                  │
  │  L3: + 合并前向       │ + MergedForwardWrapper                   │
  │  L4: 全套优化         │ L1+L2+L3 组合                             │
  └──────────────────────┴───────────────────────────────────────────┘

  × batch_size = [1, 2, 4, 8, 16, 32]

用法:
  python run_pipeline.py --stage qps_benchmark --device cuda
  python run_pipeline.py --stage qps_benchmark --device cpu  # CPU 对照

输出:
  - 分优化级别 × 分 batch_size 的延迟/QPS 矩阵
  - GPU 显存使用
  - 最终结论: 推荐部署配置

V100 特性说明:
  - SM 7.0 架构, 不支持 Flash Attention 2 (需要 SM 8.0+)
  - FP16 Tensor Core 120 TFLOPS, FP32 15.7 TFLOPS
  - INT8 Tensor Core 不如 A10/A100 成熟, 故不测 INT8
  - 显存带宽 900 GB/s, 适合大 batch 推理
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ================================================================
# 数据类
# ================================================================
@dataclass
class BenchmarkResult:
    """单组测试结果。"""
    opt_level: str
    batch_size: int
    num_runs: int

    # 延迟 (ms)
    latency_mean_ms: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p90_ms: float = 0.0
    latency_p99_ms: float = 0.0
    latency_min_ms: float = 0.0
    latency_max_ms: float = 0.0

    # 吞吐
    throughput_qps: float = 0.0  # batch_size * 1000 / latency_mean_ms

    # 显存 (MB)
    gpu_mem_allocated_mb: float = 0.0
    gpu_mem_reserved_mb: float = 0.0

    # 模型信息
    model_params_m: float = 0.0
    optimizations: list = field(default_factory=list)


@dataclass
class QPSBenchmarkReport:
    """完整的 QPS 基准测试报告。"""
    device_name: str
    gpu_memory_gb: float
    results: list[BenchmarkResult] = field(default_factory=list)
    test_date: str = ""


# ================================================================
# V100 QPS 基准测试引擎
# ================================================================
class QPSBenchmarkEngine:
    """V100 QPS 基准测试引擎。"""

    # V100 不支持 INT8 Tensor Core (不像 A10/A100)，所以不测 INT8
    OPT_LEVELS = [
        ("L0_baseline", "原始 FP32 (无优化)"),
        ("L1_fp16", "FP16 半精度 (Tensor Core)"),
        ("L2_fp16_prune", "FP16 + 层裁剪 (32→6层)"),
        ("L3_fp16_prune_merged", "FP16 + 层裁剪 + 合并前向"),
    ]

    BATCH_SIZES = [1, 2, 4, 8, 16, 32]

    def __init__(self, config: dict, device: torch.device):
        self.config = config
        self.device = device
        self.num_warmup = config.get("optimization", {}).get("benchmark_warmup", 30)
        self.num_runs = config.get("optimization", {}).get("benchmark_runs", 200)

    def run(self, vocab_info: dict) -> QPSBenchmarkReport:
        """执行完整的 QPS 基准测试。"""
        logger.info("=" * 80)
        logger.info("  V100 QPS 基准测试 — 混元 LLM 纯架构验证")
        logger.info("=" * 80)

        # GPU 信息
        device_name = "CPU"
        gpu_mem_gb = 0.0
        if self.device.type == "cuda":
            device_name = torch.cuda.get_device_name(self.device)
            gpu_mem_gb = torch.cuda.get_device_properties(
                self.device
            ).total_memory / (1024 ** 3)
            logger.info(f"  GPU: {device_name}")
            logger.info(f"  显存: {gpu_mem_gb:.1f} GB")
            logger.info(f"  CUDA: {torch.version.cuda}")
        else:
            logger.info(f"  设备: CPU")
        logger.info(f"  PyTorch: {torch.__version__}")
        logger.info(f"  预热: {self.num_warmup} 次, 测试: {self.num_runs} 次")
        logger.info("")

        report = QPSBenchmarkReport(
            device_name=device_name,
            gpu_memory_gb=gpu_mem_gb,
        )

        # 遍历每个优化级别
        for opt_key, opt_desc in self.OPT_LEVELS:
            logger.info(f"{'─' * 70}")
            logger.info(f"  优化级别: [{opt_key}] {opt_desc}")
            logger.info(f"{'─' * 70}")

            # 创建模型 (每个优化级别重新创建，避免状态干扰)
            model, applied_opts = self._create_optimized_model(opt_key)
            if model is None:
                logger.warning(f"  [{opt_key}] 模型创建失败，跳过")
                continue

            model_params = sum(p.numel() for p in model.parameters()) / 1e6

            # 遍历每个 batch size
            for bs in self.BATCH_SIZES:
                try:
                    result = self._benchmark_single(
                        model, opt_key, bs, model_params, applied_opts, vocab_info
                    )
                    report.results.append(result)
                    logger.info(
                        f"    batch={bs:>3}: "
                        f"latency={result.latency_mean_ms:>8.2f}ms "
                        f"(P99={result.latency_p99_ms:>8.2f}ms)  "
                        f"QPS={result.throughput_qps:>8.0f}"
                        + (f"  GPU mem={result.gpu_mem_allocated_mb:.0f}MB"
                           if result.gpu_mem_allocated_mb > 0 else "")
                    )
                except torch.cuda.OutOfMemoryError:
                    logger.warning(f"    batch={bs}: OOM (显存不足), 跳过更大 batch")
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()
                    break
                except Exception as e:
                    logger.warning(f"    batch={bs}: 测试失败 ({e})")
                    break

            # 释放模型显存
            del model
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        # 打印汇总报告
        self._print_report(report)

        return report

    def _create_optimized_model(self, opt_key: str
                                ) -> tuple[nn.Module | None, list[str]]:
        """根据优化级别创建模型。"""
        from src.model.hunyuan_model import create_hunyuan_model
        from src.model.inference_optimizer import MergedForwardWrapper

        data_cfg = self.config["data"]
        vocab_path = f"{data_cfg['processed_dir']}/vocab_sizes.yaml"

        # 词表信息
        vocab_info = {}
        try:
            import yaml
            if __import__("os").path.exists(vocab_path):
                with open(vocab_path, "r") as f:
                    vocab_info = yaml.safe_load(f) or {}
        except Exception:
            pass

        user_vocab = {n: vocab_info.get(n, 100) for n in data_cfg["user_features"]}
        ad_vocab = {n: vocab_info.get(n, 100) for n in data_cfg["ad_features"]}
        ctx_vocab = {n: vocab_info.get(n, 100) for n in data_cfg["context_features"]}
        stat_vocab = {n: vocab_info.get(n, 20) for n in data_cfg.get("stat_features", [])}

        applied = []

        try:
            # 创建基础模型 (禁用 LoRA 注入以简化推理)
            infer_config = {
                **self.config,
                "llm": {**self.config["llm"], "use_lora": False},
            }
            model = create_hunyuan_model(
                config=infer_config,
                user_feature_vocab_sizes=user_vocab,
                ad_feature_vocab_sizes=ad_vocab,
                context_feature_vocab_sizes=ctx_vocab,
                stat_feature_vocab_sizes=stat_vocab,
                behavior_vocab_size=50000,
            )
            model.to(self.device)
            model.eval()

            # 应用优化
            if opt_key in ("L1_fp16", "L2_fp16_prune", "L3_fp16_prune_merged"):
                if self.device.type == "cuda":
                    model = model.half()
                    applied.append("fp16")

            if opt_key in ("L2_fp16_prune", "L3_fp16_prune_merged"):
                num_layers = self.config.get("optimization", {}).get(
                    "num_inference_layers", 6
                )
                model = self._prune_model_layers(model, num_layers)
                applied.append(f"layer_prune_{num_layers}")

            if opt_key == "L3_fp16_prune_merged":
                model = MergedForwardWrapper(model)
                applied.append("merged_forward")

            return model, applied

        except Exception as e:
            logger.error(f"模型创建失败: {e}")
            return None, []

    def _prune_model_layers(self, model: nn.Module, num_layers: int) -> nn.Module:
        """裁剪 LLM 层数。"""
        if hasattr(model, 'llm'):
            llm = model.llm
            base = llm
            if hasattr(llm, 'base_model'):
                base = llm.base_model
            if hasattr(base, 'model'):
                base = base.model
            inner = base.model if hasattr(base, 'model') else base
            if hasattr(inner, 'layers'):
                total = len(inner.layers)
                if num_layers < total:
                    inner.layers = inner.layers[:num_layers]
                    logger.info(f"    层裁剪: {total} → {num_layers} 层")
        return model

    def _make_sample_input(self, vocab_info: dict,
                           batch_size: int = 1) -> dict[str, torch.Tensor]:
        """构造基准测试样本输入。"""
        data_cfg = self.config["data"]
        device = self.device
        bs = batch_size

        multi_value_feats = {"interest_tags", "interest_tags_l2", "creative_label"}
        batch = {}

        for feat in data_cfg["user_features"]:
            vsize = vocab_info.get(feat, 100)
            if feat in multi_value_feats:
                batch[f"user_{feat}"] = torch.randint(
                    2, max(3, vsize), (bs, 5), device=device
                )
            else:
                batch[f"user_{feat}"] = torch.randint(
                    2, max(3, vsize), (bs,), device=device
                )

        for feat in data_cfg["ad_features"]:
            vsize = vocab_info.get(feat, 100)
            if feat in multi_value_feats:
                batch[f"ad_{feat}"] = torch.randint(
                    2, max(3, vsize), (bs, 4), device=device
                )
            else:
                batch[f"ad_{feat}"] = torch.randint(
                    2, max(3, vsize), (bs,), device=device
                )

        for feat in data_cfg["context_features"]:
            vsize = vocab_info.get(feat, 50)
            batch[f"ctx_{feat}"] = torch.randint(
                2, max(3, vsize), (bs,), device=device
            )

        for feat in data_cfg.get("stat_features", []):
            vsize = vocab_info.get(feat, 20)
            batch[f"stat_{feat}"] = torch.randint(
                2, max(3, vsize), (bs,), device=device
            )

        max_seq = data_cfg["behavior"]["max_seq_len"]
        batch["behavior_seq"] = torch.randint(
            0, 50000, (bs, max_seq), device=device
        )

        return batch

    @torch.inference_mode()
    def _benchmark_single(self, model: nn.Module, opt_key: str,
                          batch_size: int, model_params_m: float,
                          optimizations: list, vocab_info: dict
                          ) -> BenchmarkResult:
        """对单个 (优化级别, batch_size) 组合做基准测试。"""
        from src.model.inference_optimizer import MergedForwardWrapper

        sample = self._make_sample_input(vocab_info, batch_size)
        use_cuda = self.device.type == "cuda"

        # 确定前向函数
        if isinstance(model, MergedForwardWrapper):
            forward_fn = lambda s: model.merged_inference(s)
        else:
            forward_fn = lambda s: model(s)

        # Warmup
        for _ in range(self.num_warmup):
            forward_fn(sample)
        if use_cuda:
            torch.cuda.synchronize()

        # 显存快照 (warmup 后)
        gpu_mem_alloc = 0.0
        gpu_mem_reserved = 0.0
        if use_cuda:
            gpu_mem_alloc = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
            gpu_mem_reserved = torch.cuda.memory_reserved(self.device) / (1024 ** 2)

        # 测量
        latencies = []
        for _ in range(self.num_runs):
            if use_cuda:
                se = torch.cuda.Event(enable_timing=True)
                ee = torch.cuda.Event(enable_timing=True)
                se.record()
                forward_fn(sample)
                ee.record()
                torch.cuda.synchronize()
                latencies.append(se.elapsed_time(ee))
            else:
                t0 = time.perf_counter()
                forward_fn(sample)
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000)

        lat = np.array(latencies)
        mean_ms = float(np.mean(lat))

        return BenchmarkResult(
            opt_level=opt_key,
            batch_size=batch_size,
            num_runs=self.num_runs,
            latency_mean_ms=mean_ms,
            latency_p50_ms=float(np.percentile(lat, 50)),
            latency_p90_ms=float(np.percentile(lat, 90)),
            latency_p99_ms=float(np.percentile(lat, 99)),
            latency_min_ms=float(np.min(lat)),
            latency_max_ms=float(np.max(lat)),
            throughput_qps=batch_size * 1000.0 / max(mean_ms, 0.001),
            gpu_mem_allocated_mb=gpu_mem_alloc,
            gpu_mem_reserved_mb=gpu_mem_reserved,
            model_params_m=model_params_m,
            optimizations=optimizations,
        )

    def _print_report(self, report: QPSBenchmarkReport):
        """打印汇总报告。"""
        logger.info("")
        logger.info("=" * 90)
        logger.info("  V100 QPS 基准测试报告 — 混元 LLM 纯架构 (召回+排序)")
        logger.info("=" * 90)
        logger.info(f"  GPU: {report.device_name}  ({report.gpu_memory_gb:.1f} GB)")
        logger.info(f"  模型: 混元 LLM ({self.config['llm']['model_name_or_path']})")
        logger.info(f"  行为序列长度: {self.config['data']['behavior']['max_seq_len']}")
        logger.info(f"  层裁剪配置: {self.config.get('optimization', {}).get('num_inference_layers', 'off')}")
        logger.info("")

        # ── QPS 矩阵表 ──
        logger.info("  ┌─ QPS 矩阵 (行: 优化级别, 列: batch_size) ─────────────────────┐")
        logger.info("")

        # 表头
        bs_list = sorted(set(r.batch_size for r in report.results))
        header = f"  {'优化级别':<30}"
        for bs in bs_list:
            header += f" {'bs=' + str(bs):>10}"
        logger.info(header)
        logger.info("  " + "─" * (30 + 11 * len(bs_list)))

        # 数据行
        for opt_key, opt_desc in self.OPT_LEVELS:
            row = f"  {opt_desc:<30}"
            for bs in bs_list:
                matches = [r for r in report.results
                           if r.opt_level == opt_key and r.batch_size == bs]
                if matches:
                    qps = matches[0].throughput_qps
                    row += f" {qps:>10.0f}"
                else:
                    row += f" {'—':>10}"
            logger.info(row)

        logger.info("")

        # ── 延迟详情表 ──
        logger.info("  ┌─ 延迟详情 (batch=1, 单请求延迟) ──────────────────────────────┐")
        logger.info("")
        logger.info(f"  {'优化级别':<30} {'Mean(ms)':>10} {'P50(ms)':>10} "
                    f"{'P90(ms)':>10} {'P99(ms)':>10} {'QPS':>10}")
        logger.info("  " + "─" * 80)

        for opt_key, opt_desc in self.OPT_LEVELS:
            matches = [r for r in report.results
                       if r.opt_level == opt_key and r.batch_size == 1]
            if matches:
                r = matches[0]
                logger.info(
                    f"  {opt_desc:<30} {r.latency_mean_ms:>10.2f} "
                    f"{r.latency_p50_ms:>10.2f} {r.latency_p90_ms:>10.2f} "
                    f"{r.latency_p99_ms:>10.2f} {r.throughput_qps:>10.0f}"
                )

        logger.info("")

        # ── 显存使用 ──
        if any(r.gpu_mem_allocated_mb > 0 for r in report.results):
            logger.info("  ┌─ GPU 显存使用 (batch=1 时) ─────────────────────────────────┐")
            logger.info("")
            logger.info(f"  {'优化级别':<30} {'已分配(MB)':>12} {'已预留(MB)':>12} {'参数量(M)':>12}")
            logger.info("  " + "─" * 66)

            for opt_key, opt_desc in self.OPT_LEVELS:
                matches = [r for r in report.results
                           if r.opt_level == opt_key and r.batch_size == 1]
                if matches:
                    r = matches[0]
                    logger.info(
                        f"  {opt_desc:<30} {r.gpu_mem_allocated_mb:>12.0f} "
                        f"{r.gpu_mem_reserved_mb:>12.0f} {r.model_params_m:>12.1f}"
                    )

            logger.info("")

        # ── 加速比分析 ──
        baseline_bs1 = [r for r in report.results
                        if r.opt_level == "L0_baseline" and r.batch_size == 1]
        if baseline_bs1:
            base_qps = baseline_bs1[0].throughput_qps
            base_lat = baseline_bs1[0].latency_mean_ms

            logger.info("  ┌─ 加速比 (相对于 L0 baseline batch=1) ─────────────────────┐")
            logger.info("")

            for opt_key, opt_desc in self.OPT_LEVELS:
                for bs in bs_list:
                    matches = [r for r in report.results
                               if r.opt_level == opt_key and r.batch_size == bs]
                    if matches:
                        r = matches[0]
                        speedup = r.throughput_qps / max(base_qps, 0.01)
                        lat_reduction = (1 - r.latency_mean_ms / max(base_lat, 0.01)) * 100
                        if bs == 1:
                            logger.info(
                                f"    {opt_desc:<30} bs={bs:>2}: "
                                f"QPS={r.throughput_qps:>8.0f} "
                                f"({speedup:>5.1f}x)  "
                                f"延迟={r.latency_mean_ms:>7.2f}ms "
                                f"({lat_reduction:>+6.1f}%)"
                            )
                        else:
                            logger.info(
                                f"    {'':30} bs={bs:>2}: "
                                f"QPS={r.throughput_qps:>8.0f} "
                                f"({speedup:>5.1f}x)"
                            )

            logger.info("")

        # ── 最佳配置 & 结论 ──
        if report.results:
            best = max(report.results, key=lambda r: r.throughput_qps)
            best_bs1 = max(
                [r for r in report.results if r.batch_size == 1],
                key=lambda r: r.throughput_qps,
                default=None,
            )

            logger.info("  ┌─ 结论 & 部署建议 ──────────────────────────────────────────┐")
            logger.info("")
            logger.info(f"    最佳 QPS (单请求): "
                        f"[{best_bs1.opt_level}] bs=1 → QPS={best_bs1.throughput_qps:.0f}"
                        if best_bs1 else "    N/A")
            logger.info(f"    最佳 QPS (批处理): "
                        f"[{best.opt_level}] bs={best.batch_size} → "
                        f"QPS={best.throughput_qps:.0f}")
            logger.info("")

            # V100 特定建议
            logger.info("    V100 部署建议:")
            logger.info("    ─────────────")
            logger.info("    1. 必选: FP16 + 层裁剪(6层) + 合并前向 → 单请求延迟大幅下降")
            logger.info("    2. 推荐: 动态批处理 (batch=8~16) → GPU 利用率从 <10% 提升到 50%+")
            logger.info("    3. 注意: V100 不支持 Flash Attention 2 (需 SM80+), 已配置 SDPA")
            logger.info("    4. 注意: V100 INT8 Tensor Core 效率不如 A10/A100, 建议用 FP16")
            logger.info("    5. 进阶: ONNX Runtime + FP16 可进一步加速 10-20%")
            logger.info("")

            # 目标达成判断
            target_ms = self.config.get("optimization", {}).get("target_latency_ms", 20.0)
            if best_bs1:
                if best_bs1.latency_p99_ms <= target_ms:
                    logger.info(f"    ✅ 延迟目标达成: P99={best_bs1.latency_p99_ms:.2f}ms "
                                f"≤ {target_ms}ms")
                else:
                    logger.info(f"    ⚠️  延迟目标未达成: P99={best_bs1.latency_p99_ms:.2f}ms "
                                f"> {target_ms}ms")
                    logger.info(f"       建议: 进一步裁剪层数 / 减少 behavior_seq_len / "
                                f"启用 ONNX Runtime")

            logger.info("")

            # 工业级 QPS 评估
            logger.info("    工业级 QPS 评估:")
            logger.info("    ─────────────────")
            tiers = [
                (3000, "信息流广告 (高并发)"),
                (1000, "品牌广告 / 搜索广告"),
                (500, "展示广告 (中等并发)"),
                (100, "低并发场景"),
            ]
            for threshold, scenario in tiers:
                matches = [
                    r for r in report.results if r.throughput_qps >= threshold
                ]
                if matches:
                    best_match = min(matches, key=lambda r: r.batch_size)
                    logger.info(
                        f"    ✅ {scenario}: "
                        f"QPS ≥ {threshold} → [{best_match.opt_level}] "
                        f"bs={best_match.batch_size}"
                    )
                else:
                    logger.info(
                        f"    ❌ {scenario}: QPS ≥ {threshold} → "
                        f"需要多卡或进一步优化"
                    )

            logger.info("")

            # 多卡扩展预估
            if best:
                logger.info("    多卡扩展预估 (DataParallel):")
                for n_gpu in [2, 4, 8]:
                    est_qps = best.throughput_qps * n_gpu * 0.9  # 90% 线性效率
                    logger.info(
                        f"      {n_gpu}x V100: QPS ≈ {est_qps:.0f} "
                        f"(基于 {best.opt_level} bs={best.batch_size})"
                    )

        logger.info("")
        logger.info("=" * 90)
