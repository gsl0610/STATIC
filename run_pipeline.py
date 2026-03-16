"""
STATIC-DSP 全流程入口脚本

端到端运行：数据准备 → 语义ID生成 → STATIC索引构建 → 模型训练 → 推理优化 → 竞价

支持两种 backbone:
  --backbone custom   → 原始自定义 Transformer (~1M 参数)
  --backbone hunyuan  → 混元 LLM + LoRA (~1.8B 参数, 可训练 ~20M)

用法:
  python run_pipeline.py --stage all          # 运行全部阶段
  python run_pipeline.py --stage data         # 仅数据准备
  python run_pipeline.py --stage rqvae        # 仅 RQ-VAE 训练
  python run_pipeline.py --stage index        # 仅构建 STATIC 索引
  python run_pipeline.py --stage train        # 仅训练多任务模型
  python run_pipeline.py --stage optimize     # 推理延迟优化 (INT8/TensorRT/torch.compile等)
  python run_pipeline.py --stage benchmark    # 延迟基准测试
  python run_pipeline.py --stage inference    # 仅推理演示
  python run_pipeline.py --stage ab_test      # AB对比测试: 混元LLM vs DeepFM
  python run_pipeline.py --stage model_compare # 三方对比: 混元LLM vs DeepFM vs PEPNet
  python run_pipeline.py --stage qps_benchmark # V100 QPS基准测试 (纯混元架构)
  python run_pipeline.py --backbone hunyuan   # 使用混元 LLM
  python run_pipeline.py --config config/default.yaml  # 指定配置文件
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# 添加项目根目录到 path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.feature_engineering import DSPDatasetBuilder, FeatureEncoder, generate_synthetic_data
from src.data.dataset import create_dataloader
from src.semantic_id.rqvae import RQVAE, RQVAETrainer, create_ad_embeddings
from src.static_index.csr_builder import build_static_index, STATICIndex
from src.model.trainer import MultiTaskTrainer
from src.model.constrained_decoder import STATICConstrainedDecoder
from src.bidding.dsp_engine import DSPBidEngine, BidRequest
from src.model.inference_optimizer import InferenceOptimizer

class _FlushHandler(logging.StreamHandler):
    """每次 emit 后立即 flush，避免 nohup 重定向时日志丢失。"""
    def emit(self, record):
        super().emit(record)
        self.flush()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[_FlushHandler()],
)
logger = logging.getLogger("STATIC-DSP")


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_model(config: dict, vocab_info: dict, device: torch.device):
    """根据 backbone 配置创建模型。"""
    data_cfg = config["data"]
    user_vocab = {name: vocab_info.get(name, 100) for name in data_cfg["user_features"]}
    ad_vocab = {name: vocab_info.get(name, 100) for name in data_cfg["ad_features"]}
    ctx_vocab = {name: vocab_info.get(name, 100) for name in data_cfg["context_features"]}
    stat_vocab = {name: vocab_info.get(name, 20) for name in data_cfg.get("stat_features", [])}

    backbone = config["model"].get("backbone", "custom")

    if backbone == "hunyuan":
        # 混元 LLM backbone
        from src.model.hunyuan_model import create_hunyuan_model
        model = create_hunyuan_model(
            config=config,
            user_feature_vocab_sizes=user_vocab,
            ad_feature_vocab_sizes=ad_vocab,
            context_feature_vocab_sizes=ctx_vocab,
            stat_feature_vocab_sizes=stat_vocab,
            behavior_vocab_size=50000,
        )
    else:
        # 原始自定义 Transformer
        from src.model.transformer import DSPMultiTaskModel
        model = DSPMultiTaskModel(
            config=config,
            user_feature_vocab_sizes=user_vocab,
            ad_feature_vocab_sizes=ad_vocab,
            context_feature_vocab_sizes=ctx_vocab,
            behavior_vocab_size=50000,
        )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"模型创建完成 [{backbone}]: "
        f"总参数 {total_params / 1e6:.1f}M, "
        f"可训练 {trainable_params / 1e6:.1f}M"
    )

    return model


# ================================================================
# Stage 1: 数据准备
# ================================================================
def stage_data(config: dict) -> dict:
    """数据准备阶段：生成模拟数据 → 特征工程 → 数据集划分。"""
    logger.info("=" * 60)
    logger.info("Stage 1: 数据准备")
    logger.info("=" * 60)

    output_dir = config["data"]["processed_dir"]

    # 生成模拟数据（生产环境替换为真实日志）
    df = generate_synthetic_data(num_samples=200000)

    # 构建数据集
    builder = DSPDatasetBuilder(config)
    paths = builder.build_from_dataframe(df, output_dir)

    # 保存特征词表大小信息
    encoder = builder.encoder
    vocab_info = {col: encoder.get_vocab_size(col) for col in encoder.vocab}
    vocab_info_path = os.path.join(output_dir, "vocab_sizes.yaml")
    with open(vocab_info_path, "w") as f:
        yaml.dump(vocab_info, f)

    logger.info(f"数据准备完成: {paths}")
    return {"data_paths": paths, "vocab_info": vocab_info, "encoder": encoder}


# ================================================================
# Stage 2: RQ-VAE 语义ID生成
# ================================================================
def stage_rqvae(config: dict, device: torch.device) -> dict:
    """RQ-VAE 训练阶段：生成广告语义ID。"""
    logger.info("=" * 60)
    logger.info("Stage 2: RQ-VAE 语义ID生成")
    logger.info("=" * 60)

    rq_cfg = config["rqvae"]
    num_ads = 10000  # 模拟广告数

    # 生成广告 embedding（生产环境应来自预训练广告模型）
    ad_embeddings = create_ad_embeddings(num_ads, rq_cfg["input_dim"])
    logger.info(f"广告 embedding shape: {ad_embeddings.shape}")

    # 创建 RQ-VAE 模型
    model = RQVAE(
        input_dim=rq_cfg["input_dim"],
        hidden_dim=rq_cfg["hidden_dim"],
        latent_dim=rq_cfg["latent_dim"],
        num_quantizers=rq_cfg["num_quantizers"],
        codebook_size=rq_cfg["codebook_size"],
        commitment_cost=rq_cfg["commitment_cost"],
        ema_decay=rq_cfg["ema_decay"],
    )

    # 训练
    trainer = RQVAETrainer(model, config)
    trainer.train(ad_embeddings, device)

    # 生成所有广告的语义ID
    semantic_ids = trainer.generate_all_semantic_ids(ad_embeddings, device)

    # 保存
    output_dir = config["data"]["processed_dir"]
    sid_path = os.path.join(output_dir, "ad_semantic_ids.npy")
    np.save(sid_path, semantic_ids)
    logger.info(f"语义ID已保存: {sid_path}, shape={semantic_ids.shape}")

    # 构建语义ID → 广告ID 的映射
    sid_to_ad = {}
    for i, sid in enumerate(semantic_ids):
        sid_to_ad[tuple(sid.tolist())] = i

    return {"semantic_ids": semantic_ids, "sid_to_ad_map": sid_to_ad}


# ================================================================
# Stage 3: STATIC 索引构建
# ================================================================
def stage_index(config: dict, semantic_ids: np.ndarray) -> STATICIndex:
    """构建 STATIC 索引。"""
    logger.info("=" * 60)
    logger.info("Stage 3: STATIC 索引构建")
    logger.info("=" * 60)

    idx_cfg = config["static_index"]

    # 去重 + 排序
    unique_sids = np.unique(semantic_ids, axis=0)
    sorted_indices = np.lexsort(unique_sids[:, ::-1].T)
    sorted_sids = unique_sids[sorted_indices]
    logger.info(f"去重后广告数: {len(sorted_sids)}")

    # 构建索引
    static_index = build_static_index(
        fresh_sids=sorted_sids,
        vocab_size=idx_cfg["vocab_size"],
        dense_lookup_layers=idx_cfg["dense_lookup_layers"],
    )

    # 保存
    output_dir = config["data"]["processed_dir"]
    index_path = os.path.join(output_dir, "static_index.npz")
    static_index.save(index_path)

    logger.info(f"STATIC 索引构建完成: {len(sorted_sids)} 个约束, max_branches={static_index.layer_max_branches}")
    return static_index


# ================================================================
# Stage 4: 多任务模型训练
# ================================================================
def stage_train(config: dict, vocab_info: dict,
                semantic_ids: np.ndarray, device: torch.device,
                resume_checkpoint: str | None = None):
    """多任务模型训练。"""
    logger.info("=" * 60)
    backbone = config["model"].get("backbone", "custom")
    logger.info(f"Stage 4: 多任务模型训练 [backbone={backbone}]")
    logger.info("=" * 60)

    data_cfg = config["data"]
    processed_dir = data_cfg["processed_dir"]

    # 创建模型
    model = create_model(config, vocab_info, device)

    # 从 checkpoint 恢复
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        logger.info(f"从 checkpoint 恢复: {resume_checkpoint}")
        model = MultiTaskTrainer.load_checkpoint(model, resume_checkpoint, device)

    # 创建数据加载器
    train_loader = create_dataloader(
        os.path.join(processed_dir, "train.npz"),
        data_cfg,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=4,
        ad_semantic_ids=semantic_ids,
    )
    val_loader = create_dataloader(
        os.path.join(processed_dir, "val.npz"),
        data_cfg,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=4,
        ad_semantic_ids=semantic_ids,
    )

    # 训练
    trainer = MultiTaskTrainer(model, config, device)
    history = trainer.train(train_loader, val_loader)

    logger.info(f"训练完成, 最终训练损失: {history['train_loss'][-1]:.4f}")
    return model


# ================================================================
# Stage 5: 推理 & 竞价演示
# ================================================================
def stage_inference(config: dict, model, static_index: STATICIndex,
                    sid_to_ad_map: dict[tuple, int],
                    device: torch.device):
    """推理与竞价演示。"""
    logger.info("=" * 60)
    logger.info("Stage 5: 推理 & 竞价演示")
    logger.info("=" * 60)

    # 准备 STATIC 索引张量
    index_tensors = {
        "packed_csr": torch.from_numpy(static_index.packed_csr).int(),
        "csr_indptr": torch.from_numpy(static_index.indptr).int(),
        "start_mask": torch.from_numpy(static_index.start_mask).bool(),
        "dense_mask": torch.from_numpy(static_index.dense_mask).bool(),
        "dense_states": torch.from_numpy(static_index.dense_states).int(),
        "max_branch_factors": static_index.layer_max_branches,
    }

    # 创建约束解码器
    decoder = STATICConstrainedDecoder(model, index_tensors, config, device)

    # 创建竞价引擎
    ad_info = {i: {"advertiser_bid": 0.5 + np.random.random() * 2.0} for i in range(10000)}
    engine = DSPBidEngine(model, decoder, sid_to_ad_map, ad_info, config, device)

    # 模拟竞价请求
    logger.info("模拟竞价请求...")
    rng = np.random.RandomState(123)

    requests = []
    for i in range(5):
        req = BidRequest(
            request_id=f"req_{i:04d}",
            user_features={
                "user_id": rng.randint(2, 100),
                "age_bucket": rng.randint(2, 7),
                "gender": rng.randint(2, 5),
                "city_level": rng.randint(2, 6),
                "device_type": rng.randint(2, 5),
                "os_type": rng.randint(2, 6),
                "interest_tags": list(rng.randint(2, 12, size=3)),
            },
            context_features={
                "hour_of_day": rng.randint(2, 26),
                "day_of_week": rng.randint(2, 9),
                "media_id": rng.randint(2, 50),
                "slot_id": rng.randint(2, 100),
                "slot_type": rng.randint(2, 7),
            },
            behavior_seq=rng.randint(0, 10000, size=rng.randint(5, 20)).tolist(),
            slot_info={"type": "banner", "size": "320x50"},
            floor_price=0.5,
        )
        requests.append(req)

    # 逐个处理
    for req in requests:
        response = engine.process_request(req)
        winner = response.winner
        bid_price = f"{winner.bid_price:.4f}" if winner else "0.0000"
        ecpm = f"{winner.ecpm:.4f}" if winner else "0.0000"
        logger.info(
            f"请求 [{response.request_id}] - "
            f"候选数: {len(response.candidates)}, "
            f"胜出广告: {winner.ad_id if winner else 'None'}, "
            f"出价: {bid_price}, "
            f"eCPM: {ecpm}, "
            f"延迟: {response.latency_ms:.1f}ms"
        )

    # 批量处理
    logger.info("\n批量竞价测试...")
    responses = engine.process_batch(requests)
    for resp in responses:
        logger.info(
            f"请求 [{resp.request_id}] - "
            f"候选数: {len(resp.candidates)}, "
            f"胜出: {resp.winner.ad_id if resp.winner else 'None'}"
        )

    logger.info("推理与竞价演示完成！")


# ================================================================
# Stage 6: 推理延迟优化
# ================================================================
def stage_optimize(config: dict, model, device: torch.device,
                   calibration_loader=None):
    """推理延迟优化：INT8量化 / torch.compile / TensorRT 等。"""
    logger.info("=" * 60)
    logger.info("Stage 6: 推理延迟优化")
    logger.info("=" * 60)

    optimizer = InferenceOptimizer(model, config, device)
    optimized_model = optimizer.optimize(calibration_loader)

    # 输出优化报告
    report = optimizer.get_optimization_report()
    logger.info("\n" + report)

    return optimized_model, optimizer


# ================================================================
# Stage 7: 延迟基准测试
# ================================================================
def stage_benchmark(config: dict, model, device: torch.device,
                    optimizer: InferenceOptimizer = None):
    """延迟基准测试：测量优化前后的推理延迟。"""
    logger.info("=" * 60)
    logger.info("Stage 7: 延迟基准测试")
    logger.info("=" * 60)

    opt_cfg = config.get("optimization", {})
    num_warmup = opt_cfg.get("benchmark_warmup", 10)
    num_runs = opt_cfg.get("benchmark_runs", 100)

    if optimizer is None:
        optimizer = InferenceOptimizer(model, config, device)
        optimizer.optimized_model = model

    result = optimizer.benchmark(num_warmup=num_warmup, num_runs=num_runs)

    # 打印结论
    if result["meets_target"]:
        logger.info(
            f"延迟达标: p99={result['p99_ms']:.2f}ms "
            f"≤ {result['target_ms']}ms"
        )
    else:
        logger.info(
            f"延迟未达标: p99={result['p99_ms']:.2f}ms "
            f"> {result['target_ms']}ms — "
            f"建议启用更多优化方案 (TensorRT/ONNX/CUDA Graph)"
        )

    return result


# ================================================================
# Main
# ================================================================
def main():
    parser = argparse.ArgumentParser(description="STATIC-DSP 全流程运行")
    parser.add_argument("--config", type=str, default="config/default.yaml", help="配置文件路径")
    parser.add_argument(
        "--stage", type=str, default="all",
        choices=["all", "data", "rqvae", "index", "train",
                 "optimize", "benchmark", "inference", "ab_test",
                 "model_compare", "qps_benchmark"],
        help="运行阶段"
    )
    parser.add_argument(
        "--backbone", type=str, default=None,
        choices=["custom", "hunyuan"],
        help="模型 backbone 选择 (覆盖配置文件)"
    )
    parser.add_argument("--device", type=str, default=None, help="计算设备 (cuda/cpu)")
    parser.add_argument(
        "--model-path", type=str, default=None,
        help="混元模型路径 (覆盖配置文件中的 llm.model_name_or_path)"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="从 checkpoint 恢复训练 (如 checkpoints/transformer/best_model.pt)"
    )
    args = parser.parse_args()

    # 加载配置
    config_path = os.path.join(PROJECT_ROOT, args.config)
    config = load_config(config_path)

    # 命令行参数覆盖配置
    if args.backbone:
        config["model"]["backbone"] = args.backbone
    if args.model_path:
        config["llm"]["model_name_or_path"] = args.model_path

    # 设备
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    logger.info(f"Backbone: {config['model'].get('backbone', 'custom')}")

    # 工作目录
    os.chdir(PROJECT_ROOT)
    os.makedirs(config["data"]["processed_dir"], exist_ok=True)

    stage = args.stage
    run_all = (stage == "all")

    # ---------- Stage 1: 数据准备 ----------
    if run_all or stage == "data":
        data_result = stage_data(config)
        vocab_info = data_result["vocab_info"]
    else:
        # 尝试加载已有的词表信息
        vocab_path = os.path.join(config["data"]["processed_dir"], "vocab_sizes.yaml")
        if os.path.exists(vocab_path):
            with open(vocab_path, "r") as f:
                vocab_info = yaml.safe_load(f)
        else:
            vocab_info = {}

    # ---------- Stage 2: RQ-VAE ----------
    if run_all or stage == "rqvae":
        rqvae_result = stage_rqvae(config, device)
        semantic_ids = rqvae_result["semantic_ids"]
        sid_to_ad_map = rqvae_result["sid_to_ad_map"]
    else:
        sid_path = os.path.join(config["data"]["processed_dir"], "ad_semantic_ids.npy")
        if os.path.exists(sid_path):
            semantic_ids = np.load(sid_path)
            sid_to_ad_map = {tuple(sid.tolist()): i for i, sid in enumerate(semantic_ids)}
        else:
            semantic_ids = None
            sid_to_ad_map = {}

    # ---------- Stage 3: STATIC 索引 ----------
    if (run_all or stage == "index") and semantic_ids is not None:
        static_index = stage_index(config, semantic_ids)
    else:
        index_path = os.path.join(config["data"]["processed_dir"], "static_index.npz")
        if os.path.exists(index_path):
            static_index = STATICIndex.load(index_path)
        else:
            static_index = None

    # ---------- Stage 4: 训练 ----------
    if (run_all or stage == "train") and semantic_ids is not None:
        model = stage_train(config, vocab_info, semantic_ids, device,
                            resume_checkpoint=args.resume)
    else:
        model = None

    # 如果跳过训练阶段，尝试从 checkpoint 加载模型
    if model is None and stage in ("optimize", "benchmark", "inference"):
        ckpt_dir = config["training"]["checkpoint_dir"]
        for ckpt_name in ["best_model.pt", "final_model.pt"]:
            ckpt_path = os.path.join(ckpt_dir, ckpt_name)
            if os.path.exists(ckpt_path):
                logger.info(f"从 checkpoint 加载模型: {ckpt_path}")
                # 推理时禁用 LoRA 注入，由 load_checkpoint 加载 LoRA 权重
                infer_config = {**config, "llm": {**config["llm"], "use_lora": False}}
                model = create_model(infer_config, vocab_info, device)
                try:
                    model = MultiTaskTrainer.load_checkpoint(model, ckpt_path, device)
                except RuntimeError as e:
                    if "size mismatch" in str(e):
                        logger.warning(f"Checkpoint 与当前模型结构不兼容 (特征维度变化), "
                                       f"使用随机初始化模型进行 benchmark: {e}")
                    else:
                        raise
                model.to(device)
                model.eval()
                break
        if model is None:
            logger.warning("未找到可用的 checkpoint，跳过后续阶段")

    # ---------- Stage 5: 推理优化 ----------
    optimizer = None
    if (run_all or stage in ("optimize", "benchmark", "inference")) and model is not None:
        # benchmark 和 inference 也需要先做优化
        cal_loader = None
        data_cfg = config["data"]
        processed_dir = data_cfg["processed_dir"]
        val_path = os.path.join(processed_dir, "val.npz")
        if (os.path.exists(val_path) and
                config.get("optimization", {}).get("quantization_method") == "static"):
            cal_loader = create_dataloader(
                val_path, data_cfg,
                batch_size=1, shuffle=False, num_workers=0,
                ad_semantic_ids=semantic_ids,
            )
        model, optimizer = stage_optimize(config, model, device, cal_loader)

    # ---------- Stage 6: 延迟基准测试 ----------
    if (run_all or stage == "benchmark") and model is not None:
        stage_benchmark(config, model, device, optimizer)

    # ---------- Stage 7: 推理 & 竞价 ----------
    if (run_all or stage == "inference") and model is not None and static_index is not None:
        stage_inference(config, model, static_index, sid_to_ad_map, device)

    # ---------- AB 对比测试: 混元 LLM vs DeepFM ----------
    if stage == "ab_test":
        from src.ab_test import ABTestEngine
        ab_engine = ABTestEngine(config, device)
        ab_result = ab_engine.run(vocab_info, semantic_ids)

    # ---------- 三方对比测试: 混元 LLM vs DeepFM vs PEPNet ----------
    if stage == "model_compare":
        from src.model_compare import ModelCompareEngine
        compare_engine = ModelCompareEngine(config, device)
        compare_result = compare_engine.run(vocab_info, semantic_ids)

    # ---------- V100 QPS 基准测试 (纯混元架构) ----------
    if stage == "qps_benchmark":
        from src.qps_benchmark import QPSBenchmarkEngine
        qps_engine = QPSBenchmarkEngine(config, device)
        qps_report = qps_engine.run(vocab_info)

    logger.info("=" * 60)
    logger.info("STATIC-DSP Pipeline 执行完毕！")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
