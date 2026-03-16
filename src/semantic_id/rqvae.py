"""
RQ-VAE (Residual Quantized Variational Autoencoder) - 广告语义ID生成

将广告的高维特征 embedding 量化为离散的语义ID序列，用于生成式检索。

核心流程：
  广告 embedding → Encoder → 连续 latent → RQ量化 → (c1, c2, ..., cL) 语义ID
                                               ↓
                 广告 embedding ← Decoder ← 量化 latent

每个广告被映射为长度为 L 的语义ID，每个位置取值范围 [0, V-1]，
其中 L = num_quantizers, V = codebook_size。
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class VectorQuantizer(nn.Module):
    """单层向量量化器，使用 EMA 更新码本。"""

    def __init__(self, num_embeddings: int, embedding_dim: int,
                 commitment_cost: float = 0.25, ema_decay: float = 0.99):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.ema_decay = ema_decay

        # 码本
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / num_embeddings, 1.0 / num_embeddings)

        # EMA 统计
        self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("ema_dw", self.embedding.weight.clone())

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z: 输入连续向量, shape (B, D)

        Returns:
            quantized: 量化后向量, shape (B, D)
            indices: 码本索引, shape (B,)
            loss: commitment loss
        """
        # 计算距离
        distances = (
            z.pow(2).sum(dim=1, keepdim=True)
            + self.embedding.weight.pow(2).sum(dim=1)
            - 2 * z @ self.embedding.weight.t()
        )

        # 最近邻
        indices = distances.argmin(dim=1)
        quantized = self.embedding(indices)

        # EMA 更新码本
        if self.training:
            encodings = F.one_hot(indices, self.num_embeddings).float()
            self.ema_cluster_size.mul_(self.ema_decay).add_(
                encodings.sum(0), alpha=1 - self.ema_decay
            )
            dw = encodings.t() @ z
            self.ema_dw.mul_(self.ema_decay).add_(dw, alpha=1 - self.ema_decay)

            n = self.ema_cluster_size.sum()
            cluster_size = (
                (self.ema_cluster_size + 1e-5)
                / (n + self.num_embeddings * 1e-5)
                * n
            )
            self.embedding.weight.data.copy_(self.ema_dw / cluster_size.unsqueeze(1))

        # Straight-through estimator
        commitment_loss = self.commitment_cost * F.mse_loss(z, quantized.detach())
        quantized = z + (quantized - z).detach()

        return quantized, indices, commitment_loss


class RQVAE(nn.Module):
    """残差量化 VAE，将广告 embedding 编码为多层语义ID。

    Architecture:
        Input (input_dim) → Encoder → Latent (latent_dim)
              → RQ Layer 1 → residual → RQ Layer 2 → ... → RQ Layer L
        Quantized Latent → Decoder → Reconstruction (input_dim)
    """

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int,
                 num_quantizers: int, codebook_size: int,
                 commitment_cost: float = 0.25, ema_decay: float = 0.99):
        super().__init__()

        self.num_quantizers = num_quantizers
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # 残差量化层
        self.quantizers = nn.ModuleList([
            VectorQuantizer(codebook_size, latent_dim, commitment_cost, ema_decay)
            for _ in range(num_quantizers)
        ])

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        """编码并量化。

        Returns:
            quantized: 量化后的 latent, shape (B, D)
            all_indices: 各层码本索引列表, 每个 shape (B,)
            total_commit_loss: 总 commitment loss
        """
        z = self.encoder(x)
        residual = z
        quantized_sum = torch.zeros_like(z)
        all_indices = []
        total_commit_loss = torch.tensor(0.0, device=x.device)

        for quantizer in self.quantizers:
            quantized, indices, commit_loss = quantizer(residual)
            residual = residual - quantized
            quantized_sum = quantized_sum + quantized
            all_indices.append(indices)
            total_commit_loss = total_commit_loss + commit_loss

        return quantized_sum, all_indices, total_commit_loss

    def decode(self, quantized: torch.Tensor) -> torch.Tensor:
        return self.decoder(quantized)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        quantized, all_indices, commit_loss = self.encode(x)
        x_recon = self.decode(quantized)
        recon_loss = F.mse_loss(x_recon, x)

        # 语义ID: stack 所有层索引 → (B, L)
        semantic_ids = torch.stack(all_indices, dim=1)

        return {
            "recon_loss": recon_loss,
            "commit_loss": commit_loss,
            "total_loss": recon_loss + commit_loss,
            "semantic_ids": semantic_ids,
            "reconstructed": x_recon,
        }

    @torch.no_grad()
    def get_semantic_ids(self, x: torch.Tensor) -> np.ndarray:
        """推理模式：获取输入的语义ID。

        Returns:
            semantic_ids: shape (B, L), numpy array
        """
        self.eval()
        _, all_indices, _ = self.encode(x)
        semantic_ids = torch.stack(all_indices, dim=1)
        return semantic_ids.cpu().numpy()


class RQVAETrainer:
    """RQ-VAE 训练器。"""

    def __init__(self, model: RQVAE, config: dict):
        self.model = model
        self.config = config
        rq_cfg = config["rqvae"]

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=rq_cfg["learning_rate"],
            weight_decay=1e-5,
        )
        self.num_epochs = rq_cfg["num_epochs"]
        self.batch_size = rq_cfg["batch_size"]
        self.checkpoint_dir = Path(rq_cfg["checkpoint_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train(self, ad_embeddings: np.ndarray, device: torch.device) -> RQVAE:
        """训练 RQ-VAE。

        Args:
            ad_embeddings: 广告 embedding 矩阵, shape (num_ads, input_dim)
            device: 训练设备

        Returns:
            训练完成的模型
        """
        self.model.to(device)
        dataset = torch.from_numpy(ad_embeddings).float()
        n = len(dataset)

        logger.info(f"开始训练 RQ-VAE: {n} 个广告, {self.num_epochs} 轮")

        for epoch in range(self.num_epochs):
            self.model.train()
            perm = torch.randperm(n)
            epoch_loss = 0.0
            num_batches = 0

            for i in range(0, n, self.batch_size):
                batch_idx = perm[i:i + self.batch_size]
                batch = dataset[batch_idx].to(device)

                self.optimizer.zero_grad()
                output = self.model(batch)
                loss = output["total_loss"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / max(num_batches, 1)
            logger.info(f"Epoch {epoch + 1}/{self.num_epochs} - Loss: {avg_loss:.6f}")

            if (epoch + 1) % 5 == 0:
                self._save_checkpoint(epoch + 1)

        self._save_checkpoint(self.num_epochs, is_final=True)
        return self.model

    def _save_checkpoint(self, epoch: int, is_final: bool = False):
        suffix = "final" if is_final else f"epoch_{epoch}"
        path = self.checkpoint_dir / f"rqvae_{suffix}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)
        logger.info(f"Checkpoint 已保存: {path}")

    @torch.no_grad()
    def generate_all_semantic_ids(self, ad_embeddings: np.ndarray,
                                  device: torch.device) -> np.ndarray:
        """为所有广告生成语义ID。

        Returns:
            semantic_ids: shape (num_ads, L)
        """
        self.model.eval()
        self.model.to(device)
        dataset = torch.from_numpy(ad_embeddings).float()
        all_ids = []

        for i in range(0, len(dataset), self.batch_size):
            batch = dataset[i:i + self.batch_size].to(device)
            ids = self.model.get_semantic_ids(batch)
            all_ids.append(ids)

        semantic_ids = np.concatenate(all_ids, axis=0)
        logger.info(f"已生成 {len(semantic_ids)} 个广告的语义ID, shape={semantic_ids.shape}")
        return semantic_ids


def create_ad_embeddings(num_ads: int, embed_dim: int) -> np.ndarray:
    """生成模拟的广告 embedding（实际生产中应来自预训练模型）。"""
    rng = np.random.RandomState(42)
    embeddings = rng.randn(num_ads, embed_dim).astype(np.float32)
    # 归一化
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-8)
    return embeddings
