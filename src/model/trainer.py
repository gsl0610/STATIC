"""
多任务模型训练器

联合训练 CTR + CVR + 生成式检索三个任务，使用加权损失。

兼容两种 backbone:
  - DSPMultiTaskModel (原始小型 Transformer)
  - HunyuanDSPModel (混元 LLM + LoRA)

对 LLM backbone 新增:
  - 混合精度训练 (bf16/fp16)
  - 梯度累积
  - LoRA 参数独立保存/加载
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class MultiTaskTrainer:
    """多任务模型训练器。"""

    def __init__(self, model: nn.Module, config: dict[str, Any],
                 device: torch.device):
        self.model = model.to(device)
        self.device = device
        self.config = config

        train_cfg = config["training"]
        self.task_weights = train_cfg["task_weights"]
        self.max_steps = train_cfg["max_steps"]
        self.eval_every = train_cfg["eval_every"]
        self.save_every = train_cfg["save_every"]
        self.gradient_clip = train_cfg["gradient_clip"]
        self.gradient_accumulation_steps = train_cfg.get("gradient_accumulation_steps", 1)
        self.early_stopping_patience = train_cfg.get("early_stopping_patience", 0)

        self.checkpoint_dir = Path(train_cfg["checkpoint_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # 混合精度
        self.use_amp = train_cfg.get("use_amp", False)
        self.amp_dtype = torch.bfloat16 if train_cfg.get("bf16", False) else torch.float16
        self.scaler = torch.amp.GradScaler("cuda", enabled=(self.use_amp and self.amp_dtype == torch.float16))

        # 只优化 requires_grad=True 的参数
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        total_trainable = sum(p.numel() for p in trainable_params)
        logger.info(f"可训练参数: {total_trainable / 1e6:.1f}M")

        # 优化器
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=train_cfg["learning_rate"],
            weight_decay=train_cfg["weight_decay"],
        )

        # 学习率调度器 (线性 warmup + cosine decay)
        self.warmup_steps = train_cfg["warmup_steps"]
        effective_steps = self.max_steps // self.gradient_accumulation_steps
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=train_cfg["learning_rate"],
            total_steps=max(effective_steps, 1),
            pct_start=min(self.warmup_steps / max(effective_steps, 1), 0.3),
            anneal_strategy="cos",
        )

        # 检测模型类型
        self._is_hunyuan = hasattr(model, 'llm')

    def train(self, train_loader: DataLoader,
              val_loader: DataLoader | None = None) -> dict[str, list[float]]:
        """训练循环。

        Returns:
            训练历史 {metric_name: [values]}
        """
        self.model.train()
        history = {"train_loss": [], "ctr_loss": [], "cvr_loss": [], "retrieval_loss": []}
        if val_loader:
            history["val_loss"] = []

        global_step = 0
        epoch = 0
        best_val_loss = float("inf")
        accum_loss = 0.0
        patience_counter = 0

        logger.info(
            f"开始训练: max_steps={self.max_steps}, "
            f"task_weights={self.task_weights}, "
            f"grad_accum={self.gradient_accumulation_steps}, "
            f"amp={self.use_amp} ({self.amp_dtype if self.use_amp else 'off'}), "
            f"backbone={'HunyuanLLM' if self._is_hunyuan else 'CustomTransformer'}"
        )

        while global_step < self.max_steps:
            epoch += 1
            epoch_loss = 0.0
            epoch_batches = 0

            for batch in train_loader:
                if global_step >= self.max_steps:
                    break

                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}

                # 前向传播 (可选混合精度)
                with torch.amp.autocast(
                    device_type="cuda" if self.device.type == "cuda" else "cpu",
                    dtype=self.amp_dtype,
                    enabled=self.use_amp,
                ):
                    target_sids = batch.get("ad_semantic_id", None)
                    output = self.model(batch, target_sids=target_sids)
                    loss, loss_dict = self._compute_loss(output, batch)
                    loss = loss / self.gradient_accumulation_steps

                # 反向传播
                if self.use_amp and self.amp_dtype == torch.float16:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                accum_loss += loss.item()

                # 梯度累积
                if (global_step + 1) % self.gradient_accumulation_steps == 0:
                    if self.gradient_clip > 0:
                        if self.use_amp and self.amp_dtype == torch.float16:
                            self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(
                            [p for p in self.model.parameters() if p.requires_grad],
                            self.gradient_clip
                        )

                    if self.use_amp and self.amp_dtype == torch.float16:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    self.scheduler.step()
                    self.optimizer.zero_grad()

                global_step += 1
                epoch_loss += loss.item() * self.gradient_accumulation_steps
                epoch_batches += 1

                # 日志
                if global_step % 100 == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    avg_loss = accum_loss / min(global_step, 100)
                    log_str = (
                        f"Step {global_step}/{self.max_steps} - "
                        f"Loss: {loss.item() * self.gradient_accumulation_steps:.4f} "
                        f"[CTR: {loss_dict.get('ctr', 0):.4f}, "
                        f"CVR: {loss_dict.get('cvr', 0):.4f}, "
                        f"Ret: {loss_dict.get('retrieval', 0):.4f}] "
                        f"LR: {lr:.2e}"
                    )
                    logger.info(log_str)
                    for h in logging.root.handlers:
                        h.flush()
                    sys.stdout.flush()
                    sys.stderr.flush()

                # 评估
                if val_loader and global_step % self.eval_every == 0:
                    val_loss = self._evaluate(val_loader)
                    history["val_loss"].append(val_loss)
                    logger.info(f"Step {global_step} - Val Loss: {val_loss:.4f}")
                    for h in logging.root.handlers:
                        h.flush()
                    sys.stdout.flush()

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        self._save_checkpoint(global_step, is_best=True)
                    else:
                        patience_counter += 1
                        if self.early_stopping_patience > 0 and patience_counter >= self.early_stopping_patience:
                            logger.info(
                                f"Early stopping: val_loss 连续 {patience_counter} 次未改善 "
                                f"(best={best_val_loss:.4f}, current={val_loss:.4f})"
                            )
                            self._save_checkpoint(global_step, is_final=True)
                            return history

                    self.model.train()

                # 保存检查点
                if global_step % self.save_every == 0:
                    self._save_checkpoint(global_step)

            avg_epoch_loss = epoch_loss / max(epoch_batches, 1)
            history["train_loss"].append(avg_epoch_loss)
            logger.info(f"Epoch {epoch} 完成 - Avg Loss: {avg_epoch_loss:.4f}")

        # 保存最终模型
        self._save_checkpoint(global_step, is_final=True)
        return history

    def _compute_loss(self, output: dict[str, torch.Tensor],
                      batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        """计算加权多任务损失（含 Label Smoothing 防止过拟合）。"""
        total_loss = torch.tensor(0.0, device=self.device)
        loss_dict = {}
        label_smoothing = self.config["training"].get("label_smoothing", 0.0)

        # CTR 损失 (Label Smoothing for BCE)
        if "label_click" in batch and "ctr_logit" in output:
            targets = batch["label_click"].float()
            if label_smoothing > 0:
                targets = targets * (1.0 - label_smoothing) + 0.5 * label_smoothing
            ctr_loss = F.binary_cross_entropy_with_logits(
                output["ctr_logit"].float(), targets
            )
            total_loss = total_loss + self.task_weights["ctr"] * ctr_loss
            loss_dict["ctr"] = ctr_loss.item()

        # CVR 损失 (仅在点击样本上计算，ESMM 思路)
        if "label_conversion" in batch and "cvr_logit" in output:
            click_mask = batch.get("label_click", torch.ones_like(batch["label_conversion"])) > 0.5
            if click_mask.any():
                targets = batch["label_conversion"][click_mask].float()
                if label_smoothing > 0:
                    targets = targets * (1.0 - label_smoothing) + 0.5 * label_smoothing
                cvr_loss = F.binary_cross_entropy_with_logits(
                    output["cvr_logit"][click_mask].float(), targets
                )
            else:
                cvr_loss = torch.tensor(0.0, device=self.device)
            total_loss = total_loss + self.task_weights["cvr"] * cvr_loss
            loss_dict["cvr"] = cvr_loss.item()

        # 生成式检索损失 (交叉熵 + Label Smoothing)
        if "retrieval_logits" in output and "ad_semantic_id" in batch:
            logits = output["retrieval_logits"].float()  # (B, L, V)
            targets = batch["ad_semantic_id"].long()  # (B, L)
            B, L, V = logits.shape
            retrieval_loss = F.cross_entropy(
                logits.reshape(B * L, V), targets.reshape(B * L),
                label_smoothing=label_smoothing,
            )
            total_loss = total_loss + self.task_weights["retrieval"] * retrieval_loss
            loss_dict["retrieval"] = retrieval_loss.item()

        return total_loss, loss_dict

    @torch.no_grad()
    def _evaluate(self, val_loader: DataLoader) -> float:
        """评估。"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in val_loader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            target_sids = batch.get("ad_semantic_id", None)

            with torch.amp.autocast(
                device_type="cuda" if self.device.type == "cuda" else "cpu",
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                output = self.model(batch, target_sids=target_sids)
                loss, _ = self._compute_loss(output, batch)

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def _save_checkpoint(self, step: int, is_best: bool = False, is_final: bool = False):
        if is_best:
            path = self.checkpoint_dir / "best_model.pt"
        elif is_final:
            path = self.checkpoint_dir / "final_model.pt"
        else:
            path = self.checkpoint_dir / f"checkpoint_step_{step}.pt"

        save_dict = {
            "step": step,
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

        if self._is_hunyuan:
            # 对于 LLM 模型，分开保存 LoRA 和任务头
            # LoRA 权重 (如果有 peft)
            try:
                lora_path = self.checkpoint_dir / f"lora_{'best' if is_best else 'final' if is_final else f'step_{step}'}"
                self.model.llm.save_pretrained(str(lora_path))
                save_dict["lora_path"] = str(lora_path)
                logger.info(f"LoRA 权重已保存: {lora_path}")
            except Exception as e:
                logger.warning(f"LoRA 保存失败 (可能未使用 peft): {e}")

            # 任务头和投影层的权重
            task_state = {}
            for name, param in self.model.named_parameters():
                if not name.startswith("llm."):
                    task_state[name] = param.data.cpu()
            save_dict["task_head_state_dict"] = task_state
        else:
            save_dict["model_state_dict"] = self.model.state_dict()

        torch.save(save_dict, path)
        logger.info(f"模型已保存: {path}")

    @staticmethod
    def load_checkpoint(model: nn.Module, path: str,
                        device: torch.device) -> nn.Module:
        """加载模型检查点。"""
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        if "model_state_dict" in checkpoint:
            # 原始模型: 完整 state_dict
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "task_head_state_dict" in checkpoint:
            # LLM 模型: 加载任务头
            task_state = checkpoint["task_head_state_dict"]
            model_state = model.state_dict()
            for name, param in task_state.items():
                if name in model_state:
                    model_state[name] = param
            model.load_state_dict(model_state, strict=False)

            # 加载 LoRA (如果有)
            lora_path = checkpoint.get("lora_path")
            if lora_path:
                try:
                    from peft import PeftModel
                    model.llm = PeftModel.from_pretrained(model.llm, lora_path)
                    logger.info(f"LoRA 权重已加载: {lora_path}")
                except Exception as e:
                    logger.warning(f"LoRA 加载失败: {e}")

        logger.info(f"模型已加载: {path}, step={checkpoint.get('step', 'N/A')}")
        return model
