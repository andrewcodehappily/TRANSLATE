"""
訓練管理模塊
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import numpy as np
try:
    from evaluate import load as evaluate_load
except ImportError:
    evaluate_load = None
from transformers import (
    Trainer,
    TrainingArguments,
    get_scheduler,
    AutoTokenizer
)
from torch.optim import AdamW
from accelerate import Accelerator
import wandb

logger = logging.getLogger(__name__)


class TranslationTrainer:
    """
    翻譯模型訓練器
    使用 Hugging Face Trainer 或自定義訓練循環
    """
    
    def __init__(self, config: Dict, model, tokenizer, train_dataset, eval_dataset):
        """
        初始化訓練器
        """
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        self.training_config = config.get("training_config", {})
        self.output_dir = Path(config.get("checkpoint_config", {}).get("output_dir", "./models/checkpoints"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = self._setup_device()
        self.accelerator = None
        self.trainer = None
        if evaluate_load is not None:
            try:
                self.bleu_metric = evaluate_load("bleu")
            except Exception as e:
                self.bleu_metric = None
                logger.warning(f"Unable to load BLEU metric: {e}")
        else:
            self.bleu_metric = None
            logger.warning("BLEU metric package not available; falling back to eval_loss")
        
    def _setup_device(self) -> torch.device:
        """設置計算設備"""
        hardware_config = self.config.get("hardware_config", {})
        device_name = hardware_config.get("device", "cpu")
        
        if device_name == "mps":
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        elif device_name == "cuda":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")
        
        logger.info(f"Using device: {device}")
        return device
    
    def setup_training_args(self) -> TrainingArguments:
        """
        設置 Hugging Face TrainingArguments
        """
        logging_config = self.config.get("logging_config", {})
        checkpoint_config = self.config.get("checkpoint_config", {})
        
        args = TrainingArguments(
            output_dir=str(self.output_dir),
            
            # 訓練參數
            num_train_epochs=self.training_config.get("num_epochs", 10),
            per_device_train_batch_size=self.training_config.get("batch_size", 32),
            per_device_eval_batch_size=self.training_config.get("eval_batch_size", 64),
            gradient_accumulation_steps=self.training_config.get("gradient_accumulation_steps", 2),
            
            # 學習率和優化器
            learning_rate=self.training_config.get("learning_rate", 1e-4),
            warmup_steps=self.training_config.get("warmup_steps", 4000),
            weight_decay=self.training_config.get("weight_decay", 0.01),
            max_grad_norm=self.training_config.get("max_grad_norm", 1.0),
            
            # 評估和保存
            eval_strategy="steps",
            eval_steps=self.config.get("evaluation_config", {}).get("eval_steps", 500),
            save_strategy=checkpoint_config.get("save_strategy", "steps"),
            save_steps=checkpoint_config.get("save_steps", 500),
            save_total_limit=checkpoint_config.get("save_total_limit", 3),
            load_best_model_at_end=self.config.get("evaluation_config", {}).get("load_best_model_at_end", True),
            metric_for_best_model=self._resolve_metric_for_best_model(),
            greater_is_better=self.config.get("evaluation_config", {}).get("greater_is_better", False),
            
            # 日誌
            logging_steps=logging_config.get("log_steps", 100),
            logging_dir=logging_config.get("log_dir", "./logs"),
            logging_strategy="steps",
            
            # 分散訓練（如果適用）
            dataloader_pin_memory=(self.training_config.get("pin_memory", True) and self.device.type != "mps"),
            dataloader_num_workers=self.training_config.get("num_workers", 4),
            
            # 精度設置
            bf16=self.training_config.get("mixed_precision", "bf16") == "bf16",
            fp16=self.training_config.get("mixed_precision", "bf16") == "fp16",
            gradient_checkpointing=self.training_config.get("gradient_checkpointing", True),
            
            # 其他
            seed=self.training_config.get("seed", 42),
            push_to_hub=False,
            report_to=["tensorboard"] if logging_config.get("use_tensorboard") else [],
            remove_unused_columns=False,
        )
        
        return args
    
    def _resolve_metric_for_best_model(self):
        metric = self.config.get("evaluation_config", {}).get("metric_for_best_model", "eval_loss")
        if metric == "bleu" and self.bleu_metric is None:
            logger.warning("BLEU metric unavailable; using eval_loss for best model selection")
            return "eval_loss"
        return metric
    
    def compute_metrics(self, eval_pred):
        """
        使用 BLEU 評估指標計算模型性能
        """
        predictions, labels = eval_pred
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]
        references = [[label] for label in decoded_labels]

        bleu_result = self.bleu_metric.compute(predictions=decoded_preds, references=references)
        return {"bleu": bleu_result["score"]}

    def train_with_hugging_face(self):
        """
        使用 Hugging Face Trainer API 訓練
        """
        logger.info("Starting training with Hugging Face Trainer...")
        
        training_args = self.setup_training_args()
        
        # 創建 Trainer
        trainer_kwargs = {
            "model": self.model,
            "args": training_args,
            "train_dataset": self.train_dataset,
            "eval_dataset": self.eval_dataset,
        }
        if self.bleu_metric is not None:
            trainer_kwargs["compute_metrics"] = self.compute_metrics

        self.trainer = Trainer(**trainer_kwargs)
        
        # 開始訓練
        try:
            self.trainer.train()
            logger.info("Training completed successfully!")
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def train_with_accelerate(self):
        """
        使用 Accelerate 進行自定義訓練循環
        適合需要更多控制的場景
        """
        logger.info("Starting training with Accelerate...")
        
        # 初始化 Accelerator
        self.accelerator = Accelerator(
            device_placement=True,
            mixed_precision=self.training_config.get("mixed_precision", "bf16"),
            gradient_accumulation_steps=self.training_config.get("gradient_accumulation_steps", 2)
        )
        
        # 訓練參數
        batch_size = self.training_config.get("batch_size", 32)
        num_epochs = self.training_config.get("num_epochs", 10)
        learning_rate = self.training_config.get("learning_rate", 1e-4)
        
        # 創建數據加載器
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.training_config.get("num_workers", 0),
            pin_memory=self.training_config.get("pin_memory", False)
        )
        
        eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=self.training_config.get("eval_batch_size", 64),
            num_workers=self.training_config.get("num_workers", 0),
            pin_memory=self.training_config.get("pin_memory", False)
        )
        
        # 優化器和調度器
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        num_training_steps = len(train_dataloader) * num_epochs
        
        lr_scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=self.training_config.get("warmup_steps", 0),
            num_training_steps=num_training_steps
        )
        
        # 準備模型、優化器和數據加載器
        self.model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = self.accelerator.prepare(
            self.model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
        )
        
        # 訓練循環
        logger.info(f"Total training steps: {num_training_steps}")
        
        for epoch in range(num_epochs):
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            logger.info(f"{'='*50}")
            
            # 訓練階段
            self.model.train()
            total_loss = 0
            progress_bar = tqdm(train_dataloader, desc="Training")
            
            for step, batch in enumerate(progress_bar):
                # 前向傳播
                outputs = self.model(**batch)
                loss = outputs.loss
                
                # 反向傳播
                self.accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                total_loss += loss.item()
                progress_bar.set_postfix({"loss": loss.item()})
                
                # 定期保存檢查點
                if (step + 1) % self.config.get("checkpoint_config", {}).get("save_steps", 500) == 0:
                    self._save_checkpoint(epoch, step)
            
            # 計算平均損失
            avg_loss = total_loss / len(train_dataloader)
            logger.info(f"Average training loss: {avg_loss:.4f}")
            
            # 評估階段
            logger.info("Running evaluation...")
            eval_loss = self._evaluate(eval_dataloader)
            logger.info(f"Evaluation loss: {eval_loss:.4f}")
        
        logger.info("Training completed!")
    
    def _evaluate(self, eval_dataloader) -> float:
        """評估模型"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                outputs = self.model(**batch)
                loss = outputs.loss
                total_loss += loss.item()
        
        avg_loss = total_loss / len(eval_dataloader)
        return avg_loss
    
    def _save_checkpoint(self, epoch: int, step: int):
        """保存檢查點"""
        checkpoint_dir = self.output_dir / f"checkpoint-epoch{epoch}-step{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        if self.accelerator:
            self.accelerator.save_model(self.model, str(checkpoint_dir))
        else:
            self.model.save_pretrained(str(checkpoint_dir))
        
        self.tokenizer.save_pretrained(str(checkpoint_dir))
        
        logger.info(f"Checkpoint saved to {checkpoint_dir}")
    
    def save_final_model(self):
        """保存最終模型"""
        final_dir = self.output_dir / "final_model"
        final_dir.mkdir(parents=True, exist_ok=True)
        
        if self.accelerator:
            self.accelerator.save_model(self.model, str(final_dir))
        else:
            self.model.save_pretrained(str(final_dir))
        
        self.tokenizer.save_pretrained(str(final_dir))
        
        logger.info(f"Final model saved to {final_dir}")
