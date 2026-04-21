"""
模型定義和實用工具
"""

import logging
from typing import Dict, Optional
import torch
import torch.nn as nn

from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    T5ForConditionalGeneration,
    T5Config
)

logger = logging.getLogger(__name__)


class TranslationModel:
    """翻譯模型包裝類"""
    
    def __init__(self, config: Dict, model_name: str = "google-t5/t5-base", tokenizer=None):
        """
        初始化模型
        
        Args:
            config: 配置字典
            model_name: 預訓練模型名稱或本地路徑
            tokenizer: optional tokenizer to align vocab size when training from scratch
        """
        self.config = config
        self.model_config = config.get("model_config", {})
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.model = None
        self.device = None
        
    def load_pretrained(self):
        """加載預訓練模型"""
        logger.info(f"Loading pretrained model: {self.model_name}")
        
        try:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                cache_dir=self.config.get("data_config", {}).get("cache_dir")
            )
        except Exception as e:
            logger.error(f"Failed to load pretrained model: {e}")
            raise
        
        logger.info(f"Model loaded successfully")
        return self.model
    
    def create_from_config(self):
        """從配置創建新模型（從零開始訓練）"""
        logger.info("Creating model from config (training from scratch)")
        
        # 使用提供的配置創建 T5 模型配置
        vocab_size = self.model_config.get("vocab_size", 32128)
        if self.tokenizer is not None and hasattr(self.tokenizer, "vocab_size"):
            vocab_size = self.tokenizer.vocab_size

        model_config = T5Config(
            vocab_size=vocab_size,
            d_model=self.model_config.get("d_model", 768),
            d_kv=self.model_config.get("d_kv", 64),
            d_ff=self.model_config.get("d_ff", 3072),
            num_layers=self.model_config.get("num_layers", 12),
            num_decoder_layers=self.model_config.get("num_decoder_layers", 12),
            num_heads=self.model_config.get("num_heads", 12),
            relative_attention_num_buckets=self.model_config.get("relative_attention_num_buckets", 32),
            dropout_rate=self.model_config.get("dropout_rate", 0.1),
            layer_norm_epsilon=self.model_config.get("layer_norm_epsilon", 1e-6),
            initializer_factor=self.model_config.get("initializer_factor", 1.0),
            feed_forward_proj=self.model_config.get("feed_forward_proj", "relu"),
            is_encoder_decoder=self.model_config.get("is_encoder_decoder", True),
            decoder_start_token_id=self.model_config.get("decoder_start_token_id", 0),
            pad_token_id=self.model_config.get("pad_token_id", 0),
        )
        
        self.model = T5ForConditionalGeneration(model_config)
        
        logger.info(f"Model created from config")
        logger.info(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        return self.model
    
    def get_model(self):
        """獲取模型"""
        return self.model
    
    def get_num_parameters(self) -> Dict[str, int]:
        """獲取參數統計"""
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable
        }
    
    def print_model_info(self):
        """打印模型信息"""
        if self.model is None:
            logger.warning("Model not loaded yet")
            return
        
        params = self.get_num_parameters()
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Total parameters: {params['total']:,}")
        logger.info(f"Trainable parameters: {params['trainable']:,}")
        logger.info(f"Frozen parameters: {params['frozen']:,}")
        logger.info(f"\nModel architecture:\n{self.model}")


def create_model(config: Dict, from_scratch: bool = False, tokenizer=None) -> TranslationModel:
    """
    工廠函數：創建翻譯模型
    
    Args:
        config: 配置字典
        from_scratch: 是否從零開始訓練（True）還是使用預訓練模型（False）
        tokenizer: 如果從零開始訓練，則用於對齊詞彙表大小的分詞器
    """
    model = TranslationModel(config, tokenizer=tokenizer)
    
    if from_scratch:
        model.create_from_config()
    else:
        model.load_pretrained()
    
    return model
