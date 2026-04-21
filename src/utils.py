"""
工具函數集合
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """加載 JSON 配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_config(config: Dict[str, Any], config_path: str):
    """保存 JSON 配置文件"""
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def setup_logging(log_dir: str = "./logs", log_name: str = "training.log"):
    """設置日誌"""
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, log_name)
    
    # 文件處理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # 控制台處理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 日誌格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 根日誌記錄器
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    root_logger.setLevel(logging.INFO)
    
    return log_file


def get_device(device_name: str = "auto"):
    """獲取計算設備"""
    import torch
    
    if device_name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    elif device_name == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    elif device_name == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device("cpu")


def count_parameters(model) -> Dict[str, int]:
    """計算模型參數數量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable
    }


def format_parameters(num_params: int) -> str:
    """格式化參數數量"""
    if num_params >= 1e9:
        return f"{num_params / 1e9:.2f}B"
    elif num_params >= 1e6:
        return f"{num_params / 1e6:.2f}M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.2f}K"
    else:
        return str(num_params)


def print_model_summary(model, model_name: str = "Model"):
    """打印模型摘要"""
    params = count_parameters(model)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"{model_name} Summary")
    logger.info(f"{'='*60}")
    logger.info(f"Total parameters: {format_parameters(params['total'])}")
    logger.info(f"Trainable parameters: {format_parameters(params['trainable'])}")
    logger.info(f"Frozen parameters: {format_parameters(params['frozen'])}")
    logger.info(f"{'='*60}\n")
