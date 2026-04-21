#!/usr/bin/env python3
"""
環境檢查和設置指南
"""

import sys
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_environment():
    """檢查開發環境"""
    
    logger.info("="*60)
    logger.info("環境檢查")
    logger.info("="*60)
    
    # Python 版本
    py_version = sys.version_info
    logger.info(f"✓ Python {py_version.major}.{py_version.minor}.{py_version.micro}")
    
    # PyTorch
    logger.info(f"✓ PyTorch {torch.__version__}")
    
    # CUDA/MPS 支持
    if torch.cuda.is_available():
        logger.info(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        logger.info(f"✓ MPS (Apple Metal) available")
    else:
        logger.warning("⚠ CUDA/MPS not available, will use CPU (slower)")
    
    # 檢查主要包
    packages = [
        "transformers",
        "datasets",
        "tokenizers",
        "accelerate",
        "numpy",
        "tqdm",
        "jieba"
    ]
    
    missing = []
    for package in packages:
        try:
            __import__(package)
            logger.info(f"✓ {package}")
        except ImportError:
            logger.error(f"✗ {package} NOT INSTALLED")
            missing.append(package)
    
    if missing:
        logger.error(f"\n缺少以下包: {', '.join(missing)}")
        logger.error("Please install using: pip install " + " ".join(missing))
        return False
    
    logger.info("\n✓ 所有環境檢查通過！")
    return True


if __name__ == "__main__":
    check_environment()
