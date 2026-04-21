#!/usr/bin/env python3
"""
主訓練腳本：中英翻譯模型訓練
Chinese-English Translation Model Training Script

Usage:
    python scripts/train.py --config config/config.json --from-scratch
    python scripts/train.py --config config/config.json --use-pretrained
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path

# 添加項目路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from transformers import AutoTokenizer

from src.data_loader import create_data_loader
from src.model import create_model
from src.trainer import TranslationTrainer

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """加載配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config


def main():
    parser = argparse.ArgumentParser(
        description="訓練中英翻譯模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  從零開始訓練:
    python scripts/train.py --config config/config.json --from-scratch
  
  使用預訓練模型微調:
    python scripts/train.py --config config/config.json --pretrained google-t5/t5-base
  
  恢復訓練:
    python scripts/train.py --config config/config.json --resume-from models/checkpoints/checkpoint-epoch1-step500
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.json",
        help="配置文件路徑"
    )
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="從零開始訓練新模型"
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="google-t5/t5-base",
        help="預訓練模型名稱或路徑"
    )
    parser.add_argument(
        "--use-accelerate",
        action="store_true",
        help="使用 Accelerate 訓練（自定義訓練循環）"
    )
    parser.add_argument(
        "--use-trainer",
        action="store_true",
        default=True,
        help="使用 Hugging Face Trainer"
    )
    parser.add_argument(
        "--use-backtranslation",
        action="store_true",
        help="對訓練數據使用 back-translation 增強"
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="從檢查點恢復訓練"
    )
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("中英翻譯模型訓練")
    logger.info("="*60)
    
    # 1. 加載配置
    logger.info(f"Loading config from {args.config}")
    config = load_config(args.config)
    logger.info(f"Config loaded: {json.dumps(config, indent=2, ensure_ascii=False)[:200]}...")
    
    # 2. 初始化數據加載器
    logger.info("\n[Step 1] Initializing data loader...")
    data_loader = create_data_loader(config)
    
    # 初始化分詞器
    tokenizer_name = args.pretrained if not args.from_scratch else "google-t5/t5-base"
    tokenizer = data_loader.initialize_tokenizer(tokenizer_name)
    logger.info(f"Tokenizer initialized: {tokenizer_name}")
    
    # 加載數據集
    logger.info("Loading dataset from Hugging Face Hub...")
    logger.info("Note: First run will download a large dataset (~10-50GB depending on language pair)")
    logger.info("      Subsequent runs will use cached data")
    
    dataset = data_loader.load_dataset()
    
    if args.use_backtranslation:
        config.setdefault("back_translation_config", {})["enabled"] = True
        logger.info("Applying back-translation augmentation to raw dataset")
        dataset = data_loader.add_back_translation(dataset, split="train")
        processed_dataset = None
    else:
        processed_dataset = data_loader.load_preprocessed_dataset()
    
    if processed_dataset is None:
        logger.info("Preprocessing dataset...")
        processed_dataset = data_loader.prepare_dataset(dataset)
        logger.info("Saving preprocessed dataset...")
        data_loader.save_preprocessed_dataset(processed_dataset)
    
    train_dataset = processed_dataset["train"]
    eval_dataset = processed_dataset.get("validation", processed_dataset.get("test"))
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Eval dataset size: {len(eval_dataset)}")
    
    # 3. 初始化模型
    logger.info("\n[Step 2] Initializing model...")
    
    if args.resume_from:
        logger.info(f"Resuming from checkpoint: {args.resume_from}")
        from transformers import AutoModelForSeq2SeqLM
        model_obj = AutoModelForSeq2SeqLM.from_pretrained(args.resume_from)
    elif args.from_scratch:
        logger.info("Creating model from scratch (training from zero)...")
        model_obj_wrapper = create_model(config, from_scratch=True, tokenizer=tokenizer)
        model_obj = model_obj_wrapper.get_model()
        model_obj_wrapper.print_model_info()
    else:
        logger.info(f"Loading pretrained model: {args.pretrained}")
        model_obj_wrapper = create_model(config, from_scratch=False)
        model_obj = model_obj_wrapper.get_model()
        model_obj_wrapper.print_model_info()
    
    # 4. 初始化訓練器
    logger.info("\n[Step 3] Initializing trainer...")
    trainer = TranslationTrainer(
        config=config,
        model=model_obj,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )
    
    # 5. 開始訓練
    logger.info("\n[Step 4] Starting training...")
    logger.info(f"Device: {trainer.device}")
    logger.info(f"Batch size: {config['training_config']['batch_size']}")
    logger.info(f"Number of epochs: {config['training_config']['num_epochs']}")
    logger.info(f"Learning rate: {config['training_config']['learning_rate']}")
    
    try:
        if args.use_accelerate:
            logger.info("Using Accelerate backend...")
            trainer.train_with_accelerate()
        else:
            logger.info("Using Hugging Face Trainer backend...")
            trainer.train_with_hugging_face()
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)
    
    # 6. 保存最終模型
    logger.info("\n[Step 5] Saving final model...")
    trainer.save_final_model()
    
    logger.info("\n" + "="*60)
    logger.info("訓練完成！")
    logger.info("="*60)
    logger.info(f"模型已保存到: {trainer.output_dir}")


if __name__ == "__main__":
    main()
