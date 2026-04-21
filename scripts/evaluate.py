#!/usr/bin/env python3
"""
評估和推理腳本
Chinese-English Translation Evaluation and Inference

Usage:
    # 翻譯單個句子
    python scripts/evaluate.py --model models/checkpoint-xxx --translate "你好世界"
    
    # 評估模型
    python scripts/evaluate.py --model models/final_model --eval-dataset data/processed
    
    # 批量翻譯
    python scripts/evaluate.py --model models/final_model --input-file input.txt --output-file output.txt
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_from_disk
from tqdm import tqdm
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TranslationInference:
    """翻譯推理類"""
    
    def __init__(self, model_path: str, device: str = "mps"):
        """
        初始化推理模型
        
        Args:
            model_path: 模型路徑
            device: 計算設備 (mps, cuda, cpu)
        """
        self.model_path = model_path
        self.device = self._setup_device(device)
        
        logger.info(f"Loading model from {model_path}")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Model loaded successfully on {self.device}")
    
    def _setup_device(self, device_name: str) -> torch.device:
        """設置設備"""
        if device_name == "mps":
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        elif device_name == "cuda":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")
        
        logger.info(f"Using device: {device}")
        return device
    
    def translate(
        self,
        texts: list,
        source_lang: str = "zh",
        target_lang: str = "en",
        max_length: int = 512,
        batch_size: int = 32,
        num_beams: int = 4,
        early_stopping: bool = True,
    ) -> list:
        """
        翻譯文本
        
        Args:
            texts: 文本列表或單個文本
            source_lang: 源語言代碼
            target_lang: 目標語言代碼
            max_length: 最大序列長度
            batch_size: 批次大小
            num_beams: Beam search 数量
            early_stopping: 提前停止
        
        Returns:
            翻譯結果列表
        """
        # 將單個文本轉換為列表
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        else:
            single_input = False
        
        translations = []
        
        # 批量處理
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # 準備輸入
            input_texts = [
                f"translate {source_lang} to {target_lang}: {text}"
                for text in batch_texts
            ]
            
            # 分詞
            inputs = self.tokenizer(
                input_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            ).to(self.device)
            
            # 生成
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=max_length,
                    num_beams=num_beams,
                    early_stopping=early_stopping,
                    temperature=0.9,
                    top_k=50,
                    top_p=0.95,
                )
            
            # 解碼
            batch_translations = self.tokenizer.batch_decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            translations.extend(batch_translations)
        
        return translations[0] if single_input else translations
    
    def translate_file(
        self,
        input_file: str,
        output_file: str,
        source_lang: str = "zh",
        target_lang: str = "en",
        batch_size: int = 16,
    ):
        """
        翻譯文件（每行一個句子）
        
        Args:
            input_file: 輸入文件路徑
            output_file: 輸出文件路徑
            source_lang: 源語言
            target_lang: 目標語言
            batch_size: 批次大小
        """
        logger.info(f"Reading input file: {input_file}")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        
        logger.info(f"Found {len(lines)} lines to translate")
        
        logger.info("Translating...")
        with open(output_file, 'w', encoding='utf-8') as out_f:
            for i in range(0, len(lines), batch_size):
                batch = lines[i:i+batch_size]
                translations = self.translate(
                    batch,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    batch_size=batch_size
                )
                
                for translation in translations:
                    out_f.write(translation + '\n')
        
        logger.info(f"Translated output saved to: {output_file}")


class TranslationEvaluator:
    """翻譯評估類"""
    
    def __init__(self, model_path: str, device: str = "mps"):
        """初始化評估器"""
        self.inference = TranslationInference(model_path, device)
        self.model = self.inference.model
        self.tokenizer = self.inference.tokenizer
        self.device = self.inference.device
    
    def evaluate_dataset(
        self,
        dataset_path: str,
        num_samples: int = None,
        compute_metrics: bool = True,
    ):
        """
        評估數據集
        
        Args:
            dataset_path: 數據集路徑
            num_samples: 評估樣本數量（None 表示全部）
            compute_metrics: 是否計算 BLEU 等指標
        """
        logger.info(f"Loading dataset from {dataset_path}")
        dataset = load_from_disk(dataset_path)
        
        if "validation" in dataset:
            eval_dataset = dataset["validation"]
        else:
            eval_dataset = dataset["test"]
        
        if num_samples:
            eval_dataset = eval_dataset.select(range(min(num_samples, len(eval_dataset))))
        
        logger.info(f"Evaluating on {len(eval_dataset)} samples")
        
        # 計算驗證損失
        total_loss = 0
        num_batches = 0
        
        for sample in tqdm(eval_dataset, desc="Computing loss"):
            input_ids = torch.tensor([sample["input_ids"]]).to(self.device)
            labels = torch.tensor([sample["labels"]]).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, labels=labels)
                total_loss += outputs.loss.item()
            
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        logger.info(f"Average validation loss: {avg_loss:.4f}")
        logger.info(f"Perplexity: {np.exp(avg_loss):.4f}")
        
        return {"loss": avg_loss, "perplexity": np.exp(avg_loss)}


def main():
    parser = argparse.ArgumentParser(
        description="翻譯模型評估和推理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  翻譯單個句子:
    python scripts/evaluate.py --model models/final_model --translate "你好世界"
  
  批量翻譯:
    python scripts/evaluate.py --model models/final_model \\
      --input-file test.txt --output-file predictions.txt
  
  評估模型:
    python scripts/evaluate.py --model models/final_model \\
      --eval-dataset data/processed --num-samples 100
        """
    )
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="模型路徑"
    )
    parser.add_argument(
        "--translate",
        type=str,
        help="翻譯單個句子"
    )
    parser.add_argument(
        "--input-file",
        type=str,
        help="輸入文件路徑（每行一個句子）"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        help="輸出文件路徑"
    )
    parser.add_argument(
        "--eval-dataset",
        type=str,
        help="評估數據集路徑"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="評估樣本數量"
    )
    parser.add_argument(
        "--source-lang",
        type=str,
        default="zh",
        help="源語言代碼"
    )
    parser.add_argument(
        "--target-lang",
        type=str,
        default="en",
        help="目標語言代碼"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help="計算設備 (mps, cuda, cpu)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="批次大小"
    )
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("翻譯模型評估和推理")
    logger.info("="*60)
    
    try:
        # 翻譯單個句子
        if args.translate:
            logger.info(f"\nTranslating: {args.translate}")
            inference = TranslationInference(args.model, device=args.device)
            translation = inference.translate(
                args.translate,
                source_lang=args.source_lang,
                target_lang=args.target_lang
            )
            logger.info(f"Translation: {translation}")
        
        # 批量翻譯
        if args.input_file and args.output_file:
            logger.info(f"\nTranslating from {args.input_file} to {args.output_file}")
            inference = TranslationInference(args.model, device=args.device)
            inference.translate_file(
                args.input_file,
                args.output_file,
                source_lang=args.source_lang,
                target_lang=args.target_lang,
                batch_size=args.batch_size
            )
        
        # 評估數據集
        if args.eval_dataset:
            logger.info(f"\nEvaluating dataset: {args.eval_dataset}")
            evaluator = TranslationEvaluator(args.model, device=args.device)
            metrics = evaluator.evaluate_dataset(
                args.eval_dataset,
                num_samples=args.num_samples
            )
            logger.info(f"Metrics: {json.dumps(metrics, indent=2)}")
    
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
