"""
數據加載和預處理模塊
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import torch
import numpy as np
from datasets import load_dataset, DatasetDict, concatenate_datasets, Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import jieba

logger = logging.getLogger(__name__)


class TranslationDataLoader:
    """
    加載、預處理中英翻譯數據集
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.data_config = config.get("data_config", {})
        self.model_config = config.get("model_config", {})
        
        self.dataset_name = self.data_config.get("dataset_name")
        self.language_pair = self.data_config.get("language_pair", ["zh", "en"])
        self.max_seq_length = self.data_config.get("max_seq_length", 512)
        self.cache_dir = Path(self.data_config.get("cache_dir", "./data/cache"))
        self.preprocessed_dir = Path(self.data_config.get("preprocessed_dir", "./data/processed"))
        
        # 創建必要目錄
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.preprocessed_dir.mkdir(parents=True, exist_ok=True)
        
        self.tokenizer = None
        self.loaded_dataset_config = None
        self.reversed_language_pair = False
        
    def load_dataset(self) -> DatasetDict:
        """
        從 Hugging Face Hub 加載 OPUS-100 數據集
        """
        logger.info(f"Loading dataset: {self.dataset_name}")
        logger.info(f"Requested language pair: {self.language_pair}")
        
        source_lang, target_lang = self.language_pair
        config_str = f"{source_lang}-{target_lang}"
        
        try:
            dataset = load_dataset(
                self.dataset_name,
                config_str,
                split=None,
                cache_dir=str(self.cache_dir),
                download_mode="reuse_cache_if_exists"
            )
            self.loaded_dataset_config = config_str
            self.reversed_language_pair = False
        except ValueError as e:
            logger.warning(f"Failed to load dataset config '{config_str}': {e}")
            reversed_config_str = f"{target_lang}-{source_lang}"
            logger.info(f"Trying reversed config: {reversed_config_str}")
            try:
                dataset = load_dataset(
                    self.dataset_name,
                    reversed_config_str,
                    split=None,
                    cache_dir=str(self.cache_dir),
                    download_mode="reuse_cache_if_exists"
                )
                self.loaded_dataset_config = reversed_config_str
                self.reversed_language_pair = True
            except Exception as e2:
                logger.error(f"Failed to load dataset with reversed config '{reversed_config_str}': {e2}")
                logger.warning("Retrying without explicit config... this may still fail")
                dataset = load_dataset(
                    self.dataset_name,
                    cache_dir=str(self.cache_dir),
                    download_mode="reuse_cache_if_exists"
                )
        
        logger.info(f"Dataset loaded. Splits: {dataset.keys()}")
        logger.info(f"Loaded dataset config: {self.loaded_dataset_config}")
        return dataset

    def _get_back_translation_device(self) -> torch.device:
        bt_config = self.config.get("back_translation_config", {})
        device_name = bt_config.get("device", "mps")
        if device_name == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        if device_name == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _resolve_back_translation_model(self) -> str:
        bt_config = self.config.get("back_translation_config", {})
        model_name = bt_config.get("model_name")
        if model_name:
            return model_name

        source_lang, target_lang = self.language_pair
        reverse_pair = f"{target_lang}-{source_lang}"
        if reverse_pair == "en-zh":
            return "Helsinki-NLP/opus-mt-en-zh"
        if reverse_pair == "zh-en":
            return "Helsinki-NLP/opus-mt-zh-en"
        return f"Helsinki-NLP/opus-mt-{reverse_pair}"

    def add_back_translation(self, dataset: DatasetDict, split: str = "train") -> DatasetDict:
        """
        對訓練數據集使用後向翻譯增強
        """
        bt_config = self.config.get("back_translation_config", {})
        if not bt_config.get("enabled", False):
            logger.info("Back-translation is disabled in config")
            return dataset

        if split not in dataset:
            logger.warning(f"Back-translation split '{split}' not found in dataset")
            return dataset

        if "translation" not in dataset[split].column_names:
            logger.warning(f"Dataset split '{split}' does not contain 'translation' field")
            return dataset

        logger.info("Starting back-translation augmentation...")
        back_model_name = self._resolve_back_translation_model()
        logger.info(f"Using back-translation model: {back_model_name}")

        device = self._get_back_translation_device()
        logger.info(f"Back-translation generation device: {device}")

        model = AutoModelForSeq2SeqLM.from_pretrained(back_model_name, cache_dir=str(self.cache_dir)).to(device)
        tokenizer = AutoTokenizer.from_pretrained(back_model_name, cache_dir=str(self.cache_dir))

        source_lang, target_lang = self.language_pair
        reverse_pair = f"{target_lang}-{source_lang}"

        max_samples = min(
            bt_config.get("max_samples", len(dataset[split])),
            len(dataset[split])
        )
        batch_size = bt_config.get("batch_size", 16)
        num_beams = bt_config.get("num_beams", 4)

        logger.info(f"Back-translation sample count: {max_samples}")
        logger.info(f"Back-translation batch size: {batch_size}")

        examples = dataset[split].select(range(max_samples))
        target_texts = [sample["translation"][target_lang] for sample in examples]

        synthetic_sources = []
        for i in range(0, len(target_texts), batch_size):
            batch_texts = target_texts[i:i + batch_size]
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_seq_length
            ).to(device)
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=self.max_seq_length,
                    num_beams=num_beams,
                    early_stopping=True,
                )
            batch_sources = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            synthetic_sources.extend(batch_sources)

        synthetic_records = {
            "translation": [
                {source_lang: src_text, target_lang: tgt_text}
                for src_text, tgt_text in zip(synthetic_sources, target_texts)
            ]
        }

        synthetic_dataset = Dataset.from_dict(synthetic_records)
        synthetic_dataset = synthetic_dataset.cast(dataset[split].features)
        logger.info(f"Generated {len(synthetic_dataset)} synthetic back-translation examples")

        augmented = concatenate_datasets([dataset[split], synthetic_dataset])
        dataset[split] = augmented
        logger.info(f"Augmented {split} split size: {len(dataset[split])}")
        return dataset
    
    def initialize_tokenizer(self, tokenizer_name: str = "google-t5/t5-base"):
        """
        初始化分詞器
        """
        logger.info(f"Loading tokenizer: {tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        return self.tokenizer
    
    def preprocess_text(self, text: str, lang: str = "en") -> str:
        """
        文本預處理
        """
        if text is None:
            return ""
        
        text = str(text).strip()
        
        # 語言特定處理
        if lang == "zh":
            # 中文分詞
            tokens = jieba.cut(text, cut_all=False)
            text = " ".join(tokens)
        
        return text
    
    def tokenize_examples(self, examples: Dict, max_length: int = None) -> Dict:
        """
        對輸入和標籤進行分詞
        """
        if max_length is None:
            max_length = self.max_seq_length
        
        source_lang, target_lang = self.language_pair[0], self.language_pair[1]
        
        def _extract_texts(batch: Dict, lang: str) -> List[str]:
            if "translation" in batch:
                texts = [
                    self.preprocess_text(item.get(lang, ""), lang)
                    for item in batch["translation"]
                ]
                return texts
            if lang in batch:
                return [self.preprocess_text(text, lang) for text in batch[lang]]
            return []
        
        source_texts = _extract_texts(examples, source_lang)
        target_texts = _extract_texts(examples, target_lang)
        
        if not source_texts or not target_texts:
            logger.warning(f"Expected keys {source_lang} and {target_lang} not found")
            logger.warning(f"Available keys: {examples.keys()}")
            return examples
        
        source_texts = [
            f"translate {source_lang} to {target_lang}: " + text
            for text in source_texts
        ]
        
        inputs = self.tokenizer(
            source_texts,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors=None
        )
        
        if hasattr(self.tokenizer, "as_target_tokenizer") and callable(self.tokenizer.as_target_tokenizer):
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(
                    target_texts,
                    max_length=max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors=None
                )
        else:
            labels = self.tokenizer(
                target_texts,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors=None
            )
        
        inputs["labels"] = labels["input_ids"]
        return inputs
    
    def prepare_dataset(self, dataset: DatasetDict, num_proc: int = 4) -> DatasetDict:
        """
        準備數據集：分詞、隨機分割
        """
        logger.info("Preprocessing dataset...")
        
        # 分詞
        tokenized_dataset = dataset.map(
            self.tokenize_examples,
            batched=True,
            num_proc=num_proc,
            remove_columns=[col for col in dataset.column_names.get(next(iter(dataset.keys())), []) 
                          if col not in ["input_ids", "attention_mask", "labels"]]
        )
        
        # 隨機分割訓練/驗證集
        train_ratio = self.data_config.get("train_split_ratio", 0.95)
        
        if "train" in tokenized_dataset:
            # 如果已有 train/test 分割
            processed = tokenized_dataset
            if "validation" not in processed:
                split_dataset = processed["train"].train_test_split(
                    test_size=1 - train_ratio,
                    seed=self.config.get("training_config", {}).get("seed", 42)
                )
                processed["train"] = split_dataset["train"]
                processed["validation"] = split_dataset["test"]
        else:
            # 合併所有數據並重新分割
            all_data = concatenate_datasets([tokenized_dataset[k] for k in tokenized_dataset.keys()])
            split_dataset = all_data.train_test_split(
                test_size=1 - train_ratio,
                seed=self.config.get("training_config", {}).get("seed", 42)
            )
            processed = DatasetDict({
                "train": split_dataset["train"],
                "validation": split_dataset["test"]
            })
        
        logger.info(f"Train samples: {len(processed['train'])}")
        logger.info(f"Validation samples: {len(processed['validation'])}")
        
        return processed
    
    def save_preprocessed_dataset(self, dataset: DatasetDict):
        """
        保存預處理後的數據集以加快後續加載
        """
        logger.info(f"Saving preprocessed dataset to {self.preprocessed_dir}")
        dataset.save_to_disk(str(self.preprocessed_dir))
    
    def load_preprocessed_dataset(self) -> Optional[DatasetDict]:
        """
        加載預處理的數據集
        """
        if (self.preprocessed_dir / "train").exists():
            logger.info(f"Loading preprocessed dataset from {self.preprocessed_dir}")
            from datasets import load_from_disk
            return load_from_disk(str(self.preprocessed_dir))
        return None


def create_data_loader(config: Dict) -> TranslationDataLoader:
    """工廠函數：創建數據加載器"""
    return TranslationDataLoader(config)
