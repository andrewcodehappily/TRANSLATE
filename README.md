# 中英翻譯模型訓練系統

## 🚀 概述

這是一個完整的本地訓練系統，用於從零開始訓練中文-英文翻譯模型。

**特點:**
- ✅ 從零開始訓練（不是微調）
- ✅ 使用 OPUS-100 高質量多語言語料庫
- ✅ 支持 Apple M4 Max MPS 加速
- ✅ 模塊化設計，易於自定義
- ✅ 完整的評估和推理功能

---

## 📋 環境準備

### 系統要求

你的電腦配置：
```
Mac Studio M4 Max (2025)
64GB 內存
Apple Silicon (MPS 支持)
```

### 1️⃣ 創建 Conda 環境

```bash
# 安裝環境
conda env create -f environment.yml

# 激活環境
conda activate translation-zh-en

# 驗證環境
python scripts/check_env.py
```

### 2️⃣ 檢查環境

```bash
python scripts/check_env.py
```

應該看到所有包都已安裝。

---

## 📁 項目結構

```
translate/
├── config/
│   └── config.json           # 訓練超參數配置
├── data/
│   ├── cache/                # 下載的原始數據
│   └── processed/            # 預處理後的數據集
├── models/
│   ├── checkpoints/          # 訓練檢查點
│   └── final_model/          # 最終模型
├── scripts/
│   ├── train.py              # 主訓練腳本 🔴 從這裡開始
│   ├── evaluate.py           # 評估和推理
│   └── check_env.py          # 環境檢查
├── src/
│   ├── data_loader.py        # 數據加載和預處理
│   ├── model.py              # 模型架構
│   ├── trainer.py            # 訓練循環
│   └── __init__.py
├── environment.yml           # Conda 環境定義
├── config.json               # 全局配置
└── README.md                 # 本文檔
```

---

## 🎯 快速開始

### 方案 A：從零開始訓練（推薦）

這會創建一個全新的 Transformer 模型，並使用 OPUS-100 數據集訓練。

```bash
# 開始訓練
python scripts/train.py \
    --config config/config.json \
    --from-scratch \
    --use-trainer

# 顯示進度，預計幾小時到幾天取決於數據量和硬件
```

**預期時間:**
- 首次運行：1-2 小時（下載 OPUS-100 語料库）
- 訓練時間：取決於配置的 batch size 和 epochs
  - M4 Max 使用 MPS：較快（相比 CPU）
  - 實際速度取決於數據量和模型大小

### 方案 B：使用預訓練模型微調（快速）

如果你想快速看到結果，也可以微調預訓練模型：

```bash
python scripts/train.py \
    --config config/config.json \
    --pretrained google-t5/t5-base
```

---

## ⚙️ 配置說明

### config/config.json

主要配置項：

```json
{
  "model_config": {
    "d_model": 768,           // 模型隱藏層大小
    "num_layers": 12,         // Transformer 層數
    "num_heads": 12,          // 注意力頭數
    "vocab_size": 50000       // 詞彙表大小
  },
  
  "data_config": {
    "dataset_name": "Helsinki-NLP/opus-100",
    "language_pair": ["zh", "en"],
    "max_seq_length": 512
  },
  
  "training_config": {
    "batch_size": 32,         // 批次大小
    "num_epochs": 10,         // 訓練周期
    "learning_rate": 1e-4,    // 學習率
    "max_grad_norm": 1.0      // 梯度裁剪
  }
}
```

**M4 Max 最適化建議：**

```json
{
  "training_config": {
    "batch_size": 32,              // M4 Max 可以處理
    "gradient_accumulation_steps": 2,  // 累積梯度
    "mixed_precision": "bf16"      // Apple Silicon 優化
  }
}
```

---

## 📊 訓練過程

### 1. 數據下載和預處理

```
第一次運行時：
[1] 下載 OPUS-100 (簡中↔英文) 語料库
    ├─ 約 10-50 GB，取決於語言對
    └─ 保存到 data/cache/
[2] 預處理數據
    ├─ 中文分詞（使用 jieba）
    ├─ 對齐句子對
    └─ 編碼為 token IDs
[3] 保存預處理數據
    └─ 下次運行將直接加載
```

### 2. 模型初始化

**從零開始訓練：**
```
新建 T5-base 架構的模型
├─ 768 維隱層
├─ 12 層 Transformer
├─ 隨機初始化權重
└─ ≈ 223M 參數
```

### 3. 訓練循環

```
每個 Epoch：
├─ Forward pass (前向傳播)
├─ 計算損失函數
├─ Backward pass (反向傳播)
├─ 更新權重
└─ 每 500 步保存檢查點

MPS 加速：
├─ 自動使用 Apple Metal Performance Shaders
├─ 相比 CPU 快 5-10 倍
└─ 不會自動超過內存限制
```

### 4. 評估和保存

```
每 500 步：
├─ 在驗證集上評估
├─ 計算 loss
├─ 保存檢查點

訓練結束：
├─ 保存最佳模型
├─ 保存最終模型
└─ 生成訓練日誌
```

---

## 🔍 推理和評估

### 翻譯單個句子

```bash
python scripts/evaluate.py \
    --model models/final_model \
    --translate "你好世界"

# 輸出:
# Translation: Hello world
```

### 批量翻譯

```bash
# 準備輸入文件（每行一個句子）
cat > input.txt << 'EOF'
你好世界
機器學習很有趣
Python 是最好的編程語言
EOF

# 翻譯
python scripts/evaluate.py \
    --model models/final_model \
    --input-file input.txt \
    --output-file output.txt
```

### 評估模型

```bash
python scripts/evaluate.py \
    --model models/final_model \
    --eval-dataset data/processed \
    --num-samples 1000

# 顯示 loss 和 perplexity
```

---

## 🛠️ 高級用法

### 恢復訓練

```bash
# 從檢查點恢復
python scripts/train.py \
    --config config/config.json \
    --resume-from models/checkpoints/checkpoint-epoch1-step500
```

### 自定義訓練

編輯 `config/config.json` 以調整：

```json
{
  "training_config": {
    "num_epochs": 20,           // 更多周期
    "learning_rate": 5e-5,      // 更小的學習率
    "batch_size": 64,           // 更大的批次
    "warmup_steps": 8000        // 預熱步數
  }
}
```

### 使用 Accelerate 加速（更靈活）

```bash
python scripts/train.py \
    --config config/config.json \
    --from-scratch \
    --use-accelerate
```

### Back-translation 增強

```bash
python scripts/train.py \
    --config config/config.json \
    --from-scratch \
    --use-backtranslation
```

這會利用反向翻譯模型自動生成合成源語句，並將它們加入訓練集以提高翻譯泛化能力。

---

## ⚠️ 常見問題

### Q: 訓練太慢了怎麼辦？

**A:** 調整配置：

```json
{
  "training_config": {
    "batch_size": 16,                    // 減小批次
    "gradient_accumulation_steps": 4,    // 增加梯度累積
    "eval_steps": 1000                   // 減少評估頻率
  }
}
```

### Q: 內存不足？

**A:** 減小 batch size：

```json
{
  "training_config": {
    "batch_size": 8,
    "gradient_checkpointing": true       // 啟用梯度檢查點
  }
}
```

### Q: 如何只用英文數據訓練？

**A:** 修改配置中的語言對：

```json
{
  "data_config": {
    "language_pair": ["en", "en"]
  }
}
```

但更好的做法是在 OPUS-100 中選擇不同的語言對。

### Q: 下載數據集很慢？

**A:** OPUS-100 很大。使用較小的語言對或：

1. 設置更大的超時時間
2. 使用可靠的網絡連接
3. 數據會被緩存，第二次運行會快得多

---

## 📈 訓練監控

### TensorBoard

```bash
# 在訓練過程中
tensorboard --logdir logs/

# 在瀏覽器中打開
open http://localhost:6006
```

### 日誌文件

```
logs/
├── events.out.tfevents.*
└── training_args.bin

models/checkpoints/
├── checkpoint-epoch0-step500/
├── checkpoint-epoch1-step500/
└── final_model/
```

---

## 🎓 了解更多

### 數據集
- [Helsinki-NLP/OPUS-100](https://huggingface.co/datasets/Helsinki-NLP/opus-100)
- 100+ 語言對，高質量翻譯語料

### 模型架構
- [T5: Text-to-Text Transfer Transformer](https://arxiv.org/abs/1910.10683)
- 預訓練和微調在同一框架中

### 相關工具
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Accelerate](https://huggingface.co/docs/accelerate/)
- [Datasets](https://huggingface.co/datasets/)

---

## 🎯 下一步

1. ✅ 完成環境設置
2. 🔴 **開始訓練**: `python scripts/train.py --config config/config.json --from-scratch`
3. 📊 監控訓練進度
4. 🔍 評估模型效果
5. 🚀 部署到生產環境

---

## 📞 技術支持

如遇到問題，檢查：

1. Python 版本 >= 3.8
2. PyTorch 安裝正確
3. MPS 驅動程序更新
4. 足夠的磁盤空間（數據 + 模型 > 100GB）
5. 網絡連接穩定

---

**祝你訓練順利！🚀**
