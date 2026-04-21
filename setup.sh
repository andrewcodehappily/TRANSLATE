#!/bin/bash

# 中英翻譯模型訓練初始化脚本
# Setup script for Chinese-English Translation Model Training

set -e  # 遇到錯誤時退出

echo "🚀 中英翻譯模型訓練環境初始化"
echo "================================"
echo ""

# 1. 檢查 conda 是否安裝
echo "[1/5] 檢查 Conda..."
if ! command -v conda &> /dev/null; then
    echo "❌ Conda 未安裝！"
    echo "請從 https://docs.conda.io/en/latest/miniconda.html 安裝 Miniconda"
    exit 1
fi
echo "✓ Conda 已安裝"

# 2. 創建環境
echo ""
echo "[2/5] 創建 Conda 環境..."
if conda env list | grep -q "translation-zh-en"; then
    echo "⚠ 環境已存在，跳過創建"
else
    conda env create -f environment.yml
    echo "✓ 環境已創建"
fi

# 3. 激活環境
echo ""
echo "[3/5] 激活環境..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate translation-zh-en
echo "✓ 環境已激活: translation-zh-en"

# 4. 創建必要目錄
echo ""
echo "[4/5] 創建目錄結構..."
mkdir -p data/cache
mkdir -p data/processed
mkdir -p models/checkpoints
mkdir -p logss
echo "✓ 目錄已創建"

# 5. 驗證環境
echo ""
echo "[5/5] 驗證環境..."
python scripts/check_env.py

echo ""
echo "✅ 初始化完成！"
echo ""
echo "下一步，運行訓練："
echo "  python scripts/train.py --config config/config.json --from-scratch"
echo ""
