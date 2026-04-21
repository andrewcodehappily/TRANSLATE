#!/usr/bin/env python3
"""
快速啟動腳本：一行命令開始訓練
Quick Start: Setup and begin training in one script
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(cmd, description=""):
    """執行命令並報告結果"""
    if description:
        print(f"\n{'='*60}")
        print(f"📋 {description}")
        print(f"{'='*60}\n")
    
    print(f"$ {cmd}\n")
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print(f"\n❌ 命令失敗: {cmd}")
        sys.exit(1)
    
    return True


def main():
    print("""
╔════════════════════════════════════════════════════════════╗
║  🚀 中英翻譯模型訓練快速啟動                             ║
║  Chinese-English Translation Model Training               ║
╚════════════════════════════════════════════════════════════╝
    """)
    
    # 1. 檢查環境
    print("⏳ 檢查環境...")
    run_command(
        "python scripts/check_env.py",
        "1. 環境驗證"
    )
    
    # 2. 詢問用戶選擇
    print("""
    
🎯 選擇訓練模式:

  [1] 從零開始訓練 (推薦) - 完全自定義的新模型
      ✓ 耗時較長，但完全自主
      ✓ 適合長期項目
      
  [2] 快速開始 - 使用 Google T5 base 模型微調
      ✓ 速度快，質量好
      ✓ 適合快速驗證
    """)
    
    choice = input("請選擇 [1/2] (默認: 1): ").strip() or "1"
    
    if choice == "1":
        mode = "from-scratch"
        description = "從零開始訓練新模型"
    else:
        mode = "use-trainer"
        description = "使用預訓練模型微調"

    back_choice = input("是否啟用 back-translation 增強? [y/N]: ").strip().lower() or "n"
    use_backtranslation = back_choice in ["y", "yes"]
    
    # 3. 確認配置
    print(f"""
    
✅ 訓練配置:
  • 模式: {description}
  • Back-translation: {'已啟用' if use_backtranslation else '未啟用'}
  • 數據集: OPUS-100 (簡中 ↔ 英文)
  • 語言對: zh-en
  • 硬件: M4 Max (MPS 加速)
  • 檢查點: models/checkpoints/
  • 最終模型: models/final_model/

注意:
  • 首次運行會下載語料库 (~10-50GB)
  • 數據會被緩存，後續運行更快
  • 訓練時間取決於數據量和配置
    """)
    
    confirm = input("\n開始訓練? [Y/n]: ").strip().lower()
    
    if confirm in ["n", "no"]:
        print("❌ 取消訓練")
        return
    
    # 4. 開始訓練
    print("\n")
    
    if mode == "from-scratch":
        cmd = "python scripts/train.py --config config/config.json --from-scratch --use-trainer"
    else:
        cmd = "python scripts/train.py --config config/config.json --use-trainer"

    if use_backtranslation:
        cmd += " --use-backtranslation"
    
    run_command(cmd, "🔴 開始訓練")
    
    # 5. 完成
    print("""
    
╔════════════════════════════════════════════════════════════╗
║  ✅ 訓練完成！                                            ║
╚════════════════════════════════════════════════════════════╝

📁 成果:
  • 最終模型: models/final_model/
  • 檢查點: models/checkpoints/
  • 日誌: logs/

🧪 測試模型:
  
  # 翻譯單個句子
  python scripts/evaluate.py \\
    --model models/final_model \\
    --translate "你好世界"
  
  # 批量翻譯
  echo "測試句子" > test.txt
  python scripts/evaluate.py \\
    --model models/final_model \\
    --input-file test.txt \\
    --output-file output.txt

📖 更多信息: 查看 README.md
    """)


if __name__ == "__main__":
    main()
