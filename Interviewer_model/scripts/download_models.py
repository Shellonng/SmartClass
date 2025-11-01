"""
Model Download Script
模型下载脚本 - 预先下载所有需要的模型
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
import whisper


def download_models():
    """下载所有需要的模型"""
    
    print("=" * 60)
    print("AI Interview Coach - 模型下载")
    print("=" * 60)
    
    # 1. 下载Qwen-1.8B-Chat (仅下载不加载)
    print("\n[1/5] Downloading Qwen-1.8B-Chat...")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id="Qwen/Qwen-1_8B-Chat",
            cache_dir="./cache",
            resume_download=True
        )
        print("[OK] Qwen-1.8B-Chat downloaded")
    except Exception as e:
        print(f"[FAIL] Download failed: {e}")
    
    # 2. 下载Whisper
    print("\n[2/5] Downloading Whisper-medium...")
    try:
        whisper.load_model("medium")
        print("[OK] Whisper-medium downloaded")
    except Exception as e:
        print(f"[FAIL] Download failed: {e}")
    
    # 3. 下载SentenceTransformer
    print("\n[3/5] Downloading SentenceTransformer...")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            cache_dir="./cache",
            resume_download=True
        )
        print("[OK] SentenceTransformer downloaded")
    except Exception as e:
        print(f"[FAIL] Download failed: {e}")
    
    # 4. 下载BERT
    print("\n[4/5] Downloading BERT-base-chinese...")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id="google-bert/bert-base-chinese",
            cache_dir="./cache",
            resume_download=True
        )
        print("[OK] BERT-base-chinese downloaded")
    except Exception as e:
        print(f"[FAIL] Download failed: {e}")
    
    # 5. 下载RoBERTa
    print("\n[5/5] Downloading RoBERTa-wwm-ext...")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id="hfl/chinese-roberta-wwm-ext",
            cache_dir="./cache",
            resume_download=True
        )
        print("[OK] RoBERTa-wwm-ext downloaded")
    except Exception as e:
        print(f"[FAIL] Download failed: {e}")
    
    print("\n" + "=" * 60)
    print("模型下载完成！")
    print("=" * 60)


if __name__ == "__main__":
    download_models()

