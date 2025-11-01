"""
直接下载RoBERTa并保存到本地
"""
import torch
from transformers import AutoModel, AutoTokenizer
from pathlib import Path
import os

print("=" * 60)
print("下载并转换RoBERTa模型")
print("=" * 60)

model_name = "hfl/chinese-roberta-wwm-ext"
output_dir = Path("./models/chinese-roberta-wwm-ext")

print(f"\n1. 下载模型和tokenizer...")
print(f"模型: {model_name}")

# 忽略torch版本检查，强制下载
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# 下载tokenizer
print("\n  下载tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 下载模型
print("\n  下载模型权重...")
import warnings
warnings.filterwarnings('ignore')

model = AutoModel.from_pretrained(model_name)
print("  [OK] 模型下载成功")

# 保存到本地（使用标准方法）
print(f"\n2. 保存模型到: {output_dir}")
os.makedirs(output_dir, exist_ok=True)

# 使用save_pretrained方法（自动处理所有必要文件）
model.save_pretrained(output_dir, safe_serialization=True)
tokenizer.save_pretrained(output_dir)

print("  [OK] 模型和Tokenizer已保存")

# 检查文件
files = list(output_dir.iterdir())
total_size = sum(f.stat().st_size for f in files if f.is_file()) / 1024 / 1024
print(f"  文件数量: {len(files)}")
print(f"  总大小: {total_size:.2f} MB")

print("\n" + "=" * 60)
print("转换完成！")
print(f"模型位置: {output_dir}")
print("=" * 60)


"""
import torch
from transformers import AutoModel, AutoTokenizer
from pathlib import Path
import os

print("=" * 60)
print("下载并转换RoBERTa模型")
print("=" * 60)

model_name = "hfl/chinese-roberta-wwm-ext"
output_dir = Path("./models/chinese-roberta-wwm-ext")

print(f"\n1. 下载模型和tokenizer...")
print(f"模型: {model_name}")

# 忽略torch版本检查，强制下载
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# 下载tokenizer
print("\n  下载tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 下载模型
print("\n  下载模型权重...")
import warnings
warnings.filterwarnings('ignore')

model = AutoModel.from_pretrained(model_name)
print("  [OK] 模型下载成功")

# 保存到本地（使用标准方法）
print(f"\n2. 保存模型到: {output_dir}")
os.makedirs(output_dir, exist_ok=True)

# 使用save_pretrained方法（自动处理所有必要文件）
model.save_pretrained(output_dir, safe_serialization=True)
tokenizer.save_pretrained(output_dir)

print("  [OK] 模型和Tokenizer已保存")

# 检查文件
files = list(output_dir.iterdir())
total_size = sum(f.stat().st_size for f in files if f.is_file()) / 1024 / 1024
print(f"  文件数量: {len(files)}")
print(f"  总大小: {total_size:.2f} MB")

print("\n" + "=" * 60)
print("转换完成！")
print(f"模型位置: {output_dir}")
print("=" * 60)





