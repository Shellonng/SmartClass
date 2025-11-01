"""
将HuggingFace模型缓存从C盘迁移到E盘
"""
import os
import shutil
from pathlib import Path

# 源目录（C盘）
source_dir = Path("C:/Users/28255/.cache/huggingface")

# 目标目录（E盘）
target_dir = Path("E:/HuggingFace_Cache")

print("="*60)
print("HuggingFace模型缓存迁移工具")
print("="*60)

# 检查源目录
if not source_dir.exists():
    print(f"[ERROR] 源目录不存在: {source_dir}")
    exit(1)

# 计算源目录大小
print(f"\n正在计算源目录大小...")
total_size = sum(f.stat().st_size for f in source_dir.rglob('*') if f.is_file())
print(f"源目录: {source_dir}")
print(f"大小: {total_size / 1024 / 1024 / 1024:.2f} GB")

# 创建目标目录
print(f"\n目标目录: {target_dir}")
target_dir.mkdir(parents=True, exist_ok=True)

# 复制文件
print(f"\n开始迁移（这可能需要几分钟）...")
try:
    # 使用copytree复制整个目录
    if target_dir.exists() and any(target_dir.iterdir()):
        print(f"[WARNING] 目标目录已存在且非空，将合并内容...")
    
    # 复制所有内容
    for item in source_dir.iterdir():
        target_item = target_dir / item.name
        if item.is_dir():
            if target_item.exists():
                print(f"跳过已存在: {item.name}")
            else:
                print(f"复制目录: {item.name}")
                shutil.copytree(item, target_item)
        else:
            if target_item.exists():
                print(f"跳过已存在: {item.name}")
            else:
                print(f"复制文件: {item.name}")
                shutil.copy2(item, target_item)
    
    print(f"\n[OK] 迁移完成！")
    
    # 验证
    print(f"\n正在验证...")
    target_size = sum(f.stat().st_size for f in target_dir.rglob('*') if f.is_file())
    print(f"目标目录大小: {target_size / 1024 / 1024 / 1024:.2f} GB")
    
    if target_size >= total_size * 0.95:  # 允许5%误差
        print(f"\n[VERIFY] 验证成功！")
        print(f"\n现在可以安全删除源目录以释放C盘空间：")
        print(f"源目录: {source_dir}")
        print(f"可释放空间: {total_size / 1024 / 1024 / 1024:.2f} GB")
        print(f"\n建议：")
        print(f"1. 手动检查目标目录: {target_dir}")
        print(f"2. 确认无误后，手动删除源目录（或使用下面的命令）")
        print(f"3. 设置环境变量 HF_HOME={target_dir}")
    else:
        print(f"\n[ERROR] 验证失败！大小不匹配，请检查")
        
except Exception as e:
    print(f"\n[ERROR] 迁移失败: {e}")
    exit(1)

print("\n"+"="*60)
print("迁移完成！")
print("="*60)

