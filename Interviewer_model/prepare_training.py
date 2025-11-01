#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
准备训练环境 - 检查数据和依赖
"""

import json
import os
import sys

# 确保输出UTF-8
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def check_data():
    """检查训练数据"""
    print("=" * 60)
    print("检查训练数据")
    print("=" * 60)
    
    data_files = {
        'RoBERTa': 'training_data/roberta_data.json',
        'BERT': 'training_data/bert_data.json',
        'Qwen': 'training_data/qwen_data.json'
    }
    
    for model, path in data_files.items():
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(f"✓ {model}: {len(data)} 条数据")
        else:
            print(f"✗ {model}: 文件不存在 ({path})")
    
    print()

def check_dependencies():
    """检查依赖包"""
    print("=" * 60)
    print("检查依赖包")
    print("=" * 60)
    
    required_packages = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'peft': 'PEFT (LoRA)',
        'datasets': 'Datasets',
        'accelerate': 'Accelerate',
        'sklearn': 'Scikit-learn',
        'tqdm': 'TQDM'
    }
    
    missing = []
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} - 未安装")
            missing.append(package)
    
    if missing:
        print(f"\n需要安装: pip install {' '.join(missing)}")
    print()

def check_gpu():
    """检查GPU"""
    print("=" * 60)
    print("检查GPU")
    print("=" * 60)
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ GPU可用")
            print(f"  名称: {torch.cuda.get_device_name(0)}")
            print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("✗ GPU不可用 (将使用CPU训练，速度较慢)")
    except:
        print("✗ 无法检查GPU")
    print()

def check_training_scripts():
    """检查训练脚本"""
    print("=" * 60)
    print("检查训练脚本")
    print("=" * 60)
    
    scripts = {
        'BERT': 'scripts/train_follow_up_classifier.py',
        'RoBERTa': 'scripts/train_roberta_multitask.py',
        'Qwen': 'scripts/train_qwen_lora.py'
    }
    
    for model, path in scripts.items():
        if os.path.exists(path):
            print(f"✓ {model}: {path}")
        else:
            print(f"✗ {model}: 文件不存在 ({path})")
    print()

def show_training_plan():
    """显示训练计划"""
    print("=" * 60)
    print("推荐训练顺序")
    print("=" * 60)
    print("""
1. BERT (最容易，建议先训练)
   - 任务: 二分类 (FOLLOW_UP / SWITCH_TOPIC)
   - 数据: 1,747条 ✓
   - 时间: 1-2小时
   - 预期: 准确率 90%+

2. RoBERTa (中等难度)
   - 任务: 回归评分 + 分类
   - 数据: 1,747条 ✓
   - 时间: 2-3小时
   - 预期: 准确率 85%+

3. Qwen (最复杂)
   - 任务: 条件文本生成
   - 数据: 1,173条 ○
   - 时间: 4-6小时
   - 预期: 可用性 70-80%

总时间: 7-11小时
""")

if __name__ == "__main__":
    print("\n")
    print("=" * 60)
    print("训练准备检查")
    print("=" * 60)
    print()
    
    check_data()
    check_dependencies()
    check_gpu()
    check_training_scripts()
    show_training_plan()
    
    print("=" * 60)
    print("准备完成！")
    print("=" * 60)
    print()
    print("查看详细训练指南: START_TRAINING_NOW.md")
    print()

