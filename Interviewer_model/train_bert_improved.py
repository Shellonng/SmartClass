#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进版BERT训练 - 解决70%准确率问题
改进点：
1. 添加更丰富的特征（分数、趋势等）
2. 不让模型直接看到轮数
3. 增加训练轮数和early stopping
"""

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import sys

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ========== 配置 ==========
CONFIG = {
    "model_name": "bert-base-chinese",
    "data_path": "./training_data/bert_data.json",
    "output_dir": "./checkpoints/bert_decision_v2",
    "plots_dir": "./plots",
    "max_length": 384,  # 增加长度
    "batch_size": 16,
    "epochs": 10,  # 增加epoch
    "learning_rate": 1e-5,  # 降低学习率
    "warmup_ratio": 0.1,
    "patience": 3,  # early stopping
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# ========== 改进的数据集 ==========
class ImprovedBERTDataset(Dataset):
    """改进的BERT数据集 - 添加更多特征"""
    
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_map = {"FOLLOW_UP": 0, "SWITCH_TOPIC": 1}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 构建改进的输入
        text = self._build_improved_input(item)
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 标签
        label = self.label_map.get(item['action'], 0)
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }
    
    def _build_improved_input(self, item):
        """
        改进的输入构建
        关键改进：不直接暴露轮数，而是通过内容和分数让模型学习
        """
        topic = item.get('topic_name', '技术讨论')
        rounds = item.get('rounds', [])
        
        if not rounds:
            return f"话题:{topic} [SEP] 无对话内容"
        
        # 计算统计特征
        scores = [r.get('roberta_score', 70) for r in rounds]
        avg_score = np.mean(scores)
        min_score = min(scores)
        max_score = max(scores)
        
        # 计算趋势（最近的分数 vs 早期的分数）
        if len(scores) >= 2:
            recent_avg = np.mean(scores[-2:])
            early_avg = np.mean(scores[:2])
            trend = "上升" if recent_avg > early_avg + 5 else "下降" if recent_avg < early_avg - 5 else "稳定"
        else:
            trend = "稳定"
        
        # 回答长度特征
        answer_lengths = [len(r.get('answer', '')) for r in rounds]
        avg_length = np.mean(answer_lengths)
        
        # 构建特征描述（不包含轮数！）
        features = f"[平均分{avg_score:.0f},趋势{trend},分数范围{min_score}-{max_score},平均长度{avg_length:.0f}字]"
        
        # 构建对话内容（只取最近2-3轮，避免模型数轮数）
        recent_rounds = rounds[-3:]  # 只用最近3轮
        
        rounds_text = []
        for i, r in enumerate(recent_rounds):
            q = r.get('question', '')[:100]  # 截断
            a = r.get('answer', '')[:200]
            score = r.get('roberta_score', 70)
            
            # 构建问答对，包含分数
            rounds_text.append(f"Q({score}分): {q} A: {a}")
        
        text = f"{features} 话题:{topic} [SEP] " + " [SEP] ".join(rounds_text)
        return text
    
# ========== 训练函数（与之前相同）==========
def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        logits = outputs.logits
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return avg_loss, accuracy, f1

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return avg_loss, accuracy, f1, all_preds, all_labels

def plot_training_history(train_losses, eval_losses, train_accs, eval_accs, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss图
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, eval_losses, 'r--', label='Eval Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Convergence (Improved)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy图
    ax2.plot(epochs, train_accs, 'b-', label='Train Accuracy', linewidth=2)
    ax2.plot(epochs, eval_accs, 'r--', label='Eval Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Model Accuracy (Improved)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.0])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n[保存] 训练图表: {save_path}")
    plt.close()

# ========== 主训练流程 ==========
def main():
    print("=" * 70)
    print("BERT改进版训练 - 解决70%准确率问题".center(70))
    print("=" * 70)
    print()
    print("改进点:")
    print("  1. ✓ 不直接暴露轮数信息")
    print("  2. ✓ 添加分数趋势特征")
    print("  3. ✓ 只使用最近3轮对话")
    print("  4. ✓ 增加训练epoch到10")
    print("  5. ✓ 添加early stopping")
    print()
    
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    os.makedirs(CONFIG['plots_dir'], exist_ok=True)
    
    device = CONFIG['device']
    print(f"使用设备: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # 加载数据
    print("加载数据...")
    with open(CONFIG['data_path'], 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"总数据量: {len(data)}条\n")
    
    # 划分数据
    train_data, eval_data = train_test_split(data, test_size=0.15, random_state=42)
    print(f"训练集: {len(train_data)}条")
    print(f"验证集: {len(eval_data)}条\n")
    
    # 加载模型
    print("加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    model = AutoModelForSequenceClassification.from_pretrained(
        CONFIG['model_name'],
        num_labels=2,
        use_safetensors=True
    )
    model.to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M\n")
    
    # 创建数据集
    train_dataset = ImprovedBERTDataset(train_data, tokenizer, CONFIG['max_length'])
    eval_dataset = ImprovedBERTDataset(eval_data, tokenizer, CONFIG['max_length'])
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    
    # 优化器
    optimizer = AdamW(model.parameters(), lr=CONFIG['learning_rate'])
    
    total_steps = len(train_loader) * CONFIG['epochs']
    warmup_steps = int(total_steps * CONFIG['warmup_ratio'])
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # 训练
    print("=" * 70)
    print("开始训练")
    print("=" * 70)
    print()
    
    train_losses = []
    eval_losses = []
    train_accs = []
    eval_accs = []
    
    best_eval_acc = 0
    patience_counter = 0
    
    for epoch in range(CONFIG['epochs']):
        print(f"\n[Epoch {epoch + 1}/{CONFIG['epochs']}]")
        print("-" * 70)
        
        # 训练
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, optimizer, scheduler, device
        )
        
        # 验证
        eval_loss, eval_acc, eval_f1, _, _ = evaluate(
            model, eval_loader, device
        )
        
        # 记录
        train_losses.append(train_loss)
        eval_losses.append(eval_loss)
        train_accs.append(train_acc)
        eval_accs.append(eval_acc)
        
        # 打印
        print(f"\n训练 - Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
        print(f"验证 - Loss: {eval_loss:.4f} | Acc: {eval_acc:.4f} | F1: {eval_f1:.4f}")
        
        # Early stopping
        if eval_acc > best_eval_acc:
            best_eval_acc = eval_acc
            patience_counter = 0
            model.save_pretrained(CONFIG['output_dir'])
            tokenizer.save_pretrained(CONFIG['output_dir'])
            print(f"\n✓ 保存最佳模型 (准确率: {eval_acc:.4f})")
        else:
            patience_counter += 1
            print(f"\n○ 验证准确率未提升 (patience: {patience_counter}/{CONFIG['patience']})")
            
            if patience_counter >= CONFIG['patience']:
                print(f"\n[Early Stopping] 在epoch {epoch+1}停止训练")
                break
    
    # 最终评估
    print("\n" + "=" * 70)
    print("最终评估")
    print("=" * 70)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        CONFIG['output_dir'],
        use_safetensors=True
    )
    model.to(device)
    
    _, _, _, final_preds, final_labels = evaluate(model, eval_loader, device)
    
    print("\n分类报告:")
    label_names = ["FOLLOW_UP", "SWITCH_TOPIC"]
    print(classification_report(final_labels, final_preds, target_names=label_names))
    
    # 绘制图表
    plot_path = os.path.join(CONFIG['plots_dir'], 'bert_training_improved.png')
    plot_training_history(train_losses, eval_losses, train_accs, eval_accs, plot_path)
    
    print("\n" + "=" * 70)
    print("训练完成！")
    print("=" * 70)
    print(f"\n模型保存在: {CONFIG['output_dir']}")
    print(f"图表保存在: {plot_path}")
    print(f"最佳验证准确率: {best_eval_acc:.4f}")
    
    if best_eval_acc > 0.80:
        print("\n✓✓ 改进成功！准确率超过80%")
    elif best_eval_acc > 0.75:
        print("\n✓ 有改进，但还需要更多数据")
    else:
        print("\n○ 改进有限，建议补充数据")
    print()

if __name__ == "__main__":
    main()


