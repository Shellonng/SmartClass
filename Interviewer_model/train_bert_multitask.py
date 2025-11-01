#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多任务BERT训练 - 同时训练action分类和guidance生成
任务1: action分类 (FOLLOW_UP / SWITCH_TOPIC)
任务2: guidance生成 (给Qwen的提示文本)
"""

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModel,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
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
    "output_dir": "./checkpoints/bert_multitask",
    "plots_dir": "./plots",
    "max_input_length": 384,
    "max_guidance_length": 128,  # guidance的最大长度
    "batch_size": 8,  # 多任务训练，显存占用更大
    "epochs": 10,
    "learning_rate": 2e-5,
    "warmup_ratio": 0.1,
    "patience": 3,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "task_weights": {
        "action": 1.0,      # action分类的权重
        "guidance": 0.5,    # guidance生成的权重（稍低，因为是辅助任务）
    }
}

# ========== 多任务BERT模型 ==========
class MultiTaskBERT(nn.Module):
    """
    多任务BERT模型
    - 任务1: action分类 (2类)
    - 任务2: guidance生成 (seq2seq)
    """
    
    def __init__(self, model_name, num_labels=2):
        super(MultiTaskBERT, self).__init__()
        
        # 共享的BERT编码器
        self.bert = AutoModel.from_pretrained(model_name, use_safetensors=True)
        hidden_size = self.bert.config.hidden_size
        
        # 任务1: action分类头
        self.action_classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_labels)
        )
        
        # 任务2: guidance生成头（简化为多标签分类，预测每个token）
        # 使用seq2seq解码器
        self.guidance_decoder = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, self.bert.config.vocab_size)
        )
        
    def forward(self, input_ids, attention_mask, guidance_input_ids=None, guidance_labels=None):
        """
        前向传播
        
        Args:
            input_ids: 输入token ids
            attention_mask: 注意力mask
            guidance_input_ids: guidance的输入token ids (用于teacher forcing)
            guidance_labels: guidance的标签
        """
        # BERT编码
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # [CLS] token的表示
        cls_output = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        
        # 任务1: action分类
        action_logits = self.action_classifier(cls_output)  # [batch_size, 2]
        
        # 任务2: guidance生成
        # 使用整个序列的hidden states
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # 简化版：使用CLS表示来生成整个guidance序列
        # 实际更复杂的实现可以用解码器
        guidance_logits = self.guidance_decoder(cls_output)  # [batch_size, vocab_size]
        
        return {
            'action_logits': action_logits,
            'guidance_logits': guidance_logits,
            'sequence_output': sequence_output
        }

# ========== 数据集 ==========
class MultiTaskDataset(Dataset):
    """多任务数据集"""
    
    def __init__(self, data, tokenizer, max_input_length, max_guidance_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_guidance_length = max_guidance_length
        self.label_map = {"FOLLOW_UP": 0, "SWITCH_TOPIC": 1}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 构建输入（改进版，隐藏轮数）
        input_text = self._build_input(item)
        
        # Tokenize输入
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Action标签
        action_label = self.label_map.get(item['action'], 0)
        
        # Guidance文本
        guidance_text = item.get('guidance', '继续追问')
        
        # Tokenize guidance
        guidance_encoding = self.tokenizer(
            guidance_text,
            max_length=self.max_guidance_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': input_encoding['input_ids'].squeeze(0),
            'attention_mask': input_encoding['attention_mask'].squeeze(0),
            'action_labels': torch.tensor(action_label, dtype=torch.long),
            'guidance_input_ids': guidance_encoding['input_ids'].squeeze(0),
            'guidance_labels': guidance_encoding['input_ids'].squeeze(0),  # teacher forcing
        }
    
    def _build_input(self, item):
        """构建改进的输入（隐藏轮数）"""
        topic = item.get('topic_name', '技术讨论')
        rounds = item.get('rounds', [])
        
        if not rounds:
            return f"话题:{topic}"
        
        # 计算特征
        scores = [r.get('roberta_score', 70) for r in rounds]
        avg_score = np.mean(scores)
        
        if len(scores) >= 2:
            recent_avg = np.mean(scores[-2:])
            early_avg = np.mean(scores[:2])
            trend = "上升" if recent_avg > early_avg + 5 else "下降" if recent_avg < early_avg - 5 else "稳定"
        else:
            trend = "稳定"
        
        features = f"[分数{avg_score:.0f},趋势{trend}]"
        
        # 只用最近2-3轮
        recent_rounds = rounds[-3:]
        
        rounds_text = []
        for r in recent_rounds:
            q = r.get('question', '')[:80]
            a = r.get('answer', '')[:150]
            score = r.get('roberta_score', 70)
            rounds_text.append(f"Q({score}分):{q} A:{a}")
        
        text = f"{features} 话题:{topic} [SEP] " + " [SEP] ".join(rounds_text)
        return text

# ========== 训练函数 ==========
def train_epoch(model, dataloader, optimizer, scheduler, device, task_weights):
    """训练一个epoch"""
    model.train()
    
    total_loss = 0
    total_action_loss = 0
    total_guidance_loss = 0
    
    all_action_preds = []
    all_action_labels = []
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    # 损失函数
    action_criterion = nn.CrossEntropyLoss()
    guidance_criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略padding
    
    for batch in progress_bar:
        # 移到设备
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        action_labels = batch['action_labels'].to(device)
        guidance_labels = batch['guidance_labels'].to(device)
        
        # 前向传播
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            guidance_labels=guidance_labels
        )
        
        # 任务1: action分类loss
        action_loss = action_criterion(outputs['action_logits'], action_labels)
        
        # 任务2: guidance生成loss（简化版：只预测第一个token）
        # 更完整的实现需要序列到序列的loss
        guidance_loss = guidance_criterion(
            outputs['guidance_logits'],
            guidance_labels[:, 0]  # 简化：只预测第一个token
        )
        
        # 综合loss
        loss = (task_weights['action'] * action_loss + 
                task_weights['guidance'] * guidance_loss)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        # 记录
        total_loss += loss.item()
        total_action_loss += action_loss.item()
        total_guidance_loss += guidance_loss.item()
        
        # Action准确率
        action_preds = torch.argmax(outputs['action_logits'], dim=1).cpu().numpy()
        all_action_preds.extend(action_preds)
        all_action_labels.extend(action_labels.cpu().numpy())
        
        # 更新进度条
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'act': f'{action_loss.item():.4f}',
            'gui': f'{guidance_loss.item():.4f}'
        })
    
    avg_loss = total_loss / len(dataloader)
    avg_action_loss = total_action_loss / len(dataloader)
    avg_guidance_loss = total_guidance_loss / len(dataloader)
    action_accuracy = accuracy_score(all_action_labels, all_action_preds)
    
    return avg_loss, avg_action_loss, avg_guidance_loss, action_accuracy

def evaluate(model, dataloader, device, task_weights):
    """评估模型"""
    model.eval()
    
    total_loss = 0
    total_action_loss = 0
    total_guidance_loss = 0
    
    all_action_preds = []
    all_action_labels = []
    
    action_criterion = nn.CrossEntropyLoss()
    guidance_criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            action_labels = batch['action_labels'].to(device)
            guidance_labels = batch['guidance_labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                guidance_labels=guidance_labels
            )
            
            action_loss = action_criterion(outputs['action_logits'], action_labels)
            guidance_loss = guidance_criterion(
                outputs['guidance_logits'],
                guidance_labels[:, 0]
            )
            
            loss = (task_weights['action'] * action_loss + 
                    task_weights['guidance'] * guidance_loss)
            
            total_loss += loss.item()
            total_action_loss += action_loss.item()
            total_guidance_loss += guidance_loss.item()
            
            action_preds = torch.argmax(outputs['action_logits'], dim=1).cpu().numpy()
            all_action_preds.extend(action_preds)
            all_action_labels.extend(action_labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    avg_action_loss = total_action_loss / len(dataloader)
    avg_guidance_loss = total_guidance_loss / len(dataloader)
    action_accuracy = accuracy_score(all_action_labels, all_action_preds)
    
    return avg_loss, avg_action_loss, avg_guidance_loss, action_accuracy, all_action_preds, all_action_labels

def plot_training_history(history, save_path):
    """绘制训练历史"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 总Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['eval_loss'], 'r--', label='Eval Loss', linewidth=2)
    axes[0, 0].set_title('Total Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Action分类Loss
    axes[0, 1].plot(epochs, history['train_action_loss'], 'b-', label='Train Action Loss', linewidth=2)
    axes[0, 1].plot(epochs, history['eval_action_loss'], 'r--', label='Eval Action Loss', linewidth=2)
    axes[0, 1].set_title('Action Classification Loss', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Guidance生成Loss
    axes[1, 0].plot(epochs, history['train_guidance_loss'], 'b-', label='Train Guidance Loss', linewidth=2)
    axes[1, 0].plot(epochs, history['eval_guidance_loss'], 'r--', label='Eval Guidance Loss', linewidth=2)
    axes[1, 0].set_title('Guidance Generation Loss', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Action准确率
    axes[1, 1].plot(epochs, history['train_action_acc'], 'b-', label='Train Accuracy', linewidth=2)
    axes[1, 1].plot(epochs, history['eval_action_acc'], 'r--', label='Eval Accuracy', linewidth=2)
    axes[1, 1].set_title('Action Classification Accuracy', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n[保存] 训练图表: {save_path}")
    plt.close()

# ========== 主训练流程 ==========
def main():
    print("=" * 70)
    print("多任务BERT训练 - Action分类 + Guidance生成".center(70))
    print("=" * 70)
    print()
    
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    os.makedirs(CONFIG['plots_dir'], exist_ok=True)
    
    device = CONFIG['device']
    print(f"使用设备: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()
    
    # 加载数据
    print("加载数据...")
    with open(CONFIG['data_path'], 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"总数据量: {len(data)}条")
    
    # 检查guidance字段
    has_guidance = sum(1 for d in data if d.get('guidance'))
    print(f"包含guidance的数据: {has_guidance}条 ({has_guidance/len(data)*100:.1f}%)")
    print()
    
    # 划分数据
    train_data, eval_data = train_test_split(data, test_size=0.15, random_state=42)
    print(f"训练集: {len(train_data)}条")
    print(f"验证集: {len(eval_data)}条\n")
    
    # 加载tokenizer
    print("加载tokenizer和模型...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    
    # 创建多任务模型
    model = MultiTaskBERT(CONFIG['model_name'], num_labels=2)
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params / 1e6:.1f}M")
    print(f"可训练参数: {trainable_params / 1e6:.1f}M\n")
    
    # 创建数据集
    train_dataset = MultiTaskDataset(
        train_data, tokenizer, 
        CONFIG['max_input_length'], 
        CONFIG['max_guidance_length']
    )
    eval_dataset = MultiTaskDataset(
        eval_data, tokenizer,
        CONFIG['max_input_length'],
        CONFIG['max_guidance_length']
    )
    
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
    
    print(f"总训练步数: {total_steps}")
    print(f"Warmup步数: {warmup_steps}\n")
    
    # 训练
    print("=" * 70)
    print("开始训练")
    print("=" * 70)
    print()
    
    history = {
        'train_loss': [],
        'eval_loss': [],
        'train_action_loss': [],
        'eval_action_loss': [],
        'train_guidance_loss': [],
        'eval_guidance_loss': [],
        'train_action_acc': [],
        'eval_action_acc': []
    }
    
    best_eval_acc = 0
    patience_counter = 0
    
    for epoch in range(CONFIG['epochs']):
        print(f"\n[Epoch {epoch + 1}/{CONFIG['epochs']}]")
        print("-" * 70)
        
        # 训练
        train_loss, train_act_loss, train_gui_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, device, CONFIG['task_weights']
        )
        
        # 验证
        eval_loss, eval_act_loss, eval_gui_loss, eval_acc, _, _ = evaluate(
            model, eval_loader, device, CONFIG['task_weights']
        )
        
        # 记录
        history['train_loss'].append(train_loss)
        history['eval_loss'].append(eval_loss)
        history['train_action_loss'].append(train_act_loss)
        history['eval_action_loss'].append(eval_act_loss)
        history['train_guidance_loss'].append(train_gui_loss)
        history['eval_guidance_loss'].append(eval_gui_loss)
        history['train_action_acc'].append(train_acc)
        history['eval_action_acc'].append(eval_acc)
        
        # 打印
        print(f"\n训练:")
        print(f"  总Loss: {train_loss:.4f} | Action Loss: {train_act_loss:.4f} | Guidance Loss: {train_gui_loss:.4f}")
        print(f"  Action准确率: {train_acc:.4f}")
        
        print(f"验证:")
        print(f"  总Loss: {eval_loss:.4f} | Action Loss: {eval_act_loss:.4f} | Guidance Loss: {eval_gui_loss:.4f}")
        print(f"  Action准确率: {eval_acc:.4f}")
        
        # Early stopping
        if eval_acc > best_eval_acc:
            best_eval_acc = eval_acc
            patience_counter = 0
            
            # 保存模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_eval_acc,
            }, os.path.join(CONFIG['output_dir'], 'best_model.pt'))
            
            tokenizer.save_pretrained(CONFIG['output_dir'])
            
            print(f"\n✓ 保存最佳模型 (准确率: {eval_acc:.4f})")
        else:
            patience_counter += 1
            print(f"\n○ 验证准确率未提升 (patience: {patience_counter}/{CONFIG['patience']})")
            
            if patience_counter >= CONFIG['patience']:
                print(f"\n[Early Stopping] 在epoch {epoch+1}停止训练")
                break
    
    # 绘制图表
    plot_path = os.path.join(CONFIG['plots_dir'], 'bert_multitask_training.png')
    plot_training_history(history, plot_path)
    
    print("\n" + "=" * 70)
    print("训练完成！")
    print("=" * 70)
    print(f"\n模型保存在: {CONFIG['output_dir']}")
    print(f"图表保存在: {plot_path}")
    print(f"最佳验证准确率: {best_eval_acc:.4f}")
    
    if best_eval_acc > 0.85:
        print("\n✓✓ 优秀！准确率超过85%")
    elif best_eval_acc > 0.75:
        print("\n✓ 良好！准确率超过75%")
    else:
        print("\n○ 需要改进")
    print()

if __name__ == "__main__":
    main()


