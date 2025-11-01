"""
训练追问决策分类器（BERT微调）
"""
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np
from tqdm import tqdm
import os

# ========== 配置 ==========
CONFIG = {
    "model_name": "bert-base-chinese",
    "train_data_path": "./data/bert_training_500_v2.json",  # 使用V2版本：真实多句回答
    "output_dir": "./checkpoints/follow_up_classifier",
    "max_length": 256,
    "batch_size": 8,  # 减小batch size避免OOM
    "epochs": 3,  # 减少epochs避免过拟合
    "learning_rate": 2e-5,
    "warmup_ratio": 0.1,
    "num_labels": 2,  # 只有2个标签：FOLLOW_UP和NEXT_TOPIC
    "label_map": {
        "FOLLOW_UP": 0,
        "NEXT_TOPIC": 1
    }
}

# ========== 数据集类 ==========
class FollowUpDataset(Dataset):
    """追问决策数据集"""
    
    def __init__(self, data, tokenizer, max_length, label_map):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_map = label_map
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 构建输入文本：问题 + 回答 + 上下文特征
        text = self._build_input_text(item)
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 标签
        label = self.label_map[item['label']]
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }
    
    def _build_input_text(self, item):
        """构建BERT的输入文本"""
        # 问题 [SEP] 回答 [SEP] 特征
        context = item['context']
        
        features = f"追问深度:{context['follow_up_depth']} " \
                  f"犹豫度:{context['hesitation_score']:.2f} " \
                  f"长度:{context['answer_length']}字 " \
                  f"话题:{context.get('topic', '技术')}"
        
        text = f"{item['question']}[SEP]{item['answer']}[SEP]{features}"
        return text

# ========== 训练函数 ==========
def train_epoch(model, dataloader, optimizer, scheduler, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    predictions = []
    true_labels = []
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        # 数据转移到GPU
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # 前向传播
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        total_loss += loss.item()
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # 记录预测
        preds = torch.argmax(outputs.logits, dim=1)
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
        
        progress_bar.set_postfix({'loss': loss.item()})
    
    # 计算指标
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='weighted')
    
    return avg_loss, accuracy, f1

def evaluate(model, dataloader, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []
    
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
            
            total_loss += outputs.loss.item()
            
            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='weighted')
    
    return avg_loss, accuracy, f1, predictions, true_labels

# ========== 主函数 ==========
def main():
    print("=" * 50)
    print("BERT追问决策分类器微调")
    print("=" * 50)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据
    print("\n1. 加载训练数据...")
    with open(CONFIG['train_data_path'], 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    print(f"总样本数: {len(all_data)}")
    
    # 划分训练集和验证集
    train_data, val_data = train_test_split(all_data, test_size=0.2, random_state=42)
    print(f"训练集: {len(train_data)}, 验证集: {len(val_data)}")
    
    # 加载tokenizer和模型
    print("\n2. 加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    model = AutoModelForSequenceClassification.from_pretrained(
        CONFIG['model_name'],
        num_labels=CONFIG['num_labels']
    )
    model.to(device)
    print(f"模型: {CONFIG['model_name']}")
    
    # 创建数据集
    print("\n3. 创建数据加载器...")
    train_dataset = FollowUpDataset(
        train_data, tokenizer, CONFIG['max_length'], CONFIG['label_map']
    )
    val_dataset = FollowUpDataset(
        val_data, tokenizer, CONFIG['max_length'], CONFIG['label_map']
    )
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'])
    
    # 优化器和调度器
    print("\n4. 配置优化器...")
    optimizer = AdamW(model.parameters(), lr=CONFIG['learning_rate'])
    
    total_steps = len(train_loader) * CONFIG['epochs']
    warmup_steps = int(total_steps * CONFIG['warmup_ratio'])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    
    # 训练
    print("\n5. 开始训练...")
    print("=" * 50)
    
    best_f1 = 0
    
    for epoch in range(CONFIG['epochs']):
        print(f"\nEpoch {epoch + 1}/{CONFIG['epochs']}")
        print("-" * 50)
        
        # 训练
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, optimizer, scheduler, device
        )
        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        
        # 验证
        val_loss, val_acc, val_f1, val_preds, val_labels = evaluate(
            model, val_loader, device
        )
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        
        # 保存最佳模型
        if val_f1 > best_f1:
            best_f1 = val_f1
            print(f"[BEST] 新的最佳F1: {best_f1:.4f}，保存模型...")
            
            os.makedirs(CONFIG['output_dir'], exist_ok=True)
            model.save_pretrained(CONFIG['output_dir'])
            tokenizer.save_pretrained(CONFIG['output_dir'])
    
    # 最终评估
    print("\n" + "=" * 50)
    print("6. 最终评估（使用最佳模型）")
    print("=" * 50)
    
    # 重新加载最佳模型
    model = AutoModelForSequenceClassification.from_pretrained(CONFIG['output_dir'])
    model.to(device)
    
    _, _, _, final_preds, final_labels = evaluate(model, val_loader, device)
    
    # 详细分类报告
    label_names = ['FOLLOW_UP', 'NEXT_TOPIC']
    print("\n分类报告：")
    print(classification_report(final_labels, final_preds, target_names=label_names))
    
    print(f"\n[DONE] 训练完成！模型已保存到: {CONFIG['output_dir']}")
    print(f"最佳验证F1: {best_f1:.4f}")

if __name__ == "__main__":
    main()

