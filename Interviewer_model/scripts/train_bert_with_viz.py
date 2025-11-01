"""
BERT追问决策分类器训练（带完整可视化）
实时显示：
1. 训练集损失 vs 验证集损失（检测过拟合）
2. 训练集准确率 vs 验证集准确率
3. 训练集F1 vs 验证集F1
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
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# ========== 配置 ==========
CONFIG = {
    "model_name": "bert-base-chinese",
    "train_data_path": "./data/bert_training_1500.json",  # 使用1500条数据
    "output_dir": "./checkpoints/follow_up_classifier_1500",
    "max_length": 256,
    "batch_size": 8,
    "epochs": 5,  # 增加到5个epoch观察过拟合
    "learning_rate": 2e-5,
    "warmup_ratio": 0.1,
    "num_labels": 2,
    "label_map": {
        "FOLLOW_UP": 0,
        "NEXT_TOPIC": 1
    }
}

# ========== 全局可视化数据 ==========
viz_data = {
    "epoch": [],
    "train_loss": [],
    "val_loss": [],
    "train_acc": [],
    "val_acc": [],
    "train_f1": [],
    "val_f1": []
}

# ========== 数据集类 ==========
class FollowUpDataset(Dataset):
    """追问决策数据集"""
    
    def __init__(self, data, tokenizer, label_map, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = self._build_input_text(item)
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        label = self.label_map[item['label']]
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }
    
    def _build_input_text(self, item):
        """构建BERT的输入文本"""
        context = item['context']
        
        features = f"追问深度:{context['follow_up_depth']} " \
                  f"犹豫度:{context['hesitation_score']:.2f} " \
                  f"长度:{context['answer_length']}字 " \
                  f"话题:{context.get('topic', '技术')}"
        
        text = f"{item['question']}[SEP]{item['answer']}[SEP]{features}"
        return text

# ========== 训练和评估函数 ==========
def train_epoch(model, dataloader, optimizer, scheduler, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        # 预测
        preds = torch.argmax(outputs.logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return avg_loss, accuracy, f1

def evaluate(model, dataloader, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating")
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
            total_loss += loss.item()
            
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return avg_loss, accuracy, f1, all_preds, all_labels

# ========== 可视化函数 ==========
def setup_visualization():
    """设置可视化窗口"""
    plt.ion()  # 开启交互模式
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('BERT追问决策模型训练监控', fontsize=16, fontweight='bold')
    
    return fig, axes

def update_visualization(fig, axes, epoch):
    """更新可视化图表"""
    # 清空所有子图
    for ax in axes.flat:
        ax.clear()
    
    epochs = viz_data["epoch"]
    
    # 子图1: 损失曲线
    ax1 = axes[0, 0]
    ax1.plot(epochs, viz_data["train_loss"], 'b-o', label='训练集损失', linewidth=2)
    ax1.plot(epochs, viz_data["val_loss"], 'r-s', label='验证集损失', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('损失曲线（检测过拟合）', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 标注过拟合警告
    if len(epochs) >= 2:
        if viz_data["val_loss"][-1] > viz_data["val_loss"][-2]:
            ax1.text(0.5, 0.95, '⚠️ 验证集损失上升，可能过拟合', 
                    transform=ax1.transAxes, ha='center', va='top',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
                    fontsize=10)
    
    # 子图2: 准确率曲线
    ax2 = axes[0, 1]
    ax2.plot(epochs, viz_data["train_acc"], 'b-o', label='训练集准确率', linewidth=2)
    ax2.plot(epochs, viz_data["val_acc"], 'r-s', label='验证集准确率', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('准确率曲线', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    # 子图3: F1分数曲线
    ax3 = axes[1, 0]
    ax3.plot(epochs, viz_data["train_f1"], 'b-o', label='训练集F1', linewidth=2)
    ax3.plot(epochs, viz_data["val_f1"], 'r-s', label='验证集F1', linewidth=2)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('F1 Score', fontsize=12)
    ax3.set_title('F1分数曲线', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1])
    
    # 子图4: 训练/验证差距分析
    ax4 = axes[1, 1]
    if len(epochs) > 0:
        loss_gap = [t - v for t, v in zip(viz_data["train_loss"], viz_data["val_loss"])]
        acc_gap = [v - t for t, v in zip(viz_data["train_acc"], viz_data["val_acc"])]
        f1_gap = [v - t for t, v in zip(viz_data["train_f1"], viz_data["val_f1"])]
        
        ax4.plot(epochs, loss_gap, 'g-o', label='损失差距(训练-验证)', linewidth=2)
        ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax4.set_xlabel('Epoch', fontsize=12)
        ax4.set_ylabel('Gap', fontsize=12)
        ax4.set_title('过拟合/欠拟合分析', fontsize=14, fontweight='bold')
        ax4.legend(fontsize=11)
        ax4.grid(True, alpha=0.3)
        
        # 添加分析文本
        latest_loss_gap = loss_gap[-1]
        latest_acc_gap = acc_gap[-1]
        
        status_text = ""
        if latest_loss_gap < -0.1:
            status_text = "🔴 过拟合：训练损失远低于验证损失"
            color = 'red'
        elif latest_loss_gap > 0.05:
            status_text = "🟡 欠拟合：验证损失低于训练损失"
            color = 'orange'
        else:
            status_text = "🟢 拟合良好：训练和验证损失接近"
            color = 'green'
        
        ax4.text(0.5, 0.95, status_text, 
                transform=ax4.transAxes, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.3),
                fontsize=10)
        
        # 显示最新指标
        metrics_text = f"最新指标 (Epoch {epoch}):\n"
        metrics_text += f"训练: Loss={viz_data['train_loss'][-1]:.4f}, Acc={viz_data['train_acc'][-1]:.4f}, F1={viz_data['train_f1'][-1]:.4f}\n"
        metrics_text += f"验证: Loss={viz_data['val_loss'][-1]:.4f}, Acc={viz_data['val_acc'][-1]:.4f}, F1={viz_data['val_f1'][-1]:.4f}"
        
        ax4.text(0.5, 0.5, metrics_text,
                transform=ax4.transAxes, ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5),
                fontsize=9, family='monospace')
    
    plt.tight_layout()
    plt.pause(0.1)

# ========== 主函数 ==========
def main():
    print("="*50)
    print("BERT追问决策分类器微调（带可视化）")
    print("="*50)
    
    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 1. 加载数据
    print("\n1. 加载训练数据...")
    with open(CONFIG['train_data_path'], 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    
    print(f"数据总数: {len(all_data)}")
    
    # 分割数据集（80%训练，20%验证）
    train_data, val_data = train_test_split(all_data, test_size=0.2, random_state=42)
    print(f"训练集: {len(train_data)}, 验证集: {len(val_data)}")
    
    # 2. 加载模型
    print("\n2. 加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    model = AutoModelForSequenceClassification.from_pretrained(
        CONFIG['model_name'],
        num_labels=CONFIG['num_labels']
    )
    model.to(device)
    print(f"模型: {CONFIG['model_name']}")
    
    # 3. 创建数据集和数据加载器
    print("\n3. 创建数据集和加载器...")
    train_dataset = FollowUpDataset(train_data, tokenizer, CONFIG['label_map'], CONFIG['max_length'])
    val_dataset = FollowUpDataset(val_data, tokenizer, CONFIG['label_map'], CONFIG['max_length'])
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'])
    
    # 4. 设置优化器
    print("\n4. 设置优化器...")
    optimizer = AdamW(model.parameters(), lr=CONFIG['learning_rate'])
    
    total_steps = len(train_loader) * CONFIG['epochs']
    warmup_steps = int(total_steps * CONFIG['warmup_ratio'])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # 5. 设置可视化
    print("\n5. 设置可视化窗口...")
    fig, axes = setup_visualization()
    
    # 6. 开始训练
    print("\n6. 开始训练...")
    print("="*50)
    
    best_f1 = 0
    
    for epoch in range(1, CONFIG['epochs'] + 1):
        print(f"\nEpoch {epoch}/{CONFIG['epochs']}")
        print("-"*50)
        
        # 训练
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, optimizer, scheduler, device
        )
        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        
        # 验证
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, device)
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        
        # 更新可视化数据
        viz_data["epoch"].append(epoch)
        viz_data["train_loss"].append(train_loss)
        viz_data["val_loss"].append(val_loss)
        viz_data["train_acc"].append(train_acc)
        viz_data["val_acc"].append(val_acc)
        viz_data["train_f1"].append(train_f1)
        viz_data["val_f1"].append(val_f1)
        
        # 更新可视化
        update_visualization(fig, axes, epoch)
        
        # 保存最佳模型
        if val_f1 > best_f1:
            best_f1 = val_f1
            print(f"[BEST] 新的最佳F1: {best_f1:.4f}，保存模型...")
            
            os.makedirs(CONFIG['output_dir'], exist_ok=True)
            model.save_pretrained(CONFIG['output_dir'])
            tokenizer.save_pretrained(CONFIG['output_dir'])
        
        # 过拟合检测
        if epoch > 1 and val_loss > viz_data["val_loss"][-2]:
            print("[WARNING] 验证集损失上升，可能出现过拟合！")
    
    # 7. 最终评估
    print("\n" + "="*50)
    print("7. 最终评估（使用最佳模型）：")
    print("="*50)
    
    model = AutoModelForSequenceClassification.from_pretrained(CONFIG['output_dir'])
    model.to(device)
    
    _, _, _, final_preds, final_labels = evaluate(model, val_loader, device)
    
    # 详细分类报告
    label_names = ['FOLLOW_UP', 'NEXT_TOPIC']
    print("\n分类报告：")
    print(classification_report(final_labels, final_preds, target_names=label_names))
    
    print(f"\n[DONE] 训练完成！模型已保存到: {CONFIG['output_dir']}")
    print(f"最佳验证F1: {best_f1:.4f}")
    
    # 保存可视化图表
    viz_path = "./training_visualization.png"
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    print(f"\n可视化图表已保存到: {viz_path}")
    
    print("\n按任意键关闭可视化窗口...")
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()

