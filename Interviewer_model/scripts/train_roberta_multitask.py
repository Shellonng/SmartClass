"""
RoBERTa多任务训练脚本
任务1: 当前回答质量分类（4分类：0=差, 1=一般, 2=良好, 3=优秀）
任务2: 整体能力评分（回归：0-100分）
"""
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error
import numpy as np
import os
from tqdm import tqdm

# 配置
CONFIG = {
    "model_name": "./models/chinese-roberta-wwm-ext",  # 使用本地模型
    "train_data_path": "./data/roberta_training_2000.json",
    "output_dir": "./checkpoints/answer_evaluator",
    "max_length": 256,  # 降低到256避免显存溢出
    "batch_size": 8,  # 降低batch size
    "gradient_accumulation_steps": 4,  # 梯度累积模拟batch_size=32
    "epochs": 3,  # 减少到3 epochs加快训练
    "learning_rate": 2e-5,
    "warmup_ratio": 0.1,
    "classification_weight": 1.0,  # 分类任务权重
    "regression_weight": 0.01,  # 回归任务权重（缩放到0-1范围）
}

print("=" * 60)
print("RoBERTa多任务训练 - 回答评估模型")
print("=" * 60)

# 检查设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 清理GPU缓存
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"GPU显存清理完成")

# 1. 加载数据
print("\n1. 加载训练数据...")
with open(CONFIG['train_data_path'], 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"数据总数: {len(data)}")

# 2. 定义数据集
class AnswerEvaluationDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 构建输入文本
        input_text = self._build_input_text(item)
        
        # Tokenize（不padding，在collate时动态padding）
        encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            truncation=True,
            padding=False,  # 使用动态padding
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'classification_label': torch.tensor(item['current_label'], dtype=torch.long),
            'regression_label': torch.tensor(item['overall_score'] / 100.0, dtype=torch.float)  # 归一化到0-1
        }
    
    def _build_input_text(self, item):
        """构建模型输入"""
        parts = []
        
        # 岗位信息
        parts.append(f"[岗位] {item['job_position']}")
        
        # 历史问答
        if item['history_qa']:
            parts.append("[历史问答]")
            for i, qa in enumerate(item['history_qa'], 1):
                parts.append(f"Q{i}: {qa['question']}")
                parts.append(f"A{i}: {qa['answer'][:100]}")  # 限制长度
                parts.append(f"质量: {qa['current_quality']}")
        
        # 当前问答
        parts.append("[当前问答]")
        parts.append(f"问题: {item['current_qa']['question']}")
        parts.append(f"回答: {item['current_qa']['answer']}")
        parts.append(f"流畅度: {1 - item['current_qa']['hesitation_score']:.2f}")
        parts.append(f"重要性: {item['current_qa']['question_importance']}")
        
        return "\n".join(parts)

# 动态padding collate函数
def collate_fn(batch):
    """动态padding到batch内最大长度"""
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    
    # Padding
    from torch.nn.utils.rnn import pad_sequence
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'classification_label': torch.stack([item['classification_label'] for item in batch]),
        'regression_label': torch.stack([item['regression_label'] for item in batch])
    }

# 3. 定义多任务模型
class MultiTaskRoBERTa(nn.Module):
    def __init__(self, model_name, num_labels=4):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(model_name)
        hidden_size = self.roberta.config.hidden_size
        
        # 分类头（当前回答质量）
        self.classification_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_labels)
        )
        
        # 回归头（整体能力评分）
        self.regression_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()  # 输出0-1范围
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        
        classification_logits = self.classification_head(pooled_output)
        regression_score = self.regression_head(pooled_output).squeeze(-1)
        
        return classification_logits, regression_score

# 4. 加载tokenizer和模型
print("\n2. 加载模型...")
tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
model = MultiTaskRoBERTa(CONFIG['model_name'], num_labels=4)
model.to(device)

print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

# 5. 准备数据集
print("\n3. 准备数据集...")
train_data, val_data = train_test_split(data, test_size=0.1, random_state=42)

train_dataset = AnswerEvaluationDataset(train_data, tokenizer, CONFIG['max_length'])
val_dataset = AnswerEvaluationDataset(val_data, tokenizer, CONFIG['max_length'])

# DataLoader配置（Windows上使用num_workers=0更稳定）
train_loader = DataLoader(
    train_dataset, 
    batch_size=CONFIG['batch_size'], 
    shuffle=True,
    num_workers=0,  # Windows上设为0
    pin_memory=True,  # 加速GPU传输
    collate_fn=collate_fn  # 动态padding
)
val_loader = DataLoader(
    val_dataset, 
    batch_size=CONFIG['batch_size'],
    num_workers=0,  # Windows上设为0
    pin_memory=True,
    collate_fn=collate_fn
)

print(f"训练集: {len(train_dataset)}")
print(f"验证集: {len(val_dataset)}")

# 6. 优化器和调度器
print("\n4. 配置优化器...")
optimizer = AdamW(model.parameters(), lr=CONFIG['learning_rate'])
total_steps = len(train_loader) * CONFIG['epochs']
warmup_steps = int(total_steps * CONFIG['warmup_ratio'])
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

# 损失函数
classification_criterion = nn.CrossEntropyLoss()
regression_criterion = nn.MSELoss()

# 7. 训练函数（支持梯度累积）
def train_epoch(model, loader, optimizer, scheduler, device, epoch_num):
    model.train()
    total_loss = 0
    cls_losses = []
    reg_losses = []
    
    accumulation_steps = CONFIG.get('gradient_accumulation_steps', 1)
    
    pbar = tqdm(loader, desc=f"Epoch {epoch_num}")
    for step, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        cls_labels = batch['classification_label'].to(device)
        reg_labels = batch['regression_label'].to(device)
        
        # 前向传播
        cls_logits, reg_scores = model(input_ids, attention_mask)
        
        # 计算损失
        cls_loss = classification_criterion(cls_logits, cls_labels)
        reg_loss = regression_criterion(reg_scores, reg_labels)
        
        # 多任务损失
        loss = (CONFIG['classification_weight'] * cls_loss + 
                CONFIG['regression_weight'] * reg_loss)
        
        # 梯度累积
        loss = loss / accumulation_steps
        loss.backward()
        
        # 每accumulation_steps步更新一次
        if (step + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
        cls_losses.append(cls_loss.item())
        reg_losses.append(reg_loss.item())
        
        pbar.set_postfix({
            'loss': f'{loss.item() * accumulation_steps:.4f}',
            'cls': f'{cls_loss.item():.4f}',
            'reg': f'{reg_loss.item():.4f}'
        })
    
    return total_loss / len(loader), np.mean(cls_losses), np.mean(reg_losses)

# 8. 评估函数
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    cls_preds, cls_labels = [], []
    reg_preds, reg_labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            cls_label = batch['classification_label'].to(device)
            reg_label = batch['regression_label'].to(device)
            
            cls_logits, reg_scores = model(input_ids, attention_mask)
            
            cls_loss = classification_criterion(cls_logits, cls_label)
            reg_loss = regression_criterion(reg_scores, reg_label)
            loss = CONFIG['classification_weight'] * cls_loss + CONFIG['regression_weight'] * reg_loss
            
            total_loss += loss.item()
            
            cls_preds.extend(cls_logits.argmax(dim=-1).cpu().numpy())
            cls_labels.extend(cls_label.cpu().numpy())
            reg_preds.extend((reg_scores * 100).cpu().numpy())  # 反归一化
            reg_labels.extend((reg_label * 100).cpu().numpy())
    
    # 计算指标
    cls_acc = accuracy_score(cls_labels, cls_preds)
    cls_f1 = f1_score(cls_labels, cls_preds, average='weighted')
    reg_mae = mean_absolute_error(reg_labels, reg_preds)
    reg_rmse = np.sqrt(mean_squared_error(reg_labels, reg_preds))
    
    return (total_loss / len(loader), cls_acc, cls_f1, reg_mae, reg_rmse)

# 9. 训练循环
print("\n5. 开始训练...")
print("=" * 60)

best_val_loss = float('inf')

for epoch in range(CONFIG['epochs']):
    print(f"\nEpoch {epoch + 1}/{CONFIG['epochs']}")
    print("-" * 60)
    
    train_loss, train_cls_loss, train_reg_loss = train_epoch(
        model, train_loader, optimizer, scheduler, device, epoch + 1
    )
    
    val_loss, val_cls_acc, val_cls_f1, val_reg_mae, val_reg_rmse = evaluate(
        model, val_loader, device
    )
    
    print(f"\nTrain Loss: {train_loss:.4f} (cls={train_cls_loss:.4f}, reg={train_reg_loss:.4f})")
    print(f"Val Loss: {val_loss:.4f}")
    print(f"Classification - Acc: {val_cls_acc:.4f}, F1: {val_cls_f1:.4f}")
    print(f"Regression - MAE: {val_reg_mae:.2f}, RMSE: {val_reg_rmse:.2f}")
    
    # 保存最佳模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        print(f"[BEST] 新的最佳模型，保存中...")
        os.makedirs(CONFIG['output_dir'], exist_ok=True)
        
        # 保存模型
        torch.save(model.state_dict(), f"{CONFIG['output_dir']}/pytorch_model.bin")
        tokenizer.save_pretrained(CONFIG['output_dir'])
        
        # 保存配置
        config_dict = {
            "model_name": CONFIG['model_name'],
            "num_labels": 4,
            "max_length": CONFIG['max_length'],
            "label_names": ["差", "一般", "良好", "优秀"],
            "score_mapping": [50, 70, 85, 95]
        }
        with open(f"{CONFIG['output_dir']}/config.json", 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=2)

print("\n" + "=" * 60)
print("[DONE] 训练完成！")
print(f"最佳模型已保存到: {CONFIG['output_dir']}")
print("=" * 60)


任务1: 当前回答质量分类（4分类：0=差, 1=一般, 2=良好, 3=优秀）
任务2: 整体能力评分（回归：0-100分）
"""
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error
import numpy as np
import os
from tqdm import tqdm

# 配置
CONFIG = {
    "model_name": "./models/chinese-roberta-wwm-ext",  # 使用本地模型
    "train_data_path": "./data/roberta_training_2000.json",
    "output_dir": "./checkpoints/answer_evaluator",
    "max_length": 256,  # 降低到256避免显存溢出
    "batch_size": 8,  # 降低batch size
    "gradient_accumulation_steps": 4,  # 梯度累积模拟batch_size=32
    "epochs": 3,  # 减少到3 epochs加快训练
    "learning_rate": 2e-5,
    "warmup_ratio": 0.1,
    "classification_weight": 1.0,  # 分类任务权重
    "regression_weight": 0.01,  # 回归任务权重（缩放到0-1范围）
}

print("=" * 60)
print("RoBERTa多任务训练 - 回答评估模型")
print("=" * 60)

# 检查设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 清理GPU缓存
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"GPU显存清理完成")

# 1. 加载数据
print("\n1. 加载训练数据...")
with open(CONFIG['train_data_path'], 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"数据总数: {len(data)}")

# 2. 定义数据集
class AnswerEvaluationDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 构建输入文本
        input_text = self._build_input_text(item)
        
        # Tokenize（不padding，在collate时动态padding）
        encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            truncation=True,
            padding=False,  # 使用动态padding
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'classification_label': torch.tensor(item['current_label'], dtype=torch.long),
            'regression_label': torch.tensor(item['overall_score'] / 100.0, dtype=torch.float)  # 归一化到0-1
        }
    
    def _build_input_text(self, item):
        """构建模型输入"""
        parts = []
        
        # 岗位信息
        parts.append(f"[岗位] {item['job_position']}")
        
        # 历史问答
        if item['history_qa']:
            parts.append("[历史问答]")
            for i, qa in enumerate(item['history_qa'], 1):
                parts.append(f"Q{i}: {qa['question']}")
                parts.append(f"A{i}: {qa['answer'][:100]}")  # 限制长度
                parts.append(f"质量: {qa['current_quality']}")
        
        # 当前问答
        parts.append("[当前问答]")
        parts.append(f"问题: {item['current_qa']['question']}")
        parts.append(f"回答: {item['current_qa']['answer']}")
        parts.append(f"流畅度: {1 - item['current_qa']['hesitation_score']:.2f}")
        parts.append(f"重要性: {item['current_qa']['question_importance']}")
        
        return "\n".join(parts)

# 动态padding collate函数
def collate_fn(batch):
    """动态padding到batch内最大长度"""
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    
    # Padding
    from torch.nn.utils.rnn import pad_sequence
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'classification_label': torch.stack([item['classification_label'] for item in batch]),
        'regression_label': torch.stack([item['regression_label'] for item in batch])
    }

# 3. 定义多任务模型
class MultiTaskRoBERTa(nn.Module):
    def __init__(self, model_name, num_labels=4):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(model_name)
        hidden_size = self.roberta.config.hidden_size
        
        # 分类头（当前回答质量）
        self.classification_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_labels)
        )
        
        # 回归头（整体能力评分）
        self.regression_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()  # 输出0-1范围
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        
        classification_logits = self.classification_head(pooled_output)
        regression_score = self.regression_head(pooled_output).squeeze(-1)
        
        return classification_logits, regression_score

# 4. 加载tokenizer和模型
print("\n2. 加载模型...")
tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
model = MultiTaskRoBERTa(CONFIG['model_name'], num_labels=4)
model.to(device)

print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

# 5. 准备数据集
print("\n3. 准备数据集...")
train_data, val_data = train_test_split(data, test_size=0.1, random_state=42)

train_dataset = AnswerEvaluationDataset(train_data, tokenizer, CONFIG['max_length'])
val_dataset = AnswerEvaluationDataset(val_data, tokenizer, CONFIG['max_length'])

# DataLoader配置（Windows上使用num_workers=0更稳定）
train_loader = DataLoader(
    train_dataset, 
    batch_size=CONFIG['batch_size'], 
    shuffle=True,
    num_workers=0,  # Windows上设为0
    pin_memory=True,  # 加速GPU传输
    collate_fn=collate_fn  # 动态padding
)
val_loader = DataLoader(
    val_dataset, 
    batch_size=CONFIG['batch_size'],
    num_workers=0,  # Windows上设为0
    pin_memory=True,
    collate_fn=collate_fn
)

print(f"训练集: {len(train_dataset)}")
print(f"验证集: {len(val_dataset)}")

# 6. 优化器和调度器
print("\n4. 配置优化器...")
optimizer = AdamW(model.parameters(), lr=CONFIG['learning_rate'])
total_steps = len(train_loader) * CONFIG['epochs']
warmup_steps = int(total_steps * CONFIG['warmup_ratio'])
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

# 损失函数
classification_criterion = nn.CrossEntropyLoss()
regression_criterion = nn.MSELoss()

# 7. 训练函数（支持梯度累积）
def train_epoch(model, loader, optimizer, scheduler, device, epoch_num):
    model.train()
    total_loss = 0
    cls_losses = []
    reg_losses = []
    
    accumulation_steps = CONFIG.get('gradient_accumulation_steps', 1)
    
    pbar = tqdm(loader, desc=f"Epoch {epoch_num}")
    for step, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        cls_labels = batch['classification_label'].to(device)
        reg_labels = batch['regression_label'].to(device)
        
        # 前向传播
        cls_logits, reg_scores = model(input_ids, attention_mask)
        
        # 计算损失
        cls_loss = classification_criterion(cls_logits, cls_labels)
        reg_loss = regression_criterion(reg_scores, reg_labels)
        
        # 多任务损失
        loss = (CONFIG['classification_weight'] * cls_loss + 
                CONFIG['regression_weight'] * reg_loss)
        
        # 梯度累积
        loss = loss / accumulation_steps
        loss.backward()
        
        # 每accumulation_steps步更新一次
        if (step + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
        cls_losses.append(cls_loss.item())
        reg_losses.append(reg_loss.item())
        
        pbar.set_postfix({
            'loss': f'{loss.item() * accumulation_steps:.4f}',
            'cls': f'{cls_loss.item():.4f}',
            'reg': f'{reg_loss.item():.4f}'
        })
    
    return total_loss / len(loader), np.mean(cls_losses), np.mean(reg_losses)

# 8. 评估函数
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    cls_preds, cls_labels = [], []
    reg_preds, reg_labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            cls_label = batch['classification_label'].to(device)
            reg_label = batch['regression_label'].to(device)
            
            cls_logits, reg_scores = model(input_ids, attention_mask)
            
            cls_loss = classification_criterion(cls_logits, cls_label)
            reg_loss = regression_criterion(reg_scores, reg_label)
            loss = CONFIG['classification_weight'] * cls_loss + CONFIG['regression_weight'] * reg_loss
            
            total_loss += loss.item()
            
            cls_preds.extend(cls_logits.argmax(dim=-1).cpu().numpy())
            cls_labels.extend(cls_label.cpu().numpy())
            reg_preds.extend((reg_scores * 100).cpu().numpy())  # 反归一化
            reg_labels.extend((reg_label * 100).cpu().numpy())
    
    # 计算指标
    cls_acc = accuracy_score(cls_labels, cls_preds)
    cls_f1 = f1_score(cls_labels, cls_preds, average='weighted')
    reg_mae = mean_absolute_error(reg_labels, reg_preds)
    reg_rmse = np.sqrt(mean_squared_error(reg_labels, reg_preds))
    
    return (total_loss / len(loader), cls_acc, cls_f1, reg_mae, reg_rmse)

# 9. 训练循环
print("\n5. 开始训练...")
print("=" * 60)

best_val_loss = float('inf')

for epoch in range(CONFIG['epochs']):
    print(f"\nEpoch {epoch + 1}/{CONFIG['epochs']}")
    print("-" * 60)
    
    train_loss, train_cls_loss, train_reg_loss = train_epoch(
        model, train_loader, optimizer, scheduler, device, epoch + 1
    )
    
    val_loss, val_cls_acc, val_cls_f1, val_reg_mae, val_reg_rmse = evaluate(
        model, val_loader, device
    )
    
    print(f"\nTrain Loss: {train_loss:.4f} (cls={train_cls_loss:.4f}, reg={train_reg_loss:.4f})")
    print(f"Val Loss: {val_loss:.4f}")
    print(f"Classification - Acc: {val_cls_acc:.4f}, F1: {val_cls_f1:.4f}")
    print(f"Regression - MAE: {val_reg_mae:.2f}, RMSE: {val_reg_rmse:.2f}")
    
    # 保存最佳模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        print(f"[BEST] 新的最佳模型，保存中...")
        os.makedirs(CONFIG['output_dir'], exist_ok=True)
        
        # 保存模型
        torch.save(model.state_dict(), f"{CONFIG['output_dir']}/pytorch_model.bin")
        tokenizer.save_pretrained(CONFIG['output_dir'])
        
        # 保存配置
        config_dict = {
            "model_name": CONFIG['model_name'],
            "num_labels": 4,
            "max_length": CONFIG['max_length'],
            "label_names": ["差", "一般", "良好", "优秀"],
            "score_mapping": [50, 70, 85, 95]
        }
        with open(f"{CONFIG['output_dir']}/config.json", 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=2)

print("\n" + "=" * 60)
print("[DONE] 训练完成！")
print(f"最佳模型已保存到: {CONFIG['output_dir']}")
print("=" * 60)





