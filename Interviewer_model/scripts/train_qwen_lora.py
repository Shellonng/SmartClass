"""
Qwen2-1.5B-Instruct LoRA微调脚本
用于训练AI面试官对话生成能力
"""
import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel, prepare_model_for_kbit_training
from datasets import Dataset
import os

# ========== 配置 ==========
CONFIG = {
    "model_name": "Qwen/Qwen2-1.5B-Instruct",
    "train_data_path": "./data/qwen_training_2000.json",
    "output_dir": "./checkpoints/qwen_interviewer_lora",
    "lora_r": 8,  # LoRA rank
    "lora_alpha": 16,  # LoRA alpha
    "lora_dropout": 0.05,
    "max_length": 512,
    "batch_size": 4,  # 减小batch size适应显存
    "gradient_accumulation_steps": 4,  # 梯度累积，等效batch size=16
    "epochs": 3,
    "learning_rate": 1e-4,
    "warmup_ratio": 0.1,
    "logging_steps": 50,
    "save_steps": 200,
}

print("="*60)
print("Qwen2面试官对话生成模型 - LoRA微调")
print("="*60)

# 检查设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# 1. 加载数据
print("\n1. 加载训练数据...")
with open(CONFIG['train_data_path'], 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"数据总数: {len(data)}")

# 2. 加载模型和tokenizer
print("\n2. 加载模型...")
tokenizer = AutoTokenizer.from_pretrained(
    CONFIG['model_name'],
    trust_remote_code=True,
    padding_side='right'  # 确保padding在右侧
)

# 设置pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 配置8bit量化
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    CONFIG['model_name'],
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# 准备模型以进行8bit训练
model = prepare_model_for_kbit_training(model)

print(f"模型: {CONFIG['model_name']}")
print(f"参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

# 3. 配置LoRA
print("\n3. 配置LoRA...")
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=CONFIG['lora_r'],
    lora_alpha=CONFIG['lora_alpha'],
    lora_dropout=CONFIG['lora_dropout'],
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Qwen2的attention模块
    bias="none",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 4. 准备数据集
print("\n4. 准备数据集...")

def process_func(example):
    """处理单条数据"""
    messages = example['messages']
    
    # 使用chat_template格式化完整对话
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    
    # Tokenize完整对话
    model_inputs = tokenizer(
        text,
        max_length=CONFIG['max_length'],
        truncation=True,
        padding=False,
        return_tensors=None
    )
    
    input_ids = model_inputs['input_ids']
    
    # 计算system和user部分的长度（这部分loss设为-100）
    prefix_text = tokenizer.apply_chat_template(
        messages[:-1],  # system + user
        tokenize=False,
        add_generation_prompt=True
    )
    prefix_ids = tokenizer(prefix_text, add_special_tokens=False)['input_ids']
    prefix_len = len(prefix_ids)
    
    # labels：前缀部分为-100，assistant回复部分为实际token
    labels = [-100] * prefix_len + input_ids[prefix_len:]
    
    # 确保长度一致
    if len(labels) < len(input_ids):
        labels = labels + input_ids[len(labels):]
    labels = labels[:len(input_ids)]
    
    return {
        'input_ids': input_ids,
        'labels': labels,
        'attention_mask': [1] * len(input_ids)
    }

# 转换为Dataset
dataset = Dataset.from_list(data)
processed_dataset = dataset.map(
    process_func,
    remove_columns=dataset.column_names,
    desc="Processing dataset"
)

# 分割训练集和验证集
split_dataset = processed_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset['train']
eval_dataset = split_dataset['test']

print(f"训练集: {len(train_dataset)}")
print(f"验证集: {len(eval_dataset)}")

# 5. 设置训练参数
print("\n5. 设置训练参数...")
training_args = TrainingArguments(
    output_dir=CONFIG['output_dir'],
    per_device_train_batch_size=CONFIG['batch_size'],
    per_device_eval_batch_size=CONFIG['batch_size'],
    gradient_accumulation_steps=CONFIG['gradient_accumulation_steps'],
    num_train_epochs=CONFIG['epochs'],
    learning_rate=CONFIG['learning_rate'],
    warmup_ratio=CONFIG['warmup_ratio'],
    logging_steps=CONFIG['logging_steps'],
    save_steps=CONFIG['save_steps'],
    eval_strategy="steps",
    eval_steps=CONFIG['save_steps'],
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    fp16=False,  # 8bit量化模型不使用fp16
    gradient_checkpointing=False,  # 禁用，避免梯度问题
    report_to="none",  # 不使用wandb等
    remove_unused_columns=False,
)

# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    padding=True,
    return_tensors="pt"
)

# 6. 创建Trainer
print("\n6. 创建Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

# 7. 开始训练
print("\n7. 开始训练...")
print("="*60)

try:
    trainer.train()
    
    # 8. 保存模型
    print("\n8. 保存LoRA模型...")
    model.save_pretrained(CONFIG['output_dir'])
    tokenizer.save_pretrained(CONFIG['output_dir'])
    
    print(f"\n[DONE] 训练完成！")
    print(f"LoRA权重已保存到: {CONFIG['output_dir']}")
    print(f"\n使用方法：")
    print(f"1. 加载基础模型: Qwen/Qwen2-1.5B-Instruct")
    print(f"2. 加载LoRA权重: {CONFIG['output_dir']}")
    
except Exception as e:
    print(f"\n[ERROR] 训练失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
