"""
训练Qwen-Scorer V2模型
输出格式：score + comment（移除label）
使用V3优化配置（减少过拟合）
"""
import sys
import io

# 修复Windows中文输出问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import json
from datetime import datetime

def load_data(train_path, val_path):
    """加载训练和验证数据"""
    with open(train_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    with open(val_path, 'r', encoding='utf-8') as f:
        val_data = json.load(f)
    
    return Dataset.from_list(train_data), Dataset.from_list(val_data)

def format_prompt(sample):
    """格式化prompt为Qwen2对话格式"""
    instruction = "你是一位经验丰富的技术面试官。评估候选人回答的质量，给出评分（0-100分）和详细评价。"
    
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": sample['input']},
        {"role": "assistant", "content": sample['output']}
    ]
    
    return messages

def main():
    print("="*60)
    print("Qwen-Scorer V2 训练 (移除label，只输出score+comment)")
    print("="*60)
    
    # 1. 加载数据
    print("\n[1/6] 加载数据...")
    train_dataset, val_dataset = load_data(
        'training_data/qwen_scorer_v2_train.json',
        'training_data/qwen_scorer_v2_val.json'
    )
    print(f"训练集: {len(train_dataset)} 条")
    print(f"验证集: {len(val_dataset)} 条")
    
    # 2. 加载模型和分词器
    print("\n[2/6] 加载Qwen2-1.5B-Instruct模型...")
    model_name = "Qwen/Qwen2-1.5B-Instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 4bit量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )
    model = prepare_model_for_kbit_training(model)
    
    # 3. LoRA配置（V3优化）
    print("\n[3/6] 配置LoRA参数...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.15,  # V3: 增加dropout减少过拟合
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 4. 数据预处理
    print("\n[4/6] 预处理数据...")
    def preprocess_function(examples):
        messages = format_prompt(examples)
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        model_inputs = tokenizer(
            text,
            max_length=512,
            truncation=True,
            padding="max_length"
        )
        
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        return model_inputs
    
    train_dataset = train_dataset.map(preprocess_function, remove_columns=train_dataset.column_names)
    val_dataset = val_dataset.map(preprocess_function, remove_columns=val_dataset.column_names)
    
    # 5. 训练配置（V3优化）
    print("\n[5/6] 配置训练参数...")
    output_dir = f"checkpoints/qwen_scorer_v2_lora_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,  # V3: 减少到3个epoch
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=1.0e-4,  # V3: 降低学习率
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,  # V3: 添加权重衰减
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        report_to="none",
        remove_unused_columns=False,
    )
    
    # 6. 训练
    print("\n[6/6] 开始训练...")
    print(f"输出目录: {output_dir}")
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=10,  # V3: 早停机制
                early_stopping_threshold=0.001
            )
        ]
    )
    
    trainer.train()
    
    # 保存最终模型
    final_output_dir = "checkpoints/qwen_scorer_v2_lora"
    trainer.save_model(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    
    print("\n" + "="*60)
    print(f"✅ 训练完成！模型已保存到: {final_output_dir}")
    print("="*60)

if __name__ == "__main__":
    main()

