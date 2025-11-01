"""
训练Question V4模型（Instruction格式 + Topic Queue架构）
使用与Question V3相同的instruction-input-output格式，预期loss降至0.4-0.5
"""
import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from datasets import Dataset
import sys
import io
from datetime import datetime

# 修复Windows中文输出
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 配置
MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"
OUTPUT_DIR = f"checkpoints/qwen_question_v4_lora_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
FINAL_OUTPUT_DIR = "checkpoints/qwen_question_v4_lora"

# V3优化配置（与Question V3相同）
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.15
NUM_EPOCHS = 3
LEARNING_RATE = 1.0e-4
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
EVAL_STEPS = 50
SAVE_STEPS = 50
WEIGHT_DECAY = 0.01


def prepare_dataset(data_path):
    """
    准备训练数据集（Instruction格式）
    """
    print(f"加载数据: {data_path}")
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"样本数: {len(data)}")
    
    # 转换为Hugging Face Dataset
    formatted_data = []
    for sample in data:
        # 新格式已经包含instruction、input、output
        instruction = sample['instruction']
        user_input = sample['input']
        output = sample['output']
        
        # 组合instruction和input作为user消息
        user_message = f"{instruction}\n\n{user_input}"
        
        # 使用Qwen的chat模板
        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": output}
        ]
        
        formatted_data.append({"messages": messages})
    
    return Dataset.from_list(formatted_data)


def tokenize_function(examples, tokenizer):
    """
    Tokenize函数
    """
    texts = []
    for messages in examples["messages"]:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        texts.append(text)
    
    return tokenizer(
        texts,
        truncation=True,
        max_length=2048,
        padding=False
    )


def main():
    print("="*60)
    print("Question V4 模型训练（基于Topic Queue）")
    print("="*60)
    
    # 1. 加载tokenizer和模型
    print("\n[1/6] 加载tokenizer和模型...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        cache_dir="E:/HuggingFace_Cache"
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # 4-bit量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        trust_remote_code=True,
        cache_dir="E:/HuggingFace_Cache",
        quantization_config=bnb_config
    )
    
    # 准备模型用于k-bit训练
    model = prepare_model_for_kbit_training(model)
    print("✓ 模型加载完成")
    
    # 2. 配置LoRA
    print("\n[2/6] 配置LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 3. 准备数据集
    print("\n[3/6] 准备数据集...")
    train_dataset = prepare_dataset("training_data/question_v4_train.json")
    val_dataset = prepare_dataset("training_data/question_v4_val.json")
    
    # Tokenize
    print("Tokenizing训练集...")
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    print("Tokenizing验证集...")
    val_dataset = val_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=val_dataset.column_names
    )
    
    # 4. 配置训练参数
    print("\n[4/6] 配置训练参数...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        logging_steps=10,
        eval_steps=EVAL_STEPS,
        save_steps=SAVE_STEPS,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=3,
        fp16=True,
        report_to="none",
        gradient_checkpointing=True,
        warmup_steps=20
    )
    
    print(f"总训练步数: {len(train_dataset) // (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS) * NUM_EPOCHS}")
    
    # 5. 创建Trainer
    print("\n[5/6] 创建Trainer...")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=10,
                early_stopping_threshold=0.001
            )
        ]
    )
    
    # 6. 开始训练
    print("\n[6/6] 开始训练...")
    print(f"输出目录: {OUTPUT_DIR}")
    print("-"*60)
    
    trainer.train()
    
    # 7. 保存最终模型
    print("\n" + "="*60)
    print("保存最终模型...")
    trainer.save_model(FINAL_OUTPUT_DIR)
    tokenizer.save_pretrained(FINAL_OUTPUT_DIR)
    
    print(f"✅ 训练完成！模型已保存到: {FINAL_OUTPUT_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()

