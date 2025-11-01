"""
训练 Qwen-Question 模型（原有功能）
功能: 根据话题、历史对话和指导建议，生成下一个问题和重要程度

针对 RTX 4060 8GB 显存优化:
- 4bit量化
- 小batch size (2)
- 梯度累积 (4)
"""

import json
import sys
import io
import torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
import matplotlib.pyplot as plt

# 修复Windows控制台编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 配置（V3优化版：针对过拟合优化）
CONFIG = {
    # 模型配置
    "base_model": "Qwen/Qwen2-1.5B-Instruct",  # 使用Qwen2（与test_qwen.py相同）
    "max_length": 1024,
    
    # LoRA配置（增加正则化）
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.15,  # V3: 从0.1增至0.15，增强正则化
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", 
                            "gate_proj", "up_proj", "down_proj"],
    
    # 训练配置V3（防止过拟合）
    "batch_size": 2,
    "gradient_accumulation": 4,
    "num_epochs": 3,  # V3: 从5降至3，避免后期过拟合
    "learning_rate": 1.0e-4,  # V3: 从1.5e-4降至1.0e-4，更稳定训练
    "warmup_steps": 30,  # V3: 减少预热步数
    "logging_steps": 5,  # V3: 更频繁的日志
    "save_steps": 50,  # V3: 更频繁保存，便于选择最佳模型
    "eval_steps": 10,  # V3: 每10步验证
    
    # 早停配置（新增）
    "early_stopping_patience": 10,  # V3: 10次验证不改善则停止
    "early_stopping_threshold": 0.001,  # V3: 改善阈值
    
    # 4bit量化配置
    "use_4bit": True,
    "bnb_4bit_compute_dtype": torch.float16,
    "bnb_4bit_quant_type": "nf4",
    
    # 路径
    "train_data": "dual_qwen_data/qwen_question_train_split.json",
    "val_data": "dual_qwen_data/qwen_question_val_split.json",
    "output_dir": "checkpoints/qwen_question_lora",
}

def load_data(filepath):
    """加载训练数据"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return Dataset.from_list(data)

def format_prompt(example):
    """
    格式化为Qwen的对话格式
    使用Qwen的prompt模板
    """
    instruction = example['instruction']
    input_text = example['input']
    output_text = example.get('output', '')
    
    # Qwen Chat格式
    prompt = f"""<|im_start|>system
{instruction}<|im_end|>
<|im_start|>user
{input_text}<|im_end|>
<|im_start|>assistant
{output_text}<|im_end|>"""
    
    return prompt

def tokenize_function(examples, tokenizer):
    """分词函数"""
    prompts = [format_prompt(ex) for ex in examples]
    
    # 分词
    model_inputs = tokenizer(
        prompts,
        max_length=CONFIG["max_length"],
        truncation=True,
        padding=False,
        return_tensors=None
    )
    
    # 设置labels
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    
    return model_inputs

def print_trainable_parameters(model):
    """打印可训练参数"""
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    print(f"\n📊 模型参数统计:")
    print(f"  可训练参数: {trainable_params:,} ({100 * trainable_params / all_params:.2f}%)")
    print(f"  总参数: {all_params:,}")
    print(f"  节省参数: {100 * (all_params - trainable_params) / all_params:.2f}%")

def plot_training_history(trainer, output_path):
    """绘制训练曲线（包含训练loss和验证loss）"""
    log_history = trainer.state.log_history
    
    # 提取训练loss
    train_steps = []
    train_losses = []
    
    # 提取验证loss
    eval_steps = []
    eval_losses = []
    
    for log in log_history:
        if 'loss' in log and 'eval_loss' not in log:
            train_steps.append(log['step'])
            train_losses.append(log['loss'])
        if 'eval_loss' in log:
            eval_steps.append(log['step'])
            eval_losses.append(log['eval_loss'])
    
    # 绘图
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 绘制训练loss
    ax.plot(train_steps, train_losses, label='Training Loss', 
            linewidth=2, color='#2E86AB', alpha=0.8)
    
    # 绘制验证loss
    ax.plot(eval_steps, eval_losses, label='Validation Loss', 
            linewidth=2.5, color='#E63946', alpha=0.9, marker='o', 
            markersize=3, markevery=5)
    
    ax.set_xlabel('Steps', fontsize=13, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=13, fontweight='bold')
    ax.set_title('Qwen-Question Training History', fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 添加最终loss值的文本标注
    if train_losses and eval_losses:
        final_train = train_losses[-1]
        final_eval = eval_losses[-1]
        ax.text(0.02, 0.98, f'Final Train Loss: {final_train:.4f}\nFinal Eval Loss: {final_eval:.4f}',
                transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ 训练曲线已保存: {output_path}")

def main():
    print("="*60)
    print("🚀 训练 Qwen-Question 模型（问题生成）")
    print("="*60)
    
    # 检查GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"\n✓ GPU: {gpu_name}")
        print(f"✓ 显存: {gpu_memory:.1f} GB")
    else:
        print("\n⚠️  未检测到GPU，将使用CPU训练（会很慢）")
    
    # 加载数据
    print(f"\n📥 加载训练数据...")
    train_dataset = load_data(CONFIG["train_data"])
    val_dataset = load_data(CONFIG["val_data"])
    
    print(f"✓ 训练集: {len(train_dataset)} 条")
    print(f"✓ 验证集: {len(val_dataset)} 条")
    
    # 加载分词器
    print(f"\n📥 加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG["base_model"],
        padding_side="right"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"✓ 分词器加载完成")
    
    # 分词
    print(f"\n🔄 对数据进行分词...")
    
    def tokenize_batch(examples):
        batch = []
        for i in range(len(examples['instruction'])):
            batch.append({
                'instruction': examples['instruction'][i],
                'input': examples['input'][i],
                'output': examples['output'][i]
            })
        return tokenize_function(batch, tokenizer)
    
    train_dataset = train_dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="分词训练集"
    )
    
    val_dataset = val_dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=val_dataset.column_names,
        desc="分词验证集"
    )
    
    print(f"✓ 分词完成")
    
    # 4bit量化配置
    print(f"\n⚙️  配置4bit量化...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=CONFIG["use_4bit"],
        bnb_4bit_compute_dtype=CONFIG["bnb_4bit_compute_dtype"],
        bnb_4bit_quant_type=CONFIG["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=True,
    )
    
    # 加载基座模型
    print(f"\n📥 加载基座模型: {CONFIG['base_model']}")
    print(f"   使用4bit量化以节省显存...")
    
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["base_model"],
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    
    print(f"✓ 基座模型加载完成")
    
    # 准备模型
    model = prepare_model_for_kbit_training(model)
    
    # LoRA配置
    print(f"\n⚙️  配置LoRA...")
    lora_config = LoraConfig(
        r=CONFIG["lora_r"],
        lora_alpha=CONFIG["lora_alpha"],
        target_modules=CONFIG["lora_target_modules"],
        lora_dropout=CONFIG["lora_dropout"],
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    # 应用LoRA
    model = get_peft_model(model, lora_config)
    print_trainable_parameters(model)
    
    # 训练参数
    print(f"\n⚙️  配置训练参数...")
    training_args = TrainingArguments(
        output_dir=CONFIG["output_dir"],
        num_train_epochs=CONFIG["num_epochs"],
        per_device_train_batch_size=CONFIG["batch_size"],
        per_device_eval_batch_size=CONFIG["batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation"],
        learning_rate=CONFIG["learning_rate"],
        warmup_steps=CONFIG["warmup_steps"],
        logging_steps=CONFIG["logging_steps"],
        save_steps=CONFIG["save_steps"],
        eval_steps=CONFIG["eval_steps"],  # V3: 使用独立的eval_steps
        eval_strategy="steps",  # 新版本使用eval_strategy
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",  # V3: 明确指定用eval_loss选最佳模型
        greater_is_better=False,  # V3: loss越小越好
        fp16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        max_grad_norm=0.3,
        weight_decay=0.01,  # V3: 添加权重衰减
        logging_dir=f"{CONFIG['output_dir']}/logs",
        report_to="none",
    )
    
    print(f"✓ 训练配置 (V3 - 防过拟合优化):")
    print(f"  - Batch size: {CONFIG['batch_size']}")
    print(f"  - 梯度累积: {CONFIG['gradient_accumulation']} (等效batch={CONFIG['batch_size']*CONFIG['gradient_accumulation']})")
    print(f"  - Epochs: {CONFIG['num_epochs']} (从5降至3)")
    print(f"  - Learning rate: {CONFIG['learning_rate']} (从1.5e-4降至1.0e-4)")
    print(f"  - LoRA dropout: 0.15 (从0.1增至0.15)")
    print(f"  - 早停: patience={CONFIG['early_stopping_patience']}")
    print(f"  - 4bit量化: ✓")
    print(f"  - 梯度检查点: ✓")
    print(f"  - 预期显存: 5-6GB")
    
    # Data Collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # 创建Trainer（添加早停）
    print(f"\n🔄 初始化训练器（含早停机制）...")
    
    # 早停回调
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=CONFIG["early_stopping_patience"],
        early_stopping_threshold=CONFIG["early_stopping_threshold"]
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[early_stopping],  # V3: 添加早停回调
    )
    
    # 开始训练
    print(f"\n" + "="*60)
    print("🚀 开始训练 V3（防过拟合优化版）...")
    print("="*60)
    print(f"\n配置变化:")
    print(f"  - Epochs: 5 → 3")
    print(f"  - Learning Rate: 1.5e-4 → 1.0e-4")
    print(f"  - LoRA Dropout: 0.1 → 0.15")
    print(f"  - 早停: 新增（patience=10）")
    print(f"  - 权重衰减: 新增（0.01）")
    print(f"\n预计训练时间: 20-30分钟（3 epochs + 早停可能更快）")
    print(f"可以使用 nvidia-smi 监控显存使用情况\n")
    
    try:
        trainer.train()
        
        print(f"\n" + "="*60)
        print("✅ 训练完成！")
        print("="*60)
        
        # 保存模型
        print(f"\n💾 保存模型...")
        trainer.save_model()
        print(f"✓ 模型已保存至: {CONFIG['output_dir']}")
        
        # 绘制训练曲线
        plot_path = Path("plots")
        plot_path.mkdir(exist_ok=True)
        plot_training_history(trainer, plot_path / "qwen_question_training.png")
        
        # 显示最终指标
        print(f"\n📊 最终训练指标:")
        final_metrics = trainer.state.log_history[-1]
        for key, value in final_metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
        
        print(f"\n🎉 Qwen-Question 训练成功！")
        print(f"\n✅ 双Qwen训练全部完成！")
        print(f"\n下一步:")
        print(f"  使用: python test_dual_qwen.py 进行测试")
        
    except KeyboardInterrupt:
        print(f"\n\n⚠️  训练被用户中断")
        print(f"模型checkpoint已保存至: {CONFIG['output_dir']}")
    except Exception as e:
        print(f"\n\n❌ 训练出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

