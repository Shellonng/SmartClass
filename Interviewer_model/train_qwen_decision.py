"""
训练 Qwen-Decision 模型（替代BERT）
功能: 根据对话历史和评分，输出决策(action)和指导建议(guidance)

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
    DataCollatorForSeq2Seq
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

# 配置
CONFIG = {
    # 模型配置
    "base_model": "Qwen/Qwen2-1.5B-Instruct",  # 使用Qwen2，不需要自定义代码
    "max_length": 1024,
    
    # LoRA配置
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    
    # 训练配置（针对8GB显存优化 + 更好收敛）
    "batch_size": 2,              # 小batch
    "gradient_accumulation": 4,   # 梯度累积，等效batch=8
    "num_epochs": 5,              # 增加到5个epochs
    "learning_rate": 1.5e-4,      # 降低学习率，更稳定收敛
    "warmup_steps": 50,           # 减少warmup，更快进入学习
    "logging_steps": 5,           # 更频繁的日志
    "save_steps": 100,            # 每100步保存
    "eval_steps": 10,             # 每10步验证一次
    
    # 4bit量化配置
    "use_4bit": True,
    "bnb_4bit_compute_dtype": torch.float16,
    "bnb_4bit_quant_type": "nf4",
    
    # 路径
    "train_data": "dual_qwen_data/qwen_decision_train_split.json",
    "val_data": "dual_qwen_data/qwen_decision_val_split.json",
    "output_dir": "checkpoints/qwen_decision_lora",
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
        padding=False,  # 动态padding
        return_tensors=None
    )
    
    # 设置labels（用于计算loss）
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
    ax.set_title('Qwen-Decision Training History', fontsize=16, fontweight='bold', pad=20)
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
    print("🚀 训练 Qwen-Decision 模型（替代BERT）")
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
    
    # Qwen tokenizer可能没有pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"✓ 分词器加载完成")
    
    # 分词
    print(f"\n🔄 对数据进行分词...")
    
    def tokenize_batch(examples):
        # 将字典列表转换为单个example的列表
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
        bnb_4bit_use_double_quant=True,  # 双重量化，进一步节省显存
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
    
    # 准备模型以进行k-bit训练
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
        eval_steps=CONFIG["eval_steps"],  # 使用独立的eval_steps
        eval_strategy="steps",  # 新版本使用eval_strategy
        save_strategy="steps",
        load_best_model_at_end=True,
        fp16=True,  # 混合精度训练
        gradient_checkpointing=True,  # 梯度检查点，用时间换空间
        optim="paged_adamw_8bit",  # 8bit优化器
        max_grad_norm=0.3,
        logging_dir=f"{CONFIG['output_dir']}/logs",
        report_to="none",  # 不使用wandb等
    )
    
    print(f"✓ 训练配置:")
    print(f"  - Batch size: {CONFIG['batch_size']}")
    print(f"  - 梯度累积: {CONFIG['gradient_accumulation']} (等效batch={CONFIG['batch_size']*CONFIG['gradient_accumulation']})")
    print(f"  - Epochs: {CONFIG['num_epochs']}")
    print(f"  - Learning rate: {CONFIG['learning_rate']}")
    print(f"  - 4bit量化: ✓")
    print(f"  - 梯度检查点: ✓")
    print(f"  - 预期显存: 5-6GB")
    
    # Data Collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # 创建Trainer
    print(f"\n🔄 初始化训练器...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # 开始训练
    print(f"\n" + "="*60)
    print("🚀 开始训练...")
    print("="*60)
    print(f"\n预计训练时间: 3-4小时")
    print(f"可以使用 nvidia-smi 监控显存使用情况\n")
    
    try:
        trainer.train()
        
        print(f"\n" + "="*60)
        print("✅ 训练完成！")
        print("="*60)
        
        # 保存最终模型
        print(f"\n💾 保存模型...")
        trainer.save_model()
        print(f"✓ 模型已保存至: {CONFIG['output_dir']}")
        
        # 绘制训练曲线
        plot_path = Path("plots")
        plot_path.mkdir(exist_ok=True)
        plot_training_history(trainer, plot_path / "qwen_decision_training.png")
        
        # 显示最终指标
        print(f"\n📊 最终训练指标:")
        final_metrics = trainer.state.log_history[-1]
        for key, value in final_metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
        
        print(f"\n🎉 Qwen-Decision 训练成功！")
        print(f"\n下一步:")
        print(f"  1. 运行: python train_qwen_question.py")
        print(f"  2. 训练完成后，使用: python test_dual_qwen.py 进行测试")
        
    except KeyboardInterrupt:
        print(f"\n\n⚠️  训练被用户中断")
        print(f"模型checkpoint已保存至: {CONFIG['output_dir']}")
    except Exception as e:
        print(f"\n\n❌ 训练出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

