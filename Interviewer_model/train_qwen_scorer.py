"""
训练Qwen-Scorer模型（评分+标签+评价生成）
使用V3优化配置：防过拟合
"""

import json
import torch
import sys
import io
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
import numpy as np

# 修复Windows控制台编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 配置（V3优化版：防过拟合）
CONFIG = {
    # 模型配置
    "base_model": "Qwen/Qwen2-1.5B-Instruct",
    "max_length": 1024,
    
    # LoRA配置（增强正则化）
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.15,  # V3: 增强正则化
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", 
                            "gate_proj", "up_proj", "down_proj"],
    
    # 训练配置V3（防过拟合）
    "batch_size": 2,
    "gradient_accumulation": 4,
    "num_epochs": 3,  # V3: 从5降至3
    "learning_rate": 1.0e-4,  # V3: 降低学习率
    "warmup_steps": 30,
    "logging_steps": 5,
    "save_steps": 50,  # V3: 更频繁保存
    "eval_steps": 10,
    
    # 早停配置（新增）
    "early_stopping_patience": 10,
    "early_stopping_threshold": 0.001,
    
    # 4bit量化配置
    "use_4bit": True,
    "bnb_4bit_compute_dtype": torch.float16,
    "bnb_4bit_quant_type": "nf4",
    
    # 路径
    "train_data": "dual_qwen_data/qwen_scorer_train_split.json",
    "val_data": "dual_qwen_data/qwen_scorer_val_split.json",
    "output_dir": "checkpoints/qwen_scorer_lora",
}

def print_trainable_parameters(model):
    """打印可训练参数信息"""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    print(f"\n📊 模型参数统计:")
    print(f"  可训练参数: {trainable_params:,} ({100 * trainable_params / all_param:.2f}%)")
    print(f"  总参数: {all_param:,}")
    print(f"  节省参数: {100 * (1 - trainable_params / all_param):.2f}%")

def load_data():
    """加载训练数据"""
    print(f"\n📥 加载训练数据...")
    
    with open(CONFIG["train_data"], 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    with open(CONFIG["val_data"], 'r', encoding='utf-8') as f:
        val_data = json.load(f)
    
    print(f"✓ 训练集: {len(train_data)} 条")
    print(f"✓ 验证集: {len(val_data)} 条")
    
    return train_data, val_data

def prepare_dataset(data, tokenizer):
    """准备数据集"""
    
    def format_prompt(instruction, input_text, output_text=None):
        """格式化为Qwen2对话格式"""
        if output_text:
            # 训练时包含输出
            return f"""<|im_start|>system
{instruction}<|im_end|>
<|im_start|>user
{input_text}<|im_end|>
<|im_start|>assistant
{output_text}<|im_end|>"""
        else:
            # 推理时不包含输出
            return f"""<|im_start|>system
{instruction}<|im_end|>
<|im_start|>user
{input_text}<|im_end|>
<|im_start|>assistant
"""
    
    formatted_data = []
    for item in data:
        prompt = format_prompt(
            item['instruction'],
            item['input'],
            item['output']
        )
        formatted_data.append({"text": prompt})
    
    return Dataset.from_list(formatted_data)

def tokenize_function(examples, tokenizer):
    """分词函数"""
    result = tokenizer(
        examples["text"],
        truncation=True,
        max_length=CONFIG["max_length"],
        padding=False,
    )
    result["labels"] = result["input_ids"].copy()
    return result

def plot_training_history(log_history, output_path):
    """绘制训练曲线"""
    
    # 提取训练和验证loss
    train_steps = []
    train_losses = []
    eval_steps = []
    eval_losses = []
    
    for entry in log_history:
        if 'loss' in entry and 'step' in entry:
            train_steps.append(entry['step'])
            train_losses.append(entry['loss'])
        if 'eval_loss' in entry and 'step' in entry:
            eval_steps.append(entry['step'])
            eval_losses.append(entry['eval_loss'])
    
    if not train_losses or not eval_losses:
        print("⚠️  警告: 没有足够的数据绘制训练曲线")
        return
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 训练loss
    ax.plot(train_steps, train_losses,
            label='Training Loss',
            color='#2E86AB',
            linewidth=2,
            alpha=0.8)
    
    # 验证loss
    ax.plot(eval_steps, eval_losses,
            label='Validation Loss',
            color='#A23B72',
            linewidth=2.5,
            marker='o',
            markersize=4,
            alpha=0.9)
    
    # 标记最佳点
    if eval_losses:
        best_idx = np.argmin(eval_losses)
        best_step = eval_steps[best_idx]
        best_loss = eval_losses[best_idx]
        
        ax.scatter([best_step], [best_loss],
                  color='#F18F01',
                  s=200,
                  marker='*',
                  zorder=5,
                  label=f'Best Model (Step {best_step}, Loss {best_loss:.4f})')
    
    ax.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax.set_title('Qwen-Scorer V3 Training Progress (Anti-Overfitting)',
                 fontsize=14, fontweight='bold', pad=20)
    
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 添加文本框显示最终指标
    final_train = np.mean(train_losses[-10:]) if len(train_losses) >= 10 else np.mean(train_losses)
    final_eval = eval_losses[-1]
    gap = final_train - final_eval
    
    textstr = f'Final Metrics:\n'
    textstr += f'Train Loss: {final_train:.4f}\n'
    textstr += f'Eval Loss: {final_eval:.4f}\n'
    textstr += f'Gap: {gap:.4f}\n'
    textstr += f'Best Eval: {best_loss:.4f}'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props, family='monospace')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ 训练曲线已保存: {output_path}")
    plt.close()

def main():
    """主训练流程"""
    print("="*60)
    print("🚀 训练 Qwen-Scorer 模型（评分+评价生成）")
    print("="*60)
    
    # 检查GPU
    if torch.cuda.is_available():
        print(f"\n✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("\n⚠️  警告: 未检测到GPU，训练将非常缓慢")
    
    # 加载数据
    train_data, val_data = load_data()
    
    # 加载分词器
    print(f"\n📥 加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["base_model"])
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"✓ 分词器加载完成")
    
    # 准备数据集
    print(f"\n🔄 对数据进行分词...")
    train_dataset = prepare_dataset(train_data, tokenizer)
    val_dataset = prepare_dataset(val_data, tokenizer)
    
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="处理训练集"
    )
    
    val_dataset = val_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=val_dataset.column_names,
        desc="处理验证集"
    )
    
    print(f"`torch_dtype` is deprecated! Use `dtype` instead!")
    print(f"✓ 分词完成")
    
    # 配置4bit量化
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
    )
    
    model = prepare_model_for_kbit_training(model)
    print(f"✓ 基座模型加载完成")
    
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
        eval_steps=CONFIG["eval_steps"],
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        max_grad_norm=0.3,
        weight_decay=0.01,  # V3: L2正则化
        logging_dir=f"{CONFIG['output_dir']}/logs",
        report_to="none",
    )
    
    print(f"✓ 训练配置 (V3 - 防过拟合优化):")
    print(f"  - Batch size: {CONFIG['batch_size']}")
    print(f"  - 梯度累积: {CONFIG['gradient_accumulation']} (等效batch={CONFIG['batch_size']*CONFIG['gradient_accumulation']})")
    print(f"  - Epochs: {CONFIG['num_epochs']}")
    print(f"  - Learning rate: {CONFIG['learning_rate']}")
    print(f"  - LoRA dropout: {CONFIG['lora_dropout']}")
    print(f"  - 早停: patience={CONFIG['early_stopping_patience']}")
    print(f"  - 权重衰减: 0.01")
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
        callbacks=[early_stopping],
    )
    
    # 开始训练
    print(f"\n" + "="*60)
    print("🚀 开始训练 V3（防过拟合优化版）...")
    print("="*60)
    print(f"\n配置特点:")
    print(f"  - Epochs: 3（避免过拟合）")
    print(f"  - Learning Rate: 1.0e-4（稳定训练）")
    print(f"  - LoRA Dropout: 0.15（增强正则化）")
    print(f"  - 早停: patience=10")
    print(f"  - 权重衰减: 0.01（L2正则）")
    print(f"\n预计训练时间: 25-35分钟（数据量: {len(train_data)}条）")
    print(f"可以使用 nvidia-smi 监控显存使用情况\n")
    
    try:
        trainer.train()
        
        print(f"\n" + "="*60)
        print("✅ 训练完成！")
        print("="*60)
        
        # 保存模型
        print(f"\n💾 保存模型...")
        trainer.save_model(CONFIG["output_dir"])
        tokenizer.save_pretrained(CONFIG["output_dir"])
        print(f"✓ 模型已保存至: {CONFIG['output_dir']}")
        
        # 绘制训练曲线
        plot_path = Path("plots/qwen_scorer_training.png")
        plot_path.parent.mkdir(exist_ok=True)
        
        if hasattr(trainer.state, 'log_history'):
            plot_training_history(trainer.state.log_history, plot_path)
        
        # 显示最终指标
        if trainer.state.log_history:
            final_report = [x for x in trainer.state.log_history if 'train_loss' in x]
            if final_report:
                final_metrics = final_report[-1]
                print(f"\n📊 最终训练指标:")
                for key, value in final_metrics.items():
                    if isinstance(value, float):
                        print(f"  {key}: {value:.4f}")
                    else:
                        print(f"  {key}: {value}")
        
        print(f"\n🎉 Qwen-Scorer 训练成功！")
        
        print(f"\n✅ 三个Qwen模型训练全部完成！")
        print(f"\n已训练模型:")
        print(f"  1. Qwen-Decision (决策+指导)")
        print(f"  2. Qwen-Question (提问+重要性)")
        print(f"  3. Qwen-Scorer (评分+评价) [NEW]")
        
        print(f"\n下一步:")
        print(f"  使用: python test_triple_qwen.py 进行完整测试")
        
    except KeyboardInterrupt:
        print(f"\n\n⚠️  训练被中断")
        print(f"已保存的checkpoint可以在 {CONFIG['output_dir']} 找到")
    except Exception as e:
        print(f"\n\n❌ 训练出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


