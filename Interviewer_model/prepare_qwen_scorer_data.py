"""
准备Qwen-Scorer训练数据
从RoBERTa数据转换为Qwen格式
"""

import json
from pathlib import Path
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def convert_roberta_to_qwen_scorer():
    """将RoBERTa数据转换为Qwen-Scorer格式"""
    
    print("="*70)
    print("📊 准备Qwen-Scorer训练数据")
    print("="*70)
    
    # 读取RoBERTa数据
    roberta_file = Path("training_data/roberta_data.json")
    
    if not roberta_file.exists():
        print(f"❌ 文件不存在: {roberta_file}")
        return
    
    with open(roberta_file, 'r', encoding='utf-8') as f:
        roberta_data = json.load(f)
    
    print(f"\n✅ 读取RoBERTa数据: {len(roberta_data)} 条")
    
    # 转换为Qwen格式
    qwen_scorer_data = []
    
    for item in roberta_data:
        question = item['question']
        answer = item['answer']
        score = item['score']
        label = item['label']
        comment = item.get('comment', '')
        
        # 构建输入
        input_text = f"""面试问题: {question}

候选人回答:
{answer}

请评估这个回答的质量，给出评分（0-100分）、标签（excellent/good/average/poor）和评价。"""
        
        # 构建输出
        output_text = f"""评分: {score}分
标签: {label}
评价: {comment}"""
        
        qwen_scorer_data.append({
            "instruction": "你是一位经验丰富的技术面试官。你的任务是评估候选人对技术问题的回答质量，给出评分（0-100分）、标签（excellent/good/average/poor）和详细评价。评分标准：excellent(85-100)表示回答准确、深入、有实战经验；good(70-84)表示回答正确但不够深入；average(50-69)表示回答部分正确或较浅；poor(0-49)表示回答错误或完全不会。",
            "input": input_text,
            "output": output_text,
            "metadata": {
                "question": question,
                "answer": answer,
                "score": score,
                "label": label
            }
        })
    
    print(f"✅ 转换完成: {len(qwen_scorer_data)} 条")
    
    # 划分训练集和验证集（90/10）
    import random
    random.seed(42)
    
    # 打乱数据
    shuffled_data = qwen_scorer_data.copy()
    random.shuffle(shuffled_data)
    
    # 90/10划分
    split_idx = int(len(shuffled_data) * 0.9)
    train_data = shuffled_data[:split_idx]
    val_data = shuffled_data[split_idx:]
    
    print(f"\n📊 数据划分:")
    print(f"  训练集: {len(train_data)} 条")
    print(f"  验证集: {len(val_data)} 条")
    
    # 保存数据
    output_dir = Path("dual_qwen_data")
    output_dir.mkdir(exist_ok=True)
    
    train_file = output_dir / "qwen_scorer_train_split.json"
    val_file = output_dir / "qwen_scorer_val_split.json"
    
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    with open(val_file, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 数据已保存:")
    print(f"  训练集: {train_file}")
    print(f"  验证集: {val_file}")
    
    # 显示样例
    print(f"\n📝 训练数据样例:")
    print(f"{'-'*70}")
    sample = train_data[0]
    print(f"Instruction: {sample['instruction'][:100]}...")
    print(f"\nInput: {sample['input'][:150]}...")
    print(f"\nOutput: {sample['output']}")
    print(f"{'-'*70}")
    
    return len(train_data), len(val_data)

if __name__ == "__main__":
    convert_roberta_to_qwen_scorer()

