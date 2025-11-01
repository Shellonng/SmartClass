"""
准备Qwen-Scorer V2训练数据
输出格式：只包含 score 和 comment，移除 label
"""
import json
import random
import sys
import io

# 修复Windows中文输出问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def prepare_scorer_v2_data():
    """将roberta_data.json转换为Qwen-Scorer V2训练格式"""
    
    # 读取原始数据
    with open('training_data/roberta_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"原始数据数量: {len(data)}")
    
    # 转换为Qwen格式（移除label）
    qwen_data = []
    for item in data:
        # 构建输入
        input_text = f"""面试问题: {item['question']}

候选人回答:
{item['answer']}

请评估这个回答的质量，给出评分（0-100分）和详细评价。"""
        
        # 构建输出（只有score和comment，移除label）
        output_text = f"""评分: {item['score']}分
评价: {item['comment']}"""
        
        qwen_data.append({
            "input": input_text,
            "output": output_text
        })
    
    # 打乱数据
    random.seed(42)
    random.shuffle(qwen_data)
    
    # 划分训练集和验证集 (90% / 10%)
    split_idx = int(len(qwen_data) * 0.9)
    train_data = qwen_data[:split_idx]
    val_data = qwen_data[split_idx:]
    
    # 保存训练集
    with open('training_data/qwen_scorer_v2_train.json', 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    # 保存验证集
    with open('training_data/qwen_scorer_v2_val.json', 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 训练集: {len(train_data)} 条")
    print(f"✅ 验证集: {len(val_data)} 条")
    print("\n示例数据:")
    print("="*50)
    print("INPUT:")
    print(train_data[0]['input'])
    print("\nOUTPUT:")
    print(train_data[0]['output'])
    print("="*50)

if __name__ == "__main__":
    prepare_scorer_v2_data()

