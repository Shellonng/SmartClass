"""
准备双Qwen训练数据
将现有的BERT和Qwen数据转换为适合Qwen-LoRA训练的格式
"""

import json
import sys
import io
from pathlib import Path

# 修复Windows控制台编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def load_json(filepath):
    """加载JSON数据"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"⚠️  文件未找到: {filepath}")
        return []

def prepare_decision_data(bert_data):
    """
    准备Qwen-Decision训练数据
    替代BERT的功能: 输出 action + guidance
    
    输入格式 (BERT):
    {
        "topic": "...",
        "round_number": 3,
        "history": [...],
        "scores": [85, 90, 80],
        "avg_score": 85,
        "recent_trend": "stable",
        "action": "FOLLOW_UP",
        "guidance": "..."
    }
    
    输出格式 (Qwen训练):
    {
        "instruction": "你是一位专业的技术面试官...",
        "input": "当前话题: ...\n对话历史: ...\n评分情况: ...",
        "output": "决策: FOLLOW_UP\n指导建议: ..."
    }
    """
    decision_data = []
    
    for idx, item in enumerate(bert_data):
        # 构建输入
        topic = item.get('topic', '未知话题')
        round_num = item.get('round_number', 0)
        history = item.get('history', [])
        scores = item.get('scores', [])
        avg_score = item.get('avg_score', 0)
        recent_trend = item.get('recent_trend', 'unknown')
        
        # 格式化对话历史（最近3轮）
        history_text = ""
        recent_history = history[-3:] if len(history) > 3 else history
        for h in recent_history:
            q = h.get('question', '')
            a = h.get('answer', '')
            s = h.get('score', 0)
            history_text += f"问: {q}\n答: {a}\n评分: {s}分\n\n"
        
        # 格式化评分情况
        score_text = f"平均分: {avg_score}分\n"
        score_text += f"分数趋势: {recent_trend}\n"
        if scores:
            score_text += f"最近3次: {scores[-3:]}\n"
        
        # 构建输入文本
        input_text = f"""当前话题: {topic}
当前轮次: 第{round_num}轮

对话历史:
{history_text.strip()}

评分情况:
{score_text.strip()}

请根据以上信息，做出面试决策并提供指导建议。"""
        
        # 构建输出（标准答案）
        action = item.get('action', 'SWITCH_TOPIC')
        guidance = item.get('guidance', '...')
        
        output_text = f"""决策: {action}

指导建议: {guidance}"""
        
        # 添加到训练数据
        decision_data.append({
            "instruction": "你是一位经验丰富的技术面试官。你的任务是根据当前面试话题、对话历史和候选人的表现评分，做出面试决策（FOLLOW_UP继续深入 或 SWITCH_TOPIC切换话题），并为问题生成器提供详细的指导建议。",
            "input": input_text,
            "output": output_text
        })
    
    return decision_data

def prepare_question_data(qwen_data):
    """
    准备Qwen-Question训练数据
    原有功能: 输出 question + importance
    
    输入格式 (Qwen):
    {
        "topic": "...",
        "full_history": [...],
        "guidance": "...",
        "question": "...",
        "importance": 4
    }
    
    输出格式 (Qwen训练):
    {
        "instruction": "你是一位专业的技术面试官...",
        "input": "话题: ...\n历史对话: ...\n指导建议: ...",
        "output": "问题: ...\n重要程度: 4"
    }
    """
    question_data = []
    
    for idx, item in enumerate(qwen_data):
        # 构建输入
        topic = item.get('topic', '未知话题')
        full_history = item.get('full_history', [])
        guidance = item.get('guidance', '请提出相关问题')
        
        # 格式化完整对话历史（所有轮次）
        history_text = ""
        for h in full_history:
            q = h.get('question', '')
            a = h.get('answer', '')
            history_text += f"Q: {q}\nA: {a}\n\n"
        
        # 构建输入文本
        input_text = f"""面试话题: {topic}

完整对话历史:
{history_text.strip() if history_text else '（这是第一个问题）'}

决策指导:
{guidance}

请根据以上信息生成下一个面试问题，并评估其重要程度（1-5分）。"""
        
        # 构建输出（标准答案）
        question = item.get('question', '')
        importance = item.get('importance', 3)
        
        output_text = f"""问题: {question}

重要程度: {importance}分"""
        
        # 添加到训练数据
        question_data.append({
            "instruction": "你是一位经验丰富的技术面试官。你的任务是根据面试话题、完整对话历史和决策指导，生成下一个合适的面试问题，并评估该问题的重要程度（1-5分，其中1分为闲聊，5分为核心技能考察）。",
            "input": input_text,
            "output": output_text
        })
    
    return question_data

def save_json(data, filepath):
    """保存JSON数据"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"✓ 已保存: {filepath} ({len(data)} 条)")

def analyze_data(data, name):
    """分析数据统计信息"""
    print(f"\n{'='*60}")
    print(f"📊 {name} 数据统计")
    print(f"{'='*60}")
    print(f"总数据量: {len(data)}")
    
    if data:
        # 计算平均长度
        input_lengths = [len(item['input']) for item in data]
        output_lengths = [len(item['output']) for item in data]
        
        print(f"\n输入文本:")
        print(f"  平均长度: {sum(input_lengths) / len(input_lengths):.0f} 字符")
        print(f"  最短: {min(input_lengths)} 字符")
        print(f"  最长: {max(input_lengths)} 字符")
        
        print(f"\n输出文本:")
        print(f"  平均长度: {sum(output_lengths) / len(output_lengths):.0f} 字符")
        print(f"  最短: {min(output_lengths)} 字符")
        print(f"  最长: {max(output_lengths)} 字符")
        
        # 显示示例
        print(f"\n示例数据 (第1条):")
        print(f"\n【指令】")
        print(data[0]['instruction'][:100] + "...")
        print(f"\n【输入】")
        print(data[0]['input'][:200] + "...")
        print(f"\n【输出】")
        print(data[0]['output'][:200] + "...")

def main():
    print("="*60)
    print("🚀 准备双Qwen训练数据")
    print("="*60)
    
    # 数据路径
    data_dir = Path("training_data")
    output_dir = Path("dual_qwen_data")
    output_dir.mkdir(exist_ok=True)
    
    # 加载原始数据
    print("\n📥 加载原始数据...")
    bert_data = load_json(data_dir / "bert_data.json")
    qwen_data = load_json(data_dir / "qwen_data.json")
    
    print(f"✓ BERT数据: {len(bert_data)} 条")
    print(f"✓ Qwen数据: {len(qwen_data)} 条")
    
    if not bert_data or not qwen_data:
        print("\n❌ 数据加载失败，请检查 training_data 目录")
        return
    
    # 准备决策数据
    print("\n🔄 准备 Qwen-Decision 训练数据...")
    decision_data = prepare_decision_data(bert_data)
    save_json(decision_data, output_dir / "qwen_decision_train.json")
    analyze_data(decision_data, "Qwen-Decision")
    
    # 准备提问数据
    print("\n🔄 准备 Qwen-Question 训练数据...")
    question_data = prepare_question_data(qwen_data)
    save_json(question_data, output_dir / "qwen_question_train.json")
    analyze_data(question_data, "Qwen-Question")
    
    # 划分训练集和验证集
    print("\n🔄 划分训练集和验证集（90% 训练，10% 验证）...")
    
    # Decision数据划分
    split_idx = int(len(decision_data) * 0.9)
    decision_train = decision_data[:split_idx]
    decision_val = decision_data[split_idx:]
    
    save_json(decision_train, output_dir / "qwen_decision_train_split.json")
    save_json(decision_val, output_dir / "qwen_decision_val_split.json")
    
    print(f"\nQwen-Decision:")
    print(f"  训练集: {len(decision_train)} 条")
    print(f"  验证集: {len(decision_val)} 条")
    
    # Question数据划分
    split_idx = int(len(question_data) * 0.9)
    question_train = question_data[:split_idx]
    question_val = question_data[split_idx:]
    
    save_json(question_train, output_dir / "qwen_question_train_split.json")
    save_json(question_val, output_dir / "qwen_question_val_split.json")
    
    print(f"\nQwen-Question:")
    print(f"  训练集: {len(question_train)} 条")
    print(f"  验证集: {len(question_val)} 条")
    
    # 总结
    print("\n" + "="*60)
    print("✅ 数据准备完成！")
    print("="*60)
    print(f"\n输出目录: {output_dir}/")
    print(f"\n文件列表:")
    print(f"  - qwen_decision_train.json (完整决策数据)")
    print(f"  - qwen_question_train.json (完整提问数据)")
    print(f"  - qwen_decision_train_split.json (决策训练集)")
    print(f"  - qwen_decision_val_split.json (决策验证集)")
    print(f"  - qwen_question_train_split.json (提问训练集)")
    print(f"  - qwen_question_val_split.json (提问验证集)")
    
    print(f"\n下一步:")
    print(f"  1. 运行: python train_qwen_decision.py")
    print(f"  2. 运行: python train_qwen_question.py")
    print(f"  3. 使用训练好的模型进行推理")

if __name__ == "__main__":
    main()


