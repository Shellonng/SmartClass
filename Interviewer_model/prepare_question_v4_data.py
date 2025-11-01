"""
从qwen_data.json生成Question V4训练数据
基于topic queue架构
"""
import json
import random
import sys
import io

# 修复Windows中文输出
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def infer_topic_info_from_name(topic_name):
    """
    从topic_name生成topic_info（最简单方案）
    
    返回：去掉"（项目）"标记的名称
    """
    # 直接移除"（项目）"标记
    clean_name = topic_name.replace("（项目）", "").strip()
    return clean_name


def prepare_question_v4_data(qwen_data_path, output_train_path, output_val_path):
    """
    从qwen_data.json生成Question V4训练数据（Instruction格式）
    """
    print("="*60)
    print("Question V4 数据准备（Instruction格式 + Topic Queue）")
    print("="*60)
    
    # 1. 读取数据
    print("\n[1/5] 读取qwen_data.json...")
    with open(qwen_data_path, 'r', encoding='utf-8') as f:
        qwen_data = json.load(f)
    print(f"✓ 总样本数: {len(qwen_data)}")
    
    # 2. 转换数据
    print("\n[2/5] 转换数据格式...")
    new_data = []
    
    for idx, sample in enumerate(qwen_data):
        # 提取当前topic（full_history的最后一个）
        current_topic = sample['full_history'][-1]
        topic_name = current_topic['topic_name']
        
        # 推断topic_info（灵活格式）
        topic_info = infer_topic_info_from_name(topic_name)
        
        # 提取当前topic内的对话历史（不包含score）
        topic_history = []
        for round_data in current_topic['rounds']:
            topic_history.append({
                "question": round_data['question'],
                "answer": round_data['answer']
            })
        
        # Action
        action = sample['bert_decision']
        
        # 目标问题
        target_question = sample['question']
        importance = sample['importance']
        
        # 根据action构建instruction-input-output格式
        if action == "FOLLOW_UP":
            # FOLLOW_UP样本
            instruction = "你是一位经验丰富的技术面试官。你的任务是根据当前话题和对话历史，生成下一个深入追问的问题，并评估该问题的重要程度（1-5分，其中1分为闲聊，5分为核心技能考察）。"
            
            # 构建input
            input_text = f"当前话题：{topic_info}\n\n话题内对话历史：\n"
            for i, h in enumerate(topic_history, 1):
                input_text += f"Q: {h['question']}\n"
                input_text += f"A: {h['answer']}\n\n"
            
            input_text += "请根据候选人的回答质量，生成下一个深入追问的问题，并评估其重要程度（1-5分）。"
        
        else:  # SWITCH_TOPIC
            # SWITCH_TOPIC样本
            instruction = "你是一位经验丰富的技术面试官。你的任务是根据新话题信息，生成一个自然过渡的开场问题，并评估该问题的重要程度（1-5分，其中1分为闲聊，5分为核心技能考察）。"
            
            # 构建input
            input_text = f"新话题：{topic_name}\n"
            if topic_info != topic_name:
                input_text += f"话题信息：{topic_info}\n"
            input_text += "\n这是切换到新话题的第一个问题。请生成一个自然引导候选人介绍相关经验的开场问题，并评估其重要程度（1-5分）。"
        
        # 构建output（自然语言格式）
        output_text = f"问题：{target_question}\n\n重要程度：{importance}分"
        
        # 构建新样本
        new_sample = {
            "instruction": instruction,
            "input": input_text,
            "output": output_text
        }
        
        new_data.append(new_sample)
        
        # 进度提示
        if (idx + 1) % 10000 == 0:
            print(f"  已处理: {idx + 1}/{len(qwen_data)}")
    
    print(f"✓ 转换完成: {len(new_data)} 个样本")
    
    # 3. 统计
    print("\n[3/5] 数据统计...")
    # 通过instruction区分类型
    follow_up_count = sum(1 for s in new_data if "当前话题和对话历史" in s['instruction'])
    switch_count = sum(1 for s in new_data if "新话题信息" in s['instruction'])
    print(f"  FOLLOW_UP样本: {follow_up_count} ({follow_up_count/len(new_data)*100:.1f}%)")
    print(f"  SWITCH_TOPIC样本: {switch_count} ({switch_count/len(new_data)*100:.1f}%)")
    
    # 统计input长度
    input_lengths = [len(s['input']) for s in new_data]
    print(f"\n  Input长度统计:")
    print(f"    平均: {sum(input_lengths)/len(input_lengths):.0f} 字符")
    print(f"    最小: {min(input_lengths)} 字符")
    print(f"    最大: {max(input_lengths)} 字符")
    
    # 4. 划分训练集和验证集
    print("\n[4/5] 划分数据集...")
    random.seed(42)
    random.shuffle(new_data)
    
    split_idx = int(len(new_data) * 0.9)
    train_data = new_data[:split_idx]
    val_data = new_data[split_idx:]
    
    print(f"  训练集: {len(train_data)} 样本")
    print(f"  验证集: {len(val_data)} 样本")
    
    # 5. 保存
    print("\n[5/5] 保存数据...")
    with open(output_train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    print(f"✓ 训练集已保存: {output_train_path}")
    
    with open(output_val_path, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    print(f"✓ 验证集已保存: {output_val_path}")
    
    # 6. 显示样本示例
    print("\n" + "="*60)
    print("样本示例")
    print("="*60)
    
    # FOLLOW_UP样本
    follow_up_example = next((s for s in train_data if "当前话题和对话历史" in s['instruction']), None)
    if follow_up_example:
        print("\n【FOLLOW_UP样本】")
        print(f"Instruction: {follow_up_example['instruction'][:80]}...")
        print(f"\nInput (前200字符):\n{follow_up_example['input'][:200]}...")
        print(f"\nOutput:\n{follow_up_example['output']}")
    
    # SWITCH_TOPIC样本
    switch_example = next((s for s in train_data if "新话题信息" in s['instruction']), None)
    if switch_example:
        print("\n【SWITCH_TOPIC样本】")
        print(f"Instruction: {switch_example['instruction'][:80]}...")
        print(f"\nInput:\n{switch_example['input']}")
        print(f"\nOutput:\n{switch_example['output']}")
    
    print("\n" + "="*60)
    print("✅ 数据准备完成！")
    print("="*60)


if __name__ == "__main__":
    prepare_question_v4_data(
        qwen_data_path="training_data/qwen_data.json",
        output_train_path="training_data/question_v4_train.json",
        output_val_path="training_data/question_v4_val.json"
    )

