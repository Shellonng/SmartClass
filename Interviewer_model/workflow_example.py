"""
AI面试系统工作流程 - 实际运行示例
演示一个完整的问答循环
"""

def simulate_interview_cycle():
    """模拟一个完整的问答循环"""
    
    print("="*80)
    print("[AI Interview System] Workflow Demonstration")
    print("="*80)
    
    # ============ 初始状态 ============
    print("\n【初始状态】")
    current_question = "能说说Python的装饰器吗？"
    current_topic = "Python基础"
    follow_up_depth = 0
    resume_skills = ['Python基础', 'Django', 'MySQL', 'Redis']
    
    print(f"面试官问题: {current_question}")
    print(f"当前话题: {current_topic}")
    print(f"追问深度: {follow_up_depth}")
    
    # ============ 用户回答 ============
    print("\n【用户回答】")
    user_answer = "装饰器是Python的高阶函数，我在项目中用它实现了日志记录和权限校验。"
    print(f"候选人: {user_answer}")
    
    # ============ 阶段1: RoBERTa评估 ============
    print("\n" + "="*80)
    print("【阶段1：RoBERTa评估回答质量】")
    print("="*80)
    
    # 构建输入
    print("\n步骤1.1: 构建评估输入")
    roberta_input = """
[历史问答]
Q1: 你熟悉哪些Python技术栈？
A1: Flask、Django、装饰器、生成器等
质量: 良好

[当前问答]
问题: 能说说Python的装饰器吗？
回答: 装饰器是Python的高阶函数，我在项目中用它实现了日志记录和权限校验。
流畅度: 0.85
"""
    print(f"RoBERTa输入:\n{roberta_input}")
    
    # 模拟推理
    print("\n步骤1.2: RoBERTa模型推理")
    print("  → 多任务推理（分类 + 回归）")
    print("  → 分类logits: [1.2, 2.5, 4.8, 2.1]  (差, 一般, 良好, 优秀)")
    print("  → Softmax后概率: [0.05, 0.15, 0.60, 0.20]")
    print("  → 预测类别: 良好 (index=2, 概率=0.60)")
    print("  → 回归输出: 0.825 * 100 = 82.5分")
    
    # 输出结果
    print("\n步骤1.3: RoBERTa输出结果")
    roberta_result = {
        'current_label': '良好',
        'current_score': 85,        # 映射分数
        'overall_score': 82.5,      # 回归分数
        'confidence': 0.60
    }
    print(f"  ✅ 质量标签: {roberta_result['current_label']}")
    print(f"  ✅ 映射分数: {roberta_result['current_score']}")
    print(f"  ✅ 回归分数: {roberta_result['overall_score']}")
    print(f"  ✅ 置信度: {roberta_result['confidence']:.1%}")
    
    print("\n💡 RoBERTa的作用:")
    print("  1. 评估回答质量（良好 = 85分）")
    print("  2. 这个分数会传给Qwen，影响Qwen的追问方式")
    print("  3. 不直接决策，但提供重要的质量信号")
    
    # ============ 阶段2: BERT决策 ============
    print("\n" + "="*80)
    print("【阶段2：BERT决策下一步动作】")
    print("="*80)
    
    # 特征提取
    print("\n步骤2.1: 提取回答特征")
    answer_length = len(user_answer)
    hesitation_words = ['嗯', '啊', '这个', '那个', '就是']
    hesitation_count = sum(user_answer.count(w) for w in hesitation_words)
    hesitation_score = min(0.9, hesitation_count * 0.15)
    
    print(f"  → 回答长度: {answer_length}字")
    print(f"  → 犹豫词数量: {hesitation_count}个")
    print(f"  → 犹豫度分数: {hesitation_score:.2f}")
    print(f"  → 追问深度: {follow_up_depth}")
    print(f"  → 当前话题: {current_topic}")
    
    # 构建输入
    print("\n步骤2.2: 构建BERT输入")
    features = f"追问深度:{follow_up_depth} 犹豫度:{hesitation_score:.2f} 长度:{answer_length}字 话题:{current_topic}"
    bert_input = f"{current_question}[SEP]{user_answer}[SEP]{features}"
    print(f"BERT输入:\n  {bert_input}")
    
    # 模拟推理
    print("\n步骤2.3: BERT模型推理（二分类）")
    print("  → BERT logits: [2.8, -0.5]  (FOLLOW_UP, NEXT_TOPIC)")
    print("  → Softmax后概率: [0.85, 0.15]")
    print("  → 预测: FOLLOW_UP (概率=0.85)")
    
    # 推断理由
    print("\n步骤2.4: 推断决策理由")
    print("  → 检查关键词: '项目' ✓, '用过' ✓")
    print("  → 回答长度: 37字 < 80字")
    print("  → 匹配规则: '候选者提到了使用场景但缺少细节'")
    
    # 输出结果
    print("\n步骤2.5: BERT输出决策")
    bert_decision = {
        'action': 'FOLLOW_UP',
        'confidence': 0.85,
        'probs': [0.85, 0.15],
        'reason': '候选者提到了使用场景但缺少细节，可以继续追问更深入的技术细节',
        'hesitation_score': 0.0
    }
    print(f"  ✅ 决策动作: {bert_decision['action']}")
    print(f"  ✅ 置信度: {bert_decision['confidence']:.1%}")
    print(f"  ✅ 决策理由: {bert_decision['reason']}")
    print(f"  ✅ 概率分布: FOLLOW_UP={bert_decision['probs'][0]:.1%}, NEXT_TOPIC={bert_decision['probs'][1]:.1%}")
    
    print("\n💡 BERT的作用:")
    print("  1. 根据回答内容、长度、犹豫度、追问深度综合决策")
    print("  2. 决策：继续追问 or 换话题（二选一）")
    print("  3. 不管具体问什么，只管该不该追问")
    print("  4. 控制面试节奏，避免过度追问或过早换题")
    
    # ============ 阶段3: Qwen生成问题 ============
    print("\n" + "="*80)
    print("【阶段3：Qwen生成下一个问题】")
    print("="*80)
    
    # 构建上下文
    print("\n步骤3.1: 构建Qwen上下文")
    qwen_context = {
        'last_answer': user_answer,
        'question': current_question,
        'topic': current_topic,
        'score': roberta_result['current_score']  # ← RoBERTa的评分传入
    }
    print(f"  → 上一个问题: {qwen_context['question']}")
    print(f"  → 候选人回答: {qwen_context['last_answer'][:40]}...")
    print(f"  → 当前话题: {qwen_context['topic']}")
    print(f"  → RoBERTa评分: {qwen_context['score']}分")
    print(f"  → BERT决策: {bert_decision['action']}")
    
    # 根据决策分支
    print("\n步骤3.2: 根据BERT决策选择生成策略")
    print(f"  → BERT决策 = {bert_decision['action']}")
    
    if bert_decision['action'] == 'FOLLOW_UP':
        print("  → 走追问分支")
        print(f"  → RoBERTa评分 = {qwen_context['score']}分")
        
        if qwen_context['score'] >= 80:
            feedback_guide = "候选人回答得不错，可以适当肯定（不要过于客套），然后追问更深入的问题"
            print(f"  → 评分≥80 → 策略: {feedback_guide}")
        elif qwen_context['score'] >= 60:
            feedback_guide = "候选人的回答比较笼统，需要引导他说得更具体"
            print(f"  → 60≤评分<80 → 策略: {feedback_guide}")
        else:
            feedback_guide = "候选人的回答不太理想，可以换个角度问，或者给个提示"
            print(f"  → 评分<60 → 策略: {feedback_guide}")
    
    # 构建Prompt
    print("\n步骤3.3: 构建Qwen Prompt")
    system_msg = f"你是一位经验丰富的{current_topic}技术面试官，面试风格专业、平和，善于引导候选人展示真实水平。"
    user_prompt = f"""你刚才问了候选人："{current_question}"

候选人的回答是："{user_answer}"

{feedback_guide}。请生成你的下一个问题或回复。注意：
- 语气自然，像正常对话
- 不要使用"非常棒！"、"很好！"这种过于热情的表扬
- 如果需要肯定，可以说"嗯，理解了"、"可以"、"听起来不错"等
- 直接问下一个问题，不要啰嗦
- 长度适中（40-80字）

你的回复："""
    
    print(f"System:\n  {system_msg}\n")
    print(f"User:\n  {user_prompt[:200]}...\n")
    
    # 模拟生成
    print("\n步骤3.4: Qwen模型生成")
    print("  → 使用基座模型 (Qwen2-1.5B-Instruct)")
    print("  → 生成参数: max_new_tokens=100, temperature=0.7, top_p=0.9")
    print("  → 生成中...")
    
    # 输出结果
    print("\n步骤3.5: Qwen输出新问题")
    new_question = "嗯，理解了。那在实际项目中，你是怎么处理装饰器的执行顺序问题的？特别是当有多个装饰器叠加的时候？"
    print(f"  ✅ 新问题: {new_question}")
    
    print("\n💡 Qwen的作用:")
    print("  1. 接收BERT决策（FOLLOW_UP）+ RoBERTa评分（85分）")
    print("  2. 根据评分决定语气（85分→适当肯定）")
    print("  3. 自主生成追问问题（关注装饰器执行顺序）")
    print("  4. 确保语气自然（'嗯，理解了'而非'非常棒！'）")
    
    # ============ 阶段4: 更新状态 ============
    print("\n" + "="*80)
    print("【阶段4：更新系统状态】")
    print("="*80)
    
    print("\n步骤4.1: 更新追问深度")
    follow_up_depth = follow_up_depth + 1 if bert_decision['action'] == 'FOLLOW_UP' else 0
    print(f"  → 追问深度: 0 → {follow_up_depth}")
    
    print("\n步骤4.2: 保存问答历史")
    qa_record = {
        'question': current_question,
        'answer': user_answer,
        'quality': roberta_result['current_label'],
        'score': roberta_result['current_score'],
        'action': bert_decision['action']
    }
    print(f"  → 记录: Q='{current_question[:30]}...'")
    print(f"  → 质量: {qa_record['quality']}({qa_record['score']}分)")
    print(f"  → 动作: {qa_record['action']}")
    
    print("\n步骤4.3: 更新当前问题")
    current_question = new_question
    print(f"  → 新问题: {current_question}")
    
    print("\n步骤4.4: 生成语音并展示")
    print(f"  → 调用TTS: text_to_speech('{new_question[:30]}...')")
    print(f"  → 音频文件: audio/tts_{hash(new_question) % 10000}.mp3")
    print(f"  → 展示问题 + 自动播放音频")
    
    # ============ 总结 ============
    print("\n" + "="*80)
    print("【流程总结】")
    print("="*80)
    
    print("\n三个模型的协作:")
    print("  1. RoBERTa评估: '良好(85分)' → 告诉Qwen回答质量不错")
    print("  2. BERT决策: 'FOLLOW_UP(85%置信度)' → 告诉Qwen要继续追问")
    print("  3. Qwen生成: 根据评分(85)和决策(追问) → 生成有肯定+深入的追问")
    
    print("\n关键信息流:")
    print("  用户回答")
    print("    → RoBERTa: 85分(良好)")
    print("    → BERT: FOLLOW_UP + 理由")
    print("    → Qwen: 接收85分 + FOLLOW_UP → 生成'嗯，理解了。那...'")
    print("    → 展示新问题")
    
    print("\n下一轮:")
    print(f"  当前问题: {new_question}")
    print(f"  当前话题: {current_topic}")
    print(f"  追问深度: {follow_up_depth}")
    print(f"  等待用户回答...")
    
    print("\n" + "="*80)
    print("✅ 演示完成！")
    print("="*80)


if __name__ == "__main__":
    simulate_interview_cycle()

