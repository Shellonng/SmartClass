"""
第二版数据生成脚本 - 改进版
重点改进：
1. 重要度多样化（降低5分比例）
2. 增加拉家常问题（1-2分）
3. 增加回答不好的场景
4. 考虑语音转文字特征（填充词）
"""

import dashscope
from dashscope import Generation
import json
import random
from typing import List, Dict
from tqdm import tqdm
import time

# API Key
dashscope.api_key = "sk-abf39dd471804664b5dce35e722f0857"

# ========================================
# 工具函数
# ========================================

def call_qwen(prompt: str, model: str = "qwen-plus", temperature: float = 0.8, max_tokens: int = 2000) -> str:
    """调用阿里云Qwen API"""
    try:
        response = Generation.call(
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            result_format='message'
        )
        
        if response.status_code == 200:
            return response.output.choices[0].message.content
        else:
            print(f"API调用失败: {response.message}")
            return None
    except Exception as e:
        print(f"API调用异常: {str(e)}")
        return None


def extract_json(text: str) -> dict:
    """从响应中提取JSON"""
    try:
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        return json.loads(text.strip())
    except:
        return None


# ========================================
# 第1步：使用已有简历
# ========================================

def load_existing_resumes():
    """加载已有简历"""
    with open('training_data/resumes.json', 'r', encoding='utf-8') as f:
        return json.load(f)


# ========================================
# 第2步：生成多样化面试对话（改进版）
# ========================================

def generate_interview_v2(resume: Dict, performance_type: str) -> Dict:
    """
    生成面试对话 - 第二版
    
    关键改进：
    1. 明确候选人表现类型（excellent/good/average/poor/nervous）
    2. 多样化话题类型（technical_core/casual_chat等）
    3. 回答包含填充词（根据表现类型）
    4. 重要度分布合理
    """
    
    performance_descriptions = {
        "excellent": "技术扎实，回答流畅准确，很少填充词",
        "good": "技术不错，基本流畅，偶有停顿和填充词（'嗯'、'就是'）",
        "average": "技术一般，经常使用填充词（'额'、'这个'、'那个'），缺少深度",
        "poor": "技术较弱，支支吾吾，大量填充词，避重就轻不直接说不会",
        "nervous": "技术可以但紧张，频繁停顿和填充词（'嗯'、'怎么说呢'）"
    }
    
    # 简化项目信息
    projects_text = ""
    for project in resume['projects']:
        projects_text += f"\n- {project['name']}: {project['background'][:100]}...\n"
        projects_text += f"  技术栈: {', '.join(project['tech_stack'])}\n"
    
    prompt = f"""
你是资深技术面试官，面试{resume['name']}（{resume['experience_level']}级别）。

【候选人信息】
技能: {', '.join([item for skill in resume['skills'] for item in skill['items']])}
项目: {projects_text}

【候选人表现类型】: {performance_type}
表现特征: {performance_descriptions[performance_type]}

【面试任务】
模拟20-30分钟真实语音面试，包含3-4个话题。

【话题类型要求】（必须包含）：
1. 核心技术深挖 (40%)：从项目中选1-2个核心技术深入追问
   - 重要度：4-5分
   - 轮数：3-5轮

2. 常规技能了解 (30%)：了解技术栈使用情况
   - 重要度：3分
   - 轮数：2-3轮

3. 项目经验分享 (20%)：了解实际项目经验
   - 重要度：3-4分
   - 轮数：2-4轮

4. **拉家常/背景了解 (10%)**：缓解紧张、了解背景 ❗新增❗
   - 重要度：1-2分
   - 轮数：1-2轮
   - 示例问题：
     * "你家是哪儿的？平时怎么通勤？"
     * "本科/研究生阶段过得怎么样？"
     * "业余时间有什么爱好？"
     * "为什么选择做{resume['education']['major']}这个专业？"
     * "团队规模多大？氛围怎么样？"
     * "有什么职业规划？"

【回答质量模式 - {performance_type}】：

excellent类型的回答：
  - 流畅、准确、有丰富细节
  - 很少填充词
  - 示例："在这个项目中，我们采用了Redis主从复制加哨兵模式。具体配置是1主2从3哨兵，哨兵负责监控主节点健康状态，当主节点故障时自动进行故障转移..."

good类型的回答：
  - 基本流畅，有一定细节，偶有停顿
  - 少量填充词："嗯"、"就是"、"然后"
  - 示例："我们用的是Redis集群，嗯，主要做缓存。配置的话...就是主从模式，然后通过哨兵来做故障转移，具体来说就是..."

average类型的回答：
  - 偏概念化，缺少实战细节
  - 较多填充词："额"、"这个"、"那个"
  - 示例："额，Redis我们也用过，这个...就是做缓存的嘛。具体怎么配置的...嗯...那个...我不太记得了，应该是运维配的吧..."

poor类型的回答：
  - 支支吾吾，大量填充词表示不确定
  - 不直接说"不会"，而是避重就轻
  - 示例："这个...额...Redis是吧？嗯...我知道它是缓存，但具体的...那个...我们项目里好像有用，但我没直接接触过...这个...我不太清楚..."

nervous类型的回答：
  - 内容可以但停顿多，紧张
  - 频繁使用填充词表示思考
  - 示例："额...让我想想...Redis的话，我们是用主从复制，嗯...对，还有哨兵。就是...怎么说呢...主要是为了高可用，然后...那个...故障时可以自动切换..."

【非语言填充词】（根据表现类型适当使用）：
- 思考/停顿："额"、"嗯"、"呃"、"啊"
- 不确定："这个"、"那个"、"好像"、"我记得"
- 组织语言："就是"、"然后"、"怎么说呢"、"让我想想"

【关键要求】：
1. 每轮只问1个问题
2. 问题长度：80-150字
3. 回答长度：根据表现类型，poor类型更短更模糊
4. 拉家常问题的回答通常比较自然流畅
5. 技术问题的回答根据{performance_type}类型调整质量

输出JSON：
{{
  "candidate_id": {resume['id']},
  "candidate_name": "{resume['name']}",
  "candidate_level": "{resume['experience_level']}",
  "performance_type": "{performance_type}",
  "topics": [
    {{
      "topic_name": "Redis缓存优化（核心技术）",
      "topic_type": "technical_core",
      "rounds": [
        {{
          "round_number": 1,
          "question": "...",
          "answer": "...",  # 根据{performance_type}生成
          "answer_quality": "average"
        }}
      ]
    }},
    {{
      "topic_name": "个人背景了解",
      "topic_type": "casual_chat",
      "rounds": [
        {{
          "round_number": 1,
          "question": "你家是哪儿的？平时怎么上班？",
          "answer": "我家是北京的，平时地铁上班，大概40分钟吧。",
          "answer_quality": "good"  # 拉家常通常比较自然
        }}
      ]
    }}
  ]
}}
"""
    
    response_text = call_qwen(prompt, model="qwen-max", temperature=0.85, max_tokens=8000)
    if response_text:
        return extract_json(response_text)
    return None


# ========================================
# 第3步：生成标注（改进版）
# ========================================

def generate_roberta_annotation_v2(question: str, answer: str, quality: str) -> Dict:
    """生成RoBERTa标注 - 考虑语音特征"""
    
    quality_map = {
        "excellent": (90, 100),
        "good": (75, 89),
        "average": (60, 74),
        "poor": (40, 59)
    }
    score_range = quality_map.get(quality, (70, 79))
    score_hint = random.randint(score_range[0], score_range[1])
    
    prompt = f"""
评估回答质量（语音转文字场景，建议{score_hint}分）：

问题：{question[:200]}
回答：{answer[:300]}

【评分标准 - 考虑语音特征】：

90-100分（优秀）：
  - 内容准确、深入、有丰富细节
  - 表达流畅，很少填充词
  - 逻辑清晰，有实战经验

75-89分（良好）：
  - 内容基本准确，有一定深度
  - 偶有停顿和填充词（"嗯"、"就是"）
  - 整体表达可以理解

60-74分（一般）：
  - 内容偏概念化，缺少实战细节
  - 较多填充词（"额"、"这个"、"那个"）
  - 表达不够清晰

40-59分（较差）：
  - 支支吾吾，大量填充词表示不确定
  - "额...嗯...这个...那个..." 表明不熟悉
  - 避重就轻，实际上不了解该领域

【特别识别 - 不熟悉的表现】：
- "额...这个...我好像...但具体的..."
- "嗯...那个...我记得...应该是..."
- "这个问题...怎么说呢...我不太确定..."
- 大量填充词 + 模糊表达 = 较差

输出JSON：
{{
  "score": <{score_range[0]}-{score_range[1]}>,
  "label": "<优秀/良好/一般/较差>",
  "comment": "<80-120字，如果有大量填充词需指出候选人可能不熟悉该领域>"
}}
"""
    
    response_text = call_qwen(prompt, model="qwen-turbo", temperature=0.3, max_tokens=300)
    result = extract_json(response_text)
    
    if not result:
        result = {}
    
    return {
        "score": result.get("score", score_range[0]),
        "label": result.get("label", "一般"),
        "comment": result.get("comment", "回答基本合理，涵盖了主要要点。")
    }


def generate_bert_annotation_v2(topic_rounds: List[Dict], topic_type: str) -> Dict:
    """生成BERT标注 - 考虑话题类型"""
    
    rounds_text = "\n".join([
        f"第{r['round_number']}轮: Q=\"{r['question'][:50]}...\" 分数={r['roberta_score']}"
        for r in topic_rounds
    ])
    
    avg_score = sum(r['roberta_score'] for r in topic_rounds) / len(topic_rounds)
    
    # 拉家常类话题通常不需要深挖
    if topic_type == "casual_chat":
        decision_hint = "拉家常类话题通常1-2轮即可，应该SWITCH_TOPIC切换到技术话题"
    else:
        decision_hint = ""
    
    prompt = f"""
决定面试节奏：

话题类型: {topic_type}
{rounds_text}
总轮数：{len(topic_rounds)}
平均分：{avg_score:.1f}

{decision_hint}

【真实面试场景的决策逻辑】：

SWITCH_TOPIC（切换话题）的情况：
1. 候选人答得很好，已充分展现能力
2. 候选人明显不熟悉，继续问也问不出东西
3. 拉家常类话题已问1-2轮
4. 已问3轮以上且充分了解

FOLLOW_UP（继续追问）的情况：
1. 候选人有一定理解但还可更深入
2. 才问1-2轮，需要继续了解
3. 表现起伏，需要再确认

任务：
1. 决策：FOLLOW_UP 或 SWITCH_TOPIC
2. 指导：针对最近一轮，给Qwen的指导（100-150字）

输出JSON：
{{
  "action": "<FOLLOW_UP或SWITCH_TOPIC>",
  "guidance": "<指导内容>"
}}
"""
    
    response_text = call_qwen(prompt, model="qwen-plus", temperature=0.4, max_tokens=400)
    result = extract_json(response_text)
    
    if not result:
        result = {}
    
    return {
        "action": result.get("action", "FOLLOW_UP"),
        "guidance": result.get("guidance", "继续深入追问，了解更多细节。")
    }


def generate_qwen_annotation_v2(history: List[Dict], bert: Dict, topic_type: str, rounds_count: int) -> Dict:
    """生成Qwen标注 - 重要度多样化"""
    
    history_text = ""
    for topic in history[-2:]:
        history_text += f"\n【{topic['topic_name']}】\n"
        for r in topic['rounds'][-2:]:
            history_text += f"Q: {r['question'][:80]}...\nA: {r['answer'][:80]}...\n"
    
    prompt = f"""
生成下一个问题：

最近对话：{history_text}

BERT决策：{bert.get('action', 'FOLLOW_UP')}
BERT指导：{bert.get('guidance', '继续深入追问')}

当前话题类型: {topic_type}
已问轮数: {rounds_count}

【重要程度判定标准】（必须多样化）：

5分 (20%)：核心技术深挖、生产问题、架构设计
  - "请详细说说你们的分布式事务解决方案，包括具体的实现细节"
  - "在高并发场景下，你们遇到过什么性能瓶颈？如何定位和解决的？"

4分 (25%)：重要技能验证、实战经验考察
  - "你们的微服务架构中，服务间通信是怎么设计的？"
  - "Redis缓存一致性你们是如何保证的？有遇到什么问题吗？"

3分 (30%)：常规技能了解、框架使用
  - "你用过哪些Python Web框架？它们有什么区别？"
  - "Django的ORM你觉得有什么优缺点？"

2分 (15%)：辅助信息、软技能、团队情况
  - "你们团队规模多大？你在其中负责什么？"
  - "平时怎么学习新技术的？有看什么技术博客吗？"
  - "遇到技术分歧时，团队怎么解决？"

1分 (10%)：拉家常、背景了解、缓解紧张
  - "你家是哪儿的？平时怎么通勤上班？"
  - "为什么当时选择学这个专业？"
  - "业余时间有什么爱好？"
  - "本科/研究生阶段过得怎么样？"

【判定规则】：
- 如果 topic_type == "casual_chat" → 必须1-2分
- 如果 topic_type == "technical_core" → 必须4-5分
- 如果 topic_type == "technical_general" → 必须3分
- 如果是首轮问题 → 3分左右
- 如果候选人答得不好 → 2-3分
- 如果候选人答得很好 → 4-5分

❗重要：请根据话题类型和上下文，生成不同重要度的问题❗

任务：
1. 生成问题（80-150字，只问1个具体问题）
2. 判断重要程度（1-5，必须根据上述标准）

输出JSON：
{{
  "question": "<问题>",
  "importance": <1-5，必须多样化>
}}
"""
    
    response_text = call_qwen(prompt, model="qwen-plus", temperature=0.7, max_tokens=500)
    result = extract_json(response_text)
    
    if not result:
        result = {}
    
    # 根据话题类型强制调整重要度
    importance = result.get("importance", 3)
    if topic_type == "casual_chat" and importance > 2:
        importance = random.choice([1, 2])
    elif topic_type == "technical_core" and importance < 4:
        importance = random.choice([4, 5])
    
    return {
        "question": result.get("question", "能详细说说这个问题吗？"),
        "importance": importance
    }


# ========================================
# 主流程
# ========================================

def generate_training_data_v2(
    resume_count: int = 50,
    output_dir: str = "./training_data"
):
    """
    生成第二批训练数据
    
    改进点：
    1. 使用已有简历
    2. 多样化候选人表现类型
    3. 增加拉家常问题
    4. 重要度分布合理
    5. 考虑语音转文字特征
    """
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print(f"生成第二批训练数据（改进版）")
    print(f"目标: {resume_count}份简历")
    print(f"API Key: {dashscope.api_key[:20]}...")
    print("=" * 60)
    
    # 步骤1：加载已有简历
    print("\n【步骤1】加载已有简历...")
    resumes = load_existing_resumes()
    print(f"[OK] 加载{len(resumes)}份简历")
    
    # 随机选择指定数量
    selected_resumes = random.sample(resumes, min(resume_count, len(resumes)))
    print(f"[OK] 随机选择{len(selected_resumes)}份用于第二批数据生成")
    
    # 步骤2：生成多样化的面试对话
    print("\n【步骤2】生成面试对话...")
    print("[提示] 会生成多种表现类型：excellent/good/average/poor/nervous")
    
    # 加载已有数据
    roberta_file = f'{output_dir}/roberta_data.json'
    bert_file = f'{output_dir}/bert_data.json'
    qwen_file = f'{output_dir}/qwen_data.json'
    
    # 加载已有数据
    with open(roberta_file, 'r', encoding='utf-8') as f:
        all_roberta = json.load(f)
    with open(bert_file, 'r', encoding='utf-8') as f:
        all_bert = json.load(f)
    with open(qwen_file, 'r', encoding='utf-8') as f:
        all_qwen = json.load(f)
    
    print(f"[OK] 加载已有数据: RoBERTa={len(all_roberta)}, BERT={len(all_bert)}, Qwen={len(all_qwen)}")
    
    # 定义表现类型分布
    performance_types = (
        ["excellent"] * 8 +
        ["good"] * 17 +
        ["average"] * 15 +
        ["poor"] * 8 +
        ["nervous"] * 2
    )
    random.shuffle(performance_types)
    
    for i, resume in enumerate(tqdm(selected_resumes, desc="处理简历")):
        performance_type = performance_types[i % len(performance_types)]
        
        print(f"\n{i+1}. {resume['name']} ({resume['experience_level']}) - 表现类型: {performance_type}")
        
        try:
            interview = generate_interview_v2(resume, performance_type)
            if not interview:
                print(f"  [警告] 面试生成失败，跳过")
                continue
        except Exception as e:
            print(f"  [错误] 面试生成异常: {str(e)}，跳过")
            continue
        
        full_history = []
        
        try:
            for topic in interview['topics']:
                print(f"  话题: {topic['topic_name']} ({topic.get('topic_type', 'unknown')})")
                topic_rounds = []
                
                for round_data in topic['rounds']:
                    if 'question' not in round_data or 'answer' not in round_data:
                        print(f"    [警告] 问答数据不完整，跳过该轮")
                        continue
                    
                    # RoBERTa
                    roberta = generate_roberta_annotation_v2(
                        round_data['question'],
                        round_data['answer'],
                        round_data.get('answer_quality', 'good')
                    )
                    
                    if roberta and 'score' in roberta:
                        roberta_score = roberta.get('score', 75)
                        all_roberta.append({
                            'question': round_data['question'],
                            'answer': round_data['answer'],
                            'score': roberta_score,
                            'label': roberta.get('label', '一般'),
                            'comment': roberta.get('comment', '回答基本合理。')
                        })
                    else:
                        roberta_score = 75
                    
                    topic_rounds.append({
                        'round_number': round_data['round_number'],
                        'question': round_data['question'],
                        'answer': round_data['answer'],
                        'roberta_score': roberta_score
                    })
                    
                    # BERT
                    bert = generate_bert_annotation_v2(topic_rounds, topic.get('topic_type', 'unknown'))
                    
                    if bert and 'action' in bert:
                        all_bert.append({
                            'topic_name': topic['topic_name'],
                            'rounds': topic_rounds.copy(),
                            'action': bert.get('action', 'FOLLOW_UP'),
                            'guidance': bert.get('guidance', '继续深入追问。')
                        })
                    else:
                        bert = {'action': 'FOLLOW_UP', 'guidance': '继续深入追问。'}
                    
                    # Qwen
                    if round_data != topic['rounds'][-1]:
                        qwen = generate_qwen_annotation_v2(
                            full_history + [{'topic_name': topic['topic_name'], 'rounds': topic_rounds}],
                            bert,
                            topic.get('topic_type', 'unknown'),
                            len(topic_rounds)
                        )
                        
                        if qwen and 'question' in qwen:
                            all_qwen.append({
                                'full_history': full_history + [{'topic_name': topic['topic_name'], 'rounds': topic_rounds}],
                                'bert_decision': bert['action'],
                                'bert_guidance': bert['guidance'],
                                'question': qwen.get('question', '继续追问'),
                                'importance': qwen.get('importance', 3)
                            })
                
                full_history.append({'topic_name': topic['topic_name'], 'rounds': topic_rounds})
                
        except Exception as e:
            print(f"  [错误] 处理面试数据时出现异常: {str(e)}")
            print(f"  [提示] 跳过该简历，继续处理下一份")
            continue
        
        # 增量保存
        with open(roberta_file, 'w', encoding='utf-8') as f:
            json.dump(all_roberta, f, ensure_ascii=False, indent=2)
        with open(bert_file, 'w', encoding='utf-8') as f:
            json.dump(all_bert, f, ensure_ascii=False, indent=2)
        with open(qwen_file, 'w', encoding='utf-8') as f:
            json.dump(all_qwen, f, ensure_ascii=False, indent=2)
        
        print(f"  [OK] RoBERTa: {len(all_roberta)} | BERT: {len(all_bert)} | Qwen: {len(all_qwen)} [已保存]")
        time.sleep(1)
    
    print("\n" + "=" * 60)
    print("[OK] 第二批数据生成完成！")
    print(f"RoBERTa: {len(all_roberta)} 条（累计）")
    print(f"BERT: {len(all_bert)} 条（累计）")
    print(f"Qwen: {len(all_qwen)} 条（累计）")
    print(f"保存位置: {output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    # 生成第二批数据（从已有100份简历中随机选50份）
    generate_training_data_v2(
        resume_count=50,  # 从100份中随机选50份
        output_dir="./training_data"  # 追加到现有数据
    )
    
    print("\n[OK] 数据生成完成！")
    print("请运行 python test_diversity.py 查看改进效果")

