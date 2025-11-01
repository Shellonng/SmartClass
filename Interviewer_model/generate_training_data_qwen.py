"""
使用阿里云Qwen API生成高质量训练数据（改进版）
- 每次只问一个问题，不在一个问题里包含多个子问题
- 问题长度：80-150字
- 回答长度：200-400字
"""

import dashscope
from dashscope import Generation
import json
import random
from typing import List, Dict
from tqdm import tqdm
import time

# ========================================
# 配置API Key
# ========================================

# 从阿里云百炼控制台获取：https://dashscope.console.aliyun.com/
dashscope.api_key = "sk-abf39dd471804664b5dce35e722f0857"

# ========================================
# 工具函数
# ========================================

def call_qwen(
    prompt: str,
    model: str = "qwen-plus",
    temperature: float = 0.8,
    max_tokens: int = 2000
) -> str:
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
# 第1步：生成候选人简历
# ========================================

def generate_resume(position: str, level: str, candidate_id: int) -> Dict:
    """生成候选人简历"""
    
    level_specs = {
        "junior": "0-2年工作经验，基础扎实但经验有限",
        "mid": "2-4年工作经验，能独立开发，有项目经验",
        "senior": "4年以上经验，技术深入，有架构能力"
    }
    
    prompt = f"""
创建一个{position}候选人简历（{level}级别：{level_specs[level]}）。

要求：
1. 基本信息：姓名、学历、毕业时间
2. 技能列表：5-8个，分类清晰
3. 项目经历：2-3个，每个包含背景、技术栈、职责、亮点
4. 工作经历：与经验等级匹配

输出JSON：
{{
  "id": {candidate_id},
  "name": "<姓名>",
  "experience_level": "{level}",
  "education": {{"degree": "<本科/硕士>", "major": "<专业>", "graduation_year": <年份>}},
  "skills": [{{"category": "编程语言", "items": ["Python", "Java"]}}, ...],
  "projects": [
    {{
      "name": "<项目名>",
      "background": "<背景150字>",
      "tech_stack": ["Python", "Django", "MySQL"],
      "responsibilities": "<职责200字>",
      "highlights": ["<亮点1>", "<亮点2>"]
    }}
  ]
}}
"""
    
    response_text = call_qwen(prompt, model="qwen-plus", temperature=0.8, max_tokens=3000)
    if response_text:
        return extract_json(response_text)
    return None


# ========================================
# 第2步：生成面试对话（改进版 - 每次只问1个问题）
# ========================================

def generate_interview(resume: Dict) -> Dict:
    """
    生成面试对话（改进版）
    
    关键改进：
    - 每轮只问1个问题（不要"首先...其次...最后..."）
    - 问题长度：80-150字
    - 回答长度：200-400字
    """
    
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

【面试任务】
模拟30分钟真实面试，包含3-4个话题。

**重要：模拟真实面试的节奏，不要固定轮数！**

每个话题的轮数应该根据候选人的表现动态调整：

1. **候选人表现很好（2-3轮即可）**
   - 第1轮：基础问题，回答excellent/good
   - 第2轮：深入追问，回答excellent，展现了深度
   - 第3轮（可选）：验证性问题，回答excellent
   - → 没必要继续了，换话题

2. **候选人表现一般（3-5轮）**
   - 第1轮：基础问题，回答good/average
   - 第2轮：追问细节，回答good，有一定深度
   - 第3轮：再深入，回答average，开始有点吃力
   - 第4轮：换个角度问，回答good，还行
   - 第5轮（可选）：最后确认，回答average
   - → 已经了解水平，可以换话题了

3. **候选人不熟悉（2-3轮就该换）**
   - 第1轮：基础问题，回答average/poor，比较表面
   - 第2轮：追问细节，回答poor，答不上来
   - 第3轮（可选）：再试一次，回答poor，确实不懂
   - → 继续问只会尴尬，换话题吧

4. **关键话题需要深入（4-6轮）**
   - 核心技能或关键项目经验
   - 候选人表现good/excellent，值得深挖
   - 从多个角度验证能力

【回答质量要求】❗ 生成多样化、符合真实场景的回答：
- excellent（优秀20%）：深入、全面、有实战经验，细节丰富
- good（良好40%）：回答合理、有一定深度，但略显简略
- average（一般30%）：浅显、缺少细节，停留在概念层面
- poor（较差10%）：不准确、偏离主题，或明显不懂

【关键要求】❗❗❗

1. **每轮只问1个问题**（非常重要）
   ❌ 错误："首先，缓存策略是什么？其次，如何保证一致性？最后，有没有遇到问题？"
   ✅ 正确："能详细说说你们的Redis缓存策略是怎样设计的吗？"
   
   如果需要追问，应该等候选人回答后，下一轮再问：
   第1轮: "缓存策略是什么？"
   第2轮: "那如何保证一致性？"（基于第1轮回答）
   第3轮: "有没有遇到问题？"（基于第2轮回答）

2. **问题长度：80-150字**
   - 包含背景说明、具体要求
   - 但只聚焦一个点
   - 例如："我看到你在XX项目中用了Redis优化性能。能详细说说你们的缓存策略是怎样设计的吗？比如是cache-aside模式还是其他模式？为什么选择这种模式？"（120字，只问缓存策略）

3. **回答长度：200-400字**
   - 候选人展开回答，包含背景、实现、细节
   - 根据{resume['experience_level']}级别调整深度

4. **话题选择**
   - 从项目中选1-2个深入追问（每个3-5轮）
   - 从技能中选2-3个考察（每个2-4轮）

输出JSON（示例展示真实的多样化节奏）：
{{
  "candidate_id": {resume['id']},
  "candidate_name": "{resume['name']}",
  "candidate_level": "{resume['experience_level']}",
  "topics": [
    {{
      "topic_name": "Redis缓存优化（项目）",
      "topic_type": "项目",
      "rounds": [  // 候选人熟悉，3轮即可
        {{"round_number": 1, "question": "...", "answer": "...", "answer_quality": "good"}},
        {{"round_number": 2, "question": "...", "answer": "...", "answer_quality": "excellent"}},
        {{"round_number": 3, "question": "...", "answer": "...", "answer_quality": "excellent"}}
      ]
    }},
    {{
      "topic_name": "微服务架构",
      "topic_type": "技能",
      "rounds": [  // 候选人一般，4轮了解清楚
        {{"round_number": 1, "question": "...", "answer": "...", "answer_quality": "good"}},
        {{"round_number": 2, "question": "...", "answer": "...", "answer_quality": "average"}},
        {{"round_number": 3, "question": "...", "answer": "...", "answer_quality": "good"}},
        {{"round_number": 4, "question": "...", "answer": "...", "answer_quality": "average"}}
      ]
    }},
    {{
      "topic_name": "Kubernetes部署",
      "topic_type": "技能",
      "rounds": [  // 候选人不熟，2轮就换了
        {{"round_number": 1, "question": "...", "answer": "...", "answer_quality": "average"}},
        {{"round_number": 2, "question": "...", "answer": "...", "answer_quality": "poor"}}
      ]
    }},
    {{
      "topic_name": "Python异步编程（核心技能）",
      "topic_type": "技能",
      "rounds": [  // 关键技能，深入5轮
        {{"round_number": 1, "question": "...", "answer": "...", "answer_quality": "excellent"}},
        {{"round_number": 2, "question": "...", "answer": "...", "answer_quality": "good"}},
        {{"round_number": 3, "question": "...", "answer": "...", "answer_quality": "excellent"}},
        {{"round_number": 4, "question": "...", "answer": "...", "answer_quality": "good"}},
        {{"round_number": 5, "question": "...", "answer": "...", "answer_quality": "excellent"}}
      ]
    }}
  ]
}}

**关键：每个话题的轮数要根据answer_quality的分布合理设计！**
"""
    
    response_text = call_qwen(prompt, model="qwen-max", temperature=0.85, max_tokens=8000)
    if response_text:
        return extract_json(response_text)
    return None


# ========================================
# 第3步：生成标注
# ========================================

def generate_roberta_annotation(question: str, answer: str, quality: str) -> Dict:
    """生成RoBERTa标注"""
    
    quality_map = {
        "excellent": (90, 100),
        "good": (75, 89),
        "average": (60, 74),
        "poor": (40, 59)
    }
    score_range = quality_map.get(quality, (70, 79))
    
    # 鼓励生成多样化的分数
    score_hint = random.randint(score_range[0], score_range[1])
    
    prompt = f"""
评估回答质量（{quality}级别，建议{score_hint}分左右）：

问题：{question[:150]}...
回答：{answer[:250]}...

**评分标准：**
- 90-100：优秀，深入且全面
- 75-89：良好，合理但略显简略
- 60-74：一般，浅显或缺少细节
- 40-59：较差，不准确或偏离主题

输出JSON：
{{
  "score": <{score_range[0]}-{score_range[1]}的具体分数>,
  "label": "<优秀/良好/一般/较差>",
  "comment": "<80-120字：优点、不足、建议>"
}}
"""
    
    response_text = call_qwen(prompt, model="qwen-turbo", temperature=0.3, max_tokens=300)
    result = extract_json(response_text)
    
    # 强化容错：确保所有字段都存在
    if not result:
        result = {}
    
    return {
        "score": result.get("score", score_range[0]),
        "label": result.get("label", "一般"),
        "comment": result.get("comment", "回答基本合理，涵盖了主要要点。")
    }


def generate_bert_annotation(topic_rounds: List[Dict]) -> Dict:
    """生成BERT标注 - 模拟真实面试官的决策逻辑"""
    
    # 构建详细的对话历史
    rounds_detail = []
    for r in topic_rounds:
        rounds_detail.append(f"""
第{r['round_number']}轮:
  问题: {r['question'][:100]}...
  回答摘要: {r['answer'][:150]}...
  评分: {r['roberta_score']}分
""")
    
    rounds_text = "\n".join(rounds_detail)
    
    # 分析趋势
    scores = [r['roberta_score'] for r in topic_rounds]
    avg_score = sum(scores) / len(scores)
    
    # 趋势分析
    if len(scores) >= 2:
        recent_trend = "上升" if scores[-1] > scores[-2] else ("下降" if scores[-1] < scores[-2] else "持平")
    else:
        recent_trend = "首轮"
    
    prompt = f"""
你是资深技术面试官，正在决定是否继续当前话题还是切换。

【当前对话】
{rounds_text}

【数据分析】
- 已问轮数: {len(topic_rounds)}轮
- 平均分: {avg_score:.1f}分
- 分数趋势: {recent_trend}

【真实面试场景的决策逻辑】

**应该 SWITCH_TOPIC（切换话题）的情况：**

1. **候选人表现很好，没必要继续了**
   - 回答全面深入，已经展现了扎实的理解
   - 从基础原理到实战经验都聊清楚了
   - 继续问下去也很难挖出更多有价值信息
   - 示例：问了Redis，候选人把原理、实现、优化、踩坑全说清楚了
   
2. **候选人明显不熟悉这个话题**
   - 连续回答都比较表面、不深入
   - 追问时支支吾吾，无法给出细节
   - 继续追问只会暴露更多短板，不如换话题
   - 示例：问微服务，只能说概念，问具体实现就答不上来

3. **当前话题已经充分了解**
   - 已经从多个角度验证过候选人的能力
   - 继续问也是重复，没有新信息
   - 应该换个话题全面评估

**应该 FOLLOW_UP（继续追问）的情况：**

1. **回答有深度但还可以更深入**
   - 候选人展现了一定理解，但细节不够
   - 提到了有趣的点，值得展开
   - 需要验证是真懂还是只会背概念

2. **刚开始提问，需要深入了解**
   - 才问了1-2轮，还没充分了解
   - 候选人表现正常，值得继续挖掘

3. **表现起伏，需要再确认**
   - 有时答得好，有时答得不好
   - 需要多问几个问题确认真实水平

【你的任务】
像真实面试官一样思考：
1. 分析候选人的表现趋势（越来越好？越来越差？持平？）
2. 判断当前话题是否已经问够了
3. 决定：FOLLOW_UP 还是 SWITCH_TOPIC
4. 给Qwen面试官的指导（100-150字）

**注意：不要机械地用轮数或分数做判断，要综合考虑实际情况**

输出JSON：
{{
  "action": "<FOLLOW_UP或SWITCH_TOPIC>",
  "guidance": "<给Qwen的指导：基于最近的回答，应该追问什么或者为什么换话题>"
}}
"""
    
    response_text = call_qwen(prompt, model="qwen-plus", temperature=0.4, max_tokens=400)
    result = extract_json(response_text)
    
    # 强化容错：确保所有字段都存在
    if not result:
        result = {}
    
    return {
        "action": result.get("action", "FOLLOW_UP"),
        "guidance": result.get("guidance", "继续深入追问，了解更多细节。")
    }


def generate_qwen_annotation(history: List[Dict], bert: Dict) -> Dict:
    """生成Qwen标注"""
    
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

任务：
1. 生成问题（80-150字，只问1个具体问题）
2. 判断重要程度（1-5）

输出JSON：
{{
  "question": "<问题>",
  "importance": <1-5>
}}
"""
    
    response_text = call_qwen(prompt, model="qwen-plus", temperature=0.7, max_tokens=500)
    result = extract_json(response_text)
    
    # 强化容错：确保所有字段都存在
    if not result:
        result = {}
    
    return {
        "question": result.get("question", "能详细说说这个问题吗？"),
        "importance": result.get("importance", 3)
    }


# ========================================
# 主流程
# ========================================

def generate_training_data(
    position: str = "Python后端工程师",
    resume_count: int = 5,  # 先用5个测试
    output_dir: str = "./training_data"
):
    """
    生成训练数据（改进版）
    """
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print(f"生成训练数据：{position}，共{resume_count}份简历")
    print(f"API Key: {dashscope.api_key[:20]}...")
    print("=" * 60)
    
    # 第1步：生成简历（支持断点续传）
    print("\n【第1步】检查简历...")
    resume_file = f'{output_dir}/resumes.json'
    
    if os.path.exists(resume_file):
        # 加载已有简历
        with open(resume_file, 'r', encoding='utf-8') as f:
            resumes = json.load(f)
        print(f"[OK] 加载已有简历: {len(resumes)}份")
        
        # 如果数量不足，继续生成
        if len(resumes) < resume_count:
            print(f"[提示] 需要{resume_count}份，当前{len(resumes)}份，继续生成{resume_count - len(resumes)}份...")
            levels = ["junior"] * int((resume_count - len(resumes)) * 0.3) + \
                     ["mid"] * int((resume_count - len(resumes)) * 0.5) + \
                     ["senior"] * int((resume_count - len(resumes)) * 0.2)
            random.shuffle(levels)
            
            start_id = len(resumes) + 1
            for i, level in enumerate(tqdm(levels, desc="生成简历")):
                resume = generate_resume(position, level, start_id + i)
                if resume:
                    resumes.append(resume)
                    print(f"  [OK] {resume['name']} ({level})")
                time.sleep(0.5)
            
            # 保存更新后的简历
            with open(resume_file, 'w', encoding='utf-8') as f:
                json.dump(resumes, f, ensure_ascii=False, indent=2)
            print(f"[OK] 总计{len(resumes)}份简历")
    else:
        # 全新生成
        print(f"[提示] 未找到已有简历，开始生成{resume_count}份...")
        resumes = []
        
        levels = ["junior"] * int(resume_count * 0.3) + ["mid"] * int(resume_count * 0.5) + ["senior"] * int(resume_count * 0.2)
        random.shuffle(levels)
        
        for i, level in enumerate(tqdm(levels[:resume_count], desc="生成简历")):
            resume = generate_resume(position, level, i + 1)
            if resume:
                resumes.append(resume)
                print(f"  [OK] {resume['name']} ({level})")
            time.sleep(0.5)
        
        with open(resume_file, 'w', encoding='utf-8') as f:
            json.dump(resumes, f, ensure_ascii=False, indent=2)
        print(f"[OK] 生成{len(resumes)}份简历")
    
    # 第2步：模拟面试并标注（支持断点续传）
    print("\n【第2步】模拟面试...")
    
    # 加载已有数据（如果存在）
    roberta_file = f'{output_dir}/roberta_data.json'
    bert_file = f'{output_dir}/bert_data.json'
    qwen_file = f'{output_dir}/qwen_data.json'
    
    all_roberta = []
    all_bert = []
    all_qwen = []
    processed_ids = set()
    
    if os.path.exists(roberta_file):
        with open(roberta_file, 'r', encoding='utf-8') as f:
            all_roberta = json.load(f)
        print(f"[OK] 加载已有RoBERTa数据: {len(all_roberta)}条")
    
    if os.path.exists(bert_file):
        with open(bert_file, 'r', encoding='utf-8') as f:
            all_bert = json.load(f)
        print(f"[OK] 加载已有BERT数据: {len(all_bert)}条")
    
    if os.path.exists(qwen_file):
        with open(qwen_file, 'r', encoding='utf-8') as f:
            all_qwen = json.load(f)
        print(f"[OK] 加载已有Qwen数据: {len(all_qwen)}条")
    
    # 找出已处理的简历ID（从面试数据文件中）
    interview_progress_file = f'{output_dir}/.progress.json'
    if os.path.exists(interview_progress_file):
        with open(interview_progress_file, 'r', encoding='utf-8') as f:
            processed_ids = set(json.load(f))
        print(f"[OK] 已处理简历: {len(processed_ids)}/{len(resumes)}")
    
    for i, resume in enumerate(tqdm(resumes, desc="处理简历")):
        # 跳过已处理的简历
        if resume['id'] in processed_ids:
            print(f"\n{i+1}. {resume['name']} - [跳过，已处理]")
            continue
        
        print(f"\n{i+1}. {resume['name']} ({resume['experience_level']})")
        
        try:
            interview = generate_interview(resume)
            if not interview:
                print(f"  [警告] 面试生成失败，跳过")
                continue
        except Exception as e:
            print(f"  [错误] 面试生成异常: {str(e)}，跳过")
            continue
        
        full_history = []
        
        try:
            for topic in interview['topics']:
                print(f"  话题: {topic['topic_name']}")
                topic_rounds = []
                
                for round_data in topic['rounds']:
                    # 容错：确保所有必需字段存在
                    if 'question' not in round_data or 'answer' not in round_data:
                        print(f"    [警告] 问答数据不完整，跳过该轮")
                        continue
                    
                    # RoBERTa
                    roberta = generate_roberta_annotation(
                        round_data['question'],
                        round_data['answer'],
                        round_data.get('answer_quality', 'good')  # 默认为 good
                    )
                    
                    # 添加容错
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
                        print(f"    [警告] RoBERTa标注生成失败，使用默认值")
                        roberta_score = 75
                    
                    topic_rounds.append({
                        'round_number': round_data['round_number'],
                        'question': round_data['question'],
                        'answer': round_data['answer'],
                        'roberta_score': roberta_score
                    })
                    
                    # BERT
                    bert = generate_bert_annotation(topic_rounds)
                    
                    # 添加容错
                    if bert and 'action' in bert:
                        all_bert.append({
                            'topic_name': topic['topic_name'],
                            'rounds': topic_rounds.copy(),
                            'action': bert.get('action', 'FOLLOW_UP'),
                            'guidance': bert.get('guidance', '继续深入追问。')
                        })
                    else:
                        print(f"    [警告] BERT标注生成失败，使用默认值")
                        bert = {'action': 'FOLLOW_UP', 'guidance': '继续深入追问。'}
                    
                    # Qwen
                    if round_data != topic['rounds'][-1]:
                        qwen = generate_qwen_annotation(
                            full_history + [{'topic_name': topic['topic_name'], 'rounds': topic_rounds}],
                            bert
                        )
                        # 添加容错处理，并包含完整历史
                        if qwen and 'question' in qwen:
                            # 包含完整的对话历史（用于训练Qwen学习如何避免重复、自然过渡）
                            all_qwen.append({
                                'full_history': full_history + [{'topic_name': topic['topic_name'], 'rounds': topic_rounds}],
                                'bert_decision': bert['action'],
                                'bert_guidance': bert['guidance'],
                                'question': qwen.get('question', '继续追问'),
                                'importance': qwen.get('importance', 3)
                            })
                        else:
                            print(f"    [警告] Qwen标注生成失败，跳过")
                
                full_history.append({'topic_name': topic['topic_name'], 'rounds': topic_rounds})
            
        except Exception as e:
            print(f"  [错误] 处理面试数据时出现异常: {str(e)}")
            print(f"  [提示] 跳过该简历，继续处理下一份")
            continue
        
        # 标记该简历已处理
        processed_ids.add(resume['id'])
        
        # 增量保存（避免中断丢失数据）
        with open(roberta_file, 'w', encoding='utf-8') as f:
            json.dump(all_roberta, f, ensure_ascii=False, indent=2)
        with open(bert_file, 'w', encoding='utf-8') as f:
            json.dump(all_bert, f, ensure_ascii=False, indent=2)
        with open(qwen_file, 'w', encoding='utf-8') as f:
            json.dump(all_qwen, f, ensure_ascii=False, indent=2)
        with open(interview_progress_file, 'w', encoding='utf-8') as f:
            json.dump(list(processed_ids), f)
        
        print(f"  [OK] RoBERTa: {len(all_roberta)} | BERT: {len(all_bert)} | Qwen: {len(all_qwen)} [已保存]")
        time.sleep(1)
    
    # 第3步：保存
    print("\n【第3步】保存数据...")
    
    with open(f'{output_dir}/roberta_data.json', 'w', encoding='utf-8') as f:
        json.dump(all_roberta, f, ensure_ascii=False, indent=2)
    
    with open(f'{output_dir}/bert_data.json', 'w', encoding='utf-8') as f:
        json.dump(all_bert, f, ensure_ascii=False, indent=2)
    
    with open(f'{output_dir}/qwen_data.json', 'w', encoding='utf-8') as f:
        json.dump(all_qwen, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 60)
    print("[OK] 完成！")
    print(f"RoBERTa: {len(all_roberta)} 条")
    print(f"BERT: {len(all_bert)} 条")
    print(f"Qwen: {len(all_qwen)} 条")
    print(f"保存位置: {output_dir}/")
    print("=" * 60)


# ========================================
# 运行
# ========================================

if __name__ == "__main__":
    # 生成大规模训练数据
    generate_training_data(
        position="Python后端工程师",
        resume_count=100,  # 扩大到100份简历
        output_dir="./training_data"  # 保存到正式目录
    )
    
    print("\n[OK] 数据生成完成！")
    print("建议检查数据质量，确认无误后可用于训练。")

