#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第三批数据生成 - 改进版
新增特性：第一个问题必须是"自我介绍"，更符合真实面试场景
"""

import dashscope
import json
import random
import time
from typing import Dict, List
import os

# API配置
dashscope.api_key = "sk-abf39dd471804664b5dce35e722f0857"


def call_qwen(prompt: str, model: str = "qwen-plus", temperature: float = 0.7, max_tokens: int = 2000) -> str:
    """调用Qwen API"""
    try:
        response = dashscope.Generation.call(
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            result_format='message'
        )
        
        if response.status_code == 200:
            return response.output.text
        else:
            print(f"  [警告] API调用失败: {response.code} - {response.message}")
            return None
    except Exception as e:
        print(f"  [错误] API调用异常: {str(e)}")
        return None


def extract_json(text: str) -> Dict:
    """从文本中提取JSON"""
    try:
        start = text.find('{')
        end = text.rfind('}') + 1
        if start >= 0 and end > start:
            json_str = text[start:end]
            return json.loads(json_str)
    except Exception as e:
        print(f"  [警告] JSON解析失败: {str(e)}")
    return None


# ========================================
# 第1步：生成虚拟简历
# ========================================

def generate_resume() -> Dict:
    """生成虚拟简历 - 多样化岗位版本"""
    
    # 定义多种岗位模板
    job_profiles = [
        # 后端开发
        {
            "position": "Python后端开发",
            "tech_stack": ["Python", "Django", "Flask", "FastAPI", "Redis", "MySQL", "Docker", "Nginx"],
            "project_types": ["电商平台", "内容管理系统", "API网关", "微服务架构"],
            "major": ["计算机科学", "软件工程", "信息技术"]
        },
        {
            "position": "Java后端开发",
            "tech_stack": ["Java", "Spring Boot", "Spring Cloud", "MyBatis", "Redis", "MySQL", "RabbitMQ", "Kafka"],
            "project_types": ["金融系统", "企业ERP", "电商平台", "支付系统"],
            "major": ["计算机科学", "软件工程"]
        },
        {
            "position": "Go后端开发",
            "tech_stack": ["Go", "Gin", "gRPC", "Redis", "MongoDB", "Kubernetes", "Docker", "Consul"],
            "project_types": ["微服务系统", "云原生应用", "分布式存储", "API网关"],
            "major": ["计算机科学", "软件工程"]
        },
        {
            "position": "Node.js后端开发",
            "tech_stack": ["JavaScript", "Node.js", "Express", "Koa", "MongoDB", "Redis", "GraphQL", "WebSocket"],
            "project_types": ["实时聊天系统", "社交平台", "内容分发网络", "IoT平台"],
            "major": ["计算机科学", "软件工程"]
        },
        
        # 前端开发
        {
            "position": "React前端开发",
            "tech_stack": ["JavaScript", "TypeScript", "React", "Redux", "Webpack", "Ant Design", "Next.js"],
            "project_types": ["中台系统", "管理后台", "数据可视化平台", "电商前端"],
            "major": ["计算机科学", "软件工程", "数字媒体"]
        },
        {
            "position": "Vue前端开发",
            "tech_stack": ["JavaScript", "TypeScript", "Vue3", "Vuex", "Element Plus", "Vite", "Pinia"],
            "project_types": ["企业管理系统", "在线教育平台", "电商平台", "CRM系统"],
            "major": ["计算机科学", "软件工程", "数字媒体"]
        },
        {
            "position": "Angular前端开发",
            "tech_stack": ["TypeScript", "Angular", "RxJS", "NgRx", "Angular Material", "Jest"],
            "project_types": ["企业级应用", "金融数据平台", "项目管理系统"],
            "major": ["计算机科学", "软件工程"]
        },
        
        # 全栈开发
        {
            "position": "全栈开发工程师",
            "tech_stack": ["JavaScript", "TypeScript", "React", "Node.js", "Express", "PostgreSQL", "Docker", "AWS"],
            "project_types": ["SaaS平台", "创业项目", "Web应用", "云服务"],
            "major": ["计算机科学", "软件工程"]
        },
        
        # 移动端开发
        {
            "position": "Android开发",
            "tech_stack": ["Java", "Kotlin", "Android SDK", "Jetpack", "Room", "Retrofit", "RxJava"],
            "project_types": ["移动电商App", "社交App", "工具类App", "音视频App"],
            "major": ["计算机科学", "软件工程", "移动开发"]
        },
        {
            "position": "iOS开发",
            "tech_stack": ["Swift", "Objective-C", "UIKit", "SwiftUI", "Core Data", "Alamofire", "RxSwift"],
            "project_types": ["移动应用", "企业App", "工具类App", "金融App"],
            "major": ["计算机科学", "软件工程", "移动开发"]
        },
        {
            "position": "Flutter开发",
            "tech_stack": ["Dart", "Flutter", "Provider", "Bloc", "GetX", "Dio", "Firebase"],
            "project_types": ["跨平台App", "移动电商", "在线教育", "生活服务"],
            "major": ["计算机科学", "软件工程"]
        },
        
        # 数据相关
        {
            "position": "数据分析师",
            "tech_stack": ["Python", "Pandas", "NumPy", "SQL", "Tableau", "Power BI", "Excel", "Jupyter"],
            "project_types": ["用户行为分析", "商业智能报表", "数据仓库", "A/B测试"],
            "major": ["数据科学", "统计学", "计算机科学", "数学"]
        },
        {
            "position": "数据工程师",
            "tech_stack": ["Python", "Spark", "Hadoop", "Kafka", "Airflow", "Hive", "Flink", "Elasticsearch"],
            "project_types": ["数据平台", "实时数据流", "数据仓库", "ETL系统"],
            "major": ["数据科学", "计算机科学", "软件工程"]
        },
        {
            "position": "爬虫工程师",
            "tech_stack": ["Python", "Scrapy", "Selenium", "BeautifulSoup", "Redis", "MongoDB", "Kafka"],
            "project_types": ["数据采集系统", "舆情监控", "价格监控", "内容聚合"],
            "major": ["计算机科学", "软件工程", "数据科学"]
        },
        
        # AI/机器学习
        {
            "position": "机器学习工程师",
            "tech_stack": ["Python", "TensorFlow", "PyTorch", "Scikit-learn", "Pandas", "NumPy", "Jupyter", "MLflow"],
            "project_types": ["推荐系统", "图像识别", "NLP应用", "预测模型"],
            "major": ["人工智能", "机器学习", "计算机科学", "数学"]
        },
        {
            "position": "深度学习工程师",
            "tech_stack": ["Python", "PyTorch", "TensorFlow", "Keras", "CUDA", "Docker", "OpenCV"],
            "project_types": ["计算机视觉", "自然语言处理", "语音识别", "自动驾驶"],
            "major": ["人工智能", "计算机视觉", "模式识别"]
        },
        {
            "position": "算法工程师",
            "tech_stack": ["Python", "C++", "PyTorch", "TensorFlow", "Scikit-learn", "LightGBM", "XGBoost"],
            "project_types": ["推荐算法", "搜索算法", "风控模型", "智能决策"],
            "major": ["计算机科学", "数学", "人工智能", "统计学"]
        },
        
        # 运维/DevOps
        {
            "position": "运维工程师",
            "tech_stack": ["Linux", "Shell", "Python", "Ansible", "Nginx", "MySQL", "Zabbix", "Prometheus"],
            "project_types": ["服务器运维", "监控系统", "自动化部署", "日志分析"],
            "major": ["计算机科学", "网络工程", "信息安全"]
        },
        {
            "position": "DevOps工程师",
            "tech_stack": ["Docker", "Kubernetes", "Jenkins", "GitLab CI", "Terraform", "Prometheus", "Grafana"],
            "project_types": ["CI/CD流水线", "容器化平台", "云原生架构", "自动化运维"],
            "major": ["计算机科学", "软件工程"]
        },
        {
            "position": "云计算工程师",
            "tech_stack": ["AWS", "Azure", "Kubernetes", "Docker", "Terraform", "Python", "Go"],
            "project_types": ["云基础设施", "云迁移", "混合云", "云原生应用"],
            "major": ["计算机科学", "网络工程", "软件工程"]
        },
        
        # 测试
        {
            "position": "测试工程师",
            "tech_stack": ["Python", "Selenium", "Pytest", "JMeter", "Postman", "Appium", "Jenkins"],
            "project_types": ["自动化测试", "性能测试", "接口测试", "移动端测试"],
            "major": ["计算机科学", "软件工程", "软件测试"]
        },
        
        # 大数据
        {
            "position": "大数据开发",
            "tech_stack": ["Java", "Spark", "Hadoop", "Hive", "Kafka", "Flink", "HBase", "Scala"],
            "project_types": ["离线数据处理", "实时计算平台", "数据仓库", "日志分析"],
            "major": ["计算机科学", "数据科学", "软件工程"]
        },
        
        # 游戏开发
        {
            "position": "Unity游戏开发",
            "tech_stack": ["C#", "Unity", "Shader", "Lua", "Photon", "PlayFab"],
            "project_types": ["手机游戏", "VR游戏", "独立游戏", "小游戏"],
            "major": ["计算机科学", "软件工程", "数字媒体"]
        },
        {
            "position": "游戏后端开发",
            "tech_stack": ["Go", "C++", "Redis", "MongoDB", "Protobuf", "gRPC", "Kafka"],
            "project_types": ["游戏服务器", "匹配系统", "聊天系统", "排行榜系统"],
            "major": ["计算机科学", "软件工程"]
        },
        
        # 嵌入式/IoT
        {
            "position": "嵌入式开发",
            "tech_stack": ["C", "C++", "Linux", "ARM", "RTOS", "Python", "Qt"],
            "project_types": ["智能硬件", "IoT设备", "车载系统", "工控系统"],
            "major": ["计算机科学", "电子工程", "自动化"]
        },
        
        # 安全
        {
            "position": "安全工程师",
            "tech_stack": ["Python", "Kali Linux", "Burp Suite", "Metasploit", "Wireshark", "Nmap"],
            "project_types": ["安全审计", "渗透测试", "安全防护", "应急响应"],
            "major": ["信息安全", "网络安全", "计算机科学"]
        },
        
        # 区块链
        {
            "position": "区块链开发",
            "tech_stack": ["Solidity", "JavaScript", "Web3.js", "Truffle", "Go", "Hyperledger"],
            "project_types": ["智能合约", "DApp", "联盟链", "NFT平台"],
            "major": ["计算机科学", "软件工程", "密码学"]
        }
    ]
    
    # 随机选择一个岗位
    profile = random.choice(job_profiles)
    experience_level = random.choice(["初级", "中级", "高级"])
    
    # 从技术栈中随机选择一部分（增加多样性）
    selected_tech = random.sample(profile["tech_stack"], min(len(profile["tech_stack"]), random.randint(5, 8)))
    
    resume = {
        "id": f"R{random.randint(10000, 99999)}",
        "name": f"候选人{random.randint(1, 999):03d}",
        "position": profile["position"],
        "experience_level": experience_level,
        "education": {
            "degree": random.choice(["本科", "硕士", "博士"]),
            "major": random.choice(profile["major"])
        },
        "skills": [
            {
                "category": "技术栈",
                "items": selected_tech
            }
        ],
        "projects": [
            {
                "name": f"{random.choice(profile['project_types'])}项目",
                "background": f"负责{profile['position']}相关工作",
                "tech_stack": random.sample(selected_tech, min(len(selected_tech), random.randint(3, 5)))
            }
            for i in range(random.randint(2, 4))
        ]
    }
    
    return resume


# ========================================
# 第2步：生成面试对话 - V3改进版
# ========================================

def generate_interview_v3(resume: Dict) -> Dict:
    """
    生成面试对话 - V3版本
    关键改进：第一个话题必须是"自我介绍"
    """
    
    performance_type = random.choices(
        ["excellent", "good", "average", "poor", "nervous"],
        weights=[0.15, 0.35, 0.30, 0.10, 0.10],
        k=1
    )[0]
    
    performance_descriptions = {
        "excellent": "表现优秀，回答流畅准确，细节丰富，很少填充词",
        "good": "表现良好，基本流畅，有一定细节，偶有停顿和少量填充词",
        "average": "表现一般，偏概念化，缺少实战细节，较多填充词",
        "poor": "表现较差，支支吾吾，大量填充词，不直接说'不会'而是避重就轻",
        "nervous": "有能力但紧张，频繁停顿和填充词，内容正确但表达不流畅"
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
模拟20-30分钟真实语音面试，包含4-5个话题。

⭐【第一个话题要求 - 必须遵守】⭐：
话题1必须是"自我介绍/开场寒暄"，这是真实面试的标准流程！

示例问题（选一个或类似变体）：
- "先简单介绍一下你自己吧"
- "请介绍一下你的项目经验和技术背景"
- "说说你最近在做什么项目？"
- "简单聊聊你的工作经历吧"

话题1要求：
- topic_name: "自我介绍" 或 "开场寒暄"
- topic_type: "opening_intro"  ⭐重要⭐
- 重要度: 2-3分（中低重要度，主要是暖场和建立初步印象）
- 轮数: 1-2轮（不用深挖，快速了解即可）
- 回答长度: 200-400字（候选人会介绍背景、项目、技能等）

【后续话题类型】（话题2-5）：

1. 核心技术深挖 (40%)：从项目中选1-2个核心技术深入追问
   - 重要度：4-5分
   - 轮数：3-5轮

2. 常规技能了解 (30%)：了解技术栈使用情况
   - 重要度：3分
   - 轮数：2-3轮

3. 项目经验分享 (20%)：了解实际项目经验
   - 重要度：3-4分
   - 轮数：2-4轮

4. 拉家常/背景了解 (10%)：缓解紧张、了解背景
   - 重要度：1-2分
   - 轮数：1-2轮
   - 示例问题：
     * "你家是哪儿的？平时怎么通勤？"
     * "本科/研究生阶段过得怎么样？"
     * "业余时间有什么爱好？"

【回答质量模式 - {performance_type}】：

excellent类型的回答：
  - 流畅、准确、有丰富细节
  - 很少填充词
  - 自我介绍示例："我是{resume['name']}，有3年后端开发经验。最近在做一个电商平台的优化项目，主要负责缓存架构设计和性能调优。技术栈是Python+Django+Redis+MySQL，日均处理100万+订单。之前还做过..."

good类型的回答：
  - 基本流畅，有一定细节，偶有停顿
  - 少量填充词："嗯"、"就是"、"然后"
  - 自我介绍示例："嗯，我叫{resume['name']}，做后端开发的，有差不多3年吧。最近就是在做一个电商项目，然后主要负责后端这块，用的是Python那一套技术栈..."

average类型的回答：
  - 偏概念化，缺少实战细节
  - 较多填充词："额"、"这个"、"那个"
  - 自我介绍示例："额，我是{resume['name']}，这个...做开发的，后端方向。项目的话...嗯...就是一些Web项目吧，用Python比较多，那个...具体的话..."

poor类型的回答：
  - 支支吾吾，大量填充词表示不确定
  - 不直接说"不会"，而是避重就轻
  - 自我介绍示例："额...我是{resume['name']}...嗯...做开发的...这个...项目的话...我想想...就是一些...那个...Web相关的吧..."

nervous类型的回答：
  - 内容可以但停顿多，紧张
  - 频繁使用填充词表示思考
  - 自我介绍示例："额...我是{resume['name']}，嗯...做后端开发的。项目的话...让我想想...最近在做电商平台，就是...那个...负责后端部分，用Python这些..."

【非语言填充词】（根据表现类型适当使用）：
- 思考/停顿："额"、"嗯"、"呃"、"啊"
- 不确定："这个"、"那个"、"好像"、"我记得"
- 组织语言："就是"、"然后"、"怎么说呢"、"让我想想"

【关键要求】：
1. 话题1必须是自我介绍/开场，topic_type必须是"opening_intro"
2. 每轮只问1个问题
3. 问题长度：80-150字
4. 自我介绍的回答长度：200-400字
5. 其他回答长度：根据表现类型调整

输出JSON格式：
{{
  "candidate_id": "{resume['id']}",
  "candidate_name": "{resume['name']}",
  "candidate_level": "{resume['experience_level']}",
  "performance_type": "{performance_type}",
  "topics": [
    {{
      "topic_name": "自我介绍",
      "topic_type": "opening_intro",
      "rounds": [
        {{
          "round_number": 1,
          "question": "先简单介绍一下你自己吧，说说你的项目经验和技术背景",
          "answer": "...",  # 根据{performance_type}生成200-400字
          "answer_quality": "good"
        }}
      ]
    }},
    {{
      "topic_name": "Redis缓存优化（核心技术）",
      "topic_type": "technical_core",
      "rounds": [
        {{
          "round_number": 1,
          "question": "...",
          "answer": "...",
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
          "answer_quality": "good"
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
# 第3步：生成标注（使用V2版本的逻辑）
# ========================================

def generate_roberta_annotation_v3(question: str, answer: str, quality: str) -> Dict:
    """生成RoBERTa标注"""
    
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
  - 即使有少量填充词（"嗯"、"就是"），但不影响整体质量

75-89分（良好）：
  - 内容基本准确，有一定细节
  - 适量填充词是正常的口语表达，不扣分

60-74分（一般）：
  - 概念化，缺少实战细节
  - 频繁使用"额"、"这个"、"那个"表示不确定

40-59分（较差）：
  - 支支吾吾，大量填充词表示避重就轻
  - 没有直接说"不会"，但通过填充词暴露了不熟悉

⚠️ 注意：不要因为填充词就大幅扣分，要看整体内容质量！

请给出JSON：
{{
  "score": {score_hint},  # 在{score_range[0]}-{score_range[1]}范围内
  "label": "{quality}",
  "comment": "简短评语（40-80字）"
}}
"""
    
    response_text = call_qwen(prompt, model="qwen-plus", temperature=0.6, max_tokens=300)
    if response_text:
        result = extract_json(response_text)
        if result:
            # 确保字段存在
            return {
                'score': result.get('score', score_hint),
                'label': result.get('label', quality),
                'comment': result.get('comment', '回答基本符合要求。')
            }
    
    # 默认返回
    return {
        'score': score_hint,
        'label': quality,
        'comment': '回答基本符合要求。'
    }


def generate_bert_annotation_v3(topic_rounds: List[Dict]) -> Dict:
    """生成BERT标注"""
    
    if not topic_rounds:
        return {'action': 'SWITCH_TOPIC', 'guidance': '开始新话题。'}
    
    # 计算平均分和趋势
    scores = [r.get('roberta_score', 70) for r in topic_rounds if 'roberta_score' in r]
    avg_score = sum(scores) / len(scores) if scores else 70
    
    recent_trend = "稳定"
    if len(scores) >= 2:
        if scores[-1] > scores[-2] + 5:
            recent_trend = "上升"
        elif scores[-1] < scores[-2] - 5:
            recent_trend = "下降"
    
    rounds_text = ""
    for r in topic_rounds[-3:]:
        rounds_text += f"\nQ: {r['question'][:100]}\nA: {r['answer'][:150]}\n评分: {r.get('roberta_score', 'N/A')}分"
    
    prompt = f"""
作为面试决策者，判断是否继续当前话题。

【当前话题情况】
轮数: {len(topic_rounds)}轮
平均分: {avg_score:.1f}
趋势: {recent_trend}

最近几轮:
{rounds_text}

【决策指南】：

FOLLOW_UP情况（继续深入）：
- 回答质量稳定或上升，值得深挖
- 轮数<3轮，还未充分考察
- 候选人展现出某个亮点，需要验证

SWITCH_TOPIC情况（切换话题）：
- 已经充分考察（3轮+）
- 回答质量持续下降，继续追问无益
- 候选人明确表示不熟悉该领域
- 已经达到考察深度

输出JSON：
{{
  "action": "FOLLOW_UP",  # 或 SWITCH_TOPIC
  "guidance": "对最近一轮回答的分析和对Qwen的提示（不包括重要程度）"
}}
"""
    
    response_text = call_qwen(prompt, model="qwen-plus", temperature=0.5, max_tokens=400)
    if response_text:
        result = extract_json(response_text)
        if result:
            return {
                'action': result.get('action', 'FOLLOW_UP'),
                'guidance': result.get('guidance', '继续深入追问。')
            }
    
    return {'action': 'FOLLOW_UP', 'guidance': '继续深入追问。'}


def generate_qwen_annotation_v3(full_history: List[Dict], bert: Dict, topic_type: str, rounds_count: int) -> Dict:
    """
    生成Qwen标注 - V3版本
    新增：识别opening_intro类型，给予适当的重要度
    """
    
    # 格式化历史对话
    history_text = ""
    for topic in full_history[-3:]:
        history_text += f"\n话题: {topic['topic_name']}\n"
        for r in topic['rounds'][-2:]:
            history_text += f"  Q: {r['question'][:80]}\n  A: {r['answer'][:120]}\n"
    
    prompt = f"""
作为AI面试官（Qwen），根据对话历史和BERT的guidance生成下一个问题。

【对话历史】
{history_text if history_text else "（面试刚开始）"}

【BERT的指导】
决策: {bert.get('action', 'FOLLOW_UP')}
建议: {bert.get('guidance', '继续追问')}

【当前话题信息】
话题类型: {topic_type}
已进行轮数: {rounds_count}

【重要程度判定标准】：

⭐ opening_intro (自我介绍/开场) → 2-3分
  - 暖场性质，建立初步印象
  - 不是技术考察重点
  - 示例："先简单介绍一下你自己"

5分（核心重点）：
  - technical_core话题的第1-2轮深入追问
  - 考察候选人核心竞争力
  - 示例："Redis的持久化机制有哪几种？区别是什么？"

4分（重要）：
  - technical_core话题的第3-5轮
  - project_experience的关键问题
  - 示例："你们项目中缓存穿透是怎么解决的？"

3分（常规）：
  - technical_basic话题
  - project_experience的一般问题
  - 示例："平时用Python哪些库比较多？"

2分（次要）：
  - 背景了解、项目背景
  - 非技术性问题
  - 示例："这个项目团队规模多大？"

1分（闲聊）：
  - casual_chat类型
  - 纯粹拉家常
  - 示例："你家是哪儿的？"

【要求】：
1. 避免重复历史问题
2. 问题要自然、口语化
3. 长度80-150字
4. 根据topic_type和rounds_count判断重要程度

输出JSON：
{{
  "question": "具体问题内容",
  "importance": 3  # 1-5分
}}
"""
    
    response_text = call_qwen(prompt, model="qwen-plus", temperature=0.7, max_tokens=300)
    if response_text:
        result = extract_json(response_text)
        if result and 'question' in result:
            # 对opening_intro类型，强制重要度为2-3
            if topic_type == "opening_intro":
                result['importance'] = random.randint(2, 3)
            return {
                'question': result.get('question', '请继续介绍。'),
                'importance': result.get('importance', 3)
            }
    
    return {'question': '请继续介绍。', 'importance': 3}


# ========================================
# 主流程：生成训练数据
# ========================================

def generate_training_data_v3(resume_count: int = 50, output_dir: str = "./training_data"):
    """
    生成第三批训练数据
    特点：第一个问题必须是自我介绍
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 文件路径
    resume_file = os.path.join(output_dir, "resumes.json")
    roberta_file = os.path.join(output_dir, "roberta_data.json")
    bert_file = os.path.join(output_dir, "bert_data.json")
    qwen_file = os.path.join(output_dir, "qwen_data.json")
    processed_file = os.path.join(output_dir, "processed_v3.json")
    
    # 加载现有数据
    if os.path.exists(resume_file):
        with open(resume_file, 'r', encoding='utf-8') as f:
            all_resumes = json.load(f)
    else:
        all_resumes = []
    
    if os.path.exists(roberta_file):
        with open(roberta_file, 'r', encoding='utf-8') as f:
            all_roberta = json.load(f)
    else:
        all_roberta = []
    
    if os.path.exists(bert_file):
        with open(bert_file, 'r', encoding='utf-8') as f:
            all_bert = json.load(f)
    else:
        all_bert = []
    
    if os.path.exists(qwen_file):
        with open(qwen_file, 'r', encoding='utf-8') as f:
            all_qwen = json.load(f)
    else:
        all_qwen = []
    
    if os.path.exists(processed_file):
        with open(processed_file, 'r', encoding='utf-8') as f:
            processed_v3_ids = set(json.load(f))
    else:
        processed_v3_ids = set()
    
    print("=" * 60)
    print("第三批数据生成 - 真实面试场景（从自我介绍开始）")
    print("=" * 60)
    print(f"目标: 新增{resume_count}份简历")
    print(f"当前: RoBERTa={len(all_roberta)} | BERT={len(all_bert)} | Qwen={len(all_qwen)}")
    print(f"已处理(V3): {len(processed_v3_ids)}份")
    print("=" * 60)
    
    # 第1步：确保有足够简历
    total_needed = len(all_resumes) + resume_count
    while len(all_resumes) < total_needed:
        resume = generate_resume()
        all_resumes.append(resume)
        print(f"[生成简历] {resume['name']} ({resume['experience_level']}) - 总数: {len(all_resumes)}")
        
        with open(resume_file, 'w', encoding='utf-8') as f:
            json.dump(all_resumes, f, ensure_ascii=False, indent=2)
        
        time.sleep(0.5)
    
    print(f"\n简历总数: {len(all_resumes)}")
    
    # 第2步：选择还未用V3处理的简历
    unprocessed_resumes = [r for r in all_resumes if r['id'] not in processed_v3_ids]
    
    if len(unprocessed_resumes) < resume_count:
        print(f"[提示] 未处理简历只有{len(unprocessed_resumes)}份，将处理全部")
        target_resumes = unprocessed_resumes
    else:
        target_resumes = random.sample(unprocessed_resumes, resume_count)
    
    print(f"本批次处理: {len(target_resumes)}份简历\n")
    
    # 第3步：生成面试数据和标注
    for idx, resume in enumerate(target_resumes, 1):
        print(f"\n[{idx}/{len(target_resumes)}] 处理简历: {resume['name']} ({resume['id']})")
        
        try:
            # 生成面试对话
            interview = generate_interview_v3(resume)
            
            if not interview or 'topics' not in interview:
                print(f"  [跳过] 面试生成失败")
                continue
            
            # 验证第一个话题是否是自我介绍
            first_topic = interview['topics'][0]
            if first_topic.get('topic_type') != 'opening_intro':
                print(f"  [警告] 第一个话题不是自我介绍: {first_topic.get('topic_name')}")
                print(f"  [提示] 已生成，但不符合预期格式")
            
            print(f"  [面试] {len(interview['topics'])}个话题 | 第一题: {first_topic.get('topic_name')}")
            
            # 处理每个话题
            full_history = []
            
            for topic in interview['topics']:
                topic_rounds = []
                
                for round_data in topic['rounds']:
                    question = round_data['question']
                    answer = round_data['answer']
                    quality = round_data.get('answer_quality', 'good')
                    
                    # RoBERTa
                    roberta = generate_roberta_annotation_v3(question, answer, quality)
                    
                    if roberta and 'score' in roberta:
                        round_data['roberta_score'] = roberta['score']
                        topic_rounds.append(round_data)
                        
                        all_roberta.append({
                            'question': question,
                            'answer': answer,
                            'score': roberta.get('score', 70),
                            'label': roberta.get('label', 'good'),
                            'comment': roberta.get('comment', '')
                        })
                    
                    # BERT
                    bert = generate_bert_annotation_v3(topic_rounds)
                    
                    if bert and 'action' in bert:
                        all_bert.append({
                            'topic_rounds': topic_rounds.copy(),
                            'action': bert.get('action', 'FOLLOW_UP'),
                            'guidance': bert.get('guidance', '')
                        })
                    else:
                        bert = {'action': 'FOLLOW_UP', 'guidance': '继续深入追问。'}
                    
                    # Qwen
                    if round_data != topic['rounds'][-1]:
                        qwen = generate_qwen_annotation_v3(
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
        
        # 标记为已处理
        processed_v3_ids.add(resume['id'])
        
        # 增量保存
        with open(roberta_file, 'w', encoding='utf-8') as f:
            json.dump(all_roberta, f, ensure_ascii=False, indent=2)
        with open(bert_file, 'w', encoding='utf-8') as f:
            json.dump(all_bert, f, ensure_ascii=False, indent=2)
        with open(qwen_file, 'w', encoding='utf-8') as f:
            json.dump(all_qwen, f, ensure_ascii=False, indent=2)
        with open(processed_file, 'w', encoding='utf-8') as f:
            json.dump(list(processed_v3_ids), f, ensure_ascii=False, indent=2)
        
        print(f"  [OK] RoBERTa: {len(all_roberta)} | BERT: {len(all_bert)} | Qwen: {len(all_qwen)} [已保存]")
        time.sleep(1)
    
    print("\n" + "=" * 60)
    print("[OK] 第三批数据生成完成！")
    print(f"RoBERTa: {len(all_roberta)} 条（累计）")
    print(f"BERT: {len(all_bert)} 条（累计）")
    print(f"Qwen: {len(all_qwen)} 条（累计）")
    print(f"V3处理简历: {len(processed_v3_ids)} 份")
    print(f"保存位置: {output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    # 生成第三批数据（新增30份简历）
    generate_training_data_v3(
        resume_count=30,  # 新增30份
        output_dir="./training_data"  # 追加到现有数据
    )
    
    print("\n[OK] 数据生成完成！")
    print("特色：所有面试都从'自我介绍'开始，更符合真实场景！")

