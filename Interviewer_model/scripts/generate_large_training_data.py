"""
生成大规模高质量BERT训练数据（500条）
追问深度动态，不固定3层
"""
import json
import random
from typing import List, Dict, Any

# ========== 技术栈和问题库 ==========
TECH_STACKS = {
    "Python": {
        "基础": ["语法", "数据结构", "函数", "类与对象", "装饰器", "生成器", "异常处理"],
        "进阶": ["多线程", "异步编程", "元类", "GIL", "内存管理", "性能优化"],
        "框架": ["Django", "Flask", "FastAPI", "Celery", "SQLAlchemy"]
    },
    "Java": {
        "基础": ["集合框架", "异常处理", "多线程", "IO流", "反射", "注解"],
        "进阶": ["JVM", "GC", "并发包", "设计模式", "性能调优"],
        "框架": ["Spring", "Spring Boot", "MyBatis", "Hibernate"]
    },
    "JavaScript": {
        "基础": ["闭包", "原型链", "异步", "Promise", "ES6", "事件循环"],
        "进阶": ["模块化", "打包工具", "性能优化", "内存泄漏"],
        "框架": ["React", "Vue", "Angular", "Node.js", "Express"]
    },
    "数据库": {
        "MySQL": ["索引", "事务", "锁", "主从复制", "分库分表", "慢查询优化"],
        "Redis": ["数据结构", "持久化", "过期策略", "缓存策略", "集群", "哨兵"],
        "MongoDB": ["文档模型", "索引", "聚合", "副本集", "分片"]
    },
    "分布式": {
        "基础": ["CAP理论", "一致性", "分布式事务", "服务治理"],
        "消息队列": ["RabbitMQ", "Kafka", "RocketMQ", "消息可靠性", "消息积压"],
        "微服务": ["服务拆分", "服务发现", "熔断降级", "限流", "链路追踪"]
    },
    "前端": {
        "基础": ["HTML", "CSS", "DOM", "BOM", "HTTP", "浏览器渲染"],
        "React": ["Hooks", "虚拟DOM", "状态管理", "组件通信", "性能优化"],
        "Vue": ["响应式原理", "组件化", "Vuex", "Router", "生命周期"]
    },
    "工具链": {
        "版本控制": ["Git", "分支管理", "冲突解决", "Git Flow", "rebase vs merge"],
        "容器化": ["Docker", "镜像", "容器", "网络", "Compose", "Kubernetes"],
        "CI/CD": ["Jenkins", "GitLab CI", "自动化测试", "部署策略"]
    },
    "算法": {
        "基础": ["数组", "链表", "栈", "队列", "哈希表", "树", "图"],
        "进阶": ["动态规划", "贪心", "回溯", "分治", "排序", "查找"],
        "机器学习": ["监督学习", "无监督学习", "神经网络", "优化算法", "过拟合"]
    }
}

# ========== 问题模板 ==========
QUESTION_TEMPLATES = {
    1: [  # 第1层：基础问题
        "你用过{tech}吗？",
        "你对{tech}了解吗？",
        "在项目中用过{tech}吗？",
        "能介绍一下你的{tech}经验吗？"
    ],
    2: [  # 第2层：具体细节
        "能具体说说你用{tech}做了什么吗？",
        "你是怎么使用{tech}的？",
        "在{tech}方面遇到过什么问题吗？",
        "能举个{tech}的使用案例吗？"
    ],
    3: [  # 第3层：深入原理
        "你知道{tech}的底层原理吗？",
        "{tech}是怎么实现的？",
        "为什么选择{tech}而不是其他方案？",
        "能说说{tech}的优缺点吗？"
    ],
    4: [  # 第4层：高级应用
        "在{tech}的性能优化方面有什么经验？",
        "遇到过{tech}的线上问题吗？怎么排查的？",
        "如果让你设计{tech}，你会怎么做？",
        "{tech}在高并发场景下怎么处理？"
    ],
    5: [  # 第5层：架构设计
        "在大规模系统中{tech}有什么挑战？",
        "如何评估{tech}在生产环境的表现？",
        "{tech}和其他技术怎么配合使用？",
        "能分享一个{tech}的复杂案例吗？"
    ]
}

# ========== 回答模板 ==========
ANSWER_TEMPLATES = {
    "excellent": {  # 优秀回答
        "templates": [
            "用过，我在{project}项目中用{tech}{action}。{detail}还做了{optimization}。",
            "{tech}我很熟悉，主要用来{purpose}。{implementation}效果{result}。",
            "有经验，我们项目用{tech}实现了{feature}。{process}最后{outcome}。"
        ],
        "hesitation_score": (0.08, 0.25),
        "filler_count": (0, 2),
        "speech_rate": (3.8, 5.0),
        "pause_count": (0, 0)
    },
    "good": {  # 良好回答
        "templates": [
            "嗯，用过，主要是{purpose}。{detail}",
            "了解一些，我在项目中{action}。{brief_detail}",
            "有一些经验，就是用{tech}{purpose}，然后{result}。"
        ],
        "hesitation_score": (0.25, 0.45),
        "filler_count": (2, 5),
        "speech_rate": (3.0, 3.8),
        "pause_count": (0, 1)
    },
    "mediocre": {  # 一般回答
        "templates": [
            "嗯...{tech}我用过一些，就是...{vague_description}",
            "这个...我好像...做过，嗯...具体的有点忘了",
            "嗯...知道一点，但是...不太熟悉具体的"
        ],
        "hesitation_score": (0.50, 0.70),
        "filler_count": (5, 8),
        "speech_rate": (2.3, 3.0),
        "pause_count": (1, 2)
    },
    "poor": {  # 较差回答
        "templates": [
            "嗯...这个...我好像...没怎么用过",
            "不太了解，嗯...我们项目没用到",
            "嗯...不太清楚，额...这块我不熟悉"
        ],
        "hesitation_score": (0.75, 0.95),
        "filler_count": (8, 15),
        "speech_rate": (1.5, 2.3),
        "pause_count": (2, 5)
    }
}

# ========== 数据生成器 ==========
class TrainingDataGenerator:
    def __init__(self):
        self.data = []
        self.id_counter = 1
    
    def generate_answer_quality_progression(self):
        """
        生成一个对话链的回答质量序列
        模拟真实面试：开始答得好，越问越深可能答不上来
        """
        patterns = [
            # 模式1：一直很好（20%）- 追问5-6层
            ["excellent", "excellent", "excellent", "good", "good", "excellent"],
            ["excellent", "excellent", "good", "excellent", "good"],
            
            # 模式2：逐渐下降（30%）- 追问3-4层
            ["excellent", "good", "mediocre", "poor"],
            ["good", "good", "mediocre", "poor"],
            ["excellent", "good", "good", "mediocre"],
            
            # 模式3：第一次就不会（25%）- 1层
            ["poor"],
            ["mediocre"],
            
            # 模式4：开始一般后来好起来（15%）- 2-3层
            ["mediocre", "good", "excellent"],
            ["good", "excellent", "excellent"],
            
            # 模式5：波动（10%）- 3-4层
            ["good", "mediocre", "good", "excellent"],
            ["excellent", "mediocre", "good", "poor"]
        ]
        
        return random.choice(patterns)
    
    def generate_speech_features(self, quality: str):
        """根据回答质量生成语音特征"""
        template = ANSWER_TEMPLATES[quality]
        
        hesitation_score = random.uniform(*template["hesitation_score"])
        filler_count = random.randint(*template["filler_count"])
        speech_rate = random.uniform(*template["speech_rate"])
        pause_count = random.randint(*template["pause_count"])
        
        # 生成具体的填充词
        filler_words = []
        if filler_count > 0:
            all_fillers = ["嗯", "这个", "那个", "就是", "然后", "额", "呃"]
            filler_words = random.sample(all_fillers, min(filler_count // 2 + 1, len(all_fillers)))
        
        # 生成停顿
        long_pauses = []
        if pause_count > 0:
            long_pauses = [round(random.uniform(1.5, 2.8), 1) for _ in range(pause_count)]
        
        return {
            "hesitation_score": round(hesitation_score, 2),
            "filler_count": filler_count,
            "filler_words": filler_words,
            "speech_rate": round(speech_rate, 1),
            "pause_count": pause_count,
            "long_pauses": long_pauses
        }
    
    def generate_answer_text(self, quality: str, tech: str, depth: int):
        """生成回答文本"""
        template = random.choice(ANSWER_TEMPLATES[quality]["templates"])
        
        # 根据质量和深度生成不同的回答内容
        if quality == "excellent":
            if depth == 1:
                answer = f"用过，我在电商项目中用{tech}做了核心功能。实现了高并发处理，还做了性能优化，响应时间控制在100ms以内。"
            elif depth == 2:
                answer = f"具体来说，我用{tech}实现了用户认证、数据缓存和异步任务。通过连接池优化性能，并发量提升了3倍。还加了监控告警，及时发现问题。"
            elif depth == 3:
                answer = f"{tech}的底层原理我了解，它基于事件驱动架构，通过非阻塞IO实现高并发。我们项目中针对热点数据做了预加载，配合LRU缓存淘汰策略，命中率达到95%。"
            elif depth == 4:
                answer = f"性能优化方面，我做过线程池调优、内存配置优化、慢查询分析。有一次发现内存泄漏，用profiler定位到是连接未释放，加了try-finally解决。还优化了批处理逻辑，吞吐量提升了50%。"
            else:
                answer = f"在大规模系统中，我们用{tech}搭建了主从架构+哨兵模式，保证高可用。数据量达到TB级别时，做了分片和冷热数据分离。还实现了数据一致性保证和故障自动切换，系统可用性达到99.9%。"
        
        elif quality == "good":
            if depth == 1:
                answer = f"嗯，用过，主要是做数据处理。我在项目中用{tech}实现了基本功能，效果还不错。"
            elif depth == 2:
                answer = f"具体就是用{tech}处理业务逻辑，然后做了一些配置优化。遇到过一些小问题，查资料解决了。"
            elif depth == 3:
                answer = f"嗯，原理的话，我知道大概的工作机制，就是通过某种方式实现功能。我们选择{tech}是因为比较成熟，社区活跃。"
            else:
                answer = f"嗯，性能优化方面，我加了缓存，然后做了一些参数调整。具体数据提升多少没仔细测过，但感觉快了。"
        
        elif quality == "mediocre":
            if depth <= 2:
                answer = f"嗯...{tech}我用过一些，就是...做了基本的功能，嗯...具体的细节有点记不清了。"
            else:
                answer = f"这个...原理的话...我好像...知道一点，但是...不太确定，嗯...可能需要查查资料。"
        
        else:  # poor
            answer = f"嗯...这个...我好像...没怎么用过，额...不太清楚，嗯...我们项目没用到{tech}。"
        
        # 根据填充词数量添加填充词
        speech_features = self.generate_speech_features(quality)
        if speech_features["filler_count"] > 3:
            # 添加更多填充词让answer和filler_count匹配
            fillers = speech_features["filler_words"]
            for _ in range(min(3, speech_features["filler_count"] - answer.count("嗯") - answer.count("这个"))):
                insert_pos = random.randint(0, len(answer) // 2)
                filler = random.choice(fillers)
                answer = answer[:insert_pos] + filler + "..." + answer[insert_pos:]
        
        return answer, round(len(answer), 0)
    
    def generate_dialogue_chain(self, tech_category: str, tech: str):
        """生成一个完整的对话链"""
        quality_sequence = self.generate_answer_quality_progression()
        samples = []
        
        for depth, quality in enumerate(quality_sequence, start=1):
            # 生成问题
            if depth <= 5:
                question_template = random.choice(QUESTION_TEMPLATES.get(depth, QUESTION_TEMPLATES[3]))
                question = question_template.format(tech=tech)
            else:
                question = f"关于{tech}还有什么想补充的吗？"
            
            # 生成回答
            answer, answer_length = self.generate_answer_text(quality, tech, depth)
            
            # 生成语音特征
            speech_features = self.generate_speech_features(quality)
            
            # 决定标签
            if depth == len(quality_sequence):  # 最后一轮
                if quality in ["excellent", "good"]:
                    label = "NEXT_TOPIC"
                    reason = f"已经追问{depth}层，候选人持续表现良好，充分展示了{tech}能力，应该换话题"
                    reason_type = "positive"
                else:
                    label = "NEXT_TOPIC"
                    reason = f"候选人答不上来了，应该换话题"
                    reason_type = "negative"
            else:  # 非最后一轮
                if quality == "poor":
                    label = "NEXT_TOPIC"
                    reason = f"候选人明显不了解{tech}（hesitation_score={speech_features['hesitation_score']}），应该换话题"
                    reason_type = "negative"
                elif quality == "mediocre" and depth >= 2:
                    label = "NEXT_TOPIC"
                    reason = f"候选人回答模糊，已追问{depth}层，继续问也问不出什么，换话题"
                    reason_type = "negative"
                else:
                    label = "FOLLOW_UP"
                    reason = f"候选人回答有内容（hesitation_score={speech_features['hesitation_score']}），可以继续追问"
            
            # 构建样本
            sample = {
                "id": self.id_counter,
                "question": question,
                "answer": answer,
                "context": {
                    "follow_up_depth": depth,
                    "hesitation_score": speech_features["hesitation_score"],
                    "filler_count": speech_features["filler_count"],
                    "filler_words": speech_features["filler_words"],
                    "speech_rate": speech_features["speech_rate"],
                    "pause_count": speech_features["pause_count"],
                    "long_pauses": speech_features["long_pauses"],
                    "answer_length": int(answer_length)
                },
                "label": label,
                "reason": reason,
                "tech_stack": tech,
                "tech_category": tech_category
            }
            
            if label == "NEXT_TOPIC":
                sample["reason_type"] = reason_type
            
            samples.append(sample)
            self.id_counter += 1
        
        return samples
    
    def generate_all_data(self, target_count: int = 500):
        """生成所有训练数据"""
        print(f"开始生成{target_count}条训练数据...")
        
        # 计算需要生成多少个对话链
        # 平均每个对话链3-4个样本
        avg_samples_per_chain = 3.5
        target_chains = int(target_count / avg_samples_per_chain) + 10
        
        chains_generated = 0
        
        while len(self.data) < target_count:
            # 随机选择技术栈
            category = random.choice(list(TECH_STACKS.keys()))
            sub_category = random.choice(list(TECH_STACKS[category].keys()))
            tech = random.choice(TECH_STACKS[category][sub_category])
            
            # 生成对话链
            chain_samples = self.generate_dialogue_chain(category, tech)
            self.data.extend(chain_samples)
            
            chains_generated += 1
            if chains_generated % 20 == 0:
                print(f"已生成{chains_generated}个对话链，共{len(self.data)}条样本...")
        
        # 截取到目标数量
        self.data = self.data[:target_count]
        
        # 统计
        follow_up_count = sum(1 for d in self.data if d["label"] == "FOLLOW_UP")
        next_topic_count = sum(1 for d in self.data if d["label"] == "NEXT_TOPIC")
        next_topic_positive = sum(1 for d in self.data if d.get("reason_type") == "positive")
        next_topic_negative = sum(1 for d in self.data if d.get("reason_type") == "negative")
        
        print(f"\n[OK] 数据生成完成！")
        print(f"总计: {len(self.data)}条")
        print(f"FOLLOW_UP: {follow_up_count}条 ({follow_up_count/len(self.data)*100:.1f}%)")
        print(f"NEXT_TOPIC: {next_topic_count}条 ({next_topic_count/len(self.data)*100:.1f}%)")
        print(f"  - 消极换话题: {next_topic_negative}条")
        print(f"  - 积极换话题: {next_topic_positive}条")
        
        # 深度统计
        depth_stats = {}
        for d in self.data:
            depth = d["context"]["follow_up_depth"]
            depth_stats[depth] = depth_stats.get(depth, 0) + 1
        
        print(f"\n追问深度分布:")
        for depth in sorted(depth_stats.keys()):
            print(f"  第{depth}层: {depth_stats[depth]}条")
        
        return self.data

# ========== 主函数 ==========
def main():
    generator = TrainingDataGenerator()
    data = generator.generate_all_data(target_count=500)
    
    # 保存
    output_file = "./data/bert_training_500.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\n[SAVE] 数据已保存到: {output_file}")
    
    # 显示几个示例
    print(f"\n[SAMPLE] 示例数据:")
    for i in [0, 100, 200, 300, 400]:
        sample = data[i]
        print(f"\n--- 样本 {i+1} ---")
        print(f"问题: {sample['question']}")
        print(f"回答: {sample['answer'][:60]}...")
        print(f"深度: {sample['context']['follow_up_depth']}")
        print(f"犹豫度: {sample['context']['hesitation_score']}")
        print(f"标签: {sample['label']}")

if __name__ == "__main__":
    main()

