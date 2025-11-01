"""
生成Qwen对话生成模型训练数据
目标：训练Qwen学习如何像真实面试官一样提问和回应
数据量：2000条
"""
import json
import random
from typing import Dict, List, Tuple

class QwenDataGenerator:
    """Qwen训练数据生成器"""
    
    def __init__(self):
        self.data = []
        
        # 技术话题（与BERT数据一致）
        self.tech_topics = {
            "Python基础": ["装饰器", "生成器", "GIL", "垃圾回收", "上下文管理器", "元类", "多继承"],
            "Python Web": ["Django ORM", "Flask", "异步视图", "RESTful API", "中间件", "Session"],
            "数据库": ["MySQL索引", "Redis缓存", "事务隔离", "慢查询优化", "主从复制", "分库分表"],
            "前端": ["React Hooks", "Vue响应式", "虚拟DOM", "状态管理", "Webpack", "性能优化"],
            "系统设计": ["微服务", "负载均衡", "限流", "熔断降级", "分布式事务", "CAP理论"],
            "消息队列": ["RabbitMQ", "Kafka", "消息丢失", "重复消费", "死信队列", "延迟队列"],
            "算法": ["动态规划", "二叉树", "哈希表", "排序", "图算法", "回溯"],
            "Git": ["rebase vs merge", "分支策略", "冲突解决", "cherry-pick", "stash"],
            "Docker": ["镜像优化", "网络模式", "数据卷", "容器编排", "资源限制"],
        }
        
        self.all_topics = []
        for domain, topics in self.tech_topics.items():
            for topic in topics:
                self.all_topics.append((domain, topic))
    
    def _generate_opening_question(self, topic: str) -> str:
        """生成开场问题"""
        templates = [
            f"请谈谈你对{topic}的理解",
            f"能说说你在项目中是如何使用{topic}的吗？",
            f"你了解{topic}吗？能简单介绍一下吗？",
            f"请介绍一下{topic}的应用场景",
            f"说说你对{topic}的看法吧",
        ]
        return random.choice(templates)
    
    def _generate_follow_up_question(
        self, 
        topic: str, 
        candidate_quality: str,
        depth: int
    ) -> str:
        """生成追问问题"""
        
        if candidate_quality == "excellent":
            # 对优秀回答，问更深入的问题
            templates = [
                f"很好！如果在高并发场景下使用{topic}，你会怎么优化？",
                f"{topic}有什么局限性或缺点吗？你会如何改进？",
                f"能对比一下{topic}和类似技术的优劣吗？",
                f"在生产环境中使用{topic}，需要注意哪些坑？",
                f"非常棒！那你能说说{topic}的底层原理吗？",
                f"你在使用{topic}时遇到过什么棘手的问题吗？",
            ]
        elif candidate_quality == "good":
            # 对良好回答，追问细节
            templates = [
                f"你在使用{topic}时遇到过什么问题吗？怎么解决的？",
                f"能详细说说你是怎么使用{topic}的吗？",
                f"可以举个具体的例子说明一下吗？",
                f"为什么选择{topic}而不是其他方案？",
                f"能说说{topic}的最佳实践吗？",
            ]
        else:  # vague
            # 对模糊回答，引导具体化
            templates = [
                f"你提到做了基本功能，能具体说说做了什么吗？",
                f"能详细讲讲你是怎么使用{topic}的吗？",
                f"可以举个具体的例子说明一下吗？",
                f"能说说具体的实现方式吗？",
            ]
        
        return random.choice(templates)
    
    def _generate_topic_change_response(
        self, 
        reason: str,  # "refuse", "poor", "positive"
        new_topic: str
    ) -> str:
        """生成换话题的回复"""
        
        if reason == "refuse" or reason == "poor":
            # 消极换话题：候选人不会
            transitions = [
                f"没关系，我们聊聊其他的。",
                f"好的，那我们换个话题。",
                f"理解，那换个方向聊聊。",
                f"没问题，我们谈谈别的。",
            ]
            transition = random.choice(transitions)
            new_question = random.choice([
                f"你熟悉{new_topic}吗？",
                f"能说说你对{new_topic}的了解吗？",
                f"请介绍一下{new_topic}",
                f"谈谈你在{new_topic}方面的经验吧",
            ])
            return f"{transition} {new_question}"
        else:
            # 积极换话题：已经充分考察
            affirmations = [
                "很好！",
                "不错！",
                "非常棒！",
                "回答得很全面！",
            ]
            affirmation = random.choice(affirmations)
            new_question = random.choice([
                f"那我们再聊聊{new_topic}吧",
                f"接下来谈谈{new_topic}",
                f"我们换个话题，说说{new_topic}",
            ])
            return f"{affirmation} {new_question}"
    
    def _generate_encouragement(self, candidate_quality: str) -> str:
        """生成鼓励性反馈"""
        
        if candidate_quality == "excellent":
            return random.choice([
                "很好！",
                "非常棒！",
                "回答得很全面！",
                "不错！",
                "很有深度！",
                "很专业！",
            ])
        elif candidate_quality == "good":
            return random.choice([
                "嗯，不错",
                "还可以",
                "有一定理解",
                "可以的",
            ])
        else:
            return ""
    
    def _generate_conversation_sample(self) -> Dict:
        """生成一条对话样本"""
        
        domain, topic = random.choice(self.all_topics)
        
        # 随机候选人水平
        skill_level = random.choice([
            "excellent",  # 30%
            "excellent",
            "excellent",
            "good",       # 30%
            "good",
            "good",
            "vague",      # 20%
            "vague",
            "poor",       # 15%
            "refuse"      # 5%
        ])
        
        # 随机对话轮次
        if skill_level == "excellent":
            num_rounds = random.randint(4, 7)
        elif skill_level == "good":
            num_rounds = random.randint(2, 4)
        elif skill_level == "vague":
            num_rounds = random.randint(1, 3)
        elif skill_level == "poor":
            num_rounds = random.randint(1, 2)
        else:  # refuse
            num_rounds = 1
        
        # 随机选择回应类型
        response_type = random.choice([
            "follow_up",      # 追问
            "follow_up",
            "topic_change",   # 换话题
            "encourage"       # 鼓励性反馈
        ])
        
        # 构建对话历史
        conversation_history = []
        
        # 第一轮
        question = self._generate_opening_question(topic)
        
        # 生成system prompt
        system_prompt = f"你是一位专业、友好的{domain}技术面试官，正在面试候选人。你需要根据候选人的回答，决定是继续深入追问、换话题，还是给予鼓励。"
        
        # 生成instruction
        if response_type == "follow_up":
            instruction = f"根据候选人的回答，生成一个追问问题，深入考察候选人对{topic}的理解"
            # 候选人回答（模拟）
            if skill_level == "excellent":
                candidate_answer = f"我在项目中深入使用过{topic}。从原理上讲，它主要解决了XX问题。我们团队在实际应用中，通过YY方式进行了优化，效果很好，性能提升了约30%。"
            elif skill_level == "good":
                candidate_answer = f"{topic}我用过。主要是在项目开发时使用，它的作用是提高效率。我们主要用它来处理业务逻辑，效果还不错。"
            else:  # vague
                candidate_answer = f"嗯...{topic}我...那个...用过一些，就是...做了基本的功能，具体的细节有点记不清了。"
            
            response = self._generate_follow_up_question(topic, skill_level, 1)
            
        elif response_type == "topic_change":
            new_domain, new_topic = random.choice(self.all_topics)
            
            if skill_level in ["refuse", "poor"]:
                instruction = f"候选人对{topic}不了解或答不上来，需要友好地换一个话题"
                if skill_level == "refuse":
                    candidate_answer = f"{topic}我不会，没用过。"
                else:
                    candidate_answer = f"嗯...这个{topic}...怎么说呢...我确实不太懂。之前没有接触过，只是听说过这个名字。"
                
                response = self._generate_topic_change_response("poor", new_topic)
            else:
                # 积极换话题
                instruction = f"候选人对{topic}回答得很好，已经充分考察，可以换一个新话题"
                candidate_answer = f"关于{topic}，我在项目中有深入实践。从原理、实践到优化都有丰富经验。我们通过XX方式解决了YY问题，最终性能提升了50%，系统非常稳定。"
                response = self._generate_topic_change_response("positive", new_topic)
        
        else:  # encourage
            instruction = f"对候选人的回答给予肯定和鼓励，然后继续追问"
            if skill_level == "excellent":
                candidate_answer = f"关于{topic}，我在实际项目中有比较深入的应用。我们团队通过XX方式优化了YY，效果很明显，性能提升了约40%。"
            else:
                candidate_answer = f"{topic}我有一些了解。之前项目中接触过，主要是实现了基本功能，效果还可以。"
            
            encouragement = self._generate_encouragement(skill_level)
            follow_up = self._generate_follow_up_question(topic, skill_level, 1)
            response = f"{encouragement} {follow_up}"
        
        # 构建训练样本（使用对话格式）
        sample = {
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": f"面试官问题：{question}\n候选人回答：{candidate_answer}\n\n任务：{instruction}"
                },
                {
                    "role": "assistant",
                    "content": response
                }
            ],
            "metadata": {
                "domain": domain,
                "topic": topic,
                "skill_level": skill_level,
                "response_type": response_type,
                "num_rounds": num_rounds
            }
        }
        
        return sample
    
    def generate_all_data(self, target_count: int = 2000) -> List[Dict]:
        """生成所有数据"""
        print(f"开始生成{target_count}条Qwen训练数据...")
        print("训练目标：学习如何像真实面试官一样提问和回应\n")
        
        for i in range(target_count):
            sample = self._generate_conversation_sample()
            self.data.append(sample)
            
            if (i + 1) % 200 == 0:
                print(f"已生成 {i + 1}/{target_count} 条数据...")
        
        # 统计
        response_types = {}
        skill_levels = {}
        
        for item in self.data:
            rt = item['metadata']['response_type']
            sl = item['metadata']['skill_level']
            
            response_types[rt] = response_types.get(rt, 0) + 1
            skill_levels[sl] = skill_levels.get(sl, 0) + 1
        
        print(f"\n[OK] 数据生成完成！")
        print(f"总计: {len(self.data)}条")
        
        print(f"\n回应类型分布:")
        for rt, count in sorted(response_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  {rt}: {count}条 ({count/len(self.data)*100:.1f}%)")
        
        print(f"\n候选者水平分布:")
        for sl, count in sorted(skill_levels.items(), key=lambda x: x[1], reverse=True):
            print(f"  {sl}: {count}条 ({count/len(self.data)*100:.1f}%)")
        
        return self.data

def main():
    generator = QwenDataGenerator()
    data = generator.generate_all_data(target_count=2000)
    
    # 保存为标准格式（适用于LoRA微调）
    output_file = "./data/qwen_training_2000.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\n[SAVE] 数据已保存到: {output_file}")
    print(f"文件大小: {len(json.dumps(data, ensure_ascii=False)) / 1024 / 1024:.2f} MB")
    
    # 显示示例
    print(f"\n[SAMPLE] 数据示例:")
    for i in [0, 500, 1000, 1500]:
        if i < len(data):
            sample = data[i]
            print(f"\n--- 样本 {i+1} ({sample['metadata']['response_type']}, {sample['metadata']['skill_level']}) ---")
            print(f"话题: {sample['metadata']['topic']}")
            print(f"System: {sample['messages'][0]['content'][:50]}...")
            print(f"User: {sample['messages'][1]['content'][:80]}...")
            print(f"Assistant: {sample['messages'][2]['content']}")

if __name__ == "__main__":
    main()

