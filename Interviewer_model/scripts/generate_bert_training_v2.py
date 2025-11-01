"""
生成BERT训练数据 V2
根据用户反馈重新设计：
1. 更真实的多句回答
2. 更精准的决策理由（基于候选者的具体表述）
3. 填充词和hesitation_score一致
4. 动态追问深度（不固定层数）
"""
import json
import random
from typing import Dict, List, Tuple

class RealisticTrainingDataGenerator:
    """生成更真实的训练数据"""
    
    def __init__(self):
        self.data = []
        
        # 技术领域和具体话题
        self.tech_domains = {
            "Python后端": ["Django ORM", "异步编程", "性能优化", "Flask蓝图", "单元测试"],
            "前端开发": ["React Hooks", "Vue响应式", "Webpack配置", "状态管理", "性能优化"],
            "数据库": ["索引优化", "事务隔离", "查询优化", "Redis缓存", "分库分表"],
            "系统设计": ["微服务", "负载均衡", "消息队列", "缓存策略", "限流降级"],
            "算法": ["动态规划", "二叉树遍历", "哈希表", "排序算法", "图算法"],
            "Git": ["rebase vs merge", "分支策略", "冲突解决", "代码回滚", "cherry-pick"],
            "容器化": ["Docker网络", "镜像优化", "Kubernetes部署", "服务编排", "监控日志"],
        }
        
        # 填充词库
        self.fillers = ["嗯...", "那个...", "就是...", "这个...", "其实...", "怎么说呢..."]
    
    def _generate_realistic_answer(
        self, 
        quality: str,  # "excellent", "good", "vague", "poor", "refuse"
        topic: str,
        depth: int
    ) -> Tuple[str, float, str]:
        """
        生成真实的回答
        返回：(回答文本, hesitation_score, 关键点)
        """
        
        if quality == "excellent":
            # 优秀回答：详细、有结构、有实例
            templates = [
                f"关于{topic}，我在实际项目中有比较深入的应用。首先，它的核心原理是{{principle}}。其次，在使用时要注意{{attention}}。我们项目中遇到过一个典型场景，{{example}}。最后通过{{solution}}解决了这个问题，效果很好，性能提升了大约{{improvement}}。",
                f"这个问题我比较熟悉。{topic}主要用于{{usage}}。我们团队在使用时总结了几个最佳实践：第一，{{practice1}}；第二，{{practice2}}；第三，{{practice3}}。特别是在{{scenario}}场景下，这些实践帮助我们避免了很多坑。",
                f"{topic}是我们项目的核心技术之一。从技术选型来说，我们选择它是因为{{reason}}。在实现上，我负责了{{responsibility}}模块，主要解决了{{problem}}的问题。具体来说，{{detail}}，最终达到了{{result}}的效果。"
            ]
            answer = random.choice(templates).format(
                principle=random.choice(["数据持久化", "状态管理", "资源调度", "请求处理"]),
                attention=random.choice(["内存泄漏", "并发安全", "异常处理", "资源释放"]),
                example=random.choice(["高并发情况下出现了瓶颈", "数据一致性问题", "响应时间过长"]),
                solution=random.choice(["引入缓存", "优化查询", "异步处理", "连接池"]),
                improvement=random.choice(["30%", "50%", "2倍", "40%"]),
                usage=random.choice(["数据处理", "状态同步", "资源管理", "请求转发"]),
                practice1=random.choice(["避免过度优化", "及时释放资源", "使用连接池"]),
                practice2=random.choice(["做好异常处理", "添加日志监控", "设置超时时间"]),
                practice3=random.choice(["编写单元测试", "做好文档记录", "代码审查"]),
                scenario=random.choice(["高并发", "大数据量", "低延迟", "高可用"]),
                reason=random.choice(["性能好", "生态完善", "团队熟悉", "社区活跃"]),
                responsibility=random.choice(["核心业务", "数据处理", "接口设计", "性能优化"]),
                problem=random.choice(["响应慢", "内存占用高", "并发冲突", "数据不一致"]),
                detail=random.choice(["采用了异步方案", "引入了缓存机制", "优化了算法", "重构了架构"]),
                result=random.choice(["稳定运行", "性能提升", "用户满意", "达到预期"])
            )
            hesitation = round(random.uniform(0.05, 0.15), 2)
            key_point = "详细描述了原理、实践和效果"
            
        elif quality == "good":
            # 良好回答：有内容但不够深入
            templates = [
                f"关于{topic}，我用过。主要是在{{context}}的时候使用。它的作用是{{function}}。我们项目中主要用它来{{application}}，效果还不错。",
                f"{topic}我有了解。它主要解决{{problem}}的问题。我在项目中使用过，比如{{example}}。总体感觉{{feeling}}。",
                f"这个我做过。{topic}在我们项目里主要用于{{usage}}。具体实现上，我记得是{{implementation}}。后来运行还算稳定。"
            ]
            answer = random.choice(templates).format(
                context=random.choice(["开发新功能", "优化性能", "重构代码", "解决bug"]),
                function=random.choice(["提高效率", "简化逻辑", "降低耦合", "提升性能"]),
                application=random.choice(["处理请求", "管理数据", "优化查询", "控制流程"]),
                problem=random.choice(["性能瓶颈", "代码复杂", "资源浪费", "响应慢"]),
                example=random.choice(["用户登录模块", "数据查询接口", "文件上传功能", "缓存系统"]),
                feeling=random.choice(["比较好用", "还算稳定", "符合需求", "有点复杂"]),
                usage=random.choice(["数据处理", "请求转发", "状态管理", "资源调度"]),
                implementation=random.choice(["配置了参数", "写了一些代码", "调用了API", "参考了文档"])
            )
            hesitation = round(random.uniform(0.15, 0.30), 2)
            key_point = "提到了使用场景但缺少细节"
            
        elif quality == "vague":
            # 模糊回答：承认用过但说不清楚
            filler1, filler2, filler3 = random.sample(self.fillers, 3)
            templates = [
                f"{filler1}{topic}我{filler2}用过一些，{filler3}做了基本的功能，具体的细节有点记不清了。好像是用来{{vague_function}}的，但是{filler1}具体怎么实现的我有点忘了。",
                f"{filler2}这个我之前{filler1}接触过，{filler3}当时项目里有用到。大概是{{vague_usage}}，但是{filler2}深入的东西我不太记得了，时间有点久了。",
                f"{filler1}{topic}的话，我{filler3}知道一点，{filler2}就是{{vague_concept}}吧。我们项目里好像有用，但是{filler1}我没有深入研究过。"
            ]
            answer = random.choice(templates).format(
                vague_function=random.choice(["处理数据", "优化性能", "管理状态", "控制流程"]),
                vague_usage=random.choice(["提高效率的", "解决问题的", "简化代码的", "优化系统的"]),
                vague_concept=random.choice(["一种技术方案", "一个工具", "一种设计模式", "一个框架"])
            )
            hesitation = round(random.uniform(0.45, 0.65), 2)
            key_point = "承认用过但说不清细节"
            
        elif quality == "poor":
            # 差回答：基本答不上来
            filler1, filler2, filler3 = random.sample(self.fillers, 3)
            templates = [
                f"{filler1}{filler2}这个{topic}{filler3}我...好像...听说过，但是{filler1}没有实际用过。{filler2}不太了解它的原理，{filler3}可能需要学习一下。",
                f"{filler2}{topic}...{filler1}怎么说呢...{filler3}我确实不太熟悉。{filler1}项目里好像没有涉及到这块，{filler2}所以我没有经验。",
                f"{filler3}抱歉，{filler1}{topic}这块{filler2}我真的不太懂。{filler1}之前没有深入了解过，{filler3}只是听说过这个概念。"
            ]
            answer = random.choice(templates)
            hesitation = round(random.uniform(0.60, 0.80), 2)
            key_point = "基本不了解"
            
        else:  # refuse
            # 明确拒绝：直接说不会
            templates = [
                f"{topic}我不会，没用过。",
                f"这个我不了解，项目里没接触过。",
                f"抱歉，{topic}我不熟悉，没有相关经验。",
                f"不好意思，这块我确实不懂，之前没学过。"
            ]
            answer = random.choice(templates)
            hesitation = round(random.uniform(0.05, 0.15), 2)  # 直接拒绝反而不犹豫
            key_point = "明确表示不会"
        
        return answer, hesitation, key_point
    
    def _generate_conversation_chain(
        self, 
        domain: str, 
        topic: str,
        job_title: str = "Python后端工程师"
    ) -> List[Dict]:
        """
        生成一个完整的对话链（多轮追问）
        """
        chain = []
        
        # 第1轮：开场问题
        first_question = f"请谈谈你对{topic}的理解"
        
        # 随机决定候选者的能力水平
        skill_level = random.choice([
            "expert",      # 专家：5-7轮追问，都答得好
            "proficient",  # 熟练：3-5轮，大部分答得好
            "basic",       # 基础：2-3轮，开始还行后面不行
            "weak",        # 薄弱：1-2轮，第一轮就模糊
            "none"         # 不会：1轮，直接说不会
        ])
        
        if skill_level == "expert":
            # 专家级：5-7轮深入追问
            num_rounds = random.randint(5, 7)
            quality_sequence = ["excellent"] * num_rounds
            
        elif skill_level == "proficient":
            # 熟练：3-5轮，前几轮好，后面可能模糊
            num_rounds = random.randint(3, 5)
            quality_sequence = ["excellent"] * 2 + ["good"] * (num_rounds - 2)
            
        elif skill_level == "basic":
            # 基础：2-3轮，第一轮还行，后面就不行了
            num_rounds = random.randint(2, 3)
            quality_sequence = ["good", "vague"] + ["poor"] * (num_rounds - 2)
            
        elif skill_level == "weak":
            # 薄弱：1-2轮，第一轮就模糊
            num_rounds = random.randint(1, 2)
            quality_sequence = ["vague"] + ["poor"] * (num_rounds - 1)
            
        else:  # none
            # 不会：1轮，直接拒绝
            num_rounds = 1
            quality_sequence = ["refuse"]
        
        # 生成对话链
        current_question = first_question
        for depth in range(1, num_rounds + 1):
            quality = quality_sequence[depth - 1]
            answer, hesitation, key_point = self._generate_realistic_answer(quality, topic, depth)
            
            # 决策逻辑
            if depth == num_rounds:
                # 最后一轮
                if quality in ["excellent", "good"]:
                    # 答得好，判定为掌握该领域
                    label = "NEXT_TOPIC"
                    reason = f"候选者在{depth}轮追问中持续展示了对{topic}的深入理解，已充分验证该领域能力"
                    reason_type = "positive"
                else:
                    # 答不好，换话题
                    label = "NEXT_TOPIC"
                    if quality == "refuse":
                        reason = f"候选者明确表示不了解{topic}，应换其他话题"
                    else:
                        reason = f"经过{depth}轮追问，候选者对{topic}的理解仍然模糊/不足，建议换话题"
                    reason_type = "negative"
            else:
                # 非最后一轮
                if quality == "refuse":
                    # 明确拒绝，应该换话题（但实际是最后一轮了）
                    label = "NEXT_TOPIC"
                    reason = f"候选者明确表示不了解{topic}，应换其他话题"
                    reason_type = "negative"
                    
                elif quality == "poor":
                    # 答不上来，应该换话题
                    label = "NEXT_TOPIC"
                    reason = f"候选者对{topic}理解不足，连续追问效果不佳，建议换话题"
                    reason_type = "negative"
                    
                elif quality == "vague":
                    # 模糊回答，但承认用过 -> 继续追问
                    label = "FOLLOW_UP"
                    reason = f"候选者承认使用过{topic}但未展开细节（提到了'{key_point}'），需追问具体实现"
                    reason_type = None
                    
                elif quality == "good":
                    # 回答良好但不深入 -> 继续追问
                    label = "FOLLOW_UP"
                    reason = f"候选者{key_point}，可以继续追问更深入的技术细节"
                    reason_type = None
                    
                else:  # excellent
                    # 回答优秀 -> 继续深挖
                    label = "FOLLOW_UP"
                    reason = f"候选者回答质量高，可以继续深入考察{topic}的高级应用或原理"
                    reason_type = None
            
            # 构建数据条目
            entry = {
                "question": current_question,
                "answer": answer,
                "label": label,
                "reason": reason,
                "context": {
                    "job_title": job_title,
                    "topic": topic,
                    "domain": domain,
                    "follow_up_depth": depth,
                    "hesitation_score": hesitation,
                    "answer_length": len(answer),
                    "skill_level": skill_level
                }
            }
            
            if label == "NEXT_TOPIC":
                entry["reason_type"] = reason_type
            
            chain.append(entry)
            
            # 准备下一轮问题（如果有的话）
            if depth < num_rounds:
                if quality == "vague":
                    # 对模糊回答追问细节
                    current_question = random.choice([
                        f"你提到做了基本功能，能具体说说做了什么吗？",
                        f"能详细讲讲你是怎么使用{topic}的吗？",
                        f"可以举个具体的例子说明一下吗？"
                    ])
                elif quality == "good":
                    # 对良好回答追问更深
                    current_question = random.choice([
                        f"你在使用{topic}时遇到过什么问题吗？怎么解决的？",
                        f"{topic}的底层原理你了解吗？",
                        f"能说说{topic}的最佳实践或注意事项吗？",
                        f"为什么选择{topic}而不是其他方案？"
                    ])
                else:  # excellent
                    # 对优秀回答追问高级话题
                    current_question = random.choice([
                        f"如果在高并发场景下使用{topic}，你会怎么优化？",
                        f"{topic}有什么局限性或缺点吗？你会如何改进？",
                        f"能对比一下{topic}和类似技术的优劣吗？",
                        f"在生产环境中使用{topic}，需要注意哪些坑？"
                    ])
        
        return chain
    
    def generate_all_data(self, target_count: int = 500) -> List[Dict]:
        """生成所有训练数据"""
        print(f"开始生成{target_count}条训练数据...")
        print("采用真实对话链生成策略...\n")
        
        all_topics = [(domain, topic) for domain, topics in self.tech_domains.items() for topic in topics]
        
        conversation_count = 0
        while len(self.data) < target_count:
            # 随机选择一个话题
            domain, topic = random.choice(all_topics)
            
            # 生成一个对话链
            chain = self._generate_conversation_chain(domain, topic)
            self.data.extend(chain)
            
            conversation_count += 1
            if conversation_count % 20 == 0:
                print(f"已生成{conversation_count}个对话链，共{len(self.data)}条数据...")
        
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
        print(f"  - 积极换话题（掌握）: {next_topic_positive}条")
        print(f"  - 消极换话题（不会）: {next_topic_negative}条")
        
        # 深度统计
        depth_stats = {}
        for d in self.data:
            depth = d["context"]["follow_up_depth"]
            depth_stats[depth] = depth_stats.get(depth, 0) + 1
        
        print(f"\n追问深度分布:")
        for depth in sorted(depth_stats.keys()):
            print(f"  第{depth}层: {depth_stats[depth]}条")
        
        # 技能水平统计
        skill_stats = {}
        for d in self.data:
            skill = d["context"]["skill_level"]
            skill_stats[skill] = skill_stats.get(skill, 0) + 1
        
        print(f"\n候选者技能水平分布:")
        for skill, count in sorted(skill_stats.items(), key=lambda x: x[1], reverse=True):
            print(f"  {skill}: {count}条 ({count/len(self.data)*100:.1f}%)")
        
        return self.data

# ========== 主函数 ==========
def main():
    generator = RealisticTrainingDataGenerator()
    data = generator.generate_all_data(target_count=1500)
    
    # 保存
    output_file = "./data/bert_training_1500.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\n[SAVE] 数据已保存到: {output_file}")
    
    # 显示几个示例
    print(f"\n[SAMPLE] 示例数据（展示对话链）:")
    
    # 找几个不同skill_level的对话链
    shown_skills = set()
    for i, sample in enumerate(data):
        skill = sample['context']['skill_level']
        depth = sample['context']['follow_up_depth']
        
        # 每种skill_level只展示一次，且只展示第1轮
        if skill not in shown_skills and depth == 1:
            shown_skills.add(skill)
            print(f"\n{'='*60}")
            print(f"[{skill.upper()}] {sample['context']['domain']} - {sample['context']['topic']}")
            print(f"{'='*60}")
            
            # 找到这个话题的所有轮次
            topic = sample['context']['topic']
            topic_chain = [d for d in data if d['context']['topic'] == topic and d['context'].get('skill_level') == skill][:7]
            
            for round_data in topic_chain:
                print(f"\n第{round_data['context']['follow_up_depth']}轮:")
                print(f"  问题: {round_data['question']}")
                print(f"  回答: {round_data['answer'][:80]}...")
                print(f"  犹豫度: {round_data['context']['hesitation_score']}")
                print(f"  决策: {round_data['label']}")
                print(f"  理由: {round_data['reason'][:60]}...")
            
            if len(shown_skills) >= 3:
                break

if __name__ == "__main__":
    main()

