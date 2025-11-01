"""
生成大规模BERT训练数据集（1500条）
改进策略：
1. 更丰富的技术话题（100+个）
2. 更自然的回答模板（减少机械感）
3. 基于具体内容的精准决策理由
4. 真实的对话链逻辑
"""
import json
import random
from typing import Dict, List, Tuple

class AdvancedBERTDataGenerator:
    """改进版BERT数据生成器"""
    
    def __init__(self):
        self.data = []
        
        # 扩展技术话题库（100+个话题）
        self.tech_topics = {
            "Python基础": [
                "装饰器原理", "生成器和迭代器", "GIL全局锁", "垃圾回收机制", 
                "上下文管理器", "元类metaclass", "多继承MRO", "闭包"
            ],
            "Python Web": [
                "Django ORM优化", "Flask蓝图", "Django中间件", "RESTful API设计",
                "Django信号机制", "异步视图", "WSGI vs ASGI", "Session管理"
            ],
            "Python进阶": [
                "asyncio异步编程", "协程", "多进程vs多线程", "进程池线程池",
                "信号量Semaphore", "事件Event", "队列Queue", "锁机制"
            ],
            "数据库MySQL": [
                "索引原理B+树", "事务隔离级别", "MVCC机制", "慢查询优化",
                "主从复制", "分库分表", "联合索引", "覆盖索引"
            ],
            "Redis": [
                "数据结构实现", "持久化RDB和AOF", "缓存穿透", "缓存雪崩",
                "缓存击穿", "Redis集群", "哨兵模式", "发布订阅"
            ],
            "前端React": [
                "Hooks原理", "虚拟DOM", "Fiber架构", "状态管理Redux",
                "useEffect依赖", "useMemo优化", "React性能优化", "SSR服务端渲染"
            ],
            "前端Vue": [
                "响应式原理", "computed vs watch", "Vue3 Composition API", "虚拟DOM diff",
                "Vuex状态管理", "Vue Router", "keep-alive", "slot插槽"
            ],
            "系统设计": [
                "微服务架构", "服务注册发现", "负载均衡算法", "限流算法",
                "熔断降级", "分布式事务", "CAP理论", "一致性哈希"
            ],
            "消息队列": [
                "RabbitMQ", "Kafka原理", "消息丢失处理", "消息重复消费",
                "死信队列", "延迟队列", "消息顺序性", "Kafka分区"
            ],
            "网络协议": [
                "TCP三次握手", "TCP四次挥手", "HTTP vs HTTPS", "HTTPS加密过程",
                "HTTP2特性", "WebSocket", "Cookie vs Session", "跨域CORS"
            ],
            "算法基础": [
                "动态规划", "贪心算法", "回溯算法", "分治算法",
                "二叉树遍历", "图的遍历", "排序算法", "查找算法"
            ],
            "数据结构": [
                "哈希表原理", "红黑树", "B树B+树", "跳表",
                "堆heap", "Trie树", "并查集", "LRU缓存"
            ],
            "Linux": [
                "进程vs线程", "内存管理", "文件系统", "IO模型",
                "select poll epoll", "软链接硬链接", "常用命令", "shell脚本"
            ],
            "Docker": [
                "镜像分层", "容器vs虚拟机", "Dockerfile优化", "网络模式",
                "数据卷", "镜像仓库", "容器编排", "资源限制"
            ],
            "Git": [
                "rebase vs merge", "分支管理策略", "冲突解决", "cherry-pick",
                "stash暂存", "reset vs revert", "git工作流", "子模块"
            ],
            "设计模式": [
                "单例模式", "工厂模式", "观察者模式", "策略模式",
                "装饰器模式", "代理模式", "适配器模式", "模板方法"
            ],
            "安全": [
                "XSS攻击", "CSRF攻击", "SQL注入", "密码加密",
                "JWT认证", "OAuth2.0", "HTTPS原理", "同源策略"
            ],
            "性能优化": [
                "前端性能优化", "后端性能优化", "数据库优化", "CDN加速",
                "懒加载", "代码分割", "缓存策略", "图片优化"
            ]
        }
        
        # 所有话题列表
        self.all_topics = []
        for domain, topics in self.tech_topics.items():
            for topic in topics:
                self.all_topics.append((domain, topic))
        
        # 填充词
        self.fillers = ["嗯...", "那个...", "就是...", "这个...", "其实...", "怎么说呢...", "emmm..."]
        
        # 技术术语库（用于生成真实感）
        self.tech_terms = {
            "数据库": ["事务", "索引", "查询", "优化", "锁", "隔离级别", "ACID", "范式"],
            "缓存": ["Redis", "Memcached", "穿透", "雪崩", "击穿", "过期策略", "淘汰算法"],
            "性能": ["QPS", "TPS", "延迟", "吞吐量", "并发", "响应时间", "CPU", "内存"],
            "架构": ["微服务", "SOA", "中台", "网关", "服务发现", "负载均衡", "容错"],
        }
    
    def _generate_answer_with_details(
        self, 
        quality: str, 
        topic: str, 
        depth: int
    ) -> Tuple[str, float, str]:
        """生成包含细节的真实回答"""
        
        if quality == "excellent":
            # 专家级回答：结构化+细节+数字+经验
            patterns = [
                # 模式1：原理+实践+效果
                lambda: f"关于{topic}，我在项目中有深入实践。从原理上讲，它{random.choice(['解决了', '优化了', '提供了'])}{random.choice(['性能瓶颈', '扩展性', '一致性', '可维护性'])}问题。我们团队在{random.choice(['电商系统', '金融平台', '社交应用', '内容平台'])}中{random.choice(['大规模应用', '深度优化', '完整实现']}了这个技术。具体来说，{random.choice(['第一', '首先'])}，{random.choice(['梳理了业务流程', '分析了性能瓶颈', '设计了技术方案'])}；{random.choice(['第二', '然后'])}，{random.choice(['实现了核心逻辑', '引入了监控体系', '优化了关键路径'])}；{random.choice(['第三', '最后'])}，通过{random.choice(['压测验证', 'AB测试', '灰度发布'])}，{random.choice(['QPS从800提升到3000', '响应时间降低60%', 'CPU使用率下降40%', '成本节省50%'])}。{random.choice(['整体效果非常好', '达到了预期目标', '获得了业务认可'])}。",
                
                # 模式2：对比+选型+踩坑
                lambda: f"{topic}这块我比较熟悉。{random.choice(['当时', '之前', '我们']}在技术选型时对比过几个方案。{random.choice(['最终', '综合考虑后', '权衡利弊'])}选择了{topic}，主要原因是{random.choice(['性能更好', '生态成熟', '团队熟悉', '社区活跃'])}。在实施过程中也踩过坑，比如{random.choice(['初期配置不当导致性能不达标', '并发场景下出现数据不一致', '资源占用过高影响其他服务'])}。后来通过{random.choice(['调整参数', '优化架构', '引入监控'])}解决了。{random.choice(['现在运行很稳定', '已经支撑千万级用户', '日处理量达到百万级'])}。",
                
                # 模式3：场景+方案+优化
                lambda: f"说到{topic}，我印象很深。{random.choice(['我们', '团队', '项目']}遇到的典型场景是{random.choice(['高并发秒杀', '大数据量查询', '实时数据同步', '分布式事务'])}。{random.choice(['最开始', '初版方案', '1.0版本'])}用的是{random.choice(['传统方式', '简单实现', '开源方案'])}，但{random.choice(['性能不理想', '扩展性差', '维护成本高'])}。{random.choice(['后来', '迭代时', '重构后'])}深入研究了{topic}的{random.choice(['底层原理', '最佳实践', '高级特性'])}，{random.choice(['重新设计了架构', '优化了核心逻辑', '引入了新技术栈'])}。{random.choice(['效果立竿见影', '问题彻底解决', '性能提升明显'])}，{random.choice(['用户体验提升很多', '运维压力大幅降低', '系统稳定性大幅提高'])}。"
            ]
            answer = random.choice(patterns)()
            hesitation = round(random.uniform(0.05, 0.15), 2)
            key_point = f"详细阐述了{topic}的{random.choice(['原理、实践和优化经验', '选型理由、踩坑经历和解决方案', '应用场景、技术方案和实施效果'])}"
            
        elif quality == "good":
            # 良好回答：有内容但不够深
            patterns = [
                lambda: f"{topic}我用过。主要是在{random.choice(['项目开发', '功能实现', '性能优化', '问题排查'])}时使用。它的作用是{random.choice(['提高效率', '简化流程', '降低复杂度', '提升性能'])}。{random.choice(['我们', '团队', '项目里'])}主要用它来{random.choice(['处理业务逻辑', '管理数据', '优化查询', '控制流程'])}。{random.choice(['效果还不错', '基本满足需求', '运行比较稳定', '达到了预期'])}，{random.choice(['用户反馈也比较好', '后续还在持续优化', '准备进一步改进'])}。",
                
                lambda: f"关于{topic}，我有一些了解。{random.choice(['之前', '曾经', '项目中'])}接触过，主要是{random.choice(['实现了基本功能', '解决了具体问题', '完成了业务需求'])}。{random.choice(['虽然', '不过', '但是'])}深入的原理{random.choice(['还在学习', '了解不多', '掌握不够']}。{random.choice(['主要', '基本', '大概'])}知道{random.choice(['怎么用', '如何配置', '基本操作']}，{random.choice(['能满足日常开发', '应付常见场景', '解决一般问题']}。",
                
                lambda: f"{topic}这个{random.choice(['我做过', '有经验', '用过一段时间'])}。{random.choice(['记得', '印象中', '当时'])}是用来{random.choice(['实现某个功能', '解决某个问题', '优化某个模块']}。{random.choice(['具体的', '详细的', '深入的']}技术细节{random.choice(['记得不太清了', '有点模糊', '需要回忆一下']}，但{random.choice(['大致思路', '基本原理', '核心概念'])}还是了解的。{random.choice(['如果需要', '实际用的时候', '遇到问题']]}可以{random.choice(['查查文档', '看看资料', '请教同事']}。"
            ]
            answer = random.choice(patterns)()
            hesitation = round(random.uniform(0.18, 0.35), 2)
            key_point = f"提到了{topic}的{random.choice(['使用场景', '基本功能', '应用经验'])}但{random.choice(['缺少深入细节', '原理理解不够', '实践经验有限'])}"
            
        elif quality == "vague":
            # 模糊回答：承认用过但说不清
            f1, f2, f3 = random.sample(self.fillers, 3)
            patterns = [
                lambda: f"{f1}{topic}我{f2}用过一些，{f3}{random.choice(['做了基本的功能', '实现了简单的逻辑', '完成了基础需求'])}，但是{f1}{random.choice(['具体的细节', '深入的原理', '高级的特性'])}{random.choice(['有点记不清了', '不太确定', '说不太清楚']}。{random.choice(['好像', '大概', '应该']}是用来{random.choice(['处理数据的', '优化性能的', '管理状态的', '控制流程的'])}，{f2}具体{random.choice(['怎么实现', '如何配置', '为什么这样']}我{random.choice(['有点忘了', '不太记得', '需要想想']}。",
                
                lambda: f"{f2}这个{f1}我之前{random.choice(['接触过', '了解过', '学习过'])}，{f3}当时{random.choice(['项目里', '开发时', '工作中']}有用到。{random.choice(['大概', '好像', '应该']}是{random.choice(['提高效率的', '解决问题的', '优化系统的']}，但是{f1}{random.choice(['深入的东西', '底层原理', '高级用法']}我{random.choice(['不太记得了', '掌握不够', '了解不多']}。{f2}时间{random.choice(['有点久了', '比较长了', '过去挺久']}，{random.choice(['细节有点模糊', '记不太清楚', '需要复习一下']}。",
                
                lambda: f"{f1}{topic}的话，我{f3}{random.choice(['知道一点', '了解一些', '听说过'])}，{f2}就是{random.choice(['那个...', '嗯...', '怎么说呢...']}一种{random.choice(['技术方案', '工具', '框架', '方法']}吧。{random.choice(['我们', '项目里', '团队']}{random.choice(['好像有用', '应该用过', '可能涉及']}，但{f1}我{random.choice(['没有深入研究', '不太负责这块', '接触不多']}。{random.choice(['大致', '基本', '简单']}的{random.choice(['概念', '用法', '原理']}知道，{random.choice(['详细的', '具体的', '深入的']]}就{random.choice(['不太清楚了', '说不上来', '不确定']}。"
            ]
            answer = random.choice(patterns)()
            hesitation = round(random.uniform(0.48, 0.68), 2)
            key_point = f"承认接触过{topic}但{random.choice(['细节模糊', '理解浅显', '经验不足', '记忆不清'])}"
            
        elif quality == "poor":
            # 差回答：基本答不上来
            f1, f2, f3 = random.sample(self.fillers, 3)
            patterns = [
                lambda: f"{f1}{f2}这个{topic}{f3}我...{random.choice(['好像', '应该', '可能']}...{random.choice(['听说过', '见过', '了解一点']}，但是{f1}{random.choice(['没有实际用过', '不太熟悉', '掌握不了']}。{f2}{random.choice(['项目里', '工作中', '开发时']}{random.choice(['好像没有涉及', '应该没用到', '可能不需要']}这块，{f3}所以我{random.choice(['没有经验', '不太了解', '说不上来']}。{random.choice(['可能需要学习一下', '之后会去了解', '有机会想研究研究']}。",
                
                lambda: f"{f2}{topic}...{f1}怎么说呢...{f3}我{random.choice(['确实不太懂', '真的不熟悉', '基本不了解']}。{random.choice(['之前', '以前', '曾经']}{random.choice(['没有接触过', '没学过', '没用过']}，{f1}{random.choice(['只是听说过这个名字', '知道有这个东西', '了解个大概概念']}。{random.choice(['具体', '详细', '深入']}的{random.choice(['原理', '用法', '实现']}就{random.choice(['完全不清楚了', '真的不知道', '说不出来']}。",
                
                lambda: f"{f3}抱歉，{f1}{topic}这块{f2}我{random.choice(['真的不太会', '确实不了解', '基本不懂']}。{random.choice(['之前', '以前', '过去']}{random.choice(['没有深入学习', '没有实践机会', '没有接触过']}，{f1}只是{random.choice(['听说过名字', '知道有这个', '了解一点概念']}。{random.choice(['如果', '要是', '假如']}工作中{random.choice(['需要用到', '涉及这块', '要求掌握']}，我{random.choice(['会认真学习', '愿意快速上手', '可以努力掌握']}。"
            ]
            answer = random.choice(patterns)()
            hesitation = round(random.uniform(0.62, 0.82), 2)
            key_point = f"对{topic}{random.choice(['基本不了解', '没有经验', '掌握不足', '认知模糊'])}"
            
        else:  # refuse
            # 明确拒绝
            patterns = [
                lambda: f"{topic}我不会，{random.choice(['没用过', '不了解', '不熟悉']}。",
                lambda: f"这个我{random.choice(['不了解', '不清楚', '不懂']}，{random.choice(['项目里没接触过', '工作中没用到', '之前没学过']}。",
                lambda: f"抱歉，{topic}我{random.choice(['确实不熟悉', '真的不会', '完全不了解']}，{random.choice(['没有相关经验', '之前没学过', '没有实践过']}。",
                lambda: f"{random.choice(['不好意思', '抱歉']}，这块我{random.choice(['确实不懂', '真不会', '不了解']}，{random.choice(['之前没接触过', '没学过这个', '不在我的技术栈']}。"
            ]
            answer = random.choice(patterns)()
            hesitation = round(random.uniform(0.08, 0.18), 2)  # 直接拒绝反而不犹豫
            key_point = f"明确表示{random.choice(['不会', '不了解', '不熟悉'])}{topic}"
        
        return answer, hesitation, key_point
    
    def _generate_precise_reason(
        self, 
        label: str, 
        quality: str, 
        topic: str, 
        depth: int, 
        key_point: str,
        hesitation: float
    ) -> str:
        """生成精准的决策理由（基于具体内容）"""
        
        if label == "FOLLOW_UP":
            if quality == "excellent":
                reasons = [
                    f"候选人{key_point}，展现了扎实的理论基础和丰富的实践经验，可以继续深入考察{random.choice(['高级特性', '底层原理', '优化策略', '边界情况'])}",
                    f"候选人回答质量高，{key_point}，建议追问{random.choice(['实现细节', '性能优化', '踩坑经历', '技术选型理由'])}",
                    f"候选人对{topic}理解深入，{key_point}，值得继续探讨{random.choice(['架构设计', '最佳实践', '问题排查', '未来演进'])}方向"
                ]
            elif quality == "good":
                reasons = [
                    f"候选人{key_point}，可以继续追问{random.choice(['具体实现', '技术细节', '遇到的问题', '优化方案'])}来深入评估",
                    f"候选人提到了{topic}的应用，但{key_point}，需要追问{random.choice(['更多细节', '原理理解', '实践经验', '问题处理']}",
                    f"候选人有{topic}的基础认知，{key_point}，建议继续提问验证{random.choice(['深度理解', '实际能力', '问题解决', '技术视野']}"
                ]
            else:  # vague
                reasons = [
                    f"候选人{key_point}，应追问具体内容验证其真实掌握程度（如'你提到做了基本功能，能具体说说吗？'）",
                    f"候选人表示接触过{topic}但未展开，{key_point}，需要引导其{random.choice(['举例说明', '描述场景', '讲述细节', '展示理解'])}",
                    f"候选人对{topic}的回答{key_point}，犹豫度{hesitation}，可以继续追问判断是真会还是只了解皮毛"
                ]
            return random.choice(reasons)
        
        else:  # NEXT_TOPIC
            if quality == "refuse":
                reasons = [
                    f"候选人{key_point}，应立即换话题而非继续追问（避免尴尬）",
                    f"候选人明确表达了{key_point}，继续追问没有意义，建议友好换话题",
                    f"候选人坦诚地说{key_point}，面试官应尊重并换其他话题考察"
                ]
            elif quality == "poor":
                reasons = [
                    f"候选人{key_point}，犹豫度高达{hesitation}，继续追问意义不大，建议换话题",
                    f"候选人对{topic}理解不足，{key_point}，连续追问效果不佳，应该换话题",
                    f"候选人回答{key_point}，且犹豫度{hesitation}表明其确实不掌握，建议换其他技术方向"
                ]
            elif depth >= random.randint(4, 7):  # 积极换话题
                reasons = [
                    f"候选人在{depth}轮追问中{random.choice(['持续展示', '充分证明', '全面展现'])}了对{topic}的{random.choice(['深入理解', '扎实掌握', '丰富经验']}，已充分验证该领域能力，可以换话题考察其他技能",
                    f"经过{depth}轮深入交流，候选人对{topic}的{random.choice(['原理、实践和优化', '理论和实战', '基础和进阶'])}都有很好的掌握，该话题已充分考察，建议切换到其他技术领域",
                    f"{depth}轮问答已全面评估了候选人在{topic}方向的能力（{random.choice(['从基础到高级', '从概念到实践', '从应用到原理'])}），可以换新话题拓宽考察范围"
                ]
            else:
                reasons = [
                    f"候选人{key_point}，回答{random.choice(['模糊敷衍', '缺少实质', '避重就轻', '言之无物'])}，继续追问价值不大，建议换话题",
                    f"候选人对{topic}的{random.choice(['理解停留在表面', '掌握程度有限', '经验明显不足']}，{key_point}，应该换其他话题",
                    f"第{depth}轮追问中候选人{key_point}，且犹豫度{hesitation}，{random.choice(['说明其能力边界已达', '表明进一步追问困难', '显示该方向掌握有限'])}，建议换话题"
                ]
            return random.choice(reasons)
    
    def _generate_conversation_chain(self, domain: str, topic: str) -> List[Dict]:
        """生成完整对话链"""
        chain = []
        
        # 随机技能水平
        skill_level = random.choice([
            ("expert", 6, 7),
            ("expert", 5, 6),
            ("proficient", 4, 5),
            ("proficient", 3, 4),
            ("basic", 2, 3),
            ("weak", 1, 2),
            ("none", 1, 1)
        ])
        
        level_name, min_rounds, max_rounds = skill_level
        num_rounds = random.randint(min_rounds, max_rounds)
        
        # 质量序列
        quality_map = {
            "expert": lambda n: ["excellent"] * n,
            "proficient": lambda n: ["excellent"] * (n//2) + ["good"] * (n - n//2),
            "basic": lambda n: ["good"] + ["vague"] * (n-1) if n > 1 else ["good"],
            "weak": lambda n: ["vague"] + ["poor"] * (n-1) if n > 1 else ["vague"],
            "none": lambda n: ["refuse"]
        }
        quality_sequence = quality_map[level_name](num_rounds)
        
        # 生成对话
        questions = [
            f"请谈谈你对{topic}的理解",
            f"能说说你在项目中是如何使用{topic}的吗？",
            f"你了解{topic}吗？能简单介绍一下吗？",
            f"请介绍一下{topic}的应用场景",
        ]
        
        current_question = random.choice(questions)
        
        for depth in range(1, num_rounds + 1):
            quality = quality_sequence[depth - 1]
            answer, hesitation, key_point = self._generate_answer_with_details(quality, topic, depth)
            
            # 决策
            if depth == num_rounds:
                if quality in ["excellent", "good"]:
                    label = "NEXT_TOPIC"
                    reason_type = "positive"
                else:
                    label = "NEXT_TOPIC"
                    reason_type = "negative"
            else:
                if quality in ["refuse", "poor"]:
                    label = "NEXT_TOPIC"
                    reason_type = "negative"
                else:
                    label = "FOLLOW_UP"
                    reason_type = None
            
            reason = self._generate_precise_reason(label, quality, topic, depth, key_point, hesitation)
            
            entry = {
                "question": current_question,
                "answer": answer,
                "label": label,
                "reason": reason,
                "context": {
                    "job_title": "Python后端工程师",
                    "topic": topic,
                    "domain": domain,
                    "follow_up_depth": depth,
                    "hesitation_score": hesitation,
                    "answer_length": len(answer),
                    "skill_level": level_name
                }
            }
            
            if label == "NEXT_TOPIC":
                entry["reason_type"] = reason_type
            
            chain.append(entry)
            
            # 下一轮问题
            if depth < num_rounds:
                if quality == "vague":
                    current_question = random.choice([
                        f"你提到了{random.choice(['基本功能', '使用过', '接触过'])}，能具体说说{random.choice(['做了什么', '怎么实现的', '有什么细节'])}吗？",
                        f"能详细讲讲你是怎么使用{topic}的吗？",
                        f"可以举个具体的例子说明一下吗？"
                    ])
                elif quality == "good":
                    current_question = random.choice([
                        f"你在使用{topic}时遇到过什么问题吗？怎么解决的？",
                        f"{topic}的{random.choice(['底层原理', '工作机制', '核心思想'])}你了解吗？",
                        f"能说说{topic}的{random.choice(['最佳实践', '注意事项', '常见坑点'])}吗？",
                        f"为什么选择{topic}而不是其他{random.choice(['方案', '技术', '工具'])}？"
                    ])
                else:  # excellent
                    current_question = random.choice([
                        f"如果在{random.choice(['高并发', '大数据量', '分布式', '高可用'])}场景下使用{topic}，你会怎么{random.choice(['优化', '设计', '改进', '处理'])}？",
                        f"{topic}有什么{random.choice(['局限性', '缺点', '不足', '边界']}吗？你会如何{random.choice(['改进', '规避', '解决', '优化'])}？",
                        f"能对比一下{topic}和{random.choice(['类似技术', '竞品方案', '其他选择'])}的{random.choice(['优劣', '区别', '适用场景'])}吗？",
                        f"在生产环境中使用{topic}，需要注意哪些{random.choice(['坑点', '问题', '风险', '细节'])}？"
                    ])
        
        return chain
    
    def generate_all_data(self, target_count: int = 1500) -> List[Dict]:
        """生成所有数据"""
        print(f"开始生成{target_count}条BERT训练数据...")
        print("使用改进的生成策略：更丰富的话题、更自然的回答、更精准的理由\n")
        
        conversation_count = 0
        while len(self.data) < target_count:
            domain, topic = random.choice(self.all_topics)
            chain = self._generate_conversation_chain(domain, topic)
            self.data.extend(chain)
            
            conversation_count += 1
            if conversation_count % 50 == 0:
                print(f"已生成{conversation_count}个对话链，共{len(self.data)}条数据...")
        
        # 截取
        self.data = self.data[:target_count]
        
        # 统计
        follow_up = sum(1 for d in self.data if d["label"] == "FOLLOW_UP")
        next_topic = sum(1 for d in self.data if d["label"] == "NEXT_TOPIC")
        positive = sum(1 for d in self.data if d.get("reason_type") == "positive")
        negative = sum(1 for d in self.data if d.get("reason_type") == "negative")
        
        print(f"\n[OK] 数据生成完成！")
        print(f"总计: {len(self.data)}条")
        print(f"FOLLOW_UP: {follow_up}条 ({follow_up/len(self.data)*100:.1f}%)")
        print(f"NEXT_TOPIC: {next_topic}条 ({next_topic/len(self.data)*100:.1f}%)")
        print(f"  - 积极换话题: {positive}条")
        print(f"  - 消极换话题: {negative}条")
        
        # 深度统计
        depth_stats = {}
        for d in self.data:
            depth = d["context"]["follow_up_depth"]
            depth_stats[depth] = depth_stats.get(depth, 0) + 1
        
        print(f"\n追问深度分布:")
        for depth in sorted(depth_stats.keys()):
            print(f"  第{depth}层: {depth_stats[depth]}条")
        
        # 技能统计
        skill_stats = {}
        for d in self.data:
            skill = d["context"]["skill_level"]
            skill_stats[skill] = skill_stats.get(skill, 0) + 1
        
        print(f"\n候选者技能水平分布:")
        for skill, count in sorted(skill_stats.items(), key=lambda x: x[1], reverse=True):
            print(f"  {skill}: {count}条 ({count/len(self.data)*100:.1f}%)")
        
        return self.data

def main():
    generator = AdvancedBERTDataGenerator()
    data = generator.generate_all_data(target_count=1500)
    
    # 保存
    output_file = "./data/bert_training_1500.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\n[SAVE] 数据已保存到: {output_file}")
    print(f"文件大小: {len(json.dumps(data, ensure_ascii=False)) / 1024 / 1024:.2f} MB")
    
    # 示例
    print(f"\n[SAMPLE] 数据示例:")
    for i in [0, 500, 1000]:
        if i < len(data):
            sample = data[i]
            print(f"\n--- 样本 {i+1} ({sample['context']['skill_level']}) ---")
            print(f"话题: {sample['context']['topic']}")
            print(f"问题: {sample['question'][:50]}...")
            print(f"回答: {sample['answer'][:80]}...")
            print(f"深度: {sample['context']['follow_up_depth']} | 犹豫: {sample['context']['hesitation_score']} | 决策: {sample['label']}")
            print(f"理由: {sample['reason'][:60]}...")

if __name__ == "__main__":
    main()

