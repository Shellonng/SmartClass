"""
Topic提取工具：从问题中推断话题
"""
import re
from typing import Optional


class TopicExtractor:
    """从面试问题中提取话题"""
    
    # 技术关键词映射
    TECH_KEYWORDS = {
        # 缓存相关
        "Redis": "Redis",
        "缓存": "缓存",
        "cache": "缓存",
        
        # 容器相关
        "Kubernetes": "Kubernetes",
        "K8s": "Kubernetes",
        "Docker": "Docker",
        "容器": "容器",
        
        # 微服务相关
        "微服务": "微服务",
        "Spring Cloud": "微服务",
        
        # 消息队列
        "RabbitMQ": "消息队列",
        "Kafka": "消息队列",
        "消息队列": "消息队列",
        
        # 数据库
        "MySQL": "数据库",
        "PostgreSQL": "数据库",
        "MongoDB": "数据库",
        "数据库": "数据库",
        
        # 编程语言
        "Python": "Python",
        "Java": "Java",
        "Go": "Go",
        
        # 其他
        "异步": "异步编程",
        "asyncio": "异步编程",
        "分布式": "分布式系统",
        "性能优化": "性能优化",
        "架构设计": "架构设计",
    }
    
    # 场景关键词（配合技术词）
    SCENARIO_KEYWORDS = {
        "缓存策略": "缓存策略",
        "一致性": "一致性",
        "高并发": "高并发",
        "部署": "部署",
        "编排": "编排",
        "监控": "监控",
        "优化": "优化",
        "设计": "设计",
        "实现": "实现",
        "架构": "架构",
    }
    
    @classmethod
    def extract(cls, question: str) -> str:
        """
        从问题中提取话题
        
        Args:
            question: 面试问题
            
        Returns:
            话题名称（如"Redis缓存策略"、"Kubernetes部署"）
        """
        # 特殊处理：自我介绍
        if any(word in question for word in ["自我介绍", "介绍一下", "个人介绍"]):
            return "自我介绍"
        
        # 提取技术词
        tech = cls._extract_tech(question)
        
        # 提取场景词
        scenario = cls._extract_scenario(question)
        
        # 组合
        if tech and scenario:
            return f"{tech}{scenario}"
        elif tech:
            return tech
        elif scenario:
            return scenario
        else:
            return "其他技术话题"
    
    @classmethod
    def _extract_tech(cls, question: str) -> Optional[str]:
        """提取技术关键词"""
        for keyword, tech in cls.TECH_KEYWORDS.items():
            if keyword in question:
                return tech
        return None
    
    @classmethod
    def _extract_scenario(cls, question: str) -> Optional[str]:
        """提取场景关键词"""
        for keyword, scenario in cls.SCENARIO_KEYWORDS.items():
            if keyword in question:
                return scenario
        return None
    
    @classmethod
    def extract_batch(cls, questions: list[str]) -> list[str]:
        """批量提取话题"""
        return [cls.extract(q) for q in questions]


# ========== 应用示例 ==========

if __name__ == "__main__":
    extractor = TopicExtractor()
    
    test_questions = [
        "请先做个自我介绍",
        "我看到你在项目中使用了Redis，能详细说说缓存策略吗？",
        "在高并发场景下，如何保证Redis缓存的一致性？",
        "你们是如何使用Kubernetes进行服务部署的？",
        "在微服务架构中，服务间通信是如何实现的？",
        "说说你对Python异步编程的理解",
    ]
    
    print("话题提取测试：\n")
    for q in test_questions:
        topic = extractor.extract(q)
        print(f"问题: {q}")
        print(f"话题: {topic}")
        print("-" * 60)


