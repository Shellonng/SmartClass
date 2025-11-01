"""
Model Testing Script
模型测试脚本 - 测试各个模块是否正常工作
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config_loader import load_config
from models.resume_parser import ResumeParser
from models.simple_rag import SimpleRAG
from models.dialogue_manager import DialogueManager


def test_resume_parser():
    """测试简历解析"""
    print("\n[测试] 简历解析模块")
    print("-" * 40)
    
    parser = ResumeParser()
    
    # 测试文本提取（模拟）
    test_text = """
    张三
    电话：13800138000
    邮箱：zhangsan@example.com
    
    技能：
    Python, Django, MySQL, Redis, Docker
    
    工作经历：
    2020.01 - 至今  某科技公司  后端工程师
    """
    
    # 简单测试技能提取
    skills = parser._extract_skills(test_text)
    print(f"提取的技能: {skills}")
    
    print("✅ 简历解析模块测试通过")


def test_rag():
    """测试RAG检索"""
    print("\n[测试] RAG检索模块")
    print("-" * 40)
    
    config = load_config()
    rag = SimpleRAG(config)
    
    if len(rag.questions_db) == 0:
        print("⚠️  问题库为空，跳过测试")
        return
    
    # 测试检索
    results = rag.search(
        query="Python",
        job_title="后端工程师",
        top_k=3
    )
    
    print(f"检索到 {len(results)} 个相关问题:")
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['question'][:50]}...")
    
    # 测试统计
    stats = rag.get_statistics()
    print(f"\n问题库统计: {stats['total']}条问题")
    
    print("✅ RAG检索模块测试通过")


def test_dialogue_manager():
    """测试对话管理器"""
    print("\n[测试] 对话管理器")
    print("-" * 40)
    
    config = load_config()
    
    resume_data = {
        'skills': ['Python', 'Django', 'MySQL'],
        'name': '测试用户'
    }
    
    manager = DialogueManager(config, "后端工程师", resume_data)
    
    print(f"初始状态: {manager.get_current_state().value}")
    print(f"面试计划: {len(manager.interview_plan)}个话题")
    
    # 模拟一轮对话
    speech_analysis = {
        'hesitation_score': 0.3,
        'filler_count': 2
    }
    
    action, context = manager.next_action(
        user_answer="这是我的回答",
        speech_analysis=speech_analysis,
        follow_up_decision="NEXT_TOPIC"
    )
    
    print(f"下一步行动: {action}")
    print(f"当前话题: {context.get('current_topic')}")
    
    print("✅ 对话管理器测试通过")


def main():
    """主测试函数"""
    print("=" * 60)
    print("AI Interview Coach - 模块测试")
    print("=" * 60)
    
    try:
        test_resume_parser()
        test_rag()
        test_dialogue_manager()
        
        print("\n" + "=" * 60)
        print("✅ 所有测试通过！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

