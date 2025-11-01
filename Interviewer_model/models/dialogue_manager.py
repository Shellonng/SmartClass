"""
Dialogue Manager
对话管理器 - 管理面试流程和状态
"""

from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import random

from utils.logger import setup_logger

logger = setup_logger(__name__)


class InterviewState(Enum):
    """面试状态枚举"""
    INIT = "init"                      # 初始化
    GREETING = "greeting"              # 开场
    RESUME_QUESTIONING = "resume_q"    # 简历提问
    SKILL_ASSESSMENT = "skill_assess"  # 技能考察
    FOLLOW_UP = "follow_up"            # 追问
    SCENARIO_QUESTIONING = "scenario"  # 场景题
    CLOSING = "closing"                # 结束
    FINISHED = "finished"              # 完成


class DialogueManager:
    """
    对话管理器：控制面试流程和状态转换
    """
    
    def __init__(self, config: Dict[str, Any], job_title: str, resume_data: Dict[str, Any]):
        """
        初始化对话管理器
        
        Args:
            config: 配置字典
            job_title: 目标岗位
            resume_data: 简历数据
        """
        self.config = config
        self.interview_config = config.get('interview', {})
        self.job_title = job_title
        self.resume_data = resume_data
        
        # 面试状态
        self.state = InterviewState.INIT
        self.current_topic: Optional[str] = None
        self.current_question: Optional[str] = None
        self.follow_up_depth = 0
        
        # 历史记录
        self.conversation_history: List[Dict[str, str]] = []
        self.asked_questions: List[str] = []
        self.topics_covered: List[str] = []
        
        # 候选人表现
        self.candidate_performance: Dict[str, Any] = {
            'skill_scores': {},        # 各技能得分
            'total_questions': 0,
            'answered_well': 0,
            'hesitated_count': 0,
            'follow_up_triggered': 0
        }
        
        # 面试计划
        self.interview_plan = self._create_interview_plan()
        
        logger.info(f"对话管理器初始化: 岗位={job_title}, 计划{len(self.interview_plan)}个话题")
    
    def _create_interview_plan(self) -> List[Dict[str, Any]]:
        """
        创建面试计划
        
        Returns:
            面试计划列表
        """
        plan = []
        
        # 1. 开场
        plan.append({
            'type': 'greeting',
            'topic': '自我介绍',
            'state': InterviewState.GREETING
        })
        
        # 2. 简历相关问题（根据简历技能）
        resume_skills = self.resume_data.get('skills', [])
        for skill in resume_skills[:5]:  # 最多5个技能
            plan.append({
                'type': 'resume_question',
                'topic': skill,
                'state': InterviewState.RESUME_QUESTIONING,
                'skill': skill
            })
        
        # 3. 岗位核心技能（从岗位需求中来）
        # 这部分在运行时由RAG动态生成
        plan.append({
            'type': 'job_core',
            'topic': f'{self.job_title}核心能力',
            'state': InterviewState.SKILL_ASSESSMENT
        })
        
        # 4. 场景题
        plan.append({
            'type': 'scenario',
            'topic': '实际场景应用',
            'state': InterviewState.SCENARIO_QUESTIONING
        })
        
        # 5. 结束
        plan.append({
            'type': 'closing',
            'topic': '面试总结',
            'state': InterviewState.CLOSING
        })
        
        return plan
    
    def get_current_state(self) -> InterviewState:
        """获取当前面试状态"""
        return self.state
    
    def next_action(
        self,
        user_answer: str,
        speech_analysis: Dict[str, Any],
        follow_up_decision: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        决定下一步行动
        
        Args:
            user_answer: 用户回答
            speech_analysis: 语音分析结果
            follow_up_decision: 追问决策结果 (FOLLOW_UP/NEXT_TOPIC/END)
            
        Returns:
            (行动类型, 上下文信息)
        """
        # 记录回答
        self._record_answer(user_answer, speech_analysis)
        
        # 根据状态和决策判断下一步
        if self.state == InterviewState.FOLLOW_UP:
            # 当前在追问状态
            if follow_up_decision == "FOLLOW_UP" and self.follow_up_depth < self._get_max_follow_up():
                # 继续追问
                self.follow_up_depth += 1
                self.candidate_performance['follow_up_triggered'] += 1
                return "CONTINUE_FOLLOW_UP", self._build_context()
            else:
                # 停止追问，换话题
                self.follow_up_depth = 0
                return self._next_topic()
        
        else:
            # 不在追问状态，判断是否需要追问
            if follow_up_decision == "FOLLOW_UP":
                # 进入追问状态
                self.state = InterviewState.FOLLOW_UP
                self.follow_up_depth = 1
                self.candidate_performance['follow_up_triggered'] += 1
                return "START_FOLLOW_UP", self._build_context()
            else:
                # 直接换话题
                return self._next_topic()
    
    def _next_topic(self) -> Tuple[str, Dict[str, Any]]:
        """
        切换到下一个话题
        
        Returns:
            (行动类型, 上下文信息)
        """
        # 标记当前话题已完成
        if self.current_topic:
            self.topics_covered.append(self.current_topic)
        
        # 检查是否还有未完成的话题
        uncovered_plans = [
            p for p in self.interview_plan
            if p['topic'] not in self.topics_covered
        ]
        
        if not uncovered_plans:
            # 所有话题完成，结束面试
            self.state = InterviewState.FINISHED
            return "END_INTERVIEW", self._build_context()
        
        # 获取下一个话题
        next_plan = uncovered_plans[0]
        self.state = next_plan['state']
        self.current_topic = next_plan['topic']
        
        logger.info(f"切换话题: {self.current_topic} (状态={self.state.value})")
        
        return "NEW_TOPIC", self._build_context()
    
    def _record_answer(self, answer: str, speech_analysis: Dict[str, Any]):
        """
        记录用户回答和表现
        
        Args:
            answer: 回答文本
            speech_analysis: 语音分析
        """
        self.conversation_history.append({
            'question': self.current_question or "",
            'answer': answer,
            'hesitation_score': speech_analysis.get('hesitation_score', 0),
            'filler_count': speech_analysis.get('filler_count', 0)
        })
        
        self.candidate_performance['total_questions'] += 1
        
        # 简单评估（实际会用评估模型）
        if speech_analysis.get('hesitation_score', 0) < 0.4:
            self.candidate_performance['answered_well'] += 1
        
        if speech_analysis.get('hesitation_score', 0) > 0.6:
            self.candidate_performance['hesitated_count'] += 1
    
    def set_current_question(self, question: str):
        """
        设置当前问题
        
        Args:
            question: 问题文本
        """
        self.current_question = question
        self.asked_questions.append(question)
    
    def _build_context(self) -> Dict[str, Any]:
        """
        构建当前上下文信息（供LLM使用）
        
        Returns:
            上下文字典
        """
        context = {
            'job_title': self.job_title,
            'current_state': self.state.value,
            'current_topic': self.current_topic,
            'current_question': self.current_question,
            'follow_up_depth': self.follow_up_depth,
            'max_follow_up': self._get_max_follow_up(),
            'resume_skills': self.resume_data.get('skills', []),
            'topics_covered': self.topics_covered,
            'conversation_history': self.conversation_history[-3:],  # 最近3轮
            'performance': self.candidate_performance
        }
        return context
    
    def _get_max_follow_up(self) -> int:
        """获取最大追问深度"""
        return self.interview_config.get('max_follow_up_depth', 3)
    
    def should_end_interview(self) -> bool:
        """
        判断是否应该结束面试
        
        Returns:
            是否结束
        """
        # 检查状态
        if self.state == InterviewState.FINISHED:
            return True
        
        # 检查时长（如果有记录）
        # TODO: 实现时长检查
        
        # 检查问题数量
        max_questions = self.interview_config.get('max_questions', 20)
        if self.candidate_performance['total_questions'] >= max_questions:
            logger.info(f"达到最大问题数: {max_questions}")
            return True
        
        return False
    
    def get_summary(self) -> Dict[str, Any]:
        """
        获取面试总结
        
        Returns:
            总结信息
        """
        total = self.candidate_performance['total_questions']
        well_ratio = (
            self.candidate_performance['answered_well'] / total
            if total > 0 else 0
        )
        
        summary = {
            'job_title': self.job_title,
            'total_questions': total,
            'topics_covered': len(self.topics_covered),
            'follow_up_triggered': self.candidate_performance['follow_up_triggered'],
            'well_answered_ratio': round(well_ratio * 100, 1),
            'hesitation_rate': round(
                self.candidate_performance['hesitated_count'] / total * 100
                if total > 0 else 0, 1
            ),
            'conversation_history': self.conversation_history
        }
        
        return summary
    
    def reset(self):
        """重置对话管理器"""
        self.state = InterviewState.INIT
        self.current_topic = None
        self.current_question = None
        self.follow_up_depth = 0
        self.conversation_history = []
        self.asked_questions = []
        self.topics_covered = []
        self.candidate_performance = {
            'skill_scores': {},
            'total_questions': 0,
            'answered_well': 0,
            'hesitated_count': 0,
            'follow_up_triggered': 0
        }
        logger.info("对话管理器已重置")

