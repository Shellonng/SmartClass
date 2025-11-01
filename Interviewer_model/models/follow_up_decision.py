"""
Follow-up Decision Model
追问决策模型 - 基于BERT的分类器
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, Any, Tuple
from pathlib import Path

from utils.logger import setup_logger

logger = setup_logger(__name__)


class FollowUpDecisionModel:
    """
    追问决策模型：判断是否需要追问（BERT三分类）
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化追问决策模型
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.model_config = config['models']['follow_up_classifier']
        
        model_name = self.model_config['name']
        checkpoint_path = self.model_config.get('checkpoint')
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 尝试加载微调后的模型，如果不存在则使用基础模型
        if checkpoint_path and Path(checkpoint_path).exists():
            logger.info(f"加载微调的追问决策模型: {checkpoint_path}")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                checkpoint_path,
                num_labels=self.model_config.get('num_labels', 3)
            )
        else:
            logger.warning(f"微调模型不存在，使用基础模型: {model_name}")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=self.model_config.get('num_labels', 3)
            )
        
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # 标签映射
        self.labels = ["FOLLOW_UP", "NEXT_TOPIC", "END"]
        
        logger.info("追问决策模型初始化完成")
    
    def decide(
        self,
        question: str,
        answer: str,
        speech_analysis: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Tuple[str, float]:
        """
        决策是否追问
        
        Args:
            question: 当前问题
            answer: 候选人回答
            speech_analysis: 语音分析结果
            context: 对话上下文
            
        Returns:
            (决策结果, 置信度)
        """
        # 如果已经达到最大追问深度，直接换话题
        if context['follow_up_depth'] >= context.get('max_follow_up', 3):
            logger.info("达到最大追问深度，换话题")
            return "NEXT_TOPIC", 1.0
        
        # 如果模型未微调，使用规则决策
        if not Path(self.model_config.get('checkpoint', '')).exists():
            return self._rule_based_decision(answer, speech_analysis, context)
        
        # 使用模型决策
        return self._model_based_decision(question, answer, speech_analysis, context)
    
    def _model_based_decision(
        self,
        question: str,
        answer: str,
        speech_analysis: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Tuple[str, float]:
        """
        基于模型的决策
        
        Args:
            question: 问题
            answer: 回答
            speech_analysis: 语音分析
            context: 上下文
            
        Returns:
            (决策, 置信度)
        """
        # 构建输入文本
        input_text = self._build_input_text(question, answer, speech_analysis, context)
        
        # 编码
        inputs = self.tokenizer(
            input_text,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)
        
        # 推理
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            
            predicted_idx = probs.argmax().item()
            confidence = probs.max().item()
        
        decision = self.labels[predicted_idx]
        
        logger.info(f"模型决策: {decision} (置信度={confidence:.2f})")
        
        return decision, confidence
    
    def _rule_based_decision(
        self,
        answer: str,
        speech_analysis: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Tuple[str, float]:
        """
        基于规则的决策（模型未微调时使用）
        
        Args:
            answer: 回答
            speech_analysis: 语音分析
            context: 上下文
            
        Returns:
            (决策, 置信度)
        """
        score = 0.0
        
        # 规则1: 回答太短
        if len(answer) < 30:
            score += 0.4
        
        # 规则2: 犹豫程度高
        if speech_analysis.get('hesitation_score', 0) > 0.6:
            score += 0.3
        
        # 规则3: 包含消极词汇
        negative_words = ["不知道", "不清楚", "不太了解", "没听说过", "不会"]
        if any(word in answer for word in negative_words):
            score += 0.3
        
        # 规则4: 填充词过多
        if speech_analysis.get('filler_count', 0) > 5:
            score += 0.2
        
        # 决策
        if score > 0.6:
            return "FOLLOW_UP", min(score, 1.0)
        elif context['follow_up_depth'] > 0:
            # 如果已经追问过，更倾向于换话题
            return "NEXT_TOPIC", 0.7
        else:
            # 第一次提问且回答一般，随机决定
            if score > 0.3:
                return "FOLLOW_UP", 0.6
            else:
                return "NEXT_TOPIC", 0.7
    
    def _build_input_text(
        self,
        question: str,
        answer: str,
        speech_analysis: Dict[str, Any],
        context: Dict[str, Any]
    ) -> str:
        """
        构建模型输入文本
        
        Args:
            question: 问题
            answer: 回答
            speech_analysis: 语音分析
            context: 上下文
            
        Returns:
            输入文本
        """
        input_text = f"""
[问题] {question}
[回答] {answer}
[填充词数量] {speech_analysis.get('filler_count', 0)}
[犹豫程度] {speech_analysis.get('hesitation_score', 0):.2f}
[回答长度] {len(answer)}
[追问深度] {context['follow_up_depth']}/{context.get('max_follow_up', 3)}
"""
        return input_text.strip()
    
    def get_explanation(
        self,
        decision: str,
        answer: str,
        speech_analysis: Dict[str, Any]
    ) -> str:
        """
        生成决策解释（用于调试和展示）
        
        Args:
            decision: 决策结果
            answer: 回答
            speech_analysis: 语音分析
            
        Returns:
            解释文本
        """
        if decision == "FOLLOW_UP":
            reasons = []
            
            if len(answer) < 30:
                reasons.append("回答过于简短")
            
            if speech_analysis.get('hesitation_score', 0) > 0.6:
                reasons.append(f"犹豫程度较高({speech_analysis['hesitation_score']:.2f})")
            
            if speech_analysis.get('filler_count', 0) > 3:
                reasons.append(f"使用{speech_analysis['filler_count']}个填充词")
            
            negative_words = ["不知道", "不清楚", "不太了解"]
            found_negative = [w for w in negative_words if w in answer]
            if found_negative:
                reasons.append(f"包含消极表达: {', '.join(found_negative)}")
            
            if reasons:
                return "需要追问，原因：" + "；".join(reasons)
            else:
                return "需要追问，以深入了解候选人理解"
        
        elif decision == "NEXT_TOPIC":
            return "回答充分，可以进入下一个话题"
        
        else:
            return "面试时间充足，建议结束本话题"

