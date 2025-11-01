"""
Answer Evaluator
回答评估模块 - 多任务评估（分类+回归）
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import Dict, Any, Tuple, List
from pathlib import Path

from utils.logger import setup_logger

logger = setup_logger(__name__)


class MultiTaskRoBERTa(nn.Module):
    """多任务RoBERTa模型（分类+回归）"""
    def __init__(self, model_name, num_labels=4):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(model_name)
        hidden_size = self.roberta.config.hidden_size
        
        # 分类头（当前回答质量）
        self.classification_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_labels)
        )
        
        # 回归头（整体能力评分）
        self.regression_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()  # 输出0-1范围
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        
        classification_logits = self.classification_head(pooled_output)
        regression_score = self.regression_head(pooled_output).squeeze(-1)
        
        return classification_logits, regression_score


class AnswerEvaluator:
    """
    回答评估器：评估候选人回答的质量和准确性
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化评估器（多任务模型）
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.eval_config = config['models']['evaluator']
        
        model_name = self.eval_config['name']
        checkpoint_path = self.eval_config.get('checkpoint')
        num_labels = self.eval_config.get('num_labels', 4)
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载模型
        if checkpoint_path and Path(checkpoint_path).exists():
            checkpoint_file = Path(checkpoint_path) / "pytorch_model.bin"
            if checkpoint_file.exists():
                logger.info(f"加载微调的多任务评估模型: {checkpoint_path}")
                # 创建模型架构
                self.model = MultiTaskRoBERTa(model_name, num_labels=num_labels)
                # 加载训练好的权重
                state_dict = torch.load(checkpoint_file, map_location='cpu', weights_only=False)
                self.model.load_state_dict(state_dict)
                self.model.to(self.device)
                self.model.eval()
                self.use_multitask = True
                logger.info("多任务模型加载成功")
            else:
                logger.warning(f"多任务模型文件不存在: {checkpoint_file}")
                logger.warning("使用规则评估模式")
                self.model = None
                self.use_multitask = False
        else:
            logger.warning(f"checkpoint路径不存在: {checkpoint_path}")
            logger.warning("使用规则评估模式")
            self.model = None
            self.use_multitask = False
        
        # 评分映射和标签
        self.score_mapping = self.eval_config.get('score_mapping', [50, 70, 85, 95])
        self.label_names = self.eval_config.get('label_names', ["差", "一般", "良好", "优秀"])
        
        logger.info(f"回答评估器初始化完成 (多任务模式: {self.use_multitask})")
    
    def evaluate(
        self,
        question: str,
        answer: str,
        speech_analysis: Dict[str, Any],
        reference_answer: str = "",
        history_qa: List[Dict] = None
    ) -> Dict[str, Any]:
        """
        评估回答质量（多任务：当前质量分类 + 整体能力评分）
        
        Args:
            question: 当前问题
            answer: 候选人回答
            speech_analysis: 语音分析结果
            reference_answer: 参考答案（可选）
            history_qa: 历史问答记录（可选）
            
        Returns:
            评估结果字典
        """
        if history_qa is None:
            history_qa = []
        
        # 如果使用多任务模型
        if self.use_multitask and self.model is not None:
            return self._multitask_evaluation(
                question, answer, speech_analysis, history_qa
            )
        
        # 否则使用规则评估
        return self._rule_based_evaluation(
            question, answer, speech_analysis, reference_answer
        )
    
    def _multitask_evaluation(
        self,
        question: str,
        answer: str,
        speech_analysis: Dict[str, Any],
        history_qa: List[Dict]
    ) -> Dict[str, Any]:
        """
        多任务模型评估（分类+回归）
        
        Args:
            question: 当前问题
            answer: 当前回答
            speech_analysis: 语音分析
            history_qa: 历史问答
            
        Returns:
            评估结果
        """
        # 构建输入（包含历史）
        input_parts = []
        
        # 历史问答
        if history_qa:
            input_parts.append("[历史问答]")
            for i, qa in enumerate(history_qa[-3:], 1):  # 最多取最近3轮
                input_parts.append(f"Q{i}: {qa.get('question', '')}")
                input_parts.append(f"A{i}: {qa.get('answer', '')[:100]}")
                input_parts.append(f"质量: {qa.get('quality', '未知')}")
        
        # 当前问答
        input_parts.append("[当前问答]")
        input_parts.append(f"问题: {question}")
        input_parts.append(f"回答: {answer}")
        input_parts.append(f"流畅度: {1 - speech_analysis.get('hesitation_score', 0):.2f}")
        
        input_text = "\n".join(input_parts)
        
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
            cls_logits, reg_score = self.model(
                inputs['input_ids'],
                inputs['attention_mask']
            )
            
            # 分类结果
            cls_probs = torch.softmax(cls_logits, dim=-1)
            predicted_idx = cls_probs.argmax().item()
            confidence = cls_probs.max().item()
            
            # 回归结果（整体评分）
            overall_score = reg_score.item() * 100  # 反归一化到0-100
        
        # 映射到分数和标签
        current_label = self.label_names[predicted_idx]
        current_score = self.score_mapping[predicted_idx]
        
        # 生成反馈
        feedback = self._generate_multitask_feedback(
            current_label, overall_score, answer, speech_analysis
        )
        
        result = {
            'current_score': current_score,
            'current_label': current_label,
            'overall_score': round(overall_score, 1),
            'confidence': round(confidence, 2),
            'feedback': feedback,
            'distribution': {
                self.label_names[i]: round(float(cls_probs[0][i]), 2)
                for i in range(len(self.label_names))
            }
        }
        
        logger.info(
            f"多任务评估: 当前={current_label}({current_score}), "
            f"整体={overall_score:.1f}, 置信度={confidence:.2f}"
        )
        
        return result
    
    def _model_based_evaluation(
        self,
        question: str,
        answer: str,
        speech_analysis: Dict[str, Any],
        reference_answer: str
    ) -> Dict[str, Any]:
        """
        基于模型的评估
        
        Args:
            question: 问题
            answer: 回答
            speech_analysis: 语音分析
            reference_answer: 参考答案
            
        Returns:
            评估结果
        """
        # 构建输入
        input_text = self._build_input_for_evaluation(
            question, answer, speech_analysis, reference_answer
        )
        
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
        
        # 映射到分数
        score = self.score_mapping[predicted_idx]
        label = self.label_names[predicted_idx]
        
        # 生成反馈
        feedback = self._generate_feedback(
            label, answer, speech_analysis, probs.cpu().numpy()[0]
        )
        
        result = {
            'score': score,
            'label': label,
            'confidence': round(confidence, 2),
            'feedback': feedback,
            'distribution': {
                self.label_names[i]: round(float(probs[0][i]), 2)
                for i in range(len(self.label_names))
            }
        }
        
        logger.info(f"评估完成: {label} ({score}分), 置信度={confidence:.2f}")
        
        return result
    
    def _rule_based_evaluation(
        self,
        question: str,
        answer: str,
        speech_analysis: Dict[str, Any],
        reference_answer: str
    ) -> Dict[str, Any]:
        """
        基于规则的评估
        
        Args:
            question: 问题
            answer: 回答
            speech_analysis: 语音分析
            reference_answer: 参考答案
            
        Returns:
            评估结果
        """
        score = 60  # 基础分
        
        # 评估维度
        dimensions = {
            'length': 0,      # 长度
            'hesitation': 0,  # 流畅度
            'content': 0,     # 内容
            'structure': 0    # 结构
        }
        
        # 1. 长度评估（20分）
        answer_len = len(answer)
        if answer_len < 20:
            dimensions['length'] = -15
        elif answer_len < 50:
            dimensions['length'] = -5
        elif answer_len < 100:
            dimensions['length'] = 5
        else:
            dimensions['length'] = 10
        
        # 2. 流畅度评估（20分）
        hesitation_score = speech_analysis.get('hesitation_score', 0)
        if hesitation_score < 0.3:
            dimensions['hesitation'] = 10
        elif hesitation_score < 0.5:
            dimensions['hesitation'] = 5
        elif hesitation_score < 0.7:
            dimensions['hesitation'] = -5
        else:
            dimensions['hesitation'] = -15
        
        # 3. 内容评估（30分）
        # 检查是否包含技术词汇
        tech_words = self._count_technical_words(answer)
        if tech_words >= 5:
            dimensions['content'] = 15
        elif tech_words >= 3:
            dimensions['content'] = 10
        elif tech_words >= 1:
            dimensions['content'] = 5
        else:
            dimensions['content'] = -10
        
        # 检查消极表达
        negative_phrases = ["不知道", "不清楚", "不太了解", "没听说过", "不会"]
        if any(phrase in answer for phrase in negative_phrases):
            dimensions['content'] -= 10
        
        # 4. 结构评估（10分）
        # 检查是否有逻辑词
        logic_words = ["首先", "其次", "然后", "最后", "因为", "所以", "但是", "而且"]
        if any(word in answer for word in logic_words):
            dimensions['structure'] = 5
        
        # 计算总分
        total_score = score + sum(dimensions.values())
        total_score = max(0, min(100, total_score))  # 限制在0-100
        
        # 映射到等级
        if total_score >= 85:
            label = "优秀"
        elif total_score >= 70:
            label = "好"
        elif total_score >= 50:
            label = "一般"
        else:
            label = "差"
        
        # 生成反馈
        feedback = self._generate_detailed_feedback(dimensions, answer, speech_analysis)
        
        result = {
            'score': int(total_score),
            'label': label,
            'confidence': 0.7,
            'feedback': feedback,
            'dimensions': dimensions
        }
        
        logger.info(f"规则评估完成: {label} ({total_score:.0f}分)")
        
        return result
    
    def _build_input_for_evaluation(
        self,
        question: str,
        answer: str,
        speech_analysis: Dict[str, Any],
        reference_answer: str
    ) -> str:
        """
        构建评估输入文本
        
        Args:
            question: 问题
            answer: 回答
            speech_analysis: 语音分析
            reference_answer: 参考答案
            
        Returns:
            输入文本
        """
        input_parts = [
            f"[问题] {question}",
            f"[回答] {answer}",
            f"[流畅度] {1 - speech_analysis.get('hesitation_score', 0):.2f}",
            f"[回答长度] {len(answer)}"
        ]
        
        if reference_answer:
            input_parts.append(f"[参考答案] {reference_answer}")
        
        return "\n".join(input_parts)
    
    def _count_technical_words(self, text: str) -> int:
        """统计技术词汇数量"""
        tech_keywords = [
            'API', '接口', '算法', '数据结构', '框架', '库', '模块',
            '类', '对象', '函数', '方法', '变量', '异步', '同步',
            '并发', '线程', '进程', '缓存', '数据库', '查询',
            '优化', '性能', '架构', '设计模式', '测试', '部署'
        ]
        
        count = 0
        text_lower = text.lower()
        for keyword in tech_keywords:
            if keyword.lower() in text_lower:
                count += 1
        
        return count
    
    def _generate_feedback(
        self,
        label: str,
        answer: str,
        speech_analysis: Dict[str, Any],
        probs: Any
    ) -> str:
        """生成评估反馈"""
        feedback_parts = []
        
        if label == "优秀":
            feedback_parts.append("回答非常好！")
        elif label == "好":
            feedback_parts.append("回答不错。")
        elif label == "一般":
            feedback_parts.append("回答基本到位，但还可以更深入。")
        else:
            feedback_parts.append("回答需要改进。")
        
        # 流畅度反馈
        hesitation = speech_analysis.get('hesitation_score', 0)
        if hesitation > 0.6:
            feedback_parts.append("表达时有些犹豫，建议更加自信。")
        elif hesitation < 0.3:
            feedback_parts.append("表达流畅，很好！")
        
        # 内容反馈
        if len(answer) < 30:
            feedback_parts.append("回答过于简短，可以展开说明。")
        
        return " ".join(feedback_parts)
    
    def _generate_detailed_feedback(
        self,
        dimensions: Dict[str, float],
        answer: str,
        speech_analysis: Dict[str, Any]
    ) -> str:
        """生成详细反馈"""
        feedback_parts = []
        
        # 长度反馈
        if dimensions['length'] < 0:
            feedback_parts.append("回答偏短，建议提供更多细节和示例。")
        elif dimensions['length'] > 5:
            feedback_parts.append("回答长度适中。")
        
        # 流畅度反馈
        if dimensions['hesitation'] < 0:
            feedback_parts.append(
                f"表达有些不流畅（填充词{speech_analysis.get('filler_count', 0)}个），"
                "建议思考清楚后再回答。"
            )
        elif dimensions['hesitation'] > 5:
            feedback_parts.append("表达流畅自信。")
        
        # 内容反馈
        if dimensions['content'] < 0:
            feedback_parts.append("回答缺少技术深度，建议结合具体原理或案例说明。")
        elif dimensions['content'] > 10:
            feedback_parts.append("回答有技术深度。")
        
        # 结构反馈
        if dimensions['structure'] > 0:
            feedback_parts.append("回答有条理。")
        
        return " ".join(feedback_parts) if feedback_parts else "继续加油！"
    
    def _generate_multitask_feedback(
        self,
        current_label: str,
        overall_score: float,
        answer: str,
        speech_analysis: Dict[str, Any]
    ) -> str:
        """生成多任务评估反馈"""
        feedback_parts = []
        
        # 当前回答反馈
        if current_label == "优秀":
            feedback_parts.append("本次回答非常好！")
        elif current_label == "良好":
            feedback_parts.append("本次回答不错。")
        elif current_label == "一般":
            feedback_parts.append("本次回答还可以，但可以更深入。")
        else:
            feedback_parts.append("本次回答需要改进。")
        
        # 整体表现反馈
        if overall_score >= 85:
            feedback_parts.append(f"综合表现优秀（{overall_score:.0f}分）。")
        elif overall_score >= 70:
            feedback_parts.append(f"综合表现良好（{overall_score:.0f}分）。")
        elif overall_score >= 50:
            feedback_parts.append(f"综合表现一般（{overall_score:.0f}分），继续努力。")
        else:
            feedback_parts.append(f"综合表现需要提升（{overall_score:.0f}分）。")
        
        # 流畅度反馈
        hesitation = speech_analysis.get('hesitation_score', 0)
        if hesitation > 0.6:
            feedback_parts.append("表达时有些犹豫，建议更加自信。")
        elif hesitation < 0.3:
            feedback_parts.append("表达流畅自然。")
        
        return " ".join(feedback_parts)
        
        logger.info(f"规则评估完成: {label} ({total_score:.0f}分)")
        
        return result
    
    def _build_input_for_evaluation(
        self,
        question: str,
        answer: str,
        speech_analysis: Dict[str, Any],
        reference_answer: str
    ) -> str:
        """
        构建评估输入文本
        
        Args:
            question: 问题
            answer: 回答
            speech_analysis: 语音分析
            reference_answer: 参考答案
            
        Returns:
            输入文本
        """
        input_parts = [
            f"[问题] {question}",
            f"[回答] {answer}",
            f"[流畅度] {1 - speech_analysis.get('hesitation_score', 0):.2f}",
            f"[回答长度] {len(answer)}"
        ]
        
        if reference_answer:
            input_parts.append(f"[参考答案] {reference_answer}")
        
        return "\n".join(input_parts)
    
    def _count_technical_words(self, text: str) -> int:
        """统计技术词汇数量"""
        tech_keywords = [
            'API', '接口', '算法', '数据结构', '框架', '库', '模块',
            '类', '对象', '函数', '方法', '变量', '异步', '同步',
            '并发', '线程', '进程', '缓存', '数据库', '查询',
            '优化', '性能', '架构', '设计模式', '测试', '部署'
        ]
        
        count = 0
        text_lower = text.lower()
        for keyword in tech_keywords:
            if keyword.lower() in text_lower:
                count += 1
        
        return count
    
    def _generate_feedback(
        self,
        label: str,
        answer: str,
        speech_analysis: Dict[str, Any],
        probs: Any
    ) -> str:
        """生成评估反馈"""
        feedback_parts = []
        
        if label == "优秀":
            feedback_parts.append("回答非常好！")
        elif label == "好":
            feedback_parts.append("回答不错。")
        elif label == "一般":
            feedback_parts.append("回答基本到位，但还可以更深入。")
        else:
            feedback_parts.append("回答需要改进。")
        
        # 流畅度反馈
        hesitation = speech_analysis.get('hesitation_score', 0)
        if hesitation > 0.6:
            feedback_parts.append("表达时有些犹豫，建议更加自信。")
        elif hesitation < 0.3:
            feedback_parts.append("表达流畅，很好！")
        
        # 内容反馈
        if len(answer) < 30:
            feedback_parts.append("回答过于简短，可以展开说明。")
        
        return " ".join(feedback_parts)
    
    def _generate_detailed_feedback(
        self,
        dimensions: Dict[str, float],
        answer: str,
        speech_analysis: Dict[str, Any]
    ) -> str:
        """生成详细反馈"""
        feedback_parts = []
        
        # 长度反馈
        if dimensions['length'] < 0:
            feedback_parts.append("回答偏短，建议提供更多细节和示例。")
        elif dimensions['length'] > 5:
            feedback_parts.append("回答长度适中。")
        
        # 流畅度反馈
        if dimensions['hesitation'] < 0:
            feedback_parts.append(
                f"表达有些不流畅（填充词{speech_analysis.get('filler_count', 0)}个），"
                "建议思考清楚后再回答。"
            )
        elif dimensions['hesitation'] > 5:
            feedback_parts.append("表达流畅自信。")
        
        # 内容反馈
        if dimensions['content'] < 0:
            feedback_parts.append("回答缺少技术深度，建议结合具体原理或案例说明。")
        elif dimensions['content'] > 10:
            feedback_parts.append("回答有技术深度。")
        
        # 结构反馈
        if dimensions['structure'] > 0:
            feedback_parts.append("回答有条理。")
        
        return " ".join(feedback_parts) if feedback_parts else "继续加油！"
    
    def _generate_multitask_feedback(
        self,
        current_label: str,
        overall_score: float,
        answer: str,
        speech_analysis: Dict[str, Any]
    ) -> str:
        """生成多任务评估反馈"""
        feedback_parts = []
        
        # 当前回答反馈
        if current_label == "优秀":
            feedback_parts.append("本次回答非常好！")
        elif current_label == "良好":
            feedback_parts.append("本次回答不错。")
        elif current_label == "一般":
            feedback_parts.append("本次回答还可以，但可以更深入。")
        else:
            feedback_parts.append("本次回答需要改进。")
        
        # 整体表现反馈
        if overall_score >= 85:
            feedback_parts.append(f"综合表现优秀（{overall_score:.0f}分）。")
        elif overall_score >= 70:
            feedback_parts.append(f"综合表现良好（{overall_score:.0f}分）。")
        elif overall_score >= 50:
            feedback_parts.append(f"综合表现一般（{overall_score:.0f}分），继续努力。")
        else:
            feedback_parts.append(f"综合表现需要提升（{overall_score:.0f}分）。")
        
        # 流畅度反馈
        hesitation = speech_analysis.get('hesitation_score', 0)
        if hesitation > 0.6:
            feedback_parts.append("表达时有些犹豫，建议更加自信。")
        elif hesitation < 0.3:
            feedback_parts.append("表达流畅自然。")
        
        return " ".join(feedback_parts)

