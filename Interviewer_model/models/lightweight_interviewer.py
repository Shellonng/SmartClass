"""
Lightweight Interviewer
轻量级面试官LLM - 适配RTX 4060 + LoRA微调
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from typing import Dict, List, Any, Optional
from pathlib import Path

from utils.logger import setup_logger

logger = setup_logger(__name__)


class LightweightInterviewer:
    """
    轻量级面试官模型 - 使用Qwen-1.8B + 4bit量化
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化面试官模型（支持LoRA微调权重）
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.llm_config = config['models']['llm']
        
        model_name = self.llm_config['name']
        lora_checkpoint = self.llm_config.get('lora_checkpoint')
        
        logger.info(f"加载LLM基础模型: {model_name}")
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # 配置量化（8bit）
        use_8bit = self.llm_config.get('load_in_8bit', True)
        if use_8bit:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16
            )
            quantization_config = bnb_config
        else:
            quantization_config = None
        
        # 加载基础模型
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        
        # 加载LoRA权重
        if lora_checkpoint and Path(lora_checkpoint).exists():
            logger.info(f"加载LoRA微调权重: {lora_checkpoint}")
            self.model = PeftModel.from_pretrained(self.model, lora_checkpoint)
            logger.info("LoRA权重加载完成")
        else:
            logger.warning(f"LoRA权重不存在，使用基础模型: {lora_checkpoint}")
        
        self.model.eval()
        
        # 对话历史
        self.conversation_history: List[tuple] = []
        
        # 生成参数
        self.max_length = self.llm_config.get('max_length', 512)
        self.temperature = self.llm_config.get('temperature', 0.7)
        self.top_p = self.llm_config.get('top_p', 0.9)
        
        logger.info("面试官模型加载完成")
    
    def generate_greeting(self, context: Dict[str, Any]) -> str:
        """
        生成开场白
        
        Args:
            context: 上下文信息
            
        Returns:
            开场白文本
        """
        prompt = f"""
你是一位专业的{context['job_title']}面试官，现在要开始面试。

候选人的简历显示他/她具备以下技能：
{', '.join(context['resume_skills'])}

请用友好、专业的方式开场，包括：
1. 简单自我介绍
2. 说明面试流程
3. 让候选人放松

生成开场白（100字以内）：
"""
        response = self._generate(prompt, use_history=False)
        return response
    
    def generate_question(
        self,
        context: Dict[str, Any],
        rag_questions: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        生成面试问题
        
        Args:
            context: 上下文信息
            rag_questions: RAG检索到的相关问题
            
        Returns:
            问题文本
        """
        rag_context = ""
        if rag_questions:
            rag_context = "\n参考问题：\n" + "\n".join([
                f"- {q['question']}" for q in rag_questions[:3]
            ])
        
        # 检查是否是换话题（从上下文中获取）
        last_answer = context.get('last_answer', '')
        is_topic_change = any(word in last_answer for word in ['不会', '不了解', '不熟悉'])
        
        if is_topic_change:
            transition_phrase = "没关系，我们换个话题。"
        else:
            transition_phrase = ""
        
        prompt = f"""
你是{context['job_title']}面试官，正在进行技术面试。

【候选人简历技能】{', '.join(context['resume_skills'])}

【当前话题】{context['current_topic']}

{rag_context}

【任务】
{transition_phrase}针对"{context['current_topic']}"提出一个技术问题。

要求：
1. 如果有过渡语（换话题），先说过渡语，再问问题
2. 问题要与候选人简历技能相关
3. 自然、口语化，像真实面试官
4. 控制在50字以内

你的问题：
"""
        response = self._generate(prompt)
        return response.strip()
    
    def generate_follow_up(
        self,
        context: Dict[str, Any],
        user_answer: str,
        speech_analysis: Dict[str, Any]
    ) -> str:
        """
        生成追问
        
        Args:
            context: 上下文信息
            user_answer: 用户回答
            speech_analysis: 语音分析结果
            
        Returns:
            追问文本
        """
        # 构建语音分析描述
        hesitation_note = ""
        if speech_analysis and speech_analysis.get('hesitation_score', 0) > 0.6:
            hesitation_note = f"""
[注意] 候选人回答时有{speech_analysis.get('filler_count', 0)}个填充词，
显示出一定犹豫（犹豫度{speech_analysis['hesitation_score']:.2f}）
"""
        
        prompt = f"""
你是{context['job_title']}面试官，正在与候选人对话。

【刚才的问题】{context['current_question']}

【候选人的回答】
"{user_answer}"

{hesitation_note}

【任务】
仔细阅读候选人的回答，然后做出合适的反应：

情况1: 如果候选人明确表示"不会"、"不了解"、"不熟悉"
→ 立即换话题！说："没关系，我们聊聊其他的。" 然后问候选人简历上的其他技能

情况2: 如果候选人答非所问或答得很敷衍
→ 换个角度或换话题，不要纠缠

情况3: 如果候选人回答了但比较简略
→ 可以追问具体细节，比如"能举个例子吗？"

情况4: 如果候选人回答得很详细
→ 表扬并提出新问题

注意：
- 一定要根据候选人的实际回答内容做反应
- 如果候选人说不会，绝对不要继续问同一个话题
- 语气要自然、友好、鼓励

你的回应（40字以内）：
"""
        response = self._generate(prompt)
        return response.strip()
    
    def generate_response(
        self,
        action_type: str,
        context: Dict[str, Any],
        user_answer: str = "",
        speech_analysis: Optional[Dict[str, Any]] = None,
        rag_questions: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        根据行动类型生成回应
        
        Args:
            action_type: 行动类型
            context: 上下文
            user_answer: 用户回答
            speech_analysis: 语音分析
            rag_questions: RAG检索结果
            
        Returns:
            生成的回应
        """
        if action_type == "greeting":
            return self.generate_greeting(context)
        
        elif action_type == "NEW_TOPIC":
            return self.generate_question(context, rag_questions)
        
        elif action_type in ["START_FOLLOW_UP", "CONTINUE_FOLLOW_UP"]:
            return self.generate_follow_up(context, user_answer, speech_analysis or {})
        
        elif action_type == "END_INTERVIEW":
            return self._generate_closing(context)
        
        else:
            logger.warning(f"未知的行动类型: {action_type}")
            return "请继续。"
    
    def _generate_closing(self, context: Dict[str, Any]) -> str:
        """
        生成结束语
        
        Args:
            context: 上下文
            
        Returns:
            结束语
        """
        prompt = f"""
你是面试官，面试即将结束。

【面试情况】
- 岗位：{context['job_title']}
- 提问数量：{context['performance']['total_questions']}
- 表现良好的回答：{context['performance']['answered_well']}

【任务】
用专业、友好的方式结束面试：
1. 感谢候选人的参与
2. 简单总结表现（保持积极）
3. 告知后续流程

生成结束语（80字以内）：
"""
        response = self._generate(prompt, use_history=False)
        return response
    
    def _generate(self, prompt: str, use_history: bool = True) -> str:
        """
        调用LLM生成文本
        
        Args:
            prompt: 提示文本
            use_history: 是否使用对话历史
            
        Returns:
            生成的文本
        """
        try:
            # 构建消息列表（Qwen2使用chat template）
            messages = [
                {"role": "system", "content": "You are a professional interviewer."}
            ]
            
            # 添加历史对话
            if use_history and self.conversation_history:
                for user_msg, assistant_msg in self.conversation_history[-3:]:
                    messages.append({"role": "user", "content": user_msg})
                    messages.append({"role": "assistant", "content": assistant_msg})
            
            # 添加当前提示
            messages.append({"role": "user", "content": prompt})
            
            # 应用chat template
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize
            inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            
            # 生成
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True
            )
            
            # 解码（只取新生成的部分）
            response = self.tokenizer.decode(
                outputs[0][len(inputs.input_ids[0]):], 
                skip_special_tokens=True
            ).strip()
            
            # 更新历史
            if use_history:
                self.conversation_history.append((prompt, response))
            
            return response
            
        except Exception as e:
            logger.error(f"LLM生成失败: {str(e)}")
            return "抱歉，我需要重新组织一下问题。请稍等。"
    
    def reset_history(self):
        """重置对话历史"""
        self.conversation_history = []
        logger.info("对话历史已重置")


