"""
AI Interviewer - 完整版（使用Qwen LoRA生成追问）
"""
import streamlit as st
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    AutoModel,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import PeftModel
from pathlib import Path
import json
import tempfile
from datetime import datetime
import sys
import importlib.util

# 直接导入ResumeParser
spec = importlib.util.spec_from_file_location("resume_parser", "models/resume_parser.py")
resume_parser_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(resume_parser_module)
ResumeParser = resume_parser_module.ResumeParser

# 导入数字人
spec_dh = importlib.util.spec_from_file_location("digital_human", "models/digital_human.py")
digital_human_module = importlib.util.module_from_spec(spec_dh)
spec_dh.loader.exec_module(digital_human_module)
DigitalHuman = digital_human_module.DigitalHuman

# 导入 Linly-Talker 客户端
spec_lt = importlib.util.spec_from_file_location("linly_talker_client", "models/linly_talker_client.py")
linly_module = importlib.util.module_from_spec(spec_lt)
spec_lt.loader.exec_module(linly_module)
LinlyTalkerClient = linly_module.LinlyTalkerClient

# 页面配置
st.set_page_config(
    page_title="AI Interviewer",
    page_icon="🎯",
    layout="wide"
)

# CSS样式 - 数字人+聊天界面
st.markdown("""
<style>
    /* 虚拟形象容器 */
    .avatar-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        text-align: center;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .avatar-wrapper {
        position: relative;
        display: inline-block;
    }
    
    /* 虚拟形象 - 动画效果 */
    .virtual-avatar {
        width: 180px;
        height: 180px;
        border-radius: 50%;
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        border: 5px solid white;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2);
        position: relative;
        animation: float 3s ease-in-out infinite;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 80px;
    }
    
    @keyframes float {
        0%, 100% { 
            transform: translateY(0px);
        }
        50% { 
            transform: translateY(-10px);
        }
    }
    
    /* 声纹效果 */
    .sound-wave {
        position: absolute;
        bottom: -20px;
        left: 50%;
        transform: translateX(-50%);
        display: flex;
        gap: 4px;
        align-items: flex-end;
        height: 60px;
    }
    
    .sound-bar {
        width: 6px;
        background: linear-gradient(to top, #4ade80, #22c55e);
        border-radius: 3px;
        animation: sound-wave 0.8s ease-in-out infinite;
    }
    
    .sound-bar:nth-child(1) { animation-delay: 0s; }
    .sound-bar:nth-child(2) { animation-delay: 0.1s; }
    .sound-bar:nth-child(3) { animation-delay: 0.2s; }
    .sound-bar:nth-child(4) { animation-delay: 0.3s; }
    .sound-bar:nth-child(5) { animation-delay: 0.4s; }
    .sound-bar:nth-child(6) { animation-delay: 0.3s; }
    .sound-bar:nth-child(7) { animation-delay: 0.2s; }
    .sound-bar:nth-child(8) { animation-delay: 0.1s; }
    
    @keyframes sound-wave {
        0%, 100% { height: 15px; }
        50% { height: 45px; }
    }
    
    /* 说话状态指示器 */
    .status-indicator {
        position: absolute;
        bottom: 15px;
        right: 15px;
        width: 25px;
        height: 25px;
        background: #4ade80;
        border-radius: 50%;
        border: 3px solid white;
        animation: pulse-glow 1.5s ease-in-out infinite;
        box-shadow: 0 0 15px rgba(74, 222, 128, 0.6);
    }
    
    @keyframes pulse-glow {
        0%, 100% { 
            transform: scale(1);
            opacity: 1;
        }
        50% { 
            transform: scale(1.2);
            opacity: 0.7;
        }
    }
    
    .avatar-speech {
        background: white;
        border-radius: 20px;
        padding: 1.5rem;
        margin-top: 1.5rem;
        position: relative;
        font-size: 1.1rem;
        color: #333;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        animation: fade-in 0.5s ease;
    }
    
    @keyframes fade-in {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .avatar-speech::before {
        content: '';
        position: absolute;
        top: -10px;
        left: 50%;
        transform: translateX(-50%);
        width: 0;
        height: 0;
        border-left: 12px solid transparent;
        border-right: 12px solid transparent;
        border-bottom: 12px solid white;
    }
    
    /* 聊天消息 */
    .chat-message {
        margin-bottom: 1rem;
        display: flex;
    }
    
    .chat-message.ai {
        justify-content: flex-start;
    }
    
    .chat-message.user {
        justify-content: flex-end;
    }
    
    .message-bubble {
        max-width: 70%;
        padding: 0.8rem 1rem;
        border-radius: 12px;
    }
    
    .chat-message.ai .message-bubble {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        box-shadow: 0 3px 10px rgba(102, 126, 234, 0.3);
        border-bottom-left-radius: 4px;
    }
    
    .chat-message.user .message-bubble {
        background: white;
        color: #333;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        border-bottom-right-radius: 4px;
    }
    
    .score-card {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* 隐藏音频播放器 */
    audio {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

# ==================== 定义模型 ====================
class MultiTaskRoBERTa(nn.Module):
    """RoBERTa多任务模型"""
    def __init__(self, model_name, num_labels=4):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(model_name)
        hidden_size = self.roberta.config.hidden_size
        
        self.classification_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_labels)
        )
        
        self.regression_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        classification_logits = self.classification_head(pooled_output)
        regression_score = self.regression_head(pooled_output).squeeze(-1)
        
        return classification_logits, regression_score

# ==================== 加载模型 ====================
@st.cache_resource
def load_models():
    """加载所有微调后的模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. BERT决策模型
    bert_path = "./checkpoints/follow_up_classifier_1500"
    bert_tokenizer = AutoTokenizer.from_pretrained(bert_path)
    bert_model = AutoModelForSequenceClassification.from_pretrained(bert_path)
    bert_model.to(device)
    bert_model.eval()
    
    # 2. RoBERTa评估模型
    roberta_path = "./checkpoints/answer_evaluator"
    roberta_base = "./models/chinese-roberta-wwm-ext"
    roberta_tokenizer = AutoTokenizer.from_pretrained(roberta_path)
    roberta_model = MultiTaskRoBERTa(roberta_base, num_labels=4)
    
    model_file = Path(roberta_path) / "pytorch_model.bin"
    state_dict = torch.load(model_file, map_location='cpu', weights_only=False)
    roberta_model.load_state_dict(state_dict)
    roberta_model.to(device)
    roberta_model.eval()
    
    # 3. Qwen模型（支持基座/LoRA切换）
    qwen_base = "Qwen/Qwen2-1.5B-Instruct"
    lora_path = "./checkpoints/qwen_interviewer_lora"
    
    qwen_tokenizer = None
    qwen_base_model = None
    qwen_lora_model = None
    
    try:
        # 加载tokenizer和基座模型
        qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_base, trust_remote_code=True)
        qwen_base_model = AutoModelForCausalLM.from_pretrained(
            qwen_base,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        qwen_base_model.eval()
        print(f"[INFO] Qwen base model loaded successfully")
        
        # 尝试加载LoRA权重（可选）
        if Path(lora_path).exists():
            try:
                qwen_lora_model = PeftModel.from_pretrained(qwen_base_model, lora_path)
                qwen_lora_model.eval()
                print(f"[INFO] Qwen LoRA model loaded successfully")
            except Exception as e:
                print(f"[WARNING] LoRA loading failed: {str(e)}, will use base model only")
                qwen_lora_model = None
        else:
            print(f"[INFO] LoRA checkpoint not found, using base model only")
    except Exception as e:
        print(f"[ERROR] Qwen loading failed: {str(e)}")
        qwen_base_model = None
    
    return {
        'bert_model': bert_model,
        'bert_tokenizer': bert_tokenizer,
        'roberta_model': roberta_model,
        'roberta_tokenizer': roberta_tokenizer,
        'qwen_base_model': qwen_base_model,     # 基座模型
        'qwen_lora_model': qwen_lora_model,     # LoRA模型（可能为None）
        'qwen_tokenizer': qwen_tokenizer,
        'device': device
    }

# ==================== 辅助函数 ====================
def generate_initial_question(models, skills, job_position, use_lora=False):
    """使用Qwen生成第一个面试问题
    
    Args:
        models: 模型字典
        skills: 候选人技能列表
        job_position: 应聘职位
        use_lora: 是否使用LoRA模型（False=基座模型，更自然）
    """
    # 选择使用的模型
    if use_lora and models.get('qwen_lora_model'):
        qwen_model = models['qwen_lora_model']
        model_name = "LoRA"
    elif models.get('qwen_base_model'):
        qwen_model = models['qwen_base_model']
        model_name = "Base"
    else:
        return "[错误] Qwen模型未加载，无法生成开场问题。请检查模型文件。"
    
    if not models['qwen_tokenizer']:
        return "[错误] Qwen Tokenizer未加载。"
    
    print(f"[INFO] Using Qwen {model_name} model for initial question")
    
    try:
        # 根据简历技能和职位生成友好的开场白
        skills_str = '、'.join(skills[:3]) if skills else '技术'
        
        # 根据模型类型调整提示词
        if use_lora:
            # LoRA模型：简洁指令（训练时的格式）
            system_msg = f"你是一位专业、友好的技术面试官，正在面试应聘{job_position}的候选人。请用自然、多样化的方式开场。"
            prompt = f"""候选人简历技能：{skills_str}
应聘职位：{job_position}

任务：用友好、自然的方式开场，请候选人介绍自己或相关项目经验。风格要多样化，可以：
- 直接询问项目经验
- 让候选人自我介绍
- 询问最近的工作内容
- 询问感兴趣的技术领域

请生成一个友好的开场问候（30-50字）："""
        else:
            # 基座模型：更自然的对话式指令
            system_msg = f"你是一位经验丰富的{job_position}面试官。你的面试风格专业但不失亲和力，善于通过对话了解候选人的真实能力。"
            prompt = f"""现在要面试一位应聘{job_position}的候选人，他的简历上列出了这些技能：{skills_str}。

请作为面试官，用自然、轻松的方式开场，让候选人介绍自己或分享相关经验。注意：
- 语气自然，像真实的面试对话
- 可以适当表达对候选人的兴趣
- 不要使用过于客套的寒暄（如"非常荣幸"等）
- 长度适中（50-80字）

现在请开始面试："""
        
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt}
        ]
        
        text = models['qwen_tokenizer'].apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = models['qwen_tokenizer'](
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512
        ).to(models['device'])
        
        with torch.no_grad():
            outputs = qwen_model.generate(
                **inputs,
                max_new_tokens=100 if not use_lora else 80,  # 基座模型生成更长
                temperature=0.7 if not use_lora else 0.9,     # 基座模型降低温度
                top_p=0.9 if not use_lora else 0.92,
                top_k=50,
                repetition_penalty=1.15 if not use_lora else 1.2,
                do_sample=True,
                pad_token_id=models['qwen_tokenizer'].eos_token_id
            )
        
        # 安全解码生成结果
        input_length = inputs['input_ids'].shape[1]
        if outputs.shape[1] > input_length:
            generated = models['qwen_tokenizer'].decode(
                outputs[0][input_length:],
                skip_special_tokens=True
            ).strip()
        else:
            generated = models['qwen_tokenizer'].decode(
                outputs[0],
                skip_special_tokens=True
            ).strip()
        
        if not generated:
            return "[错误] Qwen生成失败，请重试。"
        return generated
    
    except Exception as e:
        print(f"[ERROR] Initial question generation failed: {e}")
        return f"[错误] Qwen生成异常: {str(e)}"

def generate_qwen_question(models, context, question_type='follow_up', use_lora=False):
    """使用Qwen生成问题
    
    Args:
        models: 模型字典
        context: 上下文信息
        question_type: 问题类型 ('follow_up' 或 'topic_change')
        use_lora: 是否使用LoRA模型（False=基座模型，更自然）
    """
    # 选择使用的模型
    if use_lora and models.get('qwen_lora_model'):
        qwen_model = models['qwen_lora_model']
        model_name = "LoRA"
    elif models.get('qwen_base_model'):
        qwen_model = models['qwen_base_model']
        model_name = "Base"
    else:
        return "[错误] Qwen模型未加载，无法生成问题。"
    
    if not models['qwen_tokenizer']:
        return "[错误] Qwen Tokenizer未加载。"
    
    print(f"[INFO] Using Qwen {model_name} model for {question_type}")
    
    try:
        # 构建prompt
        if question_type == 'follow_up':
            answer = context['last_answer']
            question = context['question']
            topic = context.get('topic', '技术')
            score = context.get('score', 70)
            
            if use_lora:
                # LoRA模型：使用训练数据格式（简洁）
                if score >= 80:
                    task = "对候选人的回答给予肯定和鼓励，然后继续追问"
                else:
                    task = f"根据候选人的回答，生成一个追问问题，深入考察候选人对{topic}的理解"
                
                prompt = f"""面试官问题：{question}
候选人回答：{answer}

任务：{task}"""
                system_msg = f"你是一位专业、友好的{topic}面试官，正在面试候选人。你需要根据候选人的回答，决定是继续深入追问、换话题，还是给予鼓励。"
            else:
                # 基座模型：更自然的对话式指令
                if score >= 80:
                    feedback_guide = "候选人回答得不错，可以适当肯定（不要过于客套），然后追问更深入的问题"
                elif score >= 60:
                    feedback_guide = "候选人的回答比较笼统，需要引导他说得更具体"
                else:
                    feedback_guide = "候选人的回答不太理想，可以换个角度问，或者给个提示"
                
                prompt = f"""你刚才问了候选人："{question}"

候选人的回答是："{answer}"

{feedback_guide}。请生成你的下一个问题或回复。注意：
- 语气自然，像正常对话
- 不要使用"非常棒！"、"很好！"这种过于热情的表扬
- 如果需要肯定，可以说"嗯，理解了"、"可以"、"听起来不错"等
- 直接问下一个问题，不要啰嗦
- 长度适中（40-80字）

你的回复："""
                system_msg = f"你是一位经验丰富的{topic}技术面试官，面试风格专业、平和，善于引导候选人展示真实水平。"
        
        else:
            # 换话题（topic_change）
            answer = context.get('last_answer', '')
            question = context.get('question', '')
            topic = context.get('topic', '技术')
            score = context.get('score', 70)
            skills = context.get('skills', ['技术'])
            history = context.get('history', [])
            
            # 找出还没问过的技能
            asked_topics = set()
            for qa in history[-5:]:
                q = qa.get('question', '')
                for skill in skills:
                    if skill in q:
                        asked_topics.add(skill)
            
            remaining_skills = [s for s in skills if s not in asked_topics]
            next_skill = remaining_skills[0] if remaining_skills else skills[0]
            
            if use_lora:
                # LoRA模型：使用训练数据格式
                if score >= 80:
                    task = f"候选人对{topic}回答得很好，已经充分考察，可以换一个新话题"
                else:
                    task = f"候选人对{topic}不了解或答不上来，需要友好地换一个话题"
                
                prompt = f"""面试官问题：{question}
候选人回答：{answer}

任务：{task}"""
                system_msg = f"你是一位专业、友好的技术面试官，正在面试候选人。你需要根据候选人的回答，决定是继续深入追问、换话题，还是给予鼓励。"
            else:
                # 基座模型：更自然的换话题指令
                if score >= 80:
                    transition_guide = f"候选人对{topic}的理解已经考察得差不多了，可以自然地过渡到{next_skill}这个新话题"
                else:
                    transition_guide = f"候选人对{topic}不太熟悉，不要为难他，自然地切换到{next_skill}话题"
                
                prompt = f"""你刚才问了："{question}"
候选人回答："{answer}"

{transition_guide}。请生成你的下一个问题。注意：
- 不要使用"没关系"、"没问题"这种过于安慰的话
- 如果需要过渡，可以简单地说"好的"、"那我们聊聊..."、"换个方向"等
- 直接引入新话题，不要啰嗦
- 长度适中（30-60字）

你的回复："""
                system_msg = "你是一位经验丰富的技术面试官，面试风格专业、平和，善于自然地转换话题。"
        
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt}
        ]
        
        text = models['qwen_tokenizer'].apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = models['qwen_tokenizer'](
            [text],
            return_tensors="pt",
            padding=True
        ).to(models['device'])
        
        with torch.no_grad():
            outputs = qwen_model.generate(
                **inputs,
                max_new_tokens=100 if not use_lora else 60,  # 基座模型生成更长
                temperature=0.7 if not use_lora else 0.8,     # 基座模型降低温度
                top_p=0.9 if not use_lora else 0.9,
                top_k=40,
                do_sample=True,
                repetition_penalty=1.1 if not use_lora else 1.15,
                pad_token_id=models['qwen_tokenizer'].pad_token_id
            )
        
        # 安全解码生成结果
        input_length = len(inputs.input_ids[0])
        if outputs.shape[1] > input_length:
            response = models['qwen_tokenizer'].decode(
                outputs[0][input_length:],
                skip_special_tokens=True
            ).strip()
        else:
            response = models['qwen_tokenizer'].decode(
                outputs[0],
                skip_special_tokens=True
            ).strip()
        
        # 清理输出
        response = response.split('\n')[0][:100]
        
        if not response:
            return "[错误] Qwen生成为空，请重试。"
        return response
        
    except Exception as e:
        print(f"[ERROR] Qwen generation failed: {str(e)}")
        return f"[错误] Qwen生成异常: {str(e)}"

def evaluate_answer(models, question, answer, history_qa):
    """RoBERTa评估回答"""
    input_parts = []
    if history_qa:
        input_parts.append("[历史问答]")
        for i, h in enumerate(history_qa[-3:], 1):
            input_parts.append(f"Q{i}: {h['question']}")
            input_parts.append(f"A{i}: {h['answer'][:100]}")
            input_parts.append(f"质量: {h['quality']}")
    
    input_parts.append("[当前问答]")
    input_parts.append(f"问题: {question}")
    input_parts.append(f"回答: {answer}")
    input_parts.append(f"流畅度: 0.85")
    
    input_text = "\n".join(input_parts)
    
    inputs = models['roberta_tokenizer'](
        input_text,
        return_tensors='pt',
        truncation=True,
        max_length=256,
        padding=True
    ).to(models['device'])
    
    with torch.no_grad():
        cls_logits, reg_score = models['roberta_model'](
            inputs['input_ids'],
            inputs['attention_mask']
        )
        
        cls_probs = torch.softmax(cls_logits, dim=-1)
        predicted_idx = cls_probs.argmax().item()
        confidence = cls_probs.max().item()
        overall_score = reg_score.item() * 100
    
    label_names = ["差", "一般", "良好", "优秀"]
    score_mapping = [50, 70, 85, 95]
    
    return {
        'current_label': label_names[predicted_idx],
        'current_score': score_mapping[predicted_idx],
        'overall_score': overall_score,
        'confidence': confidence
    }

def decide_next_action(models, question, answer, follow_up_depth, topic):
    """BERT决策下一步（严格按照训练格式）"""
    answer_length = len(answer)
    
    # 计算犹豫度（基于语气词和停顿）
    hesitation_words = ['嗯', '啊', '这个', '那个', '就是', '怎么说', '...']
    hesitation_count = sum(answer.count(word) for word in hesitation_words)
    hesitation_score = min(0.9, hesitation_count * 0.15)
    
    # 和训练时完全一致的格式
    features = f"追问深度:{follow_up_depth} " \
              f"犹豫度:{hesitation_score:.2f} " \
              f"长度:{answer_length}字 " \
              f"话题:{topic}"
    
    bert_input = f"{question}[SEP]{answer}[SEP]{features}"
    
    inputs = models['bert_tokenizer'](
        bert_input,
        return_tensors='pt',
        truncation=True,
        max_length=512,
        padding=True
    ).to(models['device'])
    
    with torch.no_grad():
        outputs = models['bert_model'](**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        predicted = probs.argmax().item()
        conf = probs.max().item()
    
    bert_labels = ["FOLLOW_UP", "NEXT_TOPIC"]
    
    # 推断决策理由（基于训练数据中的reason模式）
    reason = _infer_decision_reason(answer, answer_length, hesitation_score, 
                                     follow_up_depth, bert_labels[predicted], topic)
    
    return {
        'action': bert_labels[predicted],
        'confidence': conf,
        'probs': probs[0].tolist(),
        'reason': reason,
        'hesitation_score': hesitation_score
    }

def _infer_decision_reason(answer, answer_length, hesitation_score, follow_up_depth, action, topic):
    """推断决策理由（基于训练数据中的reason模式）"""
    
    if action == "NEXT_TOPIC":
        # NEXT_TOPIC的常见原因
        if any(word in answer for word in ['不了解', '不懂', '没学过', '没接触']):
            return f"候选者明确表示不了解{topic}，应换其他话题"
        elif follow_up_depth >= 2 and hesitation_score > 0.4:
            return f"经过{follow_up_depth}轮追问，候选者对{topic}的理解仍然模糊/不足，建议换话题"
        elif answer_length < 20:
            return f"候选者回答过于简短且缺乏实质内容，建议换其他话题"
        else:
            return "根据综合分析，建议换话题考察其他技能点"
    
    else:  # FOLLOW_UP
        # FOLLOW_UP的常见原因
        if any(word in answer for word in ['用过', '做过', '项目']) and answer_length < 80:
            return f"候选者提到了使用场景但缺少细节，可以继续追问更深入的技术细节"
        elif hesitation_score > 0.3 and any(word in answer for word in ['接触过', '了解', '用到']):
            return f"候选者承认使用过{topic}但未展开细节（提到了'承认用过但说不清细节'），需追问具体实现"
        elif answer_length >= 50:
            return "候选者给出了一定内容，可以针对回答中的关键点继续深入追问"
        else:
            return "可以继续追问以更全面评估候选者的技术能力"

# ==================== 主界面 ====================
st.title("🎯 AI Interviewer")

# 初始化
if 'stage' not in st.session_state:
    st.session_state.stage = 'upload'
if 'resume_data' not in st.session_state:
    st.session_state.resume_data = None
if 'current_question' not in st.session_state:
    st.session_state.current_question = None
if 'qa_history' not in st.session_state:
    st.session_state.qa_history = []
if 'follow_up_depth' not in st.session_state:
    st.session_state.follow_up_depth = 0
if 'current_topic' not in st.session_state:
    st.session_state.current_topic = 'Python'
if 'total_rounds' not in st.session_state:
    st.session_state.total_rounds = 0
if 'job_position' not in st.session_state:
    st.session_state.job_position = 'Python后端工程师'
if 'digital_human' not in st.session_state:
    st.session_state.digital_human = DigitalHuman()
if 'linly_client' not in st.session_state:
    st.session_state.linly_client = LinlyTalkerClient()
if 'digital_human_mode' not in st.session_state:
    # 默认使用轻量级模式，如果Linly服务可用则可以切换
    st.session_state.digital_human_mode = 'lightweight'
if 'avatar_image_path' not in st.session_state:
    st.session_state.avatar_image_path = "Linly-Talker/examples/source_image/full_body_1.png"

# ==================== 阶段1: 上传简历 ====================
if st.session_state.stage == 'upload':
    st.markdown("---")
    st.subheader("📄 步骤1：上传简历")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        upload_method = st.radio(
            "选择上传方式",
            ["上传文件 (PDF/DOCX)", "直接粘贴文本"],
            horizontal=True
        )
        
        resume_text = ""
        
        if upload_method == "上传文件 (PDF/DOCX)":
            uploaded_file = st.file_uploader(
                "上传简历文件",
                type=['pdf', 'docx'],
                help="支持PDF和DOCX格式"
            )
            
            if uploaded_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                    tmp_path = tmp_file.name
                
                try:
                    with st.spinner("正在解析简历..."):
                        parser = ResumeParser()
                        resume_data = parser.parse(tmp_path)
                        st.session_state.resume_data = resume_data
                    
                    st.success(f"✅ 简历解析成功！")
                    
                    with st.expander("📋 查看解析结果"):
                        st.markdown(f"**姓名：** {resume_data['name']}")
                        if resume_data['contact']:
                            st.markdown(f"**联系方式：** {resume_data['contact']}")
                        st.markdown(f"**提取到的技能：** {', '.join(resume_data['skills'][:10])}")
                    
                    resume_text = resume_data['raw_text']
                    
                except Exception as e:
                    st.error(f"简历解析失败：{str(e)}")
                finally:
                    Path(tmp_path).unlink(missing_ok=True)
        else:
            resume_text = st.text_area(
                "粘贴简历内容",
                height=300,
                placeholder="姓名：张三\n应聘职位：Python后端工程师\n\n技能：Python, Django, MySQL..."
            )
        
        job_position = st.text_input("应聘职位", value="Python后端工程师")
    
    with col2:
        st.info("💡 **智能特性**\n\n✅ PDF/DOCX解析\n\n✅ BERT智能决策\n\n✅ **Qwen LoRA追问**\n\n✅ RoBERTa评估\n\n✅ **无固定轮数**")
        
        if st.session_state.resume_data:
            st.success(f"**已识别技能：**\n\n" + "\n".join([f"• {s}" for s in st.session_state.resume_data['skills'][:5]]))
    
    if st.button("🚀 开始面试", type="primary", use_container_width=True):
        if resume_text.strip() or st.session_state.resume_data:
            if not st.session_state.resume_data and resume_text:
                parser = ResumeParser()
                st.session_state.resume_data = {
                    'name': '候选人',
                    'skills': [s for s in ['Python', 'Java', 'JavaScript'] if s.lower() in resume_text.lower()],
                    'raw_text': resume_text
                }
            
            # 标记为进入面试阶段，第一个问题将在面试阶段动态生成
            skills = st.session_state.resume_data['skills'] or ['技术']
            first_skill = skills[0] if skills else '技术'
            st.session_state.current_topic = first_skill
            st.session_state.current_question = None  # 标记为需要生成
            st.session_state.job_position = job_position
            st.session_state.stage = 'interview'
            st.rerun()
        else:
            st.error("请先上传简历或粘贴文本")

# ==================== 阶段2: 面试过程 ====================
elif st.session_state.stage == 'interview':
    with st.spinner("正在加载AI模型..."):
        models = load_models()
    
    # 在侧边栏显示模型加载状态和配置（缓存外）
    with st.sidebar:
        st.markdown("### 🔍 模型状态")
        if models['qwen_base_model'] is not None:
            st.success("✅ Qwen Base模型已加载")
        if models['qwen_lora_model'] is not None:
            st.success("✅ Qwen LoRA模型已加载")
        if models['qwen_base_model'] is None:
            st.error("❌ Qwen未加载 - 无法继续面试")
        st.success("✅ BERT决策模型")
        st.success("✅ RoBERTa评估模型")
        st.markdown("---")
        
        # 模型选择
        st.markdown("### ⚙️ 面试风格")
        if 'use_lora' not in st.session_state:
            st.session_state.use_lora = False  # 默认使用基座模型
        
        use_lora = st.checkbox(
            "使用LoRA模型（简洁风格）",
            value=st.session_state.use_lora,
            help="✅ LoRA：简洁、直接（约20字）\n❌ 基座：自然、详细（约60字）",
            disabled=(models['qwen_lora_model'] is None)
        )
        st.session_state.use_lora = use_lora
        
        if st.session_state.use_lora:
            st.info("🔹 当前：LoRA模型（简洁风格）")
        else:
            st.info("🔹 当前：基座模型（自然对话风格）")
        st.markdown("---")
    
    # 生成第一个问题（如果还没有）
    if st.session_state.current_question is None:
        with st.spinner("Qwen正在生成开场问题..."):
            skills = st.session_state.resume_data.get('skills', ['技术'])
            job_position = st.session_state.get('job_position', 'Python后端工程师')
            st.session_state.current_question = generate_initial_question(models, skills, job_position, use_lora=st.session_state.use_lora)
    
    # 顶部信息
    if st.session_state.resume_data:
        st.caption(f"👤 {st.session_state.resume_data.get('name', '候选人')} | 应聘：{st.session_state.job_position} | 已完成：{st.session_state.total_rounds}轮")
    
    # 主布局：左侧(数字人+对话) 右侧(评分)
    col_main, col_score = st.columns([2.5, 1])
    
    with col_main:
        # ========== 数字人面试官 ==========
        if st.session_state.digital_human_mode == 'linly':
            # 使用 Linly-Talker 生成视频
            st.markdown('<div style="text-align: center; margin-bottom: 1rem;">', unsafe_allow_html=True)
            
            # 生成视频（带缓存key避免重复生成）
            video_cache_key = f"video_{hash(st.session_state.current_question)}"
            if video_cache_key not in st.session_state:
                with st.spinner("🎬 数字人正在生成视频..."):
                    try:
                        video_path = st.session_state.linly_client.generate_video(
                            text=st.session_state.current_question,
                            avatar_image=st.session_state.avatar_image_path
                        )
                        st.session_state[video_cache_key] = video_path
                    except Exception as e:
                        st.error(f"视频生成失败: {e}")
                        st.info("已自动切换到轻量级模式")
                        st.session_state.digital_human_mode = 'lightweight'
                        video_path = None
            else:
                video_path = st.session_state[video_cache_key]
            
            if video_path and Path(video_path).exists():
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            border-radius: 20px; padding: 1.5rem; margin-bottom: 1rem;">
                    <div style="color: white; font-size: 1.2rem; font-weight: 600; margin-bottom: 1rem; text-align: center;">
                        Alice · {st.session_state.current_topic}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.video(video_path, autoplay=True)
            else:
                # 降级到轻量级模式
                avatar_html = st.session_state.digital_human.get_avatar_html(
                    question=st.session_state.current_question,
                    topic=st.session_state.current_topic,
                    include_audio=True
                )
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            # 虚拟形象模式：带声纹效果的动画形象
            # 显示虚拟形象
            st.markdown(f"""
            <div class="avatar-container">
                <div class="avatar-wrapper">
                    <div class="virtual-avatar">
                        🤖
                        <div class="status-indicator"></div>
                    </div>
                    <div class="sound-wave">
                        <div class="sound-bar"></div>
                        <div class="sound-bar"></div>
                        <div class="sound-bar"></div>
                        <div class="sound-bar"></div>
                        <div class="sound-bar"></div>
                        <div class="sound-bar"></div>
                        <div class="sound-bar"></div>
                        <div class="sound-bar"></div>
                    </div>
                </div>
                <div style="margin-top: 2rem; color: white;">
                    <h3 style="margin: 0;">Alice</h3>
                    <p style="margin: 0.5rem 0; opacity: 0.9;">话题：{st.session_state.current_topic}</p>
                </div>
            </div>
            
            <div class="avatar-speech">
                <div style="font-size: 0.9rem; color: #999; margin-bottom: 0.5rem;">💬 面试官提问：</div>
                <div style="font-size: 1.1rem; line-height: 1.6;">{st.session_state.current_question}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # 生成并自动播放语音
            try:
                audio_file = st.session_state.digital_human.text_to_speech(st.session_state.current_question)
                if audio_file and Path(audio_file).exists():
                    import base64
                    with open(audio_file, 'rb') as f:
                        audio_bytes = f.read()
                    audio_base64 = base64.b64encode(audio_bytes).decode()
                    
                    # 使用问题内容的hash作为唯一ID（确保不同问题有不同ID）
                    import hashlib
                    question_hash = hashlib.md5(st.session_state.current_question.encode()).hexdigest()[:8]
                    audio_id = f"tts_{question_hash}"
                    
                    # 只在session_state中没有记录这个音频ID时才播放（避免重复）
                    if 'last_audio_id' not in st.session_state or st.session_state.last_audio_id != audio_id:
                        st.session_state.last_audio_id = audio_id
                        
                        # 使用HTML5音频自动播放
                        st.components.v1.html(f"""
                        <audio id="{audio_id}" autoplay style="display:none;">
                            <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                        </audio>
                        """, height=0)
            except Exception as e:
                print(f"[WARNING] Audio generation failed: {e}")
        
        # ========== 对话历史 ==========
        st.markdown("### 💬 对话记录")
        
        # 聊天式历史记录（只显示最近5轮）
        with st.container():
            st.markdown('<div style="max-height: 350px; overflow-y: auto; padding: 1rem; background: #f8f9fa; border-radius: 10px; margin-bottom: 1rem;">', unsafe_allow_html=True)
            
            if not st.session_state.qa_history:
                st.markdown('<div style="text-align: center; color: #999; padding: 2rem;">暂无对话记录</div>', unsafe_allow_html=True)
            else:
                for i, qa in enumerate(st.session_state.qa_history[-5:]):
                    quality_emoji = {'差': '🔴', '一般': '🟡', '良好': '🔵', '优秀': '🟢'}.get(qa.get('quality', '一般'), '⚪')
                    
                    # AI消息
                    st.markdown(f'''
                    <div class="chat-message ai">
                        <div class="message-bubble">
                            <div style="font-size: 0.85rem; opacity: 0.9;">🤖 面试官</div>
                            <div style="margin-top: 0.3rem;">{qa["question"]}</div>
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)
                    
                    # 用户消息
                    st.markdown(f'''
                    <div class="chat-message user">
                        <div class="message-bubble">
                            <div style="font-size: 0.85rem; opacity: 0.7;">👤 候选人</div>
                            <div style="margin-top: 0.3rem;">{qa["answer"]}</div>
                            <div style="font-size: 0.75rem; margin-top: 0.5rem; opacity: 0.8;">{quality_emoji} {qa.get('quality', '一般')}</div>
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # ========== 回答输入 ==========
        st.markdown("### ✍️ 您的回答")
        answer = st.text_area(
            "请输入您的回答",
            height=120,
            key=f"answer_{st.session_state.total_rounds}",
            placeholder="在此输入您的回答..."
        )
        
        col_s, col_e = st.columns([3, 1])
        
        with col_s:
                if st.button("✅ 提交回答", type="primary", use_container_width=True):
                    if answer.strip():
                        # 评估
                        with st.spinner("正在评估回答..."):
                            eval_result = evaluate_answer(models, st.session_state.current_question, answer, st.session_state.qa_history)
                        
                        # 显示评估详情
                        st.sidebar.markdown("### 📊 本轮评估")
                        st.sidebar.write(f"质量: {eval_result['current_label']}")
                        st.sidebar.write(f"当前分: {eval_result['current_score']}")
                        st.sidebar.write(f"整体分: {eval_result['overall_score']:.1f}")
                        st.sidebar.write(f"置信度: {eval_result['confidence']:.1%}")
                        
                        # 记录
                        st.session_state.qa_history.append({
                            'question': st.session_state.current_question,
                            'answer': answer,
                            'quality': eval_result['current_label'],
                            'eval': eval_result
                        })
                        
                        st.session_state.total_rounds += 1
                        
                        # BERT决策
                        with st.spinner("BERT正在决策..."):
                            decision = decide_next_action(
                                models,
                                st.session_state.current_question,
                                answer,
                                st.session_state.follow_up_depth,
                                st.session_state.current_topic
                            )
                        
                        # 显示决策详情
                        st.sidebar.markdown("### 🧠 BERT决策")
                        st.sidebar.write(f"**动作**: {decision['action']}")
                        st.sidebar.write(f"**置信度**: {decision['confidence']:.1%}")
                        st.sidebar.write(f"**理由**: {decision['reason']}")
                        st.sidebar.caption(f"犹豫度: {decision['hesitation_score']:.2f} | FOLLOW_UP概率: {decision['probs'][0]:.1%}")
                        
                        # 生成下一个问题
                        if decision['action'] == 'FOLLOW_UP':
                            st.session_state.follow_up_depth += 1
                            # 使用Qwen生成追问
                            st.sidebar.markdown("### 🤖 Qwen生成追问")
                            with st.spinner("Qwen正在生成追问..."):
                                context = {
                                    'last_answer': answer,
                                    'question': st.session_state.current_question,
                                    'topic': st.session_state.current_topic,
                                    'score': eval_result['current_score']
                                }
                                new_q = generate_qwen_question(models, context, 'follow_up', use_lora=st.session_state.use_lora)
                                st.session_state.current_question = new_q
                                st.sidebar.write(f"**类型**: 追问")
                                st.sidebar.write(f"**问题**: {new_q}")
                                if '[错误]' not in new_q:
                                    with st.sidebar.expander("查看提示词"):
                                        st.code(f"问题: {context['question']}\n回答: {context['last_answer'][:50]}...", language="text")
                        else:
                            st.session_state.follow_up_depth = 0
                            # 使用Qwen生成新话题问题
                            st.sidebar.markdown("### 🤖 Qwen生成新题")
                            with st.spinner("Qwen正在生成新问题..."):
                                context = {
                                    'last_answer': answer,
                                    'question': st.session_state.current_question,
                                    'topic': st.session_state.current_topic,
                                    'score': eval_result['current_score'],
                                    'skills': st.session_state.resume_data.get('skills', []),
                                    'history': st.session_state.qa_history
                                }
                                new_q = generate_qwen_question(models, context, 'new_topic', use_lora=st.session_state.use_lora)
                                st.session_state.current_question = new_q
                                
                                # 更新当前话题
                                skills = context.get('skills', [])
                                for skill in skills:
                                    if skill in new_q:
                                        st.session_state.current_topic = skill
                                        break
                                
                                st.sidebar.write(f"**类型**: 新话题")
                                st.sidebar.write(f"**问题**: {new_q}")
                                if '[错误]' not in new_q:
                                    with st.sidebar.expander("查看提示词"):
                                        st.code(f"技能: {', '.join(context.get('skills', []))}\n已问轮数: {len(context.get('history', []))}", language="text")
                        
                        st.rerun()
                    else:
                        st.error("请输入回答")
            
        with col_e:
            if st.button("🏁 结束面试", use_container_width=True):
                if st.session_state.qa_history:
                    st.session_state.stage = 'summary'
                    st.rerun()
    
    with col_score:
        # ========== 评分面板 ==========
        st.markdown("### 📊 实时评分")
        
        if st.session_state.qa_history:
            latest = st.session_state.qa_history[-1]['eval']
            
            st.markdown(f"""
            <div class="score-card">
                <div style="color: #999; font-size: 0.9rem;">当前回答</div>
                <div style="font-size: 2rem; font-weight: bold; color: #667eea;">{latest['current_label']}</div>
                <div style="color: #999; font-size: 0.85rem;">{latest['current_score']}分</div>
            </div>
            
            <div class="score-card">
                <div style="color: #999; font-size: 0.9rem;">整体表现</div>
                <div style="font-size: 2rem; font-weight: bold; color: #667eea;">{latest['overall_score']:.1f}</div>
                <div style="color: #999; font-size: 0.85rem;">满分100</div>
            </div>
            
            <div class="score-card">
                <div style="color: #999; font-size: 0.9rem;">已完成轮数</div>
                <div style="font-size: 2rem; font-weight: bold; color: #667eea;">{st.session_state.total_rounds}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("回答问题后将显示评分")
        
        st.markdown("---")
        
        # AI模型状态
        with st.expander("🔍 AI决策详情", expanded=False):
            st.caption("查看BERT、RoBERTa详细分析")
            if st.session_state.qa_history:
                last_qa = st.session_state.qa_history[-1]
                st.json(last_qa.get('eval', {}))

# ==================== 阶段3: 总结 ====================
elif st.session_state.stage == 'summary':
    st.markdown("---")
    st.subheader("📈 面试总结报告")
    
    if st.session_state.qa_history:
        final_eval = st.session_state.qa_history[-1]['eval']
        final_score = final_eval['overall_score']
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("总体评分", f"{final_score:.1f}/100")
        with col2:
            st.metric("总轮数", st.session_state.total_rounds)
        with col3:
            avg_current = sum(qa['eval']['current_score'] for qa in st.session_state.qa_history) / len(st.session_state.qa_history)
            st.metric("平均分", f"{avg_current:.0f}")
        with col4:
            excellent = sum(1 for qa in st.session_state.qa_history if qa['quality'] == '优秀')
            st.metric("优秀", f"{excellent}个")
        
        st.markdown("---")
        if final_score >= 85:
            st.success(f"### 🌟 强烈推荐\n候选人表现优秀，技术功底扎实。")
        elif final_score >= 70:
            st.success(f"### 👍 推荐\n候选人具备相应技能，表现良好。")
        elif final_score >= 50:
            st.warning(f"### 🤔 待定\n候选人基础一般，需进一步考察。")
        else:
            st.error(f"### ❌ 不推荐\n候选人技术能力不足。")
        
        st.markdown("---")
        st.markdown("### 📝 对话详情")
        
        for i, qa in enumerate(st.session_state.qa_history, 1):
            with st.expander(f"第{i}轮 - {qa['quality']} ({qa['eval']['current_score']}分)"):
                st.markdown(f"**Q:** {qa['question']}")
                st.markdown(f"**A:** {qa['answer']}")
                st.markdown(f"**评分:** {qa['eval']['current_label']} ({qa['eval']['current_score']}分) | 整体: {qa['eval']['overall_score']:.1f}/100")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 重新开始", type="primary", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    with col2:
        if st.button("📥 导出报告", use_container_width=True):
            report = {
                'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'candidate': st.session_state.resume_data.get('name', '未知'),
                'final_score': final_score,
                'total_rounds': st.session_state.total_rounds,
                'qa_history': st.session_state.qa_history
            }
            
            st.download_button(
                "💾 下载报告",
                data=json.dumps(report, ensure_ascii=False, indent=2),
                file_name=f"interview_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )