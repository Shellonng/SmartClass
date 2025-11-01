"""
AI Interviewer - Triple Qwen版（全Qwen架构）
使用三个微调的Qwen模型：Decision、Question、Scorer
"""
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from pathlib import Path
import json
import tempfile
from datetime import datetime
import sys
import importlib.util
import re

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

# 页面配置
st.set_page_config(
    page_title="AI Interviewer - Triple Qwen",
    page_icon="🚀",
    layout="wide"
)

# CSS样式
st.markdown("""
<style>
    .avatar-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        text-align: center;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }
    
    .virtual-avatar {
        width: 180px;
        height: 180px;
        border-radius: 50%;
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        border: 5px solid white;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2);
        animation: float 3s ease-in-out infinite;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 80px;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    .avatar-speech {
        background: white;
        border-radius: 20px;
        padding: 1.5rem;
        margin-top: 1.5rem;
        font-size: 1.1rem;
        color: #333;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
    }
    
    .chat-message {
        margin-bottom: 1rem;
        display: flex;
    }
    
    .chat-message.ai { justify-content: flex-start; }
    .chat-message.user { justify-content: flex-end; }
    
    .message-bubble {
        max-width: 70%;
        padding: 0.8rem 1rem;
        border-radius: 12px;
    }
    
    .chat-message.ai .message-bubble {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        box-shadow: 0 3px 10px rgba(102, 126, 234, 0.3);
    }
    
    .chat-message.user .message-bubble {
        background: white;
        color: #333;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }
    
    .score-card {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ==================== 加载Triple Qwen模型 ====================
@st.cache_resource
def load_triple_qwen_models():
    """加载三个微调的Qwen模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_model_name = "Qwen/Qwen2-1.5B-Instruct"
    
    print("[INFO] Loading Triple Qwen models...")
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 4bit量化配置（节省显存）
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    models = {}
    
    # 加载三个模型
    model_configs = [
        ("decision", "checkpoints/qwen_decision_lora", "Qwen-Decision"),
        ("question", "checkpoints/qwen_question_lora", "Qwen-Question"),
        ("scorer", "checkpoints/qwen_scorer_lora", "Qwen-Scorer")
    ]
    
    for model_key, lora_path, model_name in model_configs:
        try:
            # 加载基座模型
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                quantization_config=bnb_config,
                device_map="auto"
            )
            
            # 加载LoRA权重
            if Path(lora_path).exists():
                model = PeftModel.from_pretrained(base_model, lora_path)
                model.eval()
                models[model_key] = model
                print(f"[INFO] {model_name} loaded successfully")
            else:
                print(f"[ERROR] LoRA path not found: {lora_path}")
                models[model_key] = None
        except Exception as e:
            print(f"[ERROR] Failed to load {model_name}: {str(e)}")
            models[model_key] = None
    
    return {
        'tokenizer': tokenizer,
        'decision_model': models['decision'],
        'question_model': models['question'],
        'scorer_model': models['scorer'],
        'device': device
    }

# ==================== Triple Qwen推理函数 ====================

def generate_with_qwen(model, tokenizer, instruction, input_text, max_tokens=256, temperature=0.7, device='cuda'):
    """通用Qwen生成函数"""
    if model is None:
        return "[错误] 模型未加载"
    
    try:
        # 构建prompt
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": input_text}
        ]
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = tokenizer(text, return_tensors='pt').to(device)
        input_length = inputs['input_ids'].shape[1]
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # 只解码生成的新token
        generated_tokens = outputs[0][input_length:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        return response
    
    except Exception as e:
        print(f"[ERROR] Generation failed: {str(e)}")
        return f"[错误] 生成失败: {str(e)}"

def decision_make(models, resume_data, history, scores, current_topic, next_topic):
    """Qwen-Decision: 做出决策，给出详细指导"""
    
    # 构建对话历史
    history_text = ""
    recent_history = history[-3:] if len(history) > 3 else history
    for h in recent_history:
        history_text += f"问: {h['question']}\n答: {h['answer'][:100]}...\n评分: {h.get('score', 70)}分\n\n"
    
    avg_score = sum(scores) / len(scores) if scores else 0
    round_number = len(history) + 1
    
    # 准备next_topic的描述
    next_topic_desc = ""
    if next_topic:
        if next_topic.startswith("项目:"):
            proj_name = next_topic.replace("项目:", "")
            # 从简历中找项目详情
            for proj in resume_data.get('projects', []):
                if proj.get('name') == proj_name:
                    tech_stack = ', '.join(proj.get('tech_stack', [])[:3])
                    next_topic_desc = f"'{proj_name}'项目（技术栈：{tech_stack}）"
                    break
        elif next_topic.startswith("技能:"):
            skill = next_topic.replace("技能:", "")
            next_topic_desc = f"'{skill}'技能"
    
    input_text = f"""当前话题: {current_topic}

对话历史:
{history_text.strip() if history_text else '（这是第一轮）'}

评分: 平均{avg_score:.0f}分

如需切换话题，下一个话题是: {next_topic_desc if next_topic_desc else '无'}"""
    
    instruction = """你是技术面试官。根据对话历史和评分，做出决策。

输出格式：
决策: FOLLOW_UP 或 SWITCH_TOPIC
指导建议: [简短说明，如"继续深入XXX"或"切换到XXX项目"]"""
    
    # 在调用模型前，先做规则判断
    force_switch = False
    force_reason = ""
    
    # 检查是否需要强制切换话题
    if len(history) >= 2:
        recent_2 = history[-2:]
        
        # 规则1: 连续2次低分 (<60分)
        if all(h.get('score', 70) < 60 for h in recent_2):
            force_switch = True
            force_reason = f"候选人连续2次低分({recent_2[-2]['score']}, {recent_2[-1]['score']}分)，建议切换话题"
        
        # 规则2: 候选人明确表示不懂
        negative_keywords = ['不知道', '忘了', '不会', '没做过', '不清楚', '不了解', '不太懂']
        last_answer = history[-1]['answer']
        if any(keyword in last_answer for keyword in negative_keywords) and history[-1].get('score', 70) < 70:
            force_switch = True
            force_reason = f"候选人表示不了解当前话题(评分{history[-1]['score']}分)，建议切换到简历中的其他项目"
    
    # 如果需要强制切换，直接返回
    if force_switch:
        # 使用提供的next_topic信息
        if next_topic_desc:
            guidance = f"{force_reason}。切换到{next_topic_desc}。"
        else:
            guidance = f"{force_reason}。建议结束面试或总结。"
        
        return {
            'action': 'SWITCH_TOPIC',
            'guidance': guidance,
            'raw_response': f"[规则强制] {guidance}",
            'force_switch': True
        }
    
    # 否则调用模型决策
    response = generate_with_qwen(
        models['decision_model'],
        models['tokenizer'],
        instruction,
        input_text,
        max_tokens=100,  # 减少token数，要求简短
        temperature=0.5,  # 降低temperature，让决策更稳定
        device=models['device']
    )
    
    # 解析决策和指导
    action = "SWITCH_TOPIC"
    guidance = response
    
    if "决策:" in response or "决策：" in response:
        parts = re.split(r'指导建议[：:]', response)
        action_part = parts[0].replace("决策:", "").replace("决策：", "").strip()
        
        if "FOLLOW_UP" in action_part.upper():
            action = "FOLLOW_UP"
        elif "SWITCH_TOPIC" in action_part.upper() or "SWITCH" in action_part.upper():
            action = "SWITCH_TOPIC"
        
        if len(parts) > 1:
            guidance = parts[1].strip()
    
    return {
        'action': action,
        'guidance': guidance,
        'raw_response': response
    }

def question_generate(models, history, guidance, topic):
    """Qwen-Question: 根据guidance和当前topic生成问题"""
    
    # 构建对话历史（最近1轮完整回答）
    history_text = ""
    last_answer = ""
    if history:
        last_qa = history[-1]
        last_answer = last_qa['answer'][:150]  # 保留更多回答内容
        history_text = f"最近一轮:\nQ: {last_qa['question'][:80]}...\nA: {last_answer}\n"
    
    # 特殊处理：如果是从"自我介绍"切换到第一个项目，使用开放性引入
    if (len(history) == 1 and 
        history[0]['question'].startswith("你好！首先") and 
        topic.startswith("项目:")):
        # 第一个项目的引入性问题
        proj_name = topic.replace("项目:", "")
        input_text = f"""话题: {topic}
指导: {guidance}

这是从自我介绍切换到第一个项目，生成一个开放性的引入问题。"""
        
        instruction = """生成面试问题。对于项目，可以用"我看到你的简历中有XXX项目，能详细介绍一下吗？"这样的开放性问题。

输出：
问题: [问题内容]
重要程度: [1-5分]"""
    else:
        # 正常流程
        input_text = f"""话题: {topic}
{history_text}
指导: {guidance}

注意：根据候选人的实际回答提问，不要假设候选人说过某些话。"""
        
        instruction = """根据指导生成面试问题。不要编造候选人没说过的内容。

输出：
问题: [问题内容]
重要程度: [1-5分]"""
    
    response = generate_with_qwen(
        models['question_model'],
        models['tokenizer'],
        instruction,
        input_text,
        max_tokens=150,
        temperature=0.5,  # 降低temperature，减少"创造性"
        device=models['device']
    )
    
    # 解析问题和重要程度
    question = response
    importance = 3
    
    if "问题:" in response or "问题：" in response:
        parts = re.split(r'重要程度[：:]', response)
        question = parts[0].replace("问题:", "").replace("问题：", "").strip()
        
        if len(parts) > 1:
            importance_str = parts[1].strip().split("分")[0].strip()
            try:
                importance = int(importance_str)
            except:
                # 尝试提取数字
                nums = re.findall(r'\d+', importance_str)
                if nums:
                    importance = int(nums[0])
                else:
                    importance = 3
    
    return {
        'question': question,
        'importance': importance,
        'raw_response': response
    }

def answer_evaluate(models, question, answer):
    """Qwen-Scorer: 评估回答"""
    input_text = f"""面试问题: {question}

候选人回答:
{answer}

请评估这个回答的质量，给出评分（0-100分）、标签（excellent/good/average/poor）和评价。"""
    
    instruction = "你是一位经验丰富的技术面试官。你的任务是评估候选人对技术问题的回答质量，给出评分（0-100分）、标签（excellent/good/average/poor）和详细评价。评分标准：excellent(85-100)表示回答准确、深入、有实战经验；good(70-84)表示回答正确但不够深入；average(50-69)表示回答部分正确或较浅；poor(0-49)表示回答错误或完全不会。"
    
    response = generate_with_qwen(
        models['scorer_model'],
        models['tokenizer'],
        instruction,
        input_text,
        max_tokens=256,
        temperature=0.7,
        device=models['device']
    )
    
    # 解析评分、标签和评价
    score = 70
    label = "average"
    comment = response
    
    # 提取评分
    if "评分:" in response or "评分：" in response:
        score_match = re.search(r'评分[：:]\s*(\d+)\s*分?', response)
        if score_match:
            score = int(score_match.group(1))
    
    # 提取标签
    if "标签:" in response or "标签：" in response:
        label_match = re.search(r'标签[：:]\s*(\w+)', response)
        if label_match:
            label_text = label_match.group(1).lower()
            if 'excellent' in label_text or '优秀' in label_text:
                label = 'excellent'
            elif 'good' in label_text or '良好' in label_text:
                label = 'good'
            elif 'poor' in label_text or '差' in label_text:
                label = 'poor'
            else:
                label = 'average'
    
    # 提取评价
    if "评价:" in response or "评价：" in response:
        comment_parts = re.split(r'评价[：:]', response)
        if len(comment_parts) > 1:
            comment = comment_parts[1].strip()
    
    # 标签映射为中文
    label_map = {
        'excellent': '优秀',
        'good': '良好',
        'average': '一般',
        'poor': '差'
    }
    
    return {
        'score': score,
        'label': label,
        'label_cn': label_map.get(label, '一般'),
        'comment': comment,
        'raw_response': response
    }

# ==================== 主界面 ====================
st.title("🚀 AI Interviewer - Triple Qwen")
st.caption("基于全Qwen架构的智能面试系统 | Decision + Question + Scorer")

# 初始化session_state
if 'stage' not in st.session_state:
    st.session_state.stage = 'upload'
if 'resume_data' not in st.session_state:
    st.session_state.resume_data = None
if 'current_question' not in st.session_state:
    st.session_state.current_question = None
if 'current_importance' not in st.session_state:
    st.session_state.current_importance = 3
if 'qa_history' not in st.session_state:
    st.session_state.qa_history = []
if 'total_rounds' not in st.session_state:
    st.session_state.total_rounds = 0
if 'job_position' not in st.session_state:
    st.session_state.job_position = 'Python后端工程师'
if 'digital_human' not in st.session_state:
    st.session_state.digital_human = DigitalHuman()
if 'current_guidance' not in st.session_state:
    st.session_state.current_guidance = None
if 'current_action' not in st.session_state:
    st.session_state.current_action = None
if 'current_topic' not in st.session_state:
    st.session_state.current_topic = "自我介绍"
if 'topic_queue' not in st.session_state:
    st.session_state.topic_queue = []
if 'topic_index' not in st.session_state:
    st.session_state.topic_index = 0

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
                    
                    with st.expander("📋 查看解析结果", expanded=True):
                        # 基本信息
                        st.markdown("### 👤 基本信息")
                        col_info1, col_info2 = st.columns(2)
                        with col_info1:
                            st.markdown(f"**姓名：** {resume_data['name']}")
                            if resume_data.get('basic_info', {}).get('gender'):
                                st.markdown(f"**性别：** {resume_data['basic_info']['gender']}")
                            if resume_data.get('basic_info', {}).get('birth_date'):
                                st.markdown(f"**出生年月：** {resume_data['basic_info']['birth_date']}")
                        with col_info2:
                            if resume_data.get('contact', {}).get('phone'):
                                st.markdown(f"**电话：** {resume_data['contact']['phone']}")
                            if resume_data.get('contact', {}).get('email'):
                                st.markdown(f"**邮箱：** {resume_data['contact']['email']}")
                            if resume_data.get('basic_info', {}).get('origin'):
                                st.markdown(f"**籍贯：** {resume_data['basic_info']['origin']}")
                        
                        # 教育背景
                        if resume_data.get('education'):
                            st.markdown("### 🎓 教育背景")
                            for edu in resume_data['education']:
                                edu_parts = []
                                if edu.get('school'):
                                    edu_parts.append(edu['school'])
                                if edu.get('degree'):
                                    edu_parts.append(edu['degree'])
                                if edu.get('major'):
                                    edu_parts.append(edu['major'])
                                if edu.get('graduation_year'):
                                    edu_parts.append(f"{edu['graduation_year']}年")
                                
                                st.markdown(f"**{' · '.join(edu_parts)}**")
                                if edu.get('gpa'):
                                    st.markdown(f"  成绩：{edu['gpa']}")
                        
                        # 技能
                        st.markdown("### 💼 技能")
                        if resume_data['skills']:
                            st.markdown(f"{', '.join(resume_data['skills'][:15])}")
                            if len(resume_data['skills']) > 15:
                                st.caption(f"等{len(resume_data['skills'])}项技能")
                        else:
                            st.caption("未提取到技能信息")
                        
                        # 项目经历
                        if resume_data.get('projects'):
                            st.markdown("### 🚀 项目经历")
                            for i, proj in enumerate(resume_data['projects'][:3], 1):
                                st.markdown(f"**{i}. {proj.get('name', '项目' + str(i))}**")
                                if proj.get('tech_stack'):
                                    st.caption(f"技术栈: {', '.join(proj['tech_stack'][:5])}")
                                if proj.get('responsibilities'):
                                    st.caption(f"{proj['responsibilities'][:100]}...")
                                elif proj.get('description'):
                                    st.caption(f"{proj['description'][:100]}...")
                        
                        # 工作经历
                        if resume_data.get('experience'):
                            st.markdown("### 💼 工作经历")
                            for exp in resume_data['experience'][:3]:
                                exp_parts = []
                                if exp.get('company'):
                                    exp_parts.append(exp['company'])
                                if exp.get('position'):
                                    exp_parts.append(exp['position'])
                                if exp.get('start_date') and exp.get('end_date'):
                                    exp_parts.append(f"({exp['start_date']} - {exp['end_date']})")
                                
                                st.markdown(f"**{' · '.join(exp_parts)}**")
                                if exp.get('description'):
                                    st.caption(f"{exp['description'][:100]}...")
                    
                    resume_text = resume_data['raw_text']
                    
                except Exception as e:
                    st.error(f"简历解析失败：{str(e)}")
                finally:
                    Path(tmp_path).unlink(missing_ok=True)
        else:
            resume_text = st.text_area(
                "粘贴简历内容",
                height=300,
                placeholder="姓名：张三\n应聘职位：Python后端工程师\n\n技能：Python, Django, Redis, MySQL..."
            )
        
        job_position = st.text_input("应聘职位", value="Python后端工程师")
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🚀 Triple Qwen</h3>
            <p style="margin-top: 1rem; font-size: 0.9rem;">
            ✅ Qwen-Decision<br/>
            ✅ Qwen-Question<br/>
            ✅ Qwen-Scorer<br/><br/>
            <strong>全Qwen架构</strong><br/>
            专业·智能·高效
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.resume_data:
            st.info(f"**👤 候选人：** {st.session_state.resume_data['name']}")
            if st.session_state.resume_data.get('education'):
                edu = st.session_state.resume_data['education'][0]
                if edu.get('school'):
                    st.info(f"**🎓 学校：** {edu['school'][:10]}...")
            st.success(f"**💼 技能：**\n\n" + "\n".join([f"• {s}" for s in st.session_state.resume_data['skills'][:5]]))
    
    if st.button("🚀 开始面试", type="primary", use_container_width=True):
        if resume_text.strip() or st.session_state.resume_data:
            if not st.session_state.resume_data and resume_text:
                # 使用ResumeParser解析文本
                parser = ResumeParser()
                try:
                    # 尝试完整解析
                    import tempfile
                    from pathlib import Path
                    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as tmp:
                        tmp.write(resume_text)
                        tmp_path = tmp.name
                    st.session_state.resume_data = parser.parse(tmp_path)
                    Path(tmp_path).unlink(missing_ok=True)
                except:
                    # 解析失败，使用简化版本
                    st.session_state.resume_data = {
                        'name': '候选人',
                        'skills': [s for s in ['Python', 'Java', 'JavaScript', 'Redis', 'MySQL'] if s.lower() in resume_text.lower()],
                        'projects': [],
                        'raw_text': resume_text
                    }
            
            # 初始化topic队列（从简历中提取）
            topic_queue = ["自我介绍"]  # 第一个topic固定为自我介绍
            
            # 从项目开始添加
            if st.session_state.resume_data.get('projects'):
                for proj in st.session_state.resume_data['projects']:
                    if proj.get('name'):
                        topic_queue.append(f"项目:{proj['name']}")
            
            # 然后添加核心技能
            if st.session_state.resume_data.get('skills'):
                core_skills = st.session_state.resume_data['skills'][:5]  # 最多5个核心技能
                for skill in core_skills:
                    topic_queue.append(f"技能:{skill}")
            
            st.session_state.topic_queue = topic_queue
            st.session_state.topic_index = 0
            st.session_state.current_topic = topic_queue[0]
            st.session_state.current_question = None
            st.session_state.job_position = job_position
            st.session_state.stage = 'interview'
            st.rerun()
        else:
            st.error("请先上传简历或粘贴文本")

# ==================== 阶段2: 面试过程 ====================
elif st.session_state.stage == 'interview':
    with st.spinner("正在加载Triple Qwen模型..."):
        models = load_triple_qwen_models()
    
    # 侧边栏 - 模型状态
    with st.sidebar:
        st.markdown("### 🔍 Triple Qwen状态")
        
        if models['decision_model']:
            st.success("✅ Qwen-Decision")
        else:
            st.error("❌ Qwen-Decision")
        
        if models['question_model']:
            st.success("✅ Qwen-Question")
        else:
            st.error("❌ Qwen-Question")
        
        if models['scorer_model']:
            st.success("✅ Qwen-Scorer")
        else:
            st.error("❌ Qwen-Scorer")
        
        st.markdown("---")
        
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            st.metric("显存占用", f"{memory_allocated:.2f} GB")
        
        # 显示话题队列
        st.markdown("---")
        st.markdown("### 📋 话题进度")
        
        for i, topic in enumerate(st.session_state.topic_queue):
            if i == st.session_state.topic_index:
                st.markdown(f"**➤ {topic}** ⬅️ 当前")
            elif i < st.session_state.topic_index:
                st.markdown(f"✅ ~~{topic}~~")
            else:
                st.markdown(f"⏳ {topic}")
        
        # 显示当前Decision的指导
        if st.session_state.current_guidance:
            st.markdown("---")
            st.markdown("### 🎯 Decision指导")
            
            if st.session_state.current_action:
                action_color = "🟢" if st.session_state.current_action == "FOLLOW_UP" else "🔵"
                st.markdown(f"**动作**: {action_color} {st.session_state.current_action}")
            
            with st.expander("📝 详细指导", expanded=True):
                st.markdown(st.session_state.current_guidance)
    
    # 生成第一个问题（如果还没有）
    if st.session_state.current_question is None:
        # 第一个问题固定为自我介绍
        st.session_state.current_question = "你好！首先请你简单介绍一下你自己，包括你的教育背景、主要技能和项目经验。"
        st.session_state.current_importance = 2  # 自我介绍重要度为2分（开场闲聊）
    
    # 顶部信息
    if st.session_state.resume_data:
        topic_progress = f"话题 {st.session_state.topic_index + 1}/{len(st.session_state.topic_queue)}: {st.session_state.current_topic}"
        st.caption(f"👤 {st.session_state.resume_data.get('name', '候选人')} | 应聘：{st.session_state.job_position} | 已完成：{st.session_state.total_rounds}轮 | {topic_progress}")
    
    # 主布局
    col_main, col_score = st.columns([2.5, 1])
    
    with col_main:
        # 虚拟形象
        st.markdown(f"""
        <div class="avatar-container">
            <div class="virtual-avatar">🤖</div>
            <div style="margin-top: 2rem; color: white;">
                <h3 style="margin: 0;">Alice · Triple Qwen</h3>
                <p style="margin: 0.5rem 0; opacity: 0.9;">📌 当前话题：{st.session_state.current_topic}</p>
                <p style="margin: 0.5rem 0; opacity: 0.9;">问题重要度：{'⭐' * st.session_state.current_importance} | 第{st.session_state.total_rounds + 1}轮</p>
            </div>
        </div>
        
        <div class="avatar-speech">
            <div style="font-size: 0.9rem; color: #999; margin-bottom: 0.5rem;">💬 面试官提问：</div>
            <div style="font-size: 1.1rem; line-height: 1.6;">{st.session_state.current_question}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # 生成并播放语音
        try:
            audio_file = st.session_state.digital_human.text_to_speech(st.session_state.current_question)
            if audio_file and Path(audio_file).exists():
                import base64
                import hashlib
                
                with open(audio_file, 'rb') as f:
                    audio_bytes = f.read()
                audio_base64 = base64.b64encode(audio_bytes).decode()
                
                question_hash = hashlib.md5(st.session_state.current_question.encode()).hexdigest()[:8]
                audio_id = f"tts_{question_hash}"
                
                if 'last_audio_id' not in st.session_state or st.session_state.last_audio_id != audio_id:
                    st.session_state.last_audio_id = audio_id
                    
                    st.components.v1.html(f"""
                    <audio id="{audio_id}" autoplay style="display:none;">
                        <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                    </audio>
                    """, height=0)
        except Exception as e:
            print(f"[WARNING] Audio generation failed: {e}")
        
        # 对话历史
        st.markdown("### 💬 对话记录")
        
        with st.container():
            st.markdown('<div style="max-height: 350px; overflow-y: auto; padding: 1rem; background: #f8f9fa; border-radius: 10px; margin-bottom: 1rem;">', unsafe_allow_html=True)
            
            if not st.session_state.qa_history:
                st.markdown('<div style="text-align: center; color: #999; padding: 2rem;">暂无对话记录</div>', unsafe_allow_html=True)
            else:
                for qa in st.session_state.qa_history[-5:]:
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
                    score_emoji = {'优秀': '🟢', '良好': '🔵', '一般': '🟡', '差': '🔴'}.get(qa.get('label_cn', '一般'), '⚪')
                    st.markdown(f'''
                    <div class="chat-message user">
                        <div class="message-bubble">
                            <div style="font-size: 0.85rem; opacity: 0.7;">👤 候选人</div>
                            <div style="margin-top: 0.3rem;">{qa["answer"]}</div>
                            <div style="font-size: 0.75rem; margin-top: 0.5rem; opacity: 0.8;">{score_emoji} {qa.get('label_cn', '一般')} ({qa.get('score', 70)}分)</div>
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # 回答输入
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
                    # 1. Qwen-Scorer评估
                    with st.spinner("🔍 Qwen-Scorer正在评估..."):
                        eval_result = answer_evaluate(models, st.session_state.current_question, answer)
                    
                    st.sidebar.markdown("### 📊 Scorer评估")
                    st.sidebar.write(f"**评分**: {eval_result['score']}分")
                    st.sidebar.write(f"**标签**: {eval_result['label_cn']}")
                    st.sidebar.write(f"**评价**: {eval_result['comment'][:100]}...")
                    
                    # 记录
                    st.session_state.qa_history.append({
                        'question': st.session_state.current_question,
                        'answer': answer,
                        'score': eval_result['score'],
                        'label': eval_result['label'],
                        'label_cn': eval_result['label_cn'],
                        'comment': eval_result['comment']
                    })
                    
                    st.session_state.total_rounds += 1
                    
                    # 2. Qwen-Decision决策
                    with st.spinner("🧠 Qwen-Decision正在决策..."):
                        scores = [qa['score'] for qa in st.session_state.qa_history]
                        
                        # 获取下一个topic
                        next_topic = None
                        if st.session_state.topic_index < len(st.session_state.topic_queue) - 1:
                            next_topic = st.session_state.topic_queue[st.session_state.topic_index + 1]
                        
                        decision_result = decision_make(
                            models,
                            st.session_state.resume_data,
                            st.session_state.qa_history,
                            scores,
                            st.session_state.current_topic,
                            next_topic
                        )
                    
                    # 保存Decision结果到session_state
                    st.session_state.current_action = decision_result['action']
                    st.session_state.current_guidance = decision_result['guidance']
                    
                    # 如果决策是SWITCH_TOPIC，切换到下一个topic
                    if decision_result['action'] == 'SWITCH_TOPIC':
                        # 从topic队列中取下一个
                        if st.session_state.topic_index < len(st.session_state.topic_queue) - 1:
                            st.session_state.topic_index += 1
                            st.session_state.current_topic = st.session_state.topic_queue[st.session_state.topic_index]
                        # guidance已经包含了topic信息，不需要再更新
                    
                    # 3. Qwen-Question生成问题
                    with st.spinner("❓ Qwen-Question正在生成..."):
                        question_result = question_generate(
                            models,
                            st.session_state.qa_history,
                            st.session_state.current_guidance,
                            st.session_state.current_topic
                        )
                    
                    # 更新状态
                    st.session_state.current_question = question_result['question']
                    st.session_state.current_importance = question_result['importance']
                    
                    st.rerun()
                else:
                    st.error("请输入回答")
        
        with col_e:
            if st.button("🏁 结束面试", use_container_width=True):
                if st.session_state.qa_history:
                    st.session_state.stage = 'summary'
                    st.rerun()
    
    with col_score:
        st.markdown("### 📊 实时评分")
        
        if st.session_state.qa_history:
            latest = st.session_state.qa_history[-1]
            
            st.markdown(f"""
            <div class="score-card">
                <div style="color: #999; font-size: 0.9rem;">当前回答</div>
                <div style="font-size: 2rem; font-weight: bold; color: #667eea;">{latest['label_cn']}</div>
                <div style="color: #999; font-size: 0.85rem;">{latest['score']}分</div>
            </div>
            
            <div class="score-card">
                <div style="color: #999; font-size: 0.9rem;">平均分</div>
                <div style="font-size: 2rem; font-weight: bold; color: #667eea;">{sum(qa['score'] for qa in st.session_state.qa_history) / len(st.session_state.qa_history):.1f}</div>
                <div style="color: #999; font-size: 0.85rem;">满分100</div>
            </div>
            
            <div class="score-card">
                <div style="color: #999; font-size: 0.9rem;">已完成轮数</div>
                <div style="font-size: 2rem; font-weight: bold; color: #667eea;">{st.session_state.total_rounds}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # 显示评价
            if latest.get('comment'):
                with st.expander("💬 AI评价"):
                    st.write(latest['comment'])
        else:
            st.info("回答问题后将显示评分")

# ==================== 阶段3: 总结 ====================
elif st.session_state.stage == 'summary':
    st.markdown("---")
    st.subheader("📈 面试总结报告")
    
    if st.session_state.qa_history:
        avg_score = sum(qa['score'] for qa in st.session_state.qa_history) / len(st.session_state.qa_history)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("平均评分", f"{avg_score:.1f}/100")
        with col2:
            st.metric("总轮数", st.session_state.total_rounds)
        with col3:
            excellent_count = sum(1 for qa in st.session_state.qa_history if qa['label'] == 'excellent')
            st.metric("优秀", f"{excellent_count}个")
        with col4:
            good_count = sum(1 for qa in st.session_state.qa_history if qa['label'] in ['excellent', 'good'])
            st.metric("良好以上", f"{good_count}个")
        
        st.markdown("---")
        
        if avg_score >= 85:
            st.success(f"### 🌟 强烈推荐\n候选人表现优秀，技术功底扎实。")
        elif avg_score >= 70:
            st.success(f"### 👍 推荐\n候选人具备相应技能，表现良好。")
        elif avg_score >= 50:
            st.warning(f"### 🤔 待定\n候选人基础一般，需进一步考察。")
        else:
            st.error(f"### ❌ 不推荐\n候选人技术能力不足。")
        
        st.markdown("---")
        st.markdown("### 📝 对话详情")
        
        for i, qa in enumerate(st.session_state.qa_history, 1):
            with st.expander(f"第{i}轮 - {qa['label_cn']} ({qa['score']}分)"):
                st.markdown(f"**Q:** {qa['question']}")
                st.markdown(f"**A:** {qa['answer']}")
                st.markdown(f"**评分:** {qa['score']}分 ({qa['label_cn']})")
                st.markdown(f"**AI评价:** {qa.get('comment', '无')}")
    
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
                'job_position': st.session_state.job_position,
                'avg_score': avg_score,
                'total_rounds': st.session_state.total_rounds,
                'qa_history': st.session_state.qa_history
            }
            
            st.download_button(
                "💾 下载报告",
                data=json.dumps(report, ensure_ascii=False, indent=2),
                file_name=f"interview_triple_qwen_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )

