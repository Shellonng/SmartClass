"""
AI Interview Coach - Main Application
主应用入口（Streamlit界面）
"""

import streamlit as st
import json
from pathlib import Path
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from utils.config_loader import load_config
from utils.logger import setup_logger
from models.resume_parser import ResumeParser
from models.simple_rag import SimpleRAG
from models.dialogue_manager import DialogueManager
from models.lightweight_interviewer import LightweightInterviewer
from models.follow_up_decision import FollowUpDecisionModel
from models.answer_evaluator import AnswerEvaluator
from models.speech_processor import SpeechProcessor

logger = setup_logger()

# 页面配置
st.set_page_config(
    page_title="AI Interview Coach",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #555;
        margin-bottom: 1rem;
    }
    .interview-question {
        background-color: #f0f8ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .candidate-answer {
        background-color: #f5f5f5;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #2ca02c;
        margin: 1rem 0;
    }
    .evaluation-box {
        background-color: #fff9e6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #ff7f0e;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def load_job_database():
    """加载岗位数据库"""
    with open('data/job_database.json', 'r', encoding='utf-8') as f:
        return json.load(f)


@st.cache_resource
def initialize_models():
    """初始化所有模型（缓存避免重复加载）"""
    logger.info("初始化模型...")
    
    config = load_config()
    
    models = {
        'config': config,
        'resume_parser': ResumeParser(config),
        'rag': SimpleRAG(config),
    }
    
    # 初始化LLM面试官模型
    try:
        logger.info("正在加载LLM模型...")
        models['interviewer'] = LightweightInterviewer(config)
        logger.info("LLM模型加载完成")
    except Exception as e:
        logger.warning(f"LLM模型加载失败: {e}，将使用RAG回退模式")
        models['interviewer'] = None
    
    logger.info("所有模型初始化完成")
    
    return models


def init_session_state():
    """初始化session state"""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.interview_started = False
        st.session_state.resume_data = None
        st.session_state.job_title = None
        st.session_state.conversation_history = []
        st.session_state.current_question = None
        st.session_state.dialogue_manager = None
        st.session_state.models_loaded = False


def main():
    """主函数"""
    init_session_state()
    
    # 标题
    st.markdown('<div class="main-header">🎯 AI Interview Coach</div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666;">基于Transformer的智能面试官系统</p>', unsafe_allow_html=True)
    
    # 侧边栏
    with st.sidebar:
        st.header("⚙️ 设置")
        
        # 加载岗位列表
        job_db = load_job_database()
        job_titles = [job['job_title'] for job in job_db]
        
        selected_job = st.selectbox(
            "选择目标岗位",
            options=job_titles,
            help="选择你要面试的岗位"
        )
        
        # 显示岗位信息
        selected_job_info = next(job for job in job_db if job['job_title'] == selected_job)
        with st.expander("岗位详情"):
            st.write(f"**描述**: {selected_job_info['description']}")
            st.write(f"**核心技能**: {', '.join(selected_job_info['core_skills'])}")
        
        st.divider()
        
        # 简历上传
        st.subheader("📄 上传简历")
        uploaded_file = st.file_uploader(
            "支持PDF和DOCX格式",
            type=['pdf', 'docx'],
            help="上传你的简历，系统将自动解析"
        )
        
        if uploaded_file is not None:
            # 保存临时文件
            temp_path = Path(f"data/temp/{uploaded_file.name}")
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(temp_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            
            # 解析简历
            if st.button("🔍 解析简历", key="parse_resume"):
                with st.spinner("正在解析简历..."):
                    models = initialize_models()
                    resume_data = models['resume_parser'].parse(str(temp_path))
                    st.session_state.resume_data = resume_data
                    st.success("✅ 简历解析完成！")
        
        # 显示简历信息
        if st.session_state.resume_data:
            with st.expander("简历信息", expanded=True):
                resume = st.session_state.resume_data
                st.write(f"**姓名**: {resume['name']}")
                if resume['contact']:
                    st.write(f"**联系方式**: {resume['contact']}")
                if resume['skills']:
                    st.write(f"**技能**: {', '.join(resume['skills'][:10])}")
        
        st.divider()
        
        # 开始面试按钮
        if st.session_state.resume_data and not st.session_state.interview_started:
            if st.button("🚀 开始面试", type="primary", use_container_width=True):
                with st.spinner("正在加载面试模型..."):
                    st.session_state.job_title = selected_job
                    st.session_state.interview_started = True
                    st.rerun()
        
        # 结束面试按钮
        if st.session_state.interview_started:
            if st.button("⏹️ 结束面试", type="secondary", use_container_width=True):
                st.session_state.interview_started = False
                st.session_state.conversation_history = []
                st.session_state.current_question = None
                st.rerun()
    
    # 主内容区
    if not st.session_state.interview_started:
        # 欢迎页面
        show_welcome_page()
    else:
        # 面试页面
        show_interview_page()


def show_welcome_page():
    """显示欢迎页面"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📝 1. 准备简历")
        st.write("上传你的简历（PDF或DOCX格式），系统将自动提取技能信息")
    
    with col2:
        st.markdown("### 🎯 2. 选择岗位")
        st.write("选择你要面试的目标岗位，系统会针对性提问")
    
    with col3:
        st.markdown("### 💬 3. 开始面试")
        st.write("通过文字或语音回答问题，获得实时评估和反馈")
    
    st.divider()
    
    # 功能介绍
    st.markdown("### ✨ 核心功能")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🤖 智能面试官**
        - 基于Transformer的大语言模型
        - 自然流畅的对话交互
        - 针对简历和岗位的个性化提问
        
        **🎤 语音分析**
        - 实时语音识别（Whisper）
        - 填充词检测（嗯、呃、额等）
        - 犹豫程度分析
        """)
    
    with col2:
        st.markdown("""
        **🔄 智能追问**
        - 根据回答质量动态追问
        - 深入挖掘技术理解
        - 自适应难度调整
        
        **📊 专业评估**
        - 多维度回答评估
        - 实时反馈和建议
        - 面试总结报告
        """)
    
    st.divider()
    
    # 使用提示
    st.info("💡 **提示**: 请在左侧上传简历并选择岗位，然后点击「开始面试」按钮")


def show_interview_page():
    """显示面试页面"""
    st.markdown('<div class="sub-header">🎤 面试进行中...</div>', unsafe_allow_html=True)
    
    # 初始化对话管理器（如果还没有）
    if st.session_state.dialogue_manager is None:
        models = initialize_models()
        config = models['config']
        
        st.session_state.dialogue_manager = DialogueManager(
            config,
            st.session_state.job_title,
            st.session_state.resume_data
        )
        
        # 生成开场白
        opening = f"""你好！我是{st.session_state.job_title}的面试官。

我看过你的简历了，你的技能背景是：{', '.join(st.session_state.resume_data['skills'][:5])}。

接下来我会针对你的简历和岗位要求提一些问题，请放轻松，展现你的真实水平就好。准备好了吗？"""
        
        st.session_state.conversation_history.append({
            'role': 'interviewer',
            'content': opening
        })
    
    # 显示对话历史
    for msg in st.session_state.conversation_history:
        if msg['role'] == 'interviewer':
            st.markdown(f'<div class="interview-question"><strong>面试官:</strong><br>{msg["content"]}</div>', 
                       unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="candidate-answer"><strong>你:</strong><br>{msg["content"]}</div>', 
                       unsafe_allow_html=True)
            
            # 显示评估结果
            if 'evaluation' in msg:
                eval_result = msg['evaluation']
                st.markdown(f"""
                <div class="evaluation-box">
                    <strong>📊 评估结果:</strong><br>
                    得分: <strong>{eval_result['score']}</strong>分 ({eval_result['label']})<br>
                    反馈: {eval_result['feedback']}
                </div>
                """, unsafe_allow_html=True)
    
    # 输入区域
    st.divider()
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_answer = st.text_area(
            "你的回答:",
            height=100,
            placeholder="在这里输入你的回答...",
            key="answer_input"
        )
    
    with col2:
        st.write("")  # 占位
        st.write("")  # 占位
        submit_button = st.button("✅ 提交回答", type="primary", use_container_width=True)
    
    # 处理提交
    if submit_button and user_answer.strip():
        with st.spinner("AI正在思考中..."):
            try:
                models = initialize_models()
                
                # 1. 评估回答质量（多维度智能评估）
                with st.spinner("评估回答质量..."):
                    # 获取最后一个问题
                    last_question = st.session_state.conversation_history[-1]['content'] if st.session_state.conversation_history else ""
                    
                    # 多维度评估
                    score = 60  # 基础分
                    feedback_parts = []
                    
                    # 维度1: 长度评估（20分）
                    answer_length = len(user_answer)
                    if answer_length < 20:
                        score -= 15
                        feedback_parts.append("回答过于简短")
                    elif answer_length < 50:
                        score -= 5
                        feedback_parts.append("可以更详细")
                    elif answer_length < 100:
                        score += 5
                        feedback_parts.append("长度适中")
                    else:
                        score += 10
                        feedback_parts.append("回答详细")
                    
                    # 维度2: 技术词汇评估（30分）
                    tech_keywords = ['项目', '实现', '使用', '开发', '设计', '优化', '问题', '解决',
                                   'Python', 'Java', 'Django', 'Flask', 'Redis', 'MySQL', 
                                   '数据库', '算法', '框架', '接口', 'API', '性能']
                    tech_count = sum(1 for word in tech_keywords if word in user_answer)
                    
                    if tech_count >= 5:
                        score += 15
                        feedback_parts.append("技术深度好")
                    elif tech_count >= 3:
                        score += 10
                        feedback_parts.append("技术点清晰")
                    elif tech_count >= 1:
                        score += 5
                        feedback_parts.append("有技术内容")
                    else:
                        feedback_parts.append("需要更多技术细节")
                    
                    # 维度3: 结构评估（20分）
                    structure_markers = ['首先', '其次', '然后', '最后', '第一', '第二', 
                                       '因为', '所以', '例如', '比如', '具体来说']
                    structure_count = sum(1 for marker in structure_markers if marker in user_answer)
                    
                    if structure_count >= 3:
                        score += 10
                        feedback_parts.append("逻辑清晰")
                    elif structure_count >= 1:
                        score += 5
                        feedback_parts.append("有条理")
                    
                    # 确保分数在合理范围
                    score = max(30, min(100, score))
                    
                    # 确定等级
                    if score >= 85:
                        label = "优秀"
                    elif score >= 70:
                        label = "良好"
                    elif score >= 55:
                        label = "一般"
                    else:
                        label = "较差"
                    
                    # 生成反馈
                    feedback = "、".join(feedback_parts) + "。"
                    if score >= 85:
                        feedback += " 继续保持！"
                    elif score >= 70:
                        feedback += " 不错！"
                    else:
                        feedback += " 可以更充实一些。"
                
                # 保存候选人回答和评估
                st.session_state.conversation_history.append({
                    'role': 'candidate',
                    'content': user_answer,
                    'evaluation': {
                        'score': score,
                        'label': label,
                        'feedback': feedback
                    }
                })
                
                # 2. 生成下一个问题（使用LLM）
                with st.spinner("生成下一个问题..."):
                    # 使用LightweightInterviewer生成问题
                    interviewer = models.get('interviewer')
                    if interviewer:
                        # 构建上下文
                        context = {
                            'job_title': st.session_state.job_title,
                            'skills': st.session_state.resume_data.get('skills', []),
                            'last_question': last_question,
                            'last_answer': user_answer,
                            'conversation_count': len([m for m in st.session_state.conversation_history if m['role'] == 'interviewer'])
                        }
                        
                        # 判断是追问还是新话题（智能决策）
                        # 检测用户是否表示"不会"
                        negative_keywords = ['不会', '不了解', '不熟悉', '不清楚', '不知道', '没学过', '没用过', '不懂']
                        user_said_no = any(keyword in user_answer for keyword in negative_keywords)
                        
                        # 检测回答质量（如果分数太低，说明答不上来）
                        answer_too_poor = score < 55
                        
                        # 检测回答是否敷衍
                        evasive_keywords = ['ai写', '忽略', '换个', '不想', '别问']
                        user_is_evasive = any(keyword in user_answer for keyword in evasive_keywords)
                        
                        # 决策逻辑
                        should_change_topic = False
                        if user_said_no or answer_too_poor or user_is_evasive:
                            # 情况1: 用户明确说不会/答不好/敷衍 → 换话题
                            action_type = "NEW_TOPIC"
                            should_change_topic = True
                        elif len([m for m in st.session_state.conversation_history if m['role'] == 'interviewer']) < 2:
                            # 情况2: 前两轮 → 新话题
                            action_type = "NEW_TOPIC"
                        else:
                            # 情况3: 用户答得还可以 → 可以追问
                            action_type = "START_FOLLOW_UP"
                        
                        # 选择话题
                        if should_change_topic:
                            # 从简历技能中随机选一个新话题
                            import random
                            available_skills = context.get('skills', ['项目经验', '技术栈'])
                            current_topic = random.choice(available_skills) if available_skills else '项目经验'
                        else:
                            current_topic = context.get('skills', ['技术'])[0] if context.get('skills') else '技术'
                        
                        # 添加必要的上下文字段
                        full_context = {
                            **context,
                            'current_topic': current_topic,
                            'current_question': last_question,
                            'follow_up_depth': 1,
                            'max_follow_up': 3,
                            'resume_skills': context.get('skills', [])
                        }
                        
                        # 调用LLM生成
                        next_question = interviewer.generate_response(
                            action_type=action_type,
                            context=full_context,
                            user_answer=user_answer,
                            speech_analysis=None,
                            rag_questions=None
                        )
                        
                        # 清理输出
                        next_question = next_question.strip()
                        if not next_question or len(next_question) < 5:
                            next_question = "请详细说明一下你在项目中遇到的技术难点。"
                    else:
                        # 回退到RAG
                        rag_results = models['rag'].search(
                            query=st.session_state.resume_data['skills'][0] if st.session_state.resume_data['skills'] else "技术",
                            job_title=st.session_state.job_title,
                            top_k=1
                        )
                        
                        if rag_results:
                            next_question = rag_results[0]['question']
                        else:
                            next_question = "请介绍一下你的项目经验。"
                
                # 保存面试官问题
                st.session_state.conversation_history.append({
                    'role': 'interviewer',
                    'content': next_question
                })
                
                st.rerun()
                
            except Exception as e:
                st.error(f"处理回答时出错: {str(e)}")
                logger.error(f"处理回答错误: {e}", exc_info=True)
    
    # 提示信息
    if not user_answer or not user_answer.strip():
        st.info("💬 请在上方输入框中输入你的回答，然后点击提交")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"应用错误: {str(e)}", exc_info=True)
        st.error(f"❌ 发生错误: {str(e)}")
        st.info("请检查日志文件获取详细信息")

