"""
测试双Qwen模型推理
同时加载两个LoRA，演示完整的面试流程
"""

import torch
import sys
import io
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

class DualQwenInterviewer:
    """双Qwen面试系统"""
    
    def __init__(self, 
                 base_model="Qwen/Qwen2-1.5B-Instruct",
                 decision_lora="checkpoints/qwen_decision_lora",
                 question_lora="checkpoints/qwen_question_lora",
                 use_4bit=True):
        """
        初始化双Qwen系统
        
        参数:
            base_model: 基座模型
            decision_lora: 决策LoRA路径
            question_lora: 提问LoRA路径
            use_4bit: 是否使用4bit量化（推理时推荐）
        """
        print("="*60)
        print("🚀 初始化双Qwen面试系统")
        print("="*60)
        
        # 配置
        self.use_4bit = use_4bit
        
        # 加载分词器
        print(f"\n📥 加载分词器...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载基座模型
        print(f"\n📥 加载基座模型: {base_model}")
        if use_4bit:
            print(f"   使用4bit量化（节省显存）")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            self.base_model = AutoModelForCausalLM.from_pretrained(
                base_model,
                quantization_config=bnb_config,
                device_map="auto"
            )
        else:
            self.base_model = AutoModelForCausalLM.from_pretrained(
                base_model,
                device_map="auto",
                torch_dtype=torch.float16
            )
        
        # 加载两个LoRA（同时加载到同一个基座上）
        print(f"\n📥 加载Qwen-Decision LoRA...")
        self.decision_model = PeftModel.from_pretrained(
            self.base_model,
            decision_lora
        )
        
        print(f"\n📥 加载Qwen-Question LoRA...")
        self.question_model = PeftModel.from_pretrained(
            self.base_model,
            question_lora
        )
        
        print(f"\n✅ 双Qwen系统初始化完成！")
        
        # 显示显存占用
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"\n💾 显存占用:")
            print(f"   已分配: {memory_allocated:.2f} GB")
            print(f"   已预留: {memory_reserved:.2f} GB")
    
    def make_decision(self, topic, round_number, history, scores):
        """
        使用Qwen-Decision做出决策
        
        返回: (action, guidance)
        """
        # 构建输入
        history_text = ""
        recent_history = history[-3:] if len(history) > 3 else history
        for h in recent_history:
            history_text += f"问: {h['question']}\n答: {h['answer']}\n评分: {h['score']}分\n\n"
        
        avg_score = sum(scores) / len(scores) if scores else 0
        recent_trend = "stable"  # 简化
        
        input_text = f"""当前话题: {topic}
当前轮次: 第{round_number}轮

对话历史:
{history_text.strip()}

评分情况:
平均分: {avg_score:.0f}分
分数趋势: {recent_trend}
最近3次: {scores[-3:]}

请根据以上信息，做出面试决策并提供指导建议。"""
        
        instruction = "你是一位经验丰富的技术面试官。你的任务是根据当前面试话题、对话历史和候选人的表现评分，做出面试决策（FOLLOW_UP继续深入 或 SWITCH_TOPIC切换话题），并为问题生成器提供详细的指导建议。"
        
        # 格式化prompt
        prompt = f"""<|im_start|>system
{instruction}<|im_end|>
<|im_start|>user
{input_text}<|im_end|>
<|im_start|>assistant
"""
        
        # 生成
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.decision_model.device)
        
        with torch.no_grad():
            outputs = self.decision_model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                top_p=0.9
            )
        
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取决策和指导
        assistant_response = result.split("<|im_start|>assistant")[-1].strip()
        
        action = "SWITCH_TOPIC"  # 默认
        guidance = assistant_response
        
        if "决策:" in assistant_response:
            parts = assistant_response.split("指导建议:")
            action_part = parts[0].replace("决策:", "").strip()
            if "FOLLOW_UP" in action_part:
                action = "FOLLOW_UP"
            elif "SWITCH_TOPIC" in action_part:
                action = "SWITCH_TOPIC"
            
            if len(parts) > 1:
                guidance = parts[1].strip()
        
        return action, guidance
    
    def generate_question(self, topic, history, guidance):
        """
        使用Qwen-Question生成问题
        
        返回: (question, importance)
        """
        # 构建输入
        history_text = ""
        for h in history:
            history_text += f"Q: {h['question']}\nA: {h['answer']}\n\n"
        
        input_text = f"""面试话题: {topic}

完整对话历史:
{history_text.strip() if history_text else '（这是第一个问题）'}

决策指导:
{guidance}

请根据以上信息生成下一个面试问题，并评估其重要程度（1-5分）。"""
        
        instruction = "你是一位经验丰富的技术面试官。你的任务是根据面试话题、完整对话历史和决策指导，生成下一个合适的面试问题，并评估该问题的重要程度（1-5分，其中1分为闲聊，5分为核心技能考察）。"
        
        # 格式化prompt
        prompt = f"""<|im_start|>system
{instruction}<|im_end|>
<|im_start|>user
{input_text}<|im_end|>
<|im_start|>assistant
"""
        
        # 生成
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.question_model.device)
        
        with torch.no_grad():
            outputs = self.question_model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.8,
                do_sample=True,
                top_p=0.95
            )
        
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取问题和重要程度
        assistant_response = result.split("<|im_start|>assistant")[-1].strip()
        
        question = assistant_response
        importance = 3  # 默认
        
        if "问题:" in assistant_response and "重要程度:" in assistant_response:
            parts = assistant_response.split("重要程度:")
            question = parts[0].replace("问题:", "").strip()
            
            if len(parts) > 1:
                importance_str = parts[1].strip().split("分")[0].strip()
                try:
                    importance = int(importance_str)
                except:
                    importance = 3
        
        return question, importance

def test_interview_flow():
    """测试完整面试流程"""
    print("\n" + "="*60)
    print("🧪 测试双Qwen面试流程")
    print("="*60)
    
    # 初始化系统
    interviewer = DualQwenInterviewer(use_4bit=True)
    
    # 模拟面试场景
    topic = "Spring Boot框架"
    history = [
        {
            "question": "请介绍一下Spring Boot的核心特性",
            "answer": "Spring Boot主要提供了自动配置、起步依赖、内嵌服务器等特性，可以快速构建Spring应用。",
            "score": 75
        },
        {
            "question": "Spring Boot的自动配置原理是什么？",
            "answer": "嗯...我知道是通过注解实现的，但具体原理不太清楚。",
            "score": 60
        }
    ]
    scores = [75, 60]
    
    print(f"\n{'='*60}")
    print("📝 测试场景")
    print(f"{'='*60}")
    print(f"话题: {topic}")
    print(f"轮次: 3")
    print(f"历史评分: {scores}")
    print(f"对话历史: {len(history)} 轮")
    
    # 步骤1: 决策
    print(f"\n{'='*60}")
    print("步骤1: Qwen-Decision 做出决策")
    print(f"{'='*60}")
    
    action, guidance = interviewer.make_decision(topic, 3, history, scores)
    
    print(f"\n决策结果:")
    print(f"  Action: {action}")
    print(f"  Guidance: {guidance[:200]}...")
    
    # 步骤2: 生成问题
    print(f"\n{'='*60}")
    print("步骤2: Qwen-Question 生成问题")
    print(f"{'='*60}")
    
    question, importance = interviewer.generate_question(topic, history, guidance)
    
    print(f"\n生成结果:")
    print(f"  问题: {question}")
    print(f"  重要程度: {importance}分")
    
    # 总结
    print(f"\n{'='*60}")
    print("✅ 测试完成！")
    print(f"{'='*60}")
    
    print(f"\n完整流程:")
    print(f"  1. 输入: 话题、历史、评分")
    print(f"  2. Qwen-Decision → {action} + 指导建议")
    print(f"  3. Qwen-Question → 新问题 + 重要程度{importance}分")
    
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**3
        print(f"\n💾 推理显存占用: {memory_allocated:.2f} GB")

if __name__ == "__main__":
    test_interview_flow()

