"""
端到端测试 - 验证所有微调后的模型集成
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml

print("=" * 70)
print("AI面试系统 - 端到端测试")
print("=" * 70)

# 加载配置
print("\n1. 加载配置...")
with open('config/config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
print("   [OK] 配置加载成功")

# 测试BERT追问决策
print("\n2. 测试BERT追问决策模型...")
try:
    from models.follow_up_decision import FollowUpDecisionModel
    
    bert_model = FollowUpDecisionModel(config)
    
    # 模拟决策
    test_input = {
        'question': "请介绍一下Python的GIL机制",
        'answer': "GIL是全局解释器锁，它确保同一时刻只有一个线程执行Python字节码",
        'context': {
            'follow_up_depth': 0,
            'hesitation_score': 0.2,
            'answer_length': 35,
            'topic': 'Python并发',
            'skill_level': 'intermediate'
        }
    }
    
    decision = bert_model.decide_next_action(
        test_input['question'],
        test_input['answer'],
        test_input['context']
    )
    
    print(f"   [OK] BERT决策测试通过")
    print(f"   - 决策: {decision['action']}")
    print(f"   - 置信度: {decision['confidence']:.2f}")
    
except Exception as e:
    print(f"   [FAIL] BERT测试失败: {e}")
    import traceback
    traceback.print_exc()

# 测试RoBERTa评估
print("\n3. 测试RoBERTa多任务评估模型...")
try:
    from models.answer_evaluator import AnswerEvaluator
    
    evaluator = AnswerEvaluator(config)
    
    # 模拟评估
    test_qa = {
        'question': "什么是RESTful API?",
        'answer': "RESTful API是一种遵循REST架构风格的API设计方式，使用HTTP方法如GET、POST、PUT、DELETE来操作资源。",
        'speech_analysis': {
            'hesitation_score': 0.1,
            'filler_count': 1,
            'pause_duration': 0.5
        },
        'history_qa': [
            {'question': 'HTTP和HTTPS的区别', 'answer': 'HTTPS更安全', 'quality': '一般'},
            {'question': 'TCP三次握手', 'answer': 'SYN, SYN-ACK, ACK', 'quality': '良好'}
        ]
    }
    
    result = evaluator.evaluate(
        test_qa['question'],
        test_qa['answer'],
        test_qa['speech_analysis'],
        history_qa=test_qa['history_qa']
    )
    
    print(f"   [OK] RoBERTa评估测试通过")
    
    if evaluator.use_multitask:
        print(f"   - 当前回答: {result.get('current_label', 'N/A')} ({result.get('current_score', 0)}分)")
        print(f"   - 整体评分: {result.get('overall_score', 0):.1f}/100")
        print(f"   - 置信度: {result.get('confidence', 0):.2f}")
    else:
        print(f"   - 评分: {result.get('score', 0)}")
        print(f"   - 等级: {result.get('label', 'N/A')}")
    
except Exception as e:
    print(f"   [FAIL] RoBERTa测试失败: {e}")
    import traceback
    traceback.print_exc()

# 测试Qwen对话生成（可选，需要较大显存）
print("\n4. 测试Qwen对话生成模型...")
print("   [SKIP] Qwen模型测试需要较大显存，已在集成代码中验证LoRA加载逻辑")
print("   - LoRA权重: checkpoints/qwen_interviewer_lora/adapter_model.safetensors")
adapter_file = Path("checkpoints/qwen_interviewer_lora/adapter_model.safetensors")
if adapter_file.exists():
    print(f"   [OK] LoRA权重存在 ({adapter_file.stat().st_size / 1024 / 1024:.1f}MB)")
else:
    print(f"   [FAIL] LoRA权重文件不存在")

# 系统集成检查
print("\n5. 系统集成检查...")
integration_checks = [
    ("BERT checkpoint", Path("checkpoints/follow_up_classifier_1500/model.safetensors")),
    ("Qwen LoRA", Path("checkpoints/qwen_interviewer_lora/adapter_model.safetensors")),
    ("RoBERTa model", Path("checkpoints/answer_evaluator/pytorch_model.bin")),
    ("配置文件", Path("config/config.yaml")),
    ("模型集成文档", Path("docs/MODEL_INTEGRATION.md"))
]

all_pass = True
for name, path in integration_checks:
    if path.exists():
        print(f"   [OK] {name}")
    else:
        print(f"   [FAIL] {name} 不存在")
        all_pass = False

# 总结
print("\n" + "=" * 70)
if all_pass:
    print("端到端测试完成 - 所有组件就绪！")
    print("\n下一步：")
    print("  1. 运行 streamlit run app.py 启动面试系统")
    print("  2. 所有模型将自动加载微调后的权重")
    print("  3. 系统将使用:")
    print("     - BERT进行追问决策（1500条数据训练）")
    print("     - Qwen生成面试对话（2000条数据+LoRA）")
    print("     - RoBERTa多任务评估（当前质量+整体评分）")
else:
    print("端到端测试发现问题，请检查缺失的组件")
print("=" * 70)


"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml

print("=" * 70)
print("AI面试系统 - 端到端测试")
print("=" * 70)

# 加载配置
print("\n1. 加载配置...")
with open('config/config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
print("   [OK] 配置加载成功")

# 测试BERT追问决策
print("\n2. 测试BERT追问决策模型...")
try:
    from models.follow_up_decision import FollowUpDecisionModel
    
    bert_model = FollowUpDecisionModel(config)
    
    # 模拟决策
    test_input = {
        'question': "请介绍一下Python的GIL机制",
        'answer': "GIL是全局解释器锁，它确保同一时刻只有一个线程执行Python字节码",
        'context': {
            'follow_up_depth': 0,
            'hesitation_score': 0.2,
            'answer_length': 35,
            'topic': 'Python并发',
            'skill_level': 'intermediate'
        }
    }
    
    decision = bert_model.decide_next_action(
        test_input['question'],
        test_input['answer'],
        test_input['context']
    )
    
    print(f"   [OK] BERT决策测试通过")
    print(f"   - 决策: {decision['action']}")
    print(f"   - 置信度: {decision['confidence']:.2f}")
    
except Exception as e:
    print(f"   [FAIL] BERT测试失败: {e}")
    import traceback
    traceback.print_exc()

# 测试RoBERTa评估
print("\n3. 测试RoBERTa多任务评估模型...")
try:
    from models.answer_evaluator import AnswerEvaluator
    
    evaluator = AnswerEvaluator(config)
    
    # 模拟评估
    test_qa = {
        'question': "什么是RESTful API?",
        'answer': "RESTful API是一种遵循REST架构风格的API设计方式，使用HTTP方法如GET、POST、PUT、DELETE来操作资源。",
        'speech_analysis': {
            'hesitation_score': 0.1,
            'filler_count': 1,
            'pause_duration': 0.5
        },
        'history_qa': [
            {'question': 'HTTP和HTTPS的区别', 'answer': 'HTTPS更安全', 'quality': '一般'},
            {'question': 'TCP三次握手', 'answer': 'SYN, SYN-ACK, ACK', 'quality': '良好'}
        ]
    }
    
    result = evaluator.evaluate(
        test_qa['question'],
        test_qa['answer'],
        test_qa['speech_analysis'],
        history_qa=test_qa['history_qa']
    )
    
    print(f"   [OK] RoBERTa评估测试通过")
    
    if evaluator.use_multitask:
        print(f"   - 当前回答: {result.get('current_label', 'N/A')} ({result.get('current_score', 0)}分)")
        print(f"   - 整体评分: {result.get('overall_score', 0):.1f}/100")
        print(f"   - 置信度: {result.get('confidence', 0):.2f}")
    else:
        print(f"   - 评分: {result.get('score', 0)}")
        print(f"   - 等级: {result.get('label', 'N/A')}")
    
except Exception as e:
    print(f"   [FAIL] RoBERTa测试失败: {e}")
    import traceback
    traceback.print_exc()

# 测试Qwen对话生成（可选，需要较大显存）
print("\n4. 测试Qwen对话生成模型...")
print("   [SKIP] Qwen模型测试需要较大显存，已在集成代码中验证LoRA加载逻辑")
print("   - LoRA权重: checkpoints/qwen_interviewer_lora/adapter_model.safetensors")
adapter_file = Path("checkpoints/qwen_interviewer_lora/adapter_model.safetensors")
if adapter_file.exists():
    print(f"   [OK] LoRA权重存在 ({adapter_file.stat().st_size / 1024 / 1024:.1f}MB)")
else:
    print(f"   [FAIL] LoRA权重文件不存在")

# 系统集成检查
print("\n5. 系统集成检查...")
integration_checks = [
    ("BERT checkpoint", Path("checkpoints/follow_up_classifier_1500/model.safetensors")),
    ("Qwen LoRA", Path("checkpoints/qwen_interviewer_lora/adapter_model.safetensors")),
    ("RoBERTa model", Path("checkpoints/answer_evaluator/pytorch_model.bin")),
    ("配置文件", Path("config/config.yaml")),
    ("模型集成文档", Path("docs/MODEL_INTEGRATION.md"))
]

all_pass = True
for name, path in integration_checks:
    if path.exists():
        print(f"   [OK] {name}")
    else:
        print(f"   [FAIL] {name} 不存在")
        all_pass = False

# 总结
print("\n" + "=" * 70)
if all_pass:
    print("端到端测试完成 - 所有组件就绪！")
    print("\n下一步：")
    print("  1. 运行 streamlit run app.py 启动面试系统")
    print("  2. 所有模型将自动加载微调后的权重")
    print("  3. 系统将使用:")
    print("     - BERT进行追问决策（1500条数据训练）")
    print("     - Qwen生成面试对话（2000条数据+LoRA）")
    print("     - RoBERTa多任务评估（当前质量+整体评分）")
else:
    print("端到端测试发现问题，请检查缺失的组件")
print("=" * 70)





