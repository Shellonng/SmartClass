"""
测试 Qwen-1.8B-Chat 在面试场景下的表现
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

print("=" * 60)
print("Qwen2-1.5B-Instruct Interview Test")
print("=" * 60)

# 1. 加载模型 (使用Qwen2，无需额外依赖)
print("\n[1/5] Loading model...")
start_time = time.time()

MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16
)

load_time = time.time() - start_time
print(f"Model loaded: {MODEL_NAME}")
print(f"Loading time: {load_time:.2f}s")
print(f"GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

# 2. 测试场景
test_cases = [
    {
        "name": "场景1: 生成技术问题",
        "prompt": """你是一位资深Python工程师面试官。候选人简历显示精通Django和Redis。
请提出1个有深度的技术问题来考察候选人对Django的理解。
要求：
1. 问题要有技术深度
2. 能考察实际应用能力
3. 30字以内

问题："""
    },
    {
        "name": "场景2: 根据回答追问",
        "prompt": """你是面试官，刚才问了候选人："Django的ORM如何避免N+1查询问题？"

候选人回答："可以使用select_related和prefetch_related来优化查询。"

这个回答比较简单。请提出1个追问来深入考察候选人的理解："""
    },
    {
        "name": "场景3: 评价答案质量",
        "prompt": """你是面试官，问题是："解释Python的GIL全局解释器锁"

候选人回答："GIL是Python的一个互斥锁，同一时刻只允许一个线程执行Python字节码。这会影响多线程性能，但对IO密集型任务影响较小。可以用多进程或asyncio来绕过GIL的限制。"

请简短评价这个回答（10字以内）："""
    },
    {
        "name": "场景4: 生成开放性问题",
        "prompt": """候选人应聘后端开发岗位，简历显示有3年Python经验。

请提出1个开放性问题，考察候选人的系统设计能力："""
    },
    {
        "name": "场景5: 判断是否继续追问",
        "prompt": """面试官问："Redis的持久化方式有哪些？"
候选人回答："有RDB和AOF两种。"

这个回答是否需要继续追问？回答"是"或"否"并说明原因（20字以内）："""
    }
]

# 3. 测试每个场景
results = []

for i, test_case in enumerate(test_cases, 1):
    print(f"\n{'='*60}")
    print(f"[{i+1}/5] {test_case['name']}")
    print(f"{'='*60}")
    print(f"\n提示词:\n{test_case['prompt']}")
    
    # 生成回复 (使用Qwen2的chat template)
    start_time = time.time()
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": test_case['prompt']}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    
    response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip()
    
    gen_time = time.time() - start_time
    tokens = len(tokenizer.encode(response))
    speed = tokens / gen_time if gen_time > 0 else 0
    
    print(f"\nModel response:\n{response}")
    print(f"\nGeneration speed: {speed:.1f} tokens/s (time: {gen_time:.2f}s, {tokens} tokens)")
    
    results.append({
        "case": test_case['name'],
        "response": response,
        "time": gen_time,
        "tokens": tokens,
        "speed": speed
    })

# 4. 总结
print(f"\n{'='*60}")
print("测试总结")
print(f"{'='*60}")

avg_speed = sum(r['speed'] for r in results) / len(results)
total_time = sum(r['time'] for r in results)

print(f"\n模型加载时间: {load_time:.2f}s")
print(f"总生成时间: {total_time:.2f}s")
print(f"平均生成速度: {avg_speed:.1f} tokens/s")
print(f"显存峰值: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")

print("\n各场景表现:")
for r in results:
    print(f"  {r['case']}: {r['speed']:.1f} tokens/s")

print(f"\n{'='*60}")
print("测试完成！")
print(f"{'='*60}")

