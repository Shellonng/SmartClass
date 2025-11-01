# 🎓 模型微调指南

## 📊 两个模型的微调对比

### **总览**

| 维度 | BERT追问决策 | Qwen对话生成 |
|------|-------------|-------------|
| **任务** | 3分类 | 文本生成 |
| **方法** | 全量微调 | LoRA微调 |
| **数据量** | 100-500条 | 500-2000条 |
| **训练时间** | 10-30分钟 | 1-3小时 |
| **显存需求** | ~2GB | ~6GB |
| **难度** | ⭐⭐ | ⭐⭐⭐⭐ |
| **提升效果** | 显著（规则→智能） | 中等（优化风格） |

---

## 🔵 一、BERT追问决策模型微调

### **1.1 微调目标**

训练BERT识别3种情况，做出正确决策：

```
输入：问题 + 回答 + 特征
  ↓
BERT分类器
  ↓
输出：FOLLOW_UP / NEXT_TOPIC / END
```

### **1.2 数据集格式**

```json
{
  "question": "你用过Redis吗？",
  "answer": "不太了解",
  "context": {
    "follow_up_depth": 1,
    "hesitation_score": 0.8,
    "filler_count": 5,
    "answer_length": 15
  },
  "label": "NEXT_TOPIC"
}
```

**关键要素**：
- `question`: 当前问题
- `answer`: 候选人回答
- `context`: 多维度特征
  - `follow_up_depth`: 已追问次数（0-3）
  - `hesitation_score`: 犹豫度（0-1）
  - `filler_count`: 填充词数量
  - `answer_length`: 回答长度
- `label`: 标签（3选1）

### **1.3 数据标注指南**

#### **标注为 FOLLOW_UP（继续追问）**
✅ 候选人回答了且有内容  
✅ 回答长度 > 30字  
✅ 犹豫度 < 0.5  
✅ 追问深度 < 3  

**示例**：
```
Q: 你用过Redis吗？
A: 用过，我在项目中用Redis做缓存。
→ FOLLOW_UP（可以追问缓存了什么）
```

#### **标注为 NEXT_TOPIC（换话题）**
✅ 候选人说"不会/不了解/不清楚"  
✅ 回答长度 < 20字  
✅ 犹豫度 > 0.7  
✅ 或追问深度已达3层  

**示例**：
```
Q: 你用过Redis吗？
A: 不太了解
→ NEXT_TOPIC（应该换话题）
```

#### **标注为 END（结束）**
✅ 追问深度达到3层且回答质量好  
✅ 该话题已充分探讨  

**示例**：
```
Q: 你知道Redis的持久化机制吗？（第3次追问）
A: 知道，有RDB和AOF两种...（详细回答）
→ END（这个话题可以结束了）
```

### **1.4 数据准备建议**

#### **最小可行数据量**
```
FOLLOW_UP:   40条
NEXT_TOPIC:  40条
END:         20条
总计：       100条
```

#### **理想数据量**
```
FOLLOW_UP:   200条
NEXT_TOPIC:  200条
END:         100条
总计：       500条
```

#### **数据来源**
1. **真实面试记录**（最好）
   - 录制真实面试
   - 整理问答对
   - 人工标注决策

2. **模拟生成**
   - 让GPT-4生成面试对话
   - 人工审核和标注
   - 调整确保多样性

3. **混合方式**
   - 核心场景用真实数据
   - 边缘场景用生成数据

### **1.5 微调命令**

```bash
# 1. 准备数据（已完成，见 data/follow_up_training.json）

# 2. 运行训练
python scripts/train_follow_up_classifier.py

# 预期输出：
# Epoch 1/5: Val Acc: 0.75, F1: 0.73
# Epoch 2/5: Val Acc: 0.82, F1: 0.81
# Epoch 3/5: Val Acc: 0.88, F1: 0.87 ← 保存最佳
# Epoch 4/5: Val Acc: 0.87, F1: 0.86
# Epoch 5/5: Val Acc: 0.86, F1: 0.85
# 
# 最佳F1: 0.87
# 模型保存: ./checkpoints/follow_up_classifier
```

### **1.6 微调后使用**

```python
from models.follow_up_decision import FollowUpDecisionModel

# 加载微调后的模型
model = FollowUpDecisionModel(config)  # 会自动加载checkpoint

# 使用
decision, confidence = model.decide(
    question="你用过Redis吗？",
    answer="不太了解",
    speech_analysis={'hesitation_score': 0.8, 'filler_count': 5},
    context={'follow_up_depth': 1}
)

print(decision)    # "NEXT_TOPIC"
print(confidence)  # 0.92
```

---

## 🔴 二、Qwen对话生成模型微调

### **2.1 微调目标**

优化Qwen在面试场景的生成质量：

```
输入：Prompt（任务描述 + 上下文）
  ↓
Qwen-LoRA
  ↓
输出：自然、专业的面试官回复
```

### **2.2 数据集格式**

```json
{
  "task": "generate_follow_up",
  "context": {
    "job_title": "后端工程师",
    "current_question": "你用过Redis吗？",
    "current_topic": "缓存技术"
  },
  "user_answer": "用过，我在项目中用Redis做缓存。",
  "speech_analysis": {
    "hesitation_score": 0.2,
    "filler_count": 1
  },
  "expected_output": "很好！能具体说说你用Redis缓存了哪些数据吗？"
}
```

**关键要素**：
- `task`: 任务类型（generate_follow_up / generate_question / closing）
- `context`: 上下文信息
- `user_answer`: 用户回答（追问任务需要）
- `expected_output`: **期望的输出**（这是训练目标）

### **2.3 数据标注指南**

#### **追问（generate_follow_up）**

**好的示例**：
```
输入：
  问题："你用过Redis吗？"
  回答："用过，做缓存"

期望输出：
  "很好！能具体说说你缓存了哪些数据吗？"

✅ 自然过渡（"很好"）
✅ 针对回答内容追问（缓存了什么）
✅ 口语化（"能具体说说"）
```

**坏的示例**：
```
❌ "请详细阐述您在Redis缓存方面的技术实现方案"
   （太正式、不像真实对话）

❌ "Redis缓存了什么？"
   （太生硬、没有过渡）

❌ "你对Redis的底层数据结构了解吗？"
   （跳跃太大、不是追问）
```

#### **换话题（识别"不会"）**

**好的示例**：
```
输入：
  问题："你用过Redis吗？"
  回答："不太了解"

期望输出：
  "没关系，我们换个话题。你在项目中用过什么数据库？"

✅ 体贴的过渡（"没关系"）
✅ 明确换话题
✅ 引入新话题且贴近简历
```

#### **新问题（generate_question）**

**好的示例**：
```
输入：
  岗位："后端工程师"
  话题："Python"
  简历：["Python", "Django", "Flask"]

期望输出：
  "我看你简历上写了Django，能讲讲你用它做过什么项目吗？"

✅ 提到简历（真实感）
✅ 开放式问题
✅ 自然口语化
```

### **2.4 数据准备建议**

#### **最小可行数据量**
```
追问（FOLLOW_UP）:      200条
换话题（NEXT_TOPIC）:   100条
新问题（NEW_QUESTION）: 150条
结束语（CLOSING）:       50条
总计：                  500条
```

#### **理想数据量**
```
追问:      800条
换话题:    400条
新问题:    600条
结束语:    200条
总计：    2000条
```

#### **数据多样性要求**

1. **不同岗位**
   - 后端、前端、算法、数据...
   - 每个岗位至少50条

2. **不同技能点**
   - 编程语言、框架、工具、理论...
   - 覆盖简历常见技能

3. **不同回答质量**
   - 优秀回答（详细、专业）
   - 一般回答（简略）
   - 差的回答（不会、犹豫）

4. **不同追问深度**
   - 第1层：基础问题
   - 第2层：深入细节
   - 第3层：原理/难点

### **2.5 微调方法：LoRA vs 全量微调**

| 对比项 | LoRA微调 | 全量微调 |
|--------|---------|---------|
| **训练参数** | ~1% | 100% |
| **显存需求** | ~6GB | ~12GB+ |
| **训练时间** | 1-3小时 | 6-12小时 |
| **数据需求** | 500-2000条 | 5000+条 |
| **效果** | 90%的全量效果 | 100% |
| **推荐** | ✅ 推荐 | 不推荐（资源限制） |

**LoRA原理**：
```
原始模型（冻结）
    ↓
只训练小的"适配器"层
    ↓
推理时合并权重
```

### **2.6 微调命令**

```bash
# 1. 安装依赖
pip install peft

# 2. 准备数据（已完成，见 data/qwen_training.json）

# 3. 运行LoRA训练
python scripts/train_qwen_lora.py

# 预期输出：
# trainable params: 2,359,296 (只有1.5%参数可训练)
# Epoch 1/3: Loss: 2.34
# Epoch 2/3: Loss: 1.87
# Epoch 3/3: Loss: 1.52
# 
# 模型保存: ./checkpoints/qwen_interviewer
```

### **2.7 微调后使用**

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-1.5B-Instruct")

# 加载LoRA适配器
model = PeftModel.from_pretrained(
    base_model, 
    "./checkpoints/qwen_interviewer"
)

tokenizer = AutoTokenizer.from_pretrained("./checkpoints/qwen_interviewer")

# 正常使用（和之前一样）
model.generate(...)
```

---

## 📈 微调效果评估

### **BERT评估指标**

```python
# 混淆矩阵
            预测FOLLOW_UP  预测NEXT_TOPIC  预测END
真实FOLLOW_UP      35            3          2
真实NEXT_TOPIC      2           37          1
真实END             1            1         18

# 分类报告
              precision    recall  f1-score
FOLLOW_UP         0.92      0.88      0.90
NEXT_TOPIC        0.90      0.93      0.91
END               0.86      0.90      0.88

avg / total       0.90      0.90      0.90
```

**目标**：
- 整体准确率 > 85%
- F1-score > 0.85
- 每个类别recall > 0.80

### **Qwen评估方法**

由于是生成任务，难以量化评估，建议：

1. **人工评估**（10-20个测试样本）
   - 自然度（1-5分）
   - 相关性（1-5分）
   - 专业性（1-5分）

2. **A/B测试**
   - 微调前 vs 微调后
   - 盲测评分

3. **实际使用反馈**
   - 在面试系统中测试
   - 记录不合理的回复
   - 持续优化

---

## 🚀 微调流程（完整步骤）

### **阶段1：数据准备（1-2天）**

```bash
# 1. 收集面试对话
#    - 真实面试录音/文字
#    - 或使用GPT-4生成

# 2. 整理为标准格式
#    - BERT: follow_up_training.json
#    - Qwen: qwen_training.json

# 3. 数据质量检查
python scripts/check_data_quality.py
```

### **阶段2：BERT微调（30分钟）**

```bash
# 1. 训练
python scripts/train_follow_up_classifier.py

# 2. 评估
python scripts/evaluate_bert.py

# 3. 如果F1 > 0.85，继续；否则增加数据
```

### **阶段3：Qwen微调（2-3小时）**

```bash
# 1. LoRA训练
python scripts/train_qwen_lora.py

# 2. 人工测试
python scripts/test_qwen_generation.py

# 3. 调整数据和参数，重新训练
```

### **阶段4：集成测试（1天）**

```bash
# 1. 更新config.yaml，指向微调后的模型

# 2. 运行完整系统
streamlit run app.py

# 3. 测试各种场景
#    - 候选人说"不会"
#    - 候选人回答详细
#    - 候选人犹豫

# 4. 记录问题，优化数据
```

---

## 💡 常见问题

### Q1: 只微调一个模型可以吗？

**可以！优先级：**
1. **只微调BERT**（推荐）
   - 效果提升最明显
   - 决策准确 = 80%的提升
   - 时间短，容易验证

2. **只微调Qwen**
   - 效果提升有限
   - 需要大量高质量数据
   - 但生成更自然

### Q2: 数据量不够怎么办？

**方案**：
1. 使用GPT-4生成
   ```python
   prompt = "生成10个后端工程师面试对话，包括追问场景"
   ```

2. 数据增强
   - 同义替换
   - 改变提问方式
   - 调整回答长度

3. 少样本学习
   - 用更少的数据
   - 接受稍低的准确率

### Q3: 微调后效果不好怎么办？

**排查**：
1. 检查数据质量
   - 标注是否准确
   - 数据是否平衡
   - 是否有噪声

2. 调整超参数
   - 学习率 (1e-5 ~ 5e-5)
   - Epoch数 (3 ~ 10)
   - Batch size (4 ~ 16)

3. 增加数据量
   - 每个类别至少50条

---

## 📝 总结

| 模型 | 微调难度 | 数据需求 | 效果提升 | 推荐度 |
|------|---------|---------|---------|--------|
| BERT | ⭐⭐ 简单 | 100-500条 | ⭐⭐⭐⭐⭐ | ✅✅✅ 强烈推荐 |
| Qwen | ⭐⭐⭐⭐ 较难 | 500-2000条 | ⭐⭐⭐ | ⚠️ 可选 |

**建议**：
1. 先微调BERT，验证决策逻辑
2. 如果BERT效果好，再考虑微调Qwen
3. 如果资源有限，只微调BERT也足够

**预期效果**：
- 微调前：机械追问，不看回答
- 微调BERT后：智能决策，该换就换
- 都微调后：决策准确 + 生成自然

