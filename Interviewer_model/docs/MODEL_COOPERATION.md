# 🤝 两个模型的协作机制

## 📌 你提出的关键问题

### **问题1：NEXT_TOPIC有两种情况**

换话题的原因有本质区别：

| 类型 | 原因 | 候选人表现 | 面试官应该说 |
|------|------|-----------|------------|
| **消极换话题** | 候选人答不上来 | "不会"、犹豫、回答短 | "没关系，我们换个话题..." |
| **积极换话题** | 已经问够了 | 回答详细、已追问3层 | "很好！XX这块你掌握得不错..." |

**相同决策，不同话术**！

### **问题2：两个模型的数据必须呼应**

**是的！** 数据集必须呼应，否则：
- BERT决定换话题
- Qwen不知道为什么换，可能生成不合适的回复

---

## 🔄 两个模型的协作流程

### **完整链路**

```
候选人回答
    ↓
┌─────────────────────────────────┐
│ Step 1: BERT决策                │
│                                  │
│ 输入：问题 + 回答 + 特征         │
│   - follow_up_depth: 3          │
│   - hesitation_score: 0.15      │
│   - answer_length: 120          │
│                                  │
│ 输出：NEXT_TOPIC                 │
│ （但不知道是积极还是消极）        │
└──────────┬──────────────────────┘
           ↓
┌─────────────────────────────────┐
│ Step 2: Qwen生成                │
│                                  │
│ 接收：                           │
│   - action_type="NEXT_TOPIC"    │
│   - 用户回答内容（全文）         │
│   - 犹豫度、填充词等特征         │
│   - 追问深度                     │
│                                  │
│ Qwen需要从这些信息推断：         │
│   如果回答包含"不会" → 消极换话题 │
│   如果犹豫度>0.7 → 消极换话题     │
│   如果depth=3且回答好 → 积极换话题│
│                                  │
│ 输出：相应的回复                 │
└─────────────────────────────────┘
```

---

## 📊 数据集呼应示例

### **示例1：消极换话题**

#### BERT训练数据
```json
{
  "question": "你对Docker了解吗？",
  "answer": "了解一些基础概念，但是没有实际使用经验。",
  "context": {
    "follow_up_depth": 1,
    "hesitation_score": 0.4,
    "filler_count": 3,
    "answer_length": 25
  },
  "label": "NEXT_TOPIC",  // ← 决策
  "reason": "候选人只知道概念没有实践，不适合深入追问",
  "reason_type": "negative"  // ← 明确是消极换话题
}
```

#### Qwen训练数据（对应）
```json
{
  "task": "generate_follow_up",
  "bert_decision": "NEXT_TOPIC",  // ← 和BERT决策对应
  "reason_type": "negative",      // ← 和BERT原因对应
  "context": {
    "current_question": "你对Docker了解吗？",
    "follow_up_depth": 1
  },
  "user_answer": "了解一些基础概念，但是没有实际使用经验。",
  "speech_analysis": {
    "hesitation_score": 0.4,     // ← 和BERT特征对应
    "filler_count": 3
  },
  "expected_output": "好的，那我们聊聊你有实践经验的技术吧。",
  // ↑ 消极换话题的话术：体贴、不纠缠
}
```

### **示例2：积极换话题**

#### BERT训练数据
```json
{
  "question": "reset和revert的区别你能详细说说吗？",
  "answer": "reset是回退到某个提交，会改变历史记录...（详细回答140字）",
  "context": {
    "follow_up_depth": 3,        // ← 已经3层
    "hesitation_score": 0.1,     // ← 回答流畅
    "filler_count": 0,
    "answer_length": 140         // ← 回答详细
  },
  "label": "NEXT_TOPIC",  // ← 决策相同
  "reason": "已经追问3层，候选人对Git掌握很扎实，应该换话题",
  "reason_type": "positive"  // ← 但原因不同！
}
```

#### Qwen训练数据（对应）
```json
{
  "task": "generate_follow_up",
  "bert_decision": "NEXT_TOPIC",  // ← 和BERT决策对应
  "reason_type": "positive",      // ← 和BERT原因对应
  "context": {
    "current_question": "reset和revert的区别你能详细说说吗？",
    "follow_up_depth": 3          // ← 关键信号
  },
  "user_answer": "reset是回退到某个提交...（详细）",
  "speech_analysis": {
    "hesitation_score": 0.1,      // ← 关键信号
    "filler_count": 0
  },
  "expected_output": "非常好！Git这块你掌握得很扎实。我们聊聊其他方面...",
  // ↑ 积极换话题的话术：充分肯定 + 自然过渡
}
```

---

## 🎯 数据集呼应的关键点

### **1. 特征对齐**

BERT和Qwen看到的**特征必须一致**：

```python
# BERT看到的
context = {
    "follow_up_depth": 3,
    "hesitation_score": 0.15,
    "filler_count": 1,
    "answer_length": 120
}

# Qwen看到的（必须相同）
speech_analysis = {
    "hesitation_score": 0.15,  # ← 相同
    "filler_count": 1          # ← 相同
}
context = {
    "follow_up_depth": 3       # ← 相同
}
```

### **2. 决策对应**

每个BERT的训练样本，都应该有对应的Qwen样本：

```
BERT数据集（20条）
  ├─ FOLLOW_UP: 8条      → Qwen有8条对应的"追问"样本
  ├─ NEXT_TOPIC: 10条
  │    ├─ negative: 6条  → Qwen有6条"消极换话题"样本
  │    └─ positive: 4条  → Qwen有4条"积极换话题"样本
  └─ END: 2条            → Qwen有2条对应的"结束"样本
```

### **3. 上下文一致**

训练数据应该模拟真实流程：

```json
// BERT训练数据（ID 3）
{
  "question": "那你遇到过缓存穿透吗？",  // ← 这是追问的第3层
  "answer": "遇到过，我用了布隆过滤器...",
  "context": {"follow_up_depth": 3},
  "label": "NEXT_TOPIC",
  "reason_type": "positive"
}

// Qwen训练数据（对应ID 3）
{
  "context": {
    "current_question": "那你遇到过缓存穿透吗？",  // ← 同样的问题
    "follow_up_depth": 3                         // ← 同样的深度
  },
  "user_answer": "遇到过，我用了布隆过滤器...", // ← 同样的回答
  "expected_output": "很好！Redis这块你掌握得不错..."
}
```

---

## 🔧 实际应用中的协作

### **运行时流程**

```python
# ========== 候选人提交回答 ==========
user_answer = "遇到过，我用了布隆过滤器..."

# ========== Step 1: BERT决策 ==========
from models.follow_up_decision import FollowUpDecisionModel

bert = FollowUpDecisionModel(config)

decision, confidence = bert.decide(
    question="你遇到过缓存穿透吗？",
    answer=user_answer,
    speech_analysis={
        'hesitation_score': 0.15,
        'filler_count': 1
    },
    context={'follow_up_depth': 3}  # 已经是第3次追问
)

print(decision)  # 输出："NEXT_TOPIC"
# 但BERT不区分是积极还是消极，只是决策"要换话题"

# ========== Step 2: Qwen生成 ==========
from models.lightweight_interviewer import LightweightInterviewer

qwen = LightweightInterviewer(config)

# Qwen接收BERT的决策 + 完整上下文
response = qwen.generate_response(
    action_type=decision,  # "NEXT_TOPIC"
    context={
        'job_title': '后端工程师',
        'current_question': '你遇到过缓存穿透吗？',
        'follow_up_depth': 3,  # ← 关键：Qwen看到深度
        ...
    },
    user_answer=user_answer,  # ← 关键：Qwen看到完整回答
    speech_analysis={
        'hesitation_score': 0.15,  # ← 关键：Qwen看到犹豫度
        'filler_count': 1
    }
)

# Qwen在prompt中推断：
# - depth=3 且 hesitation_score低 且 answer长度大
# → 这是"积极换话题"
# → 应该先肯定，再换话题

print(response)
# 输出："很好！Redis这块你掌握得不错。我们换个话题..."
```

---

## 💡 为什么需要两个模型协作？

### **单模型的问题**

#### 方案A：只用Qwen（❌）
```python
# 让Qwen既做决策又生成
prompt = f"""
候选人回答：{user_answer}
追问深度：{depth}
犹豫度：{hesitation_score}

请判断：
1. 是否继续追问？（是/否）
2. 如果是，生成追问；如果否，生成换话题的话
"""

response = qwen.generate(prompt)
```

**问题**：
- Qwen-1.5B太小，理解复杂prompt能力弱
- 容易出现"机械追问"（你遇到的问题）
- 决策不稳定

#### 方案B：只用BERT（❌）
```python
# BERT只能分类，不能生成文本
decision = bert.classify(...)  # "NEXT_TOPIC"

# 然后怎么生成自然语言？只能用模板
if decision == "NEXT_TOPIC":
    response = "好的，我们换个话题。"  # ← 生硬
```

**问题**：
- BERT不能生成，只能用固定模板
- 模板回复不自然

### **双模型协作（✅）**

```
BERT（擅长分类）         Qwen（擅长生成）
      ↓                        ↓
  智能决策              +    自然表达
      ↓                        ↓
    各司其职，效果最佳
```

**优势**：
- BERT专注决策：准确率高（微调后>90%）
- Qwen专注生成：只需根据action_type生成，压力小
- 两个简单任务 > 一个复杂任务

---

## 📈 数据标注策略

### **BERT数据标注**

```python
# 标注NEXT_TOPIC时，必须记录reason_type

if "不会" in answer or hesitation_score > 0.7:
    label = "NEXT_TOPIC"
    reason_type = "negative"  # 消极换话题
    
elif follow_up_depth >= 3 and answer_quality_good:
    label = "NEXT_TOPIC"
    reason_type = "positive"  # 积极换话题
    
else:
    label = "FOLLOW_UP"
```

### **Qwen数据标注**

```python
# 根据reason_type生成不同话术

if bert_decision == "NEXT_TOPIC":
    if reason_type == "negative":
        # 消极换话题：体贴、不尴尬
        expected_output = "没关系，我们换个话题..."
        
    elif reason_type == "positive":
        # 积极换话题：肯定成果、自然过渡
        expected_output = "很好！XX这块你掌握得不错。我们换个方向..."
```

---

## 🎯 总结

### **关键点**

1. **NEXT_TOPIC有两种情况**
   - 消极：候选人答不上来
   - 积极：已经问够了

2. **两个模型必须呼应**
   - 特征对齐
   - 决策对应
   - 上下文一致

3. **数据集需要配合**
   - BERT标注reason_type
   - Qwen针对不同reason_type生成不同话术
   - 训练数据ID要对应

### **实现方案**

**当前方案**（保持3分类）：
```
BERT: 只输出决策（FOLLOW_UP/NEXT_TOPIC/END）
Qwen: 从context推断原因，生成合适话术
```

**优化方案**（改为4分类）：
```
BERT: 输出细化决策
  - FOLLOW_UP
  - NEXT_TOPIC_NEGATIVE（答不上来）
  - NEXT_TOPIC_POSITIVE（已问够）
  - END

Qwen: 根据明确的action_type生成
```

### **你的观察非常准确！**

- ✅ 识别了NEXT_TOPIC的两种情况
- ✅ 发现了reason字段的作用
- ✅ 意识到两个模型数据需要呼应

这些都是**正确且重要**的设计考量！

