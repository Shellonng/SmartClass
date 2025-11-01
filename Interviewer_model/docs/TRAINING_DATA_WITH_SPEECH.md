# 🎤 带语音特征的训练数据说明

## ✅ 回答你的问题

### **Q: 语音识别库输出的特征格式？**

**A: 是的，我完全知道！** 

我们使用的`SpeechProcessor`（基于Whisper + Wav2Vec2）输出格式如下：

```python
{
    "text": "嗯...我用过Redis，嗯...主要是做缓存",  # 转录文本（包含填充词）
    "filler_count": 5,                           # 填充词数量
    "filler_positions": [0.2, 1.8, 3.5],         # 填充词出现时间点
    "filler_words_found": ["嗯", "然后"],         # 具体填充词类型
    "hesitation_score": 0.75,                    # 犹豫度（0-1）
    "speech_rate": 2.5,                          # 语速（字/秒）
    "pauses": [1.8, 2.3],                        # 长停顿时长
    "confidence": 0.92                           # Whisper识别置信度
}
```

---

## 🎯 在训练数据中的准确使用

### **示例1：候选人不会（高犹豫）**

```json
{
  "id": 1,
  "question": "你了解Redis的持久化机制吗？",
  "answer": "嗯...这个...我好像...嗯...不太清楚，额...就是...知道一点点。",
  //       ↑填充词 ↑填充词        ↑填充词        ↑填充词 ↑填充词
  
  "context": {
    "follow_up_depth": 2,
    
    // ===== 来自SpeechProcessor的真实特征 =====
    "hesitation_score": 0.85,      // 很犹豫（0.7-1.0）
    "filler_count": 8,              // 大量填充词
    "filler_words": ["嗯", "这个", "额", "就是"],  // 具体是哪些词
    "speech_rate": 2.1,             // 语速慢（< 2.5）
    "pause_count": 3,               // 多次停顿
    "long_pauses": [2.5, 1.8, 2.0], // 停顿时长（秒）
    "answer_length": 30
  },
  
  "label": "NEXT_TOPIC",
  "reason": "hesitation_score=0.85，大量填充词和停顿，明显答不上来"
}
```

**关键点**：
- `answer`字段中**包含填充词**（嗯、这个、额等）
- 这些填充词会被Whisper识别并统计
- `filler_count`、`hesitation_score`等特征由系统自动计算

### **示例2：候选人答得好（低犹豫）**

```json
{
  "question": "你用过Redis吗？",
  "answer": "用过，我在项目中用Redis做缓存，主要缓存用户session和热点数据。",
  //       ↑无填充词，流畅
  
  "context": {
    "follow_up_depth": 1,
    
    // ===== 流畅回答的特征 =====
    "hesitation_score": 0.15,      // 流畅（0.0-0.3）
    "filler_count": 1,              // 几乎没有填充词
    "filler_words": [],             // 或只有"就是"等轻微词
    "speech_rate": 4.2,             // 语速正常（3.5-5.0）
    "pause_count": 0,               // 无长停顿
    "long_pauses": [],
    "answer_length": 45
  },
  
  "label": "FOLLOW_UP",
  "reason": "回答流畅，可以追问"
}
```

---

## 📊 语音特征的真实范围

### **犹豫度（hesitation_score）**

| 范围 | 表现 | 特征 | 决策倾向 |
|------|------|------|---------|
| **0.0-0.3** | 非常流畅 | 无填充词、语速正常、无停顿 | FOLLOW_UP |
| **0.3-0.5** | 一般 | 少量填充词、可能有思考停顿 | FOLLOW_UP（内容好） |
| **0.5-0.7** | 明显犹豫 | 较多填充词、语速慢、有停顿 | NEXT_TOPIC（倾向） |
| **0.7-1.0** | 非常犹豫 | 大量填充词、多次停顿、语速很慢 | NEXT_TOPIC（必须） |

### **填充词数量（filler_count）**

| 回答长度 | 流畅 | 一般 | 犹豫 |
|---------|------|------|------|
| **短回答（< 30字）** | 0-1 | 2-3 | 4+ |
| **中等（30-80字）** | 0-2 | 3-5 | 6+ |
| **长回答（> 80字）** | 0-3 | 4-7 | 8+ |

### **语速（speech_rate）**

| 语速 | 字/秒 | 说明 |
|------|-------|------|
| **过慢** | < 2.5 | 可能在思考或不会 |
| **正常** | 2.5-5.0 | 流畅表达 |
| **过快** | > 5.0 | 可能紧张或背诵 |

---

## 🔧 数据标注实操

### **方法1：真实录音**

```python
# 1. 录制面试音频
# 2. 用SpeechProcessor处理
from models.speech_processor import SpeechProcessor

processor = SpeechProcessor(config)
result = processor.transcribe_with_analysis("interview_clip.wav")

# 3. 直接使用输出构建训练样本
training_sample = {
    "question": "你用过Redis吗？",
    "answer": result["text"],  # "嗯...我用过，嗯...做缓存"
    "context": {
        "follow_up_depth": 1,
        "hesitation_score": result["hesitation_score"],  # 0.65
        "filler_count": result["filler_count"],          # 5
        "filler_words": result["filler_words_found"],    # ["嗯"]
        "speech_rate": result["speech_rate"],            # 2.8
        "pause_count": len(result["pauses"]),            # 2
        "long_pauses": result["pauses"],                 # [1.8, 2.0]
        "answer_length": len(result["text"])
    },
    "label": "NEXT_TOPIC",  # 人工标注
    "reason": "虽然说用过，但犹豫度0.65，实际可能不太会"
}
```

### **方法2：模拟生成**

```python
import random

def generate_training_sample(answer_quality):
    """
    根据回答质量生成训练样本
    
    answer_quality: 'good' | 'ok' | 'bad'
    """
    
    if answer_quality == 'good':
        # 回答好：低犹豫
        answer_text = "用过，我在项目中用Redis做缓存，主要缓存session和热点数据。"
        context = {
            "hesitation_score": random.uniform(0.1, 0.25),
            "filler_count": random.randint(0, 2),
            "filler_words": random.choice([[], ["就是"], ["然后"]]),
            "speech_rate": random.uniform(3.8, 4.8),
            "pause_count": 0,
            "long_pauses": [],
            "answer_length": len(answer_text)
        }
        label = "FOLLOW_UP"
        
    elif answer_quality == 'ok':
        # 回答一般：中等犹豫
        answer_text = "嗯，用过一些，就是做缓存。"
        context = {
            "hesitation_score": random.uniform(0.35, 0.55),
            "filler_count": random.randint(3, 5),
            "filler_words": ["嗯", "就是", "然后"],
            "speech_rate": random.uniform(2.8, 3.5),
            "pause_count": random.randint(0, 1),
            "long_pauses": [random.uniform(1.5, 2.0)] if random.random() > 0.5 else [],
            "answer_length": len(answer_text)
        }
        label = "FOLLOW_UP" if random.random() > 0.5 else "NEXT_TOPIC"
        
    else:  # bad
        # 回答差：高犹豫
        answer_text = "嗯...这个...我好像...嗯...不太清楚"
        context = {
            "hesitation_score": random.uniform(0.75, 0.95),
            "filler_count": random.randint(6, 12),
            "filler_words": ["嗯", "这个", "额", "就是"],
            "speech_rate": random.uniform(1.5, 2.3),
            "pause_count": random.randint(2, 4),
            "long_pauses": [random.uniform(1.8, 2.8) for _ in range(random.randint(2, 4))],
            "answer_length": len(answer_text)
        }
        label = "NEXT_TOPIC"
    
    return {
        "answer": answer_text,
        "context": context,
        "label": label
    }

# 生成100条训练数据
training_data = []
for _ in range(40):
    training_data.append(generate_training_sample('good'))
for _ in range(30):
    training_data.append(generate_training_sample('ok'))
for _ in range(30):
    training_data.append(generate_training_sample('bad'))
```

---

## 🎯 答案中填充词的表示

### **正确的表示方式**

```json
// ✅ 正确：填充词在answer中明确出现
{
  "answer": "嗯...我用过Redis，嗯...主要是做缓存",
  //         ↑        ↑这些填充词真实存在
  "context": {
    "filler_count": 5,
    "filler_words": ["嗯"]
  }
}

// ✅ 也正确：流畅回答没有填充词
{
  "answer": "用过，我在项目中用Redis做缓存，主要缓存session和热点数据",
  "context": {
    "filler_count": 0,
    "filler_words": []
  }
}
```

### **错误的表示方式**

```json
// ❌ 错误：answer中没有填充词，但filler_count不为0
{
  "answer": "我用过Redis做缓存",
  "context": {
    "filler_count": 5  // ← 矛盾！文本中没有填充词
  }
}

// ❌ 错误：填充词用特殊符号标记
{
  "answer": "(嗯)我用过Redis(嗯)做缓存",  // ← 不要这样
  "context": {
    "filler_count": 2
  }
}

// ✅ 应该这样：
{
  "answer": "嗯...我用过Redis，嗯...做缓存",  // ← 填充词就是文本的一部分
  "context": {
    "filler_count": 2
  }
}
```

---

## 📝 完整训练样本示例

### **案例1：消极换话题（不会）**

```json
{
  "id": 101,
  "question": "你了解Redis的哨兵模式吗？",
  "answer": "嗯...哨兵模式...这个...我好像...没怎么用过，嗯...不太清楚具体原理。",
  //         ↑    ↑        ↑      ↑          ↑这些都是填充词
  
  "context": {
    "follow_up_depth": 2,
    
    // 真实的语音特征
    "hesitation_score": 0.82,
    "filler_count": 9,
    "filler_words": ["嗯", "这个", "好像"],
    "speech_rate": 2.2,
    "pause_count": 3,
    "long_pauses": [2.1, 1.9, 2.3],
    "answer_length": 35
  },
  
  "label": "NEXT_TOPIC",
  "reason": "hesitation_score=0.82，9个填充词，3次长停顿，明显不会",
  "reason_type": "negative",
  
  "qwen_should_say": "没关系，这个比较高级。我们换个话题..."
}
```

### **案例2：积极换话题（已问够）**

```json
{
  "id": 102,
  "question": "那你知道主从复制的延迟怎么优化吗？",
  "answer": "知道，可以用pipeline批量发送、减少同步频率、或者用cluster分摊写压力。我们项目中主要通过读写分离缓解，写主库读从库，配合缓存减轻数据库压力。",
  //       ↑流畅，无填充词
  
  "context": {
    "follow_up_depth": 3,  // 已经第3轮
    
    // 流畅的语音特征
    "hesitation_score": 0.14,
    "filler_count": 0,
    "filler_words": [],
    "speech_rate": 4.3,
    "pause_count": 0,
    "long_pauses": [],
    "answer_length": 110
  },
  
  "label": "NEXT_TOPIC",
  "reason": "已3轮追问，hesitation_score=0.14，回答专业流畅，Redis已充分展示，换话题",
  "reason_type": "positive",
  
  "qwen_should_say": "非常好！Redis这块你掌握得很扎实。我们聊聊其他方面..."
}
```

---

## 💡 总结

### **关键点**

1. **填充词在`answer`中真实存在**
   ```json
   "answer": "嗯...我用过，嗯...做缓存"  // 填充词是文本的一部分
   ```

2. **特征值来自SpeechProcessor的真实输出**
   ```json
   "hesitation_score": 0.75,  // 由系统计算，不是随便填的
   "filler_count": 5          // 统计answer中的填充词数量
   ```

3. **特征要符合真实范围**
   ```
   流畅：hesitation_score 0.1-0.3, filler_count 0-2
   犹豫：hesitation_score 0.7-1.0, filler_count 6+
   ```

4. **BERT和Qwen的数据必须呼应**
   - BERT标注决策和reason_type
   - Qwen根据reason_type生成不同话术

---

## 📂 已创建的文件

```
✅ data/follow_up_training_with_speech.json  # 带真实语音特征的BERT训练数据（20条）
✅ docs/SPEECH_FEATURE_FORMAT.md            # 语音特征格式文档
✅ docs/TRAINING_DATA_WITH_SPEECH.md        # 本文档
```

---

## 🚀 使用建议

1. **优先使用真实录音**
   - 录制10-20段真实面试音频
   - 用SpeechProcessor提取特征
   - 人工标注决策

2. **补充模拟数据**
   - 覆盖边界情况
   - 确保数据平衡
   - 特征值要真实

3. **验证一致性**
   - answer中的填充词 = filler_count
   - hesitation_score和其他特征匹配
   - BERT和Qwen数据呼应

