# Dify集成说明文档

## 📋 概述

本文档详细说明如何在教育平台中集成Dify AI服务，实现智能组卷、自动批改等功能。

## 🔧 环境准备

### 1. Dify服务部署

确保您已经在服务器上成功部署了Dify：

```bash
# 检查Dify服务状态
curl http://localhost:3000/health

# 如果还未部署，请参考Dify官方文档
git clone https://github.com/langgenius/dify.git
cd dify
cp .env.example .env
# 修改.env配置文件
docker-compose up -d
```

### 2. Ollama集成

确保Ollama已正确配置并与Dify连接：

```bash
# 检查Ollama状态
ollama list

# 拉取需要的模型
ollama pull llama2
ollama pull qwen
```

## 🏗️ Dify工作流配置

### 1. 智能组卷工作流

#### 创建应用
1. 登录Dify控制台（http://localhost:3000）
2. 点击"创建应用" → 选择"工作流"
3. 应用名称：`智能组卷系统`
4. 描述：`基于AI的智能试卷生成系统`

#### 工作流节点配置

**输入节点（Start）**
- 变量名：`course_id` (文本)
- 变量名：`knowledge_points` (文本)
- 变量名：`difficulty` (文本)
- 变量名：`question_count` (数字)
- 变量名：`question_types` (文本)
- 变量名：`duration` (数字)
- 变量名：`total_score` (数字)
- 变量名：`additional_requirements` (文本)

**LLM节点（题目生成）**
```prompt
你是一位经验丰富的教师，需要根据以下要求生成试卷：

课程ID：{{course_id}}
知识点范围：{{knowledge_points}}
难度级别：{{difficulty}}
题目数量：{{question_count}}
题型分布：{{question_types}}
考试时长：{{duration}}分钟
总分：{{total_score}}分
额外要求：{{additional_requirements}}

请不要在输出中包含thinking或思考过程。

请严格按照以下JSON格式输出：

{
  "title": "试卷标题",
  "questions": [
    {
      "questionText": "题目内容",
      "questionType": "SINGLE_CHOICE|MULTIPLE_CHOICE|TRUE_FALSE|FILL_BLANK|ESSAY",
      "options": ["选项A", "选项B", "选项C", "选项D"], // 仅选择题需要
      "correctAnswer": "正确答案",
      "score": 分值,
      "knowledgePoint": "知识点",
      "difficulty": "EASY|MEDIUM|HARD",
      "explanation": "解析"
    }
  ]
}

要求：
1. 题目内容要准确、清晰、符合学术规范
2. 选择题的选项要有合理的干扰项
3. 难度分布要合理：简单30%、中等50%、困难20%
4. 每个题目必须标注对应的知识点
5. 提供详细的解析说明
```

**Code节点（结果处理）**
```python
import json

def main(llm_response: str) -> dict:
    try:
        # 解析LLM响应
        result = json.loads(llm_response)
        
        # 验证数据格式
        if "questions" not in result:
            return {"error": "生成结果格式错误"}
        
        # 计算分值分布
        total_calculated = sum(q.get("score", 0) for q in result["questions"])
        
        return {
            "title": result.get("title", "智能生成试卷"),
            "questions": result["questions"],
            "question_count": len(result["questions"]),
            "total_score": total_calculated,
            "status": "success"
        }
    except Exception as e:
        return {
            "error": f"处理失败: {str(e)}",
            "status": "failed"
        }
```

**输出节点（End）**
- 输出变量：处理后的试卷数据

### 2. 智能批改工作流

#### 创建应用
1. 创建新工作流应用：`智能批改系统`
2. 描述：`基于AI的自动作业批改系统`

#### 工作流节点配置

**输入节点（Start）**
- 变量名：`assignment_id` (数字)
- 变量名：`student_id` (数字)
- 变量名：`answers` (文本，JSON格式)
- 变量名：`grading_type` (文本)
- 变量名：`grading_criteria` (文本)

**LLM节点（批改分析）**
```prompt
你是一位专业的教师，需要批改学生的作业。请根据以下信息进行批改：

作业ID：{{assignment_id}}
学生ID：{{student_id}}
批改类型：{{grading_type}}
评分标准：{{grading_criteria}}

学生答案：{{answers}}

请按照以下要求进行批改：

1. 客观题（选择题、判断题、填空题）：
   - 严格按照标准答案评分
   - 答案完全正确才给分

2. 主观题（简答题、论述题）：
   - 根据答案要点给分
   - 考虑逻辑性和完整性
   - 语言表达和结构清晰度

3. 评分标准：
   - 答案准确性（60%）
   - 表达清晰度（20%）
   - 逻辑结构（20%）

请严格按照以下JSON格式输出：

{
  "results": [
    {
      "questionId": 题目ID,
      "isCorrect": true/false,
      "score": 得分,
      "totalScore": 总分,
      "comment": "具体批改意见",
      "errorType": "错误类型（如果错误）",
      "suggestion": "改进建议"
    }
  ],
  "totalScore": 总分,
  "earnedScore": 得分,
  "percentage": 得分率,
  "overallComment": "整体评价"
}
```

**Code节点（批改结果处理）**
```python
import json

def main(llm_response: str) -> dict:
    try:
        result = json.loads(llm_response)
        
        # 验证批改结果
        if "results" not in result:
            return {"error": "批改结果格式错误"}
        
        # 计算统计信息
        total_questions = len(result["results"])
        correct_count = sum(1 for r in result["results"] if r.get("isCorrect"))
        
        # 分析错误类型
        error_types = {}
        for r in result["results"]:
            if not r.get("isCorrect") and r.get("errorType"):
                error_type = r["errorType"]
                error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            "results": result["results"],
            "totalScore": result.get("totalScore", 0),
            "earnedScore": result.get("earnedScore", 0),
            "percentage": result.get("percentage", 0),
            "overallComment": result.get("overallComment", ""),
            "statistics": {
                "totalQuestions": total_questions,
                "correctCount": correct_count,
                "errorTypes": error_types
            },
            "status": "completed"
        }
    except Exception as e:
        return {
            "error": f"批改处理失败: {str(e)}",
            "status": "failed"
        }
```

## ⚙️ 应用配置

### 1. 获取API密钥

1. 在Dify控制台中，进入对应的应用
2. 点击"API访问" → "API密钥"
3. 创建新的密钥
4. 复制密钥到应用配置文件

### 2. 更新配置文件

编辑 `backend/src/main/resources/application.yml`：

```yaml
education:
  dify:
    api-url: http://localhost:3000  # 您的Dify服务地址
    api-keys:
      paper-generation: your-paper-generation-app-token
      auto-grading: your-auto-grading-app-token
      knowledge-graph: your-knowledge-graph-app-token
    timeout: 30000
    retry-count: 3
    ollama:
      model: llama2
```

## 🧪 测试验证

### 1. 组卷功能测试

```bash
# 测试智能组卷API
curl -X POST http://localhost:8080/api/teacher/paper/generate \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "courseId": 1,
    "knowledgePoints": ["函数与极限", "导数"],
    "difficulty": "MEDIUM",
    "questionCount": 10,
    "questionTypes": {
      "SINGLE_CHOICE": 5,
      "MULTIPLE_CHOICE": 3,
      "TRUE_FALSE": 2
    },
    "duration": 90,
    "totalScore": 100,
    "additionalRequirements": "注重实际应用"
  }'
```

### 2. 批改功能测试

```bash
# 测试智能批改API
curl -X POST http://localhost:8080/api/teacher/grading/auto-grade \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "assignmentId": 1,
    "studentId": 1001,
    "answers": [
      {
        "questionId": 1,
        "questionText": "什么是函数？",
        "questionType": "ESSAY",
        "correctAnswer": "函数是定义域到值域的映射关系...",
        "studentAnswer": "函数就是一种对应关系...",
        "totalScore": 10
      }
    ],
    "gradingType": "MIXED",
    "gradingCriteria": "注重答案准确性和表达清晰度"
  }'
```

## 🔍 故障排除

### 常见问题

1. **连接Dify失败**
   - 检查Dify服务是否正常运行
   - 验证API地址和端口
   - 检查网络连接

2. **API密钥错误**
   - 确认密钥是否正确复制
   - 检查密钥是否已过期
   - 验证应用权限设置

3. **生成结果格式错误**
   - 检查Prompt模板是否正确
   - 验证模型输出格式
   - 调整温度参数

4. **批改结果不准确**
   - 优化批改Prompt
   - 调整评分标准
   - 增加示例数据

### 日志查看

```bash
# 查看应用日志
tail -f logs/education-platform.log | grep -i dify

# 查看Dify服务日志
docker logs dify-api
```

## 📈 性能优化

### 1. 缓存策略

```java
// 在DifyService中添加缓存
@Cacheable(value = "paper-generation", key = "#request.courseId + '-' + #request.difficulty")
public DifyDTO.PaperGenerationResponse generatePaper(DifyDTO.PaperGenerationRequest request, String userId) {
    // 实现逻辑
}
```

### 2. 异步处理

```java
// 使用异步处理大批量批改
@Async
public CompletableFuture<List<DifyDTO.AutoGradingResponse>> batchGradeAsync(
    List<DifyDTO.AutoGradingRequest> requests) {
    // 实现逻辑
}
```

### 3. 限流控制

```java
// 添加限流注解
@RateLimiter(name = "dify-api", fallbackMethod = "fallbackGenerate")
public DifyDTO.PaperGenerationResponse generatePaper(DifyDTO.PaperGenerationRequest request, String userId) {
    // 实现逻辑
}
```

## 🔐 安全配置

### 1. API访问控制

- 使用HTTPS加密传输
- 设置API访问频率限制
- 实现用户权限验证

### 2. 数据隐私

- 敏感数据脱敏处理
- 实现数据加密存储
- 定期清理临时数据

## 📚 扩展功能

### 1. 知识图谱生成

创建知识图谱工作流，用于分析课程知识点关系。

### 2. 学习路径推荐

基于学生学习数据，智能推荐个性化学习路径。

### 3. 智能答疑

集成对话型AI，提供24/7学习答疑服务。

## 📝 更新日志

- **v1.0.0** (2024-01-20)
  - 初始版本发布
  - 支持智能组卷和自动批改
  - 集成Ollama模型

- **v1.1.0** (预计)
  - 添加知识图谱功能
  - 支持多模型切换
  - 优化批改准确性

## 🆘 技术支持

如果您在集成过程中遇到问题，请：

1. 查看本文档的故障排除部分
2. 检查系统日志文件
3. 联系技术支持团队
4. 在项目GitHub页面提交Issue

---

*本文档将根据功能更新持续维护，请关注最新版本。* 