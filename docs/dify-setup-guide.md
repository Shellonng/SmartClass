# Dify AI平台接入配置指南

## 📋 概述

本指南将帮助您将Dify AI平台接入到SmartClass教育平台中，实现AI赋能的教育功能。

## 🔧 环境准备

### 1. Dify服务器信息
- **服务器地址**: `http://219.216.65.108`
- **API版本**: `v1`
- **完整API地址**: `http://219.216.65.108/v1`

### 2. 系统要求
- Java 8+
- Spring Boot 2.7+
- 网络连接能访问Dify服务器

## 🚀 配置步骤

### 第一步：更新应用配置

配置文件位置：`backend/src/main/resources/application.yml`

```yaml
education:
  dify:
    # Dify服务器地址
    api-url: http://219.216.65.108
    # API密钥配置
    api-keys:
      # 组卷工作流API密钥
      paper-generation: app-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
      # 自动批改工作流API密钥  
      auto-grading: app-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
      # 知识图谱生成API密钥
      knowledge-graph: app-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # 请求配置
    timeout: 60000  # 请求超时时间(毫秒)
    retry-count: 3  # 重试次数
```

### 第二步：获取API密钥

#### 2.1 访问Dify管理后台
1. 访问：`http://219.216.65.108`
2. 使用管理员账号登录

#### 2.2 创建应用并获取API密钥

**组卷工作流应用**：
1. 创建新的工作流应用
2. 应用名称：`paper-generation`
3. 配置工作流（参考后续工作流配置）
4. 获取API密钥，格式如：`app-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`

**自动批改工作流应用**：
1. 创建新的工作流应用
2. 应用名称：`auto-grading`
3. 配置工作流（参考后续工作流配置）
4. 获取API密钥

**知识图谱生成工作流应用**：
1. 创建新的工作流应用
2. 应用名称：`knowledge-graph`
3. 配置工作流（参考后续工作流配置）
4. 获取API密钥

### 第三步：配置工作流

#### 3.1 知识图谱生成工作流

创建名为 `knowledge-graph` 的工作流，配置以下输入变量：

```yaml
输入变量:
  - course_data: 课程数据 (JSON格式)
  - graph_type: 图谱类型 (concept/skill/comprehensive)
  - depth: 知识图谱深度 (1-5)
  - max_nodes: 最大节点数
  - requirements: 特殊要求
```

输出格式：
```json
{
  "status": "completed",
  "graph_data": {
    "nodes": [...],
    "edges": [...],
    "metadata": {...}
  }
}
```

#### 3.2 组卷工作流配置

创建名为 `paper-generation` 的工作流：

```yaml
输入变量:
  - course_id: 课程ID
  - knowledge_points: 知识点列表
  - difficulty: 难度级别
  - question_count: 题目数量
  - question_types: 题目类型
  - duration: 考试时长
  - total_score: 总分
```

#### 3.3 自动批改工作流配置

创建名为 `auto-grading` 的工作流：

```yaml
输入变量:
  - question_content: 题目内容
  - standard_answer: 标准答案
  - student_answer: 学生答案
  - question_type: 题目类型
  - total_score: 总分
```

### 第四步：验证配置

#### 4.1 启动应用
```bash
cd backend
mvn spring-boot:run
```

#### 4.2 测试API连接
```bash
curl -X GET http://localhost:8080/actuator/health
```

#### 4.3 测试Dify接口
查看启动日志，确认没有Dify连接错误。

## 🔧 故障排除

### 常见问题

**1. API密钥错误**
```
错误: 未配置xxx的API密钥
解决: 检查application.yml中的api-keys配置
```

**2. 连接超时**
```
错误: Dify API连接超时
解决: 检查网络连接和服务器状态
```

**3. 工作流不存在**
```
错误: 工作流未找到
解决: 确认Dify中已创建对应名称的工作流
```

### 调试步骤

1. **检查网络连接**
```bash
ping 219.216.65.108
curl -I http://219.216.65.108
```

2. **检查API密钥格式**
API密钥应该是以 `app-` 开头的32位字符串

3. **查看日志**
```bash
tail -f logs/education-platform.log
```

4. **测试Dify API**
```bash
curl -X POST http://219.216.65.108/v1/workflows/run \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "inputs": {"test": "hello"},
    "response_mode": "blocking",
    "user": "test"
  }'
```

## 🎯 使用示例

### 1. 知识图谱生成

```java
@Autowired
private KnowledgeGraphService knowledgeGraphService;

// 生成知识图谱
KnowledgeGraphDTO.GenerationRequest request = new KnowledgeGraphDTO.GenerationRequest();
request.setCourseId(1L);
request.setGraphType("concept");
request.setDepth(3);

KnowledgeGraphDTO.GenerationResponse response = 
    knowledgeGraphService.generateKnowledgeGraph(request, "user123");
```

### 2. 组卷功能

```java
@Autowired
private DifyService difyService;

// 生成试卷
DifyDTO.PaperGenerationRequest request = new DifyDTO.PaperGenerationRequest();
request.setCourseId(1L);
request.setDifficulty("medium");
request.setQuestionCount(20);

DifyDTO.PaperGenerationResponse response = 
    difyService.generatePaper(request, "teacher123");
```

### 3. 自动批改

```java
// 批改作业
DifyDTO.AutoGradingRequest request = new DifyDTO.AutoGradingRequest();
request.setQuestionContent("什么是多态？");
request.setStandardAnswer("多态是面向对象编程的特性...");
request.setStudentAnswer("多态就是一个接口多种实现");

DifyDTO.AutoGradingResponse response = 
    difyService.gradeAssignment(request, "teacher123");
```

## 📝 注意事项

1. **API密钥安全**
   - 不要将API密钥提交到版本控制系统
   - 建议使用环境变量或配置文件管理

2. **请求频率限制**
   - 避免过于频繁的API调用
   - 实施适当的缓存策略

3. **错误处理**
   - 实现完善的错误处理机制
   - 提供友好的用户反馈

4. **性能优化**
   - 对于批量操作，考虑异步处理
   - 监控API响应时间

## 📞 技术支持

如果遇到问题，请按以下步骤操作：

1. 检查本文档的故障排除部分
2. 查看应用日志文件
3. 确认Dify服务器状态
4. 联系技术支持团队

---

**版本**: 1.0.0  
**更新日期**: 2024年  
**维护团队**: SmartClass Development Team 