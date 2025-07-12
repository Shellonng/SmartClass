# Dify API配置和使用指南

## 🎉 您的API密钥配置

您已成功创建了Dify应用并获得了API密钥：

### 已获得的密钥信息
- **API密钥**: `app-D5isfWHTIwVk8t82S507Rkfs`
- **聊天机器人Token**: `SKiyotVrMpqPW2Sp`
- **服务器地址**: `http://219.216.65.108`

## 🔧 后端配置

### 1. 更新配置文件

我已经帮您更新了 `backend/src/main/resources/application.yml`：

```yaml
education:
  dify:
    api-url: http://219.216.65.108
    api-keys:
      # 您的第一个应用API密钥（可以用作知识图谱生成或通用聊天）
      knowledge-graph: app-D5isfWHTIwVk8t82S507Rkfs
      chatbot: app-D5isfWHTIwVk8t82S507Rkfs
      # 还需要创建的其他应用
      paper-generation: your-paper-generation-app-token
      auto-grading: your-auto-grading-app-token
```

### 2. 应用类型说明

您当前创建的应用可以用于：

**如果是聊天机器人应用**:
- ✅ 智能问答
- ✅ 学习辅导
- ✅ 课程咨询
- ✅ 知识问答

**如果是工作流应用**:
- ✅ 知识图谱生成
- ✅ 内容分析
- ✅ 智能推荐

## 🎯 使用示例

### 后端API调用示例

```java
@RestController
@RequestMapping("/api/test")
public class DifyTestController {

    @Autowired
    private DifyService difyService;

    /**
     * 测试聊天机器人功能
     */
    @PostMapping("/chat")
    public Result<?> testChat(@RequestBody Map<String, Object> request) {
        try {
            // 构建输入参数
            Map<String, Object> inputs = new HashMap<>();
            inputs.put("query", request.get("message"));
            inputs.put("user_id", request.get("userId"));

            // 调用Dify API
            DifyDTO.DifyResponse response = difyService.callWorkflowApi(
                "chatbot", 
                inputs, 
                (String) request.get("userId")
            );

            return Result.success("聊天成功", response);
            
        } catch (Exception e) {
            return Result.error("聊天失败: " + e.getMessage());
        }
    }

    /**
     * 测试知识图谱生成
     */
    @PostMapping("/knowledge-graph")
    public Result<?> testKnowledgeGraph(@RequestBody Map<String, Object> request) {
        try {
            KnowledgeGraphDTO.GenerationRequest kgRequest = new KnowledgeGraphDTO.GenerationRequest();
            kgRequest.setCourseId((Long) request.get("courseId"));
            kgRequest.setGraphType("concept");
            kgRequest.setDepth(3);

            // 如果您的应用支持知识图谱生成
            KnowledgeGraphDTO.GenerationResponse response = 
                knowledgeGraphService.generateKnowledgeGraph(kgRequest, "test-user");

            return Result.success("知识图谱生成成功", response);
            
        } catch (Exception e) {
            return Result.error("知识图谱生成失败: " + e.getMessage());
        }
    }
}
```

## 🌐 前端集成

### 1. 聊天机器人嵌入（推荐）

您可以直接在任何前端页面中添加以下代码：

```html
<!DOCTYPE html>
<html>
<head>
    <title>SmartClass - AI助手</title>
</head>
<body>
    <!-- 页面内容 -->
    <div id="main-content">
        <h1>欢迎使用SmartClass教育平台</h1>
        <!-- 其他内容 -->
    </div>

    <!-- Dify聊天机器人配置 -->
    <script>
        window.difyChatbotConfig = {
            token: 'SKiyotVrMpqPW2Sp',
            baseUrl: 'http://219.216.65.108',
            systemVariables: {
                // 可以传入用户ID
                // user_id: 'USER_ID_HERE',
            },
            userVariables: {
                // 可以传入用户信息
                // avatar_url: 'USER_AVATAR_URL',
                // name: 'USER_NAME',
            },
        }
    </script>
    
    <!-- 加载聊天机器人 -->
    <script
        src="http://219.216.65.108/embed.min.js"
        id="SKiyotVrMpqPW2Sp"
        defer>
    </script>
    
    <!-- 自定义样式 -->
    <style>
        #dify-chatbot-bubble-button {
            background-color: #1C64F2 !important;
            box-shadow: 0 4px 12px rgba(28, 100, 242, 0.3) !important;
        }
        #dify-chatbot-bubble-window {
            width: 24rem !important;
            height: 40rem !important;
            border-radius: 12px !important;
        }
        
        /* 移动端适配 */
        @media (max-width: 768px) {
            #dify-chatbot-bubble-window {
                width: 90vw !important;
                height: 80vh !important;
                max-width: 350px !important;
            }
        }
    </style>
</body>
</html>
```

### 2. Vue组件集成

如果您使用Vue.js，我已经创建了一个组件 `DifyChatbot.vue`，使用方法：

```vue
<template>
  <div class="page-container">
    <!-- 页面内容 -->
    <div class="main-content">
      <h1>课程学习</h1>
      <!-- 其他内容 -->
    </div>
    
    <!-- AI助手聊天机器人 -->
    <DifyChatbot 
      :user-id="currentUser.id"
      :user-name="currentUser.name"
      :avatar-url="currentUser.avatar"
    />
  </div>
</template>

<script setup>
import DifyChatbot from '@/components/common/DifyChatbot.vue'
import { useUserStore } from '@/stores/user'

const userStore = useUserStore()
const currentUser = computed(() => userStore.currentUser)
</script>
```

## 🧪 测试验证

### 1. 后端API测试

启动应用后，可以使用以下curl命令测试：

```bash
# 测试聊天功能
curl -X POST http://localhost:8080/api/test/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "你好，我想了解Java编程",
    "userId": "test-user-123"
  }'

# 测试知识图谱生成
curl -X POST http://localhost:8080/api/test/knowledge-graph \
  -H "Content-Type: application/json" \
  -d '{
    "courseId": 1
  }'
```

### 2. 前端聊天机器人测试

1. 在浏览器中打开包含聊天机器人代码的页面
2. 应该看到右下角出现蓝色的聊天按钮
3. 点击按钮打开聊天窗口
4. 输入消息测试AI响应

## 📋 下一步建议

### 1. 创建专门的工作流应用

为了获得最佳效果，建议创建专门的应用：

#### 📝 组卷工作流 (paper-generation)
```yaml
应用类型: 工作流
功能: 根据课程内容和难度要求自动生成试卷
输入变量:
  - course_id: 课程ID
  - difficulty: 难度级别 (easy/medium/hard)
  - question_count: 题目数量
  - question_types: 题目类型 (选择题/填空题/简答题)
```

#### 📝 自动批改工作流 (auto-grading)
```yaml
应用类型: 工作流
功能: 智能批改学生作业和考试
输入变量:
  - questions: 题目列表
  - student_answers: 学生答案
  - grading_criteria: 评分标准
```

### 2. 优化建议

1. **个性化配置**: 根据用户角色（学生/教师）显示不同的聊天机器人功能
2. **上下文保持**: 在聊天中保持用户的学习上下文和课程信息
3. **多语言支持**: 配置支持中英文对话
4. **权限控制**: 限制某些高级功能只对教师开放

## 🔧 故障排除

### 常见问题

1. **聊天机器人不显示**
   - 检查网络连接到 `http://219.216.65.108`
   - 确认token `SKiyotVrMpqPW2Sp` 正确
   - 查看浏览器控制台错误信息

2. **API调用失败**
   - 检查API密钥 `app-D5isfWHTIwVk8t82S507Rkfs` 是否正确
   - 确认应用类型匹配
   - 查看后端日志错误信息

3. **样式问题**
   - 检查CSS冲突
   - 确认z-index设置
   - 测试移动端适配

## 📞 技术支持

如有问题，请检查：
1. 网络连接状态
2. API密钥配置
3. 应用日志信息
4. Dify服务器状态

---

**配置状态**: ✅ API密钥已配置  
**最后更新**: 2024年  
**维护团队**: SmartClass Development Team 