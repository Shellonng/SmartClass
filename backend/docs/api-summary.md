# 智慧课堂教育平台 - 后端API功能总结

## 🏗️ 系统架构概览

### 技术栈
- **框架**: Spring Boot 3.x + Spring Security + MyBatis Plus
- **数据库**: MySQL 8.0 + Redis
- **文档**: Swagger/OpenAPI 3.0
- **日志**: Logback + SLF4J
- **验证**: Jakarta Validation
- **缓存**: Redis + Spring Cache

### 项目结构
```
backend/
├── controller/          # 控制器层
│   ├── admin/          # 管理员接口
│   ├── auth/           # 认证接口
│   ├── common/         # 通用接口
│   ├── student/        # 学生端接口
│   └── teacher/        # 教师端接口
├── service/            # 服务层
├── mapper/             # 数据访问层
├── dto/                # 数据传输对象
├── entity/             # 实体类
├── config/             # 配置类
├── utils/              # 工具类
└── exception/          # 异常处理
```

## 📋 核心功能模块

### 1. 教师端功能 (Teacher APIs)

#### 1.1 班级管理 (/api/teacher/classes)
```http
GET    /api/teacher/classes              # 分页查询班级列表
POST   /api/teacher/classes              # 创建班级
GET    /api/teacher/classes/{id}         # 获取班级详情
PUT    /api/teacher/classes/{id}         # 更新班级信息
DELETE /api/teacher/classes/{id}         # 删除班级

# 学生管理
GET    /api/teacher/classes/{id}/students        # 获取班级学生列表
POST   /api/teacher/classes/{id}/students        # 添加学生到班级
DELETE /api/teacher/classes/{id}/students/{sid}  # 移除单个学生
DELETE /api/teacher/classes/{id}/students        # 批量移除学生

# 班级操作
GET    /api/teacher/classes/{id}/statistics      # 获取班级统计
PUT    /api/teacher/classes/{id}/status          # 更新班级状态
POST   /api/teacher/classes/{id}/copy            # 复制班级
GET    /api/teacher/classes/{id}/export          # 导出学生名单
```

**核心特性**:
- ✅ 完整的CRUD操作
- ✅ 学生批量管理
- ✅ 权限验证
- ✅ 数据统计分析
- ✅ Excel导出功能

#### 1.2 任务管理 (/api/teacher/tasks)
```http
GET    /api/teacher/tasks                    # 分页查询任务列表
POST   /api/teacher/tasks                    # 创建任务
GET    /api/teacher/tasks/{id}               # 获取任务详情
PUT    /api/teacher/tasks/{id}               # 更新任务
DELETE /api/teacher/tasks/{id}               # 删除任务

# 任务发布
POST   /api/teacher/tasks/{id}/publish       # 发布任务
POST   /api/teacher/tasks/{id}/unpublish     # 取消发布

# 作业批阅
GET    /api/teacher/tasks/{id}/submissions   # 获取提交列表
POST   /api/teacher/tasks/{id}/submissions/{sid}/grade  # 批阅作业
POST   /api/teacher/tasks/{id}/submissions/batch-grade # 批量批阅

# 统计分析
GET    /api/teacher/tasks/{id}/statistics    # 获取任务统计
GET    /api/teacher/tasks/{id}/export        # 导出成绩

# 高级功能
POST   /api/teacher/tasks/{id}/copy          # 复制任务
POST   /api/teacher/tasks/{id}/extend        # 延长截止时间
POST   /api/teacher/tasks/{id}/ai-grade      # 启用AI批阅

# 模板功能
GET    /api/teacher/tasks/templates          # 获取任务模板
POST   /api/teacher/tasks/from-template/{tid} # 从模板创建
```

**核心特性**:
- ✅ 任务全生命周期管理
- ✅ 智能批阅功能
- ✅ 批量操作支持
- ✅ 统计分析报告
- ✅ 模板系统

#### 1.3 AI工具集 (/api/teacher/ai)
```http
# 智能批改
POST   /api/teacher/ai/grade               # 智能批改作业
POST   /api/teacher/ai/batch-grade         # 批量智能批改

# 学习分析
POST   /api/teacher/ai/recommend           # 生成学习推荐
POST   /api/teacher/ai/ability-analysis    # 学生能力分析
POST   /api/teacher/ai/classroom-analysis  # 课堂表现分析

# 内容生成
POST   /api/teacher/ai/knowledge-graph     # 生成知识图谱
POST   /api/teacher/ai/generate-questions  # 智能题目生成
POST   /api/teacher/ai/teaching-suggestions # 教学建议

# 路径优化
POST   /api/teacher/ai/optimize-path       # 学习路径优化

# 文档分析
POST   /api/teacher/ai/analyze-document    # 文档AI分析
GET    /api/teacher/ai/analysis-history    # 分析历史

# 模型管理
POST   /api/teacher/ai/config              # 配置AI模型
GET    /api/teacher/ai/model-status        # 获取模型状态
POST   /api/teacher/ai/train-model         # 训练个性化模型
GET    /api/teacher/ai/training-progress/{id} # 训练进度
```

**核心特性**:
- 🤖 智能批改系统
- 📊 多维度学习分析
- 🧠 知识图谱生成
- 📝 智能题目生成
- 🔄 个性化模型训练

### 2. 学生端功能 (Student APIs)

#### 2.1 AI学习助手 (/api/student/ai-learning)
```http
# 个性化推荐
GET    /api/student/ai-learning/recommendations    # 学习推荐
POST   /api/student/ai-learning/question-answer    # 智能答疑

# 学习分析
GET    /api/student/ai-learning/ability-analysis   # 能力分析
GET    /api/student/ai-learning/progress-analysis  # 进度分析
GET    /api/student/ai-learning/efficiency-analysis # 效率分析

# 学习规划
POST   /api/student/ai-learning/study-plan         # 生成学习计划
GET    /api/student/ai-learning/review-recommendations # 复习推荐
GET    /api/student/ai-learning/learning-optimization # 学习优化

# 知识掌握
GET    /api/student/ai-learning/knowledge-mastery  # 知识点掌握度

# 练习推荐
POST   /api/student/ai-learning/practice-recommendations # 智能练习

# 状态评估
POST   /api/student/ai-learning/state-assessment   # 学习状态评估

# 报告生成
GET    /api/student/ai-learning/learning-report    # AI学习报告

# 目标管理
POST   /api/student/ai-learning/learning-goals     # 设置学习目标
GET    /api/student/ai-learning/learning-history   # 学习历史

# 反馈系统
POST   /api/student/ai-learning/feedback           # 反馈AI推荐
```

**核心特性**:
- 🎯 个性化学习推荐
- 🤔 智能答疑系统
- 📈 多维度学习分析
- 📋 智能学习计划
- 🎮 游戏化学习体验

### 3. 通用功能 (Common APIs)

#### 3.1 认证授权 (/api/auth)
```http
POST   /api/auth/login              # 用户登录
POST   /api/auth/logout             # 用户登出
POST   /api/auth/refresh            # 刷新Token
GET    /api/auth/me                 # 获取当前用户信息
```

#### 3.2 文件管理 (/api/common/files)
```http
POST   /api/common/files/upload     # 文件上传
GET    /api/common/files/{id}       # 文件下载
DELETE /api/common/files/{id}       # 删除文件
GET    /api/common/files/list       # 文件列表
```

#### 3.3 用户管理 (/api/common/users)
```http
GET    /api/common/users            # 用户列表
GET    /api/common/users/{id}       # 用户详情
PUT    /api/common/users/{id}       # 更新用户
```

## 📊 数据模型设计

### 核心实体关系
```
User (用户表)
├── Teacher (教师扩展)
├── Student (学生扩展)
└── Admin (管理员扩展)

Class (班级表)
├── head_teacher_id → User.id
└── ClassStudent (班级学生关系表)

Course (课程表)
├── teacher_id → User.id
└── CourseClass (课程班级关系表)

Task (任务表)
├── course_id → Course.id
├── creator_id → User.id
└── TaskSubmission (任务提交表)

AIFeature (AI功能表)
├── user_id → User.id
└── feature_type (功能类型)
```

### DTO设计模式
- **Request DTOs**: 请求参数验证
- **Response DTOs**: 响应数据封装
- **PageRequest/PageResponse**: 统一分页
- **Result**: 统一响应格式

## 🔒 安全机制

### 认证授权
- JWT Token认证
- 基于角色的权限控制(RBAC)
- 接口级权限验证
- 数据级权限隔离

### 数据安全
- 参数验证 (Jakarta Validation)
- SQL注入防护 (MyBatis Plus)
- XSS防护
- CSRF防护

## 📈 性能优化

### 缓存策略
- Redis缓存热点数据
- 分页查询优化
- 数据库连接池

### 日志监控
- 结构化日志记录
- 性能监控埋点
- 异常追踪

## 🚀 部署配置

### 环境配置
```yaml
# application.yml
server:
  port: 8080

spring:
  datasource:
    url: jdbc:mysql://localhost:3306/education_platform
  redis:
    host: localhost
    port: 6379
  security:
    jwt:
      secret: your-secret-key
      expiration: 86400000
```

### API文档
- Swagger UI: `http://localhost:8080/swagger-ui.html`
- API Docs: `http://localhost:8080/v3/api-docs`

## 📝 开发规范

### 代码结构
- Controller: 接口层，只处理HTTP请求响应
- Service: 业务逻辑层，事务管理
- Mapper: 数据访问层，SQL操作
- DTO: 数据传输对象，参数验证

### 异常处理
- 全局异常处理器
- 业务异常统一封装
- 错误码标准化

### 日志规范
- 接口调用日志
- 业务操作日志
- 异常错误日志
- 性能监控日志

## 🎯 后续计划

### 功能扩展
- [ ] 视频直播教学
- [ ] 实时协作功能
- [ ] 移动端适配
- [ ] 微服务拆分

### 技术升级
- [ ] 引入消息队列
- [ ] 分布式存储
- [ ] 容器化部署
- [ ] 自动化测试

---

**版本**: v1.0.0  
**更新时间**: 2024-12-24  
**维护团队**: 智慧课堂开发组 