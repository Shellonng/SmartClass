# AI赋能教育管理与学习辅助平台 (SmartClass)

## 项目概述

本项目是一个面向高校教师和大学生的智能化教育管理平台，融合AI和大数据技术，实现"智能化生产、结构化管理、个性化学习"。平台提供全面的教学管理与学习功能，支持课程管理、作业布置与提交、资源共享、在线测验等多种教学场景，同时集成了AI辅助功能，提升教学效率与学习体验。

## 技术栈

### 前端
- **框架**: Vue 3.5+ + Vite 6.2+
- **状态管理**: Pinia 3.0+
- **路由管理**: Vue Router 4.5+
- **UI组件库**: Ant Design Vue 4.2+、Element Plus 2.10+
- **HTTP客户端**: Axios 1.10+
- **数据可视化**: ECharts 5.6+, Vue-ECharts 7.0+
- **媒体播放**: Video.js 8.23+
- **开发工具**: TypeScript 5.8+, ESLint 9+, Prettier 3.5+

### 后端
- **框架**: Spring Boot 3.1.8
- **安全认证**: Spring Security + JWT
- **ORM框架**: MyBatis-Plus 3.5.7
- **缓存**: Spring Data Redis
- **数据库**: MySQL 8.0+
- **连接池**: HikariCP
- **API文档**: Knife4j 4.3.0
- **对象存储**: MinIO 8.5.7
- **工具库**: Hutool 5.8+, FastJSON 2.0+
- **Excel处理**: Apache POI 5.2+, EasyExcel 3.3+

### 环境要求
- **Java**: JDK 17+
- **Node.js**: 16+
- **MySQL**: 8.0+
- **Redis**: 6.0+

## 项目结构

```
SmartClass/
├── backend/                       # 后端项目
│   ├── docs/                      # 项目文档
│   │   ├── api-summary.md         # API概述
│   │   └── database-connection-pool.md # 数据库连接池说明
│   ├── logs/                      # 日志文件
│   ├── src/                       # 源代码
│   │   ├── main/
│   │   │   ├── java/com/education/
│   │   │   │   ├── aspect/        # AOP切面（缓存、日志、性能监控等）
│   │   │   │   ├── config/        # 配置类（安全配置、Cors配置、Redis配置等）
│   │   │   │   ├── controller/    # 控制器层（API接口定义）
│   │   │   │   │   ├── auth/      # 认证相关接口
│   │   │   │   │   ├── common/    # 公共接口（文件上传、用户信息等）
│   │   │   │   │   ├── student/   # 学生端接口（作业、课程、班级等）
│   │   │   │   │   └── teacher/   # 教师端接口（班级管理、课程管理等）
│   │   │   │   ├── dto/           # 数据传输对象（请求/响应封装）
│   │   │   │   ├── entity/        # 实体类（数据库表映射对象）
│   │   │   │   ├── exception/     # 异常处理（全局异常处理、自定义异常）
│   │   │   │   ├── mapper/        # MyBatis映射接口（数据访问层）
│   │   │   │   ├── monitor/       # 监控组件（数据源监控、性能指标等）
│   │   │   │   ├── security/      # 安全相关（JWT工具、权限验证等）
│   │   │   │   ├── service/       # 服务层（业务逻辑实现）
│   │   │   │   │   ├── auth/      # 认证服务（登录、注册、密码管理）
│   │   │   │   │   ├── common/    # 公共服务（文件服务、缓存服务等）
│   │   │   │   │   ├── student/   # 学生服务（作业提交、课程学习等）
│   │   │   │   │   ├── impl/      # 服务实现类
│   │   │   │   │   └── teacher/   # 教师服务（班级管理、作业批改等）
│   │   │   │   └── utils/         # 工具类（日期工具、文件工具、密码工具等）
│   │   │   └── resources/
│   │   │       ├── application.yml # 应用配置文件（数据库、Redis、安全等配置）
│   │   │       ├── db/migration/  # 数据库迁移脚本（Flyway管理的SQL脚本）
│   │   │       └── mapper/        # MyBatis XML映射文件（复杂SQL定义）
│   ├── target/                    # 编译输出目录
│   │   ├── classes/               # 编译后的类文件
│   │   └── education-platform-1.0.0.jar # 打包后的可执行JAR文件
│   └── pom.xml                    # Maven配置（依赖管理、构建配置）
├── Vue/                           # 前端项目
│   └── frontend/
│       ├── e2e/                   # 端到端测试
│       ├── public/                # 静态资源（图标、字体等）
│       ├── src/
│       │   ├── api/               # API接口封装（按功能模块组织）
│       │   ├── assets/            # 资源文件（样式、图片等）
│       │   │   ├── styles/        # 全局样式定义
│       │   │   └── images/        # 图片资源
│       │   ├── components/        # 公共组件
│       │   │   ├── __tests__/     # 组件测试
│       │   │   ├── icons/         # 图标组件
│       │   │   └── layout/        # 布局组件（页面框架）
│       │   ├── router/            # 路由配置（按角色划分路由）
│       │   ├── stores/            # Pinia状态管理（认证状态、用户信息等）
│       │   ├── utils/             # 工具函数（请求封装、日期处理等）
│       │   └── views/             # 页面组件
│       │       ├── auth/          # 认证页面（登录、注册等）
│       │       ├── student/       # 学生端页面
│       │       │   ├── assignments/ # 作业相关页面
│       │       │   ├── classes/   # 班级相关页面
│       │       │   └── resources/ # 资源相关页面
│       │       └── teacher/       # 教师端页面
│       ├── env.d.ts               # 环境变量类型定义
│       ├── eslint.config.ts       # ESLint配置
│       └── package.json           # NPM配置（依赖管理、脚本命令）
├── resource/                      # 资源文件存储
│   ├── file/                      # 文件存储
│   │   ├── courses/               # 课程相关文件（课程封面等）
│   │   └── resources/             # 教学资源（PDF、文档等）
│   ├── photo/                     # 图片存储（用户头像、教学图片等）
│   └── video/                     # 视频存储（课程视频、教学录像等）
├── docs/                          # 项目文档
│   ├── 前后端功能对接分析报告.md    # 前后端对接文档
│   ├── 功能对接状态表.md           # 功能实现状态追踪
│   └── 接口对应表.md              # API接口对应关系
├── logs/                          # 应用日志
├── sql/                           # SQL脚本
│   └── education_platform.sql     # 数据库初始化脚本（表结构、初始数据）
└── uploads/                       # 用户上传文件临时存储
    └── assignments/               # 作业提交文件
```

### 数据库设计

平台采用关系型数据库MySQL存储核心业务数据，主要数据表包括：

#### 用户与权限
- **user**: 用户基本信息（ID、用户名、密码、角色等）
- **role**: 角色定义（教师、学生、管理员）
- **permission**: 权限定义（资源访问权限）
- **user_role**: 用户与角色多对多关系

#### 教学组织
- **class**: 班级信息（班级名称、创建时间、教师ID等）
- **class_student**: 班级学生关联（班级ID、学生ID）
- **course**: 课程信息（课程名称、简介、封面图等）
- **course_class**: 课程与班级关联
- **chapter**: 课程章节（章节名称、排序等）
- **section**: 章节小节（小节名称、内容类型、视频地址等）

#### 教学内容
- **resource**: 教学资源（资源名称、类型、文件路径等）
- **assignment**: 作业信息（标题、描述、截止时间等）
- **assignment_question**: 作业题目（题目内容、类型、分值等）
- **assignment_submission**: 作业提交记录（学生ID、作业ID、提交时间等）
- **assignment_submission_answer**: 作业答案（提交ID、题目ID、答案内容等）
- **exam**: 考试信息（考试名称、时间、总分等）
- **question**: 题库（题目内容、类型、难度等）
- **question_option**: 选择题选项

#### 学习记录
- **learning_record**: 学习进度记录（学生ID、课程ID、学习时长等）
- **resource_favorite**: 资源收藏（学生ID、资源ID）
- **user_ability**: 学生能力评估（学生ID、能力维度、评分等）

## 核心功能模块

### 用户认证与权限管理
- **账户系统**
  - 用户登录/注册/找回密码
  - 多因素认证（密码+短信/邮箱验证码）
  - 基于JWT的Token认证与刷新机制
  - 会话管理与安全退出
- **权限控制**
  - 基于角色的访问控制（RBAC）
  - 精细化的功能权限管理
  - 数据权限隔离（教师只能访问自己创建的内容）
  - 操作日志记录与审计
- **个人中心**
  - 个人信息管理（头像、个人资料更新）
  - 账户安全设置（密码修改、绑定手机/邮箱）
  - 消息通知设置
  - 操作历史记录查询

### 教师端功能

#### 班级管理
- **班级创建与配置**
  - 创建新班级并设置班级基本信息
  - 自定义班级编码与分类标签
  - 设置班级可见性与加入方式
  - 生成班级邀请码/二维码
- **成员管理**
  - 批量导入/添加学生
  - 学生分组管理
  - 成员角色设置（助教、组长等）
  - 成员权限管理
- **班级数据分析**
  - 班级活跃度统计
  - 学习进度跟踪
  - 成绩分布分析
  - 个体表现对比

#### 课程管理
- **课程设计**
  - 创建/编辑课程基本信息
  - 课程封面与简介设置
  - 课程标签与分类管理
  - 课程大纲设计
- **内容建设**
  - 多级章节管理（章-节-小节）
  - 多媒体内容支持（文本、图片、视频、音频）
  - 富文本编辑器（支持代码块、公式等）
  - 内容模板与批量导入
- **资源管理**
  - 上传各类课程资源（PPT、PDF、Word等）
  - 资源分类与标签管理
  - 资源权限设置（可下载/仅预览）
  - 版本管理与更新记录
- **发布与共享**
  - 课程发布与下架控制
  - 面向班级发布课程
  - 章节内容定时发布
  - 多教师协同编辑

#### 作业管理
- **作业创建**
  - 多种题型支持（选择、填空、问答、编程等）
  - 题目难度与分值设置
  - 从题库选择题目
  - 附件与参考资料上传
- **作业设置**
  - 截止时间与提交规则设置
  - 作业可见性控制
  - 抄袭检测设置
  - 自动评分规则配置
- **批改与反馈**
  - 在线批改界面
  - 批量评分工具
  - 详细批注与反馈
  - 评分标准模板
- **数据分析**
  - 提交率与完成率统计
  - 得分分布分析
  - 题目难度分析
  - 常见错误归纳

#### 资源库管理
- **资源上传与组织**
  - 多类型资源支持（文档、图片、视频、音频等）
  - 批量上传与处理
  - 自定义分类与目录结构
  - 元数据管理（标题、描述、标签等）
- **资源共享与权限**
  - 资源共享范围设置（个人、班级、公开）
  - 下载权限控制
  - 版权信息标注
  - 使用统计与分析
- **资源检索**
  - 全文搜索功能
  - 高级筛选（类型、标签、上传时间等）
  - 资源推荐算法
  - 个人收藏与历史记录

#### 考试与评估
- **考试管理**
  - 创建多种类型考试（章节测验、期中/期末考试等）
  - 试卷编排与题目组织
  - 考试时间与规则设置
  - 多版本试卷支持
- **题库建设**
  - 题目录入与管理
  - 题目分类与标签
  - 题目难度评估
  - 题目引用统计
- **在线考试**
  - 考试环境配置（防作弊设置）
  - 自动评分系统
  - 手动评分界面
  - 成绩公布控制
- **成绩分析**
  - 详细成绩报表
  - 班级/个人成绩对比
  - 历史数据对比分析
  - 考试质量评估

#### AI辅助教学
- **内容生成**
  - 课程大纲智能生成
  - 教学内容辅助创作
  - 题目自动生成
  - 教学素材推荐
- **智能评估**
  - 作业自动批改
  - 主观题智能评分
  - 抄袭检测
  - 学习行为分析
- **知识建模**
  - 课程知识图谱构建
  - 学科能力模型
  - 学习路径优化
  - 个性化教学推荐

### 学生端功能

#### 个人学习中心
- **学习仪表盘**
  - 课程学习进度跟踪
  - 作业/考试完成情况
  - 近期学习活动记录
  - 个人能力雷达图
- **课程管理**
  - 已选课程列表
  - 课程学习状态
  - 课程收藏与归档
  - 学习计划制定

#### 课程学习
- **内容学习**
  - 多样化课程内容浏览（文本、图片、视频）
  - 视频播放控制（倍速、断点续播）
  - 笔记与标记功能
  - 内容互动（评论、提问）
- **资源利用**
  - 课程资源下载与阅读
  - 资源收藏与整理
  - 资源搜索与筛选
  - 学习记录同步

#### 作业与考试
- **作业管理**
  - 作业列表（待完成、已完成、已批改）
  - 作业提交与修改
  - 截止时间提醒
  - 批改结果查看
- **在线考试**
  - 考试安排与提醒
  - 在线答题界面
  - 自动保存与提交
  - 成绩查询与分析
- **学习评估**
  - 个人成绩单
  - 学习表现分析
  - 知识掌握程度评估
  - 学习建议与反馈

#### 互动与协作
- **班级互动**
  - 班级信息查看
  - 同学列表与联系
  - 班级公告与活动
  - 小组协作空间
- **讨论区**
  - 课程讨论参与
  - 问题提问与回答
  - 资源分享与交流
  - 教师答疑互动

#### AI学习助手
- **智能推荐**
  - 学习资源个性化推荐
  - 知识点针对性补充
  - 相似题目推荐
  - 学习方法建议
- **学习诊断**
  - 知识掌握程度分析
  - 学习弱点识别
  - 错题归类与分析
  - 学习习惯评估
- **智能辅导**
  - 问题自动解答
  - 知识点解释与示例
  - 学习路径规划
  - 考试复习指导
- **能力图谱**
  - 多维度能力评估
  - 能力发展趋势分析
  - 同伴能力对比
  - 能力提升建议

## 系统架构

### 整体架构
本平台采用前后端分离的架构设计，主要由以下几部分组成：

1. **前端应用**：基于Vue 3的单页面应用（SPA），负责用户界面渲染与交互
2. **后端服务**：Spring Boot应用，提供REST API接口服务
3. **数据存储**：MySQL关系型数据库 + Redis缓存系统
4. **文件存储**：基于MinIO的分布式对象存储系统
5. **AI服务**：基于大模型的智能辅助服务组件

```
┌─────────────┐       ┌─────────────┐      ┌─────────────┐
│  Web浏览器  │ ─────▶│   前端应用   │ ────▶│  反向代理   │
└─────────────┘       │  (Vue 3 SPA) │      │  (Nginx)    │
                      └─────────────┘      └──────┬──────┘
                                                  │
                                                  ▼
┌─────────────┐       ┌─────────────┐      ┌─────────────┐
│ 移动端应用  │ ─────▶│ 统一认证服务 │ ◀────│  后端服务   │
│ (可选)      │       │ (JWT)        │      │ (Spring Boot)│
└─────────────┘       └─────────────┘      └──────┬──────┘
                                                  │
                                                  ▼
                      ┌─────────────┐      ┌─────────────┐
                      │  缓存服务   │ ◀────┤  数据服务   │
                      │  (Redis)    │      │ (MySQL)     │
                      └─────────────┘      └─────────────┘
                                                  │
                                                  ▼
                      ┌─────────────┐      ┌─────────────┐
                      │  文件存储   │ ◀────┤   AI服务    │
                      │  (MinIO)    │      │ (大模型API) │
                      └─────────────┘      └─────────────┘
```

### 技术架构
- **表现层**：基于Vue 3的组件化开发，使用Ant Design Vue和Element Plus构建用户界面
- **应用层**：Spring Boot核心框架，集成各种中间件和服务组件
- **领域层**：核心业务逻辑实现，包括各类Service服务和领域模型
- **数据层**：MyBatis-Plus持久层框架，与MySQL数据库交互
- **基础设施层**：Redis缓存、MinIO对象存储、日志系统等基础服务

### 安全架构
- **认证系统**：基于JWT的无状态认证机制
- **授权控制**：细粒度的RBAC权限模型
- **数据安全**：敏感数据加密、防SQL注入
- **API安全**：HTTPS传输、请求验证、防重放攻击
- **文件安全**：文件类型检测、病毒扫描、访问控制

## 快速开始

### 环境准备
1. **硬件要求**
   - 处理器：双核CPU或更高
   - 内存：至少4GB RAM（推荐8GB以上）
   - 存储：至少10GB可用空间（视数据量可能需要更多）

2. **软件要求**
   - 操作系统：Linux/Windows/MacOS
   - JDK 17+：用于运行后端服务
   - Node.js 16+：用于前端开发与构建
   - MySQL 8.0+：核心数据库
   - Redis 6.0+：缓存系统
   - MinIO：对象存储服务（可选，或使用本地文件系统）

3. **开发工具**
   - IntelliJ IDEA / Eclipse：后端开发
   - Visual Studio Code：前端开发
   - MySQL Workbench：数据库管理
   - Postman：API测试

### 后端环境配置

1. **数据库配置**
   ```bash
   # 创建数据库
   mysql -u root -p
   CREATE DATABASE education_platform DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
   
   # 导入初始数据
   mysql -u root -p education_platform < sql/education_platform.sql
   ```

2. **Redis配置**
   ```bash
   # 启动Redis服务
   redis-server
   
   # 验证Redis连接
   redis-cli ping
   ```

3. **配置文件**
   修改`backend/src/main/resources/application.yml`中的数据库连接、Redis配置等信息

### 后端启动
```bash
# 进入后端目录
cd backend

# 编译打包
mvn clean package -DskipTests

# 开发模式启动
mvn spring-boot:run

# 或生产环境启动
java -jar -Dspring.profiles.active=prod target/education-platform-1.0.0.jar
```

**注意事项**:
- 首次运行会自动执行Flyway数据库迁移脚本
- 默认创建的管理员账号：admin/admin123
- 开发环境下API文档访问地址：http://localhost:8080/doc.html

### 前端环境配置

1. **Node.js环境**
   确保Node.js和npm已正确安装：
   ```bash
   node -v  # 应显示v16.x或更高版本
   npm -v   # 应显示8.x或更高版本
   ```

2. **配置API地址**
   修改`.env.development`或`.env.production`中的API基础URL配置

### 前端启动
```bash
# 进入前端目录
cd Vue/frontend

# 安装依赖
npm install

# 开发模式启动
npm run dev

# 构建生产版本
npm run build

# 预览生产构建
npm run preview
```

**注意事项**:
- 开发服务器默认运行在 http://localhost:5173
- 如需修改端口，可在`vite.config.ts`中配置
- 开发时支持热重载和HMR

## 部署指南

### Docker部署

1. **构建Docker镜像**
   ```bash
   # 后端镜像构建
   cd backend
   docker build -t education-platform-backend:latest .
   
   # 前端镜像构建
   cd ../Vue/frontend
   docker build -t education-platform-frontend:latest .
   ```

2. **使用Docker Compose部署**
   创建`docker-compose.yml`文件：
   ```yaml
   version: '3'
   services:
     mysql:
       image: mysql:8.0
       container_name: edu-mysql
       environment:
         - MYSQL_ROOT_PASSWORD=rootpassword
         - MYSQL_DATABASE=education_platform
       volumes:
         - mysql-data:/var/lib/mysql
         - ./sql:/docker-entrypoint-initdb.d
       ports:
         - "3306:3306"
     
     redis:
       image: redis:6.0
       container_name: edu-redis
       volumes:
         - redis-data:/data
       ports:
         - "6379:6379"
     
     backend:
       image: education-platform-backend:latest
       container_name: edu-backend
       depends_on:
         - mysql
         - redis
       environment:
         - SPRING_PROFILES_ACTIVE=prod
         - SPRING_DATASOURCE_URL=jdbc:mysql://mysql:3306/education_platform
         - SPRING_REDIS_HOST=redis
       ports:
         - "8080:8080"
     
     frontend:
       image: education-platform-frontend:latest
       container_name: edu-frontend
       ports:
         - "80:80"
       depends_on:
         - backend
   
   volumes:
     mysql-data:
     redis-data:
   ```

3. **启动服务**
   ```bash
   docker-compose up -d
   ```

### 传统部署

1. **后端部署**
   - 准备JDK 17+环境
   - 配置MySQL和Redis
   - 构建生产JAR包
   - 使用systemd或nohup启动服务
   ```bash
   nohup java -jar -Dspring.profiles.active=prod backend/target/education-platform-1.0.0.jar > logs/backend.log 2>&1 &
   ```

2. **前端部署**
   - 构建静态文件：`npm run build`
   - 配置Nginx服务器
   ```nginx
   server {
       listen 80;
       server_name your-domain.com;
       
       root /path/to/frontend/dist;
       index index.html;
       
       # 处理SPA路由
       location / {
           try_files $uri $uri/ /index.html;
       }
       
       # API代理
       location /api {
           proxy_pass http://localhost:8080;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```

3. **文件存储配置**
   - 确保文件存储目录权限正确
   - 设置定期备份策略
   - 配置资源访问路径

## 开发规范

### 代码规范
- **前端规范**
  - 使用ESLint和Prettier进行代码格式化和规范检查
  - 遵循Vue 3组件开发最佳实践
  - 使用TypeScript类型定义提高代码可维护性
  - 保持组件单一职责，合理拆分组件结构

- **后端规范**
  - 遵循阿里巴巴Java开发手册规范
  - 采用统一的异常处理机制
  - 使用领域模型分层架构
  - 编写完整的单元测试和集成测试

- **API设计规范**
  - 遵循RESTful API设计原则
  - 使用统一的响应格式
  - 版本控制（如`/api/v1/resource`）
  - 完善的API文档（使用Knife4j）

- **数据库规范**
  - 表名使用小写，单词间用下划线分隔
  - 必须包含主键，优先使用自增ID
  - 字段名清晰表达含义，添加必要注释
  - 合理使用索引优化查询性能

### Git工作流
- **分支策略**
  - `main`/`master`: 主分支，保持稳定可发布状态
  - `develop`: 开发主分支，包含最新功能
  - `feature/*`: 功能分支，用于开发新功能
  - `bugfix/*`: 缺陷修复分支
  - `release/*`: 发布准备分支

- **提交规范**
  - 提交前先进行代码审查
  - 确保代码通过所有测试
  - 提交信息必须清晰描述变更内容
  - 大功能拆分为小的提交，便于回滚和审查

- **代码审查**
  - 所有代码必须经过至少一名团队成员审查
  - 关注代码质量、性能和安全性
  - 使用Pull Request进行正式审查流程

### 提交前缀规范
- `feat`: 新功能（feature）
- `fix`: 修复bug
- `docs`: 文档更新
- `style`: 代码格式调整（不影响代码运行的变动）
- `refactor`: 代码重构（既不是新增功能，也不是修改bug的代码变动）
- `test`: 测试相关（添加、修改测试用例）
- `perf`: 性能优化
- `ci`: CI配置变更
- `build`: 构建系统或外部依赖变更
- `chore`: 其他杂项变更（如更新构建任务、包管理器配置等）

示例：`feat(auth): 添加多因素认证功能`

## 测试策略

### 单元测试
- 使用JUnit 5进行Java单元测试
- 使用Vitest进行Vue组件测试
- 保持测试覆盖率在80%以上

### 集成测试
- 使用Spring Boot Test测试API接口
- 使用TestContainers进行数据库集成测试
- 模拟外部服务依赖

### 端到端测试
- 使用Cypress进行前端端到端测试
- 覆盖关键用户流程和场景
- 自动化UI测试

### 性能测试
- 使用JMeter进行API性能测试
- 负载测试和压力测试
- 数据库查询性能优化

## 贡献指南

1. Fork项目到个人仓库
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 编写代码并通过测试
4. 提交变更 (`git commit -m 'feat: add some amazing feature'`)
5. 推送到分支 (`git push origin feature/amazing-feature`)
6. 提交Pull Request
7. 等待代码审查和合并

### 贡献流程
- 先创建Issue讨论需要添加的功能或修复的问题
- 确认方案后再进行代码开发
- 提交PR时关联相关Issue
- 保持PR简洁，专注于单一功能或修复

## 常见问题

### 环境配置问题
- **Q: 数据库连接失败怎么办？**
  - A: 检查application.yml中的数据库配置是否正确，确保MySQL服务已启动

- **Q: 前端启动报错"Module not found"？**
  - A: 检查依赖是否完整安装，尝试删除node_modules后重新npm install

### 开发问题
- **Q: 如何添加新的API接口？**
  - A: 在对应的Controller中添加新方法，并在Service层实现业务逻辑

- **Q: 如何扩展用户角色和权限？**
  - A: 在数据库role表添加新角色，在permission表添加权限，然后关联到用户

## 更新日志

### v1.0.0 (2025-07-01)
- 初始版本发布
- 实现基础教学管理功能
- 支持课程、作业、资源管理

### v1.1.0 (计划中)
- 增强AI辅助教学功能
- 优化移动端适配
- 添加数据分析与报表功能

## 许可证

MIT License

Copyright (c) 2025 SmartClass Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.