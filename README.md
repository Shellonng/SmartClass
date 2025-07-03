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
│   │   │   │   ├── aspect/        # AOP切面
│   │   │   │   ├── config/        # 配置类
│   │   │   │   ├── controller/    # 控制器层
│   │   │   │   │   ├── auth/      # 认证相关
│   │   │   │   │   ├── common/    # 公共接口
│   │   │   │   │   ├── student/   # 学生端接口
│   │   │   │   │   └── teacher/   # 教师端接口
│   │   │   │   ├── dto/           # 数据传输对象
│   │   │   │   ├── entity/        # 实体类
│   │   │   │   ├── exception/     # 异常处理
│   │   │   │   ├── mapper/        # MyBatis映射接口
│   │   │   │   ├── monitor/       # 监控组件
│   │   │   │   ├── security/      # 安全相关
│   │   │   │   ├── service/       # 服务层
│   │   │   │   │   ├── auth/      # 认证服务
│   │   │   │   │   ├── common/    # 公共服务
│   │   │   │   │   ├── student/   # 学生服务
│   │   │   │   │   └── teacher/   # 教师服务
│   │   │   │   └── utils/         # 工具类
│   │   │   └── resources/
│   │   │       ├── application.yml # 配置文件
│   │   │       ├── db/migration/  # 数据库迁移脚本
│   │   │       └── mapper/        # MyBatis XML映射文件
│   └── pom.xml                    # Maven配置
├── Vue/                           # 前端项目
│   └── frontend/
│       ├── public/                # 静态资源
│       ├── src/
│       │   ├── api/               # API接口
│       │   ├── assets/            # 资源文件
│       │   ├── components/        # 公共组件
│       │   │   └── layout/        # 布局组件
│       │   ├── router/            # 路由配置
│       │   ├── stores/            # Pinia状态管理
│       │   ├── utils/             # 工具函数
│       │   └── views/             # 页面组件
│       │       ├── auth/          # 认证页面
│       │       ├── student/       # 学生端页面
│       │       └── teacher/       # 教师端页面
│       └── package.json           # NPM配置
├── resource/                      # 资源文件存储
│   ├── file/                      # 文件存储
│   │   ├── courses/               # 课程相关文件
│   │   └── resources/             # 教学资源
│   ├── photo/                     # 图片存储
│   └── video/                     # 视频存储
├── docs/                          # 项目文档
└── sql/                           # SQL脚本
    └── education_platform.sql     # 数据库初始化脚本
```

## 核心功能模块

### 用户认证与权限管理
- 用户登录/注册
- 基于JWT的认证授权
- 角色权限管理（教师/学生）
- 个人信息管理

### 教师端功能
- **班级管理**: 创建班级、管理班级成员、查看班级统计信息
- **课程管理**: 创建/编辑课程、管理课程章节和内容、上传课程资源
- **作业管理**: 创建作业、设置截止时间、批改作业、查看提交情况
- **资源管理**: 上传/管理教学资源、分类整理
- **考试管理**: 创建考试、题库管理、成绩统计
- **AI辅助功能**: 教学内容生成、自动评分、知识图谱

### 学生端功能
- **课程学习**: 浏览课程内容、观看视频、下载资源
- **作业提交**: 查看作业要求、在线提交作业、查看批改结果
- **资源访问**: 浏览/下载教学资源、收藏管理
- **成绩查看**: 查看作业/考试成绩、个人学习进度
- **AI学习辅助**: 智能推荐、个性化学习、能力图谱

## 快速开始

### 环境准备
1. 安装JDK 17+
2. 安装Node.js 16+
3. 安装并启动MySQL 8.0+
4. 安装并启动Redis 6.0+

### 后端启动
```bash
# 进入后端目录
cd backend

# 编译打包
mvn clean package

# 运行应用
java -jar target/education-platform-1.0.0.jar
```

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
```

## 开发规范

### 代码规范
- **前端**: 使用ESLint和Prettier进行代码格式化和规范检查
- **后端**: 遵循阿里巴巴Java开发手册规范
- **API设计**: RESTful API设计规范
- **提交规范**: 使用语义化的提交信息

### Git工作流
- 使用功能分支开发新功能
- 提交前先进行代码审查
- 提交信息必须清晰描述变更内容

### 提交前缀规范
- `feat`: 新功能
- `fix`: 修复bug
- `docs`: 文档更新
- `style`: 代码格式调整
- `refactor`: 代码重构
- `test`: 测试相关
- `perf`: 性能优化
- `ci`: CI配置变更
- `build`: 构建系统变更

## 贡献指南

1. Fork项目
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交变更 (`git commit -m 'feat: add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 提交Pull Request

## 许可证

MIT License