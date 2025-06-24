# AI赋能教育管理与学习辅助平台

## 项目概述

本项目是一个面向高校教师和大学生的智能化教育管理平台，融合AI和大数据技术，实现"智能化生产、结构化管理、个性化学习"。

## 技术栈

### 前端
- Vue3 + Vite
- Pinia (状态管理)
- Vue Router (路由管理)
- Ant Design Vue (UI组件库)
- Axios (HTTP客户端)
- ECharts (数据可视化)

### 后端
- Spring Boot 2.7+
- Spring Security + JWT (安全认证)
- MyBatis-Plus (ORM框架)
- Spring Data Redis (缓存)
- MySQL 8.0+ (主数据库)
- Redis 6.0+ (缓存数据库)

### 部署
- Docker + Docker Compose
- Nginx (反向代理)

## 项目结构

```
大二下实训3.0/
├── Vue/                    # 前端项目
│   ├── src/
│   │   ├── components/     # 公共组件
│   │   ├── views/         # 页面组件
│   │   ├── router/        # 路由配置
│   │   ├── store/         # Pinia状态管理
│   │   ├── api/           # API接口
│   │   ├── utils/         # 工具函数
│   │   └── assets/        # 静态资源
│   ├── package.json
│   └── vite.config.js
├── SB/                     # Spring Boot后端
│   ├── src/main/java/
│   │   └── com/education/
│   │       ├── controller/ # 控制器层
│   │       ├── service/    # 服务层
│   │       ├── mapper/     # 数据访问层
│   │       ├── entity/     # 实体类
│   │       ├── dto/        # 数据传输对象
│   │       ├── config/     # 配置类
│   │       └── utils/      # 工具类
│   ├── src/main/resources/
│   │   ├── mapper/         # MyBatis映射文件
│   │   └── application.yml # 配置文件
│   └── pom.xml
├── DB/                     # 数据库相关
│   ├── init.sql           # 初始化脚本
│   ├── tables/            # 建表脚本
│   └── data/              # 测试数据
└── docs/                   # 项目文档
    ├── api/               # API文档
    ├── database/          # 数据库设计文档
    └── frontend/          # 前端设计文档
```

## 核心功能模块

### 用户管理
- 用户注册/登录
- 角色权限管理（教师/学生）
- 个人信息管理

### 教师端功能
- 班级管理
- 学生管理
- 课程管理
- 任务管理
- 成绩管理
- 资源管理
- 知识图谱（AI扩展）
- AI创新功能（预留）

### 学生端功能
- 课程学习
- 作业提交
- 成绩查看
- 资源下载
- 知识图谱（AI扩展）
- AI学习辅助（预留）

## 快速开始

### 环境要求
- Node.js 16+
- Java 11+
- MySQL 8.0+
- Redis 6.0+

### 启动步骤

1. 启动数据库服务
```bash
# 启动MySQL和Redis
docker-compose up -d mysql redis
```

2. 启动后端服务
```bash
cd SB
mvn spring-boot:run
```

3. 启动前端服务
```bash
cd Vue
npm install
npm run dev
```

## 开发规范

### 代码规范
- 前端：ESLint + Prettier
- 后端：阿里巴巴Java开发手册
- 数据库：统一命名规范

### Git规范
- feat: 新功能
- fix: 修复bug
- docs: 文档更新
- style: 代码格式调整
- refactor: 代码重构
- test: 测试相关

## 部署说明

详见 `docs/deployment.md`

## 贡献指南

详见 `docs/contributing.md`

## 许可证

MIT License