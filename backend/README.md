# AI赋能教育管理与学习辅助平台 - 后端

## 项目简介

本项目是一个基于 Spring Boot 的教育管理与学习辅助平台后端，提供了用户认证、课程管理、任务管理、资源管理等功能。

## 技术栈

- Java 17
- Spring Boot 3.1.8
- Spring Security
- MyBatis-Plus
- Redis
- MySQL
- JWT

## 开发环境要求

- JDK 17+
- Maven 3.8+
- MySQL 8.0+
- Redis 6.0+

## 快速开始

### 1. 数据库初始化

执行 `sql/database_init_simple.sql` 脚本初始化数据库。

```bash
mysql -u your_username -p < ../sql/database_init_simple.sql
```

### 2. 修改配置

根据实际环境修改 `src/main/resources/application.yml` 中的数据库和 Redis 配置。

### 3. 构建项目

```bash
mvn clean package -DskipTests
```

### 4. 启动项目

#### 方式一：使用 Maven 启动

```bash
mvn spring-boot:run -Dspring-boot.run.profiles=dev
```

#### 方式二：使用 Java 命令启动

```bash
java -jar -Dspring.profiles.active=dev target/education-platform-1.0.0.jar
```

#### 方式三：使用启动脚本

Windows:
```bash
start.bat
```

### 5. 访问接口文档

启动成功后，访问 Swagger 文档：

```
http://localhost:8080/api/doc.html
```

## 项目结构

```
src/main/java/com/education/
├── aspect/           # 切面
├── config/           # 配置类
├── controller/       # 控制器
├── dto/              # 数据传输对象
├── entity/           # 实体类
├── exception/        # 异常处理
├── mapper/           # MyBatis 映射接口
├── monitor/          # 监控相关
├── security/         # 安全相关
├── service/          # 服务接口及实现
└── utils/            # 工具类
```

## API 接口

### 认证接口

- 登录: `POST /api/auth/login`
- 注册: `POST /api/auth/register`
- 登出: `POST /api/auth/logout`
- 刷新令牌: `POST /api/auth/refresh`
- 修改密码: `POST /api/auth/change-password`
- 重置密码: `POST /api/auth/reset-password`

### 用户接口

- 获取用户信息: `GET /api/auth/user-info`

更多接口请参考 Swagger 文档。

## 注意事项

- 开发环境使用 `dev` 配置文件
- 生产环境使用 `prod` 配置文件
- 默认用户名/密码: admin/admin 