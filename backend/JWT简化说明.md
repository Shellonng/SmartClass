# SmartClass 项目JWT简化说明

## 简化目标

根据用户要求，将原本复杂的JWT认证系统简化为基于Session的简单认证机制，降低系统复杂度。

## 已删除的组件

### 1. JWT相关文件
- `JwtUtils.java` - JWT工具类
- `JwtAuthenticationFilter.java` - JWT认证过滤器  
- `JwtAuthenticationEntryPoint.java` - JWT认证入口点
- `UserDetailsServiceImpl.java` - Spring Security用户详情服务

### 2. 复杂功能组件
- `PermissionAspect.java` - 权限切面（依赖JWT）
- `SecurityUtils.java` - 安全工具类
- `DashboardController.java`（学生和教师端） - 仪表盘控制器
- 邮件服务相关功能（保留接口但标记为废弃）

### 3. 数据库表（如需要）
- `user_session` - JWT Token管理表
- `password_reset` - 密码重置表
- `captcha` - 验证码表

## 保留的核心功能

### 1. 基础认证
- 用户登录/注册
- 密码管理（修改密码）
- 基于Session的用户状态管理

### 2. 用户角色管理
- 学生角色（STUDENT）
- 教师角色（TEACHER）
- 角色验证和权限控制

### 3. 基础配置
- Redis配置（保留）
- 数据库配置
- CORS配置
- 密码加密

## 新的认证机制

### Session管理
- 登录成功后将用户信息存储到HttpSession
- 通过Session验证用户登录状态
- 登出时清除Session

### 简化的API响应
```json
{
  "code": 200,
  "message": "success",
  "data": {
    "userInfo": {
      "id": 1,
      "username": "student1",
      "realName": "王同学",
      "email": "student1@example.com",
      "role": "student",
      "avatar": null
    },
    "sessionId": "session-id-here"
  }
}
```

## 文件结构变化

### 简化后的认证模块
```
src/main/java/com/education/
├── controller/auth/
│   └── AuthController.java          # 简化版认证控制器
├── service/auth/
│   ├── AuthService.java             # 简化版认证服务接口
│   └── impl/
│       └── AuthServiceImpl.java     # 简化版认证服务实现
├── dto/
│   └── AuthDTO.java                 # 简化版认证DTO
└── config/
    └── SecurityConfig.java          # 简化版安全配置
```

### 主要API接口
- `POST /auth/login` - 用户登录
- `POST /auth/logout` - 用户登出  
- `POST /auth/register` - 用户注册
- `GET /auth/user-info` - 获取当前用户信息
- `POST /auth/change-password` - 修改密码

## 兼容性处理

为了保持与现有代码的兼容性，保留了原有的DTO类和方法，但标记为`@Deprecated`：

- `AuthDTO.LoginResponse` - 兼容原有登录响应
- `AuthDTO.RefreshTokenRequest` - 标记为废弃
- `AuthDTO.ResetPasswordRequest` - 标记为废弃
- 各种废弃的服务方法

## 数据库简化

### 保留的核心表
- `user` - 用户基本信息
- `student` - 学生详细信息
- `teacher` - 教师详细信息

### 测试数据
- 管理员：admin / 123456
- 教师：teacher1, teacher2 / 123456
- 学生：student1, student2, student3 / 123456

## 使用方式

### 前端调用示例
```javascript
// 登录
const loginResponse = await fetch('/auth/login', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    username: 'student1',
    password: '123456',
    role: 'STUDENT'
  })
});

// 后续请求会自动携带Session Cookie
const userInfo = await fetch('/auth/user-info');
```

### 后端Session验证
```java
@GetMapping("/some-api")
public Result<Object> someApi(HttpServletRequest request) {
    HttpSession session = request.getSession(false);
    if (session == null) {
        return Result.error("用户未登录");
    }
    
    Long userId = (Long) session.getAttribute("userId");
    String role = (String) session.getAttribute("role");
    
    // 业务逻辑处理
    return Result.success(data);
}
```

## 优势

1. **简单易懂** - 去除了JWT的复杂性
2. **易于调试** - Session状态容易查看和管理
3. **降低依赖** - 减少了外部库依赖
4. **快速开发** - 专注于业务逻辑而非认证机制

## 注意事项

1. Session基于服务器内存，重启服务会丢失登录状态
2. 集群部署需要考虑Session共享问题
3. 移动端可能需要额外的Token机制
4. 如需要无状态认证，后续可以重新引入JWT

## 下一步

项目已简化完成，可以：
1. 启动项目测试基础认证功能
2. 开始开发具体的业务功能
3. 根据需要逐步添加复杂功能 