package com.education.controller.auth;

import com.education.dto.common.Result;
import com.education.dto.AuthDTO;
import com.education.service.auth.AuthService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpSession;
import jakarta.validation.Valid;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;
import java.util.HashMap;
import java.util.Map;

/**
 * 认证控制器 - 简化版（基于Session）
 * 
 * @author Education Platform Team
 * @version 1.0.0-simplified
 * @since 2024
 */
@Tag(name = "认证管理", description = "用户登录、注册等基本认证接口")
@RestController
@RequestMapping("/auth")
public class AuthController {

    @Autowired
    private AuthService authService;

    @Operation(summary = "用户登录", description = "支持学生和教师登录")
    @PostMapping("/login")
    public Result<Object> login(@Valid @RequestBody AuthDTO.LoginRequest request, HttpServletRequest httpRequest) {
        try {
            System.out.println("🔐 收到登录请求:");
            System.out.println("  - 用户名: " + request.getUsername());
            System.out.println("  - 请求路径: " + httpRequest.getRequestURI());
            System.out.println("  - 请求头: " + httpRequest.getHeader("Cookie"));
            System.out.println("  - User-Agent: " + httpRequest.getHeader("User-Agent"));
            
            // 调用简化的登录服务
            AuthDTO.SimpleLoginResponse authResponse = authService.simpleLogin(request);
            
            // 将用户信息存储到Session中
            HttpSession session = httpRequest.getSession(true); // 确保创建新会话
            session.setAttribute("userId", authResponse.getUserId());
            session.setAttribute("username", authResponse.getUsername());
            session.setAttribute("role", authResponse.getUserType());
            session.setAttribute("realName", authResponse.getRealName());
            session.setAttribute("email", authResponse.getEmail());
            
            // 设置会话超时时间为24小时
            session.setMaxInactiveInterval(86400);
            
            System.out.println("✅ 会话ID: " + session.getId());
            System.out.println("✅ 会话属性设置:");
            System.out.println("  - userId: " + session.getAttribute("userId"));
            System.out.println("  - username: " + session.getAttribute("username"));
            System.out.println("  - role: " + session.getAttribute("role"));
            System.out.println("  - realName: " + session.getAttribute("realName"));
            System.out.println("  - email: " + session.getAttribute("email"));
            System.out.println("  - 会话超时时间: " + session.getMaxInactiveInterval() + "秒");
            
            // 构建前端期望的响应结构
            Map<String, Object> data = new HashMap<>();
            
            Map<String, Object> userInfo = new HashMap<>();
            userInfo.put("id", authResponse.getUserId());
            userInfo.put("username", authResponse.getUsername());
            userInfo.put("realName", authResponse.getRealName());
            userInfo.put("email", authResponse.getEmail());
            userInfo.put("role", authResponse.getUserType());
            userInfo.put("avatar", null);
            
            data.put("userInfo", userInfo);
            data.put("sessionId", session.getId()); // 返回sessionId作为身份标识
            
            System.out.println("✅ 登录成功，用户存储到Session中");
            System.out.println("✅ 返回用户信息: " + userInfo);
            
            // 输出响应头信息
            System.out.println("✅ 响应头将包含Set-Cookie: JSESSIONID=" + session.getId());
            
            return Result.success(data);
        } catch (Exception e) {
            System.out.println("❌ 登录失败: " + e.getMessage());
            return Result.error("登录失败: " + e.getMessage());
        }
    }

    @Operation(summary = "用户登出", description = "清除用户登录状态")
    @PostMapping("/logout")
    public Result<Void> logout(HttpServletRequest request) {
        try {
            HttpSession session = request.getSession(false);
            if (session != null) {
                session.invalidate(); // 清除Session
                System.out.println("✅ 用户登出成功，Session已清除");
            }
            return Result.success();
        } catch (Exception e) {
            return Result.error("登出失败: " + e.getMessage());
        }
    }

    @Operation(summary = "用户注册", description = "新用户注册，注册成功后自动登录")
    @PostMapping("/register")
    public Result<AuthDTO.SimpleLoginResponse> register(
            @Valid @RequestBody AuthDTO.RegisterRequest request,
            HttpServletRequest httpRequest) {
        try {
            authService.simpleRegister(request);
            
            // 注册成功后自动登录
            AuthDTO.LoginRequest loginRequest = new AuthDTO.LoginRequest();
            loginRequest.setUsername(request.getUsername());
            loginRequest.setPassword(request.getPassword());
            
            AuthDTO.SimpleLoginResponse response = authService.simpleLogin(loginRequest);
            
            // 设置Session
            HttpSession session = httpRequest.getSession();
            session.setAttribute("userId", response.getUserId());
            session.setAttribute("username", response.getUsername());
            session.setAttribute("userType", response.getUserType());
            
            return Result.success(response);
        } catch (Exception e) {
            return Result.error("注册失败: " + e.getMessage());
        }
    }

    @Operation(summary = "获取当前用户信息", description = "获取当前登录用户的基本信息")
    @GetMapping("/user-info")
    public Result<Object> getCurrentUserInfo(HttpServletRequest request) {
        try {
            System.out.println("📋 获取用户信息请求:");
            System.out.println("  - 请求路径: " + request.getRequestURI());
            System.out.println("  - 请求方法: " + request.getMethod());
            System.out.println("  - 请求头Cookie: " + request.getHeader("Cookie"));
            System.out.println("  - User-Agent: " + request.getHeader("User-Agent"));
            
            HttpSession session = request.getSession(false);
            if (session == null) {
                System.out.println("❌ 未找到有效会话");
                return Result.error("用户未登录");
            }
            
            System.out.println("✅ 找到会话: " + session.getId());
            
            // 从Session中获取用户信息
            Long userId = (Long) session.getAttribute("userId");
            String username = (String) session.getAttribute("username");
            String role = (String) session.getAttribute("role");
            String realName = (String) session.getAttribute("realName");
            String email = (String) session.getAttribute("email");
            
            System.out.println("  - 会话属性:");
            System.out.println("    - userId: " + userId);
            System.out.println("    - username: " + username);
            System.out.println("    - role: " + role);
            System.out.println("    - realName: " + realName);
            System.out.println("    - email: " + email);
            
            if (userId == null) {
                System.out.println("❌ 会话中没有userId属性");
                return Result.error("用户未登录");
            }
            
            Map<String, Object> userInfo = new HashMap<>();
            userInfo.put("id", userId);
            userInfo.put("username", username);
            userInfo.put("realName", realName);
            userInfo.put("email", email);
            userInfo.put("role", role);
            userInfo.put("avatar", null);
            
            System.out.println("✅ 成功返回用户信息: " + userInfo);
            return Result.success(userInfo);
        } catch (Exception e) {
            System.out.println("❌ 获取用户信息失败: " + e.getMessage());
            e.printStackTrace();
            return Result.error("获取用户信息失败: " + e.getMessage());
        }
    }

    @Operation(summary = "修改密码", description = "用户修改登录密码")
    @PostMapping("/change-password")
    public Result<Void> changePassword(@Valid @RequestBody AuthDTO.ChangePasswordRequest request, HttpServletRequest httpRequest) {
        try {
            HttpSession session = httpRequest.getSession(false);
            if (session == null) {
                return Result.error("用户未登录");
            }
            
            Long userId = (Long) session.getAttribute("userId");
            if (userId == null) {
                return Result.error("用户未登录");
            }
            
            authService.changePassword(userId, request);
            return Result.success();
        } catch (Exception e) {
            return Result.error("密码修改失败: " + e.getMessage());
        }
    }
} 