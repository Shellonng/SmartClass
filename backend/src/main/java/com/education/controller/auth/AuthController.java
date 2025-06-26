package com.education.controller.auth;

import com.education.dto.common.Result;
import com.education.dto.AuthDTO;
import com.education.service.auth.AuthService;
import com.education.utils.JwtUtils;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.validation.Valid;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;
import java.util.HashMap;
import java.util.Map;

/**
 * 认证控制器
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
@Tag(name = "认证管理", description = "用户登录、注册、密码管理等认证相关接口")
@RestController
@RequestMapping("/auth")
public class AuthController {

    @Autowired
    private AuthService authService;

    @Autowired
    private JwtUtils jwtUtils;

    @Operation(summary = "用户登录", description = "支持学生和教师登录")
    @PostMapping("/login")
    public Result<Object> login(@Valid @RequestBody AuthDTO.LoginRequest request, HttpServletRequest httpRequest) {
        try {
            System.out.println("🔐 收到登录请求:");
            System.out.println("  - 用户名: " + request.getUsername());
            System.out.println("  - 请求路径: " + httpRequest.getRequestURI());
            System.out.println("  - 请求方法: " + httpRequest.getMethod());
            System.out.println("  - Origin: " + httpRequest.getHeader("Origin"));
            System.out.println("  - Content-Type: " + httpRequest.getContentType());
            
            AuthDTO.LoginResponse authResponse = authService.login(request);
            
            // 构建前端期望的响应结构
            Map<String, Object> data = new HashMap<>();
            data.put("token", authResponse.getToken());
            
            Map<String, Object> userInfo = new HashMap<>();
            userInfo.put("id", authResponse.getUserId());
            userInfo.put("username", authResponse.getUsername());
            userInfo.put("realName", authResponse.getRealName());
            userInfo.put("email", authResponse.getEmail());
            userInfo.put("role", authResponse.getUserType());
            userInfo.put("avatar", null); // 如果没有头像字段，设为null
            
            data.put("userInfo", userInfo);
            
            System.out.println("✅ 登录成功，返回响应");
            return Result.success(data);
        } catch (Exception e) {
            System.out.println("❌ 登录失败: " + e.getMessage());
            throw e; // 让全局异常处理器处理
        }
    }

    @Operation(summary = "用户登出", description = "清除用户登录状态")
    @PostMapping("/logout")
    public Result<Void> logout(@RequestHeader("Authorization") String token) {
        try {
            // 移除Bearer前缀
            if (token.startsWith("Bearer ")) {
                token = token.substring(7);
            }
            authService.logout(token);
            return Result.success();
        } catch (Exception e) {
            return Result.error("登出失败: " + e.getMessage());
        }
    }

    @Operation(summary = "刷新Token", description = "使用刷新Token获取新的访问Token")
    @PostMapping("/refresh")
    public Result<AuthDTO.LoginResponse> refreshToken(@Valid @RequestBody AuthDTO.RefreshTokenRequest request) {
        try {
            AuthDTO.LoginResponse response = authService.refreshToken(request);
            return Result.success(response);
        } catch (Exception e) {
            return Result.error("Token刷新失败: " + e.getMessage());
        }
    }

    @Operation(summary = "修改密码", description = "用户修改登录密码")
    @PostMapping("/change-password")
    public Result<Void> changePassword(@Valid @RequestBody AuthDTO.ChangePasswordRequest request, @RequestHeader("Authorization") String token) {
        try {
            // 移除Bearer前缀
            if (token.startsWith("Bearer ")) {
                token = token.substring(7);
            }
            // 从token中获取userId
            Long userId = jwtUtils.getUserIdFromToken(token);
            authService.changePassword(request, userId);
            return Result.success();
        } catch (Exception e) {
            return Result.error("密码修改失败: " + e.getMessage());
        }
    }

    @Operation(summary = "发送重置密码邮件", description = "向用户邮箱发送密码重置验证码")
    @PostMapping("/send-reset-email")
    public Result<Void> sendResetPasswordEmail(@RequestParam String email) {
        try {
            authService.sendResetPasswordEmail(email);
            return Result.success();
        } catch (Exception e) {
            return Result.error("发送重置密码邮件失败: " + e.getMessage());
        }
    }

    @Operation(summary = "重置密码", description = "使用邮箱验证码重置密码")
    @PostMapping("/reset-password")
    public Result<Void> resetPassword(@Valid @RequestBody AuthDTO.ResetPasswordRequest request) {
        try {
            authService.resetPassword(request);
            return Result.success();
        } catch (Exception e) {
            return Result.error("重置密码失败: " + e.getMessage());
        }
    }

    @Operation(summary = "获取验证码", description = "获取图形验证码")
    @GetMapping("/captcha")
    public Result<Map<String, String>> getCaptcha() {
        try {
            Map<String, String> captchaInfo = (Map<String, String>) authService.generateCaptcha();
            return Result.success(captchaInfo);
        } catch (Exception e) {
            return Result.error("获取验证码失败: " + e.getMessage());
        }
    }

    @Operation(summary = "用户注册", description = "支持学生和教师注册")
    @PostMapping("/register")
    public Result<Object> register(@Valid @RequestBody AuthDTO.RegisterRequest request) {
        try {
            System.out.println("📝 收到注册请求:");
            System.out.println("  - 用户名: " + request.getUsername());
            System.out.println("  - 邮箱: " + request.getEmail());
            System.out.println("  - 用户角色: " + request.getRole());
            
            AuthDTO.LoginResponse authResponse = authService.register(request);
            
            // 构建前端期望的响应结构
            Map<String, Object> data = new HashMap<>();
            data.put("token", authResponse.getToken());
            
            Map<String, Object> userInfo = new HashMap<>();
            userInfo.put("id", authResponse.getUserId());
            userInfo.put("username", authResponse.getUsername());
            userInfo.put("realName", authResponse.getRealName());
            userInfo.put("email", authResponse.getEmail());
            userInfo.put("role", authResponse.getUserType());
            userInfo.put("avatar", null); // 如果没有头像字段，设为null
            
            data.put("userInfo", userInfo);
            
            System.out.println("✅ 注册成功，返回响应");
            return Result.success(data);
        } catch (Exception e) {
            System.out.println("❌ 注册失败: " + e.getMessage());
            return Result.error("注册失败: " + e.getMessage());
        }
    }

    @Operation(summary = "获取当前用户信息", description = "获取当前登录用户的基本信息")
    @GetMapping("/user-info")
    public Result<Object> getCurrentUserInfo(@RequestHeader("Authorization") String token) {
        try {
            // 移除Bearer前缀
            if (token.startsWith("Bearer ")) {
                token = token.substring(7);
            }
            // 从token中解析userId
            Long userId = jwtUtils.getUserIdFromToken(token);
            Object userInfo = authService.getCurrentUserInfo(userId);
            return Result.success(userInfo);
        } catch (Exception e) {
            return Result.error("获取用户信息失败: " + e.getMessage());
        }
    }
}