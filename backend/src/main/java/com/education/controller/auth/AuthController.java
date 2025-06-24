package com.education.controller.auth;

import com.education.dto.common.Result;
import com.education.dto.AuthDTO;
import com.education.service.auth.AuthService;
import com.education.utils.JwtUtils;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

/**
 * 认证控制器
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
@Tag(name = "认证管理", description = "用户登录、注册、密码管理等认证相关接口")
@RestController
@RequestMapping("/api/auth")
public class AuthController {

    @Autowired
    private AuthService authService;

    @Autowired
    private JwtUtils jwtUtils;

    @Operation(summary = "用户登录", description = "支持学生和教师登录")
    @PostMapping("/login")
    public Result<AuthDTO.LoginResponse> login(@Valid @RequestBody AuthDTO.LoginRequest request) {
        try {
            AuthDTO.LoginResponse response = authService.login(request);
            return Result.success(response);
        } catch (Exception e) {
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
    public Result<Object> getCaptcha() {
        try {
            // TODO: 实现验证码生成逻辑
            // AuthService接口中没有generateCaptcha方法，需要添加或使用其他方式
            return Result.error("验证码功能暂未实现");
        } catch (Exception e) {
            return Result.error("获取验证码失败: " + e.getMessage());
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