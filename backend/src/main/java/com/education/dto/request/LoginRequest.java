package com.education.dto.request;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import jakarta.validation.constraints.NotBlank;

import jakarta.validation.constraints.Size;
/**
 * 登录请求DTO
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
@Data
@Schema(description = "登录请求参数")
public class LoginRequest {
    
    @Schema(description = "用户名", example = "admin")
    @NotBlank(message = "用户名不能为空")
    @Size(min = 3, max = 50, message = "用户名长度必须在3-50个字符之间")
    private String username;
    
    @Schema(description = "密码", example = "123456")
    @NotBlank(message = "密码不能为空")
    @Size(min = 6, max = 20, message = "密码长度必须在6-20个字符之间")
    private String password;
    
    @Schema(description = "验证码", example = "1234")
    private String captcha;
    
    @Schema(description = "验证码key", example = "captcha_key_123")
    private String captchaKey;
    
    @Schema(description = "记住我", example = "true")
    private Boolean rememberMe = false;
}