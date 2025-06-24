package com.education.dto.response;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.time.LocalDateTime;
import java.util.List;

/**
 * 登录响应DTO
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
@Data
@Schema(description = "登录响应数据")
public class LoginResponse {
    
    @Schema(description = "访问令牌", example = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...")
    private String accessToken;
    
    @Schema(description = "刷新令牌", example = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...")
    private String refreshToken;
    
    @Schema(description = "令牌类型", example = "Bearer")
    private String tokenType = "Bearer";
    
    @Schema(description = "令牌过期时间（秒）", example = "7200")
    private Long expiresIn;
    
    @Schema(description = "用户ID", example = "1")
    private Long userId;
    
    @Schema(description = "用户名", example = "admin")
    private String username;
    
    @Schema(description = "真实姓名", example = "管理员")
    private String realName;
    
    @Schema(description = "用户角色", example = "ADMIN")
    private String role;
    
    @Schema(description = "用户权限列表", example = "[\"user:read\", \"user:write\"]")
    private List<String> permissions;
    
    @Schema(description = "头像URL", example = "https://example.com/avatar.jpg")
    private String avatar;
    
    @Schema(description = "邮箱", example = "admin@example.com")
    private String email;
    
    @Schema(description = "手机号", example = "13800138000")
    private String phone;
    
    @Schema(description = "最后登录时间", example = "2024-01-01T12:00:00")
    private LocalDateTime lastLoginTime;
    
    @Schema(description = "是否首次登录", example = "false")
    private Boolean firstLogin = false;
    
    @Schema(description = "是否需要修改密码", example = "false")
    private Boolean needChangePassword = false;
    
    @Schema(description = "用户状态", example = "ACTIVE")
    private String status;
}