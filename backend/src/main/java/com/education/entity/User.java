package com.education.entity;

import com.baomidou.mybatisplus.annotation.*;
import com.fasterxml.jackson.annotation.JsonFormat;
import com.fasterxml.jackson.annotation.JsonIgnore;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.experimental.Accessors;

import jakarta.validation.constraints.Email;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;
import java.io.Serializable;
import java.time.LocalDateTime;

/**
 * 用户实体类
 * 
 * @author Education Platform Team
 * @version 1.0.0
 */
@Data
@EqualsAndHashCode(callSuper = false)
@Accessors(chain = true)
@TableName("user")
@Schema(description = "用户信息")
public class User implements Serializable {

    private static final long serialVersionUID = 1L;

    @Schema(description = "用户ID")
    @TableId(value = "id", type = IdType.AUTO)
    private Long id;

    @Schema(description = "用户名", example = "admin")
    @TableField("username")
    @NotBlank(message = "用户名不能为空")
    @Size(min = 3, max = 50, message = "用户名长度必须在3-50个字符之间")
    @Pattern(regexp = "^[a-zA-Z0-9_]+$", message = "用户名只能包含字母、数字和下划线")
    private String username;

    @Schema(description = "密码")
    @TableField("password")
    @JsonIgnore
    @NotBlank(message = "密码不能为空")
    @Size(min = 6, max = 100, message = "密码长度必须在6-100个字符之间")
    private String password;

    @Schema(description = "邮箱", example = "admin@example.com")
    @TableField("email")
    @Email(message = "邮箱格式不正确")
    @Size(max = 100, message = "邮箱长度不能超过100个字符")
    private String email;

    @Schema(description = "手机号", example = "13800138000")
    @TableField(exist = false)
    @Pattern(regexp = "^1[3-9]\\d{9}$", message = "手机号格式不正确")
    private String phone;

    @Schema(description = "真实姓名", example = "张三")
    @TableField("real_name")
    @Size(max = 50, message = "真实姓名长度不能超过50个字符")
    private String realName;

    @Schema(description = "头像URL")
    @TableField("avatar")
    @Size(max = 500, message = "头像URL长度不能超过500个字符")
    private String avatar;

    @Schema(description = "用户角色", example = "STUDENT")
    @TableField("role")
    @NotBlank(message = "用户角色不能为空")
    private String role;

    @Schema(description = "账户状态", example = "ACTIVE")
    @TableField("status")
    private String status;

    @Schema(description = "创建时间")
    @TableField(value = "create_time", fill = FieldFill.INSERT)
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime createTime;

    @Schema(description = "更新时间")
    @TableField(value = "update_time", fill = FieldFill.INSERT_UPDATE)
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime updateTime;

    /**
     * 用户角色枚举
     */
    public enum Role {
        ADMIN("ADMIN", "管理员"),
        TEACHER("TEACHER", "教师"),
        STUDENT("STUDENT", "学生");

        private final String code;
        private final String description;

        Role(String code, String description) {
            this.code = code;
            this.description = description;
        }

        public String getCode() {
            return code;
        }

        public String getDescription() {
            return description;
        }

        public static Role fromCode(String code) {
            for (Role role : values()) {
                if (role.code.equals(code)) {
                    return role;
                }
            }
            throw new IllegalArgumentException("未知的用户角色: " + code);
        }
    }

    /**
     * 用户状态枚举
     */
    public enum Status {
        ACTIVE("ACTIVE", "正常"),
        DISABLED("DISABLED", "禁用"),
        LOCKED("LOCKED", "锁定"),
        PENDING("PENDING", "待激活");

        private final String code;
        private final String description;

        Status(String code, String description) {
            this.code = code;
            this.description = description;
        }

        public String getCode() {
            return code;
        }

        public String getDescription() {
            return description;
        }

        public static Status fromCode(String code) {
            for (Status status : values()) {
                if (status.code.equals(code)) {
                    return status;
                }
            }
            throw new IllegalArgumentException("未知的用户状态: " + code);
        }
    }

    /**
     * 判断是否为管理员
     * 
     * @return 是否为管理员
     */
    public boolean isAdmin() {
        return Role.ADMIN.getCode().equals(this.role);
    }

    /**
     * 判断是否为教师
     * 
     * @return 是否为教师
     */
    public boolean isTeacher() {
        return Role.TEACHER.getCode().equals(this.role);
    }

    /**
     * 判断是否为学生
     * 
     * @return 是否为学生
     */
    public boolean isStudent() {
        return Role.STUDENT.getCode().equals(this.role);
    }

    /**
     * 判断账户是否正常
     * 
     * @return 账户是否正常
     */
    public boolean isActive() {
        return Status.ACTIVE.getCode().equals(this.status);
    }

    /**
     * 判断账户是否被禁用
     * 
     * @return 账户是否被禁用
     */
    public boolean isDisabled() {
        return Status.DISABLED.getCode().equals(this.status);
    }

    /**
     * 判断账户是否被锁定
     * 
     * @return 账户是否被锁定
     */
    public boolean isLocked() {
        return Status.LOCKED.getCode().equals(this.status);
    }

    /**
     * 判断账户是否待激活
     * 
     * @return 账户是否待激活
     */
    public boolean isPending() {
        return Status.PENDING.getCode().equals(this.status);
    }

    /**
     * 获取显示名称（优先使用真实姓名，否则使用用户名）
     * 
     * @return 显示名称
     */
    public String getDisplayName() {
        return realName != null && !realName.trim().isEmpty() ? realName : username;
    }

    /**
     * 获取角色描述
     * 
     * @return 角色描述
     */
    public String getRoleDescription() {
        try {
            return Role.fromCode(this.role).getDescription();
        } catch (IllegalArgumentException e) {
            return this.role;
        }
    }

    /**
     * 获取状态描述
     * 
     * @return 状态描述
     */
    public String getStatusDescription() {
        try {
            return Status.fromCode(this.status).getDescription();
        } catch (IllegalArgumentException e) {
            return this.status;
        }
    }

    /**
     * 脱敏手机号
     * 
     * @return 脱敏后的手机号
     */
    public String getMaskedPhone() {
        if (phone == null || phone.length() < 11) {
            return phone;
        }
        return phone.substring(0, 3) + "****" + phone.substring(7);
    }

    /**
     * 脱敏邮箱
     * 
     * @return 脱敏后的邮箱
     */
    public String getMaskedEmail() {
        if (email == null || !email.contains("@")) {
            return email;
        }
        String[] parts = email.split("@");
        String localPart = parts[0];
        String domainPart = parts[1];
        
        if (localPart.length() <= 2) {
            return localPart + "***@" + domainPart;
        }
        
        return localPart.substring(0, 2) + "***@" + domainPart;
    }

    /**
     * 重置密码前的验证
     * 
     * @return 是否可以重置密码
     */
    public boolean canResetPassword() {
        return isActive() && !isLocked();
    }

    /**
     * 账户是否可用（用于登录验证）
     * 
     * @return 账户是否可用
     */
    public boolean isAccountAvailable() {
        return isActive();
    }
}