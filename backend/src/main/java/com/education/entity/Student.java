package com.education.entity;

import com.baomidou.mybatisplus.annotation.*;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.experimental.Accessors;
import com.fasterxml.jackson.annotation.JsonFormat;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;
import java.io.Serializable;
import java.math.BigDecimal;
import java.time.LocalDateTime;

/**
 * 学生信息实体类
 * 
 * @author SmartClass
 * @since 2024-01-01
 */
@Data
@EqualsAndHashCode(callSuper = false)
@Accessors(chain = true)
@TableName("student")
@Schema(description = "学生信息")
public class Student implements Serializable {

    private static final long serialVersionUID = 1L;

    @Schema(description = "学生ID")
    @TableId(value = "id", type = IdType.AUTO)
    private Long id;

    @Schema(description = "用户ID")
    @TableField("user_id")
    @NotNull(message = "用户ID不能为空")
    private Long userId;

    @Schema(description = "学号", example = "2024001")
    @TableField("student_id")
    @NotBlank(message = "学号不能为空")
    @Size(max = 50, message = "学号长度不能超过50个字符")
    private String studentId;

    @Schema(description = "学籍状态", example = "ENROLLED")
    @TableField("enrollment_status")
    private String enrollmentStatus;

    @Schema(description = "GPA", example = "3.85")
    @TableField("gpa")
    private BigDecimal gpa;

    @Schema(description = "GPA等级", example = "A")
    @TableField("gpa_level")
    private String gpaLevel;

    @Schema(description = "创建时间")
    @TableField(value = "create_time", fill = FieldFill.INSERT)
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime createTime;

    @Schema(description = "更新时间")
    @TableField(value = "update_time", fill = FieldFill.INSERT_UPDATE)
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime updateTime;

    // 关联用户信息（非数据库字段）
    @TableField(exist = false)
    @Schema(description = "关联的用户信息")
    private User user;

    /**
     * 学籍状态枚举
     */
    public enum EnrollmentStatus {
        ENROLLED("ENROLLED", "在读"),
        SUSPENDED("SUSPENDED", "休学"),
        GRADUATED("GRADUATED", "毕业"),
        DROPPED_OUT("DROPPED_OUT", "退学");

        private final String code;
        private final String description;

        EnrollmentStatus(String code, String description) {
            this.code = code;
            this.description = description;
        }

        public String getCode() {
            return code;
        }

        public String getDescription() {
            return description;
        }

        public static EnrollmentStatus fromCode(String code) {
            for (EnrollmentStatus status : values()) {
                if (status.code.equals(code)) {
                    return status;
                }
            }
            throw new IllegalArgumentException("未知的学籍状态: " + code);
        }
    }

    /**
     * 判断学生是否在读
     * 
     * @return 是否在读
     */
    public boolean isEnrolled() {
        return EnrollmentStatus.ENROLLED.getCode().equals(this.enrollmentStatus);
    }

    /**
     * 获取显示名称
     * 
     * @return 显示名称
     */
    public String getDisplayName() {
        return user != null ? user.getDisplayName() : studentId;
    }

    /**
     * 获取学籍状态描述
     * 
     * @return 学籍状态描述
     */
    public String getEnrollmentStatusDescription() {
        try {
            return EnrollmentStatus.fromCode(this.enrollmentStatus).getDescription();
        } catch (IllegalArgumentException e) {
            return this.enrollmentStatus;
        }
    }
}