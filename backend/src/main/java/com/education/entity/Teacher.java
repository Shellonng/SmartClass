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
import java.time.LocalDateTime;

/**
 * 教师信息实体类
 * 
 * @author SmartClass
 * @since 2024-01-01
 */
@Data
@EqualsAndHashCode(callSuper = false)
@Accessors(chain = true)
@TableName("teacher")
@Schema(description = "教师信息")
public class Teacher implements Serializable {

    private static final long serialVersionUID = 1L;

    @Schema(description = "教师ID")
    @TableId(value = "id", type = IdType.AUTO)
    private Long id;

    @Schema(description = "用户ID")
    @TableField("user_id")
    @NotNull(message = "用户ID不能为空")
    private Long userId;

    @Schema(description = "院系", example = "计算机科学与技术学院")
    @TableField("department")
    @Size(max = 100, message = "院系名称长度不能超过100个字符")
    private String department;

    @Schema(description = "职称", example = "副教授")
    @TableField("title")
    @Size(max = 50, message = "职称长度不能超过50个字符")
    private String title;

    @Schema(description = "教育背景")
    @TableField("education")
    private String education;

    @Schema(description = "专业")
    @TableField("specialty")
    private String specialty;

    @Schema(description = "简介")
    @TableField("introduction")
    private String introduction;

    @Schema(description = "办公地点")
    @TableField("office_location")
    private String officeLocation;

    @Schema(description = "办公时间")
    @TableField("office_hours")
    private String officeHours;

    @Schema(description = "联系邮箱")
    @TableField("contact_email")
    private String contactEmail;

    @Schema(description = "联系电话")
    @TableField("contact_phone")
    private String contactPhone;

    @Schema(description = "状态")
    @TableField("status")
    private String status;

    @Schema(description = "入职日期")
    @TableField("hire_date")
    private LocalDateTime hireDate;

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
     * 获取显示名称
     * 
     * @return 显示名称
     */
    public String getDisplayName() {
        return user != null ? user.getDisplayName() : String.valueOf(id);
    }

    /**
     * 获取完整职称
     * 
     * @return 完整职称
     */
    public String getFullTitle() {
        if (title != null && department != null) {
            return department + " " + title;
        } else if (title != null) {
            return title;
        } else if (department != null) {
            return department;
        } else {
            return "教师";
        }
    }
}