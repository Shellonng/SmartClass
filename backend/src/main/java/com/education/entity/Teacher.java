package com.education.entity;

import com.baomidou.mybatisplus.annotation.*;
import com.fasterxml.jackson.annotation.JsonFormat;
import com.fasterxml.jackson.annotation.JsonIgnore;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.experimental.Accessors;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;
import java.io.Serializable;
import java.time.LocalDateTime;

/**
 * 教师实体类
 * 
 * @author Education Platform Team
 * @version 1.0.0
 */
@Data
@EqualsAndHashCode(callSuper = false)
@Accessors(chain = true)
@TableName("teacher")
@Schema(description = "教师信息")
public class Teacher implements Serializable {

    private static final long serialVersionUID = 1L;

    @Schema(description = "教师ID")
    @TableId(value = "id", type = IdType.ASSIGN_ID)
    private Long id;

    @Schema(description = "用户ID")
    @TableField("user_id")
    @NotNull(message = "用户ID不能为空")
    private Long userId;

    @Schema(description = "教师工号", example = "T001")
    @TableField("teacher_no")
    @NotBlank(message = "教师工号不能为空")
    @Size(max = 50, message = "教师工号长度不能超过50个字符")
    private String teacherNo;

    @Schema(description = "院系", example = "计算机科学与技术学院")
    @TableField("department")
    @Size(max = 100, message = "院系名称长度不能超过100个字符")
    private String department;

    @Schema(description = "职称", example = "副教授")
    @TableField("title")
    @Size(max = 50, message = "职称长度不能超过50个字符")
    private String title;

    @Schema(description = "学历", example = "博士")
    @TableField("education")
    @Size(max = 50, message = "学历长度不能超过50个字符")
    private String education;

    @Schema(description = "专业领域", example = "人工智能,机器学习")
    @TableField("specialization")
    @Size(max = 200, message = "专业领域长度不能超过200个字符")
    private String specialization;

    @Schema(description = "个人简介")
    @TableField("bio")
    @Size(max = 1000, message = "个人简介长度不能超过1000个字符")
    private String bio;

    @Schema(description = "办公地点", example = "教学楼A座301")
    @TableField("office_location")
    @Size(max = 100, message = "办公地点长度不能超过100个字符")
    private String officeLocation;

    @Schema(description = "办公时间", example = "周一至周五 9:00-17:00")
    @TableField("office_hours")
    @Size(max = 200, message = "办公时间长度不能超过200个字符")
    private String officeHours;

    @Schema(description = "研究方向")
    @TableField("research_interests")
    @Size(max = 500, message = "研究方向长度不能超过500个字符")
    private String researchInterests;

    @Schema(description = "学术成果")
    @TableField("academic_achievements")
    @Size(max = 2000, message = "学术成果长度不能超过2000个字符")
    private String academicAchievements;

    @Schema(description = "联系邮箱", example = "teacher@example.com")
    @TableField("contact_email")
    @Size(max = 100, message = "联系邮箱长度不能超过100个字符")
    private String contactEmail;

    @Schema(description = "联系电话", example = "13800138000")
    @TableField("contact_phone")
    @Size(max = 20, message = "联系电话长度不能超过20个字符")
    private String contactPhone;

    @Schema(description = "教师状态", example = "ACTIVE")
    @TableField("status")
    private String status;

    @Schema(description = "入职时间")
    @TableField("hire_date")
    @JsonFormat(pattern = "yyyy-MM-dd")
    private LocalDateTime hireDate;

    @Schema(description = "创建时间")
    @TableField(value = "created_time", fill = FieldFill.INSERT)
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime createTime;

    @Schema(description = "更新时间")
    @TableField(value = "updated_time", fill = FieldFill.INSERT_UPDATE)
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime updateTime;

    @Schema(description = "是否删除")
    @TableField("is_deleted")
    @TableLogic
    @JsonIgnore
    private Boolean isDeleted;

    @Schema(description = "扩展字段1")
    @TableField("ext_field1")
    private String extField1;

    @Schema(description = "扩展字段2")
    @TableField("ext_field2")
    private String extField2;

    @Schema(description = "扩展字段3")
    @TableField("ext_field3")
    private String extField3;

    // 关联用户信息（非数据库字段）
    @TableField(exist = false)
    @Schema(description = "关联的用户信息")
    private User user;

    /**
     * 教师状态枚举
     */
    public enum Status {
        ACTIVE("ACTIVE", "在职"),
        INACTIVE("INACTIVE", "离职"),
        SUSPENDED("SUSPENDED", "停职"),
        RETIRED("RETIRED", "退休");

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
            throw new IllegalArgumentException("未知的教师状态: " + code);
        }
    }

    /**
     * 职称枚举
     */
    public enum Title {
        ASSISTANT("助教", 1),
        LECTURER("讲师", 2),
        ASSOCIATE_PROFESSOR("副教授", 3),
        PROFESSOR("教授", 4);

        private final String name;
        private final int level;

        Title(String name, int level) {
            this.name = name;
            this.level = level;
        }

        public String getName() {
            return name;
        }

        public int getLevel() {
            return level;
        }

        public static Title fromName(String name) {
            for (Title title : values()) {
                if (title.name.equals(name)) {
                    return title;
                }
            }
            return null;
        }
    }

    /**
     * 学历枚举
     */
    public enum Education {
        BACHELOR("学士", 1),
        MASTER("硕士", 2),
        DOCTOR("博士", 3),
        POSTDOC("博士后", 4);

        private final String name;
        private final int level;

        Education(String name, int level) {
            this.name = name;
            this.level = level;
        }

        public String getName() {
            return name;
        }

        public int getLevel() {
            return level;
        }

        public static Education fromName(String name) {
            for (Education education : values()) {
                if (education.name.equals(name)) {
                    return education;
                }
            }
            return null;
        }
    }

    /**
     * 判断教师是否在职
     * 
     * @return 是否在职
     */
    public boolean isActive() {
        return Status.ACTIVE.getCode().equals(this.status);
    }

    /**
     * 判断教师是否离职
     * 
     * @return 是否离职
     */
    public boolean isInactive() {
        return Status.INACTIVE.getCode().equals(this.status);
    }

    /**
     * 判断教师是否停职
     * 
     * @return 是否停职
     */
    public boolean isSuspended() {
        return Status.SUSPENDED.getCode().equals(this.status);
    }

    /**
     * 判断教师是否退休
     * 
     * @return 是否退休
     */
    public boolean isRetired() {
        return Status.RETIRED.getCode().equals(this.status);
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
     * 获取职称等级
     * 
     * @return 职称等级
     */
    public int getTitleLevel() {
        Title titleEnum = Title.fromName(this.title);
        return titleEnum != null ? titleEnum.getLevel() : 0;
    }

    /**
     * 获取学历等级
     * 
     * @return 学历等级
     */
    public int getEducationLevel() {
        Education educationEnum = Education.fromName(this.education);
        return educationEnum != null ? educationEnum.getLevel() : 0;
    }

    /**
     * 获取专业领域列表
     * 
     * @return 专业领域列表
     */
    public String[] getSpecializationList() {
        if (specialization == null || specialization.trim().isEmpty()) {
            return new String[0];
        }
        return specialization.split(",");
    }

    /**
     * 设置专业领域列表
     * 
     * @param specializationList 专业领域列表
     */
    public void setSpecializationList(String[] specializationList) {
        if (specializationList == null || specializationList.length == 0) {
            this.specialization = null;
        } else {
            this.specialization = String.join(",", specializationList);
        }
    }

    /**
     * 获取显示名称（优先使用用户真实姓名，否则使用教师工号）
     * 
     * @return 显示名称
     */
    public String getDisplayName() {
        if (user != null && user.getRealName() != null && !user.getRealName().trim().isEmpty()) {
            return user.getRealName();
        }
        return teacherNo;
    }

    /**
     * 获取完整职称（包含学历）
     * 
     * @return 完整职称
     */
    public String getFullTitle() {
        StringBuilder sb = new StringBuilder();
        if (education != null && !education.trim().isEmpty()) {
            sb.append(education).append(" ");
        }
        if (title != null && !title.trim().isEmpty()) {
            sb.append(title);
        }
        return sb.toString().trim();
    }

    /**
     * 获取联系方式摘要
     * 
     * @return 联系方式摘要
     */
    public String getContactSummary() {
        StringBuilder sb = new StringBuilder();
        if (contactEmail != null && !contactEmail.trim().isEmpty()) {
            sb.append("邮箱: ").append(contactEmail);
        }
        if (contactPhone != null && !contactPhone.trim().isEmpty()) {
            if (sb.length() > 0) {
                sb.append("; ");
            }
            sb.append("电话: ").append(contactPhone);
        }
        if (officeLocation != null && !officeLocation.trim().isEmpty()) {
            if (sb.length() > 0) {
                sb.append("; ");
            }
            sb.append("办公室: ").append(officeLocation);
        }
        return sb.toString();
    }

    /**
     * 判断是否可以授课
     * 
     * @return 是否可以授课
     */
    public boolean canTeach() {
        return isActive() && !isDeleted;
    }

    /**
     * 获取工作年限（从入职时间计算）
     * 
     * @return 工作年限
     */
    public int getWorkingYears() {
        if (hireDate == null) {
            return 0;
        }
        return LocalDateTime.now().getYear() - hireDate.getYear();
    }

    /**
     * 判断是否为高级职称（副教授及以上）
     * 
     * @return 是否为高级职称
     */
    public boolean isSeniorTitle() {
        return getTitleLevel() >= Title.ASSOCIATE_PROFESSOR.getLevel();
    }

    /**
     * 判断是否具有博士学历
     * 
     * @return 是否具有博士学历
     */
    public boolean hasDoctorateDegree() {
        return getEducationLevel() >= Education.DOCTOR.getLevel();
    }
}