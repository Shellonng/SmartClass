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
import java.math.BigDecimal;
import java.time.LocalDateTime;

/**
 * 学生实体类
 * 
 * @author Education Platform Team
 * @version 1.0.0
 */
@Data
@EqualsAndHashCode(callSuper = false)
@Accessors(chain = true)
@TableName("student")
@Schema(description = "学生信息")
public class Student implements Serializable {

    private static final long serialVersionUID = 1L;

    @Schema(description = "学生ID")
    @TableId(value = "id", type = IdType.ASSIGN_ID)
    private Long id;

    @Schema(description = "用户ID")
    @TableField("user_id")
    @NotNull(message = "用户ID不能为空")
    private Long userId;

    @Schema(description = "学号", example = "2024001")
    @TableField("student_no")
    @NotBlank(message = "学号不能为空")
    @Size(max = 50, message = "学号长度不能超过50个字符")
    private String studentNo;

    @Schema(description = "班级ID")
    @TableField("class_id")
    private Long classId;

    @Schema(description = "专业", example = "计算机科学与技术")
    @TableField("major")
    @Size(max = 100, message = "专业名称长度不能超过100个字符")
    private String major;

    @Schema(description = "年级", example = "2024")
    @TableField("grade")
    private Integer grade;

    @Schema(description = "入学年份", example = "2024")
    @TableField("enrollment_year")
    private Integer enrollmentYear;

    @Schema(description = "毕业年份", example = "2028")
    @TableField("graduation_year")
    private Integer graduationYear;

    @Schema(description = "GPA", example = "3.85")
    @TableField("gpa")
    private BigDecimal gpa;

    @Schema(description = "学籍状态", example = "ENROLLED")
    @TableField("enrollment_status")
    private String enrollmentStatus;

    @Schema(description = "学生类型", example = "UNDERGRADUATE")
    @TableField("student_type")
    private String studentType;

    @Schema(description = "导师ID")
    @TableField("advisor_id")
    private Long advisorId;

    @Schema(description = "家庭地址")
    @TableField("home_address")
    @Size(max = 200, message = "家庭地址长度不能超过200个字符")
    private String homeAddress;

    @Schema(description = "紧急联系人")
    @TableField("emergency_contact")
    @Size(max = 50, message = "紧急联系人长度不能超过50个字符")
    private String emergencyContact;

    @Schema(description = "紧急联系电话")
    @TableField("emergency_phone")
    @Size(max = 20, message = "紧急联系电话长度不能超过20个字符")
    private String emergencyPhone;

    @Schema(description = "个人简介")
    @TableField("bio")
    @Size(max = 1000, message = "个人简介长度不能超过1000个字符")
    private String bio;

    @Schema(description = "兴趣爱好")
    @TableField("interests")
    @Size(max = 500, message = "兴趣爱好长度不能超过500个字符")
    private String interests;

    @Schema(description = "特长技能")
    @TableField("skills")
    @Size(max = 500, message = "特长技能长度不能超过500个字符")
    private String skills;

    @Schema(description = "学习目标")
    @TableField("learning_goals")
    @Size(max = 1000, message = "学习目标长度不能超过1000个字符")
    private String learningGoals;

    @Schema(description = "创建时间")
    @TableField(value = "create_time", fill = FieldFill.INSERT)
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime createTime;

    @Schema(description = "更新时间")
    @TableField(value = "update_time", fill = FieldFill.INSERT_UPDATE)
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

    // 关联信息（非数据库字段）
    @TableField(exist = false)
    @Schema(description = "关联的用户信息")
    private User user;

    @TableField(exist = false)
    @Schema(description = "关联的班级信息")
    private Class classInfo;

    @TableField(exist = false)
    @Schema(description = "关联的导师信息")
    private Teacher advisor;

    /**
     * 学籍状态枚举
     */
    public enum EnrollmentStatus {
        ENROLLED("ENROLLED", "在读"),
        SUSPENDED("SUSPENDED", "休学"),
        GRADUATED("GRADUATED", "毕业"),
        DROPPED_OUT("DROPPED_OUT", "退学"),
        TRANSFERRED("TRANSFERRED", "转学");

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
     * 学生类型枚举
     */
    public enum StudentType {
        UNDERGRADUATE("UNDERGRADUATE", "本科生"),
        GRADUATE("GRADUATE", "研究生"),
        DOCTORAL("DOCTORAL", "博士生"),
        EXCHANGE("EXCHANGE", "交换生"),
        INTERNATIONAL("INTERNATIONAL", "留学生");

        private final String code;
        private final String description;

        StudentType(String code, String description) {
            this.code = code;
            this.description = description;
        }

        public String getCode() {
            return code;
        }

        public String getDescription() {
            return description;
        }

        public static StudentType fromCode(String code) {
            for (StudentType type : values()) {
                if (type.code.equals(code)) {
                    return type;
                }
            }
            throw new IllegalArgumentException("未知的学生类型: " + code);
        }
    }

    /**
     * GPA等级枚举
     */
    public enum GpaLevel {
        EXCELLENT("优秀", new BigDecimal("3.7"), new BigDecimal("4.0")),
        GOOD("良好", new BigDecimal("3.0"), new BigDecimal("3.7")),
        AVERAGE("中等", new BigDecimal("2.0"), new BigDecimal("3.0")),
        POOR("较差", new BigDecimal("0.0"), new BigDecimal("2.0"));

        private final String description;
        private final BigDecimal minGpa;
        private final BigDecimal maxGpa;

        GpaLevel(String description, BigDecimal minGpa, BigDecimal maxGpa) {
            this.description = description;
            this.minGpa = minGpa;
            this.maxGpa = maxGpa;
        }

        public String getDescription() {
            return description;
        }

        public BigDecimal getMinGpa() {
            return minGpa;
        }

        public BigDecimal getMaxGpa() {
            return maxGpa;
        }

        public static GpaLevel fromGpa(BigDecimal gpa) {
            if (gpa == null) {
                return null;
            }
            for (GpaLevel level : values()) {
                if (gpa.compareTo(level.minGpa) >= 0 && gpa.compareTo(level.maxGpa) < 0) {
                    return level;
                }
            }
            return EXCELLENT; // 默认为优秀
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
     * 判断学生是否休学
     * 
     * @return 是否休学
     */
    public boolean isSuspended() {
        return EnrollmentStatus.SUSPENDED.getCode().equals(this.enrollmentStatus);
    }

    /**
     * 判断学生是否毕业
     * 
     * @return 是否毕业
     */
    public boolean isGraduated() {
        return EnrollmentStatus.GRADUATED.getCode().equals(this.enrollmentStatus);
    }

    /**
     * 判断学生是否退学
     * 
     * @return 是否退学
     */
    public boolean isDroppedOut() {
        return EnrollmentStatus.DROPPED_OUT.getCode().equals(this.enrollmentStatus);
    }

    /**
     * 判断学生是否转学
     * 
     * @return 是否转学
     */
    public boolean isTransferred() {
        return EnrollmentStatus.TRANSFERRED.getCode().equals(this.enrollmentStatus);
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

    /**
     * 获取学生类型描述
     * 
     * @return 学生类型描述
     */
    public String getStudentTypeDescription() {
        try {
            return StudentType.fromCode(this.studentType).getDescription();
        } catch (IllegalArgumentException e) {
            return this.studentType;
        }
    }

    /**
     * 获取GPA等级
     * 
     * @return GPA等级
     */
    public GpaLevel getGpaLevel() {
        return GpaLevel.fromGpa(this.gpa);
    }

    /**
     * 获取GPA等级描述
     * 
     * @return GPA等级描述
     */
    public String getGpaLevelDescription() {
        GpaLevel level = getGpaLevel();
        return level != null ? level.getDescription() : "未知";
    }

    /**
     * 获取显示名称（优先使用用户真实姓名，否则使用学号）
     * 
     * @return 显示名称
     */
    public String getDisplayName() {
        if (user != null && user.getRealName() != null && !user.getRealName().trim().isEmpty()) {
            return user.getRealName();
        }
        return studentNo;
    }

    /**
     * 获取完整学生信息（姓名 + 学号）
     * 
     * @return 完整学生信息
     */
    public String getFullInfo() {
        String name = getDisplayName();
        if (name.equals(studentNo)) {
            return studentNo;
        }
        return name + "(" + studentNo + ")";
    }

    /**
     * 获取年级描述
     * 
     * @return 年级描述
     */
    public String getGradeDescription() {
        if (grade == null) {
            return "未知年级";
        }
        return grade + "级";
    }

    /**
     * 获取学习阶段（根据年级计算）
     * 
     * @return 学习阶段
     */
    public String getStudyPhase() {
        if (enrollmentYear == null) {
            return "未知";
        }
        
        int currentYear = LocalDateTime.now().getYear();
        int studyYear = currentYear - enrollmentYear + 1;
        
        if (StudentType.UNDERGRADUATE.getCode().equals(studentType)) {
            switch (studyYear) {
                case 1: return "大一";
                case 2: return "大二";
                case 3: return "大三";
                case 4: return "大四";
                default: return studyYear > 4 ? "延期" : "未入学";
            }
        } else if (StudentType.GRADUATE.getCode().equals(studentType)) {
            switch (studyYear) {
                case 1: return "研一";
                case 2: return "研二";
                case 3: return "研三";
                default: return studyYear > 3 ? "延期" : "未入学";
            }
        } else if (StudentType.DOCTORAL.getCode().equals(studentType)) {
            switch (studyYear) {
                case 1: return "博一";
                case 2: return "博二";
                case 3: return "博三";
                case 4: return "博四";
                default: return studyYear > 4 ? "延期" : "未入学";
            }
        }
        
        return "第" + studyYear + "年";
    }

    /**
     * 获取兴趣爱好列表
     * 
     * @return 兴趣爱好列表
     */
    public String[] getInterestsList() {
        if (interests == null || interests.trim().isEmpty()) {
            return new String[0];
        }
        return interests.split(",");
    }

    /**
     * 设置兴趣爱好列表
     * 
     * @param interestsList 兴趣爱好列表
     */
    public void setInterestsList(String[] interestsList) {
        if (interestsList == null || interestsList.length == 0) {
            this.interests = null;
        } else {
            this.interests = String.join(",", interestsList);
        }
    }

    /**
     * 获取技能列表
     * 
     * @return 技能列表
     */
    public String[] getSkillsList() {
        if (skills == null || skills.trim().isEmpty()) {
            return new String[0];
        }
        return skills.split(",");
    }

    /**
     * 设置技能列表
     * 
     * @param skillsList 技能列表
     */
    public void setSkillsList(String[] skillsList) {
        if (skillsList == null || skillsList.length == 0) {
            this.skills = null;
        } else {
            this.skills = String.join(",", skillsList);
        }
    }

    /**
     * 判断是否可以选课
     * 
     * @return 是否可以选课
     */
    public boolean canEnrollCourse() {
        return isEnrolled() && !isDeleted;
    }

    /**
     * 判断是否可以提交作业
     * 
     * @return 是否可以提交作业
     */
    public boolean canSubmitAssignment() {
        return isEnrolled() && !isDeleted;
    }

    /**
     * 判断是否为优秀学生（GPA >= 3.7）
     * 
     * @return 是否为优秀学生
     */
    public boolean isExcellentStudent() {
        return gpa != null && gpa.compareTo(new BigDecimal("3.7")) >= 0;
    }

    /**
     * 判断是否需要学业预警（GPA < 2.0）
     * 
     * @return 是否需要学业预警
     */
    public boolean needsAcademicWarning() {
        return gpa != null && gpa.compareTo(new BigDecimal("2.0")) < 0;
    }

    /**
     * 计算预计毕业时间
     * 
     * @return 预计毕业时间
     */
    public LocalDateTime getExpectedGraduationTime() {
        if (graduationYear == null) {
            return null;
        }
        return LocalDateTime.of(graduationYear, 6, 30, 0, 0); // 假设6月30日毕业
    }

    /**
     * 判断是否即将毕业（距离毕业时间不足一年）
     * 
     * @return 是否即将毕业
     */
    public boolean isAboutToGraduate() {
        LocalDateTime expectedGraduation = getExpectedGraduationTime();
        if (expectedGraduation == null) {
            return false;
        }
        return LocalDateTime.now().plusYears(1).isAfter(expectedGraduation);
    }
}