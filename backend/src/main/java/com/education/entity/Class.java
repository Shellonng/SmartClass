package com.education.entity;

import com.baomidou.mybatisplus.annotation.*;
import com.fasterxml.jackson.annotation.JsonFormat;
import com.fasterxml.jackson.annotation.JsonIgnore;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.experimental.Accessors;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;
import java.io.Serializable;
import java.time.LocalDateTime;
import java.util.List;

/**
 * 班级实体类
 * 
 * @author Education Platform Team
 * @version 1.0.0
 */
@Data
@EqualsAndHashCode(callSuper = false)
@Accessors(chain = true)
@TableName("class")
@Schema(description = "班级信息")
public class Class implements Serializable {

    private static final long serialVersionUID = 1L;

    @Schema(description = "班级ID")
    @TableId(value = "id", type = IdType.ASSIGN_ID)
    private Long id;

    @Schema(description = "班级名称", example = "计算机科学与技术2024-1班")
    @TableField("class_name")
    @NotBlank(message = "班级名称不能为空")
    @Size(max = 100, message = "班级名称长度不能超过100个字符")
    private String className;

    @Schema(description = "班级代码", example = "CS2024-1")
    @TableField("class_code")
    @NotBlank(message = "班级代码不能为空")
    @Size(max = 50, message = "班级代码长度不能超过50个字符")
    private String classCode;

    @Schema(description = "班主任ID")
    @TableField("head_teacher_id")
    private Long headTeacherId;

    @Schema(description = "专业", example = "计算机科学与技术")
    @TableField("major")
    @Size(max = 100, message = "专业名称长度不能超过100个字符")
    private String major;

    @Schema(description = "年级", example = "2024")
    @TableField("grade")
    private Integer grade;

    @Schema(description = "学期", example = "2024-1")
    @TableField("semester")
    @Size(max = 20, message = "学期长度不能超过20个字符")
    private String semester;

    @Schema(description = "学生人数", example = "30")
    @TableField("student_count")
    private Integer studentCount;

    @Schema(description = "最大学生人数", example = "35")
    @TableField("max_student_count")
    private Integer maxStudentCount;

    @Schema(description = "班级描述")
    @TableField("description")
    @Size(max = 500, message = "班级描述长度不能超过500个字符")
    private String description;

    @Schema(description = "班级状态", example = "ACTIVE")
    @TableField("status")
    private String status;

    @Schema(description = "开班时间")
    @TableField("start_date")
    @JsonFormat(pattern = "yyyy-MM-dd")
    private LocalDateTime startDate;

    @Schema(description = "结班时间")
    @TableField("end_date")
    @JsonFormat(pattern = "yyyy-MM-dd")
    private LocalDateTime endDate;

    @Schema(description = "教室位置", example = "教学楼A座201")
    @TableField("classroom")
    @Size(max = 100, message = "教室位置长度不能超过100个字符")
    private String classroom;

    @Schema(description = "班级口号")
    @TableField("motto")
    @Size(max = 200, message = "班级口号长度不能超过200个字符")
    private String motto;

    @Schema(description = "班级规章制度")
    @TableField("rules")
    @Size(max = 2000, message = "班级规章制度长度不能超过2000个字符")
    private String rules;

    @Schema(description = "班级活动记录")
    @TableField("activities")
    @Size(max = 2000, message = "班级活动记录长度不能超过2000个字符")
    private String activities;

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
    @Schema(description = "班主任信息")
    private Teacher headTeacher;

    @TableField(exist = false)
    @Schema(description = "学生列表")
    private List<Student> students;

    @TableField(exist = false)
    @Schema(description = "课程列表")
    private List<Course> courses;

    /**
     * 班级状态枚举
     */
    public enum Status {
        ACTIVE("ACTIVE", "正常"),
        INACTIVE("INACTIVE", "停用"),
        GRADUATED("GRADUATED", "已毕业"),
        SUSPENDED("SUSPENDED", "暂停");

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
            throw new IllegalArgumentException("未知的班级状态: " + code);
        }
    }

    /**
     * 学期类型枚举
     */
    public enum SemesterType {
        SPRING("春季学期", 1),
        SUMMER("夏季学期", 2),
        AUTUMN("秋季学期", 3),
        WINTER("冬季学期", 4);

        private final String description;
        private final int order;

        SemesterType(String description, int order) {
            this.description = description;
            this.order = order;
        }

        public String getDescription() {
            return description;
        }

        public int getOrder() {
            return order;
        }

        public static SemesterType fromSemester(String semester) {
            if (semester == null || semester.trim().isEmpty()) {
                return null;
            }
            
            String[] parts = semester.split("-");
            if (parts.length != 2) {
                return null;
            }
            
            try {
                int semesterNum = Integer.parseInt(parts[1]);
                switch (semesterNum) {
                    case 1: return SPRING;
                    case 2: return SUMMER;
                    case 3: return AUTUMN;
                    case 4: return WINTER;
                    default: return null;
                }
            } catch (NumberFormatException e) {
                return null;
            }
        }
    }

    /**
     * 判断班级是否正常
     * 
     * @return 是否正常
     */
    public boolean isActive() {
        return Status.ACTIVE.getCode().equals(this.status);
    }

    /**
     * 判断班级是否停用
     * 
     * @return 是否停用
     */
    public boolean isInactive() {
        return Status.INACTIVE.getCode().equals(this.status);
    }

    /**
     * 判断班级是否已毕业
     * 
     * @return 是否已毕业
     */
    public boolean isGraduated() {
        return Status.GRADUATED.getCode().equals(this.status);
    }

    /**
     * 判断班级是否暂停
     * 
     * @return 是否暂停
     */
    public boolean isSuspended() {
        return Status.SUSPENDED.getCode().equals(this.status);
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
     * 获取学期类型
     * 
     * @return 学期类型
     */
    public SemesterType getSemesterType() {
        return SemesterType.fromSemester(this.semester);
    }

    /**
     * 获取学期描述
     * 
     * @return 学期描述
     */
    public String getSemesterDescription() {
        SemesterType type = getSemesterType();
        if (type == null) {
            return semester;
        }
        
        String[] parts = semester.split("-");
        return parts[0] + "年" + type.getDescription();
    }

    /**
     * 获取完整班级名称（包含年级和专业）
     * 
     * @return 完整班级名称
     */
    public String getFullClassName() {
        StringBuilder sb = new StringBuilder();
        if (grade != null) {
            sb.append(grade).append("级");
        }
        if (major != null && !major.trim().isEmpty()) {
            if (sb.length() > 0) {
                sb.append(" ");
            }
            sb.append(major);
        }
        if (className != null && !className.trim().isEmpty()) {
            if (sb.length() > 0) {
                sb.append(" ");
            }
            sb.append(className);
        }
        return sb.toString();
    }

    /**
     * 获取班级简称
     * 
     * @return 班级简称
     */
    public String getShortName() {
        if (classCode != null && !classCode.trim().isEmpty()) {
            return classCode;
        }
        return className;
    }

    /**
     * 判断是否可以添加学生
     * 
     * @return 是否可以添加学生
     */
    public boolean canAddStudent() {
        if (!isActive() || isDeleted) {
            return false;
        }
        if (maxStudentCount == null) {
            return true;
        }
        int currentCount = studentCount != null ? studentCount : 0;
        return currentCount < maxStudentCount;
    }

    /**
     * 判断班级是否已满
     * 
     * @return 是否已满
     */
    public boolean isFull() {
        if (maxStudentCount == null) {
            return false;
        }
        int currentCount = studentCount != null ? studentCount : 0;
        return currentCount >= maxStudentCount;
    }

    /**
     * 获取剩余可添加学生数
     * 
     * @return 剩余可添加学生数
     */
    public int getRemainingCapacity() {
        if (maxStudentCount == null) {
            return Integer.MAX_VALUE;
        }
        int currentCount = studentCount != null ? studentCount : 0;
        return Math.max(0, maxStudentCount - currentCount);
    }

    /**
     * 获取班级容量使用率
     * 
     * @return 容量使用率（百分比）
     */
    public double getCapacityUsageRate() {
        if (maxStudentCount == null || maxStudentCount == 0) {
            return 0.0;
        }
        int currentCount = studentCount != null ? studentCount : 0;
        return (double) currentCount / maxStudentCount * 100;
    }

    /**
     * 判断班级是否在指定时间范围内
     * 
     * @param checkDate 检查日期
     * @return 是否在时间范围内
     */
    public boolean isActiveAt(LocalDateTime checkDate) {
        if (checkDate == null) {
            return false;
        }
        
        boolean afterStart = startDate == null || !checkDate.isBefore(startDate);
        boolean beforeEnd = endDate == null || !checkDate.isAfter(endDate);
        
        return afterStart && beforeEnd;
    }

    /**
     * 判断班级是否即将结束（距离结束时间不足一个月）
     * 
     * @return 是否即将结束
     */
    public boolean isAboutToEnd() {
        if (endDate == null) {
            return false;
        }
        return LocalDateTime.now().plusMonths(1).isAfter(endDate);
    }

    /**
     * 获取班级运行天数
     * 
     * @return 运行天数
     */
    public long getRunningDays() {
        if (startDate == null) {
            return 0;
        }
        LocalDateTime now = LocalDateTime.now();
        LocalDateTime start = startDate;
        if (endDate != null && now.isAfter(endDate)) {
            now = endDate;
        }
        return java.time.Duration.between(start, now).toDays();
    }

    /**
     * 获取班级总学制天数
     * 
     * @return 总学制天数
     */
    public long getTotalDays() {
        if (startDate == null || endDate == null) {
            return 0;
        }
        return java.time.Duration.between(startDate, endDate).toDays();
    }

    /**
     * 获取班级进度百分比
     * 
     * @return 进度百分比
     */
    public double getProgressPercentage() {
        long totalDays = getTotalDays();
        if (totalDays <= 0) {
            return 0.0;
        }
        long runningDays = getRunningDays();
        return Math.min(100.0, (double) runningDays / totalDays * 100);
    }

    /**
     * 更新学生人数
     * 
     * @param count 新的学生人数
     */
    public void updateStudentCount(int count) {
        this.studentCount = Math.max(0, count);
    }

    /**
     * 增加学生人数
     * 
     * @param increment 增加数量
     */
    public void incrementStudentCount(int increment) {
        int currentCount = studentCount != null ? studentCount : 0;
        this.studentCount = Math.max(0, currentCount + increment);
    }

    /**
     * 减少学生人数
     * 
     * @param decrement 减少数量
     */
    public void decrementStudentCount(int decrement) {
        incrementStudentCount(-decrement);
    }
}