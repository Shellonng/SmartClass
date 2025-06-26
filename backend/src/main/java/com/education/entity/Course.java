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
import java.util.List;

/**
 * 课程实体类
 * 
 * @author Education Platform Team
 * @version 1.0.0
 */
@Data
@EqualsAndHashCode(callSuper = false)
@Accessors(chain = true)
@TableName("course")
@Schema(description = "课程信息")
public class Course implements Serializable {

    private static final long serialVersionUID = 1L;

    @Schema(description = "课程ID")
    @TableId(value = "id", type = IdType.ASSIGN_ID)
    private Long id;
    
    // 兼容方法
    public void setCourseId(Long courseId) {
        this.id = courseId;
    }
    
    public Long getCourseId() {
        return this.id;
    }

    @Schema(description = "课程名称", example = "Java程序设计")
    @TableField("course_name")
    @NotBlank(message = "课程名称不能为空")
    @Size(max = 100, message = "课程名称长度不能超过100个字符")
    private String courseName;

    @Schema(description = "课程代码", example = "CS101")
    @TableField("course_code")
    @NotBlank(message = "课程代码不能为空")
    @Size(max = 50, message = "课程代码长度不能超过50个字符")
    private String courseCode;

    @Schema(description = "授课教师ID")
    @TableField("teacher_id")
    @NotNull(message = "授课教师ID不能为空")
    private Long teacherId;

    @Schema(description = "班级ID")
    @TableField("class_id")
    private Long classId;

    @Schema(description = "学分", example = "3.0")
    @TableField("credits")
    private BigDecimal credits;

    @Schema(description = "课程类型", example = "REQUIRED")
    @TableField("course_type")
    private String courseType;

    @Schema(description = "课程分类", example = "COMPUTER_SCIENCE")
    @TableField("category")
    @Size(max = 100, message = "课程分类长度不能超过100个字符")
    private String category;

    @Schema(description = "学期", example = "2024-1")
    @TableField("semester")
    @Size(max = 20, message = "学期长度不能超过20个字符")
    private String semester;

    @Schema(description = "开始日期")
    @TableField("start_date")
    @JsonFormat(pattern = "yyyy-MM-dd")
    private LocalDateTime startDate;

    @Schema(description = "结束日期")
    @TableField("end_date")
    @JsonFormat(pattern = "yyyy-MM-dd")
    private LocalDateTime endDate;

    @Schema(description = "课程安排")
    @TableField("schedule")
    @Size(max = 500, message = "课程安排长度不能超过500个字符")
    private String schedule;

    @Schema(description = "上课地点", example = "教学楼A座201")
    @TableField("location")
    @Size(max = 100, message = "上课地点长度不能超过100个字符")
    private String location;

    @Schema(description = "课程描述")
    @TableField("description")
    @Size(max = 2000, message = "课程描述长度不能超过2000个字符")
    private String description;

    @Schema(description = "课程目标")
    @TableField("objectives")
    @Size(max = 2000, message = "课程目标长度不能超过2000个字符")
    private String objectives;

    @Schema(description = "课程要求")
    @TableField("requirements")
    @Size(max = 2000, message = "课程要求长度不能超过2000个字符")
    private String requirements;

    @Schema(description = "教学大纲")
    @TableField("syllabus")
    @Size(max = 5000, message = "教学大纲长度不能超过5000个字符")
    private String syllabus;

    @Schema(description = "参考教材")
    @TableField("textbooks")
    @Size(max = 1000, message = "参考教材长度不能超过1000个字符")
    private String textbooks;

    @Schema(description = "评分标准")
    @TableField("grading_criteria")
    @Size(max = 2000, message = "评分标准长度不能超过2000个字符")
    private String gradingCriteria;

    @Schema(description = "最大选课人数", example = "50")
    @TableField("max_enrollment")
    private Integer maxEnrollment;

    @Schema(description = "当前选课人数", example = "30")
    @TableField("current_enrollment")
    private Integer currentEnrollment;

    @Schema(description = "课程状态", example = "ACTIVE")
    @TableField("status")
    private String status;

    @Schema(description = "是否公开", example = "true")
    @TableField("is_public")
    private Boolean isPublic;

    @Schema(description = "课程难度", example = "INTERMEDIATE")
    @TableField("difficulty_level")
    private String difficultyLevel;

    @Schema(description = "先修课程")
    @TableField("prerequisites")
    @Size(max = 500, message = "先修课程长度不能超过500个字符")
    private String prerequisites;

    @Schema(description = "课程标签")
    @TableField("tags")
    @Size(max = 500, message = "课程标签长度不能超过500个字符")
    private String tags;

    @Schema(description = "课程封面图片")
    @TableField("cover_image")
    @Size(max = 500, message = "课程封面图片URL长度不能超过500个字符")
    private String coverImage;

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

    // 关联信息（非数据库字段）
    @TableField(exist = false)
    @Schema(description = "授课教师信息")
    private Teacher teacher;

    @TableField(exist = false)
    @Schema(description = "班级信息")
    private Class classInfo;

    @TableField(exist = false)
    @Schema(description = "选课学生列表")
    private List<Student> students;

    @TableField(exist = false)
    @Schema(description = "课程任务列表")
    private List<Task> tasks;

    @TableField(exist = false)
    @Schema(description = "课程资源列表")
    private List<Resource> resources;

    /**
     * 课程类型枚举
     */
    public enum CourseType {
        REQUIRED("REQUIRED", "必修课"),
        ELECTIVE("ELECTIVE", "选修课"),
        CORE("CORE", "核心课"),
        GENERAL("GENERAL", "通识课"),
        PRACTICAL("PRACTICAL", "实践课");

        private final String code;
        private final String description;

        CourseType(String code, String description) {
            this.code = code;
            this.description = description;
        }

        public String getCode() {
            return code;
        }

        public String getDescription() {
            return description;
        }

        public static CourseType fromCode(String code) {
            for (CourseType type : values()) {
                if (type.code.equals(code)) {
                    return type;
                }
            }
            throw new IllegalArgumentException("未知的课程类型: " + code);
        }
    }

    /**
     * 课程状态枚举
     */
    public enum Status {
        DRAFT("DRAFT", "草稿"),
        ACTIVE("ACTIVE", "进行中"),
        COMPLETED("COMPLETED", "已完成"),
        SUSPENDED("SUSPENDED", "暂停"),
        CANCELLED("CANCELLED", "已取消");

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
            throw new IllegalArgumentException("未知的课程状态: " + code);
        }
    }

    /**
     * 课程难度枚举
     */
    public enum DifficultyLevel {
        BEGINNER("BEGINNER", "初级", 1),
        INTERMEDIATE("INTERMEDIATE", "中级", 2),
        ADVANCED("ADVANCED", "高级", 3),
        EXPERT("EXPERT", "专家级", 4);

        private final String code;
        private final String description;
        private final int level;

        DifficultyLevel(String code, String description, int level) {
            this.code = code;
            this.description = description;
            this.level = level;
        }

        public String getCode() {
            return code;
        }

        public String getDescription() {
            return description;
        }

        public int getLevel() {
            return level;
        }

        public static DifficultyLevel fromCode(String code) {
            for (DifficultyLevel level : values()) {
                if (level.code.equals(code)) {
                    return level;
                }
            }
            throw new IllegalArgumentException("未知的课程难度: " + code);
        }
    }

    /**
     * 判断课程是否为草稿状态
     * 
     * @return 是否为草稿状态
     */
    public boolean isDraft() {
        return Status.DRAFT.getCode().equals(this.status);
    }

    /**
     * 判断课程是否正在进行
     * 
     * @return 是否正在进行
     */
    public boolean isActive() {
        return Status.ACTIVE.getCode().equals(this.status);
    }

    /**
     * 判断课程是否已完成
     * 
     * @return 是否已完成
     */
    public boolean isCompleted() {
        return Status.COMPLETED.getCode().equals(this.status);
    }

    /**
     * 判断课程是否暂停
     * 
     * @return 是否暂停
     */
    public boolean isSuspended() {
        return Status.SUSPENDED.getCode().equals(this.status);
    }

    /**
     * 判断课程是否已取消
     * 
     * @return 是否已取消
     */
    public boolean isCancelled() {
        return Status.CANCELLED.getCode().equals(this.status);
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
     * 获取课程类型描述
     * 
     * @return 课程类型描述
     */
    public String getCourseTypeDescription() {
        try {
            return CourseType.fromCode(this.courseType).getDescription();
        } catch (IllegalArgumentException e) {
            return this.courseType;
        }
    }

    /**
     * 获取难度等级描述
     * 
     * @return 难度等级描述
     */
    public String getDifficultyLevelDescription() {
        try {
            return DifficultyLevel.fromCode(this.difficultyLevel).getDescription();
        } catch (IllegalArgumentException e) {
            return this.difficultyLevel;
        }
    }

    /**
     * 获取难度等级数值
     * 
     * @return 难度等级数值
     */
    public int getDifficultyLevelValue() {
        try {
            return DifficultyLevel.fromCode(this.difficultyLevel).getLevel();
        } catch (IllegalArgumentException e) {
            return 0;
        }
    }

    /**
     * 判断是否可以选课
     * 
     * @return 是否可以选课
     */
    public boolean canEnroll() {
        if (!isActive() || isDeleted) {
            return false;
        }
        if (maxEnrollment == null) {
            return true;
        }
        int current = currentEnrollment != null ? currentEnrollment : 0;
        return current < maxEnrollment;
    }

    /**
     * 判断课程是否已满
     * 
     * @return 是否已满
     */
    public boolean isFull() {
        if (maxEnrollment == null) {
            return false;
        }
        int current = currentEnrollment != null ? currentEnrollment : 0;
        return current >= maxEnrollment;
    }

    /**
     * 获取剩余选课名额
     * 
     * @return 剩余选课名额
     */
    public int getRemainingCapacity() {
        if (maxEnrollment == null) {
            return Integer.MAX_VALUE;
        }
        int current = currentEnrollment != null ? currentEnrollment : 0;
        return Math.max(0, maxEnrollment - current);
    }

    /**
     * 获取选课率
     * 
     * @return 选课率（百分比）
     */
    public double getEnrollmentRate() {
        if (maxEnrollment == null || maxEnrollment == 0) {
            return 0.0;
        }
        int current = currentEnrollment != null ? currentEnrollment : 0;
        return (double) current / maxEnrollment * 100;
    }

    /**
     * 判断课程是否在指定时间范围内
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
     * 判断课程是否即将开始（距离开始时间不足一周）
     * 
     * @return 是否即将开始
     */
    public boolean isAboutToStart() {
        if (startDate == null) {
            return false;
        }
        LocalDateTime now = LocalDateTime.now();
        return now.isBefore(startDate) && now.plusWeeks(1).isAfter(startDate);
    }

    /**
     * 判断课程是否即将结束（距离结束时间不足一周）
     * 
     * @return 是否即将结束
     */
    public boolean isAboutToEnd() {
        if (endDate == null) {
            return false;
        }
        return LocalDateTime.now().plusWeeks(1).isAfter(endDate);
    }

    /**
     * 获取课程进度百分比
     * 
     * @return 进度百分比
     */
    public double getProgressPercentage() {
        if (startDate == null || endDate == null) {
            return 0.0;
        }
        
        LocalDateTime now = LocalDateTime.now();
        if (now.isBefore(startDate)) {
            return 0.0;
        }
        if (now.isAfter(endDate)) {
            return 100.0;
        }
        
        long totalDuration = java.time.Duration.between(startDate, endDate).toDays();
        long elapsedDuration = java.time.Duration.between(startDate, now).toDays();
        
        if (totalDuration <= 0) {
            return 100.0;
        }
        
        return Math.min(100.0, (double) elapsedDuration / totalDuration * 100);
    }

    /**
     * 获取课程标签列表
     * 
     * @return 标签列表
     */
    public String[] getTagsList() {
        if (tags == null || tags.trim().isEmpty()) {
            return new String[0];
        }
        return tags.split(",");
    }

    /**
     * 设置课程标签列表
     * 
     * @param tagsList 标签列表
     */
    public void setTagsList(String[] tagsList) {
        if (tagsList == null || tagsList.length == 0) {
            this.tags = null;
        } else {
            this.tags = String.join(",", tagsList);
        }
    }

    /**
     * 获取先修课程列表
     * 
     * @return 先修课程列表
     */
    public String[] getPrerequisitesList() {
        if (prerequisites == null || prerequisites.trim().isEmpty()) {
            return new String[0];
        }
        return prerequisites.split(",");
    }

    /**
     * 设置先修课程列表
     * 
     * @param prerequisitesList 先修课程列表
     */
    public void setPrerequisitesList(String[] prerequisitesList) {
        if (prerequisitesList == null || prerequisitesList.length == 0) {
            this.prerequisites = null;
        } else {
            this.prerequisites = String.join(",", prerequisitesList);
        }
    }

    /**
     * 获取完整课程名称（包含课程代码）
     * 
     * @return 完整课程名称
     */
    public String getFullCourseName() {
        if (courseCode != null && !courseCode.trim().isEmpty()) {
            return courseCode + " - " + courseName;
        }
        return courseName;
    }

    /**
     * 更新选课人数
     * 
     * @param count 新的选课人数
     */
    public void updateCurrentEnrollment(int count) {
        this.currentEnrollment = Math.max(0, count);
    }

    /**
     * 增加选课人数
     * 
     * @param increment 增加数量
     */
    public void incrementEnrollment(int increment) {
        int current = currentEnrollment != null ? currentEnrollment : 0;
        this.currentEnrollment = Math.max(0, current + increment);
    }

    /**
     * 减少选课人数
     * 
     * @param decrement 减少数量
     */
    public void decrementEnrollment(int decrement) {
        incrementEnrollment(-decrement);
    }

    /**
     * 判断是否为必修课
     * 
     * @return 是否为必修课
     */
    public boolean isRequired() {
        return CourseType.REQUIRED.getCode().equals(this.courseType);
    }

    /**
     * 判断是否为选修课
     * 
     * @return 是否为选修课
     */
    public boolean isElective() {
        return CourseType.ELECTIVE.getCode().equals(this.courseType);
    }

    /**
     * 判断是否为核心课程
     * 
     * @return 是否为核心课程
     */
    public boolean isCore() {
        return CourseType.CORE.getCode().equals(this.courseType);
    }
}