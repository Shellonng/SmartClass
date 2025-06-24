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
 * 任务实体类
 * 
 * @author Education Platform Team
 * @version 1.0.0
 */
@Data
@EqualsAndHashCode(callSuper = false)
@Accessors(chain = true)
@TableName("task")
@Schema(description = "任务信息")
public class Task implements Serializable {

    private static final long serialVersionUID = 1L;

    @Schema(description = "任务ID")
    @TableId(value = "id", type = IdType.ASSIGN_ID)
    private Long id;

    @Schema(description = "任务标题", example = "Java基础练习")
    @TableField("title")
    @NotBlank(message = "任务标题不能为空")
    @Size(max = 200, message = "任务标题长度不能超过200个字符")
    private String title;

    @Schema(description = "课程ID")
    @TableField("course_id")
    @NotNull(message = "课程ID不能为空")
    private Long courseId;

    @Schema(description = "创建教师ID")
    @TableField("teacher_id")
    @NotNull(message = "创建教师ID不能为空")
    private Long teacherId;

    @Schema(description = "任务类型", example = "ASSIGNMENT")
    @TableField("task_type")
    @NotBlank(message = "任务类型不能为空")
    private String taskType;

    @Schema(description = "任务描述")
    @TableField("description")
    @Size(max = 5000, message = "任务描述长度不能超过5000个字符")
    private String description;

    @Schema(description = "任务要求")
    @TableField("requirements")
    @Size(max = 3000, message = "任务要求长度不能超过3000个字符")
    private String requirements;

    @Schema(description = "任务内容")
    @TableField("content")
    @Size(max = 10000, message = "任务内容长度不能超过10000个字符")
    private String content;

    @Schema(description = "附件URL")
    @TableField("attachments")
    @Size(max = 2000, message = "附件URL长度不能超过2000个字符")
    private String attachments;

    @Schema(description = "开始时间")
    @TableField("start_time")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime startTime;

    @Schema(description = "截止时间")
    @TableField("due_time")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime dueTime;

    @Schema(description = "提交方式", example = "ONLINE")
    @TableField("submission_type")
    private String submissionType;

    @Schema(description = "最大分数", example = "100.0")
    @TableField("max_score")
    private BigDecimal maxScore;

    @Schema(description = "权重", example = "0.3")
    @TableField("weight")
    private BigDecimal weight;

    @Schema(description = "是否允许迟交", example = "true")
    @TableField("allow_late_submission")
    private Boolean allowLateSubmission;

    @Schema(description = "迟交扣分比例", example = "0.1")
    @TableField("late_penalty_rate")
    private BigDecimal latePenaltyRate;

    @Schema(description = "最大提交次数", example = "3")
    @TableField("max_attempts")
    private Integer maxAttempts;

    @Schema(description = "是否显示成绩", example = "true")
    @TableField("show_score")
    private Boolean showScore;

    @Schema(description = "是否自动评分", example = "false")
    @TableField("auto_grade")
    private Boolean autoGrade;

    @Schema(description = "评分标准")
    @TableField("grading_criteria")
    @Size(max = 3000, message = "评分标准长度不能超过3000个字符")
    private String gradingCriteria;

    @Schema(description = "任务状态", example = "PUBLISHED")
    @TableField("status")
    private String status;

    @Schema(description = "优先级", example = "MEDIUM")
    @TableField("priority")
    private String priority;

    @Schema(description = "难度等级", example = "INTERMEDIATE")
    @TableField("difficulty_level")
    private String difficultyLevel;

    @Schema(description = "预计完成时间（小时）", example = "2.5")
    @TableField("estimated_hours")
    private BigDecimal estimatedHours;

    @Schema(description = "任务标签")
    @TableField("tags")
    @Size(max = 500, message = "任务标签长度不能超过500个字符")
    private String tags;

    @Schema(description = "是否公开", example = "true")
    @TableField("is_public")
    private Boolean isPublic;

    @Schema(description = "发布时间")
    @TableField("publish_time")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime publishTime;

    @Schema(description = "完成人数", example = "25")
    @TableField("completion_count")
    private Integer completionCount;

    @Schema(description = "提交人数", example = "30")
    @TableField("submission_count")
    private Integer submissionCount;

    @Schema(description = "平均分", example = "85.5")
    @TableField("average_score")
    private BigDecimal averageScore;

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
    @Schema(description = "课程信息")
    private Course course;

    @TableField(exist = false)
    @Schema(description = "创建教师信息")
    private Teacher teacher;

    @TableField(exist = false)
    @Schema(description = "任务提交列表")
    private List<TaskSubmission> submissions;

    @TableField(exist = false)
    @Schema(description = "任务资源列表")
    private List<Resource> resources;

    /**
     * 任务类型枚举
     */
    public enum TaskType {
        ASSIGNMENT("ASSIGNMENT", "作业"),
        QUIZ("QUIZ", "测验"),
        EXAM("EXAM", "考试"),
        PROJECT("PROJECT", "项目"),
        DISCUSSION("DISCUSSION", "讨论"),
        READING("READING", "阅读"),
        PRACTICE("PRACTICE", "练习"),
        PRESENTATION("PRESENTATION", "演示"),
        REPORT("REPORT", "报告"),
        EXPERIMENT("EXPERIMENT", "实验");

        private final String code;
        private final String description;

        TaskType(String code, String description) {
            this.code = code;
            this.description = description;
        }

        public String getCode() {
            return code;
        }

        public String getDescription() {
            return description;
        }

        public static TaskType fromCode(String code) {
            for (TaskType type : values()) {
                if (type.code.equals(code)) {
                    return type;
                }
            }
            throw new IllegalArgumentException("未知的任务类型: " + code);
        }
    }

    /**
     * 任务状态枚举
     */
    public enum Status {
        DRAFT("DRAFT", "草稿"),
        PUBLISHED("PUBLISHED", "已发布"),
        IN_PROGRESS("IN_PROGRESS", "进行中"),
        COMPLETED("COMPLETED", "已完成"),
        OVERDUE("OVERDUE", "已逾期"),
        CANCELLED("CANCELLED", "已取消"),
        ARCHIVED("ARCHIVED", "已归档");

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
            throw new IllegalArgumentException("未知的任务状态: " + code);
        }
    }

    /**
     * 提交方式枚举
     */
    public enum SubmissionType {
        ONLINE("ONLINE", "在线提交"),
        OFFLINE("OFFLINE", "线下提交"),
        FILE_UPLOAD("FILE_UPLOAD", "文件上传"),
        TEXT_INPUT("TEXT_INPUT", "文本输入"),
        LINK_SUBMISSION("LINK_SUBMISSION", "链接提交"),
        MIXED("MIXED", "混合方式");

        private final String code;
        private final String description;

        SubmissionType(String code, String description) {
            this.code = code;
            this.description = description;
        }

        public String getCode() {
            return code;
        }

        public String getDescription() {
            return description;
        }

        public static SubmissionType fromCode(String code) {
            for (SubmissionType type : values()) {
                if (type.code.equals(code)) {
                    return type;
                }
            }
            throw new IllegalArgumentException("未知的提交方式: " + code);
        }
    }

    /**
     * 优先级枚举
     */
    public enum Priority {
        LOW("LOW", "低", 1),
        MEDIUM("MEDIUM", "中", 2),
        HIGH("HIGH", "高", 3),
        URGENT("URGENT", "紧急", 4);

        private final String code;
        private final String description;
        private final int level;

        Priority(String code, String description, int level) {
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

        public static Priority fromCode(String code) {
            for (Priority priority : values()) {
                if (priority.code.equals(code)) {
                    return priority;
                }
            }
            throw new IllegalArgumentException("未知的优先级: " + code);
        }
    }

    /**
     * 难度等级枚举
     */
    public enum DifficultyLevel {
        EASY("EASY", "简单", 1),
        MEDIUM("MEDIUM", "中等", 2),
        HARD("HARD", "困难", 3),
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
            throw new IllegalArgumentException("未知的难度等级: " + code);
        }
    }

    /**
     * 判断任务是否为草稿状态
     * 
     * @return 是否为草稿状态
     */
    public boolean isDraft() {
        return Status.DRAFT.getCode().equals(this.status);
    }

    /**
     * 判断任务是否已发布
     * 
     * @return 是否已发布
     */
    public boolean isPublished() {
        return Status.PUBLISHED.getCode().equals(this.status);
    }

    /**
     * 判断任务是否进行中
     * 
     * @return 是否进行中
     */
    public boolean isInProgress() {
        return Status.IN_PROGRESS.getCode().equals(this.status);
    }

    /**
     * 判断任务是否已完成
     * 
     * @return 是否已完成
     */
    public boolean isCompleted() {
        return Status.COMPLETED.getCode().equals(this.status);
    }

    /**
     * 判断任务是否已逾期
     * 
     * @return 是否已逾期
     */
    public boolean isOverdue() {
        return Status.OVERDUE.getCode().equals(this.status) || 
               (dueTime != null && LocalDateTime.now().isAfter(dueTime));
    }

    /**
     * 判断任务是否已取消
     * 
     * @return 是否已取消
     */
    public boolean isCancelled() {
        return Status.CANCELLED.getCode().equals(this.status);
    }

    /**
     * 判断任务是否已归档
     * 
     * @return 是否已归档
     */
    public boolean isArchived() {
        return Status.ARCHIVED.getCode().equals(this.status);
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
     * 获取任务类型描述
     * 
     * @return 任务类型描述
     */
    public String getTaskTypeDescription() {
        try {
            return TaskType.fromCode(this.taskType).getDescription();
        } catch (IllegalArgumentException e) {
            return this.taskType;
        }
    }

    /**
     * 获取提交方式描述
     * 
     * @return 提交方式描述
     */
    public String getSubmissionTypeDescription() {
        try {
            return SubmissionType.fromCode(this.submissionType).getDescription();
        } catch (IllegalArgumentException e) {
            return this.submissionType;
        }
    }

    /**
     * 获取优先级描述
     * 
     * @return 优先级描述
     */
    public String getPriorityDescription() {
        try {
            return Priority.fromCode(this.priority).getDescription();
        } catch (IllegalArgumentException e) {
            return this.priority;
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
     * 判断是否可以提交
     * 
     * @return 是否可以提交
     */
    public boolean canSubmit() {
        if (!isPublished() && !isInProgress()) {
            return false;
        }
        
        LocalDateTime now = LocalDateTime.now();
        
        // 检查开始时间
        if (startTime != null && now.isBefore(startTime)) {
            return false;
        }
        
        // 检查截止时间
        if (dueTime != null && now.isAfter(dueTime)) {
            return allowLateSubmission != null && allowLateSubmission;
        }
        
        return true;
    }

    /**
     * 判断是否在截止时间内
     * 
     * @return 是否在截止时间内
     */
    public boolean isWithinDeadline() {
        if (dueTime == null) {
            return true;
        }
        return LocalDateTime.now().isBefore(dueTime);
    }

    /**
     * 获取剩余时间（小时）
     * 
     * @return 剩余时间（小时），负数表示已逾期
     */
    public long getRemainingHours() {
        if (dueTime == null) {
            return Long.MAX_VALUE;
        }
        return java.time.Duration.between(LocalDateTime.now(), dueTime).toHours();
    }

    /**
     * 获取剩余天数
     * 
     * @return 剩余天数，负数表示已逾期
     */
    public long getRemainingDays() {
        return getRemainingHours() / 24;
    }

    /**
     * 判断是否即将到期（距离截止时间不足24小时）
     * 
     * @return 是否即将到期
     */
    public boolean isDueSoon() {
        if (dueTime == null) {
            return false;
        }
        long remainingHours = getRemainingHours();
        return remainingHours > 0 && remainingHours <= 24;
    }

    /**
     * 获取完成率
     * 
     * @return 完成率（百分比）
     */
    public double getCompletionRate() {
        if (submissionCount == null || submissionCount == 0) {
            return 0.0;
        }
        int completed = completionCount != null ? completionCount : 0;
        return (double) completed / submissionCount * 100;
    }

    /**
     * 获取提交率（相对于课程选课人数）
     * 
     * @param totalStudents 总学生数
     * @return 提交率（百分比）
     */
    public double getSubmissionRate(int totalStudents) {
        if (totalStudents <= 0) {
            return 0.0;
        }
        int submitted = submissionCount != null ? submissionCount : 0;
        return (double) submitted / totalStudents * 100;
    }

    /**
     * 获取任务标签列表
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
     * 设置任务标签列表
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
     * 获取附件列表
     * 
     * @return 附件列表
     */
    public String[] getAttachmentsList() {
        if (attachments == null || attachments.trim().isEmpty()) {
            return new String[0];
        }
        return attachments.split(",");
    }

    /**
     * 设置附件列表
     * 
     * @param attachmentsList 附件列表
     */
    public void setAttachmentsList(String[] attachmentsList) {
        if (attachmentsList == null || attachmentsList.length == 0) {
            this.attachments = null;
        } else {
            this.attachments = String.join(",", attachmentsList);
        }
    }

    /**
     * 更新统计信息
     * 
     * @param submissionCount 提交人数
     * @param completionCount 完成人数
     * @param averageScore 平均分
     */
    public void updateStatistics(int submissionCount, int completionCount, BigDecimal averageScore) {
        this.submissionCount = submissionCount;
        this.completionCount = completionCount;
        this.averageScore = averageScore;
    }

    /**
     * 计算迟交扣分
     * 
     * @param originalScore 原始分数
     * @param submissionTime 提交时间
     * @return 扣分后的分数
     */
    public BigDecimal calculateLateScore(BigDecimal originalScore, LocalDateTime submissionTime) {
        if (originalScore == null || submissionTime == null || dueTime == null) {
            return originalScore;
        }
        
        if (!submissionTime.isAfter(dueTime)) {
            return originalScore; // 未迟交
        }
        
        if (latePenaltyRate == null || latePenaltyRate.compareTo(BigDecimal.ZERO) <= 0) {
            return originalScore; // 无扣分
        }
        
        // 计算迟交天数
        long lateDays = java.time.Duration.between(dueTime, submissionTime).toDays();
        if (lateDays <= 0) {
            lateDays = 1; // 至少算一天
        }
        
        // 计算扣分
        BigDecimal penalty = latePenaltyRate.multiply(BigDecimal.valueOf(lateDays));
        BigDecimal deduction = originalScore.multiply(penalty);
        BigDecimal finalScore = originalScore.subtract(deduction);
        
        // 确保分数不为负
        return finalScore.max(BigDecimal.ZERO);
    }

    /**
     * 判断是否为作业类型
     * 
     * @return 是否为作业类型
     */
    public boolean isAssignment() {
        return TaskType.ASSIGNMENT.getCode().equals(this.taskType);
    }

    /**
     * 判断是否为测验类型
     * 
     * @return 是否为测验类型
     */
    public boolean isQuiz() {
        return TaskType.QUIZ.getCode().equals(this.taskType);
    }

    /**
     * 判断是否为考试类型
     * 
     * @return 是否为考试类型
     */
    public boolean isExam() {
        return TaskType.EXAM.getCode().equals(this.taskType);
    }

    /**
     * 判断是否为项目类型
     * 
     * @return 是否为项目类型
     */
    public boolean isProject() {
        return TaskType.PROJECT.getCode().equals(this.taskType);
    }
}