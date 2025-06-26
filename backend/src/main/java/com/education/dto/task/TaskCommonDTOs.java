package com.education.dto.task;

import lombok.Data;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.NotBlank;
import java.time.LocalDateTime;
import java.util.List;
import java.math.BigDecimal;
import io.swagger.v3.oas.annotations.media.Schema;

/**
 * 任务相关DTO类集合
 */
public class TaskCommonDTOs {

    // ========== 请求DTOs ==========

    @Data
    public static class TaskCreateRequest {
        @NotBlank(message = "任务标题不能为空")
        private String title;
        
        private String description;
        
        @NotNull(message = "任务类型不能为空")
        private String type;
        
        @NotNull(message = "截止时间不能为空")
        private LocalDateTime deadline;
        
        private Long courseId;
        private Long classId;
        private Integer totalScore;
        private String instructions;
        private List<String> attachments;
        private Boolean allowLateSubmission = false;
        private String difficultyLevel = "normal";
    }

    @Data
    public static class TaskUpdateRequest {
        private String title;
        private String description;
        private String type;
        private LocalDateTime deadline;
        private Integer totalScore;
        private String instructions;
        private List<String> attachments;
        private Boolean allowLateSubmission;
        private String difficultyLevel;
        private String status;
    }

    @Data
    public static class TaskGradeRequest {
        @NotNull(message = "分数不能为空")
        private Double score;
        
        private String feedback;
        private String status = "graded";
        private List<GradeDetail> gradeDetails;
    }

    @Data
    public static class TaskBatchGradeRequest {
        @NotNull(message = "提交ID不能为空")
        private Long submissionId;
        
        @NotNull(message = "分数不能为空")
        private Double score;
        
        private String feedback;
        private String status = "graded";
    }

    @Data
    public static class TaskCopyRequest {
        @NotBlank(message = "新任务标题不能为空")
        private String newTitle;
        
        private Long targetCourseId;
        private Long targetClassId;
        private LocalDateTime newDeadline;
        private Boolean copySubmissions = false;
    }

    @Data
    public static class TaskExtendRequest {
        @NotNull(message = "新截止时间不能为空")
        private LocalDateTime newDeadline;
        
        private String reason;
        private Boolean notifyStudents = true;
    }

    @Data
    public static class TaskFromTemplateRequest {
        @NotBlank(message = "任务标题不能为空")
        private String title;
        
        private String description;
        private LocalDateTime deadline;
        private Long courseId;
        private Long classId;
        private Integer totalScore;
    }

    // ========== 响应DTOs ==========

    @Data
    public static class TaskResponse {
        private Long id;
        private String title;
        private String description;
        private String type;
        private String status;
        private LocalDateTime deadline;
        private LocalDateTime createTime;
        private LocalDateTime updateTime;
        private Long courseId;
        private String courseName;
        private Long classId;
        private String className;
        private String creatorName;
        private Integer totalScore;
        private Integer submissionCount;
        private Integer gradedCount;
        private String difficultyLevel;
        private Boolean allowLateSubmission;
    }

    @Data
    public static class TaskDetailResponse {
        private Long id;
        private String title;
        private String description;
        private String type;
        private String status;
        private LocalDateTime deadline;
        private LocalDateTime createTime;
        private LocalDateTime updateTime;
        private Long courseId;
        private String courseName;
        private Long classId;
        private String className;
        private String creatorName;
        private Integer totalScore;
        private String instructions;
        private List<String> attachments;
        private Boolean allowLateSubmission;
        private String difficultyLevel;
        
        // 统计信息
        private TaskStatistics statistics;
        
        // 最近提交
        private List<RecentSubmission> recentSubmissions;
    }

    @Data
    public static class TaskSubmissionResponse {
        private Long id;
        private Long taskId;
        private String taskTitle;
        private Long studentId;
        private String studentName;
        private String content;
        private List<String> attachments;
        private LocalDateTime submitTime;
        private String status;
        private Double score;
        private String feedback;
        private String grade;
        private LocalDateTime gradeTime;
        private String graderName;
        private Boolean isLate;
    }

    @Data
    public static class TaskStatisticsResponse {
        private Long taskId;
        private String taskTitle;
        private Integer totalStudents;
        private Integer submissionCount;
        private Integer gradedCount;
        private Integer pendingCount;
        private Integer lateSubmissionCount;
        private Double averageScore;
        private Double highestScore;
        private Double lowestScore;
        private Double completionRate;
        private Double averageGradingTime;
        private List<ScoreDistribution> scoreDistribution;
    }

    @Data
    public static class TaskTemplateResponse {
        private Long id;
        private String name;
        private String description;
        private String type;
        private String subject;
        private String difficultyLevel;
        private String content;
        private Integer estimatedDuration;
        private String creatorName;
        private LocalDateTime createTime;
        private Integer usageCount;
    }

    /**
     * 任务列表响应
     */
    @Data
    @Schema(description = "任务列表响应")
    public static class TaskListResponse {
        @Schema(description = "任务ID")
        private Long taskId;
        
        @Schema(description = "任务标题")
        private String title;
        
        @Schema(description = "任务类型")
        private String type;
        
        @Schema(description = "课程ID")
        private Long courseId;
        
        @Schema(description = "课程名称")
        private String courseName;
        
        @Schema(description = "任务状态")
        private String status;
        
        @Schema(description = "创建时间")
        private LocalDateTime createTime;
        
        @Schema(description = "截止时间")
        private LocalDateTime dueTime;
        
        @Schema(description = "最大分数")
        private BigDecimal maxScore;
        
        @Schema(description = "提交人数")
        private Integer submissionCount;
        
        @Schema(description = "已批改数量")
        private Integer gradedCount;
    }

    // ========== 通用数据类 ==========

    @Data
    public static class GradeDetail {
        private String criterion;
        private Double score;
        private Double maxScore;
        private String comment;
    }

    @Data
    public static class TaskStatistics {
        private Integer totalStudents;
        private Integer submissionCount;
        private Integer gradedCount;
        private Integer pendingCount;
        private Double averageScore;
        private Double completionRate;
    }

    @Data
    public static class RecentSubmission {
        private Long submissionId;
        private String studentName;
        private LocalDateTime submitTime;
        private String status;
        private Double score;
    }

    @Data
    public static class ScoreDistribution {
        private String range;
        private Integer count;
        private Double percentage;
    }
} 