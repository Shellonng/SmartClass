package com.education.dto.response;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.time.LocalDateTime;
import java.util.List;

/**
 * 任务响应DTO
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
@Data
@Schema(description = "任务响应数据")
public class TaskResponse {
    
    @Schema(description = "任务ID", example = "1")
    private Long taskId;
    
    @Schema(description = "任务标题", example = "Java基础编程练习")
    private String title;
    
    @Schema(description = "任务描述", example = "完成Java基础语法的编程练习题")
    private String description;
    
    @Schema(description = "任务类型", example = "HOMEWORK")
    private String taskType;
    
    @Schema(description = "课程ID", example = "1")
    private Long courseId;
    
    @Schema(description = "课程名称", example = "Java程序设计")
    private String courseName;
    
    @Schema(description = "教师ID", example = "1")
    private Long teacherId;
    
    @Schema(description = "教师姓名", example = "张老师")
    private String teacherName;
    
    @Schema(description = "开始时间", example = "2024-01-01T08:00:00")
    private LocalDateTime startTime;
    
    @Schema(description = "截止时间", example = "2024-01-07T23:59:59")
    private LocalDateTime endTime;
    
    @Schema(description = "总分", example = "100")
    private Integer totalScore;
    
    @Schema(description = "是否允许迟交", example = "true")
    private Boolean allowLateSubmission;
    
    @Schema(description = "迟交扣分比例", example = "0.1")
    private Double lateSubmissionPenalty;
    
    @Schema(description = "最大提交次数", example = "3")
    private Integer maxSubmissions;
    
    @Schema(description = "是否需要同伴评价", example = "false")
    private Boolean requirePeerReview;
    
    @Schema(description = "任务要求", example = "请按照要求完成编程练习")
    private String requirements;
    
    @Schema(description = "评分标准", example = "代码正确性50%，代码规范30%，创新性20%")
    private String gradingCriteria;
    
    @Schema(description = "任务状态", example = "PUBLISHED")
    private String status;
    
    @Schema(description = "提交统计")
    private SubmissionStats submissionStats;
    
    @Schema(description = "任务标签", example = "[\"Java\", \"基础\", \"编程\"]")
    private List<String> tags;
    
    @Schema(description = "附件列表")
    private List<AttachmentInfo> attachments;
    
    @Schema(description = "创建时间", example = "2024-01-01T12:00:00")
    private LocalDateTime createTime;
    
    @Schema(description = "更新时间", example = "2024-01-01T12:00:00")
    private LocalDateTime updateTime;
    
    @Data
    @Schema(description = "提交统计信息")
    public static class SubmissionStats {
        @Schema(description = "总学生数", example = "50")
        private Integer totalStudents;
        
        @Schema(description = "已提交数", example = "45")
        private Integer submittedCount;
        
        @Schema(description = "未提交数", example = "5")
        private Integer notSubmittedCount;
        
        @Schema(description = "已批改数", example = "40")
        private Integer gradedCount;
        
        @Schema(description = "平均分", example = "85.5")
        private Double averageScore;
    }
    
    @Data
    @Schema(description = "附件信息")
    public static class AttachmentInfo {
        @Schema(description = "文件ID", example = "1")
        private Long fileId;
        
        @Schema(description = "文件名", example = "练习题.pdf")
        private String fileName;
        
        @Schema(description = "文件大小", example = "1024000")
        private Long fileSize;
        
        @Schema(description = "文件类型", example = "application/pdf")
        private String fileType;
        
        @Schema(description = "下载URL", example = "https://example.com/download/1")
        private String downloadUrl;
    }
}