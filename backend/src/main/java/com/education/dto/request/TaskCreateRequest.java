package com.education.dto.request;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;
import java.time.LocalDateTime;
import java.util.List;

/**
 * 任务创建请求DTO
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
@Data
@Schema(description = "任务创建请求参数")
public class TaskCreateRequest {
    
    @Schema(description = "任务标题", example = "Java基础编程练习")
    @NotBlank(message = "任务标题不能为空")
    @Size(max = 200, message = "任务标题长度不能超过200个字符")
    private String title;
    
    @Schema(description = "任务描述", example = "完成Java基础语法的编程练习题")
    @Size(max = 2000, message = "任务描述长度不能超过2000个字符")
    private String description;
    
    @Schema(description = "任务类型", example = "HOMEWORK")
    @NotBlank(message = "任务类型不能为空")
    private String taskType;
    
    @Schema(description = "课程ID", example = "1")
    @NotNull(message = "课程ID不能为空")
    private Long courseId;
    
    @Schema(description = "班级ID列表", example = "[1, 2, 3]")
    @NotNull(message = "班级ID列表不能为空")
    private List<Long> classIds;
    
    @Schema(description = "开始时间", example = "2024-01-01T08:00:00")
    @NotNull(message = "开始时间不能为空")
    private LocalDateTime startTime;
    
    @Schema(description = "截止时间", example = "2024-01-07T23:59:59")
    @NotNull(message = "截止时间不能为空")
    private LocalDateTime endTime;
    
    @Schema(description = "总分", example = "100")
    @NotNull(message = "总分不能为空")
    private Integer totalScore;
    
    @Schema(description = "是否允许迟交", example = "true")
    private Boolean allowLateSubmission = false;
    
    @Schema(description = "迟交扣分比例", example = "0.1")
    private Double lateSubmissionPenalty = 0.0;
    
    @Schema(description = "最大提交次数", example = "3")
    private Integer maxSubmissions = 1;
    
    @Schema(description = "是否需要同伴评价", example = "false")
    private Boolean requirePeerReview = false;
    
    @Schema(description = "任务要求", example = "请按照要求完成编程练习")
    private String requirements;
    
    @Schema(description = "评分标准", example = "代码正确性50%，代码规范30%，创新性20%")
    private String gradingCriteria;
    
    @Schema(description = "附件文件ID列表", example = "[1, 2, 3]")
    private List<Long> attachmentIds;
    
    @Schema(description = "任务标签", example = "[\"Java\", \"基础\", \"编程\"]")
    private List<String> tags;
}