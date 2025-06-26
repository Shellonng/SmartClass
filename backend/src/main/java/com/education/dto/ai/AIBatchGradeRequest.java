package com.education.dto.ai;

import lombok.Data;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.NotEmpty;
import java.util.List;

/**
 * AI批量批改请求DTO
 */
@Data
public class AIBatchGradeRequest {
    
    @NotNull(message = "任务ID不能为空")
    private Long taskId;
    
    @NotEmpty(message = "提交ID列表不能为空")
    private List<Long> submissionIds;
    
    private String gradeType = "auto";
    private Boolean needDetailedFeedback = true;
    private String gradingStandard = "normal";
} 