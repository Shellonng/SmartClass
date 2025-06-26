package com.education.dto.ai;

import lombok.Data;
import jakarta.validation.constraints.NotNull;

/**
 * AI批改请求DTO
 */
@Data
public class AIGradeRequest {
    
    @NotNull(message = "任务ID不能为空")
    private Long taskId;
    
    @NotNull(message = "提交ID不能为空")
    private Long submissionId;
    
    /**
     * 批改类型：auto-自动批改，manual-人工复核
     */
    private String gradeType = "auto";
    
    /**
     * 是否需要详细反馈
     */
    private Boolean needDetailedFeedback = true;
    
    /**
     * 评分标准：strict-严格，normal-正常，lenient-宽松
     */
    private String gradingStandard = "normal";
} 