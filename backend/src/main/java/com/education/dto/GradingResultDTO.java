package com.education.dto;

import lombok.Data;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

/**
 * 智能批改结果DTO
 */
@Data
public class GradingResultDTO {
    
    /**
     * 工作流运行ID
     */
    private String workflowRunId;
    
    /**
     * 任务ID
     */
    private String taskId;
    
    /**
     * 执行状态：running, succeeded, failed, stopped
     */
    private String status;
    
    /**
     * 批改得分
     */
    private Integer score;
    
    /**
     * 满分
     */
    private Integer fullScore;
    
    /**
     * 得分率（百分比）
     */
    private Double scoreRate;
    
    /**
     * 批改反馈
     */
    private String feedback;
    
    /**
     * 详细批改意见
     */
    private String detailedFeedback;
    
    /**
     * 批改建议
     */
    private String suggestions;
    
    /**
     * 错误信息
     */
    private String error;
    
    /**
     * 耗时（秒）
     */
    private Double elapsedTime;
    
    /**
     * 总使用tokens
     */
    private Integer totalTokens;
    
    /**
     * 总步数
     */
    private Integer totalSteps;
    
    /**
     * 创建时间
     */
    private LocalDateTime createdAt;
    
    /**
     * 完成时间
     */
    private LocalDateTime finishedAt;
    
    /**
     * 批改维度得分
     */
    private List<DimensionScore> dimensionScores;
    
    /**
     * 工作流输出数据
     */
    private Map<String, Object> outputs;
    
    /**
     * 是否成功
     */
    private Boolean success;
    
    /**
     * 批改维度得分
     */
    @Data
    public static class DimensionScore {
        
        /**
         * 维度名称
         */
        private String dimension;
        
        /**
         * 维度得分
         */
        private Integer score;
        
        /**
         * 维度满分
         */
        private Integer fullScore;
        
        /**
         * 维度评价
         */
        private String comment;
    }
    
    /**
     * 创建成功的批改结果
     */
    public static GradingResultDTO success(String workflowRunId, Integer score, Integer fullScore, String feedback) {
        GradingResultDTO result = new GradingResultDTO();
        result.setWorkflowRunId(workflowRunId);
        result.setStatus("succeeded");
        result.setScore(score);
        result.setFullScore(fullScore);
        result.setScoreRate(fullScore > 0 ? (double) score / fullScore * 100 : 0.0);
        result.setFeedback(feedback);
        result.setSuccess(true);
        result.setCreatedAt(LocalDateTime.now());
        result.setFinishedAt(LocalDateTime.now());
        return result;
    }
    
    /**
     * 创建失败的批改结果
     */
    public static GradingResultDTO failed(String error) {
        GradingResultDTO result = new GradingResultDTO();
        result.setStatus("failed");
        result.setError(error);
        result.setSuccess(false);
        result.setCreatedAt(LocalDateTime.now());
        result.setFinishedAt(LocalDateTime.now());
        return result;
    }
} 