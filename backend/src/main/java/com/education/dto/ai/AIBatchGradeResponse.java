package com.education.dto.ai;

import lombok.Data;
import java.time.LocalDateTime;
import java.util.List;

/**
 * AI批量批改响应DTO
 */
@Data
public class AIBatchGradeResponse {
    
    private Long taskId;
    private Integer totalSubmissions;
    private Integer successCount;
    private Integer failureCount;
    private List<AIGradeResponse> results;
    private List<BatchError> errors;
    private LocalDateTime processTime;
    private String batchId;
    
    @Data
    public static class BatchError {
        private Long submissionId;
        private String errorMessage;
        private String errorCode;
    }
} 