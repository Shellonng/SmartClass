package com.education.dto.ai;

import lombok.Data;
import java.time.LocalDateTime;
import java.util.List;

/**
 * AI批改响应DTO
 */
@Data
public class AIGradeResponse {
    
    private Long submissionId;
    private Long taskId;
    private Double score;
    private String grade;
    private String feedback;
    private List<ScoreDetail> scoreDetails;
    private String aiModel;
    private Double confidence;
    private LocalDateTime gradeTime;
    
    @Data
    public static class ScoreDetail {
        private String criterion;
        private Double score;
        private Double maxScore;
        private String comment;
    }
} 