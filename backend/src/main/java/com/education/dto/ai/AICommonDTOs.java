package com.education.dto.ai;

import lombok.Data;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.NotEmpty;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

/**
 * AI功能通用DTO类集合
 */
public class AICommonDTOs {

    // ========== 请求DTOs ==========
    
    @Data
    public static class AIGradeRequest {
        @NotNull(message = "任务ID不能为空")
        private Long taskId;
        @NotNull(message = "提交ID不能为空")
        private Long submissionId;
        private String gradeType = "auto";
        private Boolean needDetailedFeedback = true;
        private String gradingStandard = "normal";
    }

    @Data
    public static class AIBatchGradeRequest {
        @NotNull(message = "任务ID不能为空")
        private Long taskId;
        @NotEmpty(message = "提交ID列表不能为空")
        private List<Long> submissionIds;
        private String gradeType = "auto";
        private Boolean needDetailedFeedback = true;
        private String gradingStandard = "normal";
    }

    @Data
    public static class AIRecommendationRequest {
        @NotNull(message = "学生ID不能为空")
        private Long studentId;
        private Long courseId;
        private String learningGoal;
        private Integer recommendationCount = 10;
        private List<String> preferredTypes;
    }

    @Data
    public static class AIAbilityAnalysisRequest {
        @NotNull(message = "学生ID不能为空")
        private Long studentId;
        private Long courseId;
        private String timeRange = "month";
        private List<String> analysisDimensions;
    }

    @Data
    public static class AIKnowledgeGraphRequest {
        @NotNull(message = "课程ID不能为空")
        private Long courseId;
        private Integer chapterCount;
        private String difficulty = "normal";
        private Boolean includePrerequisites = true;
    }

    @Data
    public static class AIQuestionGenerationRequest {
        @NotEmpty(message = "知识点列表不能为空")
        private List<String> knowledgePoints;
        private String questionType = "multiple_choice";
        private Integer questionCount = 10;
        private String difficulty = "normal";
    }

    @Data
    public static class AILearningPathRequest {
        @NotNull(message = "学生ID不能为空")
        private Long studentId;
        private List<String> targetSkills;
        private String timeFrame = "month";
        private String learningStyle;
    }

    @Data
    public static class AIClassroomAnalysisRequest {
        @NotNull(message = "班级ID不能为空")
        private Long classId;
        private String timeRange = "week";
        private List<String> analysisTypes;
    }

    @Data
    public static class AITeachingSuggestionRequest {
        @NotNull(message = "课程ID不能为空")
        private Long courseId;
        private String studentGroup = "all";
        private String difficultyLevel = "normal";
    }

    @Data
    public static class AIModelConfigRequest {
        @NotNull(message = "模型类型不能为空")
        private String modelType;
        private Map<String, Object> parameters;
        private String description;
    }

    @Data
    public static class AIModelTrainingRequest {
        @NotNull(message = "数据集ID不能为空")
        private String datasetId;
        @NotNull(message = "模型类型不能为空")
        private String modelType;
        private Map<String, Object> hyperParameters;
        private String trainingDescription;
    }

    // ========== 响应DTOs ==========

    @Data
    public static class AIGradeResponse {
        private Long submissionId;
        private Long taskId;
        private Double score;
        private String grade;
        private String feedback;
        private List<ScoreDetail> scoreDetails;
        private String aiModel;
        private Double confidence;
        private LocalDateTime gradeTime;
    }

    @Data
    public static class AIBatchGradeResponse {
        private Long taskId;
        private Integer totalSubmissions;
        private Integer successCount;
        private Integer failureCount;
        private List<AIGradeResponse> results;
        private List<BatchError> errors;
        private LocalDateTime processTime;
        private String batchId;
    }

    @Data
    public static class AIRecommendationResponse {
        private Long studentId;
        private List<Recommendation> recommendations;
        private String recommendationType;
        private LocalDateTime generateTime;
        private String aiModel;
    }

    @Data
    public static class AIAbilityAnalysisResponse {
        private Long studentId;
        private Map<String, Double> abilityScores;
        private List<AbilityInsight> insights;
        private String overallLevel;
        private List<ImprovementSuggestion> suggestions;
        private LocalDateTime analysisTime;
    }

    @Data
    public static class AIKnowledgeGraphResponse {
        private Long courseId;
        private List<KnowledgeNode> nodes;
        private List<KnowledgeEdge> edges;
        private String graphDescription;
        private LocalDateTime generateTime;
    }

    @Data
    public static class AIQuestionGenerationResponse {
        private List<GeneratedQuestion> questions;
        private String questionType;
        private String difficulty;
        private LocalDateTime generateTime;
    }

    @Data
    public static class AILearningPathResponse {
        private Long studentId;
        private List<LearningStep> learningPath;
        private String pathDescription;
        private Integer estimatedDays;
        private LocalDateTime generateTime;
    }

    @Data
    public static class AIClassroomAnalysisResponse {
        private Long classId;
        private Map<String, Object> performanceMetrics;
        private List<ClassInsight> insights;
        private List<ActionRecommendation> recommendations;
        private LocalDateTime analysisTime;
    }

    @Data
    public static class AITeachingSuggestionResponse {
        private Long courseId;
        private List<TeachingSuggestion> suggestions;
        private String targetGroup;
        private LocalDateTime generateTime;
    }

    @Data
    public static class AIDocumentAnalysisResponse {
        private String fileName;
        private String analysisType;
        private Map<String, Object> analysisResults;
        private List<String> keyPoints;
        private String summary;
        private LocalDateTime analysisTime;
    }

    @Data
    public static class AIAnalysisHistoryResponse {
        private Long id;
        private String analysisType;
        private String description;
        private LocalDateTime createTime;
        private String status;
        private String resultSummary;
    }

    @Data
    public static class AIModelStatusResponse {
        private String modelType;
        private String status;
        private String version;
        private Map<String, Object> performance;
        private LocalDateTime lastUpdate;
    }

    @Data
    public static class AIModelTrainingResponse {
        private String trainingId;
        private String status;
        private String modelType;
        private LocalDateTime startTime;
        private String estimatedDuration;
    }

    @Data
    public static class AITrainingProgressResponse {
        private String trainingId;
        private String status;
        private Integer progress;
        private String currentStage;
        private Map<String, Object> metrics;
        private String estimatedTimeRemaining;
    }

    // ========== 通用数据类 ==========

    @Data
    public static class ScoreDetail {
        private String criterion;
        private Double score;
        private Double maxScore;
        private String comment;
    }

    @Data
    public static class BatchError {
        private Long submissionId;
        private String errorMessage;
        private String errorCode;
    }

    @Data
    public static class Recommendation {
        private String type;
        private String title;
        private String description;
        private String resourceUrl;
        private Double relevanceScore;
    }

    @Data
    public static class AbilityInsight {
        private String dimension;
        private String description;
        private String level;
        private Double score;
    }

    @Data
    public static class ImprovementSuggestion {
        private String area;
        private String suggestion;
        private String priority;
        private List<String> actions;
    }

    @Data
    public static class KnowledgeNode {
        private String id;
        private String name;
        private String type;
        private String description;
        private Integer level;
    }

    @Data
    public static class KnowledgeEdge {
        private String fromNode;
        private String toNode;
        private String relationship;
        private Double weight;
    }

    @Data
    public static class GeneratedQuestion {
        private String id;
        private String type;
        private String question;
        private List<String> options;
        private String correctAnswer;
        private String explanation;
        private String difficulty;
    }

    @Data
    public static class LearningStep {
        private Integer order;
        private String title;
        private String description;
        private List<String> resources;
        private Integer estimatedHours;
        private String type;
    }

    @Data
    public static class ClassInsight {
        private String category;
        private String insight;
        private String impact;
        private Double confidence;
    }

    @Data
    public static class ActionRecommendation {
        private String action;
        private String description;
        private String priority;
        private String expectedOutcome;
    }

    @Data
    public static class TeachingSuggestion {
        private String topic;
        private String suggestion;
        private String methodology;
        private List<String> resources;
        private String difficulty;
    }
} 