package com.education.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotEmpty;
import jakarta.validation.constraints.NotNull;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

/**
 * AI相关DTO
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
public class AIDTO {

    /**
     * AI对话请求DTO
     */
    public static class ChatRequest {
        @NotBlank(message = "消息内容不能为空")
        private String message;
        
        private String sessionId;
        private String context;
        private String chatType; // LEARNING, HOMEWORK_HELP, GENERAL
        
        // Getters and Setters
        public String getMessage() { return message; }
        public void setMessage(String message) { this.message = message; }
        public String getSessionId() { return sessionId; }
        public void setSessionId(String sessionId) { this.sessionId = sessionId; }
        public String getContext() { return context; }
        public void setContext(String context) { this.context = context; }
        public String getChatType() { return chatType; }
        public void setChatType(String chatType) { this.chatType = chatType; }
    }

    /**
     * AI对话响应DTO
     */
    public static class ChatResponse {
        private String response;
        private String sessionId;
        private LocalDateTime timestamp;
        private List<String> suggestions;
        private String confidence;
        
        // Getters and Setters
        public String getResponse() { return response; }
        public void setResponse(String response) { this.response = response; }
        public String getSessionId() { return sessionId; }
        public void setSessionId(String sessionId) { this.sessionId = sessionId; }
        public LocalDateTime getTimestamp() { return timestamp; }
        public void setTimestamp(LocalDateTime timestamp) { this.timestamp = timestamp; }
        public List<String> getSuggestions() { return suggestions; }
        public void setSuggestions(List<String> suggestions) { this.suggestions = suggestions; }
        public String getConfidence() { return confidence; }
        public void setConfidence(String confidence) { this.confidence = confidence; }
    }

    /**
     * AI学习分析请求DTO
     */
    public static class LearningAnalysisRequest {
        @NotNull(message = "学生ID不能为空")
        private Long studentId;
        
        private Long courseId;
        private String analysisType; // PROGRESS, DIFFICULTY, RECOMMENDATION
        private String timeRange;
        
        // Getters and Setters
        public Long getStudentId() { return studentId; }
        public void setStudentId(Long studentId) { this.studentId = studentId; }
        public Long getCourseId() { return courseId; }
        public void setCourseId(Long courseId) { this.courseId = courseId; }
        public String getAnalysisType() { return analysisType; }
        public void setAnalysisType(String analysisType) { this.analysisType = analysisType; }
        public String getTimeRange() { return timeRange; }
        public void setTimeRange(String timeRange) { this.timeRange = timeRange; }
    }

    /**
     * AI学习分析响应DTO
     */
    public static class LearningAnalysisResponse {
        private String analysisResult;
        private Double progressScore;
        private List<String> strengths;
        private List<String> weaknesses;
        private List<String> recommendations;
        private LocalDateTime analysisTime;
        
        // Getters and Setters
        public String getAnalysisResult() { return analysisResult; }
        public void setAnalysisResult(String analysisResult) { this.analysisResult = analysisResult; }
        public Double getProgressScore() { return progressScore; }
        public void setProgressScore(Double progressScore) { this.progressScore = progressScore; }
        public List<String> getStrengths() { return strengths; }
        public void setStrengths(List<String> strengths) { this.strengths = strengths; }
        public List<String> getWeaknesses() { return weaknesses; }
        public void setWeaknesses(List<String> weaknesses) { this.weaknesses = weaknesses; }
        public List<String> getRecommendations() { return recommendations; }
        public void setRecommendations(List<String> recommendations) { this.recommendations = recommendations; }
        public LocalDateTime getAnalysisTime() { return analysisTime; }
        public void setAnalysisTime(LocalDateTime analysisTime) { this.analysisTime = analysisTime; }
    }

    public class QuestionGenerateResponse {
    }

    public class QuestionGenerateRequest {
    }

    public class AutoGradeResponse {
    }

    public class AutoGradeRequest {
    }

    public class TeachingSuggestionResponse {
    }

    public class TeachingSuggestionRequest {
    }

    public class LearningBehaviorAnalysisResponse {
    }

    public class LearningBehaviorAnalysisRequest {
    }

    public class ContentRecommendationResponse {
    }

    public class ContentRecommendationRequest {
    }

    /**
     * 语音识别请求DTO
     */
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class SpeechRecognitionRequest {
        @NotBlank(message = "音频文件路径不能为空")
        private String audioFilePath;
        
        private String language;
        private String audioFormat;
        private Boolean enablePunctuation;
    }

    /**
     * 语音识别响应DTO
     */
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class SpeechRecognitionResponse {
        private String recognizedText;
        private Double confidence;
        private Integer duration;
        private List<String> alternatives;
        private LocalDateTime processTime;
    }

    /**
     * 文本转语音请求DTO
     */
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class TextToSpeechRequest {
        @NotBlank(message = "文本内容不能为空")
        private String text;
        
        private String voice;
        private String language;
        private Double speed;
        private String outputFormat;
    }

    /**
     * 文本转语音响应DTO
     */
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class TextToSpeechResponse {
        private String audioFilePath;
        private String audioUrl;
        private Integer duration;
        private String format;
        private LocalDateTime generateTime;
    }

    /**
     * 图像识别请求DTO
     */
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class ImageRecognitionRequest {
        @NotBlank(message = "图像文件路径不能为空")
        private String imageFilePath;
        
        private String recognitionType; // TEXT, OBJECT, SCENE, HANDWRITING
        private String language;
    }

    /**
     * 图像识别响应DTO
     */
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class ImageRecognitionResponse {
        private String recognitionResult;
        private Double confidence;
        private List<String> detectedObjects;
        private String extractedText;
        private LocalDateTime processTime;
    }

    /**
     * 智能问答请求DTO
     */
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class IntelligentQARequest {
        @NotBlank(message = "问题不能为空")
        private String question;
        
        private Long courseId;
        private String context;
        private String questionType; // FACTUAL, CONCEPTUAL, PROCEDURAL
    }

    /**
     * 智能问答响应DTO
     */
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class IntelligentQAResponse {
        private String answer;
        private Double confidence;
        private List<String> relatedQuestions;
        private List<String> references;
        private LocalDateTime responseTime;
    }

    /**
     * 学习预测请求DTO
     */
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class LearningPredictionRequest {
        @NotNull(message = "学生ID不能为空")
        private Long studentId;
        
        private Long courseId;
        private String predictionType; // PERFORMANCE, DIFFICULTY, COMPLETION_TIME
        private String timeRange;
    }

    /**
     * 学习预测响应DTO
     */
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class LearningPredictionResponse {
        private String predictionResult;
        private Double accuracy;
        private String riskLevel;
        private List<String> recommendations;
        private LocalDateTime predictionTime;
    }

    /**
     * 抄袭检测请求DTO
     */
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class PlagiarismDetectionRequest {
        @NotBlank(message = "检测内容不能为空")
        private String content;
        
        private String contentType; // TEXT, CODE, DOCUMENT
        private Long courseId;
        private String language;
    }

    /**
     * 抄袭检测响应DTO
     */
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class PlagiarismDetectionResponse {
        private Double similarityScore;
        private List<String> suspiciousSources;
        private String detectionResult;
        private List<String> matchedSegments;
        private LocalDateTime detectionTime;
    }

    /**
     * 课程大纲响应DTO
     */
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class CourseOutlineResponse {
        private String outlineContent;
        private List<String> chapters;
        private List<String> learningObjectives;
        private String estimatedDuration;
        private LocalDateTime generateTime;
    }

    /**
     * 课程大纲请求DTO
     */
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class CourseOutlineRequest {
        @NotBlank(message = "课程名称不能为空")
        private String courseName;
        
        private String courseDescription;
        private String targetAudience;
        private String difficulty;
        private Integer duration;
        private List<String> keywords;
    }

    /**
     * AI使用统计响应DTO
     */
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class AIUsageStatisticsResponse {
        private Integer totalRequests;
        private Integer successfulRequests;
        private Integer failedRequests;
        private Double averageResponseTime;
        private String mostUsedFeature;
        private LocalDateTime statisticsTime;
    }

    /**
     * 教案响应DTO
     */
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class LessonPlanResponse {
        private String lessonTitle;
        private String lessonContent;
        private List<String> objectives;
        private List<String> activities;
        private String assessment;
        private String materials;
        private LocalDateTime generateTime;
    }

    /**
     * 教案请求DTO
     */
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class LessonPlanRequest {
        @NotBlank(message = "课程主题不能为空")
        private String topic;
        
        private String grade;
        private Integer duration;
        private String learningObjectives;
        private String teachingMethod;
        private List<String> resources;
    }

    /**
     * 内容优化响应DTO
     */
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class ContentOptimizationResponse {
        private String optimizedContent;
        private List<String> improvements;
        private String readabilityScore;
        private List<String> suggestions;
        private LocalDateTime optimizationTime;
    }

    /**
     * 内容优化请求DTO
     */
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class ContentOptimizationRequest {
        @NotBlank(message = "原始内容不能为空")
        private String originalContent;
        
        private String contentType;
        private String targetAudience;
        private String optimizationGoal;
        private String language;
    }

    /**
     * 学习路径响应DTO
     */
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class LearningPathResponse {
        private String pathName;
        private List<String> steps;
        private String estimatedTime;
        private String difficulty;
        private List<String> prerequisites;
        private LocalDateTime generateTime;
    }

    /**
     * 学习路径请求DTO
     */
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class LearningPathRequest {
        @NotNull(message = "学生ID不能为空")
        private Long studentId;
        
        private Long courseId;
        private String learningGoal;
        private String currentLevel;
        private String timeConstraint;
        private List<String> preferences;
    }

    /**
     * 能力分析响应DTO
     */
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class AbilityAnalysisResponse {
        private String overallAbility;
        private List<String> strengths;
        private List<String> weaknesses;
        private String improvementPlan;
        private Double confidenceScore;
        private LocalDateTime analysisTime;
    }

    /**
     * 能力分析请求DTO
     */
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class AbilityAnalysisRequest {
        @NotNull(message = "学生ID不能为空")
        private Long studentId;
        
        private Long courseId;
        private String analysisType;
        private String timeRange;
        private List<String> focusAreas;
    }

    /**
     * 个性化练习响应DTO
     */
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class PersonalizedExerciseResponse {
        private String exerciseTitle;
        private String exerciseContent;
        private String difficulty;
        private List<String> questions;
        private String expectedTime;
        private LocalDateTime generateTime;
    }

    /**
     * 个性化练习请求DTO
     */
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class PersonalizedExerciseRequest {
        @NotNull(message = "学生ID不能为空")
        private Long studentId;
        
        private Long courseId;
        private String topic;
        private String difficulty;
        private Integer questionCount;
        private String exerciseType;
    }

    /**
     * 教学效果响应DTO
     */
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class TeachingEffectivenessResponse {
        private String effectivenessScore;
        private List<String> strengths;
        private List<String> improvements;
        private String overallAssessment;
        private List<String> recommendations;
        private LocalDateTime evaluationTime;
    }

    /**
     * 教学效果请求DTO
     */
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class TeachingEffectivenessRequest {
        @NotNull(message = "教师ID不能为空")
        private Long teacherId;
        
        private Long courseId;
        private String evaluationPeriod;
        private List<String> metrics;
        private String evaluationType;
    }

    /**
     * 反馈报告响应DTO
     */
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class FeedbackReportResponse {
        private String reportTitle;
        private String reportContent;
        private List<String> keyFindings;
        private List<String> recommendations;
        private String summary;
        private LocalDateTime generateTime;
    }

    /**
     * 反馈报告请求DTO
     */
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class FeedbackReportRequest {
        @NotNull(message = "目标ID不能为空")
        private Long targetId;
        
        private String targetType; // STUDENT, COURSE, TEACHER
        private String reportType;
        private String timeRange;
        private List<String> includeMetrics;
    }

    /**
     * 智能排课响应
     */
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class ScheduleResponse {
        /**
         * 排课方案ID
         */
        private Long scheduleId;
        
        /**
         * 排课方案名称
         */
        private String scheduleName;
        
        /**
         * 课程安排列表
         */
        private List<String> courseSchedules;
        
        /**
         * 冲突检测结果
         */
        private List<String> conflicts;
        
        /**
         * 优化建议
         */
        private List<String> optimizationSuggestions;
        
        /**
         * 生成时间
         */
        private LocalDateTime generatedTime;
    }

    /**
     * 智能排课请求
     */
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class ScheduleRequest {
        /**
         * 课程列表
         */
        @NotNull(message = "课程列表不能为空")
        private List<Long> courseIds;
        
        /**
         * 教师列表
         */
        private List<Long> teacherIds;
        
        /**
         * 教室列表
         */
        private List<Long> classroomIds;
        
        /**
         * 时间约束
         */
        private String timeConstraint;
        
        /**
         * 排课偏好
         */
        private String preference;
        
        /**
         * 学期ID
         */
        @NotNull(message = "学期ID不能为空")
        private Long semesterId;
    }

    /**
     * AI历史记录响应
     */
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class AIHistoryResponse {
        /**
         * 记录ID
         */
        private Long historyId;
        
        /**
         * 功能类型
         */
        private String functionType;
        
        /**
         * 请求内容
         */
        private String requestContent;
        
        /**
         * 响应内容
         */
        private String responseContent;
        
        /**
         * 执行状态
         */
        private String executionStatus;
        
        /**
         * 执行时间
         */
        private LocalDateTime executionTime;
        
        /**
         * 耗时（毫秒）
         */
        private Long durationMs;
    }

    /**
     * 课堂互动分析响应
     */
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    public static class ClassroomInteractionResponse {
        /**
         * 分析ID
         */
        private Long analysisId;
        
        /**
         * 互动频率
         */
        private Double interactionFrequency;
        
        /**
         * 参与度分析
         */
        private String participationAnalysis;
        
        /**
         * 互动质量评分
         */
        private Double qualityScore;
        
        /**
         * 改进建议
         */
        private List<String> improvementSuggestions;
        
        /**
         * 分析时间
         */
        private LocalDateTime analysisTime;
    }

    /**
     * 课堂互动分析请求
     */
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    public static class ClassroomInteractionRequest {
        /**
         * 课程ID
         */
        @NotNull(message = "课程ID不能为空")
        private Long courseId;
        
        /**
         * 班级ID
         */
        @NotNull(message = "班级ID不能为空")
        private Long classId;
        
        /**
         * 分析时间范围
         */
        private String dateRange;
        
        /**
         * 互动数据
         */
        private List<String> interactionData;
        
        /**
         * 分析维度
         */
        private List<String> analysisDimensions;
    }

    /**
     * 考试分析响应
     */
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    public static class ExamAnalysisResponse {
        /**
         * 分析报告ID
         */
        private Long reportId;
        
        /**
         * 整体统计
         */
        private String overallStatistics;
        
        /**
         * 题目分析
         */
        private List<String> questionAnalyses;
        
        /**
         * 学生表现分析
         */
        private List<String> studentPerformances;
        
        /**
         * 改进建议
         */
        private List<String> improvementSuggestions;
        
        /**
         * 生成时间
         */
        private LocalDateTime generatedTime;
    }

    /**
     * 考试分析请求
     */
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    public static class ExamAnalysisRequest {
        /**
         * 考试ID
         */
        @NotNull(message = "考试ID不能为空")
        private Long examId;
        
        /**
         * 分析类型
         */
        @NotBlank(message = "分析类型不能为空")
        private String analysisType;
        
        /**
         * 分析维度
         */
        private List<String> analysisDimensions;
        
        /**
         * 是否包含学生详情
         */
        private Boolean includeStudentDetails;
        
        /**
         * 比较基准
         */
        private String comparisonBaseline;
    }

    /**
     * 教学资源推荐响应
     */
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    public static class TeachingResourceResponse {
        /**
         * 推荐ID
         */
        private Long recommendationId;
        
        /**
         * 推荐资源列表
         */
        private List<String> resources;
        
        /**
         * 推荐理由
         */
        private List<String> reasons;
        
        /**
         * 相关度评分
         */
        private Double relevanceScore;
        
        /**
         * 推荐时间
         */
        private LocalDateTime recommendedTime;
    }

    /**
     * 教学资源推荐请求
     */
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    public static class TeachingResourceRequest {
        /**
         * 课程ID
         */
        @NotNull(message = "课程ID不能为空")
        private Long courseId;
        
        /**
         * 教学主题
         */
        @NotBlank(message = "教学主题不能为空")
        private String topic;
        
        /**
         * 资源类型
         */
        private List<String> resourceTypes;
        
        /**
         * 难度级别
         */
        private String difficultyLevel;
        
        /**
         * 学生特征
         */
        private String studentCharacteristics;
    }

    /**
     * 学生画像响应
     */
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    public static class StudentProfileResponse {
        /**
         * 画像ID
         */
        private Long profileId;
        
        /**
         * 学生ID
         */
        private Long studentId;
        
        /**
         * 学习能力评估
         */
        private String abilityAssessment;
        
        /**
         * 学习偏好
         */
        private String learningPreference;
        
        /**
         * 知识掌握情况
         */
        private String knowledgeMastery;
        
        /**
         * 个性化建议
         */
        private List<String> personalizedSuggestions;
        
        /**
         * 生成时间
         */
        private LocalDateTime generatedTime;
    }

    /**
     * 学生画像请求
     */
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    public static class StudentProfileRequest {
        /**
         * 学生ID
         */
        @NotNull(message = "学生ID不能为空")
        private Long studentId;
        
        /**
         * 分析时间范围
         */
        private String dateRange;
        
        /**
         * 分析维度
         */
        private List<String> analysisDimensions;
        
        /**
         * 是否包含历史数据
         */
        private Boolean includeHistoricalData;
        
        /**
         * 课程范围
         */
        private List<Long> courseIds;
    }

    /**
     * 学习风险预测响应
     */
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    public static class LearningRiskResponse {
        /**
         * 预测ID
         */
        private Long predictionId;
        
        /**
         * 风险等级
         */
        private String riskLevel;
        
        /**
         * 风险评分
         */
        private Double riskScore;
        
        /**
         * 风险因素
         */
        private List<String> riskFactors;
        
        /**
         * 预警建议
         */
        private List<String> warningAdvice;
        
        /**
         * 干预措施
         */
        private List<String> interventions;
        
        /**
         * 预测时间
         */
        private LocalDateTime predictedTime;
    }

    /**
     * 学习风险预测请求
     */
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    public static class LearningRiskRequest {
        /**
         * 学生ID
         */
        @NotNull(message = "学生ID不能为空")
        private Long studentId;
        
        /**
         * 预测时间范围
         */
        private Integer predictionDays;
        
        /**
         * 考虑因素
         */
        private List<String> considerationFactors;
        
        /**
         * 历史数据范围
         */
        private String historicalRange;
        
        /**
         * 课程范围
         */
        private List<Long> courseIds;
    }

    /**
     * 教学策略优化响应
     */
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    public static class TeachingStrategyResponse {
        /**
         * 策略ID
         */
        private Long strategyId;
        
        /**
         * 优化策略
         */
        private List<String> strategies;
        
        /**
         * 预期效果
         */
        private String expectedOutcome;
        
        /**
         * 实施建议
         */
        private List<String> implementationAdvice;
        
        /**
         * 评估指标
         */
        private List<String> evaluationMetrics;
        
        /**
         * 生成时间
         */
        private LocalDateTime generatedTime;
    }

    /**
     * 教学策略优化请求
     */
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    public static class TeachingStrategyRequest {
        /**
         * 课程ID
         */
        @NotNull(message = "课程ID不能为空")
        private Long courseId;
        
        /**
         * 班级ID
         */
        @NotNull(message = "班级ID不能为空")
        private Long classId;
        
        /**
         * 当前策略
         */
        private String currentStrategy;
        
        /**
         * 优化目标
         */
        private List<String> optimizationGoals;
        
        /**
         * 学生特征数据
         */
        private String classCharacteristics;
        
        /**
         * 约束条件
         */
        private List<String> constraints;
    }

    /**
     * 多媒体内容生成响应
     */
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    public static class MultimediaContentResponse {
        /**
         * 内容ID
         */
        private Long contentId;
        
        /**
         * 生成的内容
         */
        private List<String> contents;
        
        /**
         * 内容质量评分
         */
        private Double qualityScore;
        
        /**
         * 使用建议
         */
        private List<String> usageAdvice;
        
        /**
         * 生成时间
         */
        private LocalDateTime generatedTime;
    }

    /**
     * 多媒体内容生成请求
     */
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    public static class MultimediaContentRequest {
        /**
         * 内容主题
         */
        @NotBlank(message = "内容主题不能为空")
        private String topic;
        
        /**
         * 内容类型
         */
        @NotEmpty(message = "内容类型不能为空")
        private List<String> contentTypes;
        
        /**
         * 目标受众
         */
        private String targetAudience;
        
        /**
         * 难度级别
         */
        private String difficultyLevel;
        
        /**
         * 时长要求
         */
        private Integer durationMinutes;
        
        /**
         * 特殊要求
         */
        private List<String> specialRequirements;
    }

    /**
     * 聊天机器人响应
     */
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    public static class ChatbotResponse {
        /**
         * 会话ID
         */
        private String sessionId;
        
        /**
         * 回复内容
         */
        private String reply;
        
        /**
         * 置信度
         */
        private Double confidence;
        
        /**
         * 相关建议
         */
        private List<String> suggestions;
        
        /**
         * 回复时间
         */
        private LocalDateTime replyTime;
    }

    /**
     * 聊天机器人请求
     */
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    public static class ChatbotRequest {
        /**
         * 会话ID
         */
        private String sessionId;
        
        /**
         * 用户消息
         */
        @NotBlank(message = "用户消息不能为空")
        private String message;
        
        /**
         * 上下文信息
         */
        private String context;
        
        /**
         * 消息类型
         */
        private String messageType;
    }

    /**
     * AI模型配置响应
     */
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    public static class AIModelConfigResponse {
        /**
         * 配置ID
         */
        private Long configId;
        
        /**
         * 模型配置
         */
        private String modelConfig;
        
        /**
         * 功能开关
         */
        private Map<String, Boolean> featureSwitches;
        
        /**
         * 参数设置
         */
        private Map<String, Object> parameters;
        
        /**
         * 更新时间
         */
        private LocalDateTime updatedTime;
    }

    /**
     * AI模型配置请求
     */
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    public static class AIModelConfigRequest {
        /**
         * 模型类型
         */
        @NotBlank(message = "模型类型不能为空")
        private String modelType;
        
        /**
         * 功能开关
         */
        private Map<String, Boolean> featureSwitches;
        
        /**
         * 参数设置
         */
        private Map<String, Object> parameters;
        
        /**
         * 配置描述
         */
        private String description;
    }

    /**
     * 模型训练响应
     */
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    public static class ModelTrainingResponse {
        /**
         * 训练ID
         */
        private Long trainingId;
        
        /**
         * 训练状态
         */
        private String status;
        
        /**
         * 训练进度
         */
        private Double progress;
        
        /**
         * 预计完成时间
         */
        private LocalDateTime estimatedCompletionTime;
        
        /**
         * 训练指标
         */
        private String metrics;
    }

    /**
     * 模型训练请求
     */
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    public static class ModelTrainingRequest {
        /**
         * 模型名称
         */
        @NotBlank(message = "模型名称不能为空")
        private String modelName;
        
        /**
         * 训练数据集
         */
        @NotEmpty(message = "训练数据集不能为空")
        private List<String> trainingDatasets;
        
        /**
         * 训练参数
         */
        private String trainingParams;
        
        /**
         * 验证数据集
         */
        private List<String> validationDatasets;
    }

    /**
     * 训练状态响应
     */
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    public static class TrainingStatusResponse {
        /**
         * 训练ID
         */
        private Long trainingId;
        
        /**
         * 当前状态
         */
        private String currentStatus;
        
        /**
         * 训练进度
         */
        private Double progress;
        
        /**
         * 当前轮次
         */
        private Integer currentEpoch;
        
        /**
         * 总轮次
         */
        private Integer totalEpochs;
        
        /**
         * 实时指标
         */
        private Map<String, Double> realtimeMetrics;
        
        /**
         * 状态更新时间
         */
        private LocalDateTime statusUpdateTime;
    }

    /**
     * 模型部署响应
     */
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    public static class ModelDeploymentResponse {
        /**
         * 部署ID
         */
        private Long deploymentId;
        
        /**
         * 部署状态
         */
        private String deploymentStatus;
        
        /**
         * 服务端点
         */
        private String serviceEndpoint;
        
        /**
         * 部署配置
         */
        private String deploymentConfig;
        
        /**
         * 部署时间
         */
        private LocalDateTime deploymentTime;
    }

    /**
     * 模型部署请求
     */
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    public static class ModelDeploymentRequest {
        /**
         * 模型ID
         */
        @NotNull(message = "模型ID不能为空")
        private Long modelId;
        
        /**
         * 部署环境
         */
        @NotBlank(message = "部署环境不能为空")
        private String environment;
        
        /**
         * 资源配置
         */
        private String resourceConfig;
        
        /**
         * 部署策略
         */
        private String deploymentStrategy;
    }
}