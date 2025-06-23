package com.education.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

/**
 * AI相关DTO扩展类
 * 包含更多AI功能相关的DTO定义
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
public class AIDTOExtension {

    /**
     * 题目生成请求DTO
     */
    public static class QuestionGenerateRequest {
        @NotNull(message = "课程ID不能为空")
        private Long courseId;
        
        @NotBlank(message = "知识点不能为空")
        private String knowledgePoint;
        
        @NotBlank(message = "题目类型不能为空")
        private String questionType; // CHOICE, FILL_BLANK, SHORT_ANSWER, ESSAY
        
        private Integer questionCount;
        private String difficulty; // EASY, MEDIUM, HARD
        private String language;
        
        // Getters and Setters
        public Long getCourseId() { return courseId; }
        public void setCourseId(Long courseId) { this.courseId = courseId; }
        public String getKnowledgePoint() { return knowledgePoint; }
        public void setKnowledgePoint(String knowledgePoint) { this.knowledgePoint = knowledgePoint; }
        public String getQuestionType() { return questionType; }
        public void setQuestionType(String questionType) { this.questionType = questionType; }
        public Integer getQuestionCount() { return questionCount; }
        public void setQuestionCount(Integer questionCount) { this.questionCount = questionCount; }
        public String getDifficulty() { return difficulty; }
        public void setDifficulty(String difficulty) { this.difficulty = difficulty; }
        public String getLanguage() { return language; }
        public void setLanguage(String language) { this.language = language; }
    }

    /**
     * 题目生成响应DTO
     */
    public static class QuestionGenerateResponse {
        private List<GeneratedQuestion> questions;
        private String generationId;
        private LocalDateTime generateTime;
        private String status;
        
        // Getters and Setters
        public List<GeneratedQuestion> getQuestions() { return questions; }
        public void setQuestions(List<GeneratedQuestion> questions) { this.questions = questions; }
        public String getGenerationId() { return generationId; }
        public void setGenerationId(String generationId) { this.generationId = generationId; }
        public LocalDateTime getGenerateTime() { return generateTime; }
        public void setGenerateTime(LocalDateTime generateTime) { this.generateTime = generateTime; }
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
        
        public static class GeneratedQuestion {
            private String questionText;
            private String questionType;
            private List<String> options;
            private String correctAnswer;
            private String explanation;
            private String difficulty;
            
            // Getters and Setters
            public String getQuestionText() { return questionText; }
            public void setQuestionText(String questionText) { this.questionText = questionText; }
            public String getQuestionType() { return questionType; }
            public void setQuestionType(String questionType) { this.questionType = questionType; }
            public List<String> getOptions() { return options; }
            public void setOptions(List<String> options) { this.options = options; }
            public String getCorrectAnswer() { return correctAnswer; }
            public void setCorrectAnswer(String correctAnswer) { this.correctAnswer = correctAnswer; }
            public String getExplanation() { return explanation; }
            public void setExplanation(String explanation) { this.explanation = explanation; }
            public String getDifficulty() { return difficulty; }
            public void setDifficulty(String difficulty) { this.difficulty = difficulty; }
        }
    }

    /**
     * 自动评分请求DTO
     */
    public static class AutoGradeRequest {
        @NotNull(message = "任务ID不能为空")
        private Long taskId;
        
        @NotNull(message = "学生ID不能为空")
        private Long studentId;
        
        @NotBlank(message = "学生答案不能为空")
        private String studentAnswer;
        
        private String gradingCriteria;
        private Boolean enableDetailedFeedback;
        
        // Getters and Setters
        public Long getTaskId() { return taskId; }
        public void setTaskId(Long taskId) { this.taskId = taskId; }
        public Long getStudentId() { return studentId; }
        public void setStudentId(Long studentId) { this.studentId = studentId; }
        public String getStudentAnswer() { return studentAnswer; }
        public void setStudentAnswer(String studentAnswer) { this.studentAnswer = studentAnswer; }
        public String getGradingCriteria() { return gradingCriteria; }
        public void setGradingCriteria(String gradingCriteria) { this.gradingCriteria = gradingCriteria; }
        public Boolean getEnableDetailedFeedback() { return enableDetailedFeedback; }
        public void setEnableDetailedFeedback(Boolean enableDetailedFeedback) { this.enableDetailedFeedback = enableDetailedFeedback; }
    }

    /**
     * 自动评分响应DTO
     */
    public static class AutoGradeResponse {
        private Double score;
        private String grade;
        private String feedback;
        private List<CriteriaScore> criteriaScores;
        private String confidence;
        private LocalDateTime gradeTime;
        
        // Getters and Setters
        public Double getScore() { return score; }
        public void setScore(Double score) { this.score = score; }
        public String getGrade() { return grade; }
        public void setGrade(String grade) { this.grade = grade; }
        public String getFeedback() { return feedback; }
        public void setFeedback(String feedback) { this.feedback = feedback; }
        public List<CriteriaScore> getCriteriaScores() { return criteriaScores; }
        public void setCriteriaScores(List<CriteriaScore> criteriaScores) { this.criteriaScores = criteriaScores; }
        public String getConfidence() { return confidence; }
        public void setConfidence(String confidence) { this.confidence = confidence; }
        public LocalDateTime getGradeTime() { return gradeTime; }
        public void setGradeTime(LocalDateTime gradeTime) { this.gradeTime = gradeTime; }
        
        public static class CriteriaScore {
            private String criteriaName;
            private Double score;
            private String feedback;
            
            // Getters and Setters
            public String getCriteriaName() { return criteriaName; }
            public void setCriteriaName(String criteriaName) { this.criteriaName = criteriaName; }
            public Double getScore() { return score; }
            public void setScore(Double score) { this.score = score; }
            public String getFeedback() { return feedback; }
            public void setFeedback(String feedback) { this.feedback = feedback; }
        }
    }

    /**
     * 教学建议请求DTO
     */
    public static class TeachingSuggestionRequest {
        @NotNull(message = "课程ID不能为空")
        private Long courseId;
        
        private Long classId;
        private String teachingGoal;
        private String currentProgress;
        private List<String> studentDifficulties;
        private String timeConstraint;
        
        // Getters and Setters
        public Long getCourseId() { return courseId; }
        public void setCourseId(Long courseId) { this.courseId = courseId; }
        public Long getClassId() { return classId; }
        public void setClassId(Long classId) { this.classId = classId; }
        public String getTeachingGoal() { return teachingGoal; }
        public void setTeachingGoal(String teachingGoal) { this.teachingGoal = teachingGoal; }
        public String getCurrentProgress() { return currentProgress; }
        public void setCurrentProgress(String currentProgress) { this.currentProgress = currentProgress; }
        public List<String> getStudentDifficulties() { return studentDifficulties; }
        public void setStudentDifficulties(List<String> studentDifficulties) { this.studentDifficulties = studentDifficulties; }
        public String getTimeConstraint() { return timeConstraint; }
        public void setTimeConstraint(String timeConstraint) { this.timeConstraint = timeConstraint; }
    }

    /**
     * 教学建议响应DTO
     */
    public static class TeachingSuggestionResponse {
        private List<String> teachingMethods;
        private List<String> activities;
        private List<String> resources;
        private String timeAllocation;
        private List<String> assessmentSuggestions;
        private String rationale;
        private LocalDateTime generateTime;
        
        // Getters and Setters
        public List<String> getTeachingMethods() { return teachingMethods; }
        public void setTeachingMethods(List<String> teachingMethods) { this.teachingMethods = teachingMethods; }
        public List<String> getActivities() { return activities; }
        public void setActivities(List<String> activities) { this.activities = activities; }
        public List<String> getResources() { return resources; }
        public void setResources(List<String> resources) { this.resources = resources; }
        public String getTimeAllocation() { return timeAllocation; }
        public void setTimeAllocation(String timeAllocation) { this.timeAllocation = timeAllocation; }
        public List<String> getAssessmentSuggestions() { return assessmentSuggestions; }
        public void setAssessmentSuggestions(List<String> assessmentSuggestions) { this.assessmentSuggestions = assessmentSuggestions; }
        public String getRationale() { return rationale; }
        public void setRationale(String rationale) { this.rationale = rationale; }
        public LocalDateTime getGenerateTime() { return generateTime; }
        public void setGenerateTime(LocalDateTime generateTime) { this.generateTime = generateTime; }
    }

    /**
     * 学习行为分析请求DTO
     */
    public static class LearningBehaviorAnalysisRequest {
        @NotNull(message = "学生ID不能为空")
        private Long studentId;
        
        private Long courseId;
        private String timeRange;
        private List<String> behaviorTypes; // LOGIN, STUDY_TIME, TASK_COMPLETION, INTERACTION
        
        // Getters and Setters
        public Long getStudentId() { return studentId; }
        public void setStudentId(Long studentId) { this.studentId = studentId; }
        public Long getCourseId() { return courseId; }
        public void setCourseId(Long courseId) { this.courseId = courseId; }
        public String getTimeRange() { return timeRange; }
        public void setTimeRange(String timeRange) { this.timeRange = timeRange; }
        public List<String> getBehaviorTypes() { return behaviorTypes; }
        public void setBehaviorTypes(List<String> behaviorTypes) { this.behaviorTypes = behaviorTypes; }
    }

    /**
     * 学习行为分析响应DTO
     */
    public static class LearningBehaviorAnalysisResponse {
        private String studentName;
        private Map<String, Object> behaviorMetrics;
        private List<String> patterns;
        private List<String> insights;
        private List<String> recommendations;
        private String riskLevel; // LOW, MEDIUM, HIGH
        private LocalDateTime analysisTime;
        
        // Getters and Setters
        public String getStudentName() { return studentName; }
        public void setStudentName(String studentName) { this.studentName = studentName; }
        public Map<String, Object> getBehaviorMetrics() { return behaviorMetrics; }
        public void setBehaviorMetrics(Map<String, Object> behaviorMetrics) { this.behaviorMetrics = behaviorMetrics; }
        public List<String> getPatterns() { return patterns; }
        public void setPatterns(List<String> patterns) { this.patterns = patterns; }
        public List<String> getInsights() { return insights; }
        public void setInsights(List<String> insights) { this.insights = insights; }
        public List<String> getRecommendations() { return recommendations; }
        public void setRecommendations(List<String> recommendations) { this.recommendations = recommendations; }
        public String getRiskLevel() { return riskLevel; }
        public void setRiskLevel(String riskLevel) { this.riskLevel = riskLevel; }
        public LocalDateTime getAnalysisTime() { return analysisTime; }
        public void setAnalysisTime(LocalDateTime analysisTime) { this.analysisTime = analysisTime; }
    }

    /**
     * 内容推荐请求DTO
     */
    public static class ContentRecommendationRequest {
        @NotNull(message = "学生ID不能为空")
        private Long studentId;
        
        private Long courseId;
        private String learningGoal;
        private String currentLevel;
        private List<String> interests;
        private String contentType; // VIDEO, ARTICLE, EXERCISE, BOOK
        
        // Getters and Setters
        public Long getStudentId() { return studentId; }
        public void setStudentId(Long studentId) { this.studentId = studentId; }
        public Long getCourseId() { return courseId; }
        public void setCourseId(Long courseId) { this.courseId = courseId; }
        public String getLearningGoal() { return learningGoal; }
        public void setLearningGoal(String learningGoal) { this.learningGoal = learningGoal; }
        public String getCurrentLevel() { return currentLevel; }
        public void setCurrentLevel(String currentLevel) { this.currentLevel = currentLevel; }
        public List<String> getInterests() { return interests; }
        public void setInterests(List<String> interests) { this.interests = interests; }
        public String getContentType() { return contentType; }
        public void setContentType(String contentType) { this.contentType = contentType; }
    }

    /**
     * 内容推荐响应DTO
     */
    public static class ContentRecommendationResponse {
        private List<RecommendedContent> recommendations;
        private String recommendationReason;
        private LocalDateTime recommendTime;
        
        // Getters and Setters
        public List<RecommendedContent> getRecommendations() { return recommendations; }
        public void setRecommendations(List<RecommendedContent> recommendations) { this.recommendations = recommendations; }
        public String getRecommendationReason() { return recommendationReason; }
        public void setRecommendationReason(String recommendationReason) { this.recommendationReason = recommendationReason; }
        public LocalDateTime getRecommendTime() { return recommendTime; }
        public void setRecommendTime(LocalDateTime recommendTime) { this.recommendTime = recommendTime; }
        
        public static class RecommendedContent {
            private String title;
            private String description;
            private String contentType;
            private String url;
            private String difficulty;
            private Integer estimatedTime;
            private Double relevanceScore;
            
            // Getters and Setters
            public String getTitle() { return title; }
            public void setTitle(String title) { this.title = title; }
            public String getDescription() { return description; }
            public void setDescription(String description) { this.description = description; }
            public String getContentType() { return contentType; }
            public void setContentType(String contentType) { this.contentType = contentType; }
            public String getUrl() { return url; }
            public void setUrl(String url) { this.url = url; }
            public String getDifficulty() { return difficulty; }
            public void setDifficulty(String difficulty) { this.difficulty = difficulty; }
            public Integer getEstimatedTime() { return estimatedTime; }
            public void setEstimatedTime(Integer estimatedTime) { this.estimatedTime = estimatedTime; }
            public Double getRelevanceScore() { return relevanceScore; }
            public void setRelevanceScore(Double relevanceScore) { this.relevanceScore = relevanceScore; }
        }
    }

    /**
     * 语音识别请求DTO
     */
    public static class SpeechRecognitionRequest {
        @NotBlank(message = "音频文件路径不能为空")
        private String audioFilePath;
        
        private String language;
        private String audioFormat;
        private Boolean enablePunctuation;
        
        // Getters and Setters
        public String getAudioFilePath() { return audioFilePath; }
        public void setAudioFilePath(String audioFilePath) { this.audioFilePath = audioFilePath; }
        public String getLanguage() { return language; }
        public void setLanguage(String language) { this.language = language; }
        public String getAudioFormat() { return audioFormat; }
        public void setAudioFormat(String audioFormat) { this.audioFormat = audioFormat; }
        public Boolean getEnablePunctuation() { return enablePunctuation; }
        public void setEnablePunctuation(Boolean enablePunctuation) { this.enablePunctuation = enablePunctuation; }
    }

    /**
     * 语音识别响应DTO
     */
    public static class SpeechRecognitionResponse {
        private String recognizedText;
        private Double confidence;
        private Integer duration;
        private List<String> alternatives;
        private LocalDateTime processTime;
        
        // Getters and Setters
        public String getRecognizedText() { return recognizedText; }
        public void setRecognizedText(String recognizedText) { this.recognizedText = recognizedText; }
        public Double getConfidence() { return confidence; }
        public void setConfidence(Double confidence) { this.confidence = confidence; }
        public Integer getDuration() { return duration; }
        public void setDuration(Integer duration) { this.duration = duration; }
        public List<String> getAlternatives() { return alternatives; }
        public void setAlternatives(List<String> alternatives) { this.alternatives = alternatives; }
        public LocalDateTime getProcessTime() { return processTime; }
        public void setProcessTime(LocalDateTime processTime) { this.processTime = processTime; }
    }

    /**
     * 文本转语音请求DTO
     */
    public static class TextToSpeechRequest {
        @NotBlank(message = "文本内容不能为空")
        private String text;
        
        private String voice;
        private String language;
        private Double speed;
        private String outputFormat;
        
        // Getters and Setters
        public String getText() { return text; }
        public void setText(String text) { this.text = text; }
        public String getVoice() { return voice; }
        public void setVoice(String voice) { this.voice = voice; }
        public String getLanguage() { return language; }
        public void setLanguage(String language) { this.language = language; }
        public Double getSpeed() { return speed; }
        public void setSpeed(Double speed) { this.speed = speed; }
        public String getOutputFormat() { return outputFormat; }
        public void setOutputFormat(String outputFormat) { this.outputFormat = outputFormat; }
    }

    /**
     * 文本转语音响应DTO
     */
    public static class TextToSpeechResponse {
        private String audioFilePath;
        private String audioUrl;
        private Integer duration;
        private String format;
        private LocalDateTime generateTime;
        
        // Getters and Setters
        public String getAudioFilePath() { return audioFilePath; }
        public void setAudioFilePath(String audioFilePath) { this.audioFilePath = audioFilePath; }
        public String getAudioUrl() { return audioUrl; }
        public void setAudioUrl(String audioUrl) { this.audioUrl = audioUrl; }
        public Integer getDuration() { return duration; }
        public void setDuration(Integer duration) { this.duration = duration; }
        public String getFormat() { return format; }
        public void setFormat(String format) { this.format = format; }
        public LocalDateTime getGenerateTime() { return generateTime; }
        public void setGenerateTime(LocalDateTime generateTime) { this.generateTime = generateTime; }
    }

    /**
     * 图像识别请求DTO
     */
    public static class ImageRecognitionRequest {
        @NotBlank(message = "图像文件路径不能为空")
        private String imageFilePath;
        
        private String recognitionType; // TEXT, OBJECT, SCENE, HANDWRITING
        private String language;
        
        // Getters and Setters
        public String getImageFilePath() { return imageFilePath; }
        public void setImageFilePath(String imageFilePath) { this.imageFilePath = imageFilePath; }
        public String getRecognitionType() { return recognitionType; }
        public void setRecognitionType(String recognitionType) { this.recognitionType = recognitionType; }
        public String getLanguage() { return language; }
        public void setLanguage(String language) { this.language = language; }
    }

    /**
     * 图像识别响应DTO
     */
    public static class ImageRecognitionResponse {
        private String recognizedText;
        private List<String> detectedObjects;
        private String sceneDescription;
        private Double confidence;
        private LocalDateTime processTime;
        
        // Getters and Setters
        public String getRecognizedText() { return recognizedText; }
        public void setRecognizedText(String recognizedText) { this.recognizedText = recognizedText; }
        public List<String> getDetectedObjects() { return detectedObjects; }
        public void setDetectedObjects(List<String> detectedObjects) { this.detectedObjects = detectedObjects; }
        public String getSceneDescription() { return sceneDescription; }
        public void setSceneDescription(String sceneDescription) { this.sceneDescription = sceneDescription; }
        public Double getConfidence() { return confidence; }
        public void setConfidence(Double confidence) { this.confidence = confidence; }
        public LocalDateTime getProcessTime() { return processTime; }
        public void setProcessTime(LocalDateTime processTime) { this.processTime = processTime; }
    }

    /**
     * 智能问答请求DTO
     */
    public static class IntelligentQARequest {
        @NotBlank(message = "问题不能为空")
        private String question;
        
        private Long courseId;
        private String context;
        private String questionType; // FACTUAL, CONCEPTUAL, PROCEDURAL
        
        // Getters and Setters
        public String getQuestion() { return question; }
        public void setQuestion(String question) { this.question = question; }
        public Long getCourseId() { return courseId; }
        public void setCourseId(Long courseId) { this.courseId = courseId; }
        public String getContext() { return context; }
        public void setContext(String context) { this.context = context; }
        public String getQuestionType() { return questionType; }
        public void setQuestionType(String questionType) { this.questionType = questionType; }
    }

    /**
     * 智能问答响应DTO
     */
    public static class IntelligentQAResponse {
        private String answer;
        private List<String> relatedQuestions;
        private List<String> references;
        private Double confidence;
        private LocalDateTime responseTime;
        
        // Getters and Setters
        public String getAnswer() { return answer; }
        public void setAnswer(String answer) { this.answer = answer; }
        public List<String> getRelatedQuestions() { return relatedQuestions; }
        public void setRelatedQuestions(List<String> relatedQuestions) { this.relatedQuestions = relatedQuestions; }
        public List<String> getReferences() { return references; }
        public void setReferences(List<String> references) { this.references = references; }
        public Double getConfidence() { return confidence; }
        public void setConfidence(Double confidence) { this.confidence = confidence; }
        public LocalDateTime getResponseTime() { return responseTime; }
        public void setResponseTime(LocalDateTime responseTime) { this.responseTime = responseTime; }
    }
}