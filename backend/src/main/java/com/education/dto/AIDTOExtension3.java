package com.education.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

/**
 * AI相关DTO扩展类3
 * 包含能力分析、个性化练习等AI功能相关的DTO定义
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
public class AIDTOExtension3 {

    /**
     * 能力分析请求DTO
     */
    public static class AbilityAnalysisRequest {
        @NotNull(message = "学生ID不能为空")
        private Long studentId;
        
        @NotNull(message = "课程ID不能为空")
        private Long courseId;
        
        private String analysisType; // COMPREHENSIVE, SUBJECT_SPECIFIC, SKILL_BASED
        private String timeRange; // WEEK, MONTH, SEMESTER
        
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
     * 能力分析响应DTO
     */
    public static class AbilityAnalysisResponse {
        private String overallLevel;
        private Map<String, AbilityScore> abilityScores;
        private List<String> strengths;
        private List<String> weaknesses;
        private List<String> recommendations;
        private String learningStyle;
        private LocalDateTime analysisTime;
        
        // Getters and Setters
        public String getOverallLevel() { return overallLevel; }
        public void setOverallLevel(String overallLevel) { this.overallLevel = overallLevel; }
        public Map<String, AbilityScore> getAbilityScores() { return abilityScores; }
        public void setAbilityScores(Map<String, AbilityScore> abilityScores) { this.abilityScores = abilityScores; }
        public List<String> getStrengths() { return strengths; }
        public void setStrengths(List<String> strengths) { this.strengths = strengths; }
        public List<String> getWeaknesses() { return weaknesses; }
        public void setWeaknesses(List<String> weaknesses) { this.weaknesses = weaknesses; }
        public List<String> getRecommendations() { return recommendations; }
        public void setRecommendations(List<String> recommendations) { this.recommendations = recommendations; }
        public String getLearningStyle() { return learningStyle; }
        public void setLearningStyle(String learningStyle) { this.learningStyle = learningStyle; }
        public LocalDateTime getAnalysisTime() { return analysisTime; }
        public void setAnalysisTime(LocalDateTime analysisTime) { this.analysisTime = analysisTime; }
        
        public static class AbilityScore {
            private Double score;
            private String level;
            private String description;
            
            // Getters and Setters
            public Double getScore() { return score; }
            public void setScore(Double score) { this.score = score; }
            public String getLevel() { return level; }
            public void setLevel(String level) { this.level = level; }
            public String getDescription() { return description; }
            public void setDescription(String description) { this.description = description; }
        }
    }

    /**
     * 个性化练习请求DTO
     */
    public static class PersonalizedExerciseRequest {
        @NotNull(message = "学生ID不能为空")
        private Long studentId;
        
        @NotNull(message = "课程ID不能为空")
        private Long courseId;
        
        private String difficulty; // EASY, MEDIUM, HARD
        private String exerciseType; // CHOICE, FILL_BLANK, ESSAY, CODING
        private Integer quantity;
        private List<String> focusAreas;
        
        // Getters and Setters
        public Long getStudentId() { return studentId; }
        public void setStudentId(Long studentId) { this.studentId = studentId; }
        public Long getCourseId() { return courseId; }
        public void setCourseId(Long courseId) { this.courseId = courseId; }
        public String getDifficulty() { return difficulty; }
        public void setDifficulty(String difficulty) { this.difficulty = difficulty; }
        public String getExerciseType() { return exerciseType; }
        public void setExerciseType(String exerciseType) { this.exerciseType = exerciseType; }
        public Integer getQuantity() { return quantity; }
        public void setQuantity(Integer quantity) { this.quantity = quantity; }
        public List<String> getFocusAreas() { return focusAreas; }
        public void setFocusAreas(List<String> focusAreas) { this.focusAreas = focusAreas; }
    }

    /**
     * 个性化练习响应DTO
     */
    public static class PersonalizedExerciseResponse {
        private List<Exercise> exercises;
        private String difficulty;
        private String explanation;
        private Integer estimatedTime;
        private LocalDateTime generateTime;
        
        // Getters and Setters
        public List<Exercise> getExercises() { return exercises; }
        public void setExercises(List<Exercise> exercises) { this.exercises = exercises; }
        public String getDifficulty() { return difficulty; }
        public void setDifficulty(String difficulty) { this.difficulty = difficulty; }
        public String getExplanation() { return explanation; }
        public void setExplanation(String explanation) { this.explanation = explanation; }
        public Integer getEstimatedTime() { return estimatedTime; }
        public void setEstimatedTime(Integer estimatedTime) { this.estimatedTime = estimatedTime; }
        public LocalDateTime getGenerateTime() { return generateTime; }
        public void setGenerateTime(LocalDateTime generateTime) { this.generateTime = generateTime; }
        
        public static class Exercise {
            private String question;
            private List<String> options;
            private String correctAnswer;
            private String explanation;
            private String type;
            private String knowledgePoint;
            
            // Getters and Setters
            public String getQuestion() { return question; }
            public void setQuestion(String question) { this.question = question; }
            public List<String> getOptions() { return options; }
            public void setOptions(List<String> options) { this.options = options; }
            public String getCorrectAnswer() { return correctAnswer; }
            public void setCorrectAnswer(String correctAnswer) { this.correctAnswer = correctAnswer; }
            public String getExplanation() { return explanation; }
            public void setExplanation(String explanation) { this.explanation = explanation; }
            public String getType() { return type; }
            public void setType(String type) { this.type = type; }
            public String getKnowledgePoint() { return knowledgePoint; }
            public void setKnowledgePoint(String knowledgePoint) { this.knowledgePoint = knowledgePoint; }
        }
    }

    /**
     * 教学效果请求DTO
     */
    public static class TeachingEffectivenessRequest {
        @NotNull(message = "教师ID不能为空")
        private Long teacherId;
        
        @NotNull(message = "课程ID不能为空")
        private Long courseId;
        
        private String timeRange; // WEEK, MONTH, SEMESTER
        private String analysisType; // STUDENT_PERFORMANCE, ENGAGEMENT, COMPLETION
        
        // Getters and Setters
        public Long getTeacherId() { return teacherId; }
        public void setTeacherId(Long teacherId) { this.teacherId = teacherId; }
        public Long getCourseId() { return courseId; }
        public void setCourseId(Long courseId) { this.courseId = courseId; }
        public String getTimeRange() { return timeRange; }
        public void setTimeRange(String timeRange) { this.timeRange = timeRange; }
        public String getAnalysisType() { return analysisType; }
        public void setAnalysisType(String analysisType) { this.analysisType = analysisType; }
    }

    /**
     * 教学效果响应DTO
     */
    public static class TeachingEffectivenessResponse {
        private Double overallEffectiveness;
        private Double studentSatisfaction;
        private Double learningOutcomeAchievement;
        private Double engagementLevel;
        private List<String> strengths;
        private List<String> improvementAreas;
        private List<String> recommendations;
        private Map<String, Double> metrics;
        private LocalDateTime analysisTime;
        
        // Getters and Setters
        public Double getOverallEffectiveness() { return overallEffectiveness; }
        public void setOverallEffectiveness(Double overallEffectiveness) { this.overallEffectiveness = overallEffectiveness; }
        public Double getStudentSatisfaction() { return studentSatisfaction; }
        public void setStudentSatisfaction(Double studentSatisfaction) { this.studentSatisfaction = studentSatisfaction; }
        public Double getLearningOutcomeAchievement() { return learningOutcomeAchievement; }
        public void setLearningOutcomeAchievement(Double learningOutcomeAchievement) { this.learningOutcomeAchievement = learningOutcomeAchievement; }
        public Double getEngagementLevel() { return engagementLevel; }
        public void setEngagementLevel(Double engagementLevel) { this.engagementLevel = engagementLevel; }
        public List<String> getStrengths() { return strengths; }
        public void setStrengths(List<String> strengths) { this.strengths = strengths; }
        public List<String> getImprovementAreas() { return improvementAreas; }
        public void setImprovementAreas(List<String> improvementAreas) { this.improvementAreas = improvementAreas; }
        public List<String> getRecommendations() { return recommendations; }
        public void setRecommendations(List<String> recommendations) { this.recommendations = recommendations; }
        public Map<String, Double> getMetrics() { return metrics; }
        public void setMetrics(Map<String, Double> metrics) { this.metrics = metrics; }
        public LocalDateTime getAnalysisTime() { return analysisTime; }
        public void setAnalysisTime(LocalDateTime analysisTime) { this.analysisTime = analysisTime; }
    }

    /**
     * 反馈报告请求DTO
     */
    public static class FeedbackReportRequest {
        @NotNull(message = "课程ID不能为空")
        private Long courseId;
        
        private String reportType; // STUDENT_FEEDBACK, TEACHER_FEEDBACK, COMPREHENSIVE
        private String timeRange; // WEEK, MONTH, SEMESTER
        private List<Long> studentIds;
        
        // Getters and Setters
        public Long getCourseId() { return courseId; }
        public void setCourseId(Long courseId) { this.courseId = courseId; }
        public String getReportType() { return reportType; }
        public void setReportType(String reportType) { this.reportType = reportType; }
        public String getTimeRange() { return timeRange; }
        public void setTimeRange(String timeRange) { this.timeRange = timeRange; }
        public List<Long> getStudentIds() { return studentIds; }
        public void setStudentIds(List<Long> studentIds) { this.studentIds = studentIds; }
    }

    /**
     * 反馈报告响应DTO
     */
    public static class FeedbackReportResponse {
        private String reportTitle;
        private String summary;
        private List<FeedbackItem> feedbackItems;
        private Map<String, Double> sentimentAnalysis;
        private List<String> keyInsights;
        private List<String> actionItems;
        private LocalDateTime generateTime;
        
        // Getters and Setters
        public String getReportTitle() { return reportTitle; }
        public void setReportTitle(String reportTitle) { this.reportTitle = reportTitle; }
        public String getSummary() { return summary; }
        public void setSummary(String summary) { this.summary = summary; }
        public List<FeedbackItem> getFeedbackItems() { return feedbackItems; }
        public void setFeedbackItems(List<FeedbackItem> feedbackItems) { this.feedbackItems = feedbackItems; }
        public Map<String, Double> getSentimentAnalysis() { return sentimentAnalysis; }
        public void setSentimentAnalysis(Map<String, Double> sentimentAnalysis) { this.sentimentAnalysis = sentimentAnalysis; }
        public List<String> getKeyInsights() { return keyInsights; }
        public void setKeyInsights(List<String> keyInsights) { this.keyInsights = keyInsights; }
        public List<String> getActionItems() { return actionItems; }
        public void setActionItems(List<String> actionItems) { this.actionItems = actionItems; }
        public LocalDateTime getGenerateTime() { return generateTime; }
        public void setGenerateTime(LocalDateTime generateTime) { this.generateTime = generateTime; }
        
        public static class FeedbackItem {
            private String category;
            private String content;
            private String sentiment;
            private Double score;
            private String source;
            
            // Getters and Setters
            public String getCategory() { return category; }
            public void setCategory(String category) { this.category = category; }
            public String getContent() { return content; }
            public void setContent(String content) { this.content = content; }
            public String getSentiment() { return sentiment; }
            public void setSentiment(String sentiment) { this.sentiment = sentiment; }
            public Double getScore() { return score; }
            public void setScore(Double score) { this.score = score; }
            public String getSource() { return source; }
            public void setSource(String source) { this.source = source; }
        }
    }

    /**
     * 日程安排请求DTO
     */
    public static class ScheduleRequest {
        @NotNull(message = "用户ID不能为空")
        private Long userId;
        
        private String userType; // STUDENT, TEACHER
        private String timeRange; // WEEK, MONTH
        private List<String> preferences;
        private List<String> constraints;
        
        // Getters and Setters
        public Long getUserId() { return userId; }
        public void setUserId(Long userId) { this.userId = userId; }
        public String getUserType() { return userType; }
        public void setUserType(String userType) { this.userType = userType; }
        public String getTimeRange() { return timeRange; }
        public void setTimeRange(String timeRange) { this.timeRange = timeRange; }
        public List<String> getPreferences() { return preferences; }
        public void setPreferences(List<String> preferences) { this.preferences = preferences; }
        public List<String> getConstraints() { return constraints; }
        public void setConstraints(List<String> constraints) { this.constraints = constraints; }
    }

    /**
     * 日程安排响应DTO
     */
    public static class ScheduleResponse {
        private List<ScheduleItem> scheduleItems;
        private String optimization;
        private List<String> suggestions;
        private LocalDateTime generateTime;
        
        // Getters and Setters
        public List<ScheduleItem> getScheduleItems() { return scheduleItems; }
        public void setScheduleItems(List<ScheduleItem> scheduleItems) { this.scheduleItems = scheduleItems; }
        public String getOptimization() { return optimization; }
        public void setOptimization(String optimization) { this.optimization = optimization; }
        public List<String> getSuggestions() { return suggestions; }
        public void setSuggestions(List<String> suggestions) { this.suggestions = suggestions; }
        public LocalDateTime getGenerateTime() { return generateTime; }
        public void setGenerateTime(LocalDateTime generateTime) { this.generateTime = generateTime; }
        
        public static class ScheduleItem {
            private String title;
            private String description;
            private LocalDateTime startTime;
            private LocalDateTime endTime;
            private String type;
            private String priority;
            
            // Getters and Setters
            public String getTitle() { return title; }
            public void setTitle(String title) { this.title = title; }
            public String getDescription() { return description; }
            public void setDescription(String description) { this.description = description; }
            public LocalDateTime getStartTime() { return startTime; }
            public void setStartTime(LocalDateTime startTime) { this.startTime = startTime; }
            public LocalDateTime getEndTime() { return endTime; }
            public void setEndTime(LocalDateTime endTime) { this.endTime = endTime; }
            public String getType() { return type; }
            public void setType(String type) { this.type = type; }
            public String getPriority() { return priority; }
            public void setPriority(String priority) { this.priority = priority; }
        }
    }

    /**
     * 课堂互动请求DTO
     */
    public static class ClassroomInteractionRequest {
        @NotNull(message = "课程ID不能为空")
        private Long courseId;
        
        private String interactionType; // POLL, QUIZ, DISCUSSION, Q_AND_A
        private String content;
        private List<String> options;
        private Integer duration; // 分钟
        
        // Getters and Setters
        public Long getCourseId() { return courseId; }
        public void setCourseId(Long courseId) { this.courseId = courseId; }
        public String getInteractionType() { return interactionType; }
        public void setInteractionType(String interactionType) { this.interactionType = interactionType; }
        public String getContent() { return content; }
        public void setContent(String content) { this.content = content; }
        public List<String> getOptions() { return options; }
        public void setOptions(List<String> options) { this.options = options; }
        public Integer getDuration() { return duration; }
        public void setDuration(Integer duration) { this.duration = duration; }
    }

    /**
     * 课堂互动响应DTO
     */
    public static class ClassroomInteractionResponse {
        private String interactionId;
        private String status;
        private Map<String, Integer> responses;
        private List<String> insights;
        private String engagement;
        private LocalDateTime createTime;
        
        // Getters and Setters
        public String getInteractionId() { return interactionId; }
        public void setInteractionId(String interactionId) { this.interactionId = interactionId; }
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
        public Map<String, Integer> getResponses() { return responses; }
        public void setResponses(Map<String, Integer> responses) { this.responses = responses; }
        public List<String> getInsights() { return insights; }
        public void setInsights(List<String> insights) { this.insights = insights; }
        public String getEngagement() { return engagement; }
        public void setEngagement(String engagement) { this.engagement = engagement; }
        public LocalDateTime getCreateTime() { return createTime; }
        public void setCreateTime(LocalDateTime createTime) { this.createTime = createTime; }
    }

    /**
     * 考试分析请求DTO
     */
    public static class ExamAnalysisRequest {
        @NotNull(message = "考试ID不能为空")
        private Long examId;
        
        private String analysisType; // PERFORMANCE, DIFFICULTY, DISCRIMINATION
        private Boolean includeItemAnalysis;
        
        // Getters and Setters
        public Long getExamId() { return examId; }
        public void setExamId(Long examId) { this.examId = examId; }
        public String getAnalysisType() { return analysisType; }
        public void setAnalysisType(String analysisType) { this.analysisType = analysisType; }
        public Boolean getIncludeItemAnalysis() { return includeItemAnalysis; }
        public void setIncludeItemAnalysis(Boolean includeItemAnalysis) { this.includeItemAnalysis = includeItemAnalysis; }
    }

    /**
     * 考试分析响应DTO
     */
    public static class ExamAnalysisResponse {
        private Double averageScore;
        private Double standardDeviation;
        private String difficulty;
        private String reliability;
        private List<ItemAnalysis> itemAnalyses;
        private Map<String, Double> scoreDistribution;
        private List<String> recommendations;
        private LocalDateTime analysisTime;
        
        // Getters and Setters
        public Double getAverageScore() { return averageScore; }
        public void setAverageScore(Double averageScore) { this.averageScore = averageScore; }
        public Double getStandardDeviation() { return standardDeviation; }
        public void setStandardDeviation(Double standardDeviation) { this.standardDeviation = standardDeviation; }
        public String getDifficulty() { return difficulty; }
        public void setDifficulty(String difficulty) { this.difficulty = difficulty; }
        public String getReliability() { return reliability; }
        public void setReliability(String reliability) { this.reliability = reliability; }
        public List<ItemAnalysis> getItemAnalyses() { return itemAnalyses; }
        public void setItemAnalyses(List<ItemAnalysis> itemAnalyses) { this.itemAnalyses = itemAnalyses; }
        public Map<String, Double> getScoreDistribution() { return scoreDistribution; }
        public void setScoreDistribution(Map<String, Double> scoreDistribution) { this.scoreDistribution = scoreDistribution; }
        public List<String> getRecommendations() { return recommendations; }
        public void setRecommendations(List<String> recommendations) { this.recommendations = recommendations; }
        public LocalDateTime getAnalysisTime() { return analysisTime; }
        public void setAnalysisTime(LocalDateTime analysisTime) { this.analysisTime = analysisTime; }
        
        public static class ItemAnalysis {
            private Integer questionNumber;
            private Double difficulty;
            private Double discrimination;
            private String quality;
            private List<String> suggestions;
            
            // Getters and Setters
            public Integer getQuestionNumber() { return questionNumber; }
            public void setQuestionNumber(Integer questionNumber) { this.questionNumber = questionNumber; }
            public Double getDifficulty() { return difficulty; }
            public void setDifficulty(Double difficulty) { this.difficulty = difficulty; }
            public Double getDiscrimination() { return discrimination; }
            public void setDiscrimination(Double discrimination) { this.discrimination = discrimination; }
            public String getQuality() { return quality; }
            public void setQuality(String quality) { this.quality = quality; }
            public List<String> getSuggestions() { return suggestions; }
            public void setSuggestions(List<String> suggestions) { this.suggestions = suggestions; }
        }
    }
}