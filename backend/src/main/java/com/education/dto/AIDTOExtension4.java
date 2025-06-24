package com.education.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

/**
 * AI相关DTO扩展类4
 * 包含教学资源、学生档案、学习风险等AI功能相关的DTO定义
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
public class AIDTOExtension4 {

    /**
     * 教学资源请求DTO
     */
    public static class TeachingResourceRequest {
        @NotBlank(message = "资源类型不能为空")
        private String resourceType; // VIDEO, DOCUMENT, EXERCISE, PRESENTATION
        
        @NotBlank(message = "主题不能为空")
        private String topic;
        
        private String targetGrade;
        private String difficulty;
        private String format;
        private List<String> keywords;
        
        // Getters and Setters
        public String getResourceType() { return resourceType; }
        public void setResourceType(String resourceType) { this.resourceType = resourceType; }
        public String getTopic() { return topic; }
        public void setTopic(String topic) { this.topic = topic; }
        public String getTargetGrade() { return targetGrade; }
        public void setTargetGrade(String targetGrade) { this.targetGrade = targetGrade; }
        public String getDifficulty() { return difficulty; }
        public void setDifficulty(String difficulty) { this.difficulty = difficulty; }
        public String getFormat() { return format; }
        public void setFormat(String format) { this.format = format; }
        public List<String> getKeywords() { return keywords; }
        public void setKeywords(List<String> keywords) { this.keywords = keywords; }
    }

    /**
     * 教学资源响应DTO
     */
    public static class TeachingResourceResponse {
        private List<ResourceItem> resources;
        private String searchQuery;
        private Integer totalCount;
        private LocalDateTime generateTime;
        
        // Getters and Setters
        public List<ResourceItem> getResources() { return resources; }
        public void setResources(List<ResourceItem> resources) { this.resources = resources; }
        public String getSearchQuery() { return searchQuery; }
        public void setSearchQuery(String searchQuery) { this.searchQuery = searchQuery; }
        public Integer getTotalCount() { return totalCount; }
        public void setTotalCount(Integer totalCount) { this.totalCount = totalCount; }
        public LocalDateTime getGenerateTime() { return generateTime; }
        public void setGenerateTime(LocalDateTime generateTime) { this.generateTime = generateTime; }
        
        public static class ResourceItem {
            private String title;
            private String description;
            private String url;
            private String type;
            private String difficulty;
            private Double rating;
            private List<String> tags;
            
            // Getters and Setters
            public String getTitle() { return title; }
            public void setTitle(String title) { this.title = title; }
            public String getDescription() { return description; }
            public void setDescription(String description) { this.description = description; }
            public String getUrl() { return url; }
            public void setUrl(String url) { this.url = url; }
            public String getType() { return type; }
            public void setType(String type) { this.type = type; }
            public String getDifficulty() { return difficulty; }
            public void setDifficulty(String difficulty) { this.difficulty = difficulty; }
            public Double getRating() { return rating; }
            public void setRating(Double rating) { this.rating = rating; }
            public List<String> getTags() { return tags; }
            public void setTags(List<String> tags) { this.tags = tags; }
        }
    }

    /**
     * 学生档案请求DTO
     */
    public static class StudentProfileRequest {
        @NotNull(message = "学生ID不能为空")
        private Long studentId;
        
        private String profileType; // ACADEMIC, BEHAVIORAL, COMPREHENSIVE
        private String timeRange; // MONTH, SEMESTER, YEAR
        private Boolean includeRecommendations;
        
        // Getters and Setters
        public Long getStudentId() { return studentId; }
        public void setStudentId(Long studentId) { this.studentId = studentId; }
        public String getProfileType() { return profileType; }
        public void setProfileType(String profileType) { this.profileType = profileType; }
        public String getTimeRange() { return timeRange; }
        public void setTimeRange(String timeRange) { this.timeRange = timeRange; }
        public Boolean getIncludeRecommendations() { return includeRecommendations; }
        public void setIncludeRecommendations(Boolean includeRecommendations) { this.includeRecommendations = includeRecommendations; }
    }

    /**
     * 学生档案响应DTO
     */
    public static class StudentProfileResponse {
        private String studentName;
        private Map<String, Object> academicProfile;
        private Map<String, Object> behavioralProfile;
        private List<String> strengths;
        private List<String> challenges;
        private String learningStyle;
        private List<String> recommendations;
        private LocalDateTime profileTime;
        
        // Getters and Setters
        public String getStudentName() { return studentName; }
        public void setStudentName(String studentName) { this.studentName = studentName; }
        public Map<String, Object> getAcademicProfile() { return academicProfile; }
        public void setAcademicProfile(Map<String, Object> academicProfile) { this.academicProfile = academicProfile; }
        public Map<String, Object> getBehavioralProfile() { return behavioralProfile; }
        public void setBehavioralProfile(Map<String, Object> behavioralProfile) { this.behavioralProfile = behavioralProfile; }
        public List<String> getStrengths() { return strengths; }
        public void setStrengths(List<String> strengths) { this.strengths = strengths; }
        public List<String> getChallenges() { return challenges; }
        public void setChallenges(List<String> challenges) { this.challenges = challenges; }
        public String getLearningStyle() { return learningStyle; }
        public void setLearningStyle(String learningStyle) { this.learningStyle = learningStyle; }
        public List<String> getRecommendations() { return recommendations; }
        public void setRecommendations(List<String> recommendations) { this.recommendations = recommendations; }
        public LocalDateTime getProfileTime() { return profileTime; }
        public void setProfileTime(LocalDateTime profileTime) { this.profileTime = profileTime; }
    }

    /**
     * 学习风险请求DTO
     */
    public static class LearningRiskRequest {
        @NotNull(message = "学生ID不能为空")
        private Long studentId;
        
        private Long courseId;
        private String riskType; // DROPOUT, FAILURE, DISENGAGEMENT
        private String timeHorizon; // WEEK, MONTH, SEMESTER
        
        // Getters and Setters
        public Long getStudentId() { return studentId; }
        public void setStudentId(Long studentId) { this.studentId = studentId; }
        public Long getCourseId() { return courseId; }
        public void setCourseId(Long courseId) { this.courseId = courseId; }
        public String getRiskType() { return riskType; }
        public void setRiskType(String riskType) { this.riskType = riskType; }
        public String getTimeHorizon() { return timeHorizon; }
        public void setTimeHorizon(String timeHorizon) { this.timeHorizon = timeHorizon; }
    }

    /**
     * 学习风险响应DTO
     */
    public static class LearningRiskResponse {
        private String riskLevel; // LOW, MEDIUM, HIGH, CRITICAL
        private Double riskScore;
        private List<RiskFactor> riskFactors;
        private List<String> earlyWarnings;
        private List<String> interventions;
        private String recommendation;
        private LocalDateTime assessmentTime;
        
        // Getters and Setters
        public String getRiskLevel() { return riskLevel; }
        public void setRiskLevel(String riskLevel) { this.riskLevel = riskLevel; }
        public Double getRiskScore() { return riskScore; }
        public void setRiskScore(Double riskScore) { this.riskScore = riskScore; }
        public List<RiskFactor> getRiskFactors() { return riskFactors; }
        public void setRiskFactors(List<RiskFactor> riskFactors) { this.riskFactors = riskFactors; }
        public List<String> getEarlyWarnings() { return earlyWarnings; }
        public void setEarlyWarnings(List<String> earlyWarnings) { this.earlyWarnings = earlyWarnings; }
        public List<String> getInterventions() { return interventions; }
        public void setInterventions(List<String> interventions) { this.interventions = interventions; }
        public String getRecommendation() { return recommendation; }
        public void setRecommendation(String recommendation) { this.recommendation = recommendation; }
        public LocalDateTime getAssessmentTime() { return assessmentTime; }
        public void setAssessmentTime(LocalDateTime assessmentTime) { this.assessmentTime = assessmentTime; }
        
        public static class RiskFactor {
            private String factor;
            private String impact;
            private Double weight;
            private String description;
            
            // Getters and Setters
            public String getFactor() { return factor; }
            public void setFactor(String factor) { this.factor = factor; }
            public String getImpact() { return impact; }
            public void setImpact(String impact) { this.impact = impact; }
            public Double getWeight() { return weight; }
            public void setWeight(Double weight) { this.weight = weight; }
            public String getDescription() { return description; }
            public void setDescription(String description) { this.description = description; }
        }
    }

    /**
     * 教学策略请求DTO
     */
    public static class TeachingStrategyRequest {
        @NotNull(message = "课程ID不能为空")
        private Long courseId;
        
        private String studentLevel;
        private String learningObjective;
        private String contentType;
        private Integer classDuration; // 分钟
        private List<String> availableResources;
        
        // Getters and Setters
        public Long getCourseId() { return courseId; }
        public void setCourseId(Long courseId) { this.courseId = courseId; }
        public String getStudentLevel() { return studentLevel; }
        public void setStudentLevel(String studentLevel) { this.studentLevel = studentLevel; }
        public String getLearningObjective() { return learningObjective; }
        public void setLearningObjective(String learningObjective) { this.learningObjective = learningObjective; }
        public String getContentType() { return contentType; }
        public void setContentType(String contentType) { this.contentType = contentType; }
        public Integer getClassDuration() { return classDuration; }
        public void setClassDuration(Integer classDuration) { this.classDuration = classDuration; }
        public List<String> getAvailableResources() { return availableResources; }
        public void setAvailableResources(List<String> availableResources) { this.availableResources = availableResources; }
    }

    /**
     * 教学策略响应DTO
     */
    public static class TeachingStrategyResponse {
        private String strategyName;
        private String description;
        private List<TeachingStep> steps;
        private List<String> methods;
        private List<String> assessmentStrategies;
        private String expectedOutcome;
        private LocalDateTime generateTime;
        
        // Getters and Setters
        public String getStrategyName() { return strategyName; }
        public void setStrategyName(String strategyName) { this.strategyName = strategyName; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
        public List<TeachingStep> getSteps() { return steps; }
        public void setSteps(List<TeachingStep> steps) { this.steps = steps; }
        public List<String> getMethods() { return methods; }
        public void setMethods(List<String> methods) { this.methods = methods; }
        public List<String> getAssessmentStrategies() { return assessmentStrategies; }
        public void setAssessmentStrategies(List<String> assessmentStrategies) { this.assessmentStrategies = assessmentStrategies; }
        public String getExpectedOutcome() { return expectedOutcome; }
        public void setExpectedOutcome(String expectedOutcome) { this.expectedOutcome = expectedOutcome; }
        public LocalDateTime getGenerateTime() { return generateTime; }
        public void setGenerateTime(LocalDateTime generateTime) { this.generateTime = generateTime; }
        
        public static class TeachingStep {
            private Integer stepNumber;
            private String activity;
            private String method;
            private Integer duration;
            private String purpose;
            
            // Getters and Setters
            public Integer getStepNumber() { return stepNumber; }
            public void setStepNumber(Integer stepNumber) { this.stepNumber = stepNumber; }
            public String getActivity() { return activity; }
            public void setActivity(String activity) { this.activity = activity; }
            public String getMethod() { return method; }
            public void setMethod(String method) { this.method = method; }
            public Integer getDuration() { return duration; }
            public void setDuration(Integer duration) { this.duration = duration; }
            public String getPurpose() { return purpose; }
            public void setPurpose(String purpose) { this.purpose = purpose; }
        }
    }

    /**
     * 多媒体内容请求DTO
     */
    public static class MultimediaContentRequest {
        @NotBlank(message = "内容类型不能为空")
        private String contentType; // VIDEO, AUDIO, ANIMATION, INTERACTIVE
        
        @NotBlank(message = "主题不能为空")
        private String topic;
        
        private String style;
        private Integer duration; // 秒
        private String targetAudience;
        private List<String> requirements;
        
        // Getters and Setters
        public String getContentType() { return contentType; }
        public void setContentType(String contentType) { this.contentType = contentType; }
        public String getTopic() { return topic; }
        public void setTopic(String topic) { this.topic = topic; }
        public String getStyle() { return style; }
        public void setStyle(String style) { this.style = style; }
        public Integer getDuration() { return duration; }
        public void setDuration(Integer duration) { this.duration = duration; }
        public String getTargetAudience() { return targetAudience; }
        public void setTargetAudience(String targetAudience) { this.targetAudience = targetAudience; }
        public List<String> getRequirements() { return requirements; }
        public void setRequirements(List<String> requirements) { this.requirements = requirements; }
    }

    /**
     * 多媒体内容响应DTO
     */
    public static class MultimediaContentResponse {
        private String contentId;
        private String title;
        private String description;
        private String contentUrl;
        private String thumbnailUrl;
        private Integer duration;
        private String format;
        private Map<String, String> metadata;
        private LocalDateTime generateTime;
        
        // Getters and Setters
        public String getContentId() { return contentId; }
        public void setContentId(String contentId) { this.contentId = contentId; }
        public String getTitle() { return title; }
        public void setTitle(String title) { this.title = title; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
        public String getContentUrl() { return contentUrl; }
        public void setContentUrl(String contentUrl) { this.contentUrl = contentUrl; }
        public String getThumbnailUrl() { return thumbnailUrl; }
        public void setThumbnailUrl(String thumbnailUrl) { this.thumbnailUrl = thumbnailUrl; }
        public Integer getDuration() { return duration; }
        public void setDuration(Integer duration) { this.duration = duration; }
        public String getFormat() { return format; }
        public void setFormat(String format) { this.format = format; }
        public Map<String, String> getMetadata() { return metadata; }
        public void setMetadata(Map<String, String> metadata) { this.metadata = metadata; }
        public LocalDateTime getGenerateTime() { return generateTime; }
        public void setGenerateTime(LocalDateTime generateTime) { this.generateTime = generateTime; }
    }

    /**
     * 聊天机器人请求DTO
     */
    public static class ChatbotRequest {
        @NotBlank(message = "用户消息不能为空")
        private String userMessage;
        
        private Long userId;
        private String context; // COURSE, HOMEWORK, GENERAL
        private Long contextId; // 课程ID或作业ID
        private String sessionId;
        
        // Getters and Setters
        public String getUserMessage() { return userMessage; }
        public void setUserMessage(String userMessage) { this.userMessage = userMessage; }
        public Long getUserId() { return userId; }
        public void setUserId(Long userId) { this.userId = userId; }
        public String getContext() { return context; }
        public void setContext(String context) { this.context = context; }
        public Long getContextId() { return contextId; }
        public void setContextId(Long contextId) { this.contextId = contextId; }
        public String getSessionId() { return sessionId; }
        public void setSessionId(String sessionId) { this.sessionId = sessionId; }
    }

    /**
     * 聊天机器人响应DTO
     */
    public static class ChatbotResponse {
        private String botMessage;
        private String messageType; // TEXT, LINK, SUGGESTION, ACTION
        private List<String> suggestions;
        private Map<String, Object> actions;
        private String sessionId;
        private LocalDateTime responseTime;
        
        // Getters and Setters
        public String getBotMessage() { return botMessage; }
        public void setBotMessage(String botMessage) { this.botMessage = botMessage; }
        public String getMessageType() { return messageType; }
        public void setMessageType(String messageType) { this.messageType = messageType; }
        public List<String> getSuggestions() { return suggestions; }
        public void setSuggestions(List<String> suggestions) { this.suggestions = suggestions; }
        public Map<String, Object> getActions() { return actions; }
        public void setActions(Map<String, Object> actions) { this.actions = actions; }
        public String getSessionId() { return sessionId; }
        public void setSessionId(String sessionId) { this.sessionId = sessionId; }
        public LocalDateTime getResponseTime() { return responseTime; }
        public void setResponseTime(LocalDateTime responseTime) { this.responseTime = responseTime; }
    }

    /**
     * AI模型配置请求DTO
     */
    public static class AIModelConfigRequest {
        @NotBlank(message = "模型名称不能为空")
        private String modelName;
        
        private Map<String, Object> parameters;
        private String version;
        private Boolean isActive;
        
        // Getters and Setters
        public String getModelName() { return modelName; }
        public void setModelName(String modelName) { this.modelName = modelName; }
        public Map<String, Object> getParameters() { return parameters; }
        public void setParameters(Map<String, Object> parameters) { this.parameters = parameters; }
        public String getVersion() { return version; }
        public void setVersion(String version) { this.version = version; }
        public Boolean getIsActive() { return isActive; }
        public void setIsActive(Boolean isActive) { this.isActive = isActive; }
    }

    /**
     * AI模型配置响应DTO
     */
    public static class AIModelConfigResponse {
        private String modelId;
        private String modelName;
        private String status;
        private Map<String, Object> currentConfig;
        private String performance;
        private LocalDateTime lastUpdated;
        
        // Getters and Setters
        public String getModelId() { return modelId; }
        public void setModelId(String modelId) { this.modelId = modelId; }
        public String getModelName() { return modelName; }
        public void setModelName(String modelName) { this.modelName = modelName; }
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
        public Map<String, Object> getCurrentConfig() { return currentConfig; }
        public void setCurrentConfig(Map<String, Object> currentConfig) { this.currentConfig = currentConfig; }
        public String getPerformance() { return performance; }
        public void setPerformance(String performance) { this.performance = performance; }
        public LocalDateTime getLastUpdated() { return lastUpdated; }
        public void setLastUpdated(LocalDateTime lastUpdated) { this.lastUpdated = lastUpdated; }
    }

    /**
     * 模型训练请求DTO
     */
    public static class ModelTrainingRequest {
        @NotBlank(message = "模型类型不能为空")
        private String modelType;
        
        @NotBlank(message = "训练数据不能为空")
        private String trainingData;
        
        private Map<String, Object> hyperparameters;
        private String validationData;
        private Integer epochs;
        
        // Getters and Setters
        public String getModelType() { return modelType; }
        public void setModelType(String modelType) { this.modelType = modelType; }
        public String getTrainingData() { return trainingData; }
        public void setTrainingData(String trainingData) { this.trainingData = trainingData; }
        public Map<String, Object> getHyperparameters() { return hyperparameters; }
        public void setHyperparameters(Map<String, Object> hyperparameters) { this.hyperparameters = hyperparameters; }
        public String getValidationData() { return validationData; }
        public void setValidationData(String validationData) { this.validationData = validationData; }
        public Integer getEpochs() { return epochs; }
        public void setEpochs(Integer epochs) { this.epochs = epochs; }
    }

    /**
     * 模型训练响应DTO
     */
    public static class ModelTrainingResponse {
        private String trainingId;
        private String status;
        private Double progress;
        private String currentEpoch;
        private Map<String, Double> metrics;
        private String estimatedCompletion;
        private LocalDateTime startTime;
        
        // Getters and Setters
        public String getTrainingId() { return trainingId; }
        public void setTrainingId(String trainingId) { this.trainingId = trainingId; }
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
        public Double getProgress() { return progress; }
        public void setProgress(Double progress) { this.progress = progress; }
        public String getCurrentEpoch() { return currentEpoch; }
        public void setCurrentEpoch(String currentEpoch) { this.currentEpoch = currentEpoch; }
        public Map<String, Double> getMetrics() { return metrics; }
        public void setMetrics(Map<String, Double> metrics) { this.metrics = metrics; }
        public String getEstimatedCompletion() { return estimatedCompletion; }
        public void setEstimatedCompletion(String estimatedCompletion) { this.estimatedCompletion = estimatedCompletion; }
        public LocalDateTime getStartTime() { return startTime; }
        public void setStartTime(LocalDateTime startTime) { this.startTime = startTime; }
    }

    /**
     * 训练状态响应DTO
     */
    public static class TrainingStatusResponse {
        private String trainingId;
        private String status;
        private Double progress;
        private Map<String, Double> currentMetrics;
        private List<String> logs;
        private String error;
        private LocalDateTime lastUpdate;
        
        // Getters and Setters
        public String getTrainingId() { return trainingId; }
        public void setTrainingId(String trainingId) { this.trainingId = trainingId; }
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
        public Double getProgress() { return progress; }
        public void setProgress(Double progress) { this.progress = progress; }
        public Map<String, Double> getCurrentMetrics() { return currentMetrics; }
        public void setCurrentMetrics(Map<String, Double> currentMetrics) { this.currentMetrics = currentMetrics; }
        public List<String> getLogs() { return logs; }
        public void setLogs(List<String> logs) { this.logs = logs; }
        public String getError() { return error; }
        public void setError(String error) { this.error = error; }
        public LocalDateTime getLastUpdate() { return lastUpdate; }
        public void setLastUpdate(LocalDateTime lastUpdate) { this.lastUpdate = lastUpdate; }
    }

    /**
     * 模型部署请求DTO
     */
    public static class ModelDeploymentRequest {
        @NotBlank(message = "模型ID不能为空")
        private String modelId;
        
        private String environment; // DEVELOPMENT, STAGING, PRODUCTION
        private Map<String, Object> deploymentConfig;
        private Boolean autoScale;
        
        // Getters and Setters
        public String getModelId() { return modelId; }
        public void setModelId(String modelId) { this.modelId = modelId; }
        public String getEnvironment() { return environment; }
        public void setEnvironment(String environment) { this.environment = environment; }
        public Map<String, Object> getDeploymentConfig() { return deploymentConfig; }
        public void setDeploymentConfig(Map<String, Object> deploymentConfig) { this.deploymentConfig = deploymentConfig; }
        public Boolean getAutoScale() { return autoScale; }
        public void setAutoScale(Boolean autoScale) { this.autoScale = autoScale; }
    }

    /**
     * 模型部署响应DTO
     */
    public static class ModelDeploymentResponse {
        private String deploymentId;
        private String status;
        private String endpoint;
        private String version;
        private Map<String, String> healthCheck;
        private LocalDateTime deploymentTime;
        
        // Getters and Setters
        public String getDeploymentId() { return deploymentId; }
        public void setDeploymentId(String deploymentId) { this.deploymentId = deploymentId; }
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
        public String getEndpoint() { return endpoint; }
        public void setEndpoint(String endpoint) { this.endpoint = endpoint; }
        public String getVersion() { return version; }
        public void setVersion(String version) { this.version = version; }
        public Map<String, String> getHealthCheck() { return healthCheck; }
        public void setHealthCheck(Map<String, String> healthCheck) { this.healthCheck = healthCheck; }
        public LocalDateTime getDeploymentTime() { return deploymentTime; }
        public void setDeploymentTime(LocalDateTime deploymentTime) { this.deploymentTime = deploymentTime; }
    }

    /**
     * AI历史记录响应DTO
     */
    public static class AIHistoryResponse {
        private List<HistoryItem> historyItems;
        private String timeRange;
        private Integer totalCount;
        private LocalDateTime queryTime;
        
        // Getters and Setters
        public List<HistoryItem> getHistoryItems() { return historyItems; }
        public void setHistoryItems(List<HistoryItem> historyItems) { this.historyItems = historyItems; }
        public String getTimeRange() { return timeRange; }
        public void setTimeRange(String timeRange) { this.timeRange = timeRange; }
        public Integer getTotalCount() { return totalCount; }
        public void setTotalCount(Integer totalCount) { this.totalCount = totalCount; }
        public LocalDateTime getQueryTime() { return queryTime; }
        public void setQueryTime(LocalDateTime queryTime) { this.queryTime = queryTime; }
        
        public static class HistoryItem {
            private String actionType;
            private String description;
            private Map<String, Object> parameters;
            private String result;
            private LocalDateTime timestamp;
            private Long userId;
            
            // Getters and Setters
            public String getActionType() { return actionType; }
            public void setActionType(String actionType) { this.actionType = actionType; }
            public String getDescription() { return description; }
            public void setDescription(String description) { this.description = description; }
            public Map<String, Object> getParameters() { return parameters; }
            public void setParameters(Map<String, Object> parameters) { this.parameters = parameters; }
            public String getResult() { return result; }
            public void setResult(String result) { this.result = result; }
            public LocalDateTime getTimestamp() { return timestamp; }
            public void setTimestamp(LocalDateTime timestamp) { this.timestamp = timestamp; }
            public Long getUserId() { return userId; }
            public void setUserId(Long userId) { this.userId = userId; }
        }
    }
}