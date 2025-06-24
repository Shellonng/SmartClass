package com.education.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

/**
 * AI相关DTO扩展类2
 * 包含更多AI功能相关的DTO定义
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
public class AIDTOExtension2 {

    /**
     * 学习预测请求DTO
     */
    public static class LearningPredictionRequest {
        @NotNull(message = "学生ID不能为空")
        private Long studentId;
        
        @NotNull(message = "课程ID不能为空")
        private Long courseId;
        
        private String predictionType; // PERFORMANCE, COMPLETION_TIME, DIFFICULTY
        private String timeHorizon; // WEEK, MONTH, SEMESTER
        
        // Getters and Setters
        public Long getStudentId() { return studentId; }
        public void setStudentId(Long studentId) { this.studentId = studentId; }
        public Long getCourseId() { return courseId; }
        public void setCourseId(Long courseId) { this.courseId = courseId; }
        public String getPredictionType() { return predictionType; }
        public void setPredictionType(String predictionType) { this.predictionType = predictionType; }
        public String getTimeHorizon() { return timeHorizon; }
        public void setTimeHorizon(String timeHorizon) { this.timeHorizon = timeHorizon; }
    }

    /**
     * 学习预测响应DTO
     */
    public static class LearningPredictionResponse {
        private Double predictedScore;
        private String predictedGrade;
        private Integer estimatedCompletionDays;
        private List<String> riskFactors;
        private List<String> recommendations;
        private Double confidence;
        private LocalDateTime predictionTime;
        
        // Getters and Setters
        public Double getPredictedScore() { return predictedScore; }
        public void setPredictedScore(Double predictedScore) { this.predictedScore = predictedScore; }
        public String getPredictedGrade() { return predictedGrade; }
        public void setPredictedGrade(String predictedGrade) { this.predictedGrade = predictedGrade; }
        public Integer getEstimatedCompletionDays() { return estimatedCompletionDays; }
        public void setEstimatedCompletionDays(Integer estimatedCompletionDays) { this.estimatedCompletionDays = estimatedCompletionDays; }
        public List<String> getRiskFactors() { return riskFactors; }
        public void setRiskFactors(List<String> riskFactors) { this.riskFactors = riskFactors; }
        public List<String> getRecommendations() { return recommendations; }
        public void setRecommendations(List<String> recommendations) { this.recommendations = recommendations; }
        public Double getConfidence() { return confidence; }
        public void setConfidence(Double confidence) { this.confidence = confidence; }
        public LocalDateTime getPredictionTime() { return predictionTime; }
        public void setPredictionTime(LocalDateTime predictionTime) { this.predictionTime = predictionTime; }
    }

    /**
     * 抄袭检测请求DTO
     */
    public static class PlagiarismDetectionRequest {
        @NotNull(message = "任务ID不能为空")
        private Long taskId;
        
        @NotBlank(message = "学生提交内容不能为空")
        private String submissionContent;
        
        private String detectionScope; // CLASS, COURSE, GLOBAL
        private Double similarityThreshold;
        
        // Getters and Setters
        public Long getTaskId() { return taskId; }
        public void setTaskId(Long taskId) { this.taskId = taskId; }
        public String getSubmissionContent() { return submissionContent; }
        public void setSubmissionContent(String submissionContent) { this.submissionContent = submissionContent; }
        public String getDetectionScope() { return detectionScope; }
        public void setDetectionScope(String detectionScope) { this.detectionScope = detectionScope; }
        public Double getSimilarityThreshold() { return similarityThreshold; }
        public void setSimilarityThreshold(Double similarityThreshold) { this.similarityThreshold = similarityThreshold; }
    }

    /**
     * 抄袭检测响应DTO
     */
    public static class PlagiarismDetectionResponse {
        private Boolean isPlagiarism;
        private Double overallSimilarity;
        private List<SimilarityMatch> matches;
        private String riskLevel; // LOW, MEDIUM, HIGH
        private String report;
        private LocalDateTime detectionTime;
        
        // Getters and Setters
        public Boolean getIsPlagiarism() { return isPlagiarism; }
        public void setIsPlagiarism(Boolean isPlagiarism) { this.isPlagiarism = isPlagiarism; }
        public Double getOverallSimilarity() { return overallSimilarity; }
        public void setOverallSimilarity(Double overallSimilarity) { this.overallSimilarity = overallSimilarity; }
        public List<SimilarityMatch> getMatches() { return matches; }
        public void setMatches(List<SimilarityMatch> matches) { this.matches = matches; }
        public String getRiskLevel() { return riskLevel; }
        public void setRiskLevel(String riskLevel) { this.riskLevel = riskLevel; }
        public String getReport() { return report; }
        public void setReport(String report) { this.report = report; }
        public LocalDateTime getDetectionTime() { return detectionTime; }
        public void setDetectionTime(LocalDateTime detectionTime) { this.detectionTime = detectionTime; }
        
        public static class SimilarityMatch {
            private String sourceText;
            private String matchedText;
            private Double similarity;
            private String source;
            
            // Getters and Setters
            public String getSourceText() { return sourceText; }
            public void setSourceText(String sourceText) { this.sourceText = sourceText; }
            public String getMatchedText() { return matchedText; }
            public void setMatchedText(String matchedText) { this.matchedText = matchedText; }
            public Double getSimilarity() { return similarity; }
            public void setSimilarity(Double similarity) { this.similarity = similarity; }
            public String getSource() { return source; }
            public void setSource(String source) { this.source = source; }
        }
    }

    /**
     * 课程大纲请求DTO
     */
    public static class CourseOutlineRequest {
        @NotBlank(message = "课程主题不能为空")
        private String courseTitle;
        
        @NotBlank(message = "课程描述不能为空")
        private String courseDescription;
        
        private String targetAudience;
        private Integer courseDuration; // 课程时长（小时）
        private String difficulty; // BEGINNER, INTERMEDIATE, ADVANCED
        private List<String> learningObjectives;
        
        // Getters and Setters
        public String getCourseTitle() { return courseTitle; }
        public void setCourseTitle(String courseTitle) { this.courseTitle = courseTitle; }
        public String getCourseDescription() { return courseDescription; }
        public void setCourseDescription(String courseDescription) { this.courseDescription = courseDescription; }
        public String getTargetAudience() { return targetAudience; }
        public void setTargetAudience(String targetAudience) { this.targetAudience = targetAudience; }
        public Integer getCourseDuration() { return courseDuration; }
        public void setCourseDuration(Integer courseDuration) { this.courseDuration = courseDuration; }
        public String getDifficulty() { return difficulty; }
        public void setDifficulty(String difficulty) { this.difficulty = difficulty; }
        public List<String> getLearningObjectives() { return learningObjectives; }
        public void setLearningObjectives(List<String> learningObjectives) { this.learningObjectives = learningObjectives; }
    }

    /**
     * 课程大纲响应DTO
     */
    public static class CourseOutlineResponse {
        private String courseTitle;
        private String courseDescription;
        private List<CourseModule> modules;
        private List<String> prerequisites;
        private List<String> learningOutcomes;
        private String assessmentStrategy;
        private LocalDateTime generateTime;
        
        // Getters and Setters
        public String getCourseTitle() { return courseTitle; }
        public void setCourseTitle(String courseTitle) { this.courseTitle = courseTitle; }
        public String getCourseDescription() { return courseDescription; }
        public void setCourseDescription(String courseDescription) { this.courseDescription = courseDescription; }
        public List<CourseModule> getModules() { return modules; }
        public void setModules(List<CourseModule> modules) { this.modules = modules; }
        public List<String> getPrerequisites() { return prerequisites; }
        public void setPrerequisites(List<String> prerequisites) { this.prerequisites = prerequisites; }
        public List<String> getLearningOutcomes() { return learningOutcomes; }
        public void setLearningOutcomes(List<String> learningOutcomes) { this.learningOutcomes = learningOutcomes; }
        public String getAssessmentStrategy() { return assessmentStrategy; }
        public void setAssessmentStrategy(String assessmentStrategy) { this.assessmentStrategy = assessmentStrategy; }
        public LocalDateTime getGenerateTime() { return generateTime; }
        public void setGenerateTime(LocalDateTime generateTime) { this.generateTime = generateTime; }
        
        public static class CourseModule {
            private String moduleTitle;
            private String moduleDescription;
            private Integer duration;
            private List<String> topics;
            private List<String> activities;
            
            // Getters and Setters
            public String getModuleTitle() { return moduleTitle; }
            public void setModuleTitle(String moduleTitle) { this.moduleTitle = moduleTitle; }
            public String getModuleDescription() { return moduleDescription; }
            public void setModuleDescription(String moduleDescription) { this.moduleDescription = moduleDescription; }
            public Integer getDuration() { return duration; }
            public void setDuration(Integer duration) { this.duration = duration; }
            public List<String> getTopics() { return topics; }
            public void setTopics(List<String> topics) { this.topics = topics; }
            public List<String> getActivities() { return activities; }
            public void setActivities(List<String> activities) { this.activities = activities; }
        }
    }

    /**
     * AI使用统计响应DTO
     */
    public static class AIUsageStatisticsResponse {
        private String timeRange;
        private Integer totalRequests;
        private Map<String, Integer> functionUsage;
        private Integer totalTokensUsed;
        private Double averageResponseTime;
        private Double successRate;
        private List<UsageByDay> dailyUsage;
        private LocalDateTime statisticsTime;
        
        // Getters and Setters
        public String getTimeRange() { return timeRange; }
        public void setTimeRange(String timeRange) { this.timeRange = timeRange; }
        public Integer getTotalRequests() { return totalRequests; }
        public void setTotalRequests(Integer totalRequests) { this.totalRequests = totalRequests; }
        public Map<String, Integer> getFunctionUsage() { return functionUsage; }
        public void setFunctionUsage(Map<String, Integer> functionUsage) { this.functionUsage = functionUsage; }
        public Integer getTotalTokensUsed() { return totalTokensUsed; }
        public void setTotalTokensUsed(Integer totalTokensUsed) { this.totalTokensUsed = totalTokensUsed; }
        public Double getAverageResponseTime() { return averageResponseTime; }
        public void setAverageResponseTime(Double averageResponseTime) { this.averageResponseTime = averageResponseTime; }
        public Double getSuccessRate() { return successRate; }
        public void setSuccessRate(Double successRate) { this.successRate = successRate; }
        public List<UsageByDay> getDailyUsage() { return dailyUsage; }
        public void setDailyUsage(List<UsageByDay> dailyUsage) { this.dailyUsage = dailyUsage; }
        public LocalDateTime getStatisticsTime() { return statisticsTime; }
        public void setStatisticsTime(LocalDateTime statisticsTime) { this.statisticsTime = statisticsTime; }
        
        public static class UsageByDay {
            private String date;
            private Integer requests;
            private Integer tokens;
            
            // Getters and Setters
            public String getDate() { return date; }
            public void setDate(String date) { this.date = date; }
            public Integer getRequests() { return requests; }
            public void setRequests(Integer requests) { this.requests = requests; }
            public Integer getTokens() { return tokens; }
            public void setTokens(Integer tokens) { this.tokens = tokens; }
        }
    }

    /**
     * 教案生成请求DTO
     */
    public static class LessonPlanRequest {
        @NotBlank(message = "课程主题不能为空")
        private String lessonTopic;
        
        @NotNull(message = "课程时长不能为空")
        private Integer duration; // 分钟
        
        private String targetGrade;
        private List<String> learningObjectives;
        private String teachingMethod;
        private List<String> availableResources;
        
        // Getters and Setters
        public String getLessonTopic() { return lessonTopic; }
        public void setLessonTopic(String lessonTopic) { this.lessonTopic = lessonTopic; }
        public Integer getDuration() { return duration; }
        public void setDuration(Integer duration) { this.duration = duration; }
        public String getTargetGrade() { return targetGrade; }
        public void setTargetGrade(String targetGrade) { this.targetGrade = targetGrade; }
        public List<String> getLearningObjectives() { return learningObjectives; }
        public void setLearningObjectives(List<String> learningObjectives) { this.learningObjectives = learningObjectives; }
        public String getTeachingMethod() { return teachingMethod; }
        public void setTeachingMethod(String teachingMethod) { this.teachingMethod = teachingMethod; }
        public List<String> getAvailableResources() { return availableResources; }
        public void setAvailableResources(List<String> availableResources) { this.availableResources = availableResources; }
    }

    /**
     * 教案生成响应DTO
     */
    public static class LessonPlanResponse {
        private String lessonTitle;
        private List<String> objectives;
        private List<LessonActivity> activities;
        private List<String> materials;
        private String assessment;
        private String homework;
        private LocalDateTime generateTime;
        
        // Getters and Setters
        public String getLessonTitle() { return lessonTitle; }
        public void setLessonTitle(String lessonTitle) { this.lessonTitle = lessonTitle; }
        public List<String> getObjectives() { return objectives; }
        public void setObjectives(List<String> objectives) { this.objectives = objectives; }
        public List<LessonActivity> getActivities() { return activities; }
        public void setActivities(List<LessonActivity> activities) { this.activities = activities; }
        public List<String> getMaterials() { return materials; }
        public void setMaterials(List<String> materials) { this.materials = materials; }
        public String getAssessment() { return assessment; }
        public void setAssessment(String assessment) { this.assessment = assessment; }
        public String getHomework() { return homework; }
        public void setHomework(String homework) { this.homework = homework; }
        public LocalDateTime getGenerateTime() { return generateTime; }
        public void setGenerateTime(LocalDateTime generateTime) { this.generateTime = generateTime; }
        
        public static class LessonActivity {
            private String activityName;
            private String description;
            private Integer duration;
            private String activityType;
            
            // Getters and Setters
            public String getActivityName() { return activityName; }
            public void setActivityName(String activityName) { this.activityName = activityName; }
            public String getDescription() { return description; }
            public void setDescription(String description) { this.description = description; }
            public Integer getDuration() { return duration; }
            public void setDuration(Integer duration) { this.duration = duration; }
            public String getActivityType() { return activityType; }
            public void setActivityType(String activityType) { this.activityType = activityType; }
        }
    }

    /**
     * 内容优化请求DTO
     */
    public static class ContentOptimizationRequest {
        @NotNull(message = "课程ID不能为空")
        private Long courseId;
        
        @NotBlank(message = "内容不能为空")
        private String content;
        
        private String optimizationType; // READABILITY, ENGAGEMENT, COMPREHENSION
        private String targetAudience;
        
        // Getters and Setters
        public Long getCourseId() { return courseId; }
        public void setCourseId(Long courseId) { this.courseId = courseId; }
        public String getContent() { return content; }
        public void setContent(String content) { this.content = content; }
        public String getOptimizationType() { return optimizationType; }
        public void setOptimizationType(String optimizationType) { this.optimizationType = optimizationType; }
        public String getTargetAudience() { return targetAudience; }
        public void setTargetAudience(String targetAudience) { this.targetAudience = targetAudience; }
    }

    /**
     * 内容优化响应DTO
     */
    public static class ContentOptimizationResponse {
        private String optimizedContent;
        private List<String> improvements;
        private String readabilityScore;
        private List<String> suggestions;
        private LocalDateTime optimizationTime;
        
        // Getters and Setters
        public String getOptimizedContent() { return optimizedContent; }
        public void setOptimizedContent(String optimizedContent) { this.optimizedContent = optimizedContent; }
        public List<String> getImprovements() { return improvements; }
        public void setImprovements(List<String> improvements) { this.improvements = improvements; }
        public String getReadabilityScore() { return readabilityScore; }
        public void setReadabilityScore(String readabilityScore) { this.readabilityScore = readabilityScore; }
        public List<String> getSuggestions() { return suggestions; }
        public void setSuggestions(List<String> suggestions) { this.suggestions = suggestions; }
        public LocalDateTime getOptimizationTime() { return optimizationTime; }
        public void setOptimizationTime(LocalDateTime optimizationTime) { this.optimizationTime = optimizationTime; }
    }

    /**
     * 学习路径请求DTO
     */
    public static class LearningPathRequest {
        @NotNull(message = "学生ID不能为空")
        private Long studentId;
        
        @NotNull(message = "课程ID不能为空")
        private Long courseId;
        
        private String learningGoal;
        private String currentLevel;
        private Integer timeAvailable; // 每周可用时间（小时）
        private List<String> preferredLearningStyles;
        
        // Getters and Setters
        public Long getStudentId() { return studentId; }
        public void setStudentId(Long studentId) { this.studentId = studentId; }
        public Long getCourseId() { return courseId; }
        public void setCourseId(Long courseId) { this.courseId = courseId; }
        public String getLearningGoal() { return learningGoal; }
        public void setLearningGoal(String learningGoal) { this.learningGoal = learningGoal; }
        public String getCurrentLevel() { return currentLevel; }
        public void setCurrentLevel(String currentLevel) { this.currentLevel = currentLevel; }
        public Integer getTimeAvailable() { return timeAvailable; }
        public void setTimeAvailable(Integer timeAvailable) { this.timeAvailable = timeAvailable; }
        public List<String> getPreferredLearningStyles() { return preferredLearningStyles; }
        public void setPreferredLearningStyles(List<String> preferredLearningStyles) { this.preferredLearningStyles = preferredLearningStyles; }
    }

    /**
     * 学习路径响应DTO
     */
    public static class LearningPathResponse {
        private String pathName;
        private String description;
        private List<LearningStep> steps;
        private Integer estimatedDuration; // 天数
        private String difficulty;
        private LocalDateTime generateTime;
        
        // Getters and Setters
        public String getPathName() { return pathName; }
        public void setPathName(String pathName) { this.pathName = pathName; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
        public List<LearningStep> getSteps() { return steps; }
        public void setSteps(List<LearningStep> steps) { this.steps = steps; }
        public Integer getEstimatedDuration() { return estimatedDuration; }
        public void setEstimatedDuration(Integer estimatedDuration) { this.estimatedDuration = estimatedDuration; }
        public String getDifficulty() { return difficulty; }
        public void setDifficulty(String difficulty) { this.difficulty = difficulty; }
        public LocalDateTime getGenerateTime() { return generateTime; }
        public void setGenerateTime(LocalDateTime generateTime) { this.generateTime = generateTime; }
        
        public static class LearningStep {
            private Integer stepNumber;
            private String stepTitle;
            private String description;
            private List<String> resources;
            private List<String> activities;
            private Integer estimatedTime; // 小时
            private String prerequisite;
            
            // Getters and Setters
            public Integer getStepNumber() { return stepNumber; }
            public void setStepNumber(Integer stepNumber) { this.stepNumber = stepNumber; }
            public String getStepTitle() { return stepTitle; }
            public void setStepTitle(String stepTitle) { this.stepTitle = stepTitle; }
            public String getDescription() { return description; }
            public void setDescription(String description) { this.description = description; }
            public List<String> getResources() { return resources; }
            public void setResources(List<String> resources) { this.resources = resources; }
            public List<String> getActivities() { return activities; }
            public void setActivities(List<String> activities) { this.activities = activities; }
            public Integer getEstimatedTime() { return estimatedTime; }
            public void setEstimatedTime(Integer estimatedTime) { this.estimatedTime = estimatedTime; }
            public String getPrerequisite() { return prerequisite; }
            public void setPrerequisite(String prerequisite) { this.prerequisite = prerequisite; }
        }
    }
}