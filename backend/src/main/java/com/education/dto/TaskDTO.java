package com.education.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;
import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

/**
 * 任务相关DTO
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
public class TaskDTO {

    /**
     * 任务创建请求DTO
     */
    public static class TaskCreateRequest {
        @NotBlank(message = "任务标题不能为空")
        @Size(max = 200, message = "任务标题长度不能超过200字符")
        private String title;
        
        @Size(max = 1000, message = "任务描述长度不能超过1000字符")
        private String description;
        
        @NotNull(message = "课程ID不能为空")
        private Long courseId;
        
        private String taskType; // HOMEWORK, EXAM, PROJECT, QUIZ
        private String difficulty; // EASY, MEDIUM, HARD
        private Integer timeLimit; // 时间限制（分钟）
        private Integer maxAttempts; // 最大尝试次数
        private BigDecimal totalScore; // 总分
        private BigDecimal weight; // 任务权重
        private LocalDateTime startTime;
        private LocalDateTime endTime;
        private Boolean isVisible;
        private String instructions; // 任务说明
        private List<String> attachments; // 附件
        private List<Long> classIds; // 班级ID列表
        private Map<String, Object> settings; // 其他设置
        
        // Getters and Setters
        public String getTitle() { return title; }
        public void setTitle(String title) { this.title = title; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
        public Long getCourseId() { return courseId; }
        public void setCourseId(Long courseId) { this.courseId = courseId; }
        public String getTaskType() { return taskType; }
        public void setTaskType(String taskType) { this.taskType = taskType; }
        public String getDifficulty() { return difficulty; }
        public void setDifficulty(String difficulty) { this.difficulty = difficulty; }
        public Integer getTimeLimit() { return timeLimit; }
        public void setTimeLimit(Integer timeLimit) { this.timeLimit = timeLimit; }
        public Integer getMaxAttempts() { return maxAttempts; }
        public void setMaxAttempts(Integer maxAttempts) { this.maxAttempts = maxAttempts; }
        public BigDecimal getTotalScore() { return totalScore; }
        public void setTotalScore(BigDecimal totalScore) { this.totalScore = totalScore; }
        public BigDecimal getWeight() { return weight; }
        public void setWeight(BigDecimal weight) { this.weight = weight; }
        public LocalDateTime getStartTime() { return startTime; }
        public void setStartTime(LocalDateTime startTime) { this.startTime = startTime; }
        public LocalDateTime getEndTime() { return endTime; }
        public void setEndTime(LocalDateTime endTime) { this.endTime = endTime; }
        public Boolean getIsVisible() { return isVisible; }
        public void setIsVisible(Boolean isVisible) { this.isVisible = isVisible; }
        public String getInstructions() { return instructions; }
        public void setInstructions(String instructions) { this.instructions = instructions; }
        public List<String> getAttachments() { return attachments; }
        public void setAttachments(List<String> attachments) { this.attachments = attachments; }
        public List<Long> getClassIds() { return classIds; }
        public void setClassIds(List<Long> classIds) { this.classIds = classIds; }
        public Map<String, Object> getSettings() { return settings; }
        public void setSettings(Map<String, Object> settings) { this.settings = settings; }
    }

    /**
     * 任务响应DTO
     */
    public static class TaskResponse {
        private Long taskId;
        private String title;
        private String description;
        private Long courseId;
        private String courseName;
        private String taskType;
        private String difficulty;
        private Integer timeLimit;
        private Integer maxAttempts;
        private BigDecimal totalScore;
        private BigDecimal weight;
        private LocalDateTime startTime;
        private LocalDateTime endTime;
        private Boolean isVisible;
        private String instructions;
        private List<String> attachments;
        private String status; // DRAFT, PUBLISHED, CLOSED
        private LocalDateTime createTime;
        private LocalDateTime updateTime;
        private Long creatorId;
        private String creatorName;
        private Integer submissionCount;
        private Map<String, Object> settings;
        
        // Getters and Setters
        public Long getTaskId() { return taskId; }
        public void setTaskId(Long taskId) { this.taskId = taskId; }
        public String getTitle() { return title; }
        public void setTitle(String title) { this.title = title; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
        public Long getCourseId() { return courseId; }
        public void setCourseId(Long courseId) { this.courseId = courseId; }
        public String getCourseName() { return courseName; }
        public void setCourseName(String courseName) { this.courseName = courseName; }
        public String getTaskType() { return taskType; }
        public void setTaskType(String taskType) { this.taskType = taskType; }
        public String getDifficulty() { return difficulty; }
        public void setDifficulty(String difficulty) { this.difficulty = difficulty; }
        public Integer getTimeLimit() { return timeLimit; }
        public void setTimeLimit(Integer timeLimit) { this.timeLimit = timeLimit; }
        public Integer getMaxAttempts() { return maxAttempts; }
        public void setMaxAttempts(Integer maxAttempts) { this.maxAttempts = maxAttempts; }
        public BigDecimal getTotalScore() { return totalScore; }
        public void setTotalScore(BigDecimal totalScore) { this.totalScore = totalScore; }
        public BigDecimal getWeight() { return weight; }
        public void setWeight(BigDecimal weight) { this.weight = weight; }
        public LocalDateTime getStartTime() { return startTime; }
        public void setStartTime(LocalDateTime startTime) { this.startTime = startTime; }
        public LocalDateTime getEndTime() { return endTime; }
        public void setEndTime(LocalDateTime endTime) { this.endTime = endTime; }
        public Boolean getIsVisible() { return isVisible; }
        public void setIsVisible(Boolean isVisible) { this.isVisible = isVisible; }
        public String getInstructions() { return instructions; }
        public void setInstructions(String instructions) { this.instructions = instructions; }
        public List<String> getAttachments() { return attachments; }
        public void setAttachments(List<String> attachments) { this.attachments = attachments; }
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
        public LocalDateTime getCreateTime() { return createTime; }
        public void setCreateTime(LocalDateTime createTime) { this.createTime = createTime; }
        public LocalDateTime getUpdateTime() { return updateTime; }
        public void setUpdateTime(LocalDateTime updateTime) { this.updateTime = updateTime; }
        public Long getCreatorId() { return creatorId; }
        public void setCreatorId(Long creatorId) { this.creatorId = creatorId; }
        public String getCreatorName() { return creatorName; }
        public void setCreatorName(String creatorName) { this.creatorName = creatorName; }
        public Integer getSubmissionCount() { return submissionCount; }
        public void setSubmissionCount(Integer submissionCount) { this.submissionCount = submissionCount; }
        public Map<String, Object> getSettings() { return settings; }
        public void setSettings(Map<String, Object> settings) { this.settings = settings; }
    }

    /**
     * 任务列表响应DTO
     */
    public static class TaskListResponse {
        private Long taskId;
        private String title;
        private String taskType;
        private String difficulty;
        private BigDecimal totalScore;
        private BigDecimal weight;
        private LocalDateTime startTime;
        private LocalDateTime endTime;
        private String status;
        private String courseName;
        private Boolean hasSubmitted;
        private BigDecimal myScore;
        private LocalDateTime submitTime;
        
        // Getters and Setters
        public Long getTaskId() { return taskId; }
        public void setTaskId(Long taskId) { this.taskId = taskId; }
        public String getTitle() { return title; }
        public void setTitle(String title) { this.title = title; }
        public String getTaskType() { return taskType; }
        public void setTaskType(String taskType) { this.taskType = taskType; }
        public String getDifficulty() { return difficulty; }
        public void setDifficulty(String difficulty) { this.difficulty = difficulty; }
        public BigDecimal getTotalScore() { return totalScore; }
        public void setTotalScore(BigDecimal totalScore) { this.totalScore = totalScore; }
        public BigDecimal getWeight() { return weight; }
        public void setWeight(BigDecimal weight) { this.weight = weight; }
        public LocalDateTime getStartTime() { return startTime; }
        public void setStartTime(LocalDateTime startTime) { this.startTime = startTime; }
        public LocalDateTime getEndTime() { return endTime; }
        public void setEndTime(LocalDateTime endTime) { this.endTime = endTime; }
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
        public String getCourseName() { return courseName; }
        public void setCourseName(String courseName) { this.courseName = courseName; }
        public Boolean getHasSubmitted() { return hasSubmitted; }
        public void setHasSubmitted(Boolean hasSubmitted) { this.hasSubmitted = hasSubmitted; }
        public BigDecimal getMyScore() { return myScore; }
        public void setMyScore(BigDecimal myScore) { this.myScore = myScore; }
        public LocalDateTime getSubmitTime() { return submitTime; }
        public void setSubmitTime(LocalDateTime submitTime) { this.submitTime = submitTime; }
    }

    /**
     * 任务详情响应DTO
     */
    public static class TaskDetailResponse {
        private Long taskId;
        private String title;
        private String description;
        private String taskType;
        private String difficulty;
        private Integer timeLimit;
        private Integer maxAttempts;
        private BigDecimal totalScore;
        private LocalDateTime startTime;
        private LocalDateTime endTime;
        private String instructions;
        private List<String> attachments;
        private String status;
        private String courseName;
        private Boolean hasSubmitted;
        private Integer attemptCount;
        private BigDecimal myScore;
        private LocalDateTime submitTime;
        private String feedback;
        private Map<String, Object> settings;
        private BigDecimal weight;
        
        // Getters and Setters
        public Long getTaskId() { return taskId; }
        public void setTaskId(Long taskId) { this.taskId = taskId; }
        public String getTitle() { return title; }
        public void setTitle(String title) { this.title = title; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
        public String getTaskType() { return taskType; }
        public void setTaskType(String taskType) { this.taskType = taskType; }
        public String getDifficulty() { return difficulty; }
        public void setDifficulty(String difficulty) { this.difficulty = difficulty; }
        public Integer getTimeLimit() { return timeLimit; }
        public void setTimeLimit(Integer timeLimit) { this.timeLimit = timeLimit; }
        public Integer getMaxAttempts() { return maxAttempts; }
        public void setMaxAttempts(Integer maxAttempts) { this.maxAttempts = maxAttempts; }
        public BigDecimal getTotalScore() { return totalScore; }
        public void setTotalScore(BigDecimal totalScore) { this.totalScore = totalScore; }
        public BigDecimal getWeight() { return weight; }
        public void setWeight(BigDecimal weight) { this.weight = weight; }
        public LocalDateTime getStartTime() { return startTime; }
        public void setStartTime(LocalDateTime startTime) { this.startTime = startTime; }
        public LocalDateTime getEndTime() { return endTime; }
        public void setEndTime(LocalDateTime endTime) { this.endTime = endTime; }
        public String getInstructions() { return instructions; }
        public void setInstructions(String instructions) { this.instructions = instructions; }
        public List<String> getAttachments() { return attachments; }
        public void setAttachments(List<String> attachments) { this.attachments = attachments; }
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
        public String getCourseName() { return courseName; }
        public void setCourseName(String courseName) { this.courseName = courseName; }
        public Boolean getHasSubmitted() { return hasSubmitted; }
        public void setHasSubmitted(Boolean hasSubmitted) { this.hasSubmitted = hasSubmitted; }
        public Integer getAttemptCount() { return attemptCount; }
        public void setAttemptCount(Integer attemptCount) { this.attemptCount = attemptCount; }
        public BigDecimal getMyScore() { return myScore; }
        public void setMyScore(BigDecimal myScore) { this.myScore = myScore; }
        public LocalDateTime getSubmitTime() { return submitTime; }
        public void setSubmitTime(LocalDateTime submitTime) { this.submitTime = submitTime; }
        public String getFeedback() { return feedback; }
        public void setFeedback(String feedback) { this.feedback = feedback; }
        public Map<String, Object> getSettings() { return settings; }
        public void setSettings(Map<String, Object> settings) { this.settings = settings; }
    }

    /**
     * 任务提交请求DTO
     */
    public static class TaskSubmissionRequest {
        @NotNull(message = "任务ID不能为空")
        private Long taskId;
        
        private String content; // 提交内容
        private List<String> attachments; // 附件
        private List<String> links; // 提交链接
        private Map<String, Object> answers; // 答案（用于选择题等）
        private String submissionType; // TEXT, FILE, MIXED
        
        // Getters and Setters
        public Long getTaskId() { return taskId; }
        public void setTaskId(Long taskId) { this.taskId = taskId; }
        public String getContent() { return content; }
        public void setContent(String content) { this.content = content; }
        public List<String> getAttachments() { return attachments; }
        public void setAttachments(List<String> attachments) { this.attachments = attachments; }
        public List<String> getLinks() { return links; }
        public void setLinks(List<String> links) { this.links = links; }
        public Map<String, Object> getAnswers() { return answers; }
        public void setAnswers(Map<String, Object> answers) { this.answers = answers; }
        public String getSubmissionType() { return submissionType; }
        public void setSubmissionType(String submissionType) { this.submissionType = submissionType; }
    }

    /**
     * 任务草稿保存请求DTO
     */
    public static class TaskDraftRequest {
        @NotNull(message = "任务ID不能为空")
        private Long taskId;
        
        private String content;
        private List<String> attachments;
        private List<String> links;
        private Map<String, Object> answers;
        
        // Getters and Setters
        public Long getTaskId() { return taskId; }
        public void setTaskId(Long taskId) { this.taskId = taskId; }
        public String getContent() { return content; }
        public void setContent(String content) { this.content = content; }
        public List<String> getAttachments() { return attachments; }
        public void setAttachments(List<String> attachments) { this.attachments = attachments; }
        public List<String> getLinks() { return links; }
        public void setLinks(List<String> links) { this.links = links; }
        public Map<String, Object> getAnswers() { return answers; }
        public void setAnswers(Map<String, Object> answers) { this.answers = answers; }
    }

    /**
     * 任务更新请求DTO
     */
    public static class TaskUpdateRequest {
        private String title;
        private String description;
        private String difficulty;
        private Integer timeLimit;
        private Integer maxAttempts;
        private BigDecimal totalScore;
        private BigDecimal weight;
        private LocalDateTime startTime;
        private LocalDateTime endTime;
        private Boolean isVisible;
        private String instructions;
        private List<String> attachments;
        private List<Long> classIds; // 班级ID列表
        private Map<String, Object> settings;
        
        // Getters and Setters
        public String getTitle() { return title; }
        public void setTitle(String title) { this.title = title; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
        public String getDifficulty() { return difficulty; }
        public void setDifficulty(String difficulty) { this.difficulty = difficulty; }
        public Integer getTimeLimit() { return timeLimit; }
        public void setTimeLimit(Integer timeLimit) { this.timeLimit = timeLimit; }
        public Integer getMaxAttempts() { return maxAttempts; }
        public void setMaxAttempts(Integer maxAttempts) { this.maxAttempts = maxAttempts; }
        public BigDecimal getTotalScore() { return totalScore; }
        public void setTotalScore(BigDecimal totalScore) { this.totalScore = totalScore; }
        public BigDecimal getWeight() { return weight; }
        public void setWeight(BigDecimal weight) { this.weight = weight; }
        public LocalDateTime getStartTime() { return startTime; }
        public void setStartTime(LocalDateTime startTime) { this.startTime = startTime; }
        public LocalDateTime getEndTime() { return endTime; }
        public void setEndTime(LocalDateTime endTime) { this.endTime = endTime; }
        public Boolean getIsVisible() { return isVisible; }
        public void setIsVisible(Boolean isVisible) { this.isVisible = isVisible; }
        public String getInstructions() { return instructions; }
        public void setInstructions(String instructions) { this.instructions = instructions; }
        public List<String> getAttachments() { return attachments; }
        public void setAttachments(List<String> attachments) { this.attachments = attachments; }
        public List<Long> getClassIds() { return classIds; }
        public void setClassIds(List<Long> classIds) { this.classIds = classIds; }
        public Map<String, Object> getSettings() { return settings; }
        public void setSettings(Map<String, Object> settings) { this.settings = settings; }
    }

    /**
     * 任务草稿响应DTO
     */
    public static class TaskDraftResponse {
        private Long taskId;
        private String content;
        private List<String> attachments;
        private List<String> links;
        private Map<String, Object> answers;
        private LocalDateTime saveTime;
        
        // Getters and Setters
        public Long getTaskId() { return taskId; }
        public void setTaskId(Long taskId) { this.taskId = taskId; }
        public String getContent() { return content; }
        public void setContent(String content) { this.content = content; }
        public List<String> getAttachments() { return attachments; }
        public void setAttachments(List<String> attachments) { this.attachments = attachments; }
        public List<String> getLinks() { return links; }
        public void setLinks(List<String> links) { this.links = links; }
        public Map<String, Object> getAnswers() { return answers; }
        public void setAnswers(Map<String, Object> answers) { this.answers = answers; }
        public LocalDateTime getSaveTime() { return saveTime; }
        public void setSaveTime(LocalDateTime saveTime) { this.saveTime = saveTime; }
    }

    /**
     * 提交响应DTO
     */
    public static class SubmissionResponse {
        private Long submissionId;
        private Long taskId;
        private String content;
        private List<String> attachments;
        private List<String> links;
        private Map<String, Object> answers;
        private BigDecimal score;
        private String feedback;
        private LocalDateTime submitTime;
        private String status;
        
        // Getters and Setters
        public Long getSubmissionId() { return submissionId; }
        public void setSubmissionId(Long submissionId) { this.submissionId = submissionId; }
        public Long getTaskId() { return taskId; }
        public void setTaskId(Long taskId) { this.taskId = taskId; }
        public String getContent() { return content; }
        public void setContent(String content) { this.content = content; }
        public List<String> getAttachments() { return attachments; }
        public void setAttachments(List<String> attachments) { this.attachments = attachments; }
        public List<String> getLinks() { return links; }
        public void setLinks(List<String> links) { this.links = links; }
        public Map<String, Object> getAnswers() { return answers; }
        public void setAnswers(Map<String, Object> answers) { this.answers = answers; }
        public BigDecimal getScore() { return score; }
        public void setScore(BigDecimal score) { this.score = score; }
        public String getFeedback() { return feedback; }
        public void setFeedback(String feedback) { this.feedback = feedback; }
        public LocalDateTime getSubmitTime() { return submitTime; }
        public void setSubmitTime(LocalDateTime submitTime) { this.submitTime = submitTime; }
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
    }

    /**
     * 提交更新请求DTO
     */
    public static class SubmissionUpdateRequest {
        private String content;
        private List<String> attachments;
        private List<String> links;
        private Map<String, Object> answers;
        
        // Getters and Setters
        public String getContent() { return content; }
        public void setContent(String content) { this.content = content; }
        public List<String> getAttachments() { return attachments; }
        public void setAttachments(List<String> attachments) { this.attachments = attachments; }
        public List<String> getLinks() { return links; }
        public void setLinks(List<String> links) { this.links = links; }
        public Map<String, Object> getAnswers() { return answers; }
        public void setAnswers(Map<String, Object> answers) { this.answers = answers; }
    }

    /**
     * 任务成绩响应DTO
     */
    public static class TaskGradeResponse {
        private Long taskId;
        private String taskTitle;
        private BigDecimal score;
        private BigDecimal totalScore;
        private String grade;
        private LocalDateTime gradeTime;
        private String feedback;
        
        // Getters and Setters
        public Long getTaskId() { return taskId; }
        public void setTaskId(Long taskId) { this.taskId = taskId; }
        public String getTaskTitle() { return taskTitle; }
        public void setTaskTitle(String taskTitle) { this.taskTitle = taskTitle; }
        public BigDecimal getScore() { return score; }
        public void setScore(BigDecimal score) { this.score = score; }
        public BigDecimal getTotalScore() { return totalScore; }
        public void setTotalScore(BigDecimal totalScore) { this.totalScore = totalScore; }
        public String getGrade() { return grade; }
        public void setGrade(String grade) { this.grade = grade; }
        public LocalDateTime getGradeTime() { return gradeTime; }
        public void setGradeTime(LocalDateTime gradeTime) { this.gradeTime = gradeTime; }
        public String getFeedback() { return feedback; }
        public void setFeedback(String feedback) { this.feedback = feedback; }
    }

    /**
     * 任务反馈响应DTO
     */
    public static class TaskFeedbackResponse {
        private Long taskId;
        private String feedback;
        private BigDecimal score;
        private String graderName;
        private LocalDateTime feedbackTime;
        private List<String> suggestions;
        private LocalDateTime gradeTime;

        // Getters and Setters
        public Long getTaskId() { return taskId; }
        public void setTaskId(Long taskId) { this.taskId = taskId; }
        public String getFeedback() { return feedback; }
        public void setFeedback(String feedback) { this.feedback = feedback; }
        public BigDecimal getScore() { return score; }
        public void setScore(BigDecimal score) { this.score = score; }
        public String getGraderName() { return graderName; }
        public void setGraderName(String graderName) { this.graderName = graderName; }
        public LocalDateTime getFeedbackTime() { return feedbackTime; }
        public void setFeedbackTime(LocalDateTime feedbackTime) { this.feedbackTime = feedbackTime; }
        public List<String> getSuggestions() { return suggestions; }
        public void setSuggestions(List<String> suggestions) { this.suggestions = suggestions; }

        public void setGradeTime(LocalDateTime graTime) {
            this.gradeTime = graTime;
        }
    }

    /**
     * 任务搜索请求DTO
     */
    public static class TaskSearchRequest {
        private String keyword;
        private String taskType;
        private String difficulty;
        private String status;
        private LocalDateTime startDate;
        private LocalDateTime endDate;
        private Long courseId;
        
        // Getters and Setters
        public String getKeyword() { return keyword; }
        public void setKeyword(String keyword) { this.keyword = keyword; }
        public String getTaskType() { return taskType; }
        public void setTaskType(String taskType) { this.taskType = taskType; }
        public String getDifficulty() { return difficulty; }
        public void setDifficulty(String difficulty) { this.difficulty = difficulty; }
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
        public LocalDateTime getStartDate() { return startDate; }
        public void setStartDate(LocalDateTime startDate) { this.startDate = startDate; }
        public LocalDateTime getEndDate() { return endDate; }
        public void setEndDate(LocalDateTime endDate) { this.endDate = endDate; }
        public Long getCourseId() { return courseId; }
        public void setCourseId(Long courseId) { this.courseId = courseId; }
    }

    /**
     * 任务统计响应DTO
     */
    public static class TaskStatisticsResponse {
        private Integer totalTasks;
        private Integer completedTasks;
        private Integer pendingTasks;
        private Integer overdueTasks;
        private Double averageScore;
        private Integer totalSubmissions;
        private Integer gradedSubmissions;
        private Integer lateSubmissions;
        private Double maxScore;
        private Double minScore;
        
        // Getters and Setters
        public Integer getTotalTasks() { return totalTasks; }
        public void setTotalTasks(Integer totalTasks) { this.totalTasks = totalTasks; }
        public Integer getCompletedTasks() { return completedTasks; }
        public void setCompletedTasks(Integer completedTasks) { this.completedTasks = completedTasks; }
        public Integer getPendingTasks() { return pendingTasks; }
        public void setPendingTasks(Integer pendingTasks) { this.pendingTasks = pendingTasks; }
        public Integer getOverdueTasks() { return overdueTasks; }
        public void setOverdueTasks(Integer overdueTasks) { this.overdueTasks = overdueTasks; }
        public Double getAverageScore() { return averageScore; }
        public void setAverageScore(Double averageScore) { this.averageScore = averageScore; }
        public Integer getTotalSubmissions() { return totalSubmissions; }
        public void setTotalSubmissions(Integer totalSubmissions) { this.totalSubmissions = totalSubmissions; }
        public Integer getGradedSubmissions() { return gradedSubmissions; }
        public void setGradedSubmissions(Integer gradedSubmissions) { this.gradedSubmissions = gradedSubmissions; }
        public void setGradedSubmissions(int gradedSubmissions) { this.gradedSubmissions = gradedSubmissions; }
        public Integer getLateSubmissions() { return lateSubmissions; }
        public void setLateSubmissions(Integer lateSubmissions) { this.lateSubmissions = lateSubmissions; }
        public void setLateSubmissions(int lateSubmissions) { this.lateSubmissions = lateSubmissions; }
        public Double getMaxScore() { return maxScore; }
        public void setMaxScore(Double maxScore) { this.maxScore = maxScore; }
        public void setMaxScore(double maxScore) { this.maxScore = maxScore; }
        public Double getMinScore() { return minScore; }
        public void setMinScore(Double minScore) { this.minScore = minScore; }
        public void setMinScore(double minScore) { this.minScore = minScore; }
    }

    /**
     * 任务日历响应DTO
     */
    public static class TaskCalendarResponse {
        private Integer year;
        private Integer month;
        private List<TaskCalendarItem> tasks;
        
        // Getters and Setters
        public Integer getYear() { return year; }
        public void setYear(Integer year) { this.year = year; }
        public Integer getMonth() { return month; }
        public void setMonth(Integer month) { this.month = month; }
        public List<TaskCalendarItem> getTasks() { return tasks; }
        public void setTasks(List<TaskCalendarItem> tasks) { this.tasks = tasks; }
        
        public static class TaskCalendarItem {
            private Long taskId;
            private String title;
            private LocalDateTime dueDate;
            private String status;
            private String taskType;
            
            // Getters and Setters
            public Long getTaskId() { return taskId; }
            public void setTaskId(Long taskId) { this.taskId = taskId; }
            public String getTitle() { return title; }
            public void setTitle(String title) { this.title = title; }
            public LocalDateTime getDueDate() { return dueDate; }
            public void setDueDate(LocalDateTime dueDate) { this.dueDate = dueDate; }
            public String getStatus() { return status; }
            public void setStatus(String status) { this.status = status; }
            public String getTaskType() { return taskType; }
            public void setTaskType(String taskType) { this.taskType = taskType; }
        }
    }

    /**
     * 任务提醒响应DTO
     */
    public static class TaskReminderResponse {
        private Long reminderId;
        private Long taskId;
        private String taskTitle;
        private LocalDateTime reminderTime;
        private String reminderType;
        private Boolean isActive;
        
        // Getters and Setters
        public Long getReminderId() { return reminderId; }
        public void setReminderId(Long reminderId) { this.reminderId = reminderId; }
        public Long getTaskId() { return taskId; }
        public void setTaskId(Long taskId) { this.taskId = taskId; }
        public String getTaskTitle() { return taskTitle; }
        public void setTaskTitle(String taskTitle) { this.taskTitle = taskTitle; }
        public LocalDateTime getReminderTime() { return reminderTime; }
        public void setReminderTime(LocalDateTime reminderTime) { this.reminderTime = reminderTime; }
        public String getReminderType() { return reminderType; }
        public void setReminderType(String reminderType) { this.reminderType = reminderType; }
        public Boolean getIsActive() { return isActive; }
        public void setIsActive(Boolean isActive) { this.isActive = isActive; }
    }

    /**
     * 任务提醒请求DTO
     */
    public static class TaskReminderRequest {
        @NotNull(message = "任务ID不能为空")
        private Long taskId;
        
        @NotNull(message = "提醒时间不能为空")
        private LocalDateTime reminderTime;
        
        private String reminderType; // EMAIL, SMS, NOTIFICATION
        private String message;
        
        // Getters and Setters
        public Long getTaskId() { return taskId; }
        public void setTaskId(Long taskId) { this.taskId = taskId; }
        public LocalDateTime getReminderTime() { return reminderTime; }
        public void setReminderTime(LocalDateTime reminderTime) { this.reminderTime = reminderTime; }
        public String getReminderType() { return reminderType; }
        public void setReminderType(String reminderType) { this.reminderType = reminderType; }
        public String getMessage() { return message; }
        public void setMessage(String message) { this.message = message; }
    }

    /**
     * 同行评议任务响应DTO
     */
    public static class PeerReviewTaskResponse {
        private Long taskId;
        private String taskTitle;
        private Long revieweeId;
        private String revieweeName;
        private LocalDateTime dueDate;
        private String status;
        
        // Getters and Setters
        public Long getTaskId() { return taskId; }
        public void setTaskId(Long taskId) { this.taskId = taskId; }
        public String getTaskTitle() { return taskTitle; }
        public void setTaskTitle(String taskTitle) { this.taskTitle = taskTitle; }
        public Long getRevieweeId() { return revieweeId; }
        public void setRevieweeId(Long revieweeId) { this.revieweeId = revieweeId; }
        public String getRevieweeName() { return revieweeName; }
        public void setRevieweeName(String revieweeName) { this.revieweeName = revieweeName; }
        public LocalDateTime getDueDate() { return dueDate; }
        public void setDueDate(LocalDateTime dueDate) { this.dueDate = dueDate; }
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
    }

    /**
     * 同行评议请求DTO
     */
    public static class PeerReviewRequest {
        @NotNull(message = "任务ID不能为空")
        private Long taskId;
        
        @NotNull(message = "被评议者ID不能为空")
        private Long revieweeId;
        
        private BigDecimal score;
        private String feedback;
        private Map<String, Object> criteria;
        
        // Getters and Setters
        public Long getTaskId() { return taskId; }
        public void setTaskId(Long taskId) { this.taskId = taskId; }
        public Long getRevieweeId() { return revieweeId; }
        public void setRevieweeId(Long revieweeId) { this.revieweeId = revieweeId; }
        public BigDecimal getScore() { return score; }
        public void setScore(BigDecimal score) { this.score = score; }
        public String getFeedback() { return feedback; }
        public void setFeedback(String feedback) { this.feedback = feedback; }
        public Map<String, Object> getCriteria() { return criteria; }
        public void setCriteria(Map<String, Object> criteria) { this.criteria = criteria; }
    }

    /**
     * 同行评议响应DTO
     */
    public static class PeerReviewResponse {
        private Long reviewId;
        private String reviewerName;
        private BigDecimal score;
        private String feedback;
        private LocalDateTime reviewTime;
        private Map<String, Object> criteria;
        
        // Getters and Setters
        public Long getReviewId() { return reviewId; }
        public void setReviewId(Long reviewId) { this.reviewId = reviewId; }
        public String getReviewerName() { return reviewerName; }
        public void setReviewerName(String reviewerName) { this.reviewerName = reviewerName; }
        public BigDecimal getScore() { return score; }
        public void setScore(BigDecimal score) { this.score = score; }
        public String getFeedback() { return feedback; }
        public void setFeedback(String feedback) { this.feedback = feedback; }
        public LocalDateTime getReviewTime() { return reviewTime; }
        public void setReviewTime(LocalDateTime reviewTime) { this.reviewTime = reviewTime; }
        public Map<String, Object> getCriteria() { return criteria; }
        public void setCriteria(Map<String, Object> criteria) { this.criteria = criteria; }
    }

    /**
     * 任务延期请求DTO
     */
    public static class TaskExtensionRequest {
        @NotNull(message = "任务ID不能为空")
        private Long taskId;
        
        @NotNull(message = "延期时间不能为空")
        private LocalDateTime newDueDate;
        
        @NotBlank(message = "延期原因不能为空")
        private String reason;
        
        // Getters and Setters
        public Long getTaskId() { return taskId; }
        public void setTaskId(Long taskId) { this.taskId = taskId; }
        public LocalDateTime getNewDueDate() { return newDueDate; }
        public void setNewDueDate(LocalDateTime newDueDate) { this.newDueDate = newDueDate; }
        public String getReason() { return reason; }
        public void setReason(String reason) { this.reason = reason; }
    }

    /**
     * 延期状态响应DTO
     */
    public static class ExtensionStatusResponse {
        private Long extensionId;
        private Long taskId;
        private LocalDateTime originalDueDate;
        private LocalDateTime newDueDate;
        private String reason;
        private String status; // PENDING, APPROVED, REJECTED
        private String reviewerComment;
        private LocalDateTime requestTime;
        
        // Getters and Setters
        public Long getExtensionId() { return extensionId; }
        public void setExtensionId(Long extensionId) { this.extensionId = extensionId; }
        public Long getTaskId() { return taskId; }
        public void setTaskId(Long taskId) { this.taskId = taskId; }
        public LocalDateTime getOriginalDueDate() { return originalDueDate; }
        public void setOriginalDueDate(LocalDateTime originalDueDate) { this.originalDueDate = originalDueDate; }
        public LocalDateTime getNewDueDate() { return newDueDate; }
        public void setNewDueDate(LocalDateTime newDueDate) { this.newDueDate = newDueDate; }
        public String getReason() { return reason; }
        public void setReason(String reason) { this.reason = reason; }
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
        public String getReviewerComment() { return reviewerComment; }
        public void setReviewerComment(String reviewerComment) { this.reviewerComment = reviewerComment; }
        public LocalDateTime getRequestTime() { return requestTime; }
        public void setRequestTime(LocalDateTime requestTime) { this.requestTime = requestTime; }
    }

    /**
     * 任务讨论响应DTO
     */
    public static class TaskDiscussionResponse {
        private Long discussionId; // 讨论ID
        private Long taskId; // 任务ID
        private String title; // 讨论标题
        private String content; // 讨论内容
        private Long authorId; // 作者ID
        private String authorName; // 作者姓名
        private LocalDateTime createTime; // 创建时间
        private LocalDateTime updateTime; // 更新时间
        private Integer replyCount; // 回复数量
        private Integer likeCount; // 点赞数量
        private Boolean isSticky; // 是否置顶
        private String status; // 状态：ACTIVE, CLOSED, DELETED
        private List<TaskDiscussionReplyResponse> replies; // 回复列表
        
        // Getters and Setters
        public Long getDiscussionId() { return discussionId; }
        public void setDiscussionId(Long discussionId) { this.discussionId = discussionId; }
        public Long getTaskId() { return taskId; }
        public void setTaskId(Long taskId) { this.taskId = taskId; }
        public String getTitle() { return title; }
        public void setTitle(String title) { this.title = title; }
        public String getContent() { return content; }
        public void setContent(String content) { this.content = content; }
        public Long getAuthorId() { return authorId; }
        public void setAuthorId(Long authorId) { this.authorId = authorId; }
        public String getAuthorName() { return authorName; }
        public void setAuthorName(String authorName) { this.authorName = authorName; }
        public LocalDateTime getCreateTime() { return createTime; }
        public void setCreateTime(LocalDateTime createTime) { this.createTime = createTime; }
        public LocalDateTime getUpdateTime() { return updateTime; }
        public void setUpdateTime(LocalDateTime updateTime) { this.updateTime = updateTime; }
        public Integer getReplyCount() { return replyCount; }
        public void setReplyCount(Integer replyCount) { this.replyCount = replyCount; }
        public Integer getLikeCount() { return likeCount; }
        public void setLikeCount(Integer likeCount) { this.likeCount = likeCount; }
        public Boolean getIsSticky() { return isSticky; }
        public void setIsSticky(Boolean isSticky) { this.isSticky = isSticky; }
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
        public List<TaskDiscussionReplyResponse> getReplies() { return replies; }
        public void setReplies(List<TaskDiscussionReplyResponse> replies) { this.replies = replies; }
    }

    /**
     * 任务讨论回复响应DTO
     */
    public static class TaskDiscussionReplyResponse {
        private Long replyId; // 回复ID
        private Long discussionId; // 讨论ID
        private String content; // 回复内容
        private Long authorId; // 作者ID
        private String authorName; // 作者姓名
        private LocalDateTime createTime; // 创建时间
        private Long parentReplyId; // 父回复ID
        private Integer likeCount; // 点赞数量
        
        // Getters and Setters
        public Long getReplyId() { return replyId; }
        public void setReplyId(Long replyId) { this.replyId = replyId; }
        public Long getDiscussionId() { return discussionId; }
        public void setDiscussionId(Long discussionId) { this.discussionId = discussionId; }
        public String getContent() { return content; }
        public void setContent(String content) { this.content = content; }
        public Long getAuthorId() { return authorId; }
        public void setAuthorId(Long authorId) { this.authorId = authorId; }
        public String getAuthorName() { return authorName; }
        public void setAuthorName(String authorName) { this.authorName = authorName; }
        public LocalDateTime getCreateTime() { return createTime; }
        public void setCreateTime(LocalDateTime createTime) { this.createTime = createTime; }
        public Long getParentReplyId() { return parentReplyId; }
        public void setParentReplyId(Long parentReplyId) { this.parentReplyId = parentReplyId; }
        public Integer getLikeCount() { return likeCount; }
        public void setLikeCount(Integer likeCount) { this.likeCount = likeCount; }
    }

    /**
     * 任务讨论创建请求DTO
     */
    public static class TaskDiscussionCreateRequest {
        @NotNull(message = "任务ID不能为空")
        private Long taskId; // 任务ID
        
        @NotBlank(message = "讨论标题不能为空")
        private String title; // 讨论标题
        
        @NotBlank(message = "讨论内容不能为空")
        private String content; // 讨论内容
        
        private Boolean isSticky; // 是否置顶
        
        // Getters and Setters
        public Long getTaskId() { return taskId; }
        public void setTaskId(Long taskId) { this.taskId = taskId; }
        public String getTitle() { return title; }
        public void setTitle(String title) { this.title = title; }
        public String getContent() { return content; }
        public void setContent(String content) { this.content = content; }
        public Boolean getIsSticky() { return isSticky; }
        public void setIsSticky(Boolean isSticky) { this.isSticky = isSticky; }
    }

    /**
     * 任务讨论回复请求DTO
     */
    public static class TaskDiscussionReplyRequest {
        @NotNull(message = "讨论ID不能为空")
        private Long discussionId; // 讨论ID
        
        @NotBlank(message = "回复内容不能为空")
        private String content; // 回复内容
        
        private Long parentReplyId; // 父回复ID（可选）
        
        // Getters and Setters
        public Long getDiscussionId() { return discussionId; }
        public void setDiscussionId(Long discussionId) { this.discussionId = discussionId; }
        public String getContent() { return content; }
        public void setContent(String content) { this.content = content; }
        public Long getParentReplyId() { return parentReplyId; }
        public void setParentReplyId(Long parentReplyId) { this.parentReplyId = parentReplyId; }
    }

    /**
     * 任务资源响应DTO
     */
    public static class TaskResourceResponse {
        private Long resourceId; // 资源ID
        private String resourceName; // 资源名称
        private String resourceType; // 资源类型
        private String fileUrl; // 文件URL
        private String fileName; // 文件名
        private Long fileSize; // 文件大小
        private String description; // 资源描述
        private LocalDateTime uploadTime; // 上传时间
        private String uploaderName; // 上传者姓名
        private Integer downloadCount; // 下载次数
        
        // Getters and Setters
        public Long getResourceId() { return resourceId; }
        public void setResourceId(Long resourceId) { this.resourceId = resourceId; }
        public String getResourceName() { return resourceName; }
        public void setResourceName(String resourceName) { this.resourceName = resourceName; }
        public String getResourceType() { return resourceType; }
        public void setResourceType(String resourceType) { this.resourceType = resourceType; }
        public String getFileUrl() { return fileUrl; }
        public void setFileUrl(String fileUrl) { this.fileUrl = fileUrl; }
        public String getFileName() { return fileName; }
        public void setFileName(String fileName) { this.fileName = fileName; }
        public Long getFileSize() { return fileSize; }
        public void setFileSize(Long fileSize) { this.fileSize = fileSize; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
        public LocalDateTime getUploadTime() { return uploadTime; }
        public void setUploadTime(LocalDateTime uploadTime) { this.uploadTime = uploadTime; }
        public String getUploaderName() { return uploaderName; }
        public void setUploaderName(String uploaderName) { this.uploaderName = uploaderName; }
        public Integer getDownloadCount() { return downloadCount; }
        public void setDownloadCount(Integer downloadCount) { this.downloadCount = downloadCount; }
    }

    /**
     * 资源下载响应DTO
     */
    public static class ResourceDownloadResponse {
        private String downloadUrl; // 下载URL
        private String fileName; // 文件名
        private Long fileSize; // 文件大小
        private String contentType; // 内容类型
        private LocalDateTime expiryTime; // 过期时间
        private String downloadToken; // 下载令牌
        
        // Getters and Setters
        public String getDownloadUrl() { return downloadUrl; }
        public void setDownloadUrl(String downloadUrl) { this.downloadUrl = downloadUrl; }
        public String getFileName() { return fileName; }
        public void setFileName(String fileName) { this.fileName = fileName; }
        public Long getFileSize() { return fileSize; }
        public void setFileSize(Long fileSize) { this.fileSize = fileSize; }
        public String getContentType() { return contentType; }
        public void setContentType(String contentType) { this.contentType = contentType; }
        public LocalDateTime getExpiryTime() { return expiryTime; }
        public void setExpiryTime(LocalDateTime expiryTime) { this.expiryTime = expiryTime; }
        public String getDownloadToken() { return downloadToken; }
        public void setDownloadToken(String downloadToken) { this.downloadToken = downloadToken; }
    }

    /**
     * 任务模板响应DTO
     */
    public static class TaskTemplateResponse {
        private Long templateId; // 模板ID
        private String templateName; // 模板名称
        private String description; // 模板描述
        private String taskType; // 任务类型
        private String difficulty; // 难度级别
        private Integer timeLimit; // 时间限制
        private BigDecimal totalScore; // 总分
        private String instructions; // 任务说明
        private Map<String, Object> settings; // 模板设置
        private LocalDateTime createTime; // 创建时间
        private String creatorName; // 创建者姓名
        
        // Getters and Setters
        public Long getTemplateId() { return templateId; }
        public void setTemplateId(Long templateId) { this.templateId = templateId; }
        public String getTemplateName() { return templateName; }
        public void setTemplateName(String templateName) { this.templateName = templateName; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
        public String getTaskType() { return taskType; }
        public void setTaskType(String taskType) { this.taskType = taskType; }
        public String getDifficulty() { return difficulty; }
        public void setDifficulty(String difficulty) { this.difficulty = difficulty; }
        public Integer getTimeLimit() { return timeLimit; }
        public void setTimeLimit(Integer timeLimit) { this.timeLimit = timeLimit; }
        public BigDecimal getTotalScore() { return totalScore; }
        public void setTotalScore(BigDecimal totalScore) { this.totalScore = totalScore; }
        public String getInstructions() { return instructions; }
        public void setInstructions(String instructions) { this.instructions = instructions; }
        public Map<String, Object> getSettings() { return settings; }
        public void setSettings(Map<String, Object> settings) { this.settings = settings; }
        public LocalDateTime getCreateTime() { return createTime; }
        public void setCreateTime(LocalDateTime createTime) { this.createTime = createTime; }
        public String getCreatorName() { return creatorName; }
        public void setCreatorName(String creatorName) { this.creatorName = creatorName; }
    }

    /**
     * 评分标准响应DTO
     */
    public static class GradingCriteriaResponse {
        private Long criteriaId; // 标准ID
        private String criteriaName; // 标准名称
        private String description; // 标准描述
        private BigDecimal maxScore; // 最高分数
        private BigDecimal weight; // 权重
        private String gradingType; // 评分类型：NUMERIC, LETTER, PASS_FAIL
        private List<String> gradingLevels; // 评分等级
        private Map<String, Object> rubric; // 评分细则
        
        // Getters and Setters
        public Long getCriteriaId() { return criteriaId; }
        public void setCriteriaId(Long criteriaId) { this.criteriaId = criteriaId; }
        public String getCriteriaName() { return criteriaName; }
        public void setCriteriaName(String criteriaName) { this.criteriaName = criteriaName; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
        public BigDecimal getMaxScore() { return maxScore; }
        public void setMaxScore(BigDecimal maxScore) { this.maxScore = maxScore; }
        public BigDecimal getWeight() { return weight; }
        public void setWeight(BigDecimal weight) { this.weight = weight; }
        public String getGradingType() { return gradingType; }
        public void setGradingType(String gradingType) { this.gradingType = gradingType; }
        public List<String> getGradingLevels() { return gradingLevels; }
        public void setGradingLevels(List<String> gradingLevels) { this.gradingLevels = gradingLevels; }
        public Map<String, Object> getRubric() { return rubric; }
        public void setRubric(Map<String, Object> rubric) { this.rubric = rubric; }
    }

    /**
     * 任务进度响应DTO
     */
    public static class TaskProgressResponse {
        private Long taskId; // 任务ID
        private Long studentId; // 学生ID
        private String studentName; // 学生姓名
        private String status; // 状态：NOT_STARTED, IN_PROGRESS, SUBMITTED, GRADED
        private Integer progressPercentage; // 进度百分比
        private LocalDateTime startTime; // 开始时间
        private LocalDateTime lastUpdateTime; // 最后更新时间
        private LocalDateTime submitTime; // 提交时间
        private Integer timeSpent; // 已用时间（分钟）
        private Integer remainingTime; // 剩余时间（分钟）
        private Map<String, Object> progressDetails; // 进度详情
        
        // Getters and Setters
        public Long getTaskId() { return taskId; }
        public void setTaskId(Long taskId) { this.taskId = taskId; }
        public Long getStudentId() { return studentId; }
        public void setStudentId(Long studentId) { this.studentId = studentId; }
        public String getStudentName() { return studentName; }
        public void setStudentName(String studentName) { this.studentName = studentName; }
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
        public Integer getProgressPercentage() { return progressPercentage; }
        public void setProgressPercentage(Integer progressPercentage) { this.progressPercentage = progressPercentage; }
        public LocalDateTime getStartTime() { return startTime; }
        public void setStartTime(LocalDateTime startTime) { this.startTime = startTime; }
        public LocalDateTime getLastUpdateTime() { return lastUpdateTime; }
        public void setLastUpdateTime(LocalDateTime lastUpdateTime) { this.lastUpdateTime = lastUpdateTime; }
        public LocalDateTime getSubmitTime() { return submitTime; }
        public void setSubmitTime(LocalDateTime submitTime) { this.submitTime = submitTime; }
        public Integer getTimeSpent() { return timeSpent; }
        public void setTimeSpent(Integer timeSpent) { this.timeSpent = timeSpent; }
        public Integer getRemainingTime() { return remainingTime; }
        public void setRemainingTime(Integer remainingTime) { this.remainingTime = remainingTime; }
        public Map<String, Object> getProgressDetails() { return progressDetails; }
        public void setProgressDetails(Map<String, Object> progressDetails) { this.progressDetails = progressDetails; }
    }

    /**
     * 任务进度更新请求DTO
     */
    public static class TaskProgressUpdateRequest {
        @NotNull(message = "任务ID不能为空")
        private Long taskId; // 任务ID
        
        private String status; // 状态
        private Integer progressPercentage; // 进度百分比
        private Map<String, Object> progressData; // 进度数据
        private String notes; // 备注
        
        // Getters and Setters
        public Long getTaskId() { return taskId; }
        public void setTaskId(Long taskId) { this.taskId = taskId; }
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
        public Integer getProgressPercentage() { return progressPercentage; }
        public void setProgressPercentage(Integer progressPercentage) { this.progressPercentage = progressPercentage; }
        public Map<String, Object> getProgressData() { return progressData; }
        public void setProgressData(Map<String, Object> progressData) { this.progressData = progressData; }
        public String getNotes() { return notes; }
        public void setNotes(String notes) { this.notes = notes; }
    }

    /**
     * 任务建议响应DTO
     */
    public static class TaskSuggestionResponse {
        private Long suggestionId; // 建议ID
        private String suggestionType; // 建议类型：IMPROVEMENT, DIFFICULTY_ADJUSTMENT, TIME_EXTENSION
        private String title; // 建议标题
        private String content; // 建议内容
        private String priority; // 优先级：HIGH, MEDIUM, LOW
        private String status; // 状态：PENDING, ACCEPTED, REJECTED
        private LocalDateTime createTime; // 创建时间
        private String creatorName; // 创建者姓名
        
        // Getters and Setters
        public Long getSuggestionId() { return suggestionId; }
        public void setSuggestionId(Long suggestionId) { this.suggestionId = suggestionId; }
        public String getSuggestionType() { return suggestionType; }
        public void setSuggestionType(String suggestionType) { this.suggestionType = suggestionType; }
        public String getTitle() { return title; }
        public void setTitle(String title) { this.title = title; }
        public String getContent() { return content; }
        public void setContent(String content) { this.content = content; }
        public String getPriority() { return priority; }
        public void setPriority(String priority) { this.priority = priority; }
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
        public LocalDateTime getCreateTime() { return createTime; }
        public void setCreateTime(LocalDateTime createTime) { this.createTime = createTime; }
        public String getCreatorName() { return creatorName; }
        public void setCreatorName(String creatorName) { this.creatorName = creatorName; }
    }

    /**
     * 推荐材料响应DTO
     */
    public static class RecommendedMaterialResponse {
        private Long materialId; // 材料ID
        private String materialName; // 材料名称
        private String materialType; // 材料类型：BOOK, ARTICLE, VIDEO, WEBSITE
        private String description; // 材料描述
        private String url; // 材料链接
        private String author; // 作者
        private String difficulty; // 难度级别
        private Integer estimatedTime; // 预计学习时间（分钟）
        private BigDecimal relevanceScore; // 相关性评分
        private String recommendationReason; // 推荐理由
        
        // Getters and Setters
        public Long getMaterialId() { return materialId; }
        public void setMaterialId(Long materialId) { this.materialId = materialId; }
        public String getMaterialName() { return materialName; }
        public void setMaterialName(String materialName) { this.materialName = materialName; }
        public String getMaterialType() { return materialType; }
        public void setMaterialType(String materialType) { this.materialType = materialType; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
        public String getUrl() { return url; }
        public void setUrl(String url) { this.url = url; }
        public String getAuthor() { return author; }
        public void setAuthor(String author) { this.author = author; }
        public String getDifficulty() { return difficulty; }
        public void setDifficulty(String difficulty) { this.difficulty = difficulty; }
        public Integer getEstimatedTime() { return estimatedTime; }
        public void setEstimatedTime(Integer estimatedTime) { this.estimatedTime = estimatedTime; }
        public BigDecimal getRelevanceScore() { return relevanceScore; }
        public void setRelevanceScore(BigDecimal relevanceScore) { this.relevanceScore = relevanceScore; }
        public String getRecommendationReason() { return recommendationReason; }
        public void setRecommendationReason(String recommendationReason) { this.recommendationReason = recommendationReason; }
    }

    /**
     * 任务完成报告响应DTO
     */
    public static class TaskCompletionReportResponse {
        private Long taskId; // 任务ID
        private String taskTitle; // 任务标题
        private Integer totalStudents; // 总学生数
        private Integer completedStudents; // 完成学生数
        private Integer submittedStudents; // 提交学生数
        private Integer gradedStudents; // 已评分学生数
        private BigDecimal averageScore; // 平均分
        private BigDecimal highestScore; // 最高分
        private BigDecimal lowestScore; // 最低分
        private Integer averageTimeSpent; // 平均用时（分钟）
        private BigDecimal completionRate; // 完成率
        private BigDecimal submissionRate; // 提交率
        private LocalDateTime reportGenerateTime; // 报告生成时间
        private Map<String, Object> statisticsData; // 统计数据
        
        // Getters and Setters
        public Long getTaskId() { return taskId; }
        public void setTaskId(Long taskId) { this.taskId = taskId; }
        public String getTaskTitle() { return taskTitle; }
        public void setTaskTitle(String taskTitle) { this.taskTitle = taskTitle; }
        public Integer getTotalStudents() { return totalStudents; }
        public void setTotalStudents(Integer totalStudents) { this.totalStudents = totalStudents; }
        public Integer getCompletedStudents() { return completedStudents; }
        public void setCompletedStudents(Integer completedStudents) { this.completedStudents = completedStudents; }
        public Integer getSubmittedStudents() { return submittedStudents; }
        public void setSubmittedStudents(Integer submittedStudents) { this.submittedStudents = submittedStudents; }
        public Integer getGradedStudents() { return gradedStudents; }
        public void setGradedStudents(Integer gradedStudents) { this.gradedStudents = gradedStudents; }
        public BigDecimal getAverageScore() { return averageScore; }
        public void setAverageScore(BigDecimal averageScore) { this.averageScore = averageScore; }
        public BigDecimal getHighestScore() { return highestScore; }
        public void setHighestScore(BigDecimal highestScore) { this.highestScore = highestScore; }
        public BigDecimal getLowestScore() { return lowestScore; }
        public void setLowestScore(BigDecimal lowestScore) { this.lowestScore = lowestScore; }
        public Integer getAverageTimeSpent() { return averageTimeSpent; }
        public void setAverageTimeSpent(Integer averageTimeSpent) { this.averageTimeSpent = averageTimeSpent; }
        public BigDecimal getCompletionRate() { return completionRate; }
        public void setCompletionRate(BigDecimal completionRate) { this.completionRate = completionRate; }
        public BigDecimal getSubmissionRate() { return submissionRate; }
        public void setSubmissionRate(BigDecimal submissionRate) { this.submissionRate = submissionRate; }
        public LocalDateTime getReportGenerateTime() { return reportGenerateTime; }
        public void setReportGenerateTime(LocalDateTime reportGenerateTime) { this.reportGenerateTime = reportGenerateTime; }
        public Map<String, Object> getStatisticsData() { return statisticsData; }
        public void setStatisticsData(Map<String, Object> statisticsData) { this.statisticsData = statisticsData; }
    }

    /**
     * 导出响应DTO
     */
    public static class ExportResponse {
        private String exportId;
        private String fileName;
        private String downloadUrl;
        private String fileFormat;
        private Long fileSize;
        private LocalDateTime expiryTime;
        
        // Getters and Setters
        public String getExportId() { return exportId; }
        public void setExportId(String exportId) { this.exportId = exportId; }
        public String getFileName() { return fileName; }
        public void setFileName(String fileName) { this.fileName = fileName; }
        public String getDownloadUrl() { return downloadUrl; }
        public void setDownloadUrl(String downloadUrl) { this.downloadUrl = downloadUrl; }
        public String getFileFormat() { return fileFormat; }
        public void setFileFormat(String fileFormat) { this.fileFormat = fileFormat; }
        public Long getFileSize() { return fileSize; }
        public void setFileSize(Long fileSize) { this.fileSize = fileSize; }
        public LocalDateTime getExpiryTime() { return expiryTime; }
        public void setExpiryTime(LocalDateTime expiryTime) { this.expiryTime = expiryTime; }
    }

    /**
     * 任务数据导出请求DTO
     */
    public static class TaskDataExportRequest {
        private String exportType; // TASKS, SUBMISSIONS, GRADES, ALL
        private String fileFormat; // CSV, EXCEL, PDF
        private LocalDateTime startDate;
        private LocalDateTime endDate;
        private List<Long> taskIds;
        private List<Long> courseIds;
        
        // Getters and Setters
        public String getExportType() { return exportType; }
        public void setExportType(String exportType) { this.exportType = exportType; }
        public String getFileFormat() { return fileFormat; }
        public void setFileFormat(String fileFormat) { this.fileFormat = fileFormat; }
        public LocalDateTime getStartDate() { return startDate; }
        public void setStartDate(LocalDateTime startDate) { this.startDate = startDate; }
        public LocalDateTime getEndDate() { return endDate; }
        public void setEndDate(LocalDateTime endDate) { this.endDate = endDate; }
        public List<Long> getTaskIds() { return taskIds; }
        public void setTaskIds(List<Long> taskIds) { this.taskIds = taskIds; }
        public List<Long> getCourseIds() { return courseIds; }
        public void setCourseIds(List<Long> courseIds) { this.courseIds = courseIds; }
    }

    /**
     * 提交统计响应DTO
     */
    public static class SubmissionStatisticsResponse {
        private Long taskId;
        private Integer totalSubmissions;
        private Integer gradedSubmissions;
        private Integer pendingSubmissions;
        private BigDecimal averageScore;
        private BigDecimal highestScore;
        private BigDecimal lowestScore;
        private LocalDateTime lastSubmissionTime;
        
        // Getters and Setters
        public Long getTaskId() { return taskId; }
        public void setTaskId(Long taskId) { this.taskId = taskId; }
        public Integer getTotalSubmissions() { return totalSubmissions; }
        public void setTotalSubmissions(Integer totalSubmissions) { this.totalSubmissions = totalSubmissions; }
        public Integer getGradedSubmissions() { return gradedSubmissions; }
        public void setGradedSubmissions(Integer gradedSubmissions) { this.gradedSubmissions = gradedSubmissions; }
        public Integer getPendingSubmissions() { return pendingSubmissions; }
        public void setPendingSubmissions(Integer pendingSubmissions) { this.pendingSubmissions = pendingSubmissions; }
        public BigDecimal getAverageScore() { return averageScore; }
        public void setAverageScore(BigDecimal averageScore) { this.averageScore = averageScore; }
        public BigDecimal getHighestScore() { return highestScore; }
        public void setHighestScore(BigDecimal highestScore) { this.highestScore = highestScore; }
        public BigDecimal getLowestScore() { return lowestScore; }
        public void setLowestScore(BigDecimal lowestScore) { this.lowestScore = lowestScore; }
        public LocalDateTime getLastSubmissionTime() { return lastSubmissionTime; }
        public void setLastSubmissionTime(LocalDateTime lastSubmissionTime) { this.lastSubmissionTime = lastSubmissionTime; }
    }

    /**
     * 任务分析响应DTO
     */
    public static class TaskAnalysisResponse {
        private Long taskId;
        private String taskTitle;
        private String difficultyLevel;
        private BigDecimal completionRate;
        private BigDecimal averageScore;
        private Integer averageTimeSpent;
        private Map<String, Object> performanceMetrics;
        private List<String> recommendations;
        
        // Getters and Setters
        public Long getTaskId() { return taskId; }
        public void setTaskId(Long taskId) { this.taskId = taskId; }
        public String getTaskTitle() { return taskTitle; }
        public void setTaskTitle(String taskTitle) { this.taskTitle = taskTitle; }
        public String getDifficultyLevel() { return difficultyLevel; }
        public void setDifficultyLevel(String difficultyLevel) { this.difficultyLevel = difficultyLevel; }
        public BigDecimal getCompletionRate() { return completionRate; }
        public void setCompletionRate(BigDecimal completionRate) { this.completionRate = completionRate; }
        public BigDecimal getAverageScore() { return averageScore; }
        public void setAverageScore(BigDecimal averageScore) { this.averageScore = averageScore; }
        public Integer getAverageTimeSpent() { return averageTimeSpent; }
        public void setAverageTimeSpent(Integer averageTimeSpent) { this.averageTimeSpent = averageTimeSpent; }
        public Map<String, Object> getPerformanceMetrics() { return performanceMetrics; }
        public void setPerformanceMetrics(Map<String, Object> performanceMetrics) { this.performanceMetrics = performanceMetrics; }
        public List<String> getRecommendations() { return recommendations; }
        public void setRecommendations(List<String> recommendations) { this.recommendations = recommendations; }
    }

    /**
     * 评分标准DTO
     */
    public static class GradingCriteria {
        private String criteriaName;
        private String description;
        private BigDecimal maxScore;
        private BigDecimal weight;
        private String gradingType;
        private List<String> gradingLevels;
        
        // Getters and Setters
        public String getCriteriaName() { return criteriaName; }
        public void setCriteriaName(String criteriaName) { this.criteriaName = criteriaName; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
        public BigDecimal getMaxScore() { return maxScore; }
        public void setMaxScore(BigDecimal maxScore) { this.maxScore = maxScore; }
        public BigDecimal getWeight() { return weight; }
        public void setWeight(BigDecimal weight) { this.weight = weight; }
        public String getGradingType() { return gradingType; }
        public void setGradingType(String gradingType) { this.gradingType = gradingType; }
        public List<String> getGradingLevels() { return gradingLevels; }
        public void setGradingLevels(List<String> gradingLevels) { this.gradingLevels = gradingLevels; }
    }

    /**
     * 难度分析响应DTO
     */
    public static class DifficultyAnalysisResponse {
        private String difficultyLevel;
        private Integer taskCount;
        private BigDecimal averageScore;
        private BigDecimal completionRate;
        private Integer averageTimeSpent;
        private Map<String, Object> difficultyMetrics;
        
        // Getters and Setters
        public String getDifficultyLevel() { return difficultyLevel; }
        public void setDifficultyLevel(String difficultyLevel) { this.difficultyLevel = difficultyLevel; }
        public Integer getTaskCount() { return taskCount; }
        public void setTaskCount(Integer taskCount) { this.taskCount = taskCount; }
        public BigDecimal getAverageScore() { return averageScore; }
        public void setAverageScore(BigDecimal averageScore) { this.averageScore = averageScore; }
        public BigDecimal getCompletionRate() { return completionRate; }
        public void setCompletionRate(BigDecimal completionRate) { this.completionRate = completionRate; }
        public Integer getAverageTimeSpent() { return averageTimeSpent; }
        public void setAverageTimeSpent(Integer averageTimeSpent) { this.averageTimeSpent = averageTimeSpent; }
        public Map<String, Object> getDifficultyMetrics() { return difficultyMetrics; }
        public void setDifficultyMetrics(Map<String, Object> difficultyMetrics) { this.difficultyMetrics = difficultyMetrics; }
    }
}