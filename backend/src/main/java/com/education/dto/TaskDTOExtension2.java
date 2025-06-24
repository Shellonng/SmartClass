package com.education.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;
import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

/**
 * 任务相关DTO扩展类2
 * 包含更多任务相关的DTO定义
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
public class TaskDTOExtension2 {

    /**
     * 任务进度响应DTO
     */
    public static class TaskProgressResponse {
        private Long taskId;
        private String taskTitle;
        private Integer totalSteps;
        private Integer completedSteps;
        private Double progressPercentage;
        private String currentStep;
        private LocalDateTime lastUpdateTime;
        
        // Getters and Setters
        public Long getTaskId() { return taskId; }
        public void setTaskId(Long taskId) { this.taskId = taskId; }
        public String getTaskTitle() { return taskTitle; }
        public void setTaskTitle(String taskTitle) { this.taskTitle = taskTitle; }
        public Integer getTotalSteps() { return totalSteps; }
        public void setTotalSteps(Integer totalSteps) { this.totalSteps = totalSteps; }
        public Integer getCompletedSteps() { return completedSteps; }
        public void setCompletedSteps(Integer completedSteps) { this.completedSteps = completedSteps; }
        public Double getProgressPercentage() { return progressPercentage; }
        public void setProgressPercentage(Double progressPercentage) { this.progressPercentage = progressPercentage; }
        public String getCurrentStep() { return currentStep; }
        public void setCurrentStep(String currentStep) { this.currentStep = currentStep; }
        public LocalDateTime getLastUpdateTime() { return lastUpdateTime; }
        public void setLastUpdateTime(LocalDateTime lastUpdateTime) { this.lastUpdateTime = lastUpdateTime; }
    }

    /**
     * 任务进度更新请求DTO
     */
    public static class TaskProgressUpdateRequest {
        @NotNull(message = "任务ID不能为空")
        private Long taskId;
        
        private Integer completedSteps;
        private String currentStep;
        private String notes;
        
        // Getters and Setters
        public Long getTaskId() { return taskId; }
        public void setTaskId(Long taskId) { this.taskId = taskId; }
        public Integer getCompletedSteps() { return completedSteps; }
        public void setCompletedSteps(Integer completedSteps) { this.completedSteps = completedSteps; }
        public String getCurrentStep() { return currentStep; }
        public void setCurrentStep(String currentStep) { this.currentStep = currentStep; }
        public String getNotes() { return notes; }
        public void setNotes(String notes) { this.notes = notes; }
    }

    /**
     * 任务建议响应DTO
     */
    public static class TaskSuggestionResponse {
        private Long taskId;
        private List<String> suggestions;
        private List<String> recommendedResources;
        private String difficultyAnalysis;
        private Integer estimatedTime;
        
        // Getters and Setters
        public Long getTaskId() { return taskId; }
        public void setTaskId(Long taskId) { this.taskId = taskId; }
        public List<String> getSuggestions() { return suggestions; }
        public void setSuggestions(List<String> suggestions) { this.suggestions = suggestions; }
        public List<String> getRecommendedResources() { return recommendedResources; }
        public void setRecommendedResources(List<String> recommendedResources) { this.recommendedResources = recommendedResources; }
        public String getDifficultyAnalysis() { return difficultyAnalysis; }
        public void setDifficultyAnalysis(String difficultyAnalysis) { this.difficultyAnalysis = difficultyAnalysis; }
        public Integer getEstimatedTime() { return estimatedTime; }
        public void setEstimatedTime(Integer estimatedTime) { this.estimatedTime = estimatedTime; }
    }

    /**
     * 推荐材料响应DTO
     */
    public static class RecommendedMaterialResponse {
        private Long materialId;
        private String title;
        private String description;
        private String materialType; // VIDEO, DOCUMENT, LINK, BOOK
        private String url;
        private String author;
        private Integer difficulty;
        private Integer estimatedTime;
        
        // Getters and Setters
        public Long getMaterialId() { return materialId; }
        public void setMaterialId(Long materialId) { this.materialId = materialId; }
        public String getTitle() { return title; }
        public void setTitle(String title) { this.title = title; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
        public String getMaterialType() { return materialType; }
        public void setMaterialType(String materialType) { this.materialType = materialType; }
        public String getUrl() { return url; }
        public void setUrl(String url) { this.url = url; }
        public String getAuthor() { return author; }
        public void setAuthor(String author) { this.author = author; }
        public Integer getDifficulty() { return difficulty; }
        public void setDifficulty(Integer difficulty) { this.difficulty = difficulty; }
        public Integer getEstimatedTime() { return estimatedTime; }
        public void setEstimatedTime(Integer estimatedTime) { this.estimatedTime = estimatedTime; }
    }

    /**
     * 任务完成报告响应DTO
     */
    public static class TaskCompletionReportResponse {
        private String timeRange;
        private Integer totalTasks;
        private Integer completedTasks;
        private Integer onTimeTasks;
        private Integer lateTasks;
        private Double averageScore;
        private Double completionRate;
        private List<TaskCompletionItem> taskDetails;
        
        // Getters and Setters
        public String getTimeRange() { return timeRange; }
        public void setTimeRange(String timeRange) { this.timeRange = timeRange; }
        public Integer getTotalTasks() { return totalTasks; }
        public void setTotalTasks(Integer totalTasks) { this.totalTasks = totalTasks; }
        public Integer getCompletedTasks() { return completedTasks; }
        public void setCompletedTasks(Integer completedTasks) { this.completedTasks = completedTasks; }
        public Integer getOnTimeTasks() { return onTimeTasks; }
        public void setOnTimeTasks(Integer onTimeTasks) { this.onTimeTasks = onTimeTasks; }
        public Integer getLateTasks() { return lateTasks; }
        public void setLateTasks(Integer lateTasks) { this.lateTasks = lateTasks; }
        public Double getAverageScore() { return averageScore; }
        public void setAverageScore(Double averageScore) { this.averageScore = averageScore; }
        public Double getCompletionRate() { return completionRate; }
        public void setCompletionRate(Double completionRate) { this.completionRate = completionRate; }
        public List<TaskCompletionItem> getTaskDetails() { return taskDetails; }
        public void setTaskDetails(List<TaskCompletionItem> taskDetails) { this.taskDetails = taskDetails; }
        
        public static class TaskCompletionItem {
            private Long taskId;
            private String taskTitle;
            private LocalDateTime dueDate;
            private LocalDateTime submitTime;
            private BigDecimal score;
            private String status;
            
            // Getters and Setters
            public Long getTaskId() { return taskId; }
            public void setTaskId(Long taskId) { this.taskId = taskId; }
            public String getTaskTitle() { return taskTitle; }
            public void setTaskTitle(String taskTitle) { this.taskTitle = taskTitle; }
            public LocalDateTime getDueDate() { return dueDate; }
            public void setDueDate(LocalDateTime dueDate) { this.dueDate = dueDate; }
            public LocalDateTime getSubmitTime() { return submitTime; }
            public void setSubmitTime(LocalDateTime submitTime) { this.submitTime = submitTime; }
            public BigDecimal getScore() { return score; }
            public void setScore(BigDecimal score) { this.score = score; }
            public String getStatus() { return status; }
            public void setStatus(String status) { this.status = status; }
        }
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
}