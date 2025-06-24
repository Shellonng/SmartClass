package com.education.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

/**
 * 课程相关DTO扩展7 - 包含课程日历、学习提醒、学习报告等相关类
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
public class CourseDTOExtension7 {

    /**
     * 课程日历响应DTO
     */
    public static class CourseCalendarResponse {
        private Long courseId;
        private String courseName;
        private List<CalendarEvent> events;
        private LocalDate startDate;
        private LocalDate endDate;
        
        // Getters and Setters
        public Long getCourseId() { return courseId; }
        public void setCourseId(Long courseId) { this.courseId = courseId; }
        public String getCourseName() { return courseName; }
        public void setCourseName(String courseName) { this.courseName = courseName; }
        public List<CalendarEvent> getEvents() { return events; }
        public void setEvents(List<CalendarEvent> events) { this.events = events; }
        public LocalDate getStartDate() { return startDate; }
        public void setStartDate(LocalDate startDate) { this.startDate = startDate; }
        public LocalDate getEndDate() { return endDate; }
        public void setEndDate(LocalDate endDate) { this.endDate = endDate; }
    }

    /**
     * 日历事件DTO
     */
    public static class CalendarEvent {
        private Long eventId;
        private String title;
        private String description;
        private LocalDateTime startTime;
        private LocalDateTime endTime;
        private String eventType;
        private String location;
        private Boolean isReminder;
        private String status;
        
        // Getters and Setters
        public Long getEventId() { return eventId; }
        public void setEventId(Long eventId) { this.eventId = eventId; }
        public String getTitle() { return title; }
        public void setTitle(String title) { this.title = title; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
        public LocalDateTime getStartTime() { return startTime; }
        public void setStartTime(LocalDateTime startTime) { this.startTime = startTime; }
        public LocalDateTime getEndTime() { return endTime; }
        public void setEndTime(LocalDateTime endTime) { this.endTime = endTime; }
        public String getEventType() { return eventType; }
        public void setEventType(String eventType) { this.eventType = eventType; }
        public String getLocation() { return location; }
        public void setLocation(String location) { this.location = location; }
        public Boolean getIsReminder() { return isReminder; }
        public void setIsReminder(Boolean isReminder) { this.isReminder = isReminder; }
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
    }

    /**
     * 学习提醒响应DTO
     */
    public static class StudyReminderResponse {
        private Long reminderId;
        private Long courseId;
        private String courseName;
        private String reminderType;
        private String title;
        private String content;
        private LocalDateTime reminderTime;
        private Boolean isEnabled;
        private String frequency;
        private LocalDateTime createTime;
        private LocalDateTime lastSentTime;
        
        // Getters and Setters
        public Long getReminderId() { return reminderId; }
        public void setReminderId(Long reminderId) { this.reminderId = reminderId; }
        public Long getCourseId() { return courseId; }
        public void setCourseId(Long courseId) { this.courseId = courseId; }
        public String getCourseName() { return courseName; }
        public void setCourseName(String courseName) { this.courseName = courseName; }
        public String getReminderType() { return reminderType; }
        public void setReminderType(String reminderType) { this.reminderType = reminderType; }
        public String getTitle() { return title; }
        public void setTitle(String title) { this.title = title; }
        public String getContent() { return content; }
        public void setContent(String content) { this.content = content; }
        public LocalDateTime getReminderTime() { return reminderTime; }
        public void setReminderTime(LocalDateTime reminderTime) { this.reminderTime = reminderTime; }
        public Boolean getIsEnabled() { return isEnabled; }
        public void setIsEnabled(Boolean isEnabled) { this.isEnabled = isEnabled; }
        public String getFrequency() { return frequency; }
        public void setFrequency(String frequency) { this.frequency = frequency; }
        public LocalDateTime getCreateTime() { return createTime; }
        public void setCreateTime(LocalDateTime createTime) { this.createTime = createTime; }
        public LocalDateTime getLastSentTime() { return lastSentTime; }
        public void setLastSentTime(LocalDateTime lastSentTime) { this.lastSentTime = lastSentTime; }
    }

    /**
     * 学习提醒请求DTO
     */
    public static class StudyReminderRequest {
        @NotNull(message = "课程ID不能为空")
        private Long courseId;
        @NotBlank(message = "提醒类型不能为空")
        private String reminderType;
        @NotBlank(message = "提醒标题不能为空")
        private String title;
        private String content;
        @NotNull(message = "提醒时间不能为空")
        private LocalDateTime reminderTime;
        private Boolean isEnabled;
        private String frequency;
        
        // Getters and Setters
        public Long getCourseId() { return courseId; }
        public void setCourseId(Long courseId) { this.courseId = courseId; }
        public String getReminderType() { return reminderType; }
        public void setReminderType(String reminderType) { this.reminderType = reminderType; }
        public String getTitle() { return title; }
        public void setTitle(String title) { this.title = title; }
        public String getContent() { return content; }
        public void setContent(String content) { this.content = content; }
        public LocalDateTime getReminderTime() { return reminderTime; }
        public void setReminderTime(LocalDateTime reminderTime) { this.reminderTime = reminderTime; }
        public Boolean getIsEnabled() { return isEnabled; }
        public void setIsEnabled(Boolean isEnabled) { this.isEnabled = isEnabled; }
        public String getFrequency() { return frequency; }
        public void setFrequency(String frequency) { this.frequency = frequency; }
    }

    /**
     * 学习报告响应DTO
     */
    public static class LearningReportResponse {
        private Long reportId;
        private Long studentId;
        private String reportType;
        private String reportPeriod;
        private LocalDate startDate;
        private LocalDate endDate;
        private Integer totalStudyTime;
        private Integer completedCourses;
        private Integer ongoingCourses;
        private Double averageScore;
        private Integer totalAssignments;
        private Integer completedAssignments;
        private Map<String, Object> detailedStats;
        private List<CourseProgressSummary> courseProgress;
        private LocalDateTime generateTime;
        
        // Getters and Setters
        public Long getReportId() { return reportId; }
        public void setReportId(Long reportId) { this.reportId = reportId; }
        public Long getStudentId() { return studentId; }
        public void setStudentId(Long studentId) { this.studentId = studentId; }
        public String getReportType() { return reportType; }
        public void setReportType(String reportType) { this.reportType = reportType; }
        public String getReportPeriod() { return reportPeriod; }
        public void setReportPeriod(String reportPeriod) { this.reportPeriod = reportPeriod; }
        public LocalDate getStartDate() { return startDate; }
        public void setStartDate(LocalDate startDate) { this.startDate = startDate; }
        public LocalDate getEndDate() { return endDate; }
        public void setEndDate(LocalDate endDate) { this.endDate = endDate; }
        public Integer getTotalStudyTime() { return totalStudyTime; }
        public void setTotalStudyTime(Integer totalStudyTime) { this.totalStudyTime = totalStudyTime; }
        public Integer getCompletedCourses() { return completedCourses; }
        public void setCompletedCourses(Integer completedCourses) { this.completedCourses = completedCourses; }
        public Integer getOngoingCourses() { return ongoingCourses; }
        public void setOngoingCourses(Integer ongoingCourses) { this.ongoingCourses = ongoingCourses; }
        public Double getAverageScore() { return averageScore; }
        public void setAverageScore(Double averageScore) { this.averageScore = averageScore; }
        public Integer getTotalAssignments() { return totalAssignments; }
        public void setTotalAssignments(Integer totalAssignments) { this.totalAssignments = totalAssignments; }
        public Integer getCompletedAssignments() { return completedAssignments; }
        public void setCompletedAssignments(Integer completedAssignments) { this.completedAssignments = completedAssignments; }
        public Map<String, Object> getDetailedStats() { return detailedStats; }
        public void setDetailedStats(Map<String, Object> detailedStats) { this.detailedStats = detailedStats; }
        public List<CourseProgressSummary> getCourseProgress() { return courseProgress; }
        public void setCourseProgress(List<CourseProgressSummary> courseProgress) { this.courseProgress = courseProgress; }
        public LocalDateTime getGenerateTime() { return generateTime; }
        public void setGenerateTime(LocalDateTime generateTime) { this.generateTime = generateTime; }
    }

    /**
     * 课程进度摘要DTO
     */
    public static class CourseProgressSummary {
        private Long courseId;
        private String courseName;
        private Double progressPercentage;
        private Integer studyTime;
        private Double currentScore;
        private String status;
        private LocalDateTime lastAccessTime;
        
        // Getters and Setters
        public Long getCourseId() { return courseId; }
        public void setCourseId(Long courseId) { this.courseId = courseId; }
        public String getCourseName() { return courseName; }
        public void setCourseName(String courseName) { this.courseName = courseName; }
        public Double getProgressPercentage() { return progressPercentage; }
        public void setProgressPercentage(Double progressPercentage) { this.progressPercentage = progressPercentage; }
        public Integer getStudyTime() { return studyTime; }
        public void setStudyTime(Integer studyTime) { this.studyTime = studyTime; }
        public Double getCurrentScore() { return currentScore; }
        public void setCurrentScore(Double currentScore) { this.currentScore = currentScore; }
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
        public LocalDateTime getLastAccessTime() { return lastAccessTime; }
        public void setLastAccessTime(LocalDateTime lastAccessTime) { this.lastAccessTime = lastAccessTime; }
    }

    /**
     * 学习数据导出请求DTO
     */
    public static class LearningDataExportRequest {
        @NotBlank(message = "导出类型不能为空")
        private String exportType;
        @NotBlank(message = "数据格式不能为空")
        private String dataFormat;
        private LocalDate startDate;
        private LocalDate endDate;
        private List<Long> courseIds;
        private List<String> dataTypes;
        private Boolean includePersonalInfo;
        private Boolean includeStatistics;
        
        // Getters and Setters
        public String getExportType() { return exportType; }
        public void setExportType(String exportType) { this.exportType = exportType; }
        public String getDataFormat() { return dataFormat; }
        public void setDataFormat(String dataFormat) { this.dataFormat = dataFormat; }
        public LocalDate getStartDate() { return startDate; }
        public void setStartDate(LocalDate startDate) { this.startDate = startDate; }
        public LocalDate getEndDate() { return endDate; }
        public void setEndDate(LocalDate endDate) { this.endDate = endDate; }
        public List<Long> getCourseIds() { return courseIds; }
        public void setCourseIds(List<Long> courseIds) { this.courseIds = courseIds; }
        public List<String> getDataTypes() { return dataTypes; }
        public void setDataTypes(List<String> dataTypes) { this.dataTypes = dataTypes; }
        public Boolean getIncludePersonalInfo() { return includePersonalInfo; }
        public void setIncludePersonalInfo(Boolean includePersonalInfo) { this.includePersonalInfo = includePersonalInfo; }
        public Boolean getIncludeStatistics() { return includeStatistics; }
        public void setIncludeStatistics(Boolean includeStatistics) { this.includeStatistics = includeStatistics; }
    }

    /**
     * 数据导出响应DTO
     */
    public static class DataExportResponse {
        private String exportId;
        private String fileName;
        private String downloadUrl;
        private String status;
        private Long fileSize;
        private LocalDateTime exportTime;
        private LocalDateTime expiryTime;
        private String errorMessage;
        
        // Getters and Setters
        public String getExportId() { return exportId; }
        public void setExportId(String exportId) { this.exportId = exportId; }
        public String getFileName() { return fileName; }
        public void setFileName(String fileName) { this.fileName = fileName; }
        public String getDownloadUrl() { return downloadUrl; }
        public void setDownloadUrl(String downloadUrl) { this.downloadUrl = downloadUrl; }
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
        public Long getFileSize() { return fileSize; }
        public void setFileSize(Long fileSize) { this.fileSize = fileSize; }
        public LocalDateTime getExportTime() { return exportTime; }
        public void setExportTime(LocalDateTime exportTime) { this.exportTime = exportTime; }
        public LocalDateTime getExpiryTime() { return expiryTime; }
        public void setExpiryTime(LocalDateTime expiryTime) { this.expiryTime = expiryTime; }
        public String getErrorMessage() { return errorMessage; }
        public void setErrorMessage(String errorMessage) { this.errorMessage = errorMessage; }
    }
}