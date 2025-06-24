package com.education.dto;

import jakarta.validation.constraints.Email;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;
import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

/**
 * 课程相关DTO扩展类3
 * 包含课程日历、提醒、学习报告等相关的DTO定义
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
public class CourseDTOExtension3 {

    /**
     * 课程日历响应DTO
     */
    public static class CourseCalendarResponse {
        private String date;
        private List<CalendarEvent> events;
        private Integer totalEvents;
        private Boolean hasDeadlines;
        private Boolean hasLiveClasses;
        
        // Getters and Setters
        public String getDate() { return date; }
        public void setDate(String date) { this.date = date; }
        public List<CalendarEvent> getEvents() { return events; }
        public void setEvents(List<CalendarEvent> events) { this.events = events; }
        public Integer getTotalEvents() { return totalEvents; }
        public void setTotalEvents(Integer totalEvents) { this.totalEvents = totalEvents; }
        public Boolean getHasDeadlines() { return hasDeadlines; }
        public void setHasDeadlines(Boolean hasDeadlines) { this.hasDeadlines = hasDeadlines; }
        public Boolean getHasLiveClasses() { return hasLiveClasses; }
        public void setHasLiveClasses(Boolean hasLiveClasses) { this.hasLiveClasses = hasLiveClasses; }
        
        public static class CalendarEvent {
            private Long eventId;
            private String title;
            private String type;
            private LocalDateTime startTime;
            private LocalDateTime endTime;
            private String description;
            private String location;
            private Boolean isCompleted;
            private String priority;
            
            // Getters and Setters
            public Long getEventId() { return eventId; }
            public void setEventId(Long eventId) { this.eventId = eventId; }
            public String getTitle() { return title; }
            public void setTitle(String title) { this.title = title; }
            public String getType() { return type; }
            public void setType(String type) { this.type = type; }
            public LocalDateTime getStartTime() { return startTime; }
            public void setStartTime(LocalDateTime startTime) { this.startTime = startTime; }
            public LocalDateTime getEndTime() { return endTime; }
            public void setEndTime(LocalDateTime endTime) { this.endTime = endTime; }
            public String getDescription() { return description; }
            public void setDescription(String description) { this.description = description; }
            public String getLocation() { return location; }
            public void setLocation(String location) { this.location = location; }
            public Boolean getIsCompleted() { return isCompleted; }
            public void setIsCompleted(Boolean isCompleted) { this.isCompleted = isCompleted; }
            public String getPriority() { return priority; }
            public void setPriority(String priority) { this.priority = priority; }
        }
    }

    /**
     * 学习提醒响应DTO
     */
    public static class StudyReminderResponse {
        private Long reminderId;
        private String title;
        private String content;
        private String reminderType;
        private LocalDateTime reminderTime;
        private String frequency;
        private Boolean isActive;
        private Long courseId;
        private String courseName;
        private Long chapterId;
        private String chapterTitle;
        
        // Getters and Setters
        public Long getReminderId() { return reminderId; }
        public void setReminderId(Long reminderId) { this.reminderId = reminderId; }
        public String getTitle() { return title; }
        public void setTitle(String title) { this.title = title; }
        public String getContent() { return content; }
        public void setContent(String content) { this.content = content; }
        public String getReminderType() { return reminderType; }
        public void setReminderType(String reminderType) { this.reminderType = reminderType; }
        public LocalDateTime getReminderTime() { return reminderTime; }
        public void setReminderTime(LocalDateTime reminderTime) { this.reminderTime = reminderTime; }
        public String getFrequency() { return frequency; }
        public void setFrequency(String frequency) { this.frequency = frequency; }
        public Boolean getIsActive() { return isActive; }
        public void setIsActive(Boolean isActive) { this.isActive = isActive; }
        public Long getCourseId() { return courseId; }
        public void setCourseId(Long courseId) { this.courseId = courseId; }
        public String getCourseName() { return courseName; }
        public void setCourseName(String courseName) { this.courseName = courseName; }
        public Long getChapterId() { return chapterId; }
        public void setChapterId(Long chapterId) { this.chapterId = chapterId; }
        public String getChapterTitle() { return chapterTitle; }
        public void setChapterTitle(String chapterTitle) { this.chapterTitle = chapterTitle; }
    }

    /**
     * 学习提醒请求DTO
     */
    public static class StudyReminderRequest {
        @NotBlank(message = "提醒标题不能为空")
        private String title;
        
        private String content;
        
        @NotBlank(message = "提醒类型不能为空")
        private String reminderType;
        
        @NotNull(message = "提醒时间不能为空")
        private LocalDateTime reminderTime;
        
        private String frequency;
        private Boolean isActive;
        private Long courseId;
        private Long chapterId;
        
        // Getters and Setters
        public String getTitle() { return title; }
        public void setTitle(String title) { this.title = title; }
        public String getContent() { return content; }
        public void setContent(String content) { this.content = content; }
        public String getReminderType() { return reminderType; }
        public void setReminderType(String reminderType) { this.reminderType = reminderType; }
        public LocalDateTime getReminderTime() { return reminderTime; }
        public void setReminderTime(LocalDateTime reminderTime) { this.reminderTime = reminderTime; }
        public String getFrequency() { return frequency; }
        public void setFrequency(String frequency) { this.frequency = frequency; }
        public Boolean getIsActive() { return isActive; }
        public void setIsActive(Boolean isActive) { this.isActive = isActive; }
        public Long getCourseId() { return courseId; }
        public void setCourseId(Long courseId) { this.courseId = courseId; }
        public Long getChapterId() { return chapterId; }
        public void setChapterId(Long chapterId) { this.chapterId = chapterId; }
    }

    /**
     * 学习报告响应DTO
     */
    public static class LearningReportResponse {
        private String reportType;
        private String timeRange;
        private LocalDateTime generateTime;
        private LearningOverview overview;
        private List<CourseProgress> courseProgresses;
        private List<Achievement> achievements;
        private List<Recommendation> recommendations;
        private StudyPattern studyPattern;
        
        // Getters and Setters
        public String getReportType() { return reportType; }
        public void setReportType(String reportType) { this.reportType = reportType; }
        public String getTimeRange() { return timeRange; }
        public void setTimeRange(String timeRange) { this.timeRange = timeRange; }
        public LocalDateTime getGenerateTime() { return generateTime; }
        public void setGenerateTime(LocalDateTime generateTime) { this.generateTime = generateTime; }
        public LearningOverview getOverview() { return overview; }
        public void setOverview(LearningOverview overview) { this.overview = overview; }
        public List<CourseProgress> getCourseProgresses() { return courseProgresses; }
        public void setCourseProgresses(List<CourseProgress> courseProgresses) { this.courseProgresses = courseProgresses; }
        public List<Achievement> getAchievements() { return achievements; }
        public void setAchievements(List<Achievement> achievements) { this.achievements = achievements; }
        public List<Recommendation> getRecommendations() { return recommendations; }
        public void setRecommendations(List<Recommendation> recommendations) { this.recommendations = recommendations; }
        public StudyPattern getStudyPattern() { return studyPattern; }
        public void setStudyPattern(StudyPattern studyPattern) { this.studyPattern = studyPattern; }
        
        public static class LearningOverview {
            private Integer totalStudyTime;
            private Integer totalCourses;
            private Integer completedCourses;
            private Integer ongoingCourses;
            private Double averageScore;
            private Integer totalTasks;
            private Integer completedTasks;
            private Double taskCompletionRate;
            
            // Getters and Setters
            public Integer getTotalStudyTime() { return totalStudyTime; }
            public void setTotalStudyTime(Integer totalStudyTime) { this.totalStudyTime = totalStudyTime; }
            public Integer getTotalCourses() { return totalCourses; }
            public void setTotalCourses(Integer totalCourses) { this.totalCourses = totalCourses; }
            public Integer getCompletedCourses() { return completedCourses; }
            public void setCompletedCourses(Integer completedCourses) { this.completedCourses = completedCourses; }
            public Integer getOngoingCourses() { return ongoingCourses; }
            public void setOngoingCourses(Integer ongoingCourses) { this.ongoingCourses = ongoingCourses; }
            public Double getAverageScore() { return averageScore; }
            public void setAverageScore(Double averageScore) { this.averageScore = averageScore; }
            public Integer getTotalTasks() { return totalTasks; }
            public void setTotalTasks(Integer totalTasks) { this.totalTasks = totalTasks; }
            public Integer getCompletedTasks() { return completedTasks; }
            public void setCompletedTasks(Integer completedTasks) { this.completedTasks = completedTasks; }
            public Double getTaskCompletionRate() { return taskCompletionRate; }
            public void setTaskCompletionRate(Double taskCompletionRate) { this.taskCompletionRate = taskCompletionRate; }
        }
        
        public static class CourseProgress {
            private Long courseId;
            private String courseName;
            private Double completionRate;
            private Integer studyTime;
            private Double averageScore;
            private String status;
            private LocalDateTime lastStudyTime;
            
            // Getters and Setters
            public Long getCourseId() { return courseId; }
            public void setCourseId(Long courseId) { this.courseId = courseId; }
            public String getCourseName() { return courseName; }
            public void setCourseName(String courseName) { this.courseName = courseName; }
            public Double getCompletionRate() { return completionRate; }
            public void setCompletionRate(Double completionRate) { this.completionRate = completionRate; }
            public Integer getStudyTime() { return studyTime; }
            public void setStudyTime(Integer studyTime) { this.studyTime = studyTime; }
            public Double getAverageScore() { return averageScore; }
            public void setAverageScore(Double averageScore) { this.averageScore = averageScore; }
            public String getStatus() { return status; }
            public void setStatus(String status) { this.status = status; }
            public LocalDateTime getLastStudyTime() { return lastStudyTime; }
            public void setLastStudyTime(LocalDateTime lastStudyTime) { this.lastStudyTime = lastStudyTime; }
        }
        
        public static class Achievement {
            private String achievementType;
            private String title;
            private String description;
            private LocalDateTime achieveTime;
            private String badgeUrl;
            
            // Getters and Setters
            public String getAchievementType() { return achievementType; }
            public void setAchievementType(String achievementType) { this.achievementType = achievementType; }
            public String getTitle() { return title; }
            public void setTitle(String title) { this.title = title; }
            public String getDescription() { return description; }
            public void setDescription(String description) { this.description = description; }
            public LocalDateTime getAchieveTime() { return achieveTime; }
            public void setAchieveTime(LocalDateTime achieveTime) { this.achieveTime = achieveTime; }
            public String getBadgeUrl() { return badgeUrl; }
            public void setBadgeUrl(String badgeUrl) { this.badgeUrl = badgeUrl; }
        }
        
        public static class Recommendation {
            private String recommendationType;
            private String title;
            private String description;
            private String actionUrl;
            private String priority;
            
            // Getters and Setters
            public String getRecommendationType() { return recommendationType; }
            public void setRecommendationType(String recommendationType) { this.recommendationType = recommendationType; }
            public String getTitle() { return title; }
            public void setTitle(String title) { this.title = title; }
            public String getDescription() { return description; }
            public void setDescription(String description) { this.description = description; }
            public String getActionUrl() { return actionUrl; }
            public void setActionUrl(String actionUrl) { this.actionUrl = actionUrl; }
            public String getPriority() { return priority; }
            public void setPriority(String priority) { this.priority = priority; }
        }
        
        public static class StudyPattern {
            private String preferredStudyTime;
            private Integer averageSessionDuration;
            private List<String> mostActiveWeekdays;
            private String studyConsistency;
            private Double focusScore;
            
            // Getters and Setters
            public String getPreferredStudyTime() { return preferredStudyTime; }
            public void setPreferredStudyTime(String preferredStudyTime) { this.preferredStudyTime = preferredStudyTime; }
            public Integer getAverageSessionDuration() { return averageSessionDuration; }
            public void setAverageSessionDuration(Integer averageSessionDuration) { this.averageSessionDuration = averageSessionDuration; }
            public List<String> getMostActiveWeekdays() { return mostActiveWeekdays; }
            public void setMostActiveWeekdays(List<String> mostActiveWeekdays) { this.mostActiveWeekdays = mostActiveWeekdays; }
            public String getStudyConsistency() { return studyConsistency; }
            public void setStudyConsistency(String studyConsistency) { this.studyConsistency = studyConsistency; }
            public Double getFocusScore() { return focusScore; }
            public void setFocusScore(Double focusScore) { this.focusScore = focusScore; }
        }
    }

    /**
     * 导出响应DTO
     */
    public static class ExportResponse {
        private String exportId;
        private String exportType;
        private String status;
        private String fileName;
        private String downloadUrl;
        private LocalDateTime createTime;
        private LocalDateTime expiryTime;
        private Long fileSize;
        private String format;
        
        // Getters and Setters
        public String getExportId() { return exportId; }
        public void setExportId(String exportId) { this.exportId = exportId; }
        public String getExportType() { return exportType; }
        public void setExportType(String exportType) { this.exportType = exportType; }
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
        public String getFileName() { return fileName; }
        public void setFileName(String fileName) { this.fileName = fileName; }
        public String getDownloadUrl() { return downloadUrl; }
        public void setDownloadUrl(String downloadUrl) { this.downloadUrl = downloadUrl; }
        public LocalDateTime getCreateTime() { return createTime; }
        public void setCreateTime(LocalDateTime createTime) { this.createTime = createTime; }
        public LocalDateTime getExpiryTime() { return expiryTime; }
        public void setExpiryTime(LocalDateTime expiryTime) { this.expiryTime = expiryTime; }
        public Long getFileSize() { return fileSize; }
        public void setFileSize(Long fileSize) { this.fileSize = fileSize; }
        public String getFormat() { return format; }
        public void setFormat(String format) { this.format = format; }
    }

    /**
     * 学习数据导出请求DTO
     */
    public static class LearningDataExportRequest {
        @NotBlank(message = "导出类型不能为空")
        private String exportType;
        
        @NotBlank(message = "导出格式不能为空")
        private String format;
        
        private String timeRange;
        private List<Long> courseIds;
        private Boolean includePersonalInfo;
        private Boolean includeScores;
        private Boolean includeStudyTime;
        private Boolean includeNotes;
        
        // Getters and Setters
        public String getExportType() { return exportType; }
        public void setExportType(String exportType) { this.exportType = exportType; }
        public String getFormat() { return format; }
        public void setFormat(String format) { this.format = format; }
        public String getTimeRange() { return timeRange; }
        public void setTimeRange(String timeRange) { this.timeRange = timeRange; }
        public List<Long> getCourseIds() { return courseIds; }
        public void setCourseIds(List<Long> courseIds) { this.courseIds = courseIds; }
        public Boolean getIncludePersonalInfo() { return includePersonalInfo; }
        public void setIncludePersonalInfo(Boolean includePersonalInfo) { this.includePersonalInfo = includePersonalInfo; }
        public Boolean getIncludeScores() { return includeScores; }
        public void setIncludeScores(Boolean includeScores) { this.includeScores = includeScores; }
        public Boolean getIncludeStudyTime() { return includeStudyTime; }
        public void setIncludeStudyTime(Boolean includeStudyTime) { this.includeStudyTime = includeStudyTime; }
        public Boolean getIncludeNotes() { return includeNotes; }
        public void setIncludeNotes(Boolean includeNotes) { this.includeNotes = includeNotes; }
    }
}