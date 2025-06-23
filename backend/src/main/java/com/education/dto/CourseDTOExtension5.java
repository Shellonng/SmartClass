package com.education.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import java.time.LocalDateTime;
import java.util.List;

/**
 * 课程相关DTO扩展5 - 包含公告、资源、评价等相关类
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
public class CourseDTOExtension5 {

    /**
     * 公告响应DTO
     */
    public static class AnnouncementResponse {
        private Long announcementId;
        private String title;
        private String content;
        private String type;
        private Boolean isImportant;
        private LocalDateTime publishTime;
        private String publisherName;
        private Boolean isRead;
        
        // Getters and Setters
        public Long getAnnouncementId() { return announcementId; }
        public void setAnnouncementId(Long announcementId) { this.announcementId = announcementId; }
        public String getTitle() { return title; }
        public void setTitle(String title) { this.title = title; }
        public String getContent() { return content; }
        public void setContent(String content) { this.content = content; }
        public String getType() { return type; }
        public void setType(String type) { this.type = type; }
        public Boolean getIsImportant() { return isImportant; }
        public void setIsImportant(Boolean isImportant) { this.isImportant = isImportant; }
        public LocalDateTime getPublishTime() { return publishTime; }
        public void setPublishTime(LocalDateTime publishTime) { this.publishTime = publishTime; }
        public String getPublisherName() { return publisherName; }
        public void setPublisherName(String publisherName) { this.publisherName = publisherName; }
        public Boolean getIsRead() { return isRead; }
        public void setIsRead(Boolean isRead) { this.isRead = isRead; }
    }

    /**
     * 资源响应DTO
     */
    public static class ResourceResponse {
        private Long resourceId;
        private String resourceName;
        private String description;
        private String resourceType;
        private String fileUrl;
        private Long fileSize;
        private String fileName;
        private LocalDateTime uploadTime;
        private String uploaderName;
        private Integer downloadCount;
        
        // Getters and Setters
        public Long getResourceId() { return resourceId; }
        public void setResourceId(Long resourceId) { this.resourceId = resourceId; }
        public String getResourceName() { return resourceName; }
        public void setResourceName(String resourceName) { this.resourceName = resourceName; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
        public String getResourceType() { return resourceType; }
        public void setResourceType(String resourceType) { this.resourceType = resourceType; }
        public String getFileUrl() { return fileUrl; }
        public void setFileUrl(String fileUrl) { this.fileUrl = fileUrl; }
        public Long getFileSize() { return fileSize; }
        public void setFileSize(Long fileSize) { this.fileSize = fileSize; }
        public String getFileName() { return fileName; }
        public void setFileName(String fileName) { this.fileName = fileName; }
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
        private Long resourceId;
        private String fileName;
        private String downloadUrl;
        private Long fileSize;
        private String contentType;
        private LocalDateTime expiryTime;
        
        // Getters and Setters
        public Long getResourceId() { return resourceId; }
        public void setResourceId(Long resourceId) { this.resourceId = resourceId; }
        public String getFileName() { return fileName; }
        public void setFileName(String fileName) { this.fileName = fileName; }
        public String getDownloadUrl() { return downloadUrl; }
        public void setDownloadUrl(String downloadUrl) { this.downloadUrl = downloadUrl; }
        public Long getFileSize() { return fileSize; }
        public void setFileSize(Long fileSize) { this.fileSize = fileSize; }
        public String getContentType() { return contentType; }
        public void setContentType(String contentType) { this.contentType = contentType; }
        public LocalDateTime getExpiryTime() { return expiryTime; }
        public void setExpiryTime(LocalDateTime expiryTime) { this.expiryTime = expiryTime; }
    }

    /**
     * 课程评价请求DTO
     */
    public static class CourseEvaluationRequest {
        @NotNull(message = "课程ID不能为空")
        private Long courseId;
        @NotNull(message = "评分不能为空")
        private Integer rating;
        private String comment;
        private List<String> tags;
        
        // Getters and Setters
        public Long getCourseId() { return courseId; }
        public void setCourseId(Long courseId) { this.courseId = courseId; }
        public Integer getRating() { return rating; }
        public void setRating(Integer rating) { this.rating = rating; }
        public String getComment() { return comment; }
        public void setComment(String comment) { this.comment = comment; }
        public List<String> getTags() { return tags; }
        public void setTags(List<String> tags) { this.tags = tags; }
    }

    /**
     * 评价响应DTO
     */
    public static class EvaluationResponse {
        private Long evaluationId;
        private Long courseId;
        private Long studentId;
        private String studentName;
        private Integer rating;
        private String comment;
        private List<String> tags;
        private LocalDateTime createTime;
        private Integer likeCount;
        private Boolean isLiked;
        
        // Getters and Setters
        public Long getEvaluationId() { return evaluationId; }
        public void setEvaluationId(Long evaluationId) { this.evaluationId = evaluationId; }
        public Long getCourseId() { return courseId; }
        public void setCourseId(Long courseId) { this.courseId = courseId; }
        public Long getStudentId() { return studentId; }
        public void setStudentId(Long studentId) { this.studentId = studentId; }
        public String getStudentName() { return studentName; }
        public void setStudentName(String studentName) { this.studentName = studentName; }
        public Integer getRating() { return rating; }
        public void setRating(Integer rating) { this.rating = rating; }
        public String getComment() { return comment; }
        public void setComment(String comment) { this.comment = comment; }
        public List<String> getTags() { return tags; }
        public void setTags(List<String> tags) { this.tags = tags; }
        public LocalDateTime getCreateTime() { return createTime; }
        public void setCreateTime(LocalDateTime createTime) { this.createTime = createTime; }
        public Integer getLikeCount() { return likeCount; }
        public void setLikeCount(Integer likeCount) { this.likeCount = likeCount; }
        public Boolean getIsLiked() { return isLiked; }
        public void setIsLiked(Boolean isLiked) { this.isLiked = isLiked; }
    }

    /**
     * 课程搜索请求DTO
     */
    public static class CourseSearchRequest {
        private String keyword;
        private String category;
        private String difficulty;
        private String status;
        private String sortBy;
        private String sortOrder;
        private Integer minCredit;
        private Integer maxCredit;
        private LocalDateTime startTimeFrom;
        private LocalDateTime startTimeTo;
        
        // Getters and Setters
        public String getKeyword() { return keyword; }
        public void setKeyword(String keyword) { this.keyword = keyword; }
        public String getCategory() { return category; }
        public void setCategory(String category) { this.category = category; }
        public String getDifficulty() { return difficulty; }
        public void setDifficulty(String difficulty) { this.difficulty = difficulty; }
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
        public String getSortBy() { return sortBy; }
        public void setSortBy(String sortBy) { this.sortBy = sortBy; }
        public String getSortOrder() { return sortOrder; }
        public void setSortOrder(String sortOrder) { this.sortOrder = sortOrder; }
        public Integer getMinCredit() { return minCredit; }
        public void setMinCredit(Integer minCredit) { this.minCredit = minCredit; }
        public Integer getMaxCredit() { return maxCredit; }
        public void setMaxCredit(Integer maxCredit) { this.maxCredit = maxCredit; }
        public LocalDateTime getStartTimeFrom() { return startTimeFrom; }
        public void setStartTimeFrom(LocalDateTime startTimeFrom) { this.startTimeFrom = startTimeFrom; }
        public LocalDateTime getStartTimeTo() { return startTimeTo; }
        public void setStartTimeTo(LocalDateTime startTimeTo) { this.startTimeTo = startTimeTo; }
    }

    /**
     * 学习统计响应DTO
     */
    public static class LearningStatisticsResponse {
        private Long courseId;
        private Long studentId;
        private Integer totalStudyTime;
        private Integer todayStudyTime;
        private Integer weekStudyTime;
        private Integer monthStudyTime;
        private Double progress;
        private Integer completedChapters;
        private Integer totalChapters;
        private Integer streakDays;
        private LocalDateTime lastStudyTime;
        private List<DailyStudyRecord> dailyRecords;
        
        // Getters and Setters
        public Long getCourseId() { return courseId; }
        public void setCourseId(Long courseId) { this.courseId = courseId; }
        public Long getStudentId() { return studentId; }
        public void setStudentId(Long studentId) { this.studentId = studentId; }
        public Integer getTotalStudyTime() { return totalStudyTime; }
        public void setTotalStudyTime(Integer totalStudyTime) { this.totalStudyTime = totalStudyTime; }
        public Integer getTodayStudyTime() { return todayStudyTime; }
        public void setTodayStudyTime(Integer todayStudyTime) { this.todayStudyTime = todayStudyTime; }
        public Integer getWeekStudyTime() { return weekStudyTime; }
        public void setWeekStudyTime(Integer weekStudyTime) { this.weekStudyTime = weekStudyTime; }
        public Integer getMonthStudyTime() { return monthStudyTime; }
        public void setMonthStudyTime(Integer monthStudyTime) { this.monthStudyTime = monthStudyTime; }
        public Double getProgress() { return progress; }
        public void setProgress(Double progress) { this.progress = progress; }
        public Integer getCompletedChapters() { return completedChapters; }
        public void setCompletedChapters(Integer completedChapters) { this.completedChapters = completedChapters; }
        public Integer getTotalChapters() { return totalChapters; }
        public void setTotalChapters(Integer totalChapters) { this.totalChapters = totalChapters; }
        public Integer getStreakDays() { return streakDays; }
        public void setStreakDays(Integer streakDays) { this.streakDays = streakDays; }
        public LocalDateTime getLastStudyTime() { return lastStudyTime; }
        public void setLastStudyTime(LocalDateTime lastStudyTime) { this.lastStudyTime = lastStudyTime; }
        public List<DailyStudyRecord> getDailyRecords() { return dailyRecords; }
        public void setDailyRecords(List<DailyStudyRecord> dailyRecords) { this.dailyRecords = dailyRecords; }
    }

    /**
     * 每日学习记录DTO
     */
    public static class DailyStudyRecord {
        private String date;
        private Integer studyTime;
        private Integer completedChapters;
        
        // Getters and Setters
        public String getDate() { return date; }
        public void setDate(String date) { this.date = date; }
        public Integer getStudyTime() { return studyTime; }
        public void setStudyTime(Integer studyTime) { this.studyTime = studyTime; }
        public Integer getCompletedChapters() { return completedChapters; }
        public void setCompletedChapters(Integer completedChapters) { this.completedChapters = completedChapters; }
    }
}