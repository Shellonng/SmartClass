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
 * 课程相关DTO扩展类
 * 包含更多课程相关的DTO定义
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
public class CourseDTOExtension {

    /**
     * 课程更新请求DTO
     */
    public static class CourseUpdateRequest {
        private String courseName;
        private String description;
        private String coverImage;
        private Integer credit;
        private String category;
        private String difficulty;
        private LocalDateTime startTime;
        private LocalDateTime endTime;
        
        // Getters and Setters
        public String getCourseName() { return courseName; }
        public void setCourseName(String courseName) { this.courseName = courseName; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
        public String getCoverImage() { return coverImage; }
        public void setCoverImage(String coverImage) { this.coverImage = coverImage; }
        public Integer getCredit() { return credit; }
        public void setCredit(Integer credit) { this.credit = credit; }
        public String getCategory() { return category; }
        public void setCategory(String category) { this.category = category; }
        public String getDifficulty() { return difficulty; }
        public void setDifficulty(String difficulty) { this.difficulty = difficulty; }
        public LocalDateTime getStartTime() { return startTime; }
        public void setStartTime(LocalDateTime startTime) { this.startTime = startTime; }
        public LocalDateTime getEndTime() { return endTime; }
        public void setEndTime(LocalDateTime endTime) { this.endTime = endTime; }
    }

    /**
     * 课程详情响应DTO
     */
    public static class CourseDetailResponse {
        private Long courseId;
        private String courseName;
        private String description;
        private String coverImage;
        private Integer credit;
        private String category;
        private String difficulty;
        private String status;
        private LocalDateTime startTime;
        private LocalDateTime endTime;
        private LocalDateTime createTime;
        private String teacherName;
        private Integer studentCount;
        private List<ChapterResponse> chapters;
        private List<String> tags;
        private String syllabus;
        private String prerequisites;
        private String objectives;
        private Double rating;
        private Integer reviewCount;
        
        // Getters and Setters
        public Long getCourseId() { return courseId; }
        public void setCourseId(Long courseId) { this.courseId = courseId; }
        public String getCourseName() { return courseName; }
        public void setCourseName(String courseName) { this.courseName = courseName; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
        public String getCoverImage() { return coverImage; }
        public void setCoverImage(String coverImage) { this.coverImage = coverImage; }
        public Integer getCredit() { return credit; }
        public void setCredit(Integer credit) { this.credit = credit; }
        public String getCategory() { return category; }
        public void setCategory(String category) { this.category = category; }
        public String getDifficulty() { return difficulty; }
        public void setDifficulty(String difficulty) { this.difficulty = difficulty; }
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
        public LocalDateTime getStartTime() { return startTime; }
        public void setStartTime(LocalDateTime startTime) { this.startTime = startTime; }
        public LocalDateTime getEndTime() { return endTime; }
        public void setEndTime(LocalDateTime endTime) { this.endTime = endTime; }
        public LocalDateTime getCreateTime() { return createTime; }
        public void setCreateTime(LocalDateTime createTime) { this.createTime = createTime; }
        public String getTeacherName() { return teacherName; }
        public void setTeacherName(String teacherName) { this.teacherName = teacherName; }
        public Integer getStudentCount() { return studentCount; }
        public void setStudentCount(Integer studentCount) { this.studentCount = studentCount; }
        public List<ChapterResponse> getChapters() { return chapters; }
        public void setChapters(List<ChapterResponse> chapters) { this.chapters = chapters; }
        public List<String> getTags() { return tags; }
        public void setTags(List<String> tags) { this.tags = tags; }
        public String getSyllabus() { return syllabus; }
        public void setSyllabus(String syllabus) { this.syllabus = syllabus; }
        public String getPrerequisites() { return prerequisites; }
        public void setPrerequisites(String prerequisites) { this.prerequisites = prerequisites; }
        public String getObjectives() { return objectives; }
        public void setObjectives(String objectives) { this.objectives = objectives; }
        public Double getRating() { return rating; }
        public void setRating(Double rating) { this.rating = rating; }
        public Integer getReviewCount() { return reviewCount; }
        public void setReviewCount(Integer reviewCount) { this.reviewCount = reviewCount; }
    }

    /**
     * 章节响应DTO
     */
    public static class ChapterResponse {
        private Long chapterId;
        private String chapterTitle;
        private String description;
        private Integer orderIndex;
        private String status;
        private Integer duration;
        private String videoUrl;
        private List<String> resources;
        private LocalDateTime createTime;
        
        // Getters and Setters
        public Long getChapterId() { return chapterId; }
        public void setChapterId(Long chapterId) { this.chapterId = chapterId; }
        public String getChapterTitle() { return chapterTitle; }
        public void setChapterTitle(String chapterTitle) { this.chapterTitle = chapterTitle; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
        public Integer getOrderIndex() { return orderIndex; }
        public void setOrderIndex(Integer orderIndex) { this.orderIndex = orderIndex; }
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
        public Integer getDuration() { return duration; }
        public void setDuration(Integer duration) { this.duration = duration; }
        public String getVideoUrl() { return videoUrl; }
        public void setVideoUrl(String videoUrl) { this.videoUrl = videoUrl; }
        public List<String> getResources() { return resources; }
        public void setResources(List<String> resources) { this.resources = resources; }
        public LocalDateTime getCreateTime() { return createTime; }
        public void setCreateTime(LocalDateTime createTime) { this.createTime = createTime; }
    }

    /**
     * 章节创建请求DTO
     */
    public static class ChapterCreateRequest {
        @NotBlank(message = "章节标题不能为空")
        private String chapterTitle;
        
        private String description;
        private Integer orderIndex;
        private Integer duration;
        private String videoUrl;
        private List<String> resources;
        
        // Getters and Setters
        public String getChapterTitle() { return chapterTitle; }
        public void setChapterTitle(String chapterTitle) { this.chapterTitle = chapterTitle; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
        public Integer getOrderIndex() { return orderIndex; }
        public void setOrderIndex(Integer orderIndex) { this.orderIndex = orderIndex; }
        public Integer getDuration() { return duration; }
        public void setDuration(Integer duration) { this.duration = duration; }
        public String getVideoUrl() { return videoUrl; }
        public void setVideoUrl(String videoUrl) { this.videoUrl = videoUrl; }
        public List<String> getResources() { return resources; }
        public void setResources(List<String> resources) { this.resources = resources; }
    }

    /**
     * 章节更新请求DTO
     */
    public static class ChapterUpdateRequest {
        private String chapterTitle;
        private String description;
        private Integer orderIndex;
        private Integer duration;
        private String videoUrl;
        private List<String> resources;
        
        // Getters and Setters
        public String getChapterTitle() { return chapterTitle; }
        public void setChapterTitle(String chapterTitle) { this.chapterTitle = chapterTitle; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
        public Integer getOrderIndex() { return orderIndex; }
        public void setOrderIndex(Integer orderIndex) { this.orderIndex = orderIndex; }
        public Integer getDuration() { return duration; }
        public void setDuration(Integer duration) { this.duration = duration; }
        public String getVideoUrl() { return videoUrl; }
        public void setVideoUrl(String videoUrl) { this.videoUrl = videoUrl; }
        public List<String> getResources() { return resources; }
        public void setResources(List<String> resources) { this.resources = resources; }
    }

    /**
     * 章节详情响应DTO
     */
    public static class ChapterDetailResponse {
        private Long chapterId;
        private String chapterTitle;
        private String description;
        private Integer orderIndex;
        private String status;
        private Integer duration;
        private String videoUrl;
        private List<ResourceResponse> resources;
        private List<String> attachments;
        private String content;
        private LocalDateTime createTime;
        private Boolean isCompleted;
        private Integer progress;
        
        // Getters and Setters
        public Long getChapterId() { return chapterId; }
        public void setChapterId(Long chapterId) { this.chapterId = chapterId; }
        public String getChapterTitle() { return chapterTitle; }
        public void setChapterTitle(String chapterTitle) { this.chapterTitle = chapterTitle; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
        public Integer getOrderIndex() { return orderIndex; }
        public void setOrderIndex(Integer orderIndex) { this.orderIndex = orderIndex; }
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
        public Integer getDuration() { return duration; }
        public void setDuration(Integer duration) { this.duration = duration; }
        public String getVideoUrl() { return videoUrl; }
        public void setVideoUrl(String videoUrl) { this.videoUrl = videoUrl; }
        public List<ResourceResponse> getResources() { return resources; }
        public void setResources(List<ResourceResponse> resources) { this.resources = resources; }
        public List<String> getAttachments() { return attachments; }
        public void setAttachments(List<String> attachments) { this.attachments = attachments; }
        public String getContent() { return content; }
        public void setContent(String content) { this.content = content; }
        public LocalDateTime getCreateTime() { return createTime; }
        public void setCreateTime(LocalDateTime createTime) { this.createTime = createTime; }
        public Boolean getIsCompleted() { return isCompleted; }
        public void setIsCompleted(Boolean isCompleted) { this.isCompleted = isCompleted; }
        public Integer getProgress() { return progress; }
        public void setProgress(Integer progress) { this.progress = progress; }
    }

    /**
     * 章节排序请求DTO
     */
    public static class ChapterOrderRequest {
        @NotNull(message = "章节ID不能为空")
        private Long chapterId;
        
        @NotNull(message = "排序索引不能为空")
        private Integer orderIndex;
        
        // Getters and Setters
        public Long getChapterId() { return chapterId; }
        public void setChapterId(Long chapterId) { this.chapterId = chapterId; }
        public Integer getOrderIndex() { return orderIndex; }
        public void setOrderIndex(Integer orderIndex) { this.orderIndex = orderIndex; }
    }

    /**
     * 课程统计响应DTO
     */
    public static class CourseStatisticsResponse {
        private Integer totalStudents;
        private Integer activeStudents;
        private Integer completedStudents;
        private Double averageProgress;
        private Double completionRate;
        private Integer totalChapters;
        private Integer totalTasks;
        private Double averageRating;
        private Integer totalReviews;
        private Map<String, Integer> progressDistribution;
        private List<ChapterStatistics> chapterStatistics;
        
        // Getters and Setters
        public Integer getTotalStudents() { return totalStudents; }
        public void setTotalStudents(Integer totalStudents) { this.totalStudents = totalStudents; }
        public Integer getActiveStudents() { return activeStudents; }
        public void setActiveStudents(Integer activeStudents) { this.activeStudents = activeStudents; }
        public Integer getCompletedStudents() { return completedStudents; }
        public void setCompletedStudents(Integer completedStudents) { this.completedStudents = completedStudents; }
        public Double getAverageProgress() { return averageProgress; }
        public void setAverageProgress(Double averageProgress) { this.averageProgress = averageProgress; }
        public Double getCompletionRate() { return completionRate; }
        public void setCompletionRate(Double completionRate) { this.completionRate = completionRate; }
        public Integer getTotalChapters() { return totalChapters; }
        public void setTotalChapters(Integer totalChapters) { this.totalChapters = totalChapters; }
        public Integer getTotalTasks() { return totalTasks; }
        public void setTotalTasks(Integer totalTasks) { this.totalTasks = totalTasks; }
        public Double getAverageRating() { return averageRating; }
        public void setAverageRating(Double averageRating) { this.averageRating = averageRating; }
        public Integer getTotalReviews() { return totalReviews; }
        public void setTotalReviews(Integer totalReviews) { this.totalReviews = totalReviews; }
        public Map<String, Integer> getProgressDistribution() { return progressDistribution; }
        public void setProgressDistribution(Map<String, Integer> progressDistribution) { this.progressDistribution = progressDistribution; }
        public List<ChapterStatistics> getChapterStatistics() { return chapterStatistics; }
        public void setChapterStatistics(List<ChapterStatistics> chapterStatistics) { this.chapterStatistics = chapterStatistics; }
        
        public static class ChapterStatistics {
            private Long chapterId;
            private String chapterTitle;
            private Integer completedCount;
            private Double completionRate;
            private Integer averageTime;
            
            // Getters and Setters
            public Long getChapterId() { return chapterId; }
            public void setChapterId(Long chapterId) { this.chapterId = chapterId; }
            public String getChapterTitle() { return chapterTitle; }
            public void setChapterTitle(String chapterTitle) { this.chapterTitle = chapterTitle; }
            public Integer getCompletedCount() { return completedCount; }
            public void setCompletedCount(Integer completedCount) { this.completedCount = completedCount; }
            public Double getCompletionRate() { return completionRate; }
            public void setCompletionRate(Double completionRate) { this.completionRate = completionRate; }
            public Integer getAverageTime() { return averageTime; }
            public void setAverageTime(Integer averageTime) { this.averageTime = averageTime; }
        }
    }

    /**
     * 课程列表响应DTO
     */
    public static class CourseListResponse {
        private Long courseId;
        private String courseName;
        private String description;
        private String coverImage;
        private Integer credit;
        private String category;
        private String difficulty;
        private String status;
        private LocalDateTime startTime;
        private LocalDateTime endTime;
        private String teacherName;
        private Integer studentCount;
        private Double rating;
        private Boolean isFavorite;
        private Integer progress;
        
        // Getters and Setters
        public Long getCourseId() { return courseId; }
        public void setCourseId(Long courseId) { this.courseId = courseId; }
        public String getCourseName() { return courseName; }
        public void setCourseName(String courseName) { this.courseName = courseName; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
        public String getCoverImage() { return coverImage; }
        public void setCoverImage(String coverImage) { this.coverImage = coverImage; }
        public Integer getCredit() { return credit; }
        public void setCredit(Integer credit) { this.credit = credit; }
        public String getCategory() { return category; }
        public void setCategory(String category) { this.category = category; }
        public String getDifficulty() { return difficulty; }
        public void setDifficulty(String difficulty) { this.difficulty = difficulty; }
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
        public LocalDateTime getStartTime() { return startTime; }
        public void setStartTime(LocalDateTime startTime) { this.startTime = startTime; }
        public LocalDateTime getEndTime() { return endTime; }
        public void setEndTime(LocalDateTime endTime) { this.endTime = endTime; }
        public String getTeacherName() { return teacherName; }
        public void setTeacherName(String teacherName) { this.teacherName = teacherName; }
        public Integer getStudentCount() { return studentCount; }
        public void setStudentCount(Integer studentCount) { this.studentCount = studentCount; }
        public Double getRating() { return rating; }
        public void setRating(Double rating) { this.rating = rating; }
        public Boolean getIsFavorite() { return isFavorite; }
        public void setIsFavorite(Boolean isFavorite) { this.isFavorite = isFavorite; }
        public Integer getProgress() { return progress; }
        public void setProgress(Integer progress) { this.progress = progress; }
    }

    /**
     * 学习进度响应DTO
     */
    public static class LearningProgressResponse {
        private Long courseId;
        private String courseName;
        private Integer totalChapters;
        private Integer completedChapters;
        private Double progressPercentage;
        private Integer totalStudyTime;
        private LocalDateTime lastStudyTime;
        private List<ChapterProgress> chapterProgress;
        private String currentChapter;
        private String nextChapter;
        
        // Getters and Setters
        public Long getCourseId() { return courseId; }
        public void setCourseId(Long courseId) { this.courseId = courseId; }
        public String getCourseName() { return courseName; }
        public void setCourseName(String courseName) { this.courseName = courseName; }
        public Integer getTotalChapters() { return totalChapters; }
        public void setTotalChapters(Integer totalChapters) { this.totalChapters = totalChapters; }
        public Integer getCompletedChapters() { return completedChapters; }
        public void setCompletedChapters(Integer completedChapters) { this.completedChapters = completedChapters; }
        public Double getProgressPercentage() { return progressPercentage; }
        public void setProgressPercentage(Double progressPercentage) { this.progressPercentage = progressPercentage; }
        public Integer getTotalStudyTime() { return totalStudyTime; }
        public void setTotalStudyTime(Integer totalStudyTime) { this.totalStudyTime = totalStudyTime; }
        public LocalDateTime getLastStudyTime() { return lastStudyTime; }
        public void setLastStudyTime(LocalDateTime lastStudyTime) { this.lastStudyTime = lastStudyTime; }
        public List<ChapterProgress> getChapterProgress() { return chapterProgress; }
        public void setChapterProgress(List<ChapterProgress> chapterProgress) { this.chapterProgress = chapterProgress; }
        public String getCurrentChapter() { return currentChapter; }
        public void setCurrentChapter(String currentChapter) { this.currentChapter = currentChapter; }
        public String getNextChapter() { return nextChapter; }
        public void setNextChapter(String nextChapter) { this.nextChapter = nextChapter; }
        
        public static class ChapterProgress {
            private Long chapterId;
            private String chapterTitle;
            private Boolean isCompleted;
            private Integer progress;
            private Integer studyTime;
            private LocalDateTime lastAccessTime;
            
            // Getters and Setters
            public Long getChapterId() { return chapterId; }
            public void setChapterId(Long chapterId) { this.chapterId = chapterId; }
            public String getChapterTitle() { return chapterTitle; }
            public void setChapterTitle(String chapterTitle) { this.chapterTitle = chapterTitle; }
            public Boolean getIsCompleted() { return isCompleted; }
            public void setIsCompleted(Boolean isCompleted) { this.isCompleted = isCompleted; }
            public Integer getProgress() { return progress; }
            public void setProgress(Integer progress) { this.progress = progress; }
            public Integer getStudyTime() { return studyTime; }
            public void setStudyTime(Integer studyTime) { this.studyTime = studyTime; }
            public LocalDateTime getLastAccessTime() { return lastAccessTime; }
            public void setLastAccessTime(LocalDateTime lastAccessTime) { this.lastAccessTime = lastAccessTime; }
        }
    }

    /**
     * 进度更新请求DTO
     */
    public static class ProgressUpdateRequest {
        @NotNull(message = "章节ID不能为空")
        private Long chapterId;
        
        @NotNull(message = "进度不能为空")
        private Integer progress;
        
        private Integer studyTime;
        private Boolean isCompleted;
        
        // Getters and Setters
        public Long getChapterId() { return chapterId; }
        public void setChapterId(Long chapterId) { this.chapterId = chapterId; }
        public Integer getProgress() { return progress; }
        public void setProgress(Integer progress) { this.progress = progress; }
        public Integer getStudyTime() { return studyTime; }
        public void setStudyTime(Integer studyTime) { this.studyTime = studyTime; }
        public Boolean getIsCompleted() { return isCompleted; }
        public void setIsCompleted(Boolean isCompleted) { this.isCompleted = isCompleted; }
    }

    /**
     * 资源响应DTO
     */
    public static class ResourceResponse {
        private Long resourceId;
        private String resourceName;
        private String resourceType;
        private String resourceUrl;
        private Long fileSize;
        private String description;
        private LocalDateTime uploadTime;
        
        // Getters and Setters
        public Long getResourceId() { return resourceId; }
        public void setResourceId(Long resourceId) { this.resourceId = resourceId; }
        public String getResourceName() { return resourceName; }
        public void setResourceName(String resourceName) { this.resourceName = resourceName; }
        public String getResourceType() { return resourceType; }
        public void setResourceType(String resourceType) { this.resourceType = resourceType; }
        public String getResourceUrl() { return resourceUrl; }
        public void setResourceUrl(String resourceUrl) { this.resourceUrl = resourceUrl; }
        public Long getFileSize() { return fileSize; }
        public void setFileSize(Long fileSize) { this.fileSize = fileSize; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
        public LocalDateTime getUploadTime() { return uploadTime; }
        public void setUploadTime(LocalDateTime uploadTime) { this.uploadTime = uploadTime; }
    }

    /**
     * 资源下载响应DTO
     */
    public static class ResourceDownloadResponse {
        private String downloadUrl;
        private String fileName;
        private Long fileSize;
        private String contentType;
        private LocalDateTime expiryTime;
        
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
    }
}