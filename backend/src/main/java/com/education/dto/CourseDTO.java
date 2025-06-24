package com.education.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import java.time.LocalDateTime;
import java.util.List;

/**
 * 课程相关DTO
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
public class CourseDTO {

    /**
     * 课程创建请求DTO
     */
    public static class CourseCreateRequest {
        @NotBlank(message = "课程名称不能为空")
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
     * 课程响应DTO
     */
    public static class CourseResponse {
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
    }

    /**
     * 课程导入请求DTO
     */
    public static class CourseImportRequest {
        private String importType;
        private String fileUrl;
        private List<CourseCreateRequest> courses;
        
        // Getters and Setters
        public String getImportType() { return importType; }
        public void setImportType(String importType) { this.importType = importType; }
        public String getFileUrl() { return fileUrl; }
        public void setFileUrl(String fileUrl) { this.fileUrl = fileUrl; }
        public List<CourseCreateRequest> getCourses() { return courses; }
        public void setCourses(List<CourseCreateRequest> courses) { this.courses = courses; }
    }

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
        private Double progress;
        private Boolean isFavorite;
        
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
        public Double getProgress() { return progress; }
        public void setProgress(Double progress) { this.progress = progress; }
        public Boolean getIsFavorite() { return isFavorite; }
        public void setIsFavorite(Boolean isFavorite) { this.isFavorite = isFavorite; }
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
        private Double progress;
        private Boolean isEnrolled;
        private Boolean isFavorite;
        private List<ChapterResponse> chapters;
        
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
        public Double getProgress() { return progress; }
        public void setProgress(Double progress) { this.progress = progress; }
        public Boolean getIsEnrolled() { return isEnrolled; }
        public void setIsEnrolled(Boolean isEnrolled) { this.isEnrolled = isEnrolled; }
        public Boolean getIsFavorite() { return isFavorite; }
        public void setIsFavorite(Boolean isFavorite) { this.isFavorite = isFavorite; }
        public List<ChapterResponse> getChapters() { return chapters; }
        public void setChapters(List<ChapterResponse> chapters) { this.chapters = chapters; }
    }

    /**
     * 章节响应DTO
     */
    public static class ChapterResponse {
        private Long chapterId;
        private String chapterName;
        private String description;
        private Integer orderIndex;
        private String content;
        private String videoUrl;
        private Integer duration;
        private Boolean isLearned;
        private LocalDateTime createTime;
        
        // Getters and Setters
        public Long getChapterId() { return chapterId; }
        public void setChapterId(Long chapterId) { this.chapterId = chapterId; }
        public String getChapterName() { return chapterName; }
        public void setChapterName(String chapterName) { this.chapterName = chapterName; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
        public Integer getOrderIndex() { return orderIndex; }
        public void setOrderIndex(Integer orderIndex) { this.orderIndex = orderIndex; }
        public String getContent() { return content; }
        public void setContent(String content) { this.content = content; }
        public String getVideoUrl() { return videoUrl; }
        public void setVideoUrl(String videoUrl) { this.videoUrl = videoUrl; }
        public Integer getDuration() { return duration; }
        public void setDuration(Integer duration) { this.duration = duration; }
        public Boolean getIsLearned() { return isLearned; }
        public void setIsLearned(Boolean isLearned) { this.isLearned = isLearned; }
        public LocalDateTime getCreateTime() { return createTime; }
        public void setCreateTime(LocalDateTime createTime) { this.createTime = createTime; }
    }

    /**
     * 章节详情响应DTO
     */
    public static class ChapterDetailResponse {
        private Long chapterId;
        private String chapterName;
        private String description;
        private Integer orderIndex;
        private String content;
        private String videoUrl;
        private Integer duration;
        private Boolean isLearned;
        private LocalDateTime createTime;
        private LocalDateTime lastAccessTime;
        private Integer studyTime;
        
        // Getters and Setters
        public Long getChapterId() { return chapterId; }
        public void setChapterId(Long chapterId) { this.chapterId = chapterId; }
        public String getChapterName() { return chapterName; }
        public void setChapterName(String chapterName) { this.chapterName = chapterName; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
        public Integer getOrderIndex() { return orderIndex; }
        public void setOrderIndex(Integer orderIndex) { this.orderIndex = orderIndex; }
        public String getContent() { return content; }
        public void setContent(String content) { this.content = content; }
        public String getVideoUrl() { return videoUrl; }
        public void setVideoUrl(String videoUrl) { this.videoUrl = videoUrl; }
        public Integer getDuration() { return duration; }
        public void setDuration(Integer duration) { this.duration = duration; }
        public Boolean getIsLearned() { return isLearned; }
        public void setIsLearned(Boolean isLearned) { this.isLearned = isLearned; }
        public LocalDateTime getCreateTime() { return createTime; }
        public void setCreateTime(LocalDateTime createTime) { this.createTime = createTime; }
        public LocalDateTime getLastAccessTime() { return lastAccessTime; }
        public void setLastAccessTime(LocalDateTime lastAccessTime) { this.lastAccessTime = lastAccessTime; }
        public Integer getStudyTime() { return studyTime; }
        public void setStudyTime(Integer studyTime) { this.studyTime = studyTime; }
    }

    /**
     * 学习进度响应DTO
     */
    public static class LearningProgressResponse {
        private Long courseId;
        private Long studentId;
        private Double progress;
        private Integer completedChapters;
        private Integer totalChapters;
        private Integer studyTime;
        private LocalDateTime lastAccessTime;
        private LocalDateTime startTime;
        
        // Getters and Setters
        public Long getCourseId() { return courseId; }
        public void setCourseId(Long courseId) { this.courseId = courseId; }
        public Long getStudentId() { return studentId; }
        public void setStudentId(Long studentId) { this.studentId = studentId; }
        public Double getProgress() { return progress; }
        public void setProgress(Double progress) { this.progress = progress; }
        public Integer getCompletedChapters() { return completedChapters; }
        public void setCompletedChapters(Integer completedChapters) { this.completedChapters = completedChapters; }
        public Integer getTotalChapters() { return totalChapters; }
        public void setTotalChapters(Integer totalChapters) { this.totalChapters = totalChapters; }
        public Integer getStudyTime() { return studyTime; }
        public void setStudyTime(Integer studyTime) { this.studyTime = studyTime; }
        public LocalDateTime getLastAccessTime() { return lastAccessTime; }
        public void setLastAccessTime(LocalDateTime lastAccessTime) { this.lastAccessTime = lastAccessTime; }
        public LocalDateTime getStartTime() { return startTime; }
        public void setStartTime(LocalDateTime startTime) { this.startTime = startTime; }
    }

    /**
     * 进度更新请求DTO
     */
    public static class ProgressUpdateRequest {
        private Long courseId;
        private Long chapterId;
        private Integer studyTime;
        private Boolean isCompleted;
        
        // Getters and Setters
        public Long getCourseId() { return courseId; }
        public void setCourseId(Long courseId) { this.courseId = courseId; }
        public Long getChapterId() { return chapterId; }
        public void setChapterId(Long chapterId) { this.chapterId = chapterId; }
        public Integer getStudyTime() { return studyTime; }
        public void setStudyTime(Integer studyTime) { this.studyTime = studyTime; }
        public Boolean getIsCompleted() { return isCompleted; }
        public void setIsCompleted(Boolean isCompleted) { this.isCompleted = isCompleted; }
    }

    /**
     * 课程公告响应DTO
     */
    public static class AnnouncementResponse {
        private Long announcementId;
        private String title;
        private String content;
        private String priority;
        private LocalDateTime publishTime;
        private LocalDateTime createTime;
        private String authorName;
        
        // Getters and Setters
        public Long getAnnouncementId() { return announcementId; }
        public void setAnnouncementId(Long announcementId) { this.announcementId = announcementId; }
        public String getTitle() { return title; }
        public void setTitle(String title) { this.title = title; }
        public String getContent() { return content; }
        public void setContent(String content) { this.content = content; }
        public String getPriority() { return priority; }
        public void setPriority(String priority) { this.priority = priority; }
        public LocalDateTime getPublishTime() { return publishTime; }
        public void setPublishTime(LocalDateTime publishTime) { this.publishTime = publishTime; }
        public LocalDateTime getCreateTime() { return createTime; }
        public void setCreateTime(LocalDateTime createTime) { this.createTime = createTime; }
        public String getAuthorName() { return authorName; }
        public void setAuthorName(String authorName) { this.authorName = authorName; }
    }

    /**
     * 课程资源响应DTO
     */
    public static class ResourceResponse {
        private Long resourceId;
        private Long courseId;
        private String resourceName;
        private String resourceType;
        private String fileName;
        private String fileType;
        private String fileUrl;
        private Long fileSize;
        private String description;
        private LocalDateTime uploadTime;
        private String uploaderName;
        
        // Getters and Setters
        public Long getResourceId() { return resourceId; }
        public void setResourceId(Long resourceId) { this.resourceId = resourceId; }
        public String getResourceName() { return resourceName; }
        public void setResourceName(String resourceName) { this.resourceName = resourceName; }
        public String getResourceType() { return resourceType; }
        public void setResourceType(String resourceType) { this.resourceType = resourceType; }
        public String getFileUrl() { return fileUrl; }
        public void setFileUrl(String fileUrl) { this.fileUrl = fileUrl; }
        public Long getFileSize() { return fileSize; }
        public void setFileSize(Long fileSize) { this.fileSize = fileSize; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
        public LocalDateTime getUploadTime() { return uploadTime; }
        public void setUploadTime(LocalDateTime uploadTime) { this.uploadTime = uploadTime; }
        public String getUploaderName() { return uploaderName; }
        public void setUploaderName(String uploaderName) { this.uploaderName = uploaderName; }
        public Long getCourseId() { return courseId; }
        public void setCourseId(Long courseId) { this.courseId = courseId; }
        public String getFileName() { return fileName; }
        public void setFileName(String fileName) { this.fileName = fileName; }
        public String getFileType() { return fileType; }
        public void setFileType(String fileType) { this.fileType = fileType; }
        public void setCreatedTime(LocalDateTime createdTime) { this.uploadTime = createdTime; }
    }

    /**
     * 课程讨论响应DTO
     */
    public static class DiscussionResponse {
        private Long discussionId;
        private String title;
        private String content;
        private String authorName;
        private LocalDateTime createTime;
        private Integer replyCount;
        private Integer likeCount;
        private Boolean isLiked;
        
        // Getters and Setters
        public Long getDiscussionId() { return discussionId; }
        public void setDiscussionId(Long discussionId) { this.discussionId = discussionId; }
        public String getTitle() { return title; }
        public void setTitle(String title) { this.title = title; }
        public String getContent() { return content; }
        public void setContent(String content) { this.content = content; }
        public String getAuthorName() { return authorName; }
        public void setAuthorName(String authorName) { this.authorName = authorName; }
        public LocalDateTime getCreateTime() { return createTime; }
        public void setCreateTime(LocalDateTime createTime) { this.createTime = createTime; }
        public Integer getReplyCount() { return replyCount; }
        public void setReplyCount(Integer replyCount) { this.replyCount = replyCount; }
        public Integer getLikeCount() { return likeCount; }
        public void setLikeCount(Integer likeCount) { this.likeCount = likeCount; }
        public Boolean getIsLiked() { return isLiked; }
        public void setIsLiked(Boolean isLiked) { this.isLiked = isLiked; }
    }

    /**
     * 讨论创建请求DTO
     */
    public static class DiscussionCreateRequest {
        private String title;
        private String content;
        private Long courseId;
        
        // Getters and Setters
        public String getTitle() { return title; }
        public void setTitle(String title) { this.title = title; }
        public String getContent() { return content; }
        public void setContent(String content) { this.content = content; }
        public Long getCourseId() { return courseId; }
        public void setCourseId(Long courseId) { this.courseId = courseId; }
    }

    /**
     * 讨论回复请求DTO
     */
    public static class DiscussionReplyRequest {
        private String content;
        private Long parentId;
        
        // Getters and Setters
        public String getContent() { return content; }
        public void setContent(String content) { this.content = content; }
        public Long getParentId() { return parentId; }
        public void setParentId(Long parentId) { this.parentId = parentId; }
    }

    /**
     * 课程评价请求DTO
     */
    public static class CourseEvaluationRequest {
        private Long courseId;
        private Integer rating;
        private String comment;
        
        // Getters and Setters
        public Long getCourseId() { return courseId; }
        public void setCourseId(Long courseId) { this.courseId = courseId; }
        public Integer getRating() { return rating; }
        public void setRating(Integer rating) { this.rating = rating; }
        public String getComment() { return comment; }
        public void setComment(String comment) { this.comment = comment; }
    }

    /**
     * 评价响应DTO
     */
    public static class EvaluationResponse {
        private Long evaluationId;
        private Long courseId;
        private Long studentId;
        private Integer rating;
        private String comment;
        private String studentName;
        private LocalDateTime createTime;
        
        // Getters and Setters
        public Long getEvaluationId() { return evaluationId; }
        public void setEvaluationId(Long evaluationId) { this.evaluationId = evaluationId; }
        public Long getCourseId() { return courseId; }
        public void setCourseId(Long courseId) { this.courseId = courseId; }
        public Long getStudentId() { return studentId; }
        public void setStudentId(Long studentId) { this.studentId = studentId; }
        public Integer getRating() { return rating; }
        public void setRating(Integer rating) { this.rating = rating; }
        public String getComment() { return comment; }
        public void setComment(String comment) { this.comment = comment; }
        public String getStudentName() { return studentName; }
        public void setStudentName(String studentName) { this.studentName = studentName; }
        public LocalDateTime getCreateTime() { return createTime; }
        public void setCreateTime(LocalDateTime createTime) { this.createTime = createTime; }
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
        private Double completionRate;
        private Integer totalChapters;
        private Integer completedChapters;
        private Integer totalTasks;
        private Integer completedTasks;
        
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
        public Double getCompletionRate() { return completionRate; }
        public void setCompletionRate(Double completionRate) { this.completionRate = completionRate; }
        public Integer getTotalChapters() { return totalChapters; }
        public void setTotalChapters(Integer totalChapters) { this.totalChapters = totalChapters; }
        public Integer getCompletedChapters() { return completedChapters; }
        public void setCompletedChapters(Integer completedChapters) { this.completedChapters = completedChapters; }
        public Integer getTotalTasks() { return totalTasks; }
        public void setTotalTasks(Integer totalTasks) { this.totalTasks = totalTasks; }
        public Integer getCompletedTasks() { return completedTasks; }
        public void setCompletedTasks(Integer completedTasks) { this.completedTasks = completedTasks; }
    }

    /**
     * 资源下载响应DTO
     */
    public static class ResourceDownloadResponse {
        private Long resourceId;
        private String downloadUrl;
        private String fileName;
        private Long fileSize;
        private String contentType;
        private LocalDateTime expiryTime;
        
        // Getters and Setters
        public Long getResourceId() { return resourceId; }
        public void setResourceId(Long resourceId) { this.resourceId = resourceId; }
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
        private Integer page;
        private Integer size;
        
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
        public Integer getPage() { return page; }
        public void setPage(Integer page) { this.page = page; }
        public Integer getSize() { return size; }
        public void setSize(Integer size) { this.size = size; }
    }

    /**
     * 学习时长统计响应DTO
     */
    public static class StudyTimeStatisticsResponse {
        private Long studentId;
        private String timeRange;
        private Integer totalStudyTime;
        private Integer todayStudyTime;
        private Integer weekStudyTime;
        private Integer monthStudyTime;
        private Integer yearStudyTime;
        private Double averageDailyTime;
        private Double averageStudyTime;
        private Integer streakDays;
        private Integer maxStreakDays;
        private java.util.Map<String, Integer> dailyStudyTime;
        private java.util.Map<String, Integer> courseStudyTime;
        
        // Getters and Setters
        public Long getStudentId() { return studentId; }
        public void setStudentId(Long studentId) { this.studentId = studentId; }
        public String getTimeRange() { return timeRange; }
        public void setTimeRange(String timeRange) { this.timeRange = timeRange; }
        public Integer getTotalStudyTime() { return totalStudyTime; }
        public void setTotalStudyTime(Integer totalStudyTime) { this.totalStudyTime = totalStudyTime; }
        public Integer getTodayStudyTime() { return todayStudyTime; }
        public void setTodayStudyTime(Integer todayStudyTime) { this.todayStudyTime = todayStudyTime; }
        public Integer getWeekStudyTime() { return weekStudyTime; }
        public void setWeekStudyTime(Integer weekStudyTime) { this.weekStudyTime = weekStudyTime; }
        public Integer getMonthStudyTime() { return monthStudyTime; }
        public void setMonthStudyTime(Integer monthStudyTime) { this.monthStudyTime = monthStudyTime; }
        public Integer getYearStudyTime() { return yearStudyTime; }
        public void setYearStudyTime(Integer yearStudyTime) { this.yearStudyTime = yearStudyTime; }
        public Double getAverageDailyTime() { return averageDailyTime; }
        public void setAverageDailyTime(Double averageDailyTime) { this.averageDailyTime = averageDailyTime; }
        public Double getAverageStudyTime() { return averageStudyTime; }
        public void setAverageStudyTime(Double averageStudyTime) { this.averageStudyTime = averageStudyTime; }
        public Integer getStreakDays() { return streakDays; }
        public void setStreakDays(Integer streakDays) { this.streakDays = streakDays; }
        public Integer getMaxStreakDays() { return maxStreakDays; }
        public void setMaxStreakDays(Integer maxStreakDays) { this.maxStreakDays = maxStreakDays; }
        public java.util.Map<String, Integer> getDailyStudyTime() { return dailyStudyTime; }
        public void setDailyStudyTime(java.util.Map<String, Integer> dailyStudyTime) { this.dailyStudyTime = dailyStudyTime; }
        public java.util.Map<String, Integer> getCourseStudyTime() { return courseStudyTime; }
        public void setCourseStudyTime(java.util.Map<String, Integer> courseStudyTime) { this.courseStudyTime = courseStudyTime; }
    }

    /**
     * 学习时长记录请求DTO
     */
    public static class StudyTimeRecordRequest {
        @NotNull(message = "课程ID不能为空")
        private Long courseId;
        private Long chapterId;
        @NotNull(message = "学习时长不能为空")
        private Integer studyTime;
        private LocalDateTime startTime;
        private LocalDateTime endTime;
        
        // Getters and Setters
        public Long getCourseId() { return courseId; }
        public void setCourseId(Long courseId) { this.courseId = courseId; }
        public Long getChapterId() { return chapterId; }
        public void setChapterId(Long chapterId) { this.chapterId = chapterId; }
        public Integer getStudyTime() { return studyTime; }
        public void setStudyTime(Integer studyTime) { this.studyTime = studyTime; }
        public LocalDateTime getStartTime() { return startTime; }
        public void setStartTime(LocalDateTime startTime) { this.startTime = startTime; }
        public LocalDateTime getEndTime() { return endTime; }
        public void setEndTime(LocalDateTime endTime) { this.endTime = endTime; }
    }

    /**
     * 笔记响应DTO
     */
    public static class NoteResponse {
        private Long noteId;
        private Long courseId;
        private Long chapterId;
        private String title;
        private String content;
        private String noteType;
        private LocalDateTime createTime;
        private LocalDateTime updateTime;
        private String chapterName;
        private Boolean isPrivate;
        
        // Getters and Setters
        public Long getNoteId() { return noteId; }
        public void setNoteId(Long noteId) { this.noteId = noteId; }
        public Long getCourseId() { return courseId; }
        public void setCourseId(Long courseId) { this.courseId = courseId; }
        public Long getChapterId() { return chapterId; }
        public void setChapterId(Long chapterId) { this.chapterId = chapterId; }
        public String getTitle() { return title; }
        public void setTitle(String title) { this.title = title; }
        public String getContent() { return content; }
        public void setContent(String content) { this.content = content; }
        public String getNoteType() { return noteType; }
        public void setNoteType(String noteType) { this.noteType = noteType; }
        public LocalDateTime getCreateTime() { return createTime; }
        public void setCreateTime(LocalDateTime createTime) { this.createTime = createTime; }
        public LocalDateTime getUpdateTime() { return updateTime; }
        public void setUpdateTime(LocalDateTime updateTime) { this.updateTime = updateTime; }
        public String getChapterName() { return chapterName; }
        public void setChapterName(String chapterName) { this.chapterName = chapterName; }
        public Boolean getIsPrivate() { return isPrivate; }
        public void setIsPrivate(Boolean isPrivate) { this.isPrivate = isPrivate; }
    }

    /**
     * 笔记创建请求DTO
     */
    public static class NoteCreateRequest {
        @NotNull(message = "课程ID不能为空")
        private Long courseId;
        private Long chapterId;
        @NotBlank(message = "笔记标题不能为空")
        private String title;
        @NotBlank(message = "笔记内容不能为空")
        private String content;
        private String noteType;
        private Boolean isPrivate;
        
        // Getters and Setters
        public Long getCourseId() { return courseId; }
        public void setCourseId(Long courseId) { this.courseId = courseId; }
        public Long getChapterId() { return chapterId; }
        public void setChapterId(Long chapterId) { this.chapterId = chapterId; }
        public String getTitle() { return title; }
        public void setTitle(String title) { this.title = title; }
        public String getContent() { return content; }
        public void setContent(String content) { this.content = content; }
        public String getNoteType() { return noteType; }
        public void setNoteType(String noteType) { this.noteType = noteType; }
        public Boolean getIsPrivate() { return isPrivate; }
        public void setIsPrivate(Boolean isPrivate) { this.isPrivate = isPrivate; }
    }

    /**
     * 笔记更新请求DTO
     */
    public static class NoteUpdateRequest {
        private String title;
        private String content;
        private String noteType;
        private Boolean isPrivate;
        
        // Getters and Setters
        public String getTitle() { return title; }
        public void setTitle(String title) { this.title = title; }
        public String getContent() { return content; }
        public void setContent(String content) { this.content = content; }
        public String getNoteType() { return noteType; }
        public void setNoteType(String noteType) { this.noteType = noteType; }
        public Boolean getIsPrivate() { return isPrivate; }
        public void setIsPrivate(Boolean isPrivate) { this.isPrivate = isPrivate; }
    }

    /**
     * 证书响应DTO
     */
    public static class CertificateResponse {
        private Long certificateId;
        private Long courseId;
        private Long studentId;
        private String courseName;
        private String studentName;
        private String certificateNumber;
        private String certificateType;
        private Double finalScore;
        private LocalDateTime issueDate;
        private LocalDateTime expiryDate;
        private String certificateUrl;
        private String status;
        
        // Getters and Setters
        public Long getCertificateId() { return certificateId; }
        public void setCertificateId(Long certificateId) { this.certificateId = certificateId; }
        public Long getCourseId() { return courseId; }
        public void setCourseId(Long courseId) { this.courseId = courseId; }
        public Long getStudentId() { return studentId; }
        public void setStudentId(Long studentId) { this.studentId = studentId; }
        public String getCourseName() { return courseName; }
        public void setCourseName(String courseName) { this.courseName = courseName; }
        public String getStudentName() { return studentName; }
        public void setStudentName(String studentName) { this.studentName = studentName; }
        public String getCertificateNumber() { return certificateNumber; }
        public void setCertificateNumber(String certificateNumber) { this.certificateNumber = certificateNumber; }
        public String getCertificateType() { return certificateType; }
        public void setCertificateType(String certificateType) { this.certificateType = certificateType; }
        public Double getFinalScore() { return finalScore; }
        public void setFinalScore(Double finalScore) { this.finalScore = finalScore; }
        public LocalDateTime getIssueDate() { return issueDate; }
        public void setIssueDate(LocalDateTime issueDate) { this.issueDate = issueDate; }
        public LocalDateTime getExpiryDate() { return expiryDate; }
        public void setExpiryDate(LocalDateTime expiryDate) { this.expiryDate = expiryDate; }
        public String getCertificateUrl() { return certificateUrl; }
        public void setCertificateUrl(String certificateUrl) { this.certificateUrl = certificateUrl; }
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
    }

    /**
     * 课程日历响应DTO
     */
    public static class CourseCalendarResponse {
        private Long eventId;
        private Long courseId;
        private Long studentId;
        private String courseName;
        private String eventType;
        private String title;
        private String description;
        private LocalDateTime startTime;
        private LocalDateTime endTime;
        private String location;
        private Boolean isAllDay;
        private String status;
        private String color;
        private Integer year;
        private Integer month;
        private List<Object> events;
        
        // Getters and Setters
        public Long getEventId() { return eventId; }
        public void setEventId(Long eventId) { this.eventId = eventId; }
        public Long getCourseId() { return courseId; }
        public void setCourseId(Long courseId) { this.courseId = courseId; }
        public String getCourseName() { return courseName; }
        public void setCourseName(String courseName) { this.courseName = courseName; }
        public String getEventType() { return eventType; }
        public void setEventType(String eventType) { this.eventType = eventType; }
        public String getTitle() { return title; }
        public void setTitle(String title) { this.title = title; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
        public LocalDateTime getStartTime() { return startTime; }
        public void setStartTime(LocalDateTime startTime) { this.startTime = startTime; }
        public LocalDateTime getEndTime() { return endTime; }
        public void setEndTime(LocalDateTime endTime) { this.endTime = endTime; }
        public String getLocation() { return location; }
        public void setLocation(String location) { this.location = location; }
        public Boolean getIsAllDay() { return isAllDay; }
        public void setIsAllDay(Boolean isAllDay) { this.isAllDay = isAllDay; }
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
        public String getColor() { return color; }
        public void setColor(String color) { this.color = color; }
        public Long getStudentId() { return studentId; }
        public void setStudentId(Long studentId) { this.studentId = studentId; }
        public Integer getYear() { return year; }
        public void setYear(Integer year) { this.year = year; }
        public Integer getMonth() { return month; }
        public void setMonth(Integer month) { this.month = month; }
        public List<Object> getEvents() { return events; }
        public void setEvents(List<Object> events) { this.events = events; }
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
        private java.time.LocalDate startDate;
        private java.time.LocalDate endDate;
        private Integer totalStudyTime;
        private Integer completedCourses;
        private Integer ongoingCourses;
        private Double averageScore;
        private Integer totalAssignments;
        private Integer completedAssignments;
        private java.util.Map<String, Object> detailedStats;
        private List<CourseProgressSummary> courseProgress;
        private LocalDateTime generateTime;
        private String timeRange;
        private java.util.Map<Object, Object> reportData;
        
        // Getters and Setters
        public Long getReportId() { return reportId; }
        public void setReportId(Long reportId) { this.reportId = reportId; }
        public Long getStudentId() { return studentId; }
        public void setStudentId(Long studentId) { this.studentId = studentId; }
        public String getReportType() { return reportType; }
        public void setReportType(String reportType) { this.reportType = reportType; }
        public String getReportPeriod() { return reportPeriod; }
        public void setReportPeriod(String reportPeriod) { this.reportPeriod = reportPeriod; }
        public java.time.LocalDate getStartDate() { return startDate; }
        public void setStartDate(java.time.LocalDate startDate) { this.startDate = startDate; }
        public java.time.LocalDate getEndDate() { return endDate; }
        public void setEndDate(java.time.LocalDate endDate) { this.endDate = endDate; }
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
        public java.util.Map<String, Object> getDetailedStats() { return detailedStats; }
        public void setDetailedStats(java.util.Map<String, Object> detailedStats) { this.detailedStats = detailedStats; }
        public List<CourseProgressSummary> getCourseProgress() { return courseProgress; }
        public void setCourseProgress(List<CourseProgressSummary> courseProgress) { this.courseProgress = courseProgress; }
        public LocalDateTime getGenerateTime() { return generateTime; }
        public void setGenerateTime(LocalDateTime generateTime) { this.generateTime = generateTime; }
        public String getTimeRange() { return timeRange; }
        public void setTimeRange(String timeRange) { this.timeRange = timeRange; }
        public java.util.Map<Object, Object> getReportData() { return reportData; }
        public void setReportData(java.util.Map<Object, Object> reportData) { this.reportData = reportData; }
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
     * 导出响应DTO
     */
    public static class ExportResponse {
        private String exportId;
        private Long studentId;
        private String exportType;
        private String status;
        private String fileName;
        private String downloadUrl;
        private String exportUrl;
        private LocalDateTime createTime;
        private LocalDateTime expiryTime;
        private LocalDateTime exportTime;
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
        public Long getStudentId() { return studentId; }
        public void setStudentId(Long studentId) { this.studentId = studentId; }
        public String getExportUrl() { return exportUrl; }
        public void setExportUrl(String exportUrl) { this.exportUrl = exportUrl; }
        public LocalDateTime getExportTime() { return exportTime; }
        public void setExportTime(LocalDateTime exportTime) { this.exportTime = exportTime; }
    }

    /**
     * 学习数据导出请求DTO
     */
    public static class LearningDataExportRequest {
        @NotBlank(message = "导出类型不能为空")
        private String exportType;
        @NotBlank(message = "数据格式不能为空")
        private String dataFormat;
        private java.time.LocalDate startDate;
        private java.time.LocalDate endDate;
        private List<Long> courseIds;
        private List<String> dataTypes;
        private Boolean includePersonalInfo;
        private Boolean includeStatistics;
        
        // Getters and Setters
        public String getExportType() { return exportType; }
        public void setExportType(String exportType) { this.exportType = exportType; }
        public String getDataFormat() { return dataFormat; }
        public void setDataFormat(String dataFormat) { this.dataFormat = dataFormat; }
        public java.time.LocalDate getStartDate() { return startDate; }
        public void setStartDate(java.time.LocalDate startDate) { this.startDate = startDate; }
        public java.time.LocalDate getEndDate() { return endDate; }
        public void setEndDate(java.time.LocalDate endDate) { this.endDate = endDate; }
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
     * 章节创建请求DTO
     */
    public static class ChapterCreateRequest {
        @NotBlank(message = "章节名称不能为空")
        private String chapterName;
        private String description;
        private Integer orderIndex;
        private String content;
        private String videoUrl;
        private Integer duration;
        
        // Getters and Setters
        public String getChapterName() { return chapterName; }
        public void setChapterName(String chapterName) { this.chapterName = chapterName; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
        public Integer getOrderIndex() { return orderIndex; }
        public void setOrderIndex(Integer orderIndex) { this.orderIndex = orderIndex; }
        public String getContent() { return content; }
        public void setContent(String content) { this.content = content; }
        public String getVideoUrl() { return videoUrl; }
        public void setVideoUrl(String videoUrl) { this.videoUrl = videoUrl; }
        public Integer getDuration() { return duration; }
        public void setDuration(Integer duration) { this.duration = duration; }
    }

    /**
     * 章节更新请求DTO
     */
    public static class ChapterUpdateRequest {
        private String chapterName;
        private String description;
        private Integer orderIndex;
        private String content;
        private String videoUrl;
        private Integer duration;
        
        // Getters and Setters
        public String getChapterName() { return chapterName; }
        public void setChapterName(String chapterName) { this.chapterName = chapterName; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
        public Integer getOrderIndex() { return orderIndex; }
        public void setOrderIndex(Integer orderIndex) { this.orderIndex = orderIndex; }
        public String getContent() { return content; }
        public void setContent(String content) { this.content = content; }
        public String getVideoUrl() { return videoUrl; }
        public void setVideoUrl(String videoUrl) { this.videoUrl = videoUrl; }
        public Integer getDuration() { return duration; }
        public void setDuration(Integer duration) { this.duration = duration; }
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
        private Long courseId;
        private String courseName;
        private Integer totalStudents;
        private Integer activeStudents;
        private Double averageProgress;
        private Integer totalChapters;
        private Integer completedChapters;
        private Double completionRate;
        private Integer totalStudyTime;
        private LocalDateTime lastUpdateTime;
        
        // Getters and Setters
        public Long getCourseId() { return courseId; }
        public void setCourseId(Long courseId) { this.courseId = courseId; }
        public String getCourseName() { return courseName; }
        public void setCourseName(String courseName) { this.courseName = courseName; }
        public Integer getTotalStudents() { return totalStudents; }
        public void setTotalStudents(Integer totalStudents) { this.totalStudents = totalStudents; }
        public Integer getActiveStudents() { return activeStudents; }
        public void setActiveStudents(Integer activeStudents) { this.activeStudents = activeStudents; }
        public Double getAverageProgress() { return averageProgress; }
        public void setAverageProgress(Double averageProgress) { this.averageProgress = averageProgress; }
        public Integer getTotalChapters() { return totalChapters; }
        public void setTotalChapters(Integer totalChapters) { this.totalChapters = totalChapters; }
        public Integer getCompletedChapters() { return completedChapters; }
        public void setCompletedChapters(Integer completedChapters) { this.completedChapters = completedChapters; }
        public Double getCompletionRate() { return completionRate; }
        public void setCompletionRate(Double completionRate) { this.completionRate = completionRate; }
        public Integer getTotalStudyTime() { return totalStudyTime; }
        public void setTotalStudyTime(Integer totalStudyTime) { this.totalStudyTime = totalStudyTime; }
        public LocalDateTime getLastUpdateTime() { return lastUpdateTime; }
        public void setLastUpdateTime(LocalDateTime lastUpdateTime) { this.lastUpdateTime = lastUpdateTime; }
    }
}