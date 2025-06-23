package com.education.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import java.time.LocalDateTime;
import java.util.List;

/**
 * 课程相关DTO扩展4 - 包含更多内部类
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
public class CourseDTOExtension4 {

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
        private List<ChapterProgressResponse> chapterProgress;
        
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
        public List<ChapterProgressResponse> getChapterProgress() { return chapterProgress; }
        public void setChapterProgress(List<ChapterProgressResponse> chapterProgress) { this.chapterProgress = chapterProgress; }
    }

    /**
     * 章节进度响应DTO
     */
    public static class ChapterProgressResponse {
        private Long chapterId;
        private String chapterName;
        private Boolean isCompleted;
        private Integer studyTime;
        private LocalDateTime completedTime;
        
        // Getters and Setters
        public Long getChapterId() { return chapterId; }
        public void setChapterId(Long chapterId) { this.chapterId = chapterId; }
        public String getChapterName() { return chapterName; }
        public void setChapterName(String chapterName) { this.chapterName = chapterName; }
        public Boolean getIsCompleted() { return isCompleted; }
        public void setIsCompleted(Boolean isCompleted) { this.isCompleted = isCompleted; }
        public Integer getStudyTime() { return studyTime; }
        public void setStudyTime(Integer studyTime) { this.studyTime = studyTime; }
        public LocalDateTime getCompletedTime() { return completedTime; }
        public void setCompletedTime(LocalDateTime completedTime) { this.completedTime = completedTime; }
    }

    /**
     * 进度更新请求DTO
     */
    public static class ProgressUpdateRequest {
        @NotNull(message = "课程ID不能为空")
        private Long courseId;
        private Long chapterId;
        private Double progress;
        private Integer studyTime;
        private Boolean isCompleted;
        
        // Getters and Setters
        public Long getCourseId() { return courseId; }
        public void setCourseId(Long courseId) { this.courseId = courseId; }
        public Long getChapterId() { return chapterId; }
        public void setChapterId(Long chapterId) { this.chapterId = chapterId; }
        public Double getProgress() { return progress; }
        public void setProgress(Double progress) { this.progress = progress; }
        public Integer getStudyTime() { return studyTime; }
        public void setStudyTime(Integer studyTime) { this.studyTime = studyTime; }
        public Boolean getIsCompleted() { return isCompleted; }
        public void setIsCompleted(Boolean isCompleted) { this.isCompleted = isCompleted; }
    }
}