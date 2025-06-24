package com.education.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

/**
 * 课程相关DTO扩展6 - 包含学习时长、笔记、讨论、证书等相关类
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
public class CourseDTOExtension6 {

    /**
     * 学习时长统计响应DTO
     */
    public static class StudyTimeStatisticsResponse {
        private Long studentId;
        private Integer totalStudyTime;
        private Integer todayStudyTime;
        private Integer weekStudyTime;
        private Integer monthStudyTime;
        private Integer yearStudyTime;
        private Double averageDailyTime;
        private Integer streakDays;
        private Integer maxStreakDays;
        private Map<String, Integer> dailyStudyTime;
        private Map<String, Integer> courseStudyTime;
        
        // Getters and Setters
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
        public Integer getYearStudyTime() { return yearStudyTime; }
        public void setYearStudyTime(Integer yearStudyTime) { this.yearStudyTime = yearStudyTime; }
        public Double getAverageDailyTime() { return averageDailyTime; }
        public void setAverageDailyTime(Double averageDailyTime) { this.averageDailyTime = averageDailyTime; }
        public Integer getStreakDays() { return streakDays; }
        public void setStreakDays(Integer streakDays) { this.streakDays = streakDays; }
        public Integer getMaxStreakDays() { return maxStreakDays; }
        public void setMaxStreakDays(Integer maxStreakDays) { this.maxStreakDays = maxStreakDays; }
        public Map<String, Integer> getDailyStudyTime() { return dailyStudyTime; }
        public void setDailyStudyTime(Map<String, Integer> dailyStudyTime) { this.dailyStudyTime = dailyStudyTime; }
        public Map<String, Integer> getCourseStudyTime() { return courseStudyTime; }
        public void setCourseStudyTime(Map<String, Integer> courseStudyTime) { this.courseStudyTime = courseStudyTime; }
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
     * 讨论响应DTO
     */
    public static class DiscussionResponse {
        private Long discussionId;
        private Long courseId;
        private String title;
        private String content;
        private String discussionType;
        private Long authorId;
        private String authorName;
        private LocalDateTime createTime;
        private Integer replyCount;
        private Integer likeCount;
        private Boolean isLiked;
        private Boolean isPinned;
        private List<DiscussionReplyResponse> replies;
        
        // Getters and Setters
        public Long getDiscussionId() { return discussionId; }
        public void setDiscussionId(Long discussionId) { this.discussionId = discussionId; }
        public Long getCourseId() { return courseId; }
        public void setCourseId(Long courseId) { this.courseId = courseId; }
        public String getTitle() { return title; }
        public void setTitle(String title) { this.title = title; }
        public String getContent() { return content; }
        public void setContent(String content) { this.content = content; }
        public String getDiscussionType() { return discussionType; }
        public void setDiscussionType(String discussionType) { this.discussionType = discussionType; }
        public Long getAuthorId() { return authorId; }
        public void setAuthorId(Long authorId) { this.authorId = authorId; }
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
        public Boolean getIsPinned() { return isPinned; }
        public void setIsPinned(Boolean isPinned) { this.isPinned = isPinned; }
        public List<DiscussionReplyResponse> getReplies() { return replies; }
        public void setReplies(List<DiscussionReplyResponse> replies) { this.replies = replies; }
    }

    /**
     * 讨论创建请求DTO
     */
    public static class DiscussionCreateRequest {
        @NotNull(message = "课程ID不能为空")
        private Long courseId;
        @NotBlank(message = "讨论标题不能为空")
        private String title;
        @NotBlank(message = "讨论内容不能为空")
        private String content;
        private String discussionType;
        
        // Getters and Setters
        public Long getCourseId() { return courseId; }
        public void setCourseId(Long courseId) { this.courseId = courseId; }
        public String getTitle() { return title; }
        public void setTitle(String title) { this.title = title; }
        public String getContent() { return content; }
        public void setContent(String content) { this.content = content; }
        public String getDiscussionType() { return discussionType; }
        public void setDiscussionType(String discussionType) { this.discussionType = discussionType; }
    }

    /**
     * 讨论回复请求DTO
     */
    public static class DiscussionReplyRequest {
        @NotBlank(message = "回复内容不能为空")
        private String content;
        private Long parentReplyId;
        
        // Getters and Setters
        public String getContent() { return content; }
        public void setContent(String content) { this.content = content; }
        public Long getParentReplyId() { return parentReplyId; }
        public void setParentReplyId(Long parentReplyId) { this.parentReplyId = parentReplyId; }
    }

    /**
     * 讨论回复响应DTO
     */
    public static class DiscussionReplyResponse {
        private Long replyId;
        private Long discussionId;
        private String content;
        private Long authorId;
        private String authorName;
        private LocalDateTime createTime;
        private Long parentReplyId;
        private String parentAuthorName;
        private Integer likeCount;
        private Boolean isLiked;
        
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
        public String getParentAuthorName() { return parentAuthorName; }
        public void setParentAuthorName(String parentAuthorName) { this.parentAuthorName = parentAuthorName; }
        public Integer getLikeCount() { return likeCount; }
        public void setLikeCount(Integer likeCount) { this.likeCount = likeCount; }
        public Boolean getIsLiked() { return isLiked; }
        public void setIsLiked(Boolean isLiked) { this.isLiked = isLiked; }
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
}