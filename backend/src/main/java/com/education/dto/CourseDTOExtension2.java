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
 * 课程相关DTO扩展类2
 * 包含更多课程相关的DTO定义
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
public class CourseDTOExtension2 {

    /**
     * 公告响应DTO
     */
    public static class AnnouncementResponse {
        private Long announcementId;
        private String title;
        private String content;
        private String priority;
        private LocalDateTime publishTime;
        private LocalDateTime expiryTime;
        private String publisherName;
        private Boolean isRead;
        private List<String> attachments;
        
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
        public LocalDateTime getExpiryTime() { return expiryTime; }
        public void setExpiryTime(LocalDateTime expiryTime) { this.expiryTime = expiryTime; }
        public String getPublisherName() { return publisherName; }
        public void setPublisherName(String publisherName) { this.publisherName = publisherName; }
        public Boolean getIsRead() { return isRead; }
        public void setIsRead(Boolean isRead) { this.isRead = isRead; }
        public List<String> getAttachments() { return attachments; }
        public void setAttachments(List<String> attachments) { this.attachments = attachments; }
    }

    /**
     * 课程评价请求DTO
     */
    public static class CourseEvaluationRequest {
        @NotNull(message = "评分不能为空")
        private Integer rating;
        
        private String comment;
        private List<String> tags;
        private Boolean isAnonymous;
        
        // Getters and Setters
        public Integer getRating() { return rating; }
        public void setRating(Integer rating) { this.rating = rating; }
        public String getComment() { return comment; }
        public void setComment(String comment) { this.comment = comment; }
        public List<String> getTags() { return tags; }
        public void setTags(List<String> tags) { this.tags = tags; }
        public Boolean getIsAnonymous() { return isAnonymous; }
        public void setIsAnonymous(Boolean isAnonymous) { this.isAnonymous = isAnonymous; }
    }

    /**
     * 评价响应DTO
     */
    public static class EvaluationResponse {
        private Long evaluationId;
        private Integer rating;
        private String comment;
        private List<String> tags;
        private LocalDateTime createTime;
        private String studentName;
        private Boolean isAnonymous;
        private Integer helpfulCount;
        private Boolean isHelpful;
        
        // Getters and Setters
        public Long getEvaluationId() { return evaluationId; }
        public void setEvaluationId(Long evaluationId) { this.evaluationId = evaluationId; }
        public Integer getRating() { return rating; }
        public void setRating(Integer rating) { this.rating = rating; }
        public String getComment() { return comment; }
        public void setComment(String comment) { this.comment = comment; }
        public List<String> getTags() { return tags; }
        public void setTags(List<String> tags) { this.tags = tags; }
        public LocalDateTime getCreateTime() { return createTime; }
        public void setCreateTime(LocalDateTime createTime) { this.createTime = createTime; }
        public String getStudentName() { return studentName; }
        public void setStudentName(String studentName) { this.studentName = studentName; }
        public Boolean getIsAnonymous() { return isAnonymous; }
        public void setIsAnonymous(Boolean isAnonymous) { this.isAnonymous = isAnonymous; }
        public Integer getHelpfulCount() { return helpfulCount; }
        public void setHelpfulCount(Integer helpfulCount) { this.helpfulCount = helpfulCount; }
        public Boolean getIsHelpful() { return isHelpful; }
        public void setIsHelpful(Boolean isHelpful) { this.isHelpful = isHelpful; }
    }

    /**
     * 课程搜索请求DTO
     */
    public static class CourseSearchRequest {
        private String keyword;
        private String category;
        private String difficulty;
        private String status;
        private Integer minCredit;
        private Integer maxCredit;
        private Double minRating;
        private String sortBy;
        private String sortOrder;
        
        // Getters and Setters
        public String getKeyword() { return keyword; }
        public void setKeyword(String keyword) { this.keyword = keyword; }
        public String getCategory() { return category; }
        public void setCategory(String category) { this.category = category; }
        public String getDifficulty() { return difficulty; }
        public void setDifficulty(String difficulty) { this.difficulty = difficulty; }
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
        public Integer getMinCredit() { return minCredit; }
        public void setMinCredit(Integer minCredit) { this.minCredit = minCredit; }
        public Integer getMaxCredit() { return maxCredit; }
        public void setMaxCredit(Integer maxCredit) { this.maxCredit = maxCredit; }
        public Double getMinRating() { return minRating; }
        public void setMinRating(Double minRating) { this.minRating = minRating; }
        public String getSortBy() { return sortBy; }
        public void setSortBy(String sortBy) { this.sortBy = sortBy; }
        public String getSortOrder() { return sortOrder; }
        public void setSortOrder(String sortOrder) { this.sortOrder = sortOrder; }
    }

    /**
     * 学习统计响应DTO
     */
    public static class LearningStatisticsResponse {
        private Long courseId;
        private String courseName;
        private Integer totalStudyTime;
        private Integer totalSessions;
        private Double averageSessionTime;
        private Integer completedChapters;
        private Integer totalChapters;
        private Double completionRate;
        private Integer streak;
        private LocalDateTime lastStudyTime;
        private List<DailyStudyTime> dailyStudyTimes;
        private List<ChapterStudyTime> chapterStudyTimes;
        
        // Getters and Setters
        public Long getCourseId() { return courseId; }
        public void setCourseId(Long courseId) { this.courseId = courseId; }
        public String getCourseName() { return courseName; }
        public void setCourseName(String courseName) { this.courseName = courseName; }
        public Integer getTotalStudyTime() { return totalStudyTime; }
        public void setTotalStudyTime(Integer totalStudyTime) { this.totalStudyTime = totalStudyTime; }
        public Integer getTotalSessions() { return totalSessions; }
        public void setTotalSessions(Integer totalSessions) { this.totalSessions = totalSessions; }
        public Double getAverageSessionTime() { return averageSessionTime; }
        public void setAverageSessionTime(Double averageSessionTime) { this.averageSessionTime = averageSessionTime; }
        public Integer getCompletedChapters() { return completedChapters; }
        public void setCompletedChapters(Integer completedChapters) { this.completedChapters = completedChapters; }
        public Integer getTotalChapters() { return totalChapters; }
        public void setTotalChapters(Integer totalChapters) { this.totalChapters = totalChapters; }
        public Double getCompletionRate() { return completionRate; }
        public void setCompletionRate(Double completionRate) { this.completionRate = completionRate; }
        public Integer getStreak() { return streak; }
        public void setStreak(Integer streak) { this.streak = streak; }
        public LocalDateTime getLastStudyTime() { return lastStudyTime; }
        public void setLastStudyTime(LocalDateTime lastStudyTime) { this.lastStudyTime = lastStudyTime; }
        public List<DailyStudyTime> getDailyStudyTimes() { return dailyStudyTimes; }
        public void setDailyStudyTimes(List<DailyStudyTime> dailyStudyTimes) { this.dailyStudyTimes = dailyStudyTimes; }
        public List<ChapterStudyTime> getChapterStudyTimes() { return chapterStudyTimes; }
        public void setChapterStudyTimes(List<ChapterStudyTime> chapterStudyTimes) { this.chapterStudyTimes = chapterStudyTimes; }
        
        public static class DailyStudyTime {
            private String date;
            private Integer studyTime;
            private Integer sessions;
            
            // Getters and Setters
            public String getDate() { return date; }
            public void setDate(String date) { this.date = date; }
            public Integer getStudyTime() { return studyTime; }
            public void setStudyTime(Integer studyTime) { this.studyTime = studyTime; }
            public Integer getSessions() { return sessions; }
            public void setSessions(Integer sessions) { this.sessions = sessions; }
        }
        
        public static class ChapterStudyTime {
            private Long chapterId;
            private String chapterTitle;
            private Integer studyTime;
            private Integer sessions;
            private Boolean isCompleted;
            
            // Getters and Setters
            public Long getChapterId() { return chapterId; }
            public void setChapterId(Long chapterId) { this.chapterId = chapterId; }
            public String getChapterTitle() { return chapterTitle; }
            public void setChapterTitle(String chapterTitle) { this.chapterTitle = chapterTitle; }
            public Integer getStudyTime() { return studyTime; }
            public void setStudyTime(Integer studyTime) { this.studyTime = studyTime; }
            public Integer getSessions() { return sessions; }
            public void setSessions(Integer sessions) { this.sessions = sessions; }
            public Boolean getIsCompleted() { return isCompleted; }
            public void setIsCompleted(Boolean isCompleted) { this.isCompleted = isCompleted; }
        }
    }

    /**
     * 学习时间统计响应DTO
     */
    public static class StudyTimeStatisticsResponse {
        private String timeRange;
        private Integer totalStudyTime;
        private Integer averageDailyTime;
        private Integer studyDays;
        private Integer totalDays;
        private Double consistency;
        private List<CourseStudyTime> courseStudyTimes;
        private List<DailyStudyTime> dailyBreakdown;
        
        // Getters and Setters
        public String getTimeRange() { return timeRange; }
        public void setTimeRange(String timeRange) { this.timeRange = timeRange; }
        public Integer getTotalStudyTime() { return totalStudyTime; }
        public void setTotalStudyTime(Integer totalStudyTime) { this.totalStudyTime = totalStudyTime; }
        public Integer getAverageDailyTime() { return averageDailyTime; }
        public void setAverageDailyTime(Integer averageDailyTime) { this.averageDailyTime = averageDailyTime; }
        public Integer getStudyDays() { return studyDays; }
        public void setStudyDays(Integer studyDays) { this.studyDays = studyDays; }
        public Integer getTotalDays() { return totalDays; }
        public void setTotalDays(Integer totalDays) { this.totalDays = totalDays; }
        public Double getConsistency() { return consistency; }
        public void setConsistency(Double consistency) { this.consistency = consistency; }
        public List<CourseStudyTime> getCourseStudyTimes() { return courseStudyTimes; }
        public void setCourseStudyTimes(List<CourseStudyTime> courseStudyTimes) { this.courseStudyTimes = courseStudyTimes; }
        public List<DailyStudyTime> getDailyBreakdown() { return dailyBreakdown; }
        public void setDailyBreakdown(List<DailyStudyTime> dailyBreakdown) { this.dailyBreakdown = dailyBreakdown; }
        
        public static class CourseStudyTime {
            private Long courseId;
            private String courseName;
            private Integer studyTime;
            private Double percentage;
            
            // Getters and Setters
            public Long getCourseId() { return courseId; }
            public void setCourseId(Long courseId) { this.courseId = courseId; }
            public String getCourseName() { return courseName; }
            public void setCourseName(String courseName) { this.courseName = courseName; }
            public Integer getStudyTime() { return studyTime; }
            public void setStudyTime(Integer studyTime) { this.studyTime = studyTime; }
            public Double getPercentage() { return percentage; }
            public void setPercentage(Double percentage) { this.percentage = percentage; }
        }
        
        public static class DailyStudyTime {
            private String date;
            private Integer studyTime;
            private Integer sessions;
            
            // Getters and Setters
            public String getDate() { return date; }
            public void setDate(String date) { this.date = date; }
            public Integer getStudyTime() { return studyTime; }
            public void setStudyTime(Integer studyTime) { this.studyTime = studyTime; }
            public Integer getSessions() { return sessions; }
            public void setSessions(Integer sessions) { this.sessions = sessions; }
        }
    }

    /**
     * 学习时间记录请求DTO
     */
    public static class StudyTimeRecordRequest {
        @NotNull(message = "课程ID不能为空")
        private Long courseId;
        
        private Long chapterId;
        
        @NotNull(message = "学习时间不能为空")
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
        private String title;
        private String content;
        private Long chapterId;
        private String chapterTitle;
        private String noteType;
        private List<String> tags;
        private LocalDateTime createTime;
        private LocalDateTime updateTime;
        private Boolean isPublic;
        
        // Getters and Setters
        public Long getNoteId() { return noteId; }
        public void setNoteId(Long noteId) { this.noteId = noteId; }
        public String getTitle() { return title; }
        public void setTitle(String title) { this.title = title; }
        public String getContent() { return content; }
        public void setContent(String content) { this.content = content; }
        public Long getChapterId() { return chapterId; }
        public void setChapterId(Long chapterId) { this.chapterId = chapterId; }
        public String getChapterTitle() { return chapterTitle; }
        public void setChapterTitle(String chapterTitle) { this.chapterTitle = chapterTitle; }
        public String getNoteType() { return noteType; }
        public void setNoteType(String noteType) { this.noteType = noteType; }
        public List<String> getTags() { return tags; }
        public void setTags(List<String> tags) { this.tags = tags; }
        public LocalDateTime getCreateTime() { return createTime; }
        public void setCreateTime(LocalDateTime createTime) { this.createTime = createTime; }
        public LocalDateTime getUpdateTime() { return updateTime; }
        public void setUpdateTime(LocalDateTime updateTime) { this.updateTime = updateTime; }
        public Boolean getIsPublic() { return isPublic; }
        public void setIsPublic(Boolean isPublic) { this.isPublic = isPublic; }
    }

    /**
     * 笔记创建请求DTO
     */
    public static class NoteCreateRequest {
        @NotBlank(message = "标题不能为空")
        private String title;
        
        @NotBlank(message = "内容不能为空")
        private String content;
        
        private Long chapterId;
        private String noteType;
        private List<String> tags;
        private Boolean isPublic;
        
        // Getters and Setters
        public String getTitle() { return title; }
        public void setTitle(String title) { this.title = title; }
        public String getContent() { return content; }
        public void setContent(String content) { this.content = content; }
        public Long getChapterId() { return chapterId; }
        public void setChapterId(Long chapterId) { this.chapterId = chapterId; }
        public String getNoteType() { return noteType; }
        public void setNoteType(String noteType) { this.noteType = noteType; }
        public List<String> getTags() { return tags; }
        public void setTags(List<String> tags) { this.tags = tags; }
        public Boolean getIsPublic() { return isPublic; }
        public void setIsPublic(Boolean isPublic) { this.isPublic = isPublic; }
    }

    /**
     * 笔记更新请求DTO
     */
    public static class NoteUpdateRequest {
        private String title;
        private String content;
        private String noteType;
        private List<String> tags;
        private Boolean isPublic;
        
        // Getters and Setters
        public String getTitle() { return title; }
        public void setTitle(String title) { this.title = title; }
        public String getContent() { return content; }
        public void setContent(String content) { this.content = content; }
        public String getNoteType() { return noteType; }
        public void setNoteType(String noteType) { this.noteType = noteType; }
        public List<String> getTags() { return tags; }
        public void setTags(List<String> tags) { this.tags = tags; }
        public Boolean getIsPublic() { return isPublic; }
        public void setIsPublic(Boolean isPublic) { this.isPublic = isPublic; }
    }

    /**
     * 讨论响应DTO
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
        private Boolean isPinned;
        private List<DiscussionReply> replies;
        
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
        public Boolean getIsPinned() { return isPinned; }
        public void setIsPinned(Boolean isPinned) { this.isPinned = isPinned; }
        public List<DiscussionReply> getReplies() { return replies; }
        public void setReplies(List<DiscussionReply> replies) { this.replies = replies; }
        
        public static class DiscussionReply {
            private Long replyId;
            private String content;
            private String authorName;
            private LocalDateTime createTime;
            private Integer likeCount;
            private Boolean isLiked;
            
            // Getters and Setters
            public Long getReplyId() { return replyId; }
            public void setReplyId(Long replyId) { this.replyId = replyId; }
            public String getContent() { return content; }
            public void setContent(String content) { this.content = content; }
            public String getAuthorName() { return authorName; }
            public void setAuthorName(String authorName) { this.authorName = authorName; }
            public LocalDateTime getCreateTime() { return createTime; }
            public void setCreateTime(LocalDateTime createTime) { this.createTime = createTime; }
            public Integer getLikeCount() { return likeCount; }
            public void setLikeCount(Integer likeCount) { this.likeCount = likeCount; }
            public Boolean getIsLiked() { return isLiked; }
            public void setIsLiked(Boolean isLiked) { this.isLiked = isLiked; }
        }
    }

    /**
     * 讨论创建请求DTO
     */
    public static class DiscussionCreateRequest {
        @NotBlank(message = "标题不能为空")
        private String title;
        
        @NotBlank(message = "内容不能为空")
        private String content;
        
        private List<String> attachments;
        
        // Getters and Setters
        public String getTitle() { return title; }
        public void setTitle(String title) { this.title = title; }
        public String getContent() { return content; }
        public void setContent(String content) { this.content = content; }
        public List<String> getAttachments() { return attachments; }
        public void setAttachments(List<String> attachments) { this.attachments = attachments; }
    }

    /**
     * 讨论回复请求DTO
     */
    public static class DiscussionReplyRequest {
        @NotBlank(message = "回复内容不能为空")
        private String content;
        
        private List<String> attachments;
        
        // Getters and Setters
        public String getContent() { return content; }
        public void setContent(String content) { this.content = content; }
        public List<String> getAttachments() { return attachments; }
        public void setAttachments(List<String> attachments) { this.attachments = attachments; }
    }

    /**
     * 证书响应DTO
     */
    public static class CertificateResponse {
        private Long certificateId;
        private String certificateNumber;
        private String courseName;
        private String studentName;
        private LocalDateTime issueDate;
        private String certificateUrl;
        private String status;
        private Double finalScore;
        private String grade;
        
        // Getters and Setters
        public Long getCertificateId() { return certificateId; }
        public void setCertificateId(Long certificateId) { this.certificateId = certificateId; }
        public String getCertificateNumber() { return certificateNumber; }
        public void setCertificateNumber(String certificateNumber) { this.certificateNumber = certificateNumber; }
        public String getCourseName() { return courseName; }
        public void setCourseName(String courseName) { this.courseName = courseName; }
        public String getStudentName() { return studentName; }
        public void setStudentName(String studentName) { this.studentName = studentName; }
        public LocalDateTime getIssueDate() { return issueDate; }
        public void setIssueDate(LocalDateTime issueDate) { this.issueDate = issueDate; }
        public String getCertificateUrl() { return certificateUrl; }
        public void setCertificateUrl(String certificateUrl) { this.certificateUrl = certificateUrl; }
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
        public Double getFinalScore() { return finalScore; }
        public void setFinalScore(Double finalScore) { this.finalScore = finalScore; }
        public String getGrade() { return grade; }
        public void setGrade(String grade) { this.grade = grade; }
    }
}