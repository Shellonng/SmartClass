package com.education.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import java.time.LocalDateTime;
import java.util.List;
import java.math.BigDecimal;

/**
 * 用户DTO扩展类 - 第8部分
 * 包含学习时间统计、在线状态、用户资料、反馈等相关DTO
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
public class UserDTOExtension8 {

    /**
     * 学习时间统计响应DTO
     */
    public static class StudyTimeStatisticsResponse {
        private String timeRange;
        private Integer totalMinutes;
        private Integer totalSessions;
        private BigDecimal averageSessionLength;
        private Integer longestSession;
        private Integer shortestSession;
        private List<DailyStudyTime> dailyBreakdown;
        private List<SubjectStudyTime> subjectBreakdown;
        private String mostActiveDay;
        private String mostActiveHour;
        
        // Getters and Setters
        public String getTimeRange() { return timeRange; }
        public void setTimeRange(String timeRange) { this.timeRange = timeRange; }
        public Integer getTotalMinutes() { return totalMinutes; }
        public void setTotalMinutes(Integer totalMinutes) { this.totalMinutes = totalMinutes; }
        public Integer getTotalSessions() { return totalSessions; }
        public void setTotalSessions(Integer totalSessions) { this.totalSessions = totalSessions; }
        public BigDecimal getAverageSessionLength() { return averageSessionLength; }
        public void setAverageSessionLength(BigDecimal averageSessionLength) { this.averageSessionLength = averageSessionLength; }
        public Integer getLongestSession() { return longestSession; }
        public void setLongestSession(Integer longestSession) { this.longestSession = longestSession; }
        public Integer getShortestSession() { return shortestSession; }
        public void setShortestSession(Integer shortestSession) { this.shortestSession = shortestSession; }
        public List<DailyStudyTime> getDailyBreakdown() { return dailyBreakdown; }
        public void setDailyBreakdown(List<DailyStudyTime> dailyBreakdown) { this.dailyBreakdown = dailyBreakdown; }
        public List<SubjectStudyTime> getSubjectBreakdown() { return subjectBreakdown; }
        public void setSubjectBreakdown(List<SubjectStudyTime> subjectBreakdown) { this.subjectBreakdown = subjectBreakdown; }
        public String getMostActiveDay() { return mostActiveDay; }
        public void setMostActiveDay(String mostActiveDay) { this.mostActiveDay = mostActiveDay; }
        public String getMostActiveHour() { return mostActiveHour; }
        public void setMostActiveHour(String mostActiveHour) { this.mostActiveHour = mostActiveHour; }
        
        public static class DailyStudyTime {
            private String date;
            private Integer minutes;
            private Integer sessions;
            
            // Getters and Setters
            public String getDate() { return date; }
            public void setDate(String date) { this.date = date; }
            public Integer getMinutes() { return minutes; }
            public void setMinutes(Integer minutes) { this.minutes = minutes; }
            public Integer getSessions() { return sessions; }
            public void setSessions(Integer sessions) { this.sessions = sessions; }
        }
        
        public static class SubjectStudyTime {
            private String subject;
            private Integer minutes;
            private BigDecimal percentage;
            
            // Getters and Setters
            public String getSubject() { return subject; }
            public void setSubject(String subject) { this.subject = subject; }
            public Integer getMinutes() { return minutes; }
            public void setMinutes(Integer minutes) { this.minutes = minutes; }
            public BigDecimal getPercentage() { return percentage; }
            public void setPercentage(BigDecimal percentage) { this.percentage = percentage; }
        }
    }

    /**
     * 在线状态响应DTO
     */
    public static class OnlineStatusResponse {
        private Boolean isOnline;
        private String status; // ONLINE, OFFLINE, AWAY, BUSY
        private String customMessage;
        private LocalDateTime lastSeen;
        private LocalDateTime statusChangedTime;
        private Boolean showOnlineStatus;
        
        // Getters and Setters
        public Boolean getIsOnline() { return isOnline; }
        public void setIsOnline(Boolean isOnline) { this.isOnline = isOnline; }
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
        public String getCustomMessage() { return customMessage; }
        public void setCustomMessage(String customMessage) { this.customMessage = customMessage; }
        public LocalDateTime getLastSeen() { return lastSeen; }
        public void setLastSeen(LocalDateTime lastSeen) { this.lastSeen = lastSeen; }
        public LocalDateTime getStatusChangedTime() { return statusChangedTime; }
        public void setStatusChangedTime(LocalDateTime statusChangedTime) { this.statusChangedTime = statusChangedTime; }
        public Boolean getShowOnlineStatus() { return showOnlineStatus; }
        public void setShowOnlineStatus(Boolean showOnlineStatus) { this.showOnlineStatus = showOnlineStatus; }
    }

    /**
     * 在线状态更新请求DTO
     */
    public static class OnlineStatusUpdateRequest {
        @NotBlank(message = "状态不能为空")
        private String status; // ONLINE, OFFLINE, AWAY, BUSY
        
        private String customMessage;
        private Boolean showOnlineStatus;
        
        // Getters and Setters
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
        public String getCustomMessage() { return customMessage; }
        public void setCustomMessage(String customMessage) { this.customMessage = customMessage; }
        public Boolean getShowOnlineStatus() { return showOnlineStatus; }
        public void setShowOnlineStatus(Boolean showOnlineStatus) { this.showOnlineStatus = showOnlineStatus; }
    }

    /**
     * 用户资料响应DTO
     */
    public static class UserProfileResponse {
        private Long userId;
        private String username;
        private String realName;
        private String email;
        private String phone;
        private String avatar;
        private String userType;
        private String gender;
        private String department;
        private String bio;
        private String location;
        private String website;
        private String status;
        private LocalDateTime createTime;
        private LocalDateTime lastLoginTime;
        private Boolean isOnline;
        private Integer followersCount;
        private Integer followingCount;
        private Integer coursesCount;
        private BigDecimal averageGrade;
        private Integer totalPoints;
        private Integer currentLevel;
        private List<String> interests;
        private List<String> skills;
        private Boolean canFollow;
        private Boolean isFollowing;
        private Boolean isBlocked;
        
        // Getters and Setters
        public Long getUserId() { return userId; }
        public void setUserId(Long userId) { this.userId = userId; }
        public String getUsername() { return username; }
        public void setUsername(String username) { this.username = username; }
        public String getRealName() { return realName; }
        public void setRealName(String realName) { this.realName = realName; }
        public String getEmail() { return email; }
        public void setEmail(String email) { this.email = email; }
        public String getPhone() { return phone; }
        public void setPhone(String phone) { this.phone = phone; }
        public String getAvatar() { return avatar; }
        public void setAvatar(String avatar) { this.avatar = avatar; }
        public String getUserType() { return userType; }
        public void setUserType(String userType) { this.userType = userType; }
        public String getGender() { return gender; }
        public void setGender(String gender) { this.gender = gender; }
        public String getDepartment() { return department; }
        public void setDepartment(String department) { this.department = department; }
        public String getBio() { return bio; }
        public void setBio(String bio) { this.bio = bio; }
        public String getLocation() { return location; }
        public void setLocation(String location) { this.location = location; }
        public String getWebsite() { return website; }
        public void setWebsite(String website) { this.website = website; }
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
        public LocalDateTime getCreateTime() { return createTime; }
        public void setCreateTime(LocalDateTime createTime) { this.createTime = createTime; }
        public LocalDateTime getLastLoginTime() { return lastLoginTime; }
        public void setLastLoginTime(LocalDateTime lastLoginTime) { this.lastLoginTime = lastLoginTime; }
        public Boolean getIsOnline() { return isOnline; }
        public void setIsOnline(Boolean isOnline) { this.isOnline = isOnline; }
        public Integer getFollowersCount() { return followersCount; }
        public void setFollowersCount(Integer followersCount) { this.followersCount = followersCount; }
        public Integer getFollowingCount() { return followingCount; }
        public void setFollowingCount(Integer followingCount) { this.followingCount = followingCount; }
        public Integer getCoursesCount() { return coursesCount; }
        public void setCoursesCount(Integer coursesCount) { this.coursesCount = coursesCount; }
        public BigDecimal getAverageGrade() { return averageGrade; }
        public void setAverageGrade(BigDecimal averageGrade) { this.averageGrade = averageGrade; }
        public Integer getTotalPoints() { return totalPoints; }
        public void setTotalPoints(Integer totalPoints) { this.totalPoints = totalPoints; }
        public Integer getCurrentLevel() { return currentLevel; }
        public void setCurrentLevel(Integer currentLevel) { this.currentLevel = currentLevel; }
        public List<String> getInterests() { return interests; }
        public void setInterests(List<String> interests) { this.interests = interests; }
        public List<String> getSkills() { return skills; }
        public void setSkills(List<String> skills) { this.skills = skills; }
        public Boolean getCanFollow() { return canFollow; }
        public void setCanFollow(Boolean canFollow) { this.canFollow = canFollow; }
        public Boolean getIsFollowing() { return isFollowing; }
        public void setIsFollowing(Boolean isFollowing) { this.isFollowing = isFollowing; }
        public Boolean getIsBlocked() { return isBlocked; }
        public void setIsBlocked(Boolean isBlocked) { this.isBlocked = isBlocked; }
    }

    /**
     * 用户资料更新请求DTO
     */
    public static class UserProfileUpdateRequest {
        private String realName;
        private String avatar;
        private String gender;
        private String department;
        private String bio;
        private String location;
        private String website;
        private List<String> interests;
        private List<String> skills;
        
        // Getters and Setters
        public String getRealName() { return realName; }
        public void setRealName(String realName) { this.realName = realName; }
        public String getAvatar() { return avatar; }
        public void setAvatar(String avatar) { this.avatar = avatar; }
        public String getGender() { return gender; }
        public void setGender(String gender) { this.gender = gender; }
        public String getDepartment() { return department; }
        public void setDepartment(String department) { this.department = department; }
        public String getBio() { return bio; }
        public void setBio(String bio) { this.bio = bio; }
        public String getLocation() { return location; }
        public void setLocation(String location) { this.location = location; }
        public String getWebsite() { return website; }
        public void setWebsite(String website) { this.website = website; }
        public List<String> getInterests() { return interests; }
        public void setInterests(List<String> interests) { this.interests = interests; }
        public List<String> getSkills() { return skills; }
        public void setSkills(List<String> skills) { this.skills = skills; }
    }

    /**
     * 用户反馈响应DTO
     */
    public static class UserFeedbackResponse {
        private Long feedbackId;
        private String title;
        private String content;
        private String category;
        private String priority;
        private String status;
        private LocalDateTime submitTime;
        private LocalDateTime responseTime;
        private String response;
        private String respondedBy;
        private List<String> attachments;
        
        // Getters and Setters
        public Long getFeedbackId() { return feedbackId; }
        public void setFeedbackId(Long feedbackId) { this.feedbackId = feedbackId; }
        public String getTitle() { return title; }
        public void setTitle(String title) { this.title = title; }
        public String getContent() { return content; }
        public void setContent(String content) { this.content = content; }
        public String getCategory() { return category; }
        public void setCategory(String category) { this.category = category; }
        public String getPriority() { return priority; }
        public void setPriority(String priority) { this.priority = priority; }
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
        public LocalDateTime getSubmitTime() { return submitTime; }
        public void setSubmitTime(LocalDateTime submitTime) { this.submitTime = submitTime; }
        public LocalDateTime getResponseTime() { return responseTime; }
        public void setResponseTime(LocalDateTime responseTime) { this.responseTime = responseTime; }
        public String getResponse() { return response; }
        public void setResponse(String response) { this.response = response; }
        public String getRespondedBy() { return respondedBy; }
        public void setRespondedBy(String respondedBy) { this.respondedBy = respondedBy; }
        public List<String> getAttachments() { return attachments; }
        public void setAttachments(List<String> attachments) { this.attachments = attachments; }
    }

    /**
     * 反馈提交请求DTO
     */
    public static class FeedbackSubmitRequest {
        @NotBlank(message = "标题不能为空")
        private String title;
        
        @NotBlank(message = "内容不能为空")
        private String content;
        
        @NotBlank(message = "分类不能为空")
        private String category; // BUG, FEATURE, IMPROVEMENT, OTHER
        
        private String priority; // LOW, MEDIUM, HIGH, URGENT
        private List<String> attachments;
        private String userAgent;
        private String browserInfo;
        
        // Getters and Setters
        public String getTitle() { return title; }
        public void setTitle(String title) { this.title = title; }
        public String getContent() { return content; }
        public void setContent(String content) { this.content = content; }
        public String getCategory() { return category; }
        public void setCategory(String category) { this.category = category; }
        public String getPriority() { return priority; }
        public void setPriority(String priority) { this.priority = priority; }
        public List<String> getAttachments() { return attachments; }
        public void setAttachments(List<String> attachments) { this.attachments = attachments; }
        public String getUserAgent() { return userAgent; }
        public void setUserAgent(String userAgent) { this.userAgent = userAgent; }
        public String getBrowserInfo() { return browserInfo; }
        public void setBrowserInfo(String browserInfo) { this.browserInfo = browserInfo; }
    }
}