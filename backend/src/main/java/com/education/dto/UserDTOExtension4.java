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
 * 用户相关DTO扩展类4
 * 包含更多用户相关的DTO定义
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
public class UserDTOExtension4 {

    /**
     * 成就响应DTO
     */
    public static class AchievementResponse {
        private Long achievementId;
        private String title;
        private String description;
        private String icon;
        private String category;
        private Integer points;
        private String difficulty;
        private LocalDateTime earnedTime;
        private Double progress;
        private Boolean isCompleted;
        
        // Getters and Setters
        public Long getAchievementId() { return achievementId; }
        public void setAchievementId(Long achievementId) { this.achievementId = achievementId; }
        public String getTitle() { return title; }
        public void setTitle(String title) { this.title = title; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
        public String getIcon() { return icon; }
        public void setIcon(String icon) { this.icon = icon; }
        public String getCategory() { return category; }
        public void setCategory(String category) { this.category = category; }
        public Integer getPoints() { return points; }
        public void setPoints(Integer points) { this.points = points; }
        public String getDifficulty() { return difficulty; }
        public void setDifficulty(String difficulty) { this.difficulty = difficulty; }
        public LocalDateTime getEarnedTime() { return earnedTime; }
        public void setEarnedTime(LocalDateTime earnedTime) { this.earnedTime = earnedTime; }
        public Double getProgress() { return progress; }
        public void setProgress(Double progress) { this.progress = progress; }
        public Boolean getIsCompleted() { return isCompleted; }
        public void setIsCompleted(Boolean isCompleted) { this.isCompleted = isCompleted; }
    }

    /**
     * 用户积分响应DTO
     */
    public static class UserPointsResponse {
        private Integer totalPoints;
        private Integer availablePoints;
        private Integer usedPoints;
        private Integer currentLevelPoints;
        private Integer nextLevelPoints;
        private String currentLevel;
        private String nextLevel;
        private Double progressToNextLevel;
        
        // Getters and Setters
        public Integer getTotalPoints() { return totalPoints; }
        public void setTotalPoints(Integer totalPoints) { this.totalPoints = totalPoints; }
        public Integer getAvailablePoints() { return availablePoints; }
        public void setAvailablePoints(Integer availablePoints) { this.availablePoints = availablePoints; }
        public Integer getUsedPoints() { return usedPoints; }
        public void setUsedPoints(Integer usedPoints) { this.usedPoints = usedPoints; }
        public Integer getCurrentLevelPoints() { return currentLevelPoints; }
        public void setCurrentLevelPoints(Integer currentLevelPoints) { this.currentLevelPoints = currentLevelPoints; }
        public Integer getNextLevelPoints() { return nextLevelPoints; }
        public void setNextLevelPoints(Integer nextLevelPoints) { this.nextLevelPoints = nextLevelPoints; }
        public String getCurrentLevel() { return currentLevel; }
        public void setCurrentLevel(String currentLevel) { this.currentLevel = currentLevel; }
        public String getNextLevel() { return nextLevel; }
        public void setNextLevel(String nextLevel) { this.nextLevel = nextLevel; }
        public Double getProgressToNextLevel() { return progressToNextLevel; }
        public void setProgressToNextLevel(Double progressToNextLevel) { this.progressToNextLevel = progressToNextLevel; }
    }

    /**
     * 积分历史响应DTO
     */
    public static class PointsHistoryResponse {
        private Long recordId;
        private String action;
        private String description;
        private Integer points;
        private String type; // EARNED, USED, EXPIRED
        private String source;
        private LocalDateTime timestamp;
        private Integer balanceAfter;
        
        // Getters and Setters
        public Long getRecordId() { return recordId; }
        public void setRecordId(Long recordId) { this.recordId = recordId; }
        public String getAction() { return action; }
        public void setAction(String action) { this.action = action; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
        public Integer getPoints() { return points; }
        public void setPoints(Integer points) { this.points = points; }
        public String getType() { return type; }
        public void setType(String type) { this.type = type; }
        public String getSource() { return source; }
        public void setSource(String source) { this.source = source; }
        public LocalDateTime getTimestamp() { return timestamp; }
        public void setTimestamp(LocalDateTime timestamp) { this.timestamp = timestamp; }
        public Integer getBalanceAfter() { return balanceAfter; }
        public void setBalanceAfter(Integer balanceAfter) { this.balanceAfter = balanceAfter; }
    }

    /**
     * 用户等级响应DTO
     */
    public static class UserLevelResponse {
        private String currentLevel;
        private String nextLevel;
        private Integer currentLevelPoints;
        private Integer nextLevelPoints;
        private Integer userPoints;
        private Double progressPercentage;
        private List<String> currentLevelBenefits;
        private List<String> nextLevelBenefits;
        
        // Getters and Setters
        public String getCurrentLevel() { return currentLevel; }
        public void setCurrentLevel(String currentLevel) { this.currentLevel = currentLevel; }
        public String getNextLevel() { return nextLevel; }
        public void setNextLevel(String nextLevel) { this.nextLevel = nextLevel; }
        public Integer getCurrentLevelPoints() { return currentLevelPoints; }
        public void setCurrentLevelPoints(Integer currentLevelPoints) { this.currentLevelPoints = currentLevelPoints; }
        public Integer getNextLevelPoints() { return nextLevelPoints; }
        public void setNextLevelPoints(Integer nextLevelPoints) { this.nextLevelPoints = nextLevelPoints; }
        public Integer getUserPoints() { return userPoints; }
        public void setUserPoints(Integer userPoints) { this.userPoints = userPoints; }
        public Double getProgressPercentage() { return progressPercentage; }
        public void setProgressPercentage(Double progressPercentage) { this.progressPercentage = progressPercentage; }
        public List<String> getCurrentLevelBenefits() { return currentLevelBenefits; }
        public void setCurrentLevelBenefits(List<String> currentLevelBenefits) { this.currentLevelBenefits = currentLevelBenefits; }
        public List<String> getNextLevelBenefits() { return nextLevelBenefits; }
        public void setNextLevelBenefits(List<String> nextLevelBenefits) { this.nextLevelBenefits = nextLevelBenefits; }
    }

    /**
     * 徽章响应DTO
     */
    public static class BadgeResponse {
        private Long badgeId;
        private String name;
        private String description;
        private String icon;
        private String color;
        private String category;
        private String rarity;
        private LocalDateTime earnedTime;
        private String earnedFor;
        
        // Getters and Setters
        public Long getBadgeId() { return badgeId; }
        public void setBadgeId(Long badgeId) { this.badgeId = badgeId; }
        public String getName() { return name; }
        public void setName(String name) { this.name = name; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
        public String getIcon() { return icon; }
        public void setIcon(String icon) { this.icon = icon; }
        public String getColor() { return color; }
        public void setColor(String color) { this.color = color; }
        public String getCategory() { return category; }
        public void setCategory(String category) { this.category = category; }
        public String getRarity() { return rarity; }
        public void setRarity(String rarity) { this.rarity = rarity; }
        public LocalDateTime getEarnedTime() { return earnedTime; }
        public void setEarnedTime(LocalDateTime earnedTime) { this.earnedTime = earnedTime; }
        public String getEarnedFor() { return earnedFor; }
        public void setEarnedFor(String earnedFor) { this.earnedFor = earnedFor; }
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
        private Double studyConsistency;
        private List<DailyStudyTime> dailyBreakdown;
        private List<SubjectStudyTime> subjectBreakdown;
        
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
        public Double getStudyConsistency() { return studyConsistency; }
        public void setStudyConsistency(Double studyConsistency) { this.studyConsistency = studyConsistency; }
        public List<DailyStudyTime> getDailyBreakdown() { return dailyBreakdown; }
        public void setDailyBreakdown(List<DailyStudyTime> dailyBreakdown) { this.dailyBreakdown = dailyBreakdown; }
        public List<SubjectStudyTime> getSubjectBreakdown() { return subjectBreakdown; }
        public void setSubjectBreakdown(List<SubjectStudyTime> subjectBreakdown) { this.subjectBreakdown = subjectBreakdown; }
        
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
        
        public static class SubjectStudyTime {
            private String subject;
            private Integer studyTime;
            private Double percentage;
            
            // Getters and Setters
            public String getSubject() { return subject; }
            public void setSubject(String subject) { this.subject = subject; }
            public Integer getStudyTime() { return studyTime; }
            public void setStudyTime(Integer studyTime) { this.studyTime = studyTime; }
            public Double getPercentage() { return percentage; }
            public void setPercentage(Double percentage) { this.percentage = percentage; }
        }
    }

    /**
     * 在线状态响应DTO
     */
    public static class OnlineStatusResponse {
        private String status; // ONLINE, OFFLINE, AWAY, BUSY
        private LocalDateTime lastSeen;
        private String customMessage;
        private Boolean showOnlineStatus;
        
        // Getters and Setters
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
        public LocalDateTime getLastSeen() { return lastSeen; }
        public void setLastSeen(LocalDateTime lastSeen) { this.lastSeen = lastSeen; }
        public String getCustomMessage() { return customMessage; }
        public void setCustomMessage(String customMessage) { this.customMessage = customMessage; }
        public Boolean getShowOnlineStatus() { return showOnlineStatus; }
        public void setShowOnlineStatus(Boolean showOnlineStatus) { this.showOnlineStatus = showOnlineStatus; }
    }

    /**
     * 在线状态更新请求DTO
     */
    public static class OnlineStatusUpdateRequest {
        @NotBlank(message = "状态不能为空")
        private String status;
        
        private String customMessage;
        
        // Getters and Setters
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
        public String getCustomMessage() { return customMessage; }
        public void setCustomMessage(String customMessage) { this.customMessage = customMessage; }
    }

    /**
     * 用户档案响应DTO
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
        private Integer followersCount;
        private Integer followingCount;
        private Integer coursesCount;
        private Integer achievementsCount;
        private Integer points;
        private String level;
        private Boolean isFollowing;
        private Boolean canMessage;
        
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
        public Integer getFollowersCount() { return followersCount; }
        public void setFollowersCount(Integer followersCount) { this.followersCount = followersCount; }
        public Integer getFollowingCount() { return followingCount; }
        public void setFollowingCount(Integer followingCount) { this.followingCount = followingCount; }
        public Integer getCoursesCount() { return coursesCount; }
        public void setCoursesCount(Integer coursesCount) { this.coursesCount = coursesCount; }
        public Integer getAchievementsCount() { return achievementsCount; }
        public void setAchievementsCount(Integer achievementsCount) { this.achievementsCount = achievementsCount; }
        public Integer getPoints() { return points; }
        public void setPoints(Integer points) { this.points = points; }
        public String getLevel() { return level; }
        public void setLevel(String level) { this.level = level; }
        public Boolean getIsFollowing() { return isFollowing; }
        public void setIsFollowing(Boolean isFollowing) { this.isFollowing = isFollowing; }
        public Boolean getCanMessage() { return canMessage; }
        public void setCanMessage(Boolean canMessage) { this.canMessage = canMessage; }
    }

    /**
     * 用户档案更新请求DTO
     */
    public static class UserProfileUpdateRequest {
        private String realName;
        private String phone;
        private String avatar;
        private String gender;
        private String department;
        private String bio;
        private String location;
        private String website;
        
        // Getters and Setters
        public String getRealName() { return realName; }
        public void setRealName(String realName) { this.realName = realName; }
        public String getPhone() { return phone; }
        public void setPhone(String phone) { this.phone = phone; }
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
        private LocalDateTime updateTime;
        private String adminReply;
        
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
        public LocalDateTime getUpdateTime() { return updateTime; }
        public void setUpdateTime(LocalDateTime updateTime) { this.updateTime = updateTime; }
        public String getAdminReply() { return adminReply; }
        public void setAdminReply(String adminReply) { this.adminReply = adminReply; }
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
        private String category;
        
        private String priority;
        private List<String> attachments;
        
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
    }
}