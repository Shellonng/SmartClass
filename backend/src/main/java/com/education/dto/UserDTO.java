package com.education.dto;

import jakarta.validation.constraints.Email;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.List;

/**
 * 用户相关DTO
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
public class UserDTO {

    /**
     * 用户创建请求DTO
     */
    public static class UserCreateRequest {
        @NotBlank(message = "用户名不能为空")
        private String username;
        
        @NotBlank(message = "密码不能为空")
        private String password;
        
        @Email(message = "邮箱格式不正确")
        private String email;
        
        private String realName;
        private String phone;
        private String avatar;
        private String userType; // STUDENT, TEACHER, ADMIN
        private String gender;
        private String department;
        
        // Getters and Setters
        public String getUsername() { return username; }
        public void setUsername(String username) { this.username = username; }
        public String getPassword() { return password; }
        public void setPassword(String password) { this.password = password; }
        public String getEmail() { return email; }
        public void setEmail(String email) { this.email = email; }
        public String getRealName() { return realName; }
        public void setRealName(String realName) { this.realName = realName; }
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
    }

    /**
     * 用户响应DTO
     */
    public static class UserResponse {
        private Long userId;
        private String username;
        private String email;
        private String realName;
        private String phone;
        private String avatar;
        private String userType;
        private String gender;
        private String department;
        private String status;
        private LocalDateTime createTime;
        private LocalDateTime lastLoginTime;
        
        // Getters and Setters
        public Long getUserId() { return userId; }
        public void setUserId(Long userId) { this.userId = userId; }
        public String getUsername() { return username; }
        public void setUsername(String username) { this.username = username; }
        public String getEmail() { return email; }
        public void setEmail(String email) { this.email = email; }
        public String getRealName() { return realName; }
        public void setRealName(String realName) { this.realName = realName; }
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
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
        public LocalDateTime getCreateTime() { return createTime; }
        public void setCreateTime(LocalDateTime createTime) { this.createTime = createTime; }
        public LocalDateTime getLastLoginTime() { return lastLoginTime; }
        public void setLastLoginTime(LocalDateTime lastLoginTime) { this.lastLoginTime = lastLoginTime; }
    }

    /**
     * 用户更新请求DTO
     */
    public static class UserUpdateRequest {
        private String email;
        private String realName;
        private String phone;
        private String avatar;
        private String gender;
        private String department;
        
        // Getters and Setters
        public String getEmail() { return email; }
        public void setEmail(String email) { this.email = email; }
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
    }

    /**
     * 密码修改请求DTO
     */
    public static class PasswordChangeRequest {
        @NotBlank(message = "原密码不能为空")
        private String oldPassword;
        
        @NotBlank(message = "新密码不能为空")
        private String newPassword;
        
        // Getters and Setters
        public String getOldPassword() { return oldPassword; }
        public void setOldPassword(String oldPassword) { this.oldPassword = oldPassword; }
        public String getNewPassword() { return newPassword; }
        public void setNewPassword(String newPassword) { this.newPassword = newPassword; }
    }

    /**
     * 用户信息响应DTO
     */
    public static class UserInfoResponse {
        private Long userId;
        private String username;
        private String email;
        private String realName;
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
        private Integer totalLogins;
        private Boolean isOnline;
        
        // Getters and Setters
        public Long getUserId() { return userId; }
        public void setUserId(Long userId) { this.userId = userId; }
        public String getUsername() { return username; }
        public void setUsername(String username) { this.username = username; }
        public String getEmail() { return email; }
        public void setEmail(String email) { this.email = email; }
        public String getRealName() { return realName; }
        public void setRealName(String realName) { this.realName = realName; }
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
        public Integer getTotalLogins() { return totalLogins; }
        public void setTotalLogins(Integer totalLogins) { this.totalLogins = totalLogins; }
        public Boolean getIsOnline() { return isOnline; }
        public void setIsOnline(Boolean isOnline) { this.isOnline = isOnline; }
    }

    /**
     * 通知设置更新请求DTO
     */
    public static class NotificationSettingsUpdateRequest {
        private Boolean emailNotifications;
        private Boolean pushNotifications;
        private Boolean smsNotifications;
        
        // Getters and Setters
        public Boolean getEmailNotifications() { return emailNotifications; }
        public void setEmailNotifications(Boolean emailNotifications) { this.emailNotifications = emailNotifications; }
        public Boolean getPushNotifications() { return pushNotifications; }
        public void setPushNotifications(Boolean pushNotifications) { this.pushNotifications = pushNotifications; }
        public Boolean getSmsNotifications() { return smsNotifications; }
        public void setSmsNotifications(Boolean smsNotifications) { this.smsNotifications = smsNotifications; }
    }

    /**
     * 活动日志响应DTO
     */
    public static class ActivityLogResponse {
        private Long logId;
        private String action;
        private String description;
        private String ipAddress;
        private String userAgent;
        private LocalDateTime timestamp;
        private String status;
        
        // Getters and Setters
        public Long getLogId() { return logId; }
        public void setLogId(Long logId) { this.logId = logId; }
        public String getAction() { return action; }
        public void setAction(String action) { this.action = action; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
        public String getIpAddress() { return ipAddress; }
        public void setIpAddress(String ipAddress) { this.ipAddress = ipAddress; }
        public String getUserAgent() { return userAgent; }
        public void setUserAgent(String userAgent) { this.userAgent = userAgent; }
        public LocalDateTime getTimestamp() { return timestamp; }
        public void setTimestamp(LocalDateTime timestamp) { this.timestamp = timestamp; }
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
    }

    /**
     * 用户信息更新请求DTO
     */
    public static class UserInfoUpdateRequest {
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
     * 邮箱绑定请求DTO
     */
    public static class EmailBindRequest {
        @Email(message = "邮箱格式不正确")
        @NotBlank(message = "邮箱不能为空")
        private String email;
        
        @NotBlank(message = "验证码不能为空")
        private String verificationCode;
        
        // Getters and Setters
        public String getEmail() { return email; }
        public void setEmail(String email) { this.email = email; }
        public String getVerificationCode() { return verificationCode; }
        public void setVerificationCode(String verificationCode) { this.verificationCode = verificationCode; }
    }

    /**
     * 手机绑定请求DTO
     */
    public static class PhoneBindRequest {
        @NotBlank(message = "手机号不能为空")
        private String phone;
        
        @NotBlank(message = "验证码不能为空")
        private String verificationCode;
        
        // Getters and Setters
        public String getPhone() { return phone; }
        public void setPhone(String phone) { this.phone = phone; }
        public String getVerificationCode() { return verificationCode; }
        public void setVerificationCode(String verificationCode) { this.verificationCode = verificationCode; }
    }

    /**
     * 用户设置响应DTO
     */
    public static class UserSettingsResponse {
        private String language;
        private String timezone;
        private String theme;
        private Boolean emailNotifications;
        private Boolean pushNotifications;
        private Boolean smsNotifications;
        private String privacyLevel;
        
        // Getters and Setters
        public String getLanguage() { return language; }
        public void setLanguage(String language) { this.language = language; }
        public String getTimezone() { return timezone; }
        public void setTimezone(String timezone) { this.timezone = timezone; }
        public String getTheme() { return theme; }
        public void setTheme(String theme) { this.theme = theme; }
        public Boolean getEmailNotifications() { return emailNotifications; }
        public void setEmailNotifications(Boolean emailNotifications) { this.emailNotifications = emailNotifications; }
        public Boolean getPushNotifications() { return pushNotifications; }
        public void setPushNotifications(Boolean pushNotifications) { this.pushNotifications = pushNotifications; }
        public Boolean getSmsNotifications() { return smsNotifications; }
        public void setSmsNotifications(Boolean smsNotifications) { this.smsNotifications = smsNotifications; }
        public String getPrivacyLevel() { return privacyLevel; }
        public void setPrivacyLevel(String privacyLevel) { this.privacyLevel = privacyLevel; }
    }

    /**
     * 用户设置更新请求DTO
     */
    public static class UserSettingsUpdateRequest {
        private String language;
        private String timezone;
        private String theme;
        private Boolean emailNotifications;
        private Boolean pushNotifications;
        private Boolean smsNotifications;
        private String privacyLevel;
        
        // Getters and Setters
        public String getLanguage() { return language; }
        public void setLanguage(String language) { this.language = language; }
        public String getTimezone() { return timezone; }
        public void setTimezone(String timezone) { this.timezone = timezone; }
        public String getTheme() { return theme; }
        public void setTheme(String theme) { this.theme = theme; }
        public Boolean getEmailNotifications() { return emailNotifications; }
        public void setEmailNotifications(Boolean emailNotifications) { this.emailNotifications = emailNotifications; }
        public Boolean getPushNotifications() { return pushNotifications; }
        public void setPushNotifications(Boolean pushNotifications) { this.pushNotifications = pushNotifications; }
        public Boolean getSmsNotifications() { return smsNotifications; }
        public void setSmsNotifications(Boolean smsNotifications) { this.smsNotifications = smsNotifications; }
        public String getPrivacyLevel() { return privacyLevel; }
        public void setPrivacyLevel(String privacyLevel) { this.privacyLevel = privacyLevel; }
    }

    /**
     * 用户积分响应DTO
     */
    public static class UserPointsResponse {
        private Integer totalPoints;
        private Integer availablePoints;
        private Integer usedPoints;
        private Integer pendingPoints;
        private List<PointSource> pointSources;
        private String rank;
        private Integer rankPosition;
        private LocalDateTime lastUpdated;
        
        // Getters and Setters
        public Integer getTotalPoints() { return totalPoints; }
        public void setTotalPoints(Integer totalPoints) { this.totalPoints = totalPoints; }
        public Integer getAvailablePoints() { return availablePoints; }
        public void setAvailablePoints(Integer availablePoints) { this.availablePoints = availablePoints; }
        public Integer getUsedPoints() { return usedPoints; }
        public void setUsedPoints(Integer usedPoints) { this.usedPoints = usedPoints; }
        public Integer getPendingPoints() { return pendingPoints; }
        public void setPendingPoints(Integer pendingPoints) { this.pendingPoints = pendingPoints; }
        public List<PointSource> getPointSources() { return pointSources; }
        public void setPointSources(List<PointSource> pointSources) { this.pointSources = pointSources; }
        public String getRank() { return rank; }
        public void setRank(String rank) { this.rank = rank; }
        public Integer getRankPosition() { return rankPosition; }
        public void setRankPosition(Integer rankPosition) { this.rankPosition = rankPosition; }
        public LocalDateTime getLastUpdated() { return lastUpdated; }
        public void setLastUpdated(LocalDateTime lastUpdated) { this.lastUpdated = lastUpdated; }
        
        public static class PointSource {
            private String source;
            private Integer points;
            private String description;
            
            // Getters and Setters
            public String getSource() { return source; }
            public void setSource(String source) { this.source = source; }
            public Integer getPoints() { return points; }
            public void setPoints(Integer points) { this.points = points; }
            public String getDescription() { return description; }
            public void setDescription(String description) { this.description = description; }
        }
    }

    /**
     * 积分历史响应DTO
     */
    public static class PointsHistoryResponse {
        private Long historyId;
        private String action;
        private Integer points;
        private String description;
        private String source;
        private LocalDateTime createTime;
        private Integer balanceAfter;
        private String status;
        
        // Getters and Setters
        public Long getHistoryId() { return historyId; }
        public void setHistoryId(Long historyId) { this.historyId = historyId; }
        public String getAction() { return action; }
        public void setAction(String action) { this.action = action; }
        public Integer getPoints() { return points; }
        public void setPoints(Integer points) { this.points = points; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
        public String getSource() { return source; }
        public void setSource(String source) { this.source = source; }
        public LocalDateTime getCreateTime() { return createTime; }
        public void setCreateTime(LocalDateTime createTime) { this.createTime = createTime; }
        public Integer getBalanceAfter() { return balanceAfter; }
        public void setBalanceAfter(Integer balanceAfter) { this.balanceAfter = balanceAfter; }
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
    }

    /**
     * 用户等级响应DTO
     */
    public static class UserLevelResponse {
        private Integer currentLevel;
        private String levelName;
        private String levelDescription;
        private Integer currentExp;
        private Integer expToNextLevel;
        private Integer totalExpForCurrentLevel;
        private Integer totalExpForNextLevel;
        private BigDecimal progressPercentage;
        private List<String> privileges;
        private String levelIcon;
        private LocalDateTime levelAchievedTime;
        
        // Getters and Setters
        public Integer getCurrentLevel() { return currentLevel; }
        public void setCurrentLevel(Integer currentLevel) { this.currentLevel = currentLevel; }
        public String getLevelName() { return levelName; }
        public void setLevelName(String levelName) { this.levelName = levelName; }
        public String getLevelDescription() { return levelDescription; }
        public void setLevelDescription(String levelDescription) { this.levelDescription = levelDescription; }
        public Integer getCurrentExp() { return currentExp; }
        public void setCurrentExp(Integer currentExp) { this.currentExp = currentExp; }
        public Integer getExpToNextLevel() { return expToNextLevel; }
        public void setExpToNextLevel(Integer expToNextLevel) { this.expToNextLevel = expToNextLevel; }
        public Integer getTotalExpForCurrentLevel() { return totalExpForCurrentLevel; }
        public void setTotalExpForCurrentLevel(Integer totalExpForCurrentLevel) { this.totalExpForCurrentLevel = totalExpForCurrentLevel; }
        public Integer getTotalExpForNextLevel() { return totalExpForNextLevel; }
        public void setTotalExpForNextLevel(Integer totalExpForNextLevel) { this.totalExpForNextLevel = totalExpForNextLevel; }
        public BigDecimal getProgressPercentage() { return progressPercentage; }
        public void setProgressPercentage(BigDecimal progressPercentage) { this.progressPercentage = progressPercentage; }
        public List<String> getPrivileges() { return privileges; }
        public void setPrivileges(List<String> privileges) { this.privileges = privileges; }
        public String getLevelIcon() { return levelIcon; }
        public void setLevelIcon(String levelIcon) { this.levelIcon = levelIcon; }
        public LocalDateTime getLevelAchievedTime() { return levelAchievedTime; }
        public void setLevelAchievedTime(LocalDateTime levelAchievedTime) { this.levelAchievedTime = levelAchievedTime; }
    }

    /**
     * 徽章响应DTO
     */
    public static class BadgeResponse {
        private Long badgeId;
        private String badgeName;
        private String badgeDescription;
        private String badgeIcon;
        private String badgeType;
        private String rarity;
        private LocalDateTime earnedTime;
        private Boolean isVisible;
        private String category;
        private Integer pointsAwarded;
        private String requirements;
        
        // Getters and Setters
        public Long getBadgeId() { return badgeId; }
        public void setBadgeId(Long badgeId) { this.badgeId = badgeId; }
        public String getBadgeName() { return badgeName; }
        public void setBadgeName(String badgeName) { this.badgeName = badgeName; }
        public String getBadgeDescription() { return badgeDescription; }
        public void setBadgeDescription(String badgeDescription) { this.badgeDescription = badgeDescription; }
        public String getBadgeIcon() { return badgeIcon; }
        public void setBadgeIcon(String badgeIcon) { this.badgeIcon = badgeIcon; }
        public String getBadgeType() { return badgeType; }
        public void setBadgeType(String badgeType) { this.badgeType = badgeType; }
        public String getRarity() { return rarity; }
        public void setRarity(String rarity) { this.rarity = rarity; }
        public LocalDateTime getEarnedTime() { return earnedTime; }
        public void setEarnedTime(LocalDateTime earnedTime) { this.earnedTime = earnedTime; }
        public Boolean getIsVisible() { return isVisible; }
        public void setIsVisible(Boolean isVisible) { this.isVisible = isVisible; }
        public String getCategory() { return category; }
        public void setCategory(String category) { this.category = category; }
        public Integer getPointsAwarded() { return pointsAwarded; }
        public void setPointsAwarded(Integer pointsAwarded) { this.pointsAwarded = pointsAwarded; }
        public String getRequirements() { return requirements; }
        public void setRequirements(String requirements) { this.requirements = requirements; }
    }

    /**
     * 通知设置响应DTO
     */
    public static class NotificationSettingsResponse {
        private Boolean emailNotifications;
        private Boolean pushNotifications;
        private Boolean smsNotifications;
        private Boolean courseReminders;
        private Boolean assignmentDeadlines;
        private Boolean gradeUpdates;
        private Boolean systemAnnouncements;
        private Boolean socialInteractions;
        private String notificationFrequency;
        private List<String> mutedCategories;
        
        // Getters and Setters
        public Boolean getEmailNotifications() { return emailNotifications; }
        public void setEmailNotifications(Boolean emailNotifications) { this.emailNotifications = emailNotifications; }
        public Boolean getPushNotifications() { return pushNotifications; }
        public void setPushNotifications(Boolean pushNotifications) { this.pushNotifications = pushNotifications; }
        public Boolean getSmsNotifications() { return smsNotifications; }
        public void setSmsNotifications(Boolean smsNotifications) { this.smsNotifications = smsNotifications; }
        public Boolean getCourseReminders() { return courseReminders; }
        public void setCourseReminders(Boolean courseReminders) { this.courseReminders = courseReminders; }
        public Boolean getAssignmentDeadlines() { return assignmentDeadlines; }
        public void setAssignmentDeadlines(Boolean assignmentDeadlines) { this.assignmentDeadlines = assignmentDeadlines; }
        public Boolean getGradeUpdates() { return gradeUpdates; }
        public void setGradeUpdates(Boolean gradeUpdates) { this.gradeUpdates = gradeUpdates; }
        public Boolean getSystemAnnouncements() { return systemAnnouncements; }
        public void setSystemAnnouncements(Boolean systemAnnouncements) { this.systemAnnouncements = systemAnnouncements; }
        public Boolean getSocialInteractions() { return socialInteractions; }
        public void setSocialInteractions(Boolean socialInteractions) { this.socialInteractions = socialInteractions; }
        public String getNotificationFrequency() { return notificationFrequency; }
        public void setNotificationFrequency(String notificationFrequency) { this.notificationFrequency = notificationFrequency; }
        public List<String> getMutedCategories() { return mutedCategories; }
        public void setMutedCategories(List<String> mutedCategories) { this.mutedCategories = mutedCategories; }
    }

    /**
     * 登录历史响应DTO
     */
    public static class LoginHistoryResponse {
        private Long loginId;
        private LocalDateTime loginTime;
        private String ipAddress;
        private String userAgent;
        private String deviceType;
        private String location;
        private String status;
        private LocalDateTime logoutTime;
        private String sessionDuration;
        private Boolean isCurrentSession;
        
        // Getters and Setters
        public Long getLoginId() { return loginId; }
        public void setLoginId(Long loginId) { this.loginId = loginId; }
        public LocalDateTime getLoginTime() { return loginTime; }
        public void setLoginTime(LocalDateTime loginTime) { this.loginTime = loginTime; }
        public String getIpAddress() { return ipAddress; }
        public void setIpAddress(String ipAddress) { this.ipAddress = ipAddress; }
        public String getUserAgent() { return userAgent; }
        public void setUserAgent(String userAgent) { this.userAgent = userAgent; }
        public String getDeviceType() { return deviceType; }
        public void setDeviceType(String deviceType) { this.deviceType = deviceType; }
        public String getLocation() { return location; }
        public void setLocation(String location) { this.location = location; }
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
        public LocalDateTime getLogoutTime() { return logoutTime; }
        public void setLogoutTime(LocalDateTime logoutTime) { this.logoutTime = logoutTime; }
        public String getSessionDuration() { return sessionDuration; }
        public void setSessionDuration(String sessionDuration) { this.sessionDuration = sessionDuration; }
        public Boolean getIsCurrentSession() { return isCurrentSession; }
        public void setIsCurrentSession(Boolean isCurrentSession) { this.isCurrentSession = isCurrentSession; }
    }

    /**
     * 账户注销请求DTO
     */
    public static class AccountDeactivateRequest {
        @NotBlank(message = "注销原因不能为空")
        private String reason;
        
        @NotBlank(message = "密码不能为空")
        private String password;
        
        private String feedback;
        private Boolean deleteData;
        
        // Getters and Setters
        public String getReason() { return reason; }
        public void setReason(String reason) { this.reason = reason; }
        public String getPassword() { return password; }
        public void setPassword(String password) { this.password = password; }
        public String getFeedback() { return feedback; }
        public void setFeedback(String feedback) { this.feedback = feedback; }
        public Boolean getDeleteData() { return deleteData; }
        public void setDeleteData(Boolean deleteData) { this.deleteData = deleteData; }
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

    public class DataExportResponse {
    }

    public class DataExportRequest {
    }

    public class IdentityVerificationRequest {
    }

    public class UserStatisticsResponse {
    }

    public class UserSearchResponse {
    }

    public class UserSearchRequest {
    }

    public class UserDetailResponse {
    }

    public class UserRoleResponse {
    }

    public class UserPermissionResponse {
    }

    public class UserPreferencesResponse {
    }

    public class UserPreferencesUpdateRequest {
    }

    public class SecuritySettingsResponse {
    }

    public class SecuritySettingsUpdateRequest {
    }

    public class TwoFactorAuthResponse {
    }

    public class TwoFactorAuthRequest {
    }

    public class TwoFactorAuthDisableRequest {
    }

    public class NotificationResponse {
    }

    public class UserFollowResponse {
    }

    public class LearningReportResponse {
    }

    public class AchievementResponse {
    }

    public class StudyTimeStatisticsResponse {
    }

    public class OnlineStatusResponse {
    }
}