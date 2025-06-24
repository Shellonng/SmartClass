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
 * 用户相关DTO扩展类3
 * 包含更多用户相关的DTO定义
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
public class UserDTOExtension3 {

    /**
     * 用户偏好响应DTO
     */
    public static class UserPreferencesResponse {
        private String language;
        private String timezone;
        private String theme;
        private String dateFormat;
        private String timeFormat;
        private Boolean showOnlineStatus;
        private Boolean allowDirectMessages;
        private String emailFrequency;
        private Map<String, Object> customSettings;
        
        // Getters and Setters
        public String getLanguage() { return language; }
        public void setLanguage(String language) { this.language = language; }
        public String getTimezone() { return timezone; }
        public void setTimezone(String timezone) { this.timezone = timezone; }
        public String getTheme() { return theme; }
        public void setTheme(String theme) { this.theme = theme; }
        public String getDateFormat() { return dateFormat; }
        public void setDateFormat(String dateFormat) { this.dateFormat = dateFormat; }
        public String getTimeFormat() { return timeFormat; }
        public void setTimeFormat(String timeFormat) { this.timeFormat = timeFormat; }
        public Boolean getShowOnlineStatus() { return showOnlineStatus; }
        public void setShowOnlineStatus(Boolean showOnlineStatus) { this.showOnlineStatus = showOnlineStatus; }
        public Boolean getAllowDirectMessages() { return allowDirectMessages; }
        public void setAllowDirectMessages(Boolean allowDirectMessages) { this.allowDirectMessages = allowDirectMessages; }
        public String getEmailFrequency() { return emailFrequency; }
        public void setEmailFrequency(String emailFrequency) { this.emailFrequency = emailFrequency; }
        public Map<String, Object> getCustomSettings() { return customSettings; }
        public void setCustomSettings(Map<String, Object> customSettings) { this.customSettings = customSettings; }
    }

    /**
     * 用户偏好更新请求DTO
     */
    public static class UserPreferencesUpdateRequest {
        private String language;
        private String timezone;
        private String theme;
        private String dateFormat;
        private String timeFormat;
        private Boolean showOnlineStatus;
        private Boolean allowDirectMessages;
        private String emailFrequency;
        private Map<String, Object> customSettings;
        
        // Getters and Setters
        public String getLanguage() { return language; }
        public void setLanguage(String language) { this.language = language; }
        public String getTimezone() { return timezone; }
        public void setTimezone(String timezone) { this.timezone = timezone; }
        public String getTheme() { return theme; }
        public void setTheme(String theme) { this.theme = theme; }
        public String getDateFormat() { return dateFormat; }
        public void setDateFormat(String dateFormat) { this.dateFormat = dateFormat; }
        public String getTimeFormat() { return timeFormat; }
        public void setTimeFormat(String timeFormat) { this.timeFormat = timeFormat; }
        public Boolean getShowOnlineStatus() { return showOnlineStatus; }
        public void setShowOnlineStatus(Boolean showOnlineStatus) { this.showOnlineStatus = showOnlineStatus; }
        public Boolean getAllowDirectMessages() { return allowDirectMessages; }
        public void setAllowDirectMessages(Boolean allowDirectMessages) { this.allowDirectMessages = allowDirectMessages; }
        public String getEmailFrequency() { return emailFrequency; }
        public void setEmailFrequency(String emailFrequency) { this.emailFrequency = emailFrequency; }
        public Map<String, Object> getCustomSettings() { return customSettings; }
        public void setCustomSettings(Map<String, Object> customSettings) { this.customSettings = customSettings; }
    }

    /**
     * 安全设置响应DTO
     */
    public static class SecuritySettingsResponse {
        private Boolean twoFactorEnabled;
        private Boolean loginNotifications;
        private Boolean sessionTimeout;
        private Integer sessionTimeoutMinutes;
        private Boolean deviceTracking;
        private List<String> trustedDevices;
        private Boolean passwordExpiry;
        private Integer passwordExpiryDays;
        
        // Getters and Setters
        public Boolean getTwoFactorEnabled() { return twoFactorEnabled; }
        public void setTwoFactorEnabled(Boolean twoFactorEnabled) { this.twoFactorEnabled = twoFactorEnabled; }
        public Boolean getLoginNotifications() { return loginNotifications; }
        public void setLoginNotifications(Boolean loginNotifications) { this.loginNotifications = loginNotifications; }
        public Boolean getSessionTimeout() { return sessionTimeout; }
        public void setSessionTimeout(Boolean sessionTimeout) { this.sessionTimeout = sessionTimeout; }
        public Integer getSessionTimeoutMinutes() { return sessionTimeoutMinutes; }
        public void setSessionTimeoutMinutes(Integer sessionTimeoutMinutes) { this.sessionTimeoutMinutes = sessionTimeoutMinutes; }
        public Boolean getDeviceTracking() { return deviceTracking; }
        public void setDeviceTracking(Boolean deviceTracking) { this.deviceTracking = deviceTracking; }
        public List<String> getTrustedDevices() { return trustedDevices; }
        public void setTrustedDevices(List<String> trustedDevices) { this.trustedDevices = trustedDevices; }
        public Boolean getPasswordExpiry() { return passwordExpiry; }
        public void setPasswordExpiry(Boolean passwordExpiry) { this.passwordExpiry = passwordExpiry; }
        public Integer getPasswordExpiryDays() { return passwordExpiryDays; }
        public void setPasswordExpiryDays(Integer passwordExpiryDays) { this.passwordExpiryDays = passwordExpiryDays; }
    }

    /**
     * 安全设置更新请求DTO
     */
    public static class SecuritySettingsUpdateRequest {
        private Boolean loginNotifications;
        private Boolean sessionTimeout;
        private Integer sessionTimeoutMinutes;
        private Boolean deviceTracking;
        private Boolean passwordExpiry;
        private Integer passwordExpiryDays;
        
        // Getters and Setters
        public Boolean getLoginNotifications() { return loginNotifications; }
        public void setLoginNotifications(Boolean loginNotifications) { this.loginNotifications = loginNotifications; }
        public Boolean getSessionTimeout() { return sessionTimeout; }
        public void setSessionTimeout(Boolean sessionTimeout) { this.sessionTimeout = sessionTimeout; }
        public Integer getSessionTimeoutMinutes() { return sessionTimeoutMinutes; }
        public void setSessionTimeoutMinutes(Integer sessionTimeoutMinutes) { this.sessionTimeoutMinutes = sessionTimeoutMinutes; }
        public Boolean getDeviceTracking() { return deviceTracking; }
        public void setDeviceTracking(Boolean deviceTracking) { this.deviceTracking = deviceTracking; }
        public Boolean getPasswordExpiry() { return passwordExpiry; }
        public void setPasswordExpiry(Boolean passwordExpiry) { this.passwordExpiry = passwordExpiry; }
        public Integer getPasswordExpiryDays() { return passwordExpiryDays; }
        public void setPasswordExpiryDays(Integer passwordExpiryDays) { this.passwordExpiryDays = passwordExpiryDays; }
    }

    /**
     * 双因子认证响应DTO
     */
    public static class TwoFactorAuthResponse {
        private String qrCodeUrl;
        private String secretKey;
        private List<String> backupCodes;
        private Boolean isEnabled;
        
        // Getters and Setters
        public String getQrCodeUrl() { return qrCodeUrl; }
        public void setQrCodeUrl(String qrCodeUrl) { this.qrCodeUrl = qrCodeUrl; }
        public String getSecretKey() { return secretKey; }
        public void setSecretKey(String secretKey) { this.secretKey = secretKey; }
        public List<String> getBackupCodes() { return backupCodes; }
        public void setBackupCodes(List<String> backupCodes) { this.backupCodes = backupCodes; }
        public Boolean getIsEnabled() { return isEnabled; }
        public void setIsEnabled(Boolean isEnabled) { this.isEnabled = isEnabled; }
    }

    /**
     * 双因子认证请求DTO
     */
    public static class TwoFactorAuthRequest {
        @NotBlank(message = "验证码不能为空")
        private String verificationCode;
        
        @NotBlank(message = "密码不能为空")
        private String password;
        
        // Getters and Setters
        public String getVerificationCode() { return verificationCode; }
        public void setVerificationCode(String verificationCode) { this.verificationCode = verificationCode; }
        public String getPassword() { return password; }
        public void setPassword(String password) { this.password = password; }
    }

    /**
     * 双因子认证禁用请求DTO
     */
    public static class TwoFactorAuthDisableRequest {
        @NotBlank(message = "密码不能为空")
        private String password;
        
        private String verificationCode;
        private String backupCode;
        
        // Getters and Setters
        public String getPassword() { return password; }
        public void setPassword(String password) { this.password = password; }
        public String getVerificationCode() { return verificationCode; }
        public void setVerificationCode(String verificationCode) { this.verificationCode = verificationCode; }
        public String getBackupCode() { return backupCode; }
        public void setBackupCode(String backupCode) { this.backupCode = backupCode; }
    }

    /**
     * 通知响应DTO
     */
    public static class NotificationResponse {
        private Long notificationId;
        private String title;
        private String content;
        private String type;
        private String priority;
        private Boolean isRead;
        private LocalDateTime createTime;
        private Map<String, Object> data;
        
        // Getters and Setters
        public Long getNotificationId() { return notificationId; }
        public void setNotificationId(Long notificationId) { this.notificationId = notificationId; }
        public String getTitle() { return title; }
        public void setTitle(String title) { this.title = title; }
        public String getContent() { return content; }
        public void setContent(String content) { this.content = content; }
        public String getType() { return type; }
        public void setType(String type) { this.type = type; }
        public String getPriority() { return priority; }
        public void setPriority(String priority) { this.priority = priority; }
        public Boolean getIsRead() { return isRead; }
        public void setIsRead(Boolean isRead) { this.isRead = isRead; }
        public LocalDateTime getCreateTime() { return createTime; }
        public void setCreateTime(LocalDateTime createTime) { this.createTime = createTime; }
        public Map<String, Object> getData() { return data; }
        public void setData(Map<String, Object> data) { this.data = data; }
    }

    /**
     * 用户关注响应DTO
     */
    public static class UserFollowResponse {
        private Long userId;
        private String username;
        private String realName;
        private String avatar;
        private String userType;
        private String department;
        private LocalDateTime followTime;
        private Boolean isFollowingBack;
        
        // Getters and Setters
        public Long getUserId() { return userId; }
        public void setUserId(Long userId) { this.userId = userId; }
        public String getUsername() { return username; }
        public void setUsername(String username) { this.username = username; }
        public String getRealName() { return realName; }
        public void setRealName(String realName) { this.realName = realName; }
        public String getAvatar() { return avatar; }
        public void setAvatar(String avatar) { this.avatar = avatar; }
        public String getUserType() { return userType; }
        public void setUserType(String userType) { this.userType = userType; }
        public String getDepartment() { return department; }
        public void setDepartment(String department) { this.department = department; }
        public LocalDateTime getFollowTime() { return followTime; }
        public void setFollowTime(LocalDateTime followTime) { this.followTime = followTime; }
        public Boolean getIsFollowingBack() { return isFollowingBack; }
        public void setIsFollowingBack(Boolean isFollowingBack) { this.isFollowingBack = isFollowingBack; }
    }

    /**
     * 学习报告响应DTO
     */
    public static class LearningReportResponse {
        private String reportType;
        private String timeRange;
        private Integer totalStudyTime;
        private Integer completedTasks;
        private Integer totalTasks;
        private Double averageGrade;
        private Integer coursesEnrolled;
        private Integer coursesCompleted;
        private List<CourseProgress> courseProgress;
        private List<SubjectPerformance> subjectPerformance;
        
        // Getters and Setters
        public String getReportType() { return reportType; }
        public void setReportType(String reportType) { this.reportType = reportType; }
        public String getTimeRange() { return timeRange; }
        public void setTimeRange(String timeRange) { this.timeRange = timeRange; }
        public Integer getTotalStudyTime() { return totalStudyTime; }
        public void setTotalStudyTime(Integer totalStudyTime) { this.totalStudyTime = totalStudyTime; }
        public Integer getCompletedTasks() { return completedTasks; }
        public void setCompletedTasks(Integer completedTasks) { this.completedTasks = completedTasks; }
        public Integer getTotalTasks() { return totalTasks; }
        public void setTotalTasks(Integer totalTasks) { this.totalTasks = totalTasks; }
        public Double getAverageGrade() { return averageGrade; }
        public void setAverageGrade(Double averageGrade) { this.averageGrade = averageGrade; }
        public Integer getCoursesEnrolled() { return coursesEnrolled; }
        public void setCoursesEnrolled(Integer coursesEnrolled) { this.coursesEnrolled = coursesEnrolled; }
        public Integer getCoursesCompleted() { return coursesCompleted; }
        public void setCoursesCompleted(Integer coursesCompleted) { this.coursesCompleted = coursesCompleted; }
        public List<CourseProgress> getCourseProgress() { return courseProgress; }
        public void setCourseProgress(List<CourseProgress> courseProgress) { this.courseProgress = courseProgress; }
        public List<SubjectPerformance> getSubjectPerformance() { return subjectPerformance; }
        public void setSubjectPerformance(List<SubjectPerformance> subjectPerformance) { this.subjectPerformance = subjectPerformance; }
        
        public static class CourseProgress {
            private Long courseId;
            private String courseName;
            private Double progressPercentage;
            private Integer completedTasks;
            private Integer totalTasks;
            private Double averageGrade;
            
            // Getters and Setters
            public Long getCourseId() { return courseId; }
            public void setCourseId(Long courseId) { this.courseId = courseId; }
            public String getCourseName() { return courseName; }
            public void setCourseName(String courseName) { this.courseName = courseName; }
            public Double getProgressPercentage() { return progressPercentage; }
            public void setProgressPercentage(Double progressPercentage) { this.progressPercentage = progressPercentage; }
            public Integer getCompletedTasks() { return completedTasks; }
            public void setCompletedTasks(Integer completedTasks) { this.completedTasks = completedTasks; }
            public Integer getTotalTasks() { return totalTasks; }
            public void setTotalTasks(Integer totalTasks) { this.totalTasks = totalTasks; }
            public Double getAverageGrade() { return averageGrade; }
            public void setAverageGrade(Double averageGrade) { this.averageGrade = averageGrade; }
        }
        
        public static class SubjectPerformance {
            private String subject;
            private Double averageGrade;
            private Integer completedTasks;
            private Integer totalTasks;
            private String trend;
            
            // Getters and Setters
            public String getSubject() { return subject; }
            public void setSubject(String subject) { this.subject = subject; }
            public Double getAverageGrade() { return averageGrade; }
            public void setAverageGrade(Double averageGrade) { this.averageGrade = averageGrade; }
            public Integer getCompletedTasks() { return completedTasks; }
            public void setCompletedTasks(Integer completedTasks) { this.completedTasks = completedTasks; }
            public Integer getTotalTasks() { return totalTasks; }
            public void setTotalTasks(Integer totalTasks) { this.totalTasks = totalTasks; }
            public String getTrend() { return trend; }
            public void setTrend(String trend) { this.trend = trend; }
        }
    }
}