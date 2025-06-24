package com.education.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import java.time.LocalDateTime;
import java.util.List;
import java.math.BigDecimal;

/**
 * 用户DTO扩展类 - 第6部分
 * 包含用户偏好、安全设置、双因子认证等相关DTO
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
public class UserDTOExtension6 {

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
        private Boolean showEmail;
        private Boolean showPhone;
        private String defaultView;
        private Integer itemsPerPage;
        
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
        public Boolean getShowEmail() { return showEmail; }
        public void setShowEmail(Boolean showEmail) { this.showEmail = showEmail; }
        public Boolean getShowPhone() { return showPhone; }
        public void setShowPhone(Boolean showPhone) { this.showPhone = showPhone; }
        public String getDefaultView() { return defaultView; }
        public void setDefaultView(String defaultView) { this.defaultView = defaultView; }
        public Integer getItemsPerPage() { return itemsPerPage; }
        public void setItemsPerPage(Integer itemsPerPage) { this.itemsPerPage = itemsPerPage; }
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
        private Boolean showEmail;
        private Boolean showPhone;
        private String defaultView;
        private Integer itemsPerPage;
        
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
        public Boolean getShowEmail() { return showEmail; }
        public void setShowEmail(Boolean showEmail) { this.showEmail = showEmail; }
        public Boolean getShowPhone() { return showPhone; }
        public void setShowPhone(Boolean showPhone) { this.showPhone = showPhone; }
        public String getDefaultView() { return defaultView; }
        public void setDefaultView(String defaultView) { this.defaultView = defaultView; }
        public Integer getItemsPerPage() { return itemsPerPage; }
        public void setItemsPerPage(Integer itemsPerPage) { this.itemsPerPage = itemsPerPage; }
    }

    /**
     * 安全设置响应DTO
     */
    public static class SecuritySettingsResponse {
        private Boolean twoFactorEnabled;
        private String twoFactorMethod;
        private Boolean loginNotifications;
        private Boolean suspiciousActivityAlerts;
        private Integer sessionTimeout;
        private Boolean requirePasswordChange;
        private LocalDateTime lastPasswordChange;
        private List<TrustedDevice> trustedDevices;
        private List<String> allowedIpRanges;
        
        // Getters and Setters
        public Boolean getTwoFactorEnabled() { return twoFactorEnabled; }
        public void setTwoFactorEnabled(Boolean twoFactorEnabled) { this.twoFactorEnabled = twoFactorEnabled; }
        public String getTwoFactorMethod() { return twoFactorMethod; }
        public void setTwoFactorMethod(String twoFactorMethod) { this.twoFactorMethod = twoFactorMethod; }
        public Boolean getLoginNotifications() { return loginNotifications; }
        public void setLoginNotifications(Boolean loginNotifications) { this.loginNotifications = loginNotifications; }
        public Boolean getSuspiciousActivityAlerts() { return suspiciousActivityAlerts; }
        public void setSuspiciousActivityAlerts(Boolean suspiciousActivityAlerts) { this.suspiciousActivityAlerts = suspiciousActivityAlerts; }
        public Integer getSessionTimeout() { return sessionTimeout; }
        public void setSessionTimeout(Integer sessionTimeout) { this.sessionTimeout = sessionTimeout; }
        public Boolean getRequirePasswordChange() { return requirePasswordChange; }
        public void setRequirePasswordChange(Boolean requirePasswordChange) { this.requirePasswordChange = requirePasswordChange; }
        public LocalDateTime getLastPasswordChange() { return lastPasswordChange; }
        public void setLastPasswordChange(LocalDateTime lastPasswordChange) { this.lastPasswordChange = lastPasswordChange; }
        public List<TrustedDevice> getTrustedDevices() { return trustedDevices; }
        public void setTrustedDevices(List<TrustedDevice> trustedDevices) { this.trustedDevices = trustedDevices; }
        public List<String> getAllowedIpRanges() { return allowedIpRanges; }
        public void setAllowedIpRanges(List<String> allowedIpRanges) { this.allowedIpRanges = allowedIpRanges; }
        
        public static class TrustedDevice {
            private String deviceId;
            private String deviceName;
            private String deviceType;
            private LocalDateTime addedTime;
            private LocalDateTime lastUsed;
            
            // Getters and Setters
            public String getDeviceId() { return deviceId; }
            public void setDeviceId(String deviceId) { this.deviceId = deviceId; }
            public String getDeviceName() { return deviceName; }
            public void setDeviceName(String deviceName) { this.deviceName = deviceName; }
            public String getDeviceType() { return deviceType; }
            public void setDeviceType(String deviceType) { this.deviceType = deviceType; }
            public LocalDateTime getAddedTime() { return addedTime; }
            public void setAddedTime(LocalDateTime addedTime) { this.addedTime = addedTime; }
            public LocalDateTime getLastUsed() { return lastUsed; }
            public void setLastUsed(LocalDateTime lastUsed) { this.lastUsed = lastUsed; }
        }
    }

    /**
     * 安全设置更新请求DTO
     */
    public static class SecuritySettingsUpdateRequest {
        private Boolean loginNotifications;
        private Boolean suspiciousActivityAlerts;
        private Integer sessionTimeout;
        private List<String> allowedIpRanges;
        
        // Getters and Setters
        public Boolean getLoginNotifications() { return loginNotifications; }
        public void setLoginNotifications(Boolean loginNotifications) { this.loginNotifications = loginNotifications; }
        public Boolean getSuspiciousActivityAlerts() { return suspiciousActivityAlerts; }
        public void setSuspiciousActivityAlerts(Boolean suspiciousActivityAlerts) { this.suspiciousActivityAlerts = suspiciousActivityAlerts; }
        public Integer getSessionTimeout() { return sessionTimeout; }
        public void setSessionTimeout(Integer sessionTimeout) { this.sessionTimeout = sessionTimeout; }
        public List<String> getAllowedIpRanges() { return allowedIpRanges; }
        public void setAllowedIpRanges(List<String> allowedIpRanges) { this.allowedIpRanges = allowedIpRanges; }
    }

    /**
     * 双因子认证响应DTO
     */
    public static class TwoFactorAuthResponse {
        private Boolean enabled;
        private String method;
        private String qrCode;
        private String secretKey;
        private List<String> backupCodes;
        private LocalDateTime enabledTime;
        
        // Getters and Setters
        public Boolean getEnabled() { return enabled; }
        public void setEnabled(Boolean enabled) { this.enabled = enabled; }
        public String getMethod() { return method; }
        public void setMethod(String method) { this.method = method; }
        public String getQrCode() { return qrCode; }
        public void setQrCode(String qrCode) { this.qrCode = qrCode; }
        public String getSecretKey() { return secretKey; }
        public void setSecretKey(String secretKey) { this.secretKey = secretKey; }
        public List<String> getBackupCodes() { return backupCodes; }
        public void setBackupCodes(List<String> backupCodes) { this.backupCodes = backupCodes; }
        public LocalDateTime getEnabledTime() { return enabledTime; }
        public void setEnabledTime(LocalDateTime enabledTime) { this.enabledTime = enabledTime; }
    }

    /**
     * 双因子认证请求DTO
     */
    public static class TwoFactorAuthRequest {
        @NotBlank(message = "认证方法不能为空")
        private String method; // TOTP, SMS, EMAIL
        
        @NotBlank(message = "密码不能为空")
        private String password;
        
        private String phoneNumber;
        private String email;
        
        // Getters and Setters
        public String getMethod() { return method; }
        public void setMethod(String method) { this.method = method; }
        public String getPassword() { return password; }
        public void setPassword(String password) { this.password = password; }
        public String getPhoneNumber() { return phoneNumber; }
        public void setPhoneNumber(String phoneNumber) { this.phoneNumber = phoneNumber; }
        public String getEmail() { return email; }
        public void setEmail(String email) { this.email = email; }
    }

    /**
     * 双因子认证禁用请求DTO
     */
    public static class TwoFactorAuthDisableRequest {
        @NotBlank(message = "密码不能为空")
        private String password;
        
        @NotBlank(message = "验证码不能为空")
        private String verificationCode;
        
        // Getters and Setters
        public String getPassword() { return password; }
        public void setPassword(String password) { this.password = password; }
        public String getVerificationCode() { return verificationCode; }
        public void setVerificationCode(String verificationCode) { this.verificationCode = verificationCode; }
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
        private LocalDateTime createdTime;
        private LocalDateTime readTime;
        private String actionUrl;
        private String sender;
        
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
        public LocalDateTime getCreatedTime() { return createdTime; }
        public void setCreatedTime(LocalDateTime createdTime) { this.createdTime = createdTime; }
        public LocalDateTime getReadTime() { return readTime; }
        public void setReadTime(LocalDateTime readTime) { this.readTime = readTime; }
        public String getActionUrl() { return actionUrl; }
        public void setActionUrl(String actionUrl) { this.actionUrl = actionUrl; }
        public String getSender() { return sender; }
        public void setSender(String sender) { this.sender = sender; }
    }
}