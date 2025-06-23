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
 * 用户相关DTO扩展类
 * 包含更多用户相关的DTO定义
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
public class UserDTOExtension {

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
        private String status;
        private LocalDateTime createTime;
        private LocalDateTime lastLoginTime;
        private String bio;
        private String location;
        private String website;
        
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
        public String getBio() { return bio; }
        public void setBio(String bio) { this.bio = bio; }
        public String getLocation() { return location; }
        public void setLocation(String location) { this.location = location; }
        public String getWebsite() { return website; }
        public void setWebsite(String website) { this.website = website; }
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
        private Boolean smsNotifications;
        private Boolean pushNotifications;
        private String privacy;
        
        // Getters and Setters
        public String getLanguage() { return language; }
        public void setLanguage(String language) { this.language = language; }
        public String getTimezone() { return timezone; }
        public void setTimezone(String timezone) { this.timezone = timezone; }
        public String getTheme() { return theme; }
        public void setTheme(String theme) { this.theme = theme; }
        public Boolean getEmailNotifications() { return emailNotifications; }
        public void setEmailNotifications(Boolean emailNotifications) { this.emailNotifications = emailNotifications; }
        public Boolean getSmsNotifications() { return smsNotifications; }
        public void setSmsNotifications(Boolean smsNotifications) { this.smsNotifications = smsNotifications; }
        public Boolean getPushNotifications() { return pushNotifications; }
        public void setPushNotifications(Boolean pushNotifications) { this.pushNotifications = pushNotifications; }
        public String getPrivacy() { return privacy; }
        public void setPrivacy(String privacy) { this.privacy = privacy; }
    }

    /**
     * 用户设置更新请求DTO
     */
    public static class UserSettingsUpdateRequest {
        private String language;
        private String timezone;
        private String theme;
        private Boolean emailNotifications;
        private Boolean smsNotifications;
        private Boolean pushNotifications;
        private String privacy;
        
        // Getters and Setters
        public String getLanguage() { return language; }
        public void setLanguage(String language) { this.language = language; }
        public String getTimezone() { return timezone; }
        public void setTimezone(String timezone) { this.timezone = timezone; }
        public String getTheme() { return theme; }
        public void setTheme(String theme) { this.theme = theme; }
        public Boolean getEmailNotifications() { return emailNotifications; }
        public void setEmailNotifications(Boolean emailNotifications) { this.emailNotifications = emailNotifications; }
        public Boolean getSmsNotifications() { return smsNotifications; }
        public void setSmsNotifications(Boolean smsNotifications) { this.smsNotifications = smsNotifications; }
        public Boolean getPushNotifications() { return pushNotifications; }
        public void setPushNotifications(Boolean pushNotifications) { this.pushNotifications = pushNotifications; }
        public String getPrivacy() { return privacy; }
        public void setPrivacy(String privacy) { this.privacy = privacy; }
    }

    /**
     * 通知设置响应DTO
     */
    public static class NotificationSettingsResponse {
        private Boolean emailNotifications;
        private Boolean smsNotifications;
        private Boolean pushNotifications;
        private Boolean taskReminders;
        private Boolean gradeNotifications;
        private Boolean messageNotifications;
        private Boolean systemNotifications;
        
        // Getters and Setters
        public Boolean getEmailNotifications() { return emailNotifications; }
        public void setEmailNotifications(Boolean emailNotifications) { this.emailNotifications = emailNotifications; }
        public Boolean getSmsNotifications() { return smsNotifications; }
        public void setSmsNotifications(Boolean smsNotifications) { this.smsNotifications = smsNotifications; }
        public Boolean getPushNotifications() { return pushNotifications; }
        public void setPushNotifications(Boolean pushNotifications) { this.pushNotifications = pushNotifications; }
        public Boolean getTaskReminders() { return taskReminders; }
        public void setTaskReminders(Boolean taskReminders) { this.taskReminders = taskReminders; }
        public Boolean getGradeNotifications() { return gradeNotifications; }
        public void setGradeNotifications(Boolean gradeNotifications) { this.gradeNotifications = gradeNotifications; }
        public Boolean getMessageNotifications() { return messageNotifications; }
        public void setMessageNotifications(Boolean messageNotifications) { this.messageNotifications = messageNotifications; }
        public Boolean getSystemNotifications() { return systemNotifications; }
        public void setSystemNotifications(Boolean systemNotifications) { this.systemNotifications = systemNotifications; }
    }

    /**
     * 通知设置更新请求DTO
     */
    public static class NotificationSettingsUpdateRequest {
        private Boolean emailNotifications;
        private Boolean smsNotifications;
        private Boolean pushNotifications;
        private Boolean taskReminders;
        private Boolean gradeNotifications;
        private Boolean messageNotifications;
        private Boolean systemNotifications;
        
        // Getters and Setters
        public Boolean getEmailNotifications() { return emailNotifications; }
        public void setEmailNotifications(Boolean emailNotifications) { this.emailNotifications = emailNotifications; }
        public Boolean getSmsNotifications() { return smsNotifications; }
        public void setSmsNotifications(Boolean smsNotifications) { this.smsNotifications = smsNotifications; }
        public Boolean getPushNotifications() { return pushNotifications; }
        public void setPushNotifications(Boolean pushNotifications) { this.pushNotifications = pushNotifications; }
        public Boolean getTaskReminders() { return taskReminders; }
        public void setTaskReminders(Boolean taskReminders) { this.taskReminders = taskReminders; }
        public Boolean getGradeNotifications() { return gradeNotifications; }
        public void setGradeNotifications(Boolean gradeNotifications) { this.gradeNotifications = gradeNotifications; }
        public Boolean getMessageNotifications() { return messageNotifications; }
        public void setMessageNotifications(Boolean messageNotifications) { this.messageNotifications = messageNotifications; }
        public Boolean getSystemNotifications() { return systemNotifications; }
        public void setSystemNotifications(Boolean systemNotifications) { this.systemNotifications = systemNotifications; }
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
        private String result;
        
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
        public String getResult() { return result; }
        public void setResult(String result) { this.result = result; }
    }

    /**
     * 登录历史响应DTO
     */
    public static class LoginHistoryResponse {
        private Long loginId;
        private String ipAddress;
        private String userAgent;
        private String location;
        private String device;
        private LocalDateTime loginTime;
        private LocalDateTime logoutTime;
        private String status;
        
        // Getters and Setters
        public Long getLoginId() { return loginId; }
        public void setLoginId(Long loginId) { this.loginId = loginId; }
        public String getIpAddress() { return ipAddress; }
        public void setIpAddress(String ipAddress) { this.ipAddress = ipAddress; }
        public String getUserAgent() { return userAgent; }
        public void setUserAgent(String userAgent) { this.userAgent = userAgent; }
        public String getLocation() { return location; }
        public void setLocation(String location) { this.location = location; }
        public String getDevice() { return device; }
        public void setDevice(String device) { this.device = device; }
        public LocalDateTime getLoginTime() { return loginTime; }
        public void setLoginTime(LocalDateTime loginTime) { this.loginTime = loginTime; }
        public LocalDateTime getLogoutTime() { return logoutTime; }
        public void setLogoutTime(LocalDateTime logoutTime) { this.logoutTime = logoutTime; }
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
    }
}