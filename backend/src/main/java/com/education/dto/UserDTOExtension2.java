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
 * 用户相关DTO扩展类2
 * 包含更多用户相关的DTO定义
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
public class UserDTOExtension2 {

    /**
     * 账户停用请求DTO
     */
    public static class AccountDeactivateRequest {
        @NotBlank(message = "停用原因不能为空")
        private String reason;
        
        @NotBlank(message = "密码不能为空")
        private String password;
        
        private Boolean deleteData;
        
        // Getters and Setters
        public String getReason() { return reason; }
        public void setReason(String reason) { this.reason = reason; }
        public String getPassword() { return password; }
        public void setPassword(String password) { this.password = password; }
        public Boolean getDeleteData() { return deleteData; }
        public void setDeleteData(Boolean deleteData) { this.deleteData = deleteData; }
    }

    /**
     * 数据导出响应DTO
     */
    public static class DataExportResponse {
        private String exportId;
        private String fileName;
        private String downloadUrl;
        private String fileFormat;
        private Long fileSize;
        private LocalDateTime expiryTime;
        private String status;
        
        // Getters and Setters
        public String getExportId() { return exportId; }
        public void setExportId(String exportId) { this.exportId = exportId; }
        public String getFileName() { return fileName; }
        public void setFileName(String fileName) { this.fileName = fileName; }
        public String getDownloadUrl() { return downloadUrl; }
        public void setDownloadUrl(String downloadUrl) { this.downloadUrl = downloadUrl; }
        public String getFileFormat() { return fileFormat; }
        public void setFileFormat(String fileFormat) { this.fileFormat = fileFormat; }
        public Long getFileSize() { return fileSize; }
        public void setFileSize(Long fileSize) { this.fileSize = fileSize; }
        public LocalDateTime getExpiryTime() { return expiryTime; }
        public void setExpiryTime(LocalDateTime expiryTime) { this.expiryTime = expiryTime; }
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
    }

    /**
     * 数据导出请求DTO
     */
    public static class DataExportRequest {
        private String exportType; // PROFILE, ACTIVITIES, GRADES, ALL
        private String fileFormat; // JSON, CSV, PDF
        private LocalDateTime startDate;
        private LocalDateTime endDate;
        
        // Getters and Setters
        public String getExportType() { return exportType; }
        public void setExportType(String exportType) { this.exportType = exportType; }
        public String getFileFormat() { return fileFormat; }
        public void setFileFormat(String fileFormat) { this.fileFormat = fileFormat; }
        public LocalDateTime getStartDate() { return startDate; }
        public void setStartDate(LocalDateTime startDate) { this.startDate = startDate; }
        public LocalDateTime getEndDate() { return endDate; }
        public void setEndDate(LocalDateTime endDate) { this.endDate = endDate; }
    }

    /**
     * 身份验证请求DTO
     */
    public static class IdentityVerificationRequest {
        @NotBlank(message = "验证类型不能为空")
        private String verificationType; // PASSWORD, EMAIL, SMS, TOTP
        
        private String password;
        private String verificationCode;
        
        // Getters and Setters
        public String getVerificationType() { return verificationType; }
        public void setVerificationType(String verificationType) { this.verificationType = verificationType; }
        public String getPassword() { return password; }
        public void setPassword(String password) { this.password = password; }
        public String getVerificationCode() { return verificationCode; }
        public void setVerificationCode(String verificationCode) { this.verificationCode = verificationCode; }
    }

    /**
     * 用户统计响应DTO
     */
    public static class UserStatisticsResponse {
        private Integer totalCourses;
        private Integer completedTasks;
        private Integer pendingTasks;
        private Double averageGrade;
        private Integer studyDays;
        private Integer totalStudyTime;
        private Integer achievementCount;
        private Integer points;
        
        // Getters and Setters
        public Integer getTotalCourses() { return totalCourses; }
        public void setTotalCourses(Integer totalCourses) { this.totalCourses = totalCourses; }
        public Integer getCompletedTasks() { return completedTasks; }
        public void setCompletedTasks(Integer completedTasks) { this.completedTasks = completedTasks; }
        public Integer getPendingTasks() { return pendingTasks; }
        public void setPendingTasks(Integer pendingTasks) { this.pendingTasks = pendingTasks; }
        public Double getAverageGrade() { return averageGrade; }
        public void setAverageGrade(Double averageGrade) { this.averageGrade = averageGrade; }
        public Integer getStudyDays() { return studyDays; }
        public void setStudyDays(Integer studyDays) { this.studyDays = studyDays; }
        public Integer getTotalStudyTime() { return totalStudyTime; }
        public void setTotalStudyTime(Integer totalStudyTime) { this.totalStudyTime = totalStudyTime; }
        public Integer getAchievementCount() { return achievementCount; }
        public void setAchievementCount(Integer achievementCount) { this.achievementCount = achievementCount; }
        public Integer getPoints() { return points; }
        public void setPoints(Integer points) { this.points = points; }
    }

    /**
     * 用户搜索请求DTO
     */
    public static class UserSearchRequest {
        private String keyword;
        private String userType;
        private String department;
        private String status;
        private String sortBy;
        private String sortOrder;
        
        // Getters and Setters
        public String getKeyword() { return keyword; }
        public void setKeyword(String keyword) { this.keyword = keyword; }
        public String getUserType() { return userType; }
        public void setUserType(String userType) { this.userType = userType; }
        public String getDepartment() { return department; }
        public void setDepartment(String department) { this.department = department; }
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
        public String getSortBy() { return sortBy; }
        public void setSortBy(String sortBy) { this.sortBy = sortBy; }
        public String getSortOrder() { return sortOrder; }
        public void setSortOrder(String sortOrder) { this.sortOrder = sortOrder; }
    }

    /**
     * 用户搜索响应DTO
     */
    public static class UserSearchResponse {
        private Long userId;
        private String username;
        private String realName;
        private String avatar;
        private String userType;
        private String department;
        private String status;
        private LocalDateTime lastLoginTime;
        
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
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
        public LocalDateTime getLastLoginTime() { return lastLoginTime; }
        public void setLastLoginTime(LocalDateTime lastLoginTime) { this.lastLoginTime = lastLoginTime; }
    }

    /**
     * 用户详情响应DTO
     */
    public static class UserDetailResponse {
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
        private Boolean isFollowing;
        
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
        public Boolean getIsFollowing() { return isFollowing; }
        public void setIsFollowing(Boolean isFollowing) { this.isFollowing = isFollowing; }
    }

    /**
     * 用户角色响应DTO
     */
    public static class UserRoleResponse {
        private List<RoleInfo> roles;
        
        // Getters and Setters
        public List<RoleInfo> getRoles() { return roles; }
        public void setRoles(List<RoleInfo> roles) { this.roles = roles; }
        
        public static class RoleInfo {
            private Long roleId;
            private String roleName;
            private String roleCode;
            private String description;
            private LocalDateTime assignTime;
            
            // Getters and Setters
            public Long getRoleId() { return roleId; }
            public void setRoleId(Long roleId) { this.roleId = roleId; }
            public String getRoleName() { return roleName; }
            public void setRoleName(String roleName) { this.roleName = roleName; }
            public String getRoleCode() { return roleCode; }
            public void setRoleCode(String roleCode) { this.roleCode = roleCode; }
            public String getDescription() { return description; }
            public void setDescription(String description) { this.description = description; }
            public LocalDateTime getAssignTime() { return assignTime; }
            public void setAssignTime(LocalDateTime assignTime) { this.assignTime = assignTime; }
        }
    }

    /**
     * 用户权限响应DTO
     */
    public static class UserPermissionResponse {
        private List<PermissionInfo> permissions;
        
        // Getters and Setters
        public List<PermissionInfo> getPermissions() { return permissions; }
        public void setPermissions(List<PermissionInfo> permissions) { this.permissions = permissions; }
        
        public static class PermissionInfo {
            private Long permissionId;
            private String permissionName;
            private String permissionCode;
            private String resource;
            private String action;
            private String description;
            
            // Getters and Setters
            public Long getPermissionId() { return permissionId; }
            public void setPermissionId(Long permissionId) { this.permissionId = permissionId; }
            public String getPermissionName() { return permissionName; }
            public void setPermissionName(String permissionName) { this.permissionName = permissionName; }
            public String getPermissionCode() { return permissionCode; }
            public void setPermissionCode(String permissionCode) { this.permissionCode = permissionCode; }
            public String getResource() { return resource; }
            public void setResource(String resource) { this.resource = resource; }
            public String getAction() { return action; }
            public void setAction(String action) { this.action = action; }
            public String getDescription() { return description; }
            public void setDescription(String description) { this.description = description; }
        }
    }
}