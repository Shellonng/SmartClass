package com.education.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import java.time.LocalDateTime;
import java.util.List;
import java.math.BigDecimal;

/**
 * 用户DTO扩展类 - 第5部分
 * 包含用户搜索、详情、角色权限等相关DTO
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
public class UserDTOExtension5 {

    /**
     * 用户搜索请求DTO
     */
    public static class UserSearchRequest {
        private String keyword;
        private String userType;
        private String department;
        private String status;
        private String gender;
        private LocalDateTime createTimeStart;
        private LocalDateTime createTimeEnd;
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
        public String getGender() { return gender; }
        public void setGender(String gender) { this.gender = gender; }
        public LocalDateTime getCreateTimeStart() { return createTimeStart; }
        public void setCreateTimeStart(LocalDateTime createTimeStart) { this.createTimeStart = createTimeStart; }
        public LocalDateTime getCreateTimeEnd() { return createTimeEnd; }
        public void setCreateTimeEnd(LocalDateTime createTimeEnd) { this.createTimeEnd = createTimeEnd; }
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
        private String email;
        private String userType;
        private String department;
        private String avatar;
        private String status;
        private LocalDateTime createTime;
        private LocalDateTime lastLoginTime;
        private Boolean isOnline;
        
        // Getters and Setters
        public Long getUserId() { return userId; }
        public void setUserId(Long userId) { this.userId = userId; }
        public String getUsername() { return username; }
        public void setUsername(String username) { this.username = username; }
        public String getRealName() { return realName; }
        public void setRealName(String realName) { this.realName = realName; }
        public String getEmail() { return email; }
        public void setEmail(String email) { this.email = email; }
        public String getUserType() { return userType; }
        public void setUserType(String userType) { this.userType = userType; }
        public String getDepartment() { return department; }
        public void setDepartment(String department) { this.department = department; }
        public String getAvatar() { return avatar; }
        public void setAvatar(String avatar) { this.avatar = avatar; }
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
        public LocalDateTime getCreateTime() { return createTime; }
        public void setCreateTime(LocalDateTime createTime) { this.createTime = createTime; }
        public LocalDateTime getLastLoginTime() { return lastLoginTime; }
        public void setLastLoginTime(LocalDateTime lastLoginTime) { this.lastLoginTime = lastLoginTime; }
        public Boolean getIsOnline() { return isOnline; }
        public void setIsOnline(Boolean isOnline) { this.isOnline = isOnline; }
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
        private Integer totalLogins;
        private Boolean isOnline;
        private List<String> roles;
        private List<String> permissions;
        private UserStatistics statistics;
        
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
        public Integer getTotalLogins() { return totalLogins; }
        public void setTotalLogins(Integer totalLogins) { this.totalLogins = totalLogins; }
        public Boolean getIsOnline() { return isOnline; }
        public void setIsOnline(Boolean isOnline) { this.isOnline = isOnline; }
        public List<String> getRoles() { return roles; }
        public void setRoles(List<String> roles) { this.roles = roles; }
        public List<String> getPermissions() { return permissions; }
        public void setPermissions(List<String> permissions) { this.permissions = permissions; }
        public UserStatistics getStatistics() { return statistics; }
        public void setStatistics(UserStatistics statistics) { this.statistics = statistics; }
        
        public static class UserStatistics {
            private Integer totalCourses;
            private Integer completedCourses;
            private BigDecimal averageGrade;
            private Integer totalPoints;
            
            // Getters and Setters
            public Integer getTotalCourses() { return totalCourses; }
            public void setTotalCourses(Integer totalCourses) { this.totalCourses = totalCourses; }
            public Integer getCompletedCourses() { return completedCourses; }
            public void setCompletedCourses(Integer completedCourses) { this.completedCourses = completedCourses; }
            public BigDecimal getAverageGrade() { return averageGrade; }
            public void setAverageGrade(BigDecimal averageGrade) { this.averageGrade = averageGrade; }
            public Integer getTotalPoints() { return totalPoints; }
            public void setTotalPoints(Integer totalPoints) { this.totalPoints = totalPoints; }
        }
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
            private LocalDateTime assignedTime;
            private String assignedBy;
            
            // Getters and Setters
            public Long getRoleId() { return roleId; }
            public void setRoleId(Long roleId) { this.roleId = roleId; }
            public String getRoleName() { return roleName; }
            public void setRoleName(String roleName) { this.roleName = roleName; }
            public String getRoleCode() { return roleCode; }
            public void setRoleCode(String roleCode) { this.roleCode = roleCode; }
            public String getDescription() { return description; }
            public void setDescription(String description) { this.description = description; }
            public LocalDateTime getAssignedTime() { return assignedTime; }
            public void setAssignedTime(LocalDateTime assignedTime) { this.assignedTime = assignedTime; }
            public String getAssignedBy() { return assignedBy; }
            public void setAssignedBy(String assignedBy) { this.assignedBy = assignedBy; }
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
            private String source; // ROLE, DIRECT
            
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
            public String getSource() { return source; }
            public void setSource(String source) { this.source = source; }
        }
    }
}