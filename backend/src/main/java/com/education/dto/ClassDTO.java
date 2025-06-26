package com.education.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;
import java.time.LocalDateTime;
import java.util.List;

/**
 * 班级相关DTO
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
public class ClassDTO {

    /**
     * 班级创建请求DTO
     */
    public static class ClassCreateRequest {
        @NotBlank(message = "班级名称不能为空")
        @Size(max = 100, message = "班级名称长度不能超过100字符")
        private String className;
        
        @Size(max = 500, message = "班级描述长度不能超过500字符")
        private String description;
        
        @NotNull(message = "课程ID不能为空")
        private Long courseId;
        
        private Integer maxStudents; // 最大学生数
        private String classCode; // 班级代码
        private Boolean isActive; // 是否激活
        private String semester; // 学期
        private String academicYear; // 学年
        
        // Getters and Setters
        public String getClassName() { return className; }
        public void setClassName(String className) { this.className = className; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
        public Long getCourseId() { return courseId; }
        public void setCourseId(Long courseId) { this.courseId = courseId; }
        public Integer getMaxStudents() { return maxStudents; }
        public void setMaxStudents(Integer maxStudents) { this.maxStudents = maxStudents; }
        public String getClassCode() { return classCode; }
        public void setClassCode(String classCode) { this.classCode = classCode; }
        public Boolean getIsActive() { return isActive; }
        public void setIsActive(Boolean isActive) { this.isActive = isActive; }
        public String getSemester() { return semester; }
        public void setSemester(String semester) { this.semester = semester; }
        public String getAcademicYear() { return academicYear; }
        public void setAcademicYear(String academicYear) { this.academicYear = academicYear; }
    }

    /**
     * 班级响应DTO
     */
    public static class ClassResponse {
        private Long classId;
        private String className;
        private String description;
        private Long courseId;
        private String courseName;
        private Integer maxStudents;
        private Integer currentStudents;
        private String classCode;
        private Boolean isActive;
        private String semester;
        private String academicYear;
        private LocalDateTime createTime;
        private LocalDateTime updateTime;
        private Long teacherId;
        private String teacherName;
        
        // Getters and Setters
        public Long getClassId() { return classId; }
        public void setClassId(Long classId) { this.classId = classId; }
        public String getClassName() { return className; }
        public void setClassName(String className) { this.className = className; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
        public Long getCourseId() { return courseId; }
        public void setCourseId(Long courseId) { this.courseId = courseId; }
        public String getCourseName() { return courseName; }
        public void setCourseName(String courseName) { this.courseName = courseName; }
        public Integer getMaxStudents() { return maxStudents; }
        public void setMaxStudents(Integer maxStudents) { this.maxStudents = maxStudents; }
        public Integer getCurrentStudents() { return currentStudents; }
        public void setCurrentStudents(Integer currentStudents) { this.currentStudents = currentStudents; }
        public String getClassCode() { return classCode; }
        public void setClassCode(String classCode) { this.classCode = classCode; }
        public Boolean getIsActive() { return isActive; }
        public void setIsActive(Boolean isActive) { this.isActive = isActive; }
        public String getSemester() { return semester; }
        public void setSemester(String semester) { this.semester = semester; }
        public String getAcademicYear() { return academicYear; }
        public void setAcademicYear(String academicYear) { this.academicYear = academicYear; }
        public LocalDateTime getCreateTime() { return createTime; }
        public void setCreateTime(LocalDateTime createTime) { this.createTime = createTime; }
        public LocalDateTime getUpdateTime() { return updateTime; }
        public void setUpdateTime(LocalDateTime updateTime) { this.updateTime = updateTime; }
        public Long getTeacherId() { return teacherId; }
        public void setTeacherId(Long teacherId) { this.teacherId = teacherId; }
        public String getTeacherName() { return teacherName; }
        public void setTeacherName(String teacherName) { this.teacherName = teacherName; }
    }

    /**
     * 班级列表响应DTO
     */
    public static class ClassListResponse {
        private Long classId;
        private String className;
        private String courseName;
        private Integer currentStudents;
        private Integer maxStudents;
        private Boolean isActive;
        private String semester;
        private LocalDateTime createTime;
        
        // Getters and Setters
        public Long getClassId() { return classId; }
        public void setClassId(Long classId) { this.classId = classId; }
        public String getClassName() { return className; }
        public void setClassName(String className) { this.className = className; }
        public String getCourseName() { return courseName; }
        public void setCourseName(String courseName) { this.courseName = courseName; }
        public Integer getCurrentStudents() { return currentStudents; }
        public void setCurrentStudents(Integer currentStudents) { this.currentStudents = currentStudents; }
        public Integer getMaxStudents() { return maxStudents; }
        public void setMaxStudents(Integer maxStudents) { this.maxStudents = maxStudents; }
        public Boolean getIsActive() { return isActive; }
        public void setIsActive(Boolean isActive) { this.isActive = isActive; }
        public String getSemester() { return semester; }
        public void setSemester(String semester) { this.semester = semester; }
        public LocalDateTime getCreateTime() { return createTime; }
        public void setCreateTime(LocalDateTime createTime) { this.createTime = createTime; }
    }

    /**
     * 班级更新请求DTO
     */
    public static class ClassUpdateRequest {
        private String className;
        private String description;
        private Integer maxStudents;
        private Boolean isActive;
        private String semester;
        private String academicYear;
        
        // Getters and Setters
        public String getClassName() { return className; }
        public void setClassName(String className) { this.className = className; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
        public Integer getMaxStudents() { return maxStudents; }
        public void setMaxStudents(Integer maxStudents) { this.maxStudents = maxStudents; }
        public Boolean getIsActive() { return isActive; }
        public void setIsActive(Boolean isActive) { this.isActive = isActive; }
        public String getSemester() { return semester; }
        public void setSemester(String semester) { this.semester = semester; }
        public String getAcademicYear() { return academicYear; }
        public void setAcademicYear(String academicYear) { this.academicYear = academicYear; }
    }

    /**
     * 学生加入班级请求DTO
     */
    public static class StudentJoinRequest {
        @NotNull(message = "学生ID不能为空")
        private Long studentId;
        
        @NotNull(message = "班级ID不能为空")
        private Long classId;
        
        // Getters and Setters
        public Long getStudentId() { return studentId; }
        public void setStudentId(Long studentId) { this.studentId = studentId; }
        public Long getClassId() { return classId; }
        public void setClassId(Long classId) { this.classId = classId; }
    }

    /**
     * 班级学生列表响应DTO
     */
    public static class ClassStudentResponse {
        private Long studentId;
        private String studentName;
        private String studentNumber;
        private String email;
        private LocalDateTime joinTime;
        private Boolean isActive;
        
        // Getters and Setters
        public Long getStudentId() { return studentId; }
        public void setStudentId(Long studentId) { this.studentId = studentId; }
        public String getStudentName() { return studentName; }
        public void setStudentName(String studentName) { this.studentName = studentName; }
        public String getStudentNumber() { return studentNumber; }
        public void setStudentNumber(String studentNumber) { this.studentNumber = studentNumber; }
        public String getEmail() { return email; }
        public void setEmail(String email) { this.email = email; }
        public LocalDateTime getJoinTime() { return joinTime; }
        public void setJoinTime(LocalDateTime joinTime) { this.joinTime = joinTime; }
        public Boolean getIsActive() { return isActive; }
        public void setIsActive(Boolean isActive) { this.isActive = isActive; }
    }

    /**
     * 班级统计响应DTO
     */
    public static class ClassStatisticsResponse {
        private Long classId;
        private String className;
        private Integer totalStudents;
        private Integer activeStudents;
        private Integer totalTasks;
        private Integer completedTasks;
        private Double averageScore;
        private Integer totalResources;
        
        // Getters and Setters
        public Long getClassId() { return classId; }
        public void setClassId(Long classId) { this.classId = classId; }
        public String getClassName() { return className; }
        public void setClassName(String className) { this.className = className; }
        public Integer getTotalStudents() { return totalStudents; }
        public void setTotalStudents(Integer totalStudents) { this.totalStudents = totalStudents; }
        public Integer getActiveStudents() { return activeStudents; }
        public void setActiveStudents(Integer activeStudents) { this.activeStudents = activeStudents; }
        public Integer getTotalTasks() { return totalTasks; }
        public void setTotalTasks(Integer totalTasks) { this.totalTasks = totalTasks; }
        public Integer getCompletedTasks() { return completedTasks; }
        public void setCompletedTasks(Integer completedTasks) { this.completedTasks = completedTasks; }
        public Double getAverageScore() { return averageScore; }
        public void setAverageScore(Double averageScore) { this.averageScore = averageScore; }
        public Integer getTotalResources() { return totalResources; }
        public void setTotalResources(Integer totalResources) { this.totalResources = totalResources; }
        
        // Additional setter methods for compatibility
        public void setMaxStudents(Integer maxStudents) { this.totalStudents = maxStudents; }
        public void setInactiveStudents(int inactiveStudents) { 
            this.activeStudents = this.totalStudents != null ? this.totalStudents - inactiveStudents : 0; 
        }
        public void setAverageGrade(double averageGrade) { this.averageScore = averageGrade; }
        public void setAssignmentCompletionRate(double completionRate) { 
            // Store completion rate as a percentage of completed tasks
        }
    }
    
    /**
     * 学生响应DTO (用于班级学生列表)
     */
    public static class StudentResponse {
        private Long studentId;
        private String studentNumber;
        private String realName;
        private String email;
        private String phone;
        private String gender;
        private String major;
        private String grade;
        private String className;
        private String avatar;
        private String status;
        private LocalDateTime createTime;
        private LocalDateTime lastLoginTime;
        private LocalDateTime joinTime;
        private Boolean isActive;
        
        // Getters and Setters
        public Long getStudentId() { return studentId; }
        public void setStudentId(Long studentId) { this.studentId = studentId; }
        public String getStudentNumber() { return studentNumber; }
        public void setStudentNumber(String studentNumber) { this.studentNumber = studentNumber; }
        public String getRealName() { return realName; }
        public void setRealName(String realName) { this.realName = realName; }
        public String getEmail() { return email; }
        public void setEmail(String email) { this.email = email; }
        public String getPhone() { return phone; }
        public void setPhone(String phone) { this.phone = phone; }
        public String getGender() { return gender; }
        public void setGender(String gender) { this.gender = gender; }
        public String getMajor() { return major; }
        public void setMajor(String major) { this.major = major; }
        public String getGrade() { return grade; }
        public void setGrade(String grade) { this.grade = grade; }
        public String getClassName() { return className; }
        public void setClassName(String className) { this.className = className; }
        public String getAvatar() { return avatar; }
        public void setAvatar(String avatar) { this.avatar = avatar; }
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
        public LocalDateTime getCreateTime() { return createTime; }
        public void setCreateTime(LocalDateTime createTime) { this.createTime = createTime; }
        public LocalDateTime getLastLoginTime() { return lastLoginTime; }
        public void setLastLoginTime(LocalDateTime lastLoginTime) { this.lastLoginTime = lastLoginTime; }
        public LocalDateTime getJoinTime() { return joinTime; }
        public void setJoinTime(LocalDateTime joinTime) { this.joinTime = joinTime; }
        public Boolean getIsActive() { return isActive; }
        public void setIsActive(Boolean isActive) { this.isActive = isActive; }
        /**
         * 设置学生姓名（兼容方法）
         */
        public void setStudentName(String studentName) {
            this.realName = studentName;
        }

        /**
         * 获取学生姓名（兼容方法）
         */
        public String getStudentName() {
            return this.realName;
        }
    }
    
    /**
     * 邀请码响应DTO
     */
    public static class InviteCodeResponse {
        private String inviteCode;
        private Long classId;
        private String className;
        private LocalDateTime expireTime;
        private LocalDateTime createTime;
        private Boolean isActive;
        
        // Getters and Setters
        public String getInviteCode() { return inviteCode; }
        public void setInviteCode(String inviteCode) { this.inviteCode = inviteCode; }
        public Long getClassId() { return classId; }
        public void setClassId(Long classId) { this.classId = classId; }
        public String getClassName() { return className; }
        public void setClassName(String className) { this.className = className; }
        public LocalDateTime getExpireTime() { return expireTime; }
        public void setExpireTime(LocalDateTime expireTime) { this.expireTime = expireTime; }
        public LocalDateTime getCreateTime() { return createTime; }
        public void setCreateTime(LocalDateTime createTime) { this.createTime = createTime; }
        public Boolean getIsActive() { return isActive; }
        public void setIsActive(Boolean isActive) { this.isActive = isActive; }
        

    }
}