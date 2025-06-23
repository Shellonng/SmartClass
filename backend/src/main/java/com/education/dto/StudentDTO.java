package com.education.dto;

import jakarta.validation.constraints.Email;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import java.time.LocalDateTime;
import java.util.List;

/**
 * 学生相关DTO
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
public class StudentDTO {

    /**
     * 学生创建请求DTO
     */
    public static class StudentCreateRequest {
        @NotBlank(message = "学号不能为空")
        private String studentNumber;
        
        @NotBlank(message = "姓名不能为空")
        private String realName;
        
        @Email(message = "邮箱格式不正确")
        private String email;
        
        private String phone;
        private String gender;
        private String major;
        private String grade;
        private String className;
        private String avatar;
        
        // Getters and Setters
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
    }

    /**
     * 学生响应DTO
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
    }

    /**
     * 学生导入请求DTO
     */
    public static class StudentImportRequest {
        private String importType; // EXCEL, CSV
        private String fileUrl;
        private List<StudentCreateRequest> students;
        
        // Getters and Setters
        public String getImportType() { return importType; }
        public void setImportType(String importType) { this.importType = importType; }
        public String getFileUrl() { return fileUrl; }
        public void setFileUrl(String fileUrl) { this.fileUrl = fileUrl; }
        public List<StudentCreateRequest> getStudents() { return students; }
        public void setStudents(List<StudentCreateRequest> students) { this.students = students; }
    }

    /**
     * 学生导入响应DTO
     */
    public static class StudentImportResponse {
        private Boolean success;
        private String message;
        private Integer totalCount;
        private Integer successCount;
        private Integer failCount;
        private List<String> errors;
        
        // Getters and Setters
        public Boolean getSuccess() { return success; }
        public void setSuccess(Boolean success) { this.success = success; }
        public String getMessage() { return message; }
        public void setMessage(String message) { this.message = message; }
        public Integer getTotalCount() { return totalCount; }
        public void setTotalCount(Integer totalCount) { this.totalCount = totalCount; }
        public Integer getSuccessCount() { return successCount; }
        public void setSuccessCount(Integer successCount) { this.successCount = successCount; }
        public Integer getFailCount() { return failCount; }
        public void setFailCount(Integer failCount) { this.failCount = failCount; }
        public List<String> getErrors() { return errors; }
        public void setErrors(List<String> errors) { this.errors = errors; }
    }

    /**
     * 学生更新请求DTO
     */
    public static class StudentUpdateRequest {
        private String realName;
        private String email;
        private String phone;
        private String gender;
        private String major;
        private String grade;
        private String className;
        private String avatar;
        
        // Getters and Setters
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
    }

    /**
     * 学生详细信息响应DTO
     */
    public static class StudentDetailResponse {
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
        private Integer totalCourses;
        private Integer completedCourses;
        private Double averageGrade;
        private Integer totalAssignments;
        private Integer submittedAssignments;
        private List<String> enrolledCourses;
        
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
        public Integer getTotalCourses() { return totalCourses; }
        public void setTotalCourses(Integer totalCourses) { this.totalCourses = totalCourses; }
        public Integer getCompletedCourses() { return completedCourses; }
        public void setCompletedCourses(Integer completedCourses) { this.completedCourses = completedCourses; }
        public Double getAverageGrade() { return averageGrade; }
        public void setAverageGrade(Double averageGrade) { this.averageGrade = averageGrade; }
        public Integer getTotalAssignments() { return totalAssignments; }
        public void setTotalAssignments(Integer totalAssignments) { this.totalAssignments = totalAssignments; }
        public Integer getSubmittedAssignments() { return submittedAssignments; }
        public void setSubmittedAssignments(Integer submittedAssignments) { this.submittedAssignments = submittedAssignments; }
        public List<String> getEnrolledCourses() { return enrolledCourses; }
        public void setEnrolledCourses(List<String> enrolledCourses) { this.enrolledCourses = enrolledCourses; }
    }

    /**
     * 学生进度响应DTO
     */
    public static class StudentProgressResponse {
        private Long studentId;
        private String studentName;
        private Long courseId;
        private String courseName;
        private Double progressPercentage;
        private Integer completedLessons;
        private Integer totalLessons;
        private Integer completedAssignments;
        private Integer totalAssignments;
        private Double averageScore;
        private LocalDateTime lastActivity;
        private String currentStatus;
        
        // Getters and Setters
        public Long getStudentId() { return studentId; }
        public void setStudentId(Long studentId) { this.studentId = studentId; }
        public String getStudentName() { return studentName; }
        public void setStudentName(String studentName) { this.studentName = studentName; }
        public Long getCourseId() { return courseId; }
        public void setCourseId(Long courseId) { this.courseId = courseId; }
        public String getCourseName() { return courseName; }
        public void setCourseName(String courseName) { this.courseName = courseName; }
        public Double getProgressPercentage() { return progressPercentage; }
        public void setProgressPercentage(Double progressPercentage) { this.progressPercentage = progressPercentage; }
        public Integer getCompletedLessons() { return completedLessons; }
        public void setCompletedLessons(Integer completedLessons) { this.completedLessons = completedLessons; }
        public Integer getTotalLessons() { return totalLessons; }
        public void setTotalLessons(Integer totalLessons) { this.totalLessons = totalLessons; }
        public Integer getCompletedAssignments() { return completedAssignments; }
        public void setCompletedAssignments(Integer completedAssignments) { this.completedAssignments = completedAssignments; }
        public Integer getTotalAssignments() { return totalAssignments; }
        public void setTotalAssignments(Integer totalAssignments) { this.totalAssignments = totalAssignments; }
        public Double getAverageScore() { return averageScore; }
        public void setAverageScore(Double averageScore) { this.averageScore = averageScore; }
        public LocalDateTime getLastActivity() { return lastActivity; }
        public void setLastActivity(LocalDateTime lastActivity) { this.lastActivity = lastActivity; }
        public String getCurrentStatus() { return currentStatus; }
        public void setCurrentStatus(String currentStatus) { this.currentStatus = currentStatus; }
    }

    /**
     * 学生成绩统计响应DTO
     */
    public static class StudentGradeStatisticsResponse {
        private Long studentId;
        private String studentName;
        private Double overallAverage;
        private Double highestGrade;
        private Double lowestGrade;
        private Integer totalAssignments;
        private Integer gradedAssignments;
        private Integer pendingAssignments;
        private List<CourseGradeInfo> courseGrades;
        private List<String> strengths;
        private List<String> improvements;
        
        // Getters and Setters
        public Long getStudentId() { return studentId; }
        public void setStudentId(Long studentId) { this.studentId = studentId; }
        public String getStudentName() { return studentName; }
        public void setStudentName(String studentName) { this.studentName = studentName; }
        public Double getOverallAverage() { return overallAverage; }
        public void setOverallAverage(Double overallAverage) { this.overallAverage = overallAverage; }
        public Double getHighestGrade() { return highestGrade; }
        public void setHighestGrade(Double highestGrade) { this.highestGrade = highestGrade; }
        public Double getLowestGrade() { return lowestGrade; }
        public void setLowestGrade(Double lowestGrade) { this.lowestGrade = lowestGrade; }
        public Integer getTotalAssignments() { return totalAssignments; }
        public void setTotalAssignments(Integer totalAssignments) { this.totalAssignments = totalAssignments; }
        public Integer getGradedAssignments() { return gradedAssignments; }
        public void setGradedAssignments(Integer gradedAssignments) { this.gradedAssignments = gradedAssignments; }
        public Integer getPendingAssignments() { return pendingAssignments; }
        public void setPendingAssignments(Integer pendingAssignments) { this.pendingAssignments = pendingAssignments; }
        public List<CourseGradeInfo> getCourseGrades() { return courseGrades; }
        public void setCourseGrades(List<CourseGradeInfo> courseGrades) { this.courseGrades = courseGrades; }
        public List<String> getStrengths() { return strengths; }
        public void setStrengths(List<String> strengths) { this.strengths = strengths; }
        public List<String> getImprovements() { return improvements; }
        public void setImprovements(List<String> improvements) { this.improvements = improvements; }
        
        public static class CourseGradeInfo {
            private Long courseId;
            private String courseName;
            private Double average;
            private Integer assignmentCount;
            
            // Getters and Setters
            public Long getCourseId() { return courseId; }
            public void setCourseId(Long courseId) { this.courseId = courseId; }
            public String getCourseName() { return courseName; }
            public void setCourseName(String courseName) { this.courseName = courseName; }
            public Double getAverage() { return average; }
            public void setAverage(Double average) { this.average = average; }
            public Integer getAssignmentCount() { return assignmentCount; }
            public void setAssignmentCount(Integer assignmentCount) { this.assignmentCount = assignmentCount; }
        }
    }

    /**
     * 学生提交响应DTO
     */
    public static class StudentSubmissionResponse {
        private Long submissionId;
        private Long studentId;
        private String studentName;
        private Long assignmentId;
        private String assignmentTitle;
        private String submissionContent;
        private String attachmentUrl;
        private LocalDateTime submitTime;
        private String status;
        private Double grade;
        private String feedback;
        private LocalDateTime gradeTime;
        private String graderName;
        
        // Getters and Setters
        public Long getSubmissionId() { return submissionId; }
        public void setSubmissionId(Long submissionId) { this.submissionId = submissionId; }
        public Long getStudentId() { return studentId; }
        public void setStudentId(Long studentId) { this.studentId = studentId; }
        public String getStudentName() { return studentName; }
        public void setStudentName(String studentName) { this.studentName = studentName; }
        public Long getAssignmentId() { return assignmentId; }
        public void setAssignmentId(Long assignmentId) { this.assignmentId = assignmentId; }
        public String getAssignmentTitle() { return assignmentTitle; }
        public void setAssignmentTitle(String assignmentTitle) { this.assignmentTitle = assignmentTitle; }
        public String getSubmissionContent() { return submissionContent; }
        public void setSubmissionContent(String submissionContent) { this.submissionContent = submissionContent; }
        public String getAttachmentUrl() { return attachmentUrl; }
        public void setAttachmentUrl(String attachmentUrl) { this.attachmentUrl = attachmentUrl; }
        public LocalDateTime getSubmitTime() { return submitTime; }
        public void setSubmitTime(LocalDateTime submitTime) { this.submitTime = submitTime; }
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
        public Double getGrade() { return grade; }
        public void setGrade(Double grade) { this.grade = grade; }
        public String getFeedback() { return feedback; }
        public void setFeedback(String feedback) { this.feedback = feedback; }
        public LocalDateTime getGradeTime() { return gradeTime; }
        public void setGradeTime(LocalDateTime gradeTime) { this.gradeTime = gradeTime; }
        public String getGraderName() { return graderName; }
        public void setGraderName(String graderName) { this.graderName = graderName; }
    }

    /**
     * 学生分析响应DTO
     */
    public static class StudentAnalysisResponse {
        private Long studentId;
        private String studentName;
        private String learningStyle;
        private List<String> strongSubjects;
        private List<String> weakSubjects;
        private Double engagementScore;
        private Double performanceScore;
        private String riskLevel;
        private List<String> recommendations;
        private LocalDateTime lastAnalysis;
        private String analysisType;
        
        // Getters and Setters
        public Long getStudentId() { return studentId; }
        public void setStudentId(Long studentId) { this.studentId = studentId; }
        public String getStudentName() { return studentName; }
        public void setStudentName(String studentName) { this.studentName = studentName; }
        public String getLearningStyle() { return learningStyle; }
        public void setLearningStyle(String learningStyle) { this.learningStyle = learningStyle; }
        public List<String> getStrongSubjects() { return strongSubjects; }
        public void setStrongSubjects(List<String> strongSubjects) { this.strongSubjects = strongSubjects; }
        public List<String> getWeakSubjects() { return weakSubjects; }
        public void setWeakSubjects(List<String> weakSubjects) { this.weakSubjects = weakSubjects; }
        public Double getEngagementScore() { return engagementScore; }
        public void setEngagementScore(Double engagementScore) { this.engagementScore = engagementScore; }
        public Double getPerformanceScore() { return performanceScore; }
        public void setPerformanceScore(Double performanceScore) { this.performanceScore = performanceScore; }
        public String getRiskLevel() { return riskLevel; }
        public void setRiskLevel(String riskLevel) { this.riskLevel = riskLevel; }
        public List<String> getRecommendations() { return recommendations; }
        public void setRecommendations(List<String> recommendations) { this.recommendations = recommendations; }
        public LocalDateTime getLastAnalysis() { return lastAnalysis; }
        public void setLastAnalysis(LocalDateTime lastAnalysis) { this.lastAnalysis = lastAnalysis; }
        public String getAnalysisType() { return analysisType; }
        public void setAnalysisType(String analysisType) { this.analysisType = analysisType; }
    }
}