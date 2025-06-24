package com.education.dto;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

/**
 * 学生相关DTO扩展类
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
public class StudentDTOExtension {

    /**
     * 学生详情响应DTO
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
        private Integer totalStudyHours;
        private List<CourseInfo> enrolledCourses;
        private List<RecentActivity> recentActivities;
        
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
        public Integer getTotalStudyHours() { return totalStudyHours; }
        public void setTotalStudyHours(Integer totalStudyHours) { this.totalStudyHours = totalStudyHours; }
        public List<CourseInfo> getEnrolledCourses() { return enrolledCourses; }
        public void setEnrolledCourses(List<CourseInfo> enrolledCourses) { this.enrolledCourses = enrolledCourses; }
        public List<RecentActivity> getRecentActivities() { return recentActivities; }
        public void setRecentActivities(List<RecentActivity> recentActivities) { this.recentActivities = recentActivities; }
        
        public static class CourseInfo {
            private Long courseId;
            private String courseName;
            private String status;
            private Double progress;
            private Double grade;
            private LocalDateTime enrollTime;
            
            // Getters and Setters
            public Long getCourseId() { return courseId; }
            public void setCourseId(Long courseId) { this.courseId = courseId; }
            public String getCourseName() { return courseName; }
            public void setCourseName(String courseName) { this.courseName = courseName; }
            public String getStatus() { return status; }
            public void setStatus(String status) { this.status = status; }
            public Double getProgress() { return progress; }
            public void setProgress(Double progress) { this.progress = progress; }
            public Double getGrade() { return grade; }
            public void setGrade(Double grade) { this.grade = grade; }
            public LocalDateTime getEnrollTime() { return enrollTime; }
            public void setEnrollTime(LocalDateTime enrollTime) { this.enrollTime = enrollTime; }
        }
        
        public static class RecentActivity {
            private String activityType;
            private String description;
            private LocalDateTime activityTime;
            
            // Getters and Setters
            public String getActivityType() { return activityType; }
            public void setActivityType(String activityType) { this.activityType = activityType; }
            public String getDescription() { return description; }
            public void setDescription(String description) { this.description = description; }
            public LocalDateTime getActivityTime() { return activityTime; }
            public void setActivityTime(LocalDateTime activityTime) { this.activityTime = activityTime; }
        }
    }

    /**
     * 学生进度响应DTO
     */
    public static class StudentProgressResponse {
        private Long studentId;
        private String studentName;
        private Long courseId;
        private String courseName;
        private Double overallProgress;
        private List<ChapterProgress> chapterProgresses;
        private List<TaskProgress> taskProgresses;
        private Integer totalStudyTime;
        private LocalDateTime lastStudyTime;
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
        public Double getOverallProgress() { return overallProgress; }
        public void setOverallProgress(Double overallProgress) { this.overallProgress = overallProgress; }
        public List<ChapterProgress> getChapterProgresses() { return chapterProgresses; }
        public void setChapterProgresses(List<ChapterProgress> chapterProgresses) { this.chapterProgresses = chapterProgresses; }
        public List<TaskProgress> getTaskProgresses() { return taskProgresses; }
        public void setTaskProgresses(List<TaskProgress> taskProgresses) { this.taskProgresses = taskProgresses; }
        public Integer getTotalStudyTime() { return totalStudyTime; }
        public void setTotalStudyTime(Integer totalStudyTime) { this.totalStudyTime = totalStudyTime; }
        public LocalDateTime getLastStudyTime() { return lastStudyTime; }
        public void setLastStudyTime(LocalDateTime lastStudyTime) { this.lastStudyTime = lastStudyTime; }
        public String getCurrentStatus() { return currentStatus; }
        public void setCurrentStatus(String currentStatus) { this.currentStatus = currentStatus; }
        
        public static class ChapterProgress {
            private Long chapterId;
            private String chapterName;
            private Double progress;
            private String status;
            private Integer studyTime;
            
            // Getters and Setters
            public Long getChapterId() { return chapterId; }
            public void setChapterId(Long chapterId) { this.chapterId = chapterId; }
            public String getChapterName() { return chapterName; }
            public void setChapterName(String chapterName) { this.chapterName = chapterName; }
            public Double getProgress() { return progress; }
            public void setProgress(Double progress) { this.progress = progress; }
            public String getStatus() { return status; }
            public void setStatus(String status) { this.status = status; }
            public Integer getStudyTime() { return studyTime; }
            public void setStudyTime(Integer studyTime) { this.studyTime = studyTime; }
        }
        
        public static class TaskProgress {
            private Long taskId;
            private String taskName;
            private String status;
            private Double score;
            private LocalDateTime submitTime;
            
            // Getters and Setters
            public Long getTaskId() { return taskId; }
            public void setTaskId(Long taskId) { this.taskId = taskId; }
            public String getTaskName() { return taskName; }
            public void setTaskName(String taskName) { this.taskName = taskName; }
            public String getStatus() { return status; }
            public void setStatus(String status) { this.status = status; }
            public Double getScore() { return score; }
            public void setScore(Double score) { this.score = score; }
            public LocalDateTime getSubmitTime() { return submitTime; }
            public void setSubmitTime(LocalDateTime submitTime) { this.submitTime = submitTime; }
        }
    }

    /**
     * 学生成绩统计响应DTO
     */
    public static class StudentGradeStatisticsResponse {
        private Long studentId;
        private String studentName;
        private Double averageGrade;
        private Double highestGrade;
        private Double lowestGrade;
        private Integer totalTasks;
        private Integer completedTasks;
        private Integer pendingTasks;
        private Map<String, Double> subjectGrades;
        private List<GradeTrend> gradeTrends;
        private String gradeLevel;
        private Integer classRank;
        private Integer totalStudents;
        
        // Getters and Setters
        public Long getStudentId() { return studentId; }
        public void setStudentId(Long studentId) { this.studentId = studentId; }
        public String getStudentName() { return studentName; }
        public void setStudentName(String studentName) { this.studentName = studentName; }
        public Double getAverageGrade() { return averageGrade; }
        public void setAverageGrade(Double averageGrade) { this.averageGrade = averageGrade; }
        public Double getHighestGrade() { return highestGrade; }
        public void setHighestGrade(Double highestGrade) { this.highestGrade = highestGrade; }
        public Double getLowestGrade() { return lowestGrade; }
        public void setLowestGrade(Double lowestGrade) { this.lowestGrade = lowestGrade; }
        public Integer getTotalTasks() { return totalTasks; }
        public void setTotalTasks(Integer totalTasks) { this.totalTasks = totalTasks; }
        public Integer getCompletedTasks() { return completedTasks; }
        public void setCompletedTasks(Integer completedTasks) { this.completedTasks = completedTasks; }
        public Integer getPendingTasks() { return pendingTasks; }
        public void setPendingTasks(Integer pendingTasks) { this.pendingTasks = pendingTasks; }
        public Map<String, Double> getSubjectGrades() { return subjectGrades; }
        public void setSubjectGrades(Map<String, Double> subjectGrades) { this.subjectGrades = subjectGrades; }
        public List<GradeTrend> getGradeTrends() { return gradeTrends; }
        public void setGradeTrends(List<GradeTrend> gradeTrends) { this.gradeTrends = gradeTrends; }
        public String getGradeLevel() { return gradeLevel; }
        public void setGradeLevel(String gradeLevel) { this.gradeLevel = gradeLevel; }
        public Integer getClassRank() { return classRank; }
        public void setClassRank(Integer classRank) { this.classRank = classRank; }
        public Integer getTotalStudents() { return totalStudents; }
        public void setTotalStudents(Integer totalStudents) { this.totalStudents = totalStudents; }
        
        public static class GradeTrend {
            private String period;
            private Double averageGrade;
            private LocalDateTime date;
            
            // Getters and Setters
            public String getPeriod() { return period; }
            public void setPeriod(String period) { this.period = period; }
            public Double getAverageGrade() { return averageGrade; }
            public void setAverageGrade(Double averageGrade) { this.averageGrade = averageGrade; }
            public LocalDateTime getDate() { return date; }
            public void setDate(LocalDateTime date) { this.date = date; }
        }
    }

    /**
     * 学生提交记录响应DTO
     */
    public static class StudentSubmissionResponse {
        private Long submissionId;
        private Long taskId;
        private String taskName;
        private String taskType;
        private String submissionContent;
        private String status;
        private Double score;
        private String feedback;
        private LocalDateTime submitTime;
        private LocalDateTime gradeTime;
        private String graderName;
        private List<String> attachments;
        
        // Getters and Setters
        public Long getSubmissionId() { return submissionId; }
        public void setSubmissionId(Long submissionId) { this.submissionId = submissionId; }
        public Long getTaskId() { return taskId; }
        public void setTaskId(Long taskId) { this.taskId = taskId; }
        public String getTaskName() { return taskName; }
        public void setTaskName(String taskName) { this.taskName = taskName; }
        public String getTaskType() { return taskType; }
        public void setTaskType(String taskType) { this.taskType = taskType; }
        public String getSubmissionContent() { return submissionContent; }
        public void setSubmissionContent(String submissionContent) { this.submissionContent = submissionContent; }
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
        public Double getScore() { return score; }
        public void setScore(Double score) { this.score = score; }
        public String getFeedback() { return feedback; }
        public void setFeedback(String feedback) { this.feedback = feedback; }
        public LocalDateTime getSubmitTime() { return submitTime; }
        public void setSubmitTime(LocalDateTime submitTime) { this.submitTime = submitTime; }
        public LocalDateTime getGradeTime() { return gradeTime; }
        public void setGradeTime(LocalDateTime gradeTime) { this.gradeTime = gradeTime; }
        public String getGraderName() { return graderName; }
        public void setGraderName(String graderName) { this.graderName = graderName; }
        public List<String> getAttachments() { return attachments; }
        public void setAttachments(List<String> attachments) { this.attachments = attachments; }
    }

    /**
     * 学生分析响应DTO
     */
    public static class StudentAnalysisResponse {
        private Long studentId;
        private String studentName;
        private String timeRange;
        private LearningBehavior learningBehavior;
        private PerformanceAnalysis performanceAnalysis;
        private EngagementMetrics engagementMetrics;
        private List<String> strengths;
        private List<String> weaknesses;
        private List<String> recommendations;
        private LocalDateTime analysisTime;
        
        // Getters and Setters
        public Long getStudentId() { return studentId; }
        public void setStudentId(Long studentId) { this.studentId = studentId; }
        public String getStudentName() { return studentName; }
        public void setStudentName(String studentName) { this.studentName = studentName; }
        public String getTimeRange() { return timeRange; }
        public void setTimeRange(String timeRange) { this.timeRange = timeRange; }
        public LearningBehavior getLearningBehavior() { return learningBehavior; }
        public void setLearningBehavior(LearningBehavior learningBehavior) { this.learningBehavior = learningBehavior; }
        public PerformanceAnalysis getPerformanceAnalysis() { return performanceAnalysis; }
        public void setPerformanceAnalysis(PerformanceAnalysis performanceAnalysis) { this.performanceAnalysis = performanceAnalysis; }
        public EngagementMetrics getEngagementMetrics() { return engagementMetrics; }
        public void setEngagementMetrics(EngagementMetrics engagementMetrics) { this.engagementMetrics = engagementMetrics; }
        public List<String> getStrengths() { return strengths; }
        public void setStrengths(List<String> strengths) { this.strengths = strengths; }
        public List<String> getWeaknesses() { return weaknesses; }
        public void setWeaknesses(List<String> weaknesses) { this.weaknesses = weaknesses; }
        public List<String> getRecommendations() { return recommendations; }
        public void setRecommendations(List<String> recommendations) { this.recommendations = recommendations; }
        public LocalDateTime getAnalysisTime() { return analysisTime; }
        public void setAnalysisTime(LocalDateTime analysisTime) { this.analysisTime = analysisTime; }
        
        public static class LearningBehavior {
            private Integer totalStudyTime;
            private Double averageSessionDuration;
            private Integer loginFrequency;
            private String preferredStudyTime;
            private List<String> activeSubjects;
            
            // Getters and Setters
            public Integer getTotalStudyTime() { return totalStudyTime; }
            public void setTotalStudyTime(Integer totalStudyTime) { this.totalStudyTime = totalStudyTime; }
            public Double getAverageSessionDuration() { return averageSessionDuration; }
            public void setAverageSessionDuration(Double averageSessionDuration) { this.averageSessionDuration = averageSessionDuration; }
            public Integer getLoginFrequency() { return loginFrequency; }
            public void setLoginFrequency(Integer loginFrequency) { this.loginFrequency = loginFrequency; }
            public String getPreferredStudyTime() { return preferredStudyTime; }
            public void setPreferredStudyTime(String preferredStudyTime) { this.preferredStudyTime = preferredStudyTime; }
            public List<String> getActiveSubjects() { return activeSubjects; }
            public void setActiveSubjects(List<String> activeSubjects) { this.activeSubjects = activeSubjects; }
        }
        
        public static class PerformanceAnalysis {
            private Double averageScore;
            private String performanceTrend;
            private Map<String, Double> subjectPerformance;
            private Integer improvementRate;
            
            // Getters and Setters
            public Double getAverageScore() { return averageScore; }
            public void setAverageScore(Double averageScore) { this.averageScore = averageScore; }
            public String getPerformanceTrend() { return performanceTrend; }
            public void setPerformanceTrend(String performanceTrend) { this.performanceTrend = performanceTrend; }
            public Map<String, Double> getSubjectPerformance() { return subjectPerformance; }
            public void setSubjectPerformance(Map<String, Double> subjectPerformance) { this.subjectPerformance = subjectPerformance; }
            public Integer getImprovementRate() { return improvementRate; }
            public void setImprovementRate(Integer improvementRate) { this.improvementRate = improvementRate; }
        }
        
        public static class EngagementMetrics {
            private Double participationRate;
            private Integer discussionPosts;
            private Integer resourceAccess;
            private Double completionRate;
            
            // Getters and Setters
            public Double getParticipationRate() { return participationRate; }
            public void setParticipationRate(Double participationRate) { this.participationRate = participationRate; }
            public Integer getDiscussionPosts() { return discussionPosts; }
            public void setDiscussionPosts(Integer discussionPosts) { this.discussionPosts = discussionPosts; }
            public Integer getResourceAccess() { return resourceAccess; }
            public void setResourceAccess(Integer resourceAccess) { this.resourceAccess = resourceAccess; }
            public Double getCompletionRate() { return completionRate; }
            public void setCompletionRate(Double completionRate) { this.completionRate = completionRate; }
        }
    }

    /**
     * 学生导出请求DTO
     */
    public static class StudentExportRequest {
        private String exportType; // EXCEL, CSV, PDF
        private List<Long> studentIds;
        private List<String> fields;
        private String dateRange;
        private Boolean includeGrades;
        private Boolean includeProgress;
        
        // Getters and Setters
        public String getExportType() { return exportType; }
        public void setExportType(String exportType) { this.exportType = exportType; }
        public List<Long> getStudentIds() { return studentIds; }
        public void setStudentIds(List<Long> studentIds) { this.studentIds = studentIds; }
        public List<String> getFields() { return fields; }
        public void setFields(List<String> fields) { this.fields = fields; }
        public String getDateRange() { return dateRange; }
        public void setDateRange(String dateRange) { this.dateRange = dateRange; }
        public Boolean getIncludeGrades() { return includeGrades; }
        public void setIncludeGrades(Boolean includeGrades) { this.includeGrades = includeGrades; }
        public Boolean getIncludeProgress() { return includeProgress; }
        public void setIncludeProgress(Boolean includeProgress) { this.includeProgress = includeProgress; }
    }

    /**
     * 学生排名响应DTO
     */
    public static class StudentRankResponse {
        private Long studentId;
        private String studentName;
        private String studentNumber;
        private Double score;
        private Integer rank;
        private String rankType;
        private String className;
        private Double percentile;
        private String badge;
        
        // Getters and Setters
        public Long getStudentId() { return studentId; }
        public void setStudentId(Long studentId) { this.studentId = studentId; }
        public String getStudentName() { return studentName; }
        public void setStudentName(String studentName) { this.studentName = studentName; }
        public String getStudentNumber() { return studentNumber; }
        public void setStudentNumber(String studentNumber) { this.studentNumber = studentNumber; }
        public Double getScore() { return score; }
        public void setScore(Double score) { this.score = score; }
        public Integer getRank() { return rank; }
        public void setRank(Integer rank) { this.rank = rank; }
        public String getRankType() { return rankType; }
        public void setRankType(String rankType) { this.rankType = rankType; }
        public String getClassName() { return className; }
        public void setClassName(String className) { this.className = className; }
        public Double getPercentile() { return percentile; }
        public void setPercentile(Double percentile) { this.percentile = percentile; }
        public String getBadge() { return badge; }
        public void setBadge(String badge) { this.badge = badge; }
    }
}