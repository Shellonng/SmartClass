package com.education.dto;

import jakarta.validation.constraints.Email;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;
import jakarta.validation.constraints.DecimalMax;
import jakarta.validation.constraints.DecimalMin;
import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

/**
 * 成绩相关DTO扩展类
 * 包含更多成绩相关的DTO定义
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
public class GradeDTOExtension {

    /**
     * 批量成绩响应DTO
     */
    public static class BatchGradeResponse {
        private Integer totalCount;
        private Integer successCount;
        private Integer failureCount;
        private List<GradeResponse> successGrades;
        private List<GradeError> errors;
        
        // Getters and Setters
        public Integer getTotalCount() { return totalCount; }
        public void setTotalCount(Integer totalCount) { this.totalCount = totalCount; }
        public Integer getSuccessCount() { return successCount; }
        public void setSuccessCount(Integer successCount) { this.successCount = successCount; }
        public Integer getFailureCount() { return failureCount; }
        public void setFailureCount(Integer failureCount) { this.failureCount = failureCount; }
        public List<GradeResponse> getSuccessGrades() { return successGrades; }
        public void setSuccessGrades(List<GradeResponse> successGrades) { this.successGrades = successGrades; }
        public List<GradeError> getErrors() { return errors; }
        public void setErrors(List<GradeError> errors) { this.errors = errors; }
        
        public static class GradeError {
            private Long studentId;
            private Long taskId;
            private String errorMessage;
            
            // Getters and Setters
            public Long getStudentId() { return studentId; }
            public void setStudentId(Long studentId) { this.studentId = studentId; }
            public Long getTaskId() { return taskId; }
            public void setTaskId(Long taskId) { this.taskId = taskId; }
            public String getErrorMessage() { return errorMessage; }
            public void setErrorMessage(String errorMessage) { this.errorMessage = errorMessage; }
        }
        
        public static class GradeResponse {
            private Long gradeId;
            private Long studentId;
            private String studentName;
            private Long taskId;
            private String taskTitle;
            private BigDecimal score;
            private String feedback;
            private LocalDateTime gradeTime;
            
            // Getters and Setters
            public Long getGradeId() { return gradeId; }
            public void setGradeId(Long gradeId) { this.gradeId = gradeId; }
            public Long getStudentId() { return studentId; }
            public void setStudentId(Long studentId) { this.studentId = studentId; }
            public String getStudentName() { return studentName; }
            public void setStudentName(String studentName) { this.studentName = studentName; }
            public Long getTaskId() { return taskId; }
            public void setTaskId(Long taskId) { this.taskId = taskId; }
            public String getTaskTitle() { return taskTitle; }
            public void setTaskTitle(String taskTitle) { this.taskTitle = taskTitle; }
            public BigDecimal getScore() { return score; }
            public void setScore(BigDecimal score) { this.score = score; }
            public String getFeedback() { return feedback; }
            public void setFeedback(String feedback) { this.feedback = feedback; }
            public LocalDateTime getGradeTime() { return gradeTime; }
            public void setGradeTime(LocalDateTime gradeTime) { this.gradeTime = gradeTime; }
        }
    }

    /**
     * 成绩列表响应DTO
     */
    public static class GradeListResponse {
        private Long gradeId;
        private Long studentId;
        private String studentName;
        private String studentNumber;
        private Long taskId;
        private String taskTitle;
        private String courseTitle;
        private BigDecimal score;
        private String grade;
        private String status;
        private LocalDateTime submitTime;
        private LocalDateTime gradeTime;
        private Boolean isLate;
        
        // Getters and Setters
        public Long getGradeId() { return gradeId; }
        public void setGradeId(Long gradeId) { this.gradeId = gradeId; }
        public Long getStudentId() { return studentId; }
        public void setStudentId(Long studentId) { this.studentId = studentId; }
        public String getStudentName() { return studentName; }
        public void setStudentName(String studentName) { this.studentName = studentName; }
        public String getStudentNumber() { return studentNumber; }
        public void setStudentNumber(String studentNumber) { this.studentNumber = studentNumber; }
        public Long getTaskId() { return taskId; }
        public void setTaskId(Long taskId) { this.taskId = taskId; }
        public String getTaskTitle() { return taskTitle; }
        public void setTaskTitle(String taskTitle) { this.taskTitle = taskTitle; }
        public String getCourseTitle() { return courseTitle; }
        public void setCourseTitle(String courseTitle) { this.courseTitle = courseTitle; }
        public BigDecimal getScore() { return score; }
        public void setScore(BigDecimal score) { this.score = score; }
        public String getGrade() { return grade; }
        public void setGrade(String grade) { this.grade = grade; }
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
        public LocalDateTime getSubmitTime() { return submitTime; }
        public void setSubmitTime(LocalDateTime submitTime) { this.submitTime = submitTime; }
        public LocalDateTime getGradeTime() { return gradeTime; }
        public void setGradeTime(LocalDateTime gradeTime) { this.gradeTime = gradeTime; }
        public Boolean getIsLate() { return isLate; }
        public void setIsLate(Boolean isLate) { this.isLate = isLate; }
    }

    /**
     * 课程成绩响应DTO
     */
    public static class CourseGradeResponse {
        private Long courseId;
        private String courseName;
        private BigDecimal totalScore;
        private String finalGrade;
        private Double completionRate;
        private List<TaskGrade> taskGrades;
        private List<GradeWeight> gradeWeights;
        private String status;
        private LocalDateTime lastUpdateTime;
        
        // Getters and Setters
        public Long getCourseId() { return courseId; }
        public void setCourseId(Long courseId) { this.courseId = courseId; }
        public String getCourseName() { return courseName; }
        public void setCourseName(String courseName) { this.courseName = courseName; }
        public BigDecimal getTotalScore() { return totalScore; }
        public void setTotalScore(BigDecimal totalScore) { this.totalScore = totalScore; }
        public String getFinalGrade() { return finalGrade; }
        public void setFinalGrade(String finalGrade) { this.finalGrade = finalGrade; }
        public Double getCompletionRate() { return completionRate; }
        public void setCompletionRate(Double completionRate) { this.completionRate = completionRate; }
        public List<TaskGrade> getTaskGrades() { return taskGrades; }
        public void setTaskGrades(List<TaskGrade> taskGrades) { this.taskGrades = taskGrades; }
        public List<GradeWeight> getGradeWeights() { return gradeWeights; }
        public void setGradeWeights(List<GradeWeight> gradeWeights) { this.gradeWeights = gradeWeights; }
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
        public LocalDateTime getLastUpdateTime() { return lastUpdateTime; }
        public void setLastUpdateTime(LocalDateTime lastUpdateTime) { this.lastUpdateTime = lastUpdateTime; }
        
        public static class TaskGrade {
            private Long taskId;
            private String taskTitle;
            private String taskType;
            private BigDecimal score;
            private BigDecimal maxScore;
            private String grade;
            private Double weight;
            private LocalDateTime gradeTime;
            
            // Getters and Setters
            public Long getTaskId() { return taskId; }
            public void setTaskId(Long taskId) { this.taskId = taskId; }
            public String getTaskTitle() { return taskTitle; }
            public void setTaskTitle(String taskTitle) { this.taskTitle = taskTitle; }
            public String getTaskType() { return taskType; }
            public void setTaskType(String taskType) { this.taskType = taskType; }
            public BigDecimal getScore() { return score; }
            public void setScore(BigDecimal score) { this.score = score; }
            public BigDecimal getMaxScore() { return maxScore; }
            public void setMaxScore(BigDecimal maxScore) { this.maxScore = maxScore; }
            public String getGrade() { return grade; }
            public void setGrade(String grade) { this.grade = grade; }
            public Double getWeight() { return weight; }
            public void setWeight(Double weight) { this.weight = weight; }
            public LocalDateTime getGradeTime() { return gradeTime; }
            public void setGradeTime(LocalDateTime gradeTime) { this.gradeTime = gradeTime; }
        }
        
        public static class GradeWeight {
            private String taskType;
            private Double weight;
            private String description;
            
            // Getters and Setters
            public String getTaskType() { return taskType; }
            public void setTaskType(String taskType) { this.taskType = taskType; }
            public Double getWeight() { return weight; }
            public void setWeight(Double weight) { this.weight = weight; }
            public String getDescription() { return description; }
            public void setDescription(String description) { this.description = description; }
        }
    }

    /**
     * 任务成绩详情响应DTO
     */
    public static class TaskGradeDetailResponse {
        private Long gradeId;
        private Long taskId;
        private String taskTitle;
        private String taskDescription;
        private BigDecimal score;
        private BigDecimal maxScore;
        private String grade;
        private String feedback;
        private List<GradeCriteria> gradeCriteria;
        private List<String> attachments;
        private LocalDateTime submitTime;
        private LocalDateTime gradeTime;
        private String graderName;
        private Boolean canAppeal;
        
        // Getters and Setters
        public Long getGradeId() { return gradeId; }
        public void setGradeId(Long gradeId) { this.gradeId = gradeId; }
        public Long getTaskId() { return taskId; }
        public void setTaskId(Long taskId) { this.taskId = taskId; }
        public String getTaskTitle() { return taskTitle; }
        public void setTaskTitle(String taskTitle) { this.taskTitle = taskTitle; }
        public String getTaskDescription() { return taskDescription; }
        public void setTaskDescription(String taskDescription) { this.taskDescription = taskDescription; }
        public BigDecimal getScore() { return score; }
        public void setScore(BigDecimal score) { this.score = score; }
        public BigDecimal getMaxScore() { return maxScore; }
        public void setMaxScore(BigDecimal maxScore) { this.maxScore = maxScore; }
        public String getGrade() { return grade; }
        public void setGrade(String grade) { this.grade = grade; }
        public String getFeedback() { return feedback; }
        public void setFeedback(String feedback) { this.feedback = feedback; }
        public List<GradeCriteria> getGradeCriteria() { return gradeCriteria; }
        public void setGradeCriteria(List<GradeCriteria> gradeCriteria) { this.gradeCriteria = gradeCriteria; }
        public List<String> getAttachments() { return attachments; }
        public void setAttachments(List<String> attachments) { this.attachments = attachments; }
        public LocalDateTime getSubmitTime() { return submitTime; }
        public void setSubmitTime(LocalDateTime submitTime) { this.submitTime = submitTime; }
        public LocalDateTime getGradeTime() { return gradeTime; }
        public void setGradeTime(LocalDateTime gradeTime) { this.gradeTime = gradeTime; }
        public String getGraderName() { return graderName; }
        public void setGraderName(String graderName) { this.graderName = graderName; }
        public Boolean getCanAppeal() { return canAppeal; }
        public void setCanAppeal(Boolean canAppeal) { this.canAppeal = canAppeal; }
        
        public static class GradeCriteria {
            private String criteriaName;
            private BigDecimal score;
            private BigDecimal maxScore;
            private String feedback;
            
            // Getters and Setters
            public String getCriteriaName() { return criteriaName; }
            public void setCriteriaName(String criteriaName) { this.criteriaName = criteriaName; }
            public BigDecimal getScore() { return score; }
            public void setScore(BigDecimal score) { this.score = score; }
            public BigDecimal getMaxScore() { return maxScore; }
            public void setMaxScore(BigDecimal maxScore) { this.maxScore = maxScore; }
            public String getFeedback() { return feedback; }
            public void setFeedback(String feedback) { this.feedback = feedback; }
        }
    }

    /**
     * 成绩趋势响应DTO
     */
    public static class GradeTrendResponse {
        private String timeRange;
        private List<GradeTrendData> trendData;
        private BigDecimal averageScore;
        private String trend; // IMPROVING, DECLINING, STABLE
        private String analysis;
        
        // Getters and Setters
        public String getTimeRange() { return timeRange; }
        public void setTimeRange(String timeRange) { this.timeRange = timeRange; }
        public List<GradeTrendData> getTrendData() { return trendData; }
        public void setTrendData(List<GradeTrendData> trendData) { this.trendData = trendData; }
        public BigDecimal getAverageScore() { return averageScore; }
        public void setAverageScore(BigDecimal averageScore) { this.averageScore = averageScore; }
        public String getTrend() { return trend; }
        public void setTrend(String trend) { this.trend = trend; }
        public String getAnalysis() { return analysis; }
        public void setAnalysis(String analysis) { this.analysis = analysis; }
        
        public static class GradeTrendData {
            private String period;
            private BigDecimal averageScore;
            private Integer taskCount;
            private String grade;
            
            // Getters and Setters
            public String getPeriod() { return period; }
            public void setPeriod(String period) { this.period = period; }
            public BigDecimal getAverageScore() { return averageScore; }
            public void setAverageScore(BigDecimal averageScore) { this.averageScore = averageScore; }
            public Integer getTaskCount() { return taskCount; }
            public void setTaskCount(Integer taskCount) { this.taskCount = taskCount; }
            public String getGrade() { return grade; }
            public void setGrade(String grade) { this.grade = grade; }
        }
    }

    /**
     * 成绩分布响应DTO
     */
    public static class GradeDistributionResponse {
        private Long courseId;
        private String courseName;
        private Long taskId;
        private String taskTitle;
        private Integer totalStudents;
        private BigDecimal averageScore;
        private List<ScoreRange> scoreRanges;
        private List<GradeLevel> gradeLevels;
        
        // Getters and Setters
        public Long getCourseId() { return courseId; }
        public void setCourseId(Long courseId) { this.courseId = courseId; }
        public String getCourseName() { return courseName; }
        public void setCourseName(String courseName) { this.courseName = courseName; }
        public Long getTaskId() { return taskId; }
        public void setTaskId(Long taskId) { this.taskId = taskId; }
        public String getTaskTitle() { return taskTitle; }
        public void setTaskTitle(String taskTitle) { this.taskTitle = taskTitle; }
        public Integer getTotalStudents() { return totalStudents; }
        public void setTotalStudents(Integer totalStudents) { this.totalStudents = totalStudents; }
        public BigDecimal getAverageScore() { return averageScore; }
        public void setAverageScore(BigDecimal averageScore) { this.averageScore = averageScore; }
        public List<ScoreRange> getScoreRanges() { return scoreRanges; }
        public void setScoreRanges(List<ScoreRange> scoreRanges) { this.scoreRanges = scoreRanges; }
        public List<GradeLevel> getGradeLevels() { return gradeLevels; }
        public void setGradeLevels(List<GradeLevel> gradeLevels) { this.gradeLevels = gradeLevels; }
        
        public static class ScoreRange {
            private String range;
            private Integer count;
            private Double percentage;
            
            // Getters and Setters
            public String getRange() { return range; }
            public void setRange(String range) { this.range = range; }
            public Integer getCount() { return count; }
            public void setCount(Integer count) { this.count = count; }
            public Double getPercentage() { return percentage; }
            public void setPercentage(Double percentage) { this.percentage = percentage; }
        }
        
        public static class GradeLevel {
            private String grade;
            private Integer count;
            private Double percentage;
            
            // Getters and Setters
            public String getGrade() { return grade; }
            public void setGrade(String grade) { this.grade = grade; }
            public Integer getCount() { return count; }
            public void setCount(Integer count) { this.count = count; }
            public Double getPercentage() { return percentage; }
            public void setPercentage(Double percentage) { this.percentage = percentage; }
        }
    }

    /**
     * 班级排名响应DTO
     */
    public static class ClassRankingResponse {
        private Long classId;
        private String className;
        private Integer totalStudents;
        private Integer currentRank;
        private BigDecimal currentScore;
        private String currentGrade;
        private List<StudentRank> rankings;
        private RankingStatistics statistics;
        
        // Getters and Setters
        public Long getClassId() { return classId; }
        public void setClassId(Long classId) { this.classId = classId; }
        public String getClassName() { return className; }
        public void setClassName(String className) { this.className = className; }
        public Integer getTotalStudents() { return totalStudents; }
        public void setTotalStudents(Integer totalStudents) { this.totalStudents = totalStudents; }
        public Integer getCurrentRank() { return currentRank; }
        public void setCurrentRank(Integer currentRank) { this.currentRank = currentRank; }
        public BigDecimal getCurrentScore() { return currentScore; }
        public void setCurrentScore(BigDecimal currentScore) { this.currentScore = currentScore; }
        public String getCurrentGrade() { return currentGrade; }
        public void setCurrentGrade(String currentGrade) { this.currentGrade = currentGrade; }
        public List<StudentRank> getRankings() { return rankings; }
        public void setRankings(List<StudentRank> rankings) { this.rankings = rankings; }
        public RankingStatistics getStatistics() { return statistics; }
        public void setStatistics(RankingStatistics statistics) { this.statistics = statistics; }
        
        public static class StudentRank {
            private Integer rank;
            private Long studentId;
            private String studentName;
            private String studentNumber;
            private BigDecimal score;
            private String grade;
            
            // Getters and Setters
            public Integer getRank() { return rank; }
            public void setRank(Integer rank) { this.rank = rank; }
            public Long getStudentId() { return studentId; }
            public void setStudentId(Long studentId) { this.studentId = studentId; }
            public String getStudentName() { return studentName; }
            public void setStudentName(String studentName) { this.studentName = studentName; }
            public String getStudentNumber() { return studentNumber; }
            public void setStudentNumber(String studentNumber) { this.studentNumber = studentNumber; }
            public BigDecimal getScore() { return score; }
            public void setScore(BigDecimal score) { this.score = score; }
            public String getGrade() { return grade; }
            public void setGrade(String grade) { this.grade = grade; }
        }
        
        public static class RankingStatistics {
            private BigDecimal averageScore;
            private BigDecimal medianScore;
            private BigDecimal topScore;
            private BigDecimal bottomScore;
            private Double standardDeviation;
            
            // Getters and Setters
            public BigDecimal getAverageScore() { return averageScore; }
            public void setAverageScore(BigDecimal averageScore) { this.averageScore = averageScore; }
            public BigDecimal getMedianScore() { return medianScore; }
            public void setMedianScore(BigDecimal medianScore) { this.medianScore = medianScore; }
            public BigDecimal getTopScore() { return topScore; }
            public void setTopScore(BigDecimal topScore) { this.topScore = topScore; }
            public BigDecimal getBottomScore() { return bottomScore; }
            public void setBottomScore(BigDecimal bottomScore) { this.bottomScore = bottomScore; }
            public Double getStandardDeviation() { return standardDeviation; }
            public void setStandardDeviation(Double standardDeviation) { this.standardDeviation = standardDeviation; }
        }
    }

    /**
     * 课程排名响应DTO
     */
    public static class CourseRankingResponse {
        private Long courseId;
        private String courseName;
        private Integer totalStudents;
        private Integer currentRank;
        private BigDecimal currentScore;
        private String currentGrade;
        private List<StudentRank> rankings;
        private RankingStatistics statistics;
        
        // Getters and Setters
        public Long getCourseId() { return courseId; }
        public void setCourseId(Long courseId) { this.courseId = courseId; }
        public String getCourseName() { return courseName; }
        public void setCourseName(String courseName) { this.courseName = courseName; }
        public Integer getTotalStudents() { return totalStudents; }
        public void setTotalStudents(Integer totalStudents) { this.totalStudents = totalStudents; }
        public Integer getCurrentRank() { return currentRank; }
        public void setCurrentRank(Integer currentRank) { this.currentRank = currentRank; }
        public BigDecimal getCurrentScore() { return currentScore; }
        public void setCurrentScore(BigDecimal currentScore) { this.currentScore = currentScore; }
        public String getCurrentGrade() { return currentGrade; }
        public void setCurrentGrade(String currentGrade) { this.currentGrade = currentGrade; }
        public List<StudentRank> getRankings() { return rankings; }
        public void setRankings(List<StudentRank> rankings) { this.rankings = rankings; }
        public RankingStatistics getStatistics() { return statistics; }
        public void setStatistics(RankingStatistics statistics) { this.statistics = statistics; }
        
        public static class StudentRank {
            private Integer rank;
            private Long studentId;
            private String studentName;
            private String studentNumber;
            private BigDecimal score;
            private String grade;
            
            // Getters and Setters
            public Integer getRank() { return rank; }
            public void setRank(Integer rank) { this.rank = rank; }
            public Long getStudentId() { return studentId; }
            public void setStudentId(Long studentId) { this.studentId = studentId; }
            public String getStudentName() { return studentName; }
            public void setStudentName(String studentName) { this.studentName = studentName; }
            public String getStudentNumber() { return studentNumber; }
            public void setStudentNumber(String studentNumber) { this.studentNumber = studentNumber; }
            public BigDecimal getScore() { return score; }
            public void setScore(BigDecimal score) { this.score = score; }
            public String getGrade() { return grade; }
            public void setGrade(String grade) { this.grade = grade; }
        }
        
        public static class RankingStatistics {
            private BigDecimal averageScore;
            private BigDecimal medianScore;
            private BigDecimal topScore;
            private BigDecimal bottomScore;
            private Double standardDeviation;
            
            // Getters and Setters
            public BigDecimal getAverageScore() { return averageScore; }
            public void setAverageScore(BigDecimal averageScore) { this.averageScore = averageScore; }
            public BigDecimal getMedianScore() { return medianScore; }
            public void setMedianScore(BigDecimal medianScore) { this.medianScore = medianScore; }
            public BigDecimal getTopScore() { return topScore; }
            public void setTopScore(BigDecimal topScore) { this.topScore = topScore; }
            public BigDecimal getBottomScore() { return bottomScore; }
            public void setBottomScore(BigDecimal bottomScore) { this.bottomScore = bottomScore; }
            public Double getStandardDeviation() { return standardDeviation; }
            public void setStandardDeviation(Double standardDeviation) { this.standardDeviation = standardDeviation; }
        }
    }
}