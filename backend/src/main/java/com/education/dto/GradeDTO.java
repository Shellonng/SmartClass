package com.education.dto;

import jakarta.validation.constraints.*;

import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;

import java.math.BigDecimal;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.List;

/**
 * 成绩相关DTO
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
public class GradeDTO {

    /**
     * 成绩创建请求DTO
     */
    public static class GradeCreateRequest {
        @NotNull(message = "学生ID不能为空")
        private Long studentId;
        
        @NotNull(message = "任务ID不能为空")
        private Long taskId;
        
        @DecimalMin(value = "0.0", message = "分数不能小于0")
        @DecimalMax(value = "100.0", message = "分数不能大于100")
        private BigDecimal score;
        
        private String feedback;
        private String gradeType; // AUTO, MANUAL
        private String status; // PENDING, GRADED, REVIEWED
        
        // Getters and Setters
        public Long getStudentId() { return studentId; }
        public void setStudentId(Long studentId) { this.studentId = studentId; }
        public Long getTaskId() { return taskId; }
        public void setTaskId(Long taskId) { this.taskId = taskId; }
        public BigDecimal getScore() { return score; }
        public void setScore(BigDecimal score) { this.score = score; }
        public String getFeedback() { return feedback; }
        public void setFeedback(String feedback) { this.feedback = feedback; }
        public String getGradeType() { return gradeType; }
        public void setGradeType(String gradeType) { this.gradeType = gradeType; }
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
    }

    /**
     * 成绩响应DTO
     */
    public static class GradeResponse {
        private Long gradeId;
        private Long studentId;
        private String studentName;
        private String studentNumber;
        private Long taskId;
        private String taskTitle;
        private BigDecimal score;
        private String feedback;
        private String gradeType;
        private String status;
        private LocalDateTime submitTime;
        private LocalDateTime gradeTime;
        private String graderName;
        
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
        public BigDecimal getScore() { return score; }
        public void setScore(BigDecimal score) { this.score = score; }
        public String getFeedback() { return feedback; }
        public void setFeedback(String feedback) { this.feedback = feedback; }
        public String getGradeType() { return gradeType; }
        public void setGradeType(String gradeType) { this.gradeType = gradeType; }
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
        public LocalDateTime getSubmitTime() { return submitTime; }
        public void setSubmitTime(LocalDateTime submitTime) { this.submitTime = submitTime; }
        public LocalDateTime getGradeTime() { return gradeTime; }
        public void setGradeTime(LocalDateTime gradeTime) { this.gradeTime = gradeTime; }
        public String getGraderName() { return graderName; }
        public void setGraderName(String graderName) { this.graderName = graderName; }
    }

    /**
     * 成绩更新请求DTO
     */
    public static class GradeUpdateRequest {
        @DecimalMin(value = "0.0", message = "分数不能小于0")
        @DecimalMax(value = "100.0", message = "分数不能大于100")
        private BigDecimal score;
        
        private String feedback;
        private String status;
        
        // Getters and Setters
        public BigDecimal getScore() { return score; }
        public void setScore(BigDecimal score) { this.score = score; }
        public String getFeedback() { return feedback; }
        public void setFeedback(String feedback) { this.feedback = feedback; }
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
    }

    /**
     * 成绩统计响应DTO
     */
    public static class GradeStatisticsResponse {
        private Long studentId;
        private Long taskId;
        private String taskTitle;
        private Integer totalStudents;
        private Integer gradedCount;
        private Integer pendingCount;
        private BigDecimal averageScore;
        private BigDecimal maxScore;
        private BigDecimal minScore;
        private List<ScoreDistribution> scoreDistribution;
        
        // Getters and Setters
        public Long getTaskId() { return taskId; }
        public void setTaskId(Long taskId) { this.taskId = taskId; }
        public String getTaskTitle() { return taskTitle; }
        public void setTaskTitle(String taskTitle) { this.taskTitle = taskTitle; }
        public Integer getTotalStudents() { return totalStudents; }
        public void setTotalStudents(Integer totalStudents) { this.totalStudents = totalStudents; }
        public Integer getGradedCount() { return gradedCount; }
        public void setGradedCount(Integer gradedCount) { this.gradedCount = gradedCount; }
        public Integer getPendingCount() { return pendingCount; }
        public void setPendingCount(Integer pendingCount) { this.pendingCount = pendingCount; }
        public BigDecimal getAverageScore() { return averageScore; }
        public void setAverageScore(BigDecimal averageScore) { this.averageScore = averageScore; }
        public BigDecimal getMaxScore() { return maxScore; }
        public void setMaxScore(BigDecimal maxScore) { this.maxScore = maxScore; }
        public BigDecimal getMinScore() { return minScore; }
        public void setMinScore(BigDecimal minScore) { this.minScore = minScore; }
        public List<ScoreDistribution> getScoreDistribution() { return scoreDistribution; }
        public void setScoreDistribution(List<ScoreDistribution> scoreDistribution) {
        this.scoreDistribution = scoreDistribution;
    }

    public Long getStudentId() {
        return studentId;
    }

    public void setStudentId(Long studentId) {
        this.studentId = studentId;
    }
    
    public void setTotalTasks(int totalTasks) {
        // 使用现有的totalStudents字段
        this.totalStudents = totalTasks;
    }
    
    public void setPassingRate(double passingRate) {
        // 可以添加新字段或使用现有逻辑
    }
    }

    /**
     * 分数分布DTO
     */
    public static class ScoreDistribution {
        private String range; // 0-60, 60-70, 70-80, 80-90, 90-100
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

    /**
     * 批量评分请求DTO
     */
    public static class BatchGradeRequest {
        private List<GradeCreateRequest> grades;
        private String gradeType;
        
        // Getters and Setters
        public List<GradeCreateRequest> getGrades() { return grades; }
        public void setGrades(List<GradeCreateRequest> grades) { this.grades = grades; }
        public String getGradeType() { return gradeType; }
        public void setGradeType(String gradeType) { this.gradeType = gradeType; }
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
        private Long courseId;
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
        public Long getCourseId() { return courseId; }
        public void setCourseId(Long courseId) { this.courseId = courseId; }
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

        public void setCourseName(@NotBlank(message = "课程名称不能为空") @Size(max = 100, message = "课程名称长度不能超过100个字符") String courseName) {
            this.courseTitle = courseName;
        }
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
        private Long studentId;
        private Long taskId;
        private String taskTitle;
        private String taskDescription;
        private BigDecimal score;
        private BigDecimal originalScore;
        private BigDecimal maxScore;
        private String grade;
        private String feedback;
        private List<GradeCriteria> gradeCriteria;
        private List<String> attachments;
        private LocalDateTime submitTime;
        private LocalDateTime gradeTime;
        private String graderName;
        private Boolean canAppeal;
        private Boolean isLate;
        private Integer lateDays;
        
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
        public Long getStudentId() { return studentId; }
        public void setStudentId(Long studentId) { this.studentId = studentId; }
        public BigDecimal getOriginalScore() { return originalScore; }
        public void setOriginalScore(BigDecimal originalScore) { this.originalScore = originalScore; }
        public Boolean getIsLate() { return isLate; }
        public void setIsLate(Boolean isLate) { this.isLate = isLate; }
        public Integer getLateDays() { return lateDays; }
        public void setLateDays(Integer lateDays) { this.lateDays = lateDays; }
        public void setDeductedPoints(BigDecimal deductedPoints) { /* Implementation needed */ }
        public void setGradingTime(LocalDateTime gradingTime) { this.gradeTime = gradingTime; }
        
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
        
        public void setStudentId(Long studentId) {
            // 可以添加studentId字段或使用现有逻辑
        }
        
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
        
        public void setStudentId(Long studentId) {
            // 可以添加studentId字段或使用现有逻辑
        }
        
        public void setDistributionData(List<Object> distributionData) {
            // 可以将distributionData转换为scoreRanges或gradeLevels
        }
        
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
        
        public void setStudentId(Long studentId) {
            // 可以添加studentId字段或使用现有逻辑
        }
        
        public void setRanking(int ranking) {
            this.currentRank = ranking;
        }
        
        public void setPercentile(double percentile) {
            // 可以添加percentile字段或使用现有逻辑
        }
        
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
        
        public void setStudentId(Long studentId) {
            // 可以添加studentId字段或使用现有逻辑
        }
        
        public void setRanking(int ranking) {
            this.currentRank = ranking;
        }
        
        public void setPercentile(double percentile) {
            // 可以添加percentile字段或使用现有逻辑
        }
        
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
     * 成绩对比分析响应DTO
     */
    public static class GradeComparisonResponse {
        private String comparisonType;
        private List<ComparisonData> comparisonData;
        private String analysis;
        private List<String> recommendations;
        
        // Getters and Setters
        public String getComparisonType() { return comparisonType; }
        public void setComparisonType(String comparisonType) { this.comparisonType = comparisonType; }
        public List<ComparisonData> getComparisonData() { return comparisonData; }
        public void setComparisonData(List<ComparisonData> comparisonData) { this.comparisonData = comparisonData; }
        public String getAnalysis() { return analysis; }
        public void setAnalysis(String analysis) { this.analysis = analysis; }
        
        public void setStudentId(Long studentId) {
            // 可以添加studentId字段或使用现有逻辑
        }
        public List<String> getRecommendations() { return recommendations; }
        public void setRecommendations(List<String> recommendations) { this.recommendations = recommendations; }
        public void setCourseId(Long courseId) { /* Implementation needed */ }
        public void setAbilities(List<Object> abilities) { /* Implementation needed */ }
        
        public static class ComparisonData {
            private String label;
            private BigDecimal currentScore;
            private BigDecimal compareScore;
            private BigDecimal difference;
            private String trend;
            
            // Getters and Setters
            public String getLabel() { return label; }
            public void setLabel(String label) { this.label = label; }
            public BigDecimal getCurrentScore() { return currentScore; }
            public void setCurrentScore(BigDecimal currentScore) { this.currentScore = currentScore; }
            public BigDecimal getCompareScore() { return compareScore; }
            public void setCompareScore(BigDecimal compareScore) { this.compareScore = compareScore; }
            public BigDecimal getDifference() { return difference; }
            public void setDifference(BigDecimal difference) { this.difference = difference; }
            public String getTrend() { return trend; }
            public void setTrend(String trend) { this.trend = trend; }
        }
    }

    /**
     * 成绩对比请求DTO
     */
    public static class GradeComparisonRequest {
        private String comparisonType; // SEMESTER, YEAR, COURSE, CLASS
        private String targetPeriod;
        private Long targetCourseId;
        private Long targetClassId;
        
        // Getters and Setters
        public String getComparisonType() { return comparisonType; }
        public void setComparisonType(String comparisonType) { this.comparisonType = comparisonType; }
        public String getTargetPeriod() { return targetPeriod; }
        public void setTargetPeriod(String targetPeriod) { this.targetPeriod = targetPeriod; }
        public Long getTargetCourseId() { return targetCourseId; }
        public void setTargetCourseId(Long targetCourseId) { this.targetCourseId = targetCourseId; }
        public Long getTargetClassId() { return targetClassId; }
        public void setTargetClassId(Long targetClassId) { this.targetClassId = targetClassId; }
    }

    /**
     * 学期成绩汇总响应DTO
     */
    public static class SemesterGradeSummaryResponse {
        private String semester;
        private Integer totalCourses;
        private Integer completedCourses;
        private BigDecimal averageScore;
        private String overallGrade;
        private List<CourseGradeSummary> courseGrades;
        private List<String> achievements;
        private List<String> warnings;
        
        // Getters and Setters
        public String getSemester() { return semester; }
        public void setSemester(String semester) { this.semester = semester; }
        public Integer getTotalCourses() { return totalCourses; }
        public void setTotalCourses(Integer totalCourses) { this.totalCourses = totalCourses; }
        public Integer getCompletedCourses() { return completedCourses; }
        public void setCompletedCourses(Integer completedCourses) { this.completedCourses = completedCourses; }
        public BigDecimal getAverageScore() { return averageScore; }
        public void setAverageScore(BigDecimal averageScore) { this.averageScore = averageScore; }
        public String getOverallGrade() { return overallGrade; }
        public void setOverallGrade(String overallGrade) { this.overallGrade = overallGrade; }
        public List<CourseGradeSummary> getCourseGrades() { return courseGrades; }
        public void setCourseGrades(List<CourseGradeSummary> courseGrades) { this.courseGrades = courseGrades; }
        public List<String> getAchievements() { return achievements; }
        public void setAchievements(List<String> achievements) { this.achievements = achievements; }
        public List<String> getWarnings() { return warnings; }
        public void setWarnings(List<String> warnings) { this.warnings = warnings; }
        
        public void setStudentId(Long studentId) {
            // 可以添加studentId字段或使用现有逻辑
        }
        
        public void setAverageGPA(double averageGPA) {
            // 可以将GPA转换为averageScore或添加新字段
            this.averageScore = BigDecimal.valueOf(averageGPA);
        }
        
        public static class CourseGradeSummary {
            private Long courseId;
            private String courseName;
            private BigDecimal finalScore;
            private String finalGrade;
            private String status;
            
            // Getters and Setters
            public Long getCourseId() { return courseId; }
            public void setCourseId(Long courseId) { this.courseId = courseId; }
            public String getCourseName() { return courseName; }
            public void setCourseName(String courseName) { this.courseName = courseName; }
            public BigDecimal getFinalScore() { return finalScore; }
            public void setFinalScore(BigDecimal finalScore) { this.finalScore = finalScore; }
            public String getFinalGrade() { return finalGrade; }
            public void setFinalGrade(String finalGrade) { this.finalGrade = finalGrade; }
            public String getStatus() { return status; }
            public void setStatus(String status) { this.status = status; }
        }
    }

    /**
     * 年度成绩汇总响应DTO
     */
    public static class YearlyGradeSummaryResponse {
        private Integer year;
        private Integer totalSemesters;
        private Integer completedSemesters;
        private BigDecimal yearlyAverageScore;
        private String yearlyGrade;
        private List<SemesterSummary> semesterSummaries;
        private List<String> yearlyAchievements;
        private String progressAnalysis;
        
        // Getters and Setters
        public Integer getYear() { return year; }
        public void setYear(Integer year) { this.year = year; }
        public Integer getTotalSemesters() { return totalSemesters; }
        public void setTotalSemesters(Integer totalSemesters) { this.totalSemesters = totalSemesters; }
        public Integer getCompletedSemesters() { return completedSemesters; }
        public void setCompletedSemesters(Integer completedSemesters) { this.completedSemesters = completedSemesters; }
        public BigDecimal getYearlyAverageScore() { return yearlyAverageScore; }
        public void setYearlyAverageScore(BigDecimal yearlyAverageScore) { this.yearlyAverageScore = yearlyAverageScore; }
        public String getYearlyGrade() { return yearlyGrade; }
        public void setYearlyGrade(String yearlyGrade) { this.yearlyGrade = yearlyGrade; }
        public List<SemesterSummary> getSemesterSummaries() { return semesterSummaries; }
        public void setSemesterSummaries(List<SemesterSummary> semesterSummaries) { this.semesterSummaries = semesterSummaries; }
        public List<String> getYearlyAchievements() { return yearlyAchievements; }
        public void setYearlyAchievements(List<String> yearlyAchievements) { this.yearlyAchievements = yearlyAchievements; }
        public String getProgressAnalysis() { return progressAnalysis; }
        public void setProgressAnalysis(String progressAnalysis) { this.progressAnalysis = progressAnalysis; }
        public void setStudentId(Long studentId) { /* Implementation needed */ }
        public void setGoals(List<Object> goals) { /* Implementation needed */ }
        public void setOverallProgress(double overallProgress) { /* Implementation needed */ }
        public void setTotalCourses(int totalCourses) { /* Implementation needed */ }
        public void setAverageGPA(double averageGPA) { /* Implementation needed */ }
        
        public static class SemesterSummary {
            private String semester;
            private BigDecimal averageScore;
            private String grade;
            private Integer courseCount;
            
            // Getters and Setters
            public String getSemester() { return semester; }
            public void setSemester(String semester) { this.semester = semester; }
            public BigDecimal getAverageScore() { return averageScore; }
            public void setAverageScore(BigDecimal averageScore) { this.averageScore = averageScore; }
            public String getGrade() { return grade; }
            public void setGrade(String grade) { this.grade = grade; }
            public Integer getCourseCount() { return courseCount; }
            public void setCourseCount(Integer courseCount) { this.courseCount = courseCount; }
        }
    }

    /**
     * 成绩预警响应DTO
     */
    public static class GradeWarningResponse {
        private String warningType; // LOW_SCORE, DECLINING_TREND, MISSING_SUBMISSION
        private String severity; // HIGH, MEDIUM, LOW
        private String message;
        private Long courseId;
        private String courseName;
        private Long taskId;
        private String taskTitle;
        private BigDecimal currentScore;
        private BigDecimal thresholdScore;
        private LocalDateTime warningTime;
        private List<String> suggestions;
        
        // Getters and Setters
        public String getWarningType() { return warningType; }
        public void setWarningType(String warningType) { this.warningType = warningType; }
        public String getSeverity() { return severity; }
        public void setSeverity(String severity) { this.severity = severity; }
        public String getMessage() { return message; }
        public void setMessage(String message) { this.message = message; }
        public Long getCourseId() { return courseId; }
        public void setCourseId(Long courseId) { this.courseId = courseId; }
        public String getCourseName() { return courseName; }
        public void setCourseName(String courseName) { this.courseName = courseName; }
        public Long getTaskId() { return taskId; }
        public void setTaskId(Long taskId) { this.taskId = taskId; }
        public String getTaskTitle() { return taskTitle; }
        public void setTaskTitle(String taskTitle) { this.taskTitle = taskTitle; }
        public BigDecimal getCurrentScore() { return currentScore; }
        public void setCurrentScore(BigDecimal currentScore) { this.currentScore = currentScore; }
        public BigDecimal getThresholdScore() { return thresholdScore; }
        public void setThresholdScore(BigDecimal thresholdScore) { this.thresholdScore = thresholdScore; }
        public LocalDateTime getWarningTime() { return warningTime; }
        public void setWarningTime(LocalDateTime warningTime) { this.warningTime = warningTime; }
        public List<String> getSuggestions() { return suggestions; }
        public void setSuggestions(List<String> suggestions) { this.suggestions = suggestions; }
    }

    /**
     * 改进建议响应DTO
     */
    public static class ImprovementSuggestionResponse {
        private Long courseId;
        private String courseName;
        private BigDecimal currentScore;
        private String currentGrade;
        private List<ImprovementArea> improvementAreas;
        private List<StudyPlan> studyPlans;
        private List<String> resources;
        
        // Getters and Setters
        public Long getCourseId() { return courseId; }
        public void setCourseId(Long courseId) { this.courseId = courseId; }
        public String getCourseName() { return courseName; }
        public void setCourseName(String courseName) { this.courseName = courseName; }
        public BigDecimal getCurrentScore() { return currentScore; }
        public void setCurrentScore(BigDecimal currentScore) { this.currentScore = currentScore; }
        public String getCurrentGrade() { return currentGrade; }
        public void setCurrentGrade(String currentGrade) { this.currentGrade = currentGrade; }
        public List<ImprovementArea> getImprovementAreas() { return improvementAreas; }
        public void setImprovementAreas(List<ImprovementArea> improvementAreas) { this.improvementAreas = improvementAreas; }
        public List<StudyPlan> getStudyPlans() { return studyPlans; }
        public void setStudyPlans(List<StudyPlan> studyPlans) { this.studyPlans = studyPlans; }
        public List<String> getResources() { return resources; }
        public void setResources(List<String> resources) { this.resources = resources; }
        public void setStudentId(Long studentId) { /* Implementation needed */ }
        public void setSuggestions(List<String> suggestions) { /* Implementation needed */ }
        
        public static class ImprovementArea {
            private String area;
            private String description;
            private String priority;
            private BigDecimal potentialImprovement;
            
            // Getters and Setters
            public String getArea() { return area; }
            public void setArea(String area) { this.area = area; }
            public String getDescription() { return description; }
            public void setDescription(String description) { this.description = description; }
            public String getPriority() { return priority; }
            public void setPriority(String priority) { this.priority = priority; }
            public BigDecimal getPotentialImprovement() { return potentialImprovement; }
            public void setPotentialImprovement(BigDecimal potentialImprovement) { this.potentialImprovement = potentialImprovement; }
        }
        
        public static class StudyPlan {
            private String title;
            private String description;
            private Integer estimatedHours;
            private String difficulty;
            private List<String> steps;
            
            // Getters and Setters
            public String getTitle() { return title; }
            public void setTitle(String title) { this.title = title; }
            public String getDescription() { return description; }
            public void setDescription(String description) { this.description = description; }
            public Integer getEstimatedHours() { return estimatedHours; }
            public void setEstimatedHours(Integer estimatedHours) { this.estimatedHours = estimatedHours; }
            public String getDifficulty() { return difficulty; }
            public void setDifficulty(String difficulty) { this.difficulty = difficulty; }
            public List<String> getSteps() { return steps; }
            public void setSteps(List<String> steps) { this.steps = steps; }
        }
    }

    /**
     * 学习目标完成情况响应DTO
     */
    public static class LearningGoalProgressResponse {
        private Long goalId;
        private String goalTitle;
        private String goalDescription;
        private BigDecimal targetScore;
        private BigDecimal currentScore;
        private Double completionRate;
        private String status; // NOT_STARTED, IN_PROGRESS, COMPLETED, OVERDUE
        private LocalDateTime startDate;
        private LocalDateTime targetDate;
        private List<GoalMilestone> milestones;
        private String progressAnalysis;
        
        // Getters and Setters
        public Long getGoalId() { return goalId; }
        public void setGoalId(Long goalId) { this.goalId = goalId; }
        public String getGoalTitle() { return goalTitle; }
        public void setGoalTitle(String goalTitle) { this.goalTitle = goalTitle; }
        public String getGoalDescription() { return goalDescription; }
        public void setGoalDescription(String goalDescription) { this.goalDescription = goalDescription; }
        public BigDecimal getTargetScore() { return targetScore; }
        public void setTargetScore(BigDecimal targetScore) { this.targetScore = targetScore; }
        public BigDecimal getCurrentScore() { return currentScore; }
        public void setCurrentScore(BigDecimal currentScore) { this.currentScore = currentScore; }
        public Double getCompletionRate() { return completionRate; }
        public void setCompletionRate(Double completionRate) { this.completionRate = completionRate; }
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
        public LocalDateTime getStartDate() { return startDate; }
        public void setStartDate(LocalDateTime startDate) { this.startDate = startDate; }
        public LocalDateTime getTargetDate() { return targetDate; }
        public void setTargetDate(LocalDateTime targetDate) { this.targetDate = targetDate; }
        public List<GoalMilestone> getMilestones() { return milestones; }
        public void setMilestones(List<GoalMilestone> milestones) { this.milestones = milestones; }
        public String getProgressAnalysis() { return progressAnalysis; }
        public void setProgressAnalysis(String progressAnalysis) { this.progressAnalysis = progressAnalysis; }
        public void setStudentId(Long studentId) { /* Implementation needed */ }
        public void setGoals(List<Object> goals) { /* Implementation needed */ }
        public void setOverallProgress(double overallProgress) { /* Implementation needed */ }
        public void setTotalCourses(int totalCourses) { /* Implementation needed */ }
        public void setAverageGPA(double averageGPA) { /* Implementation needed */ }
        
        public static class GoalMilestone {
            private String title;
            private String description;
            private BigDecimal targetValue;
            private BigDecimal currentValue;
            private Boolean isCompleted;
            private LocalDateTime completedDate;
            
            // Getters and Setters
            public String getTitle() { return title; }
            public void setTitle(String title) { this.title = title; }
            public String getDescription() { return description; }
            public void setDescription(String description) { this.description = description; }
            public BigDecimal getTargetValue() { return targetValue; }
            public void setTargetValue(BigDecimal targetValue) { this.targetValue = targetValue; }
            public BigDecimal getCurrentValue() { return currentValue; }
            public void setCurrentValue(BigDecimal currentValue) { this.currentValue = currentValue; }
            public Boolean getIsCompleted() { return isCompleted; }
            public void setIsCompleted(Boolean isCompleted) { this.isCompleted = isCompleted; }
            public LocalDateTime getCompletedDate() { return completedDate; }
            public void setCompletedDate(LocalDateTime completedDate) { this.completedDate = completedDate; }
        }
    }

    /**
     * 学习目标设置请求DTO
     */
    public static class LearningGoalRequest {
        @NotNull(message = "目标标题不能为空")
        private String goalTitle;
        
        private String goalDescription;
        
        @NotNull(message = "目标分数不能为空")
        @DecimalMin(value = "0.0", message = "目标分数不能小于0")
        @DecimalMax(value = "100.0", message = "目标分数不能大于100")
        private BigDecimal targetScore;
        
        @NotNull(message = "目标日期不能为空")
        private LocalDateTime targetDate;
        
        private Long courseId;
        private String goalType; // COURSE, SEMESTER, YEAR
        
        // Getters and Setters
        public String getGoalTitle() { return goalTitle; }
        public void setGoalTitle(String goalTitle) { this.goalTitle = goalTitle; }
        public String getGoalDescription() { return goalDescription; }
        public void setGoalDescription(String goalDescription) { this.goalDescription = goalDescription; }
        public BigDecimal getTargetScore() { return targetScore; }
        public void setTargetScore(BigDecimal targetScore) { this.targetScore = targetScore; }
        public LocalDateTime getTargetDate() { return targetDate; }
        public void setTargetDate(LocalDateTime targetDate) { this.targetDate = targetDate; }
        public Long getCourseId() { return courseId; }
        public void setCourseId(Long courseId) { this.courseId = courseId; }
        public String getGoalType() { return goalType; }
        public void setGoalType(String goalType) { this.goalType = goalType; }
    }

    /**
     * 学习目标更新请求DTO
     */
    public static class LearningGoalUpdateRequest {
        @NotNull(message = "目标ID不能为空")
        private Long goalId;
        
        @NotNull(message = "目标标题不能为空")
        private String goalTitle;
        
        private String goalDescription;
        
        @NotNull(message = "目标分数不能为空")
        @DecimalMin(value = "0.0", message = "目标分数不能小于0")
        @DecimalMax(value = "100.0", message = "目标分数不能大于100")
        private BigDecimal targetScore;
        
        @NotNull(message = "目标日期不能为空")
        private LocalDateTime targetDate;
        
        private String status; // ACTIVE, COMPLETED, CANCELLED
        private String priority; // HIGH, MEDIUM, LOW
        
        // Getters and Setters
        public Long getGoalId() { return goalId; }
        public void setGoalId(Long goalId) { this.goalId = goalId; }
        public String getGoalTitle() { return goalTitle; }
        public void setGoalTitle(String goalTitle) { this.goalTitle = goalTitle; }
        public String getGoalDescription() { return goalDescription; }
        public void setGoalDescription(String goalDescription) { this.goalDescription = goalDescription; }
        public BigDecimal getTargetScore() { return targetScore; }
        public void setTargetScore(BigDecimal targetScore) { this.targetScore = targetScore; }
        public LocalDateTime getTargetDate() { return targetDate; }
        public void setTargetDate(LocalDateTime targetDate) { this.targetDate = targetDate; }
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
        public String getPriority() { return priority; }
        public void setPriority(String priority) { this.priority = priority; }
    }

    /**
     * 成绩证书响应DTO
     */
    public static class GradeCertificateResponse {
        private Long certificateId;
        private String certificateType;
        private String courseName;
        private String studentName;
        private BigDecimal finalScore;
        private String grade;
        private LocalDateTime issueDate;
        private String certificateUrl;
        private String verificationCode;
        private Boolean isValid;
        
        // Getters and Setters
        public Long getCertificateId() { return certificateId; }
        public void setCertificateId(Long certificateId) { this.certificateId = certificateId; }
        public String getCertificateType() { return certificateType; }
        public void setCertificateType(String certificateType) { this.certificateType = certificateType; }
        public String getCourseName() { return courseName; }
        public void setCourseName(String courseName) { this.courseName = courseName; }
        public String getStudentName() { return studentName; }
        public void setStudentName(String studentName) { this.studentName = studentName; }
        public BigDecimal getFinalScore() { return finalScore; }
        public void setFinalScore(BigDecimal finalScore) { this.finalScore = finalScore; }
        public String getGrade() { return grade; }
        public void setGrade(String grade) { this.grade = grade; }
        public LocalDateTime getIssueDate() { return issueDate; }
        public void setIssueDate(LocalDateTime issueDate) { this.issueDate = issueDate; }
        public String getCertificateUrl() { return certificateUrl; }
        public void setCertificateUrl(String certificateUrl) { this.certificateUrl = certificateUrl; }
        public String getVerificationCode() { return verificationCode; }
        public void setVerificationCode(String verificationCode) { this.verificationCode = verificationCode; }
        public Boolean getIsValid() { return isValid; }
        public void setIsValid(Boolean isValid) { this.isValid = isValid; }
        public void setStudentId(Long studentId) { /* Implementation needed */ }
        public void setCourseId(Long courseId) { /* Implementation needed */ }
    }

    /**
     * 证书申请请求DTO
     */
    public static class CertificateApplicationRequest {
        @NotNull(message = "学生ID不能为空")
        private Long studentId;
        
        @NotNull(message = "课程ID不能为空")
        private Long courseId;
        
        @NotBlank(message = "证书类型不能为空")
        private String certificateType;
        
        private String applicationReason;
        private String contactInfo;
        
        // Getters and Setters
        public Long getStudentId() { return studentId; }
        public void setStudentId(Long studentId) { this.studentId = studentId; }
        public Long getCourseId() { return courseId; }
        public void setCourseId(Long courseId) { this.courseId = courseId; }
        public String getCertificateType() { return certificateType; }
        public void setCertificateType(String certificateType) { this.certificateType = certificateType; }
        public String getApplicationReason() { return applicationReason; }
        public void setApplicationReason(String applicationReason) { this.applicationReason = applicationReason; }
        public String getContactInfo() { return contactInfo; }
        public void setContactInfo(String contactInfo) { this.contactInfo = contactInfo; }
    }

    /**
     * 成绩历史响应DTO
     */
    public static class GradeHistoryResponse {
        private List<GradeHistoryItem> historyItems;
        private Integer totalCount;
        private String timeRange;
        
        // Getters and Setters
        public List<GradeHistoryItem> getHistoryItems() { return historyItems; }
        public void setHistoryItems(List<GradeHistoryItem> historyItems) { this.historyItems = historyItems; }
        public Integer getTotalCount() { return totalCount; }
        public void setTotalCount(Integer totalCount) { this.totalCount = totalCount; }
        public String getTimeRange() { return timeRange; }
        public void setTimeRange(String timeRange) { this.timeRange = timeRange; }
        
        public static class GradeHistoryItem {
            private Long gradeId;
            private String courseName;
            private String taskName;
            private BigDecimal score;
            private String grade;
            private LocalDateTime gradeDate;
            private String gradeType;
            private String changeReason;
            
            // Getters and Setters
            public Long getGradeId() { return gradeId; }
            public void setGradeId(Long gradeId) { this.gradeId = gradeId; }
            public String getCourseName() { return courseName; }
            public void setCourseName(String courseName) { this.courseName = courseName; }
            public String getTaskName() { return taskName; }
            public void setTaskName(String taskName) { this.taskName = taskName; }
            public BigDecimal getScore() { return score; }
            public void setScore(BigDecimal score) { this.score = score; }
            public String getGrade() { return grade; }
            public void setGrade(String grade) { this.grade = grade; }
            public LocalDateTime getGradeDate() { return gradeDate; }
            public void setGradeDate(LocalDateTime gradeDate) { this.gradeDate = gradeDate; }
            public String getGradeType() { return gradeType; }
            public void setGradeType(String gradeType) { this.gradeType = gradeType; }
            public String getChangeReason() { return changeReason; }
            public void setChangeReason(String changeReason) { this.changeReason = changeReason; }
        }
    }

    /**
     * 详细反馈响应DTO
     */
    public static class DetailedFeedbackResponse {
        private Long feedbackId;
        private String taskName;
        private BigDecimal score;
        private String overallFeedback;
        private List<CriteriaFeedback> criteriaFeedbacks;
        private List<String> strengths;
        private List<String> improvements;
        private String teacherName;
        private LocalDateTime feedbackDate;
        
        // Getters and Setters
        public Long getFeedbackId() { return feedbackId; }
        public void setFeedbackId(Long feedbackId) { this.feedbackId = feedbackId; }
        public String getTaskName() { return taskName; }
        public void setTaskName(String taskName) { this.taskName = taskName; }
        public BigDecimal getScore() { return score; }
        public void setScore(BigDecimal score) { this.score = score; }
        public String getOverallFeedback() { return overallFeedback; }
        public void setOverallFeedback(String overallFeedback) { this.overallFeedback = overallFeedback; }
        public List<CriteriaFeedback> getCriteriaFeedbacks() { return criteriaFeedbacks; }
        public void setCriteriaFeedbacks(List<CriteriaFeedback> criteriaFeedbacks) { this.criteriaFeedbacks = criteriaFeedbacks; }
        public List<String> getStrengths() { return strengths; }
        public void setStrengths(List<String> strengths) { this.strengths = strengths; }
        public List<String> getImprovements() { return improvements; }
        public void setImprovements(List<String> improvements) { this.improvements = improvements; }
        public String getTeacherName() { return teacherName; }
        public void setTeacherName(String teacherName) { this.teacherName = teacherName; }
        public LocalDateTime getFeedbackDate() { return feedbackDate; }
        public void setFeedbackDate(LocalDateTime feedbackDate) { this.feedbackDate = feedbackDate; }
        public void setTaskId(Long taskId) { /* Implementation needed */ }
        public void setStudentId(Long studentId) { /* Implementation needed */ }
        public void setFeedback(String feedback) { this.overallFeedback = feedback; }
        public void setWeaknesses(List<String> weaknesses) { /* Implementation needed */ }
        public void setSuggestions(List<String> suggestions) { /* Implementation needed */ }
        
        public static class CriteriaFeedback {
            private String criteriaName;
            private BigDecimal score;
            private String feedback;
            private Integer weight;
            
            // Getters and Setters
            public String getCriteriaName() { return criteriaName; }
            public void setCriteriaName(String criteriaName) { this.criteriaName = criteriaName; }
            public BigDecimal getScore() { return score; }
            public void setScore(BigDecimal score) { this.score = score; }
            public String getFeedback() { return feedback; }
            public void setFeedback(String feedback) { this.feedback = feedback; }
            public Integer getWeight() { return weight; }
            public void setWeight(Integer weight) { this.weight = weight; }
        }
    }

    /**
     * 同伴评价响应DTO
     */
    public static class PeerEvaluationResponse {
        private List<PeerEvaluation> evaluations;
        private BigDecimal averageScore;
        private Integer totalEvaluators;
        private String summary;
        
        // Getters and Setters
        public List<PeerEvaluation> getEvaluations() { return evaluations; }
        public void setEvaluations(List<PeerEvaluation> evaluations) { this.evaluations = evaluations; }
        public BigDecimal getAverageScore() { return averageScore; }
        public void setAverageScore(BigDecimal averageScore) { this.averageScore = averageScore; }
        public Integer getTotalEvaluators() { return totalEvaluators; }
        public void setTotalEvaluators(Integer totalEvaluators) { this.totalEvaluators = totalEvaluators; }
        public String getSummary() { return summary; }
        public void setSummary(String summary) { this.summary = summary; }
        
        public static class PeerEvaluation {
            private String evaluatorName;
            private BigDecimal score;
            private String comment;
            private LocalDateTime evaluationDate;
            private List<DimensionScore> dimensionScores;
            
            // Getters and Setters
            public String getEvaluatorName() { return evaluatorName; }
            public void setEvaluatorName(String evaluatorName) { this.evaluatorName = evaluatorName; }
            public BigDecimal getScore() { return score; }
            public void setScore(BigDecimal score) { this.score = score; }
            public String getComment() { return comment; }
            public void setComment(String comment) { this.comment = comment; }
            public LocalDateTime getEvaluationDate() { return evaluationDate; }
            public void setEvaluationDate(LocalDateTime evaluationDate) { this.evaluationDate = evaluationDate; }
            public List<DimensionScore> getDimensionScores() { return dimensionScores; }
            public void setDimensionScores(List<DimensionScore> dimensionScores) { this.dimensionScores = dimensionScores; }
        }
        
        public static class DimensionScore {
            private String dimensionName;
            private BigDecimal score;
            private String comment;
            
            // Getters and Setters
            public String getDimensionName() { return dimensionName; }
            public void setDimensionName(String dimensionName) { this.dimensionName = dimensionName; }
            public BigDecimal getScore() { return score; }
            public void setScore(BigDecimal score) { this.score = score; }
            public String getComment() { return comment; }
            public void setComment(String comment) { this.comment = comment; }
        }
    }

    /**
     * 自我评价响应DTO
     */
    public static class SelfEvaluationResponse {
        private Long evaluationId;
        private String taskName;
        private BigDecimal selfScore;
        private String selfComment;
        private List<SelfDimensionScore> dimensionScores;
        private LocalDateTime evaluationDate;
        private String reflection;
        
        // Getters and Setters
        public Long getEvaluationId() { return evaluationId; }
        public void setEvaluationId(Long evaluationId) { this.evaluationId = evaluationId; }
        public String getTaskName() { return taskName; }
        public void setTaskName(String taskName) { this.taskName = taskName; }
        public BigDecimal getSelfScore() { return selfScore; }
        public void setSelfScore(BigDecimal selfScore) { this.selfScore = selfScore; }
        public String getSelfComment() { return selfComment; }
        public void setSelfComment(String selfComment) { this.selfComment = selfComment; }
        public List<SelfDimensionScore> getDimensionScores() { return dimensionScores; }
        public void setDimensionScores(List<SelfDimensionScore> dimensionScores) { this.dimensionScores = dimensionScores; }
        public LocalDateTime getEvaluationDate() { return evaluationDate; }
        public void setEvaluationDate(LocalDateTime evaluationDate) { this.evaluationDate = evaluationDate; }
        public String getReflection() { return reflection; }
        public void setReflection(String reflection) { this.reflection = reflection; }
        public void setTaskId(Long taskId) { /* Implementation needed */ }
        public void setStudentId(Long studentId) { /* Implementation needed */ }
        
        public static class SelfDimensionScore {
            private String dimensionName;
            private BigDecimal score;
            private String comment;
            
            // Getters and Setters
            public String getDimensionName() { return dimensionName; }
            public void setDimensionName(String dimensionName) { this.dimensionName = dimensionName; }
            public BigDecimal getScore() { return score; }
            public void setScore(BigDecimal score) { this.score = score; }
            public String getComment() { return comment; }
            public void setComment(String comment) { this.comment = comment; }
        }
    }

    /**
     * 自我评价请求DTO
     */
    public static class SelfEvaluationRequest {
        @NotNull(message = "任务ID不能为空")
        private Long taskId;
        
        @NotNull(message = "学生ID不能为空")
        private Long studentId;
        
        @NotNull(message = "自评分数不能为空")
        @DecimalMin(value = "0.0", message = "自评分数不能小于0")
        @DecimalMax(value = "100.0", message = "自评分数不能大于100")
        private BigDecimal selfScore;
        
        private String selfComment;
        private List<SelfDimensionScoreRequest> dimensionScores;
        private String reflection;
        
        // Getters and Setters
        public Long getTaskId() { return taskId; }
        public void setTaskId(Long taskId) { this.taskId = taskId; }
        public Long getStudentId() { return studentId; }
        public void setStudentId(Long studentId) { this.studentId = studentId; }
        public BigDecimal getSelfScore() { return selfScore; }
        public void setSelfScore(BigDecimal selfScore) { this.selfScore = selfScore; }
        public String getSelfComment() { return selfComment; }
        public void setSelfComment(String selfComment) { this.selfComment = selfComment; }
        public List<SelfDimensionScoreRequest> getDimensionScores() { return dimensionScores; }
        public void setDimensionScores(List<SelfDimensionScoreRequest> dimensionScores) { this.dimensionScores = dimensionScores; }
        public String getReflection() { return reflection; }
        public void setReflection(String reflection) { this.reflection = reflection; }
        
        public static class SelfDimensionScoreRequest {
            private String dimensionName;
            private BigDecimal score;
            private String comment;
            
            // Getters and Setters
            public String getDimensionName() { return dimensionName; }
            public void setDimensionName(String dimensionName) { this.dimensionName = dimensionName; }
            public BigDecimal getScore() { return score; }
            public void setScore(BigDecimal score) { this.score = score; }
            public String getComment() { return comment; }
            public void setComment(String comment) { this.comment = comment; }
        }
    }

    /**
     * 能力雷达图响应DTO
     */
    public static class AbilityRadarResponse {
        private List<AbilityDimension> dimensions;
        private BigDecimal overallScore;
        private String analysis;
        private List<String> recommendations;
        
        // Getters and Setters
        public List<AbilityDimension> getDimensions() { return dimensions; }
        public void setDimensions(List<AbilityDimension> dimensions) { this.dimensions = dimensions; }
        public BigDecimal getOverallScore() { return overallScore; }
        public void setOverallScore(BigDecimal overallScore) { this.overallScore = overallScore; }
        public String getAnalysis() { return analysis; }
        public void setAnalysis(String analysis) { this.analysis = analysis; }
        
        public void setStudentId(Long studentId) {
            // 可以添加studentId字段或使用现有逻辑
        }
        public List<String> getRecommendations() { return recommendations; }
        public void setRecommendations(List<String> recommendations) { this.recommendations = recommendations; }
        public void setCourseId(Long courseId) { /* Implementation needed */ }
        public void setAbilities(List<Object> abilities) { /* Implementation needed */ }
        
        public static class AbilityDimension {
            private String dimensionName;
            private BigDecimal score;
            private BigDecimal maxScore;
            private String level;
            private String description;
            
            // Getters and Setters
            public String getDimensionName() { return dimensionName; }
            public void setDimensionName(String dimensionName) { this.dimensionName = dimensionName; }
            public BigDecimal getScore() { return score; }
            public void setScore(BigDecimal score) { this.score = score; }
            public BigDecimal getMaxScore() { return maxScore; }
            public void setMaxScore(BigDecimal maxScore) { this.maxScore = maxScore; }
            public String getLevel() { return level; }
            public void setLevel(String level) { this.level = level; }
            public String getDescription() { return description; }
            public void setDescription(String description) { this.description = description; }
        }
    }

    /**
     * 学习效率响应DTO
     */
    public static class LearningEfficiencyResponse {
        private BigDecimal efficiencyScore;
        private String efficiencyLevel;
        private List<EfficiencyMetric> metrics;
        private List<String> suggestions;
        private EfficiencyTrend trend;
        
        // Getters and Setters
        public BigDecimal getEfficiencyScore() { return efficiencyScore; }
        public void setEfficiencyScore(BigDecimal efficiencyScore) { this.efficiencyScore = efficiencyScore; }
        public String getEfficiencyLevel() { return efficiencyLevel; }
        public void setEfficiencyLevel(String efficiencyLevel) { this.efficiencyLevel = efficiencyLevel; }
        public List<EfficiencyMetric> getMetrics() { return metrics; }
        public void setMetrics(List<EfficiencyMetric> metrics) { this.metrics = metrics; }
        public List<String> getSuggestions() { return suggestions; }
        public void setSuggestions(List<String> suggestions) { this.suggestions = suggestions; }
        public EfficiencyTrend getTrend() { return trend; }
        public void setTrend(EfficiencyTrend trend) { this.trend = trend; }
        public void setStudentId(Long studentId) { /* Implementation needed */ }
        public void setTimeRange(String timeRange) { /* Implementation needed */ }
        public void setAnalysis(String analysis) { /* Implementation needed */ }
        
        public static class EfficiencyMetric {
            private String metricName;
            private BigDecimal value;
            private String unit;
            private String description;
            
            // Getters and Setters
            public String getMetricName() { return metricName; }
            public void setMetricName(String metricName) { this.metricName = metricName; }
            public BigDecimal getValue() { return value; }
            public void setValue(BigDecimal value) { this.value = value; }
            public String getUnit() { return unit; }
            public void setUnit(String unit) { this.unit = unit; }
            public String getDescription() { return description; }
            public void setDescription(String description) { this.description = description; }
        }
        
        public static class EfficiencyTrend {
            private List<EfficiencyPoint> points;
            private String trendDirection;
            
            // Getters and Setters
            public List<EfficiencyPoint> getPoints() { return points; }
            public void setPoints(List<EfficiencyPoint> points) { this.points = points; }
            public String getTrendDirection() { return trendDirection; }
            public void setTrendDirection(String trendDirection) { this.trendDirection = trendDirection; }
            
            public static class EfficiencyPoint {
                private LocalDate date;
                private BigDecimal efficiency;
                
                // Getters and Setters
                public LocalDate getDate() { return date; }
                public void setDate(LocalDate date) { this.date = date; }
                public BigDecimal getEfficiency() { return efficiency; }
                public void setEfficiency(BigDecimal efficiency) { this.efficiency = efficiency; }
            }
        }
    }

    /**
     * 知识点掌握度响应DTO
     */
    public static class KnowledgePointMasteryResponse {
        private List<KnowledgePointMastery> masteryList;
        private BigDecimal overallMastery;
        private Integer totalPoints;
        private Integer masteredPoints;
        
        // Getters and Setters
        public List<KnowledgePointMastery> getMasteryList() { return masteryList; }
        public void setMasteryList(List<KnowledgePointMastery> masteryList) { this.masteryList = masteryList; }
        public BigDecimal getOverallMastery() { return overallMastery; }
        public void setOverallMastery(BigDecimal overallMastery) { this.overallMastery = overallMastery; }
        public Integer getTotalPoints() { return totalPoints; }
        public void setTotalPoints(Integer totalPoints) { this.totalPoints = totalPoints; }
        public Integer getMasteredPoints() { return masteredPoints; }
        public void setMasteredPoints(Integer masteredPoints) { this.masteredPoints = masteredPoints; }
        public void setStudentId(Long studentId) { /* Implementation needed */ }
        public void setCourseId(Long courseId) { /* Implementation needed */ }
        public void setKnowledgePoints(List<Object> knowledgePoints) { /* Implementation needed */ }
        
        public static class KnowledgePointMastery {
            private String knowledgePoint;
            private BigDecimal masteryLevel;
            private String masteryDescription;
            private Integer practiceCount;
            private BigDecimal averageScore;
            private LocalDateTime lastPractice;
            
            // Getters and Setters
            public String getKnowledgePoint() { return knowledgePoint; }
            public void setKnowledgePoint(String knowledgePoint) { this.knowledgePoint = knowledgePoint; }
            public BigDecimal getMasteryLevel() { return masteryLevel; }
            public void setMasteryLevel(BigDecimal masteryLevel) { this.masteryLevel = masteryLevel; }
            public String getMasteryDescription() { return masteryDescription; }
            public void setMasteryDescription(String masteryDescription) { this.masteryDescription = masteryDescription; }
            public Integer getPracticeCount() { return practiceCount; }
            public void setPracticeCount(Integer practiceCount) { this.practiceCount = practiceCount; }
            public BigDecimal getAverageScore() { return averageScore; }
            public void setAverageScore(BigDecimal averageScore) { this.averageScore = averageScore; }
            public LocalDateTime getLastPractice() { return lastPractice; }
            public void setLastPractice(LocalDateTime lastPractice) { this.lastPractice = lastPractice; }
        }
    }

    /**
     * 错题分析响应DTO
     */
    public static class WrongQuestionAnalysisResponse {
        private List<WrongQuestionItem> wrongQuestions;
        private List<ErrorPattern> errorPatterns;
        private List<String> suggestions;
        private WrongQuestionStatistics statistics;
        
        // Getters and Setters
        public List<WrongQuestionItem> getWrongQuestions() { return wrongQuestions; }
        public void setWrongQuestions(List<WrongQuestionItem> wrongQuestions) { this.wrongQuestions = wrongQuestions; }
        public List<ErrorPattern> getErrorPatterns() { return errorPatterns; }
        public void setErrorPatterns(List<ErrorPattern> errorPatterns) { this.errorPatterns = errorPatterns; }
        public List<String> getSuggestions() { return suggestions; }
        public void setSuggestions(List<String> suggestions) { this.suggestions = suggestions; }
        public WrongQuestionStatistics getStatistics() { return statistics; }
        public void setStatistics(WrongQuestionStatistics statistics) { this.statistics = statistics; }
        
        public static class WrongQuestionItem {
            private Long questionId;
            private String questionContent;
            private String correctAnswer;
            private String studentAnswer;
            private String knowledgePoint;
            private String errorType;
            private LocalDateTime wrongDate;
            private Integer wrongCount;
            
            // Getters and Setters
            public Long getQuestionId() { return questionId; }
            public void setQuestionId(Long questionId) { this.questionId = questionId; }
            public String getQuestionContent() { return questionContent; }
            public void setQuestionContent(String questionContent) { this.questionContent = questionContent; }
            public String getCorrectAnswer() { return correctAnswer; }
            public void setCorrectAnswer(String correctAnswer) { this.correctAnswer = correctAnswer; }
            public String getStudentAnswer() { return studentAnswer; }
            public void setStudentAnswer(String studentAnswer) { this.studentAnswer = studentAnswer; }
            public String getKnowledgePoint() { return knowledgePoint; }
            public void setKnowledgePoint(String knowledgePoint) { this.knowledgePoint = knowledgePoint; }
            public String getErrorType() { return errorType; }
            public void setErrorType(String errorType) { this.errorType = errorType; }
            public LocalDateTime getWrongDate() { return wrongDate; }
            public void setWrongDate(LocalDateTime wrongDate) { this.wrongDate = wrongDate; }
            public Integer getWrongCount() { return wrongCount; }
            public void setWrongCount(Integer wrongCount) { this.wrongCount = wrongCount; }
        }
        
        public static class ErrorPattern {
            private String patternName;
            private Integer frequency;
            private String description;
            private List<String> relatedKnowledgePoints;
            
            // Getters and Setters
            public String getPatternName() { return patternName; }
            public void setPatternName(String patternName) { this.patternName = patternName; }
            public Integer getFrequency() { return frequency; }
            public void setFrequency(Integer frequency) { this.frequency = frequency; }
            public String getDescription() { return description; }
            public void setDescription(String description) { this.description = description; }
            public List<String> getRelatedKnowledgePoints() { return relatedKnowledgePoints; }
            public void setRelatedKnowledgePoints(List<String> relatedKnowledgePoints) { this.relatedKnowledgePoints = relatedKnowledgePoints; }
        }
        
        public static class WrongQuestionStatistics {
            private Integer totalWrongQuestions;
            private BigDecimal wrongRate;
            private String mostFrequentErrorType;
            private String weakestKnowledgePoint;
            
            // Getters and Setters
            public Integer getTotalWrongQuestions() { return totalWrongQuestions; }
            public void setTotalWrongQuestions(Integer totalWrongQuestions) { this.totalWrongQuestions = totalWrongQuestions; }
            public BigDecimal getWrongRate() { return wrongRate; }
            public void setWrongRate(BigDecimal wrongRate) { this.wrongRate = wrongRate; }
            public String getMostFrequentErrorType() { return mostFrequentErrorType; }
            public void setMostFrequentErrorType(String mostFrequentErrorType) { this.mostFrequentErrorType = mostFrequentErrorType; }
            public String getWeakestKnowledgePoint() { return weakestKnowledgePoint; }
            public void setWeakestKnowledgePoint(String weakestKnowledgePoint) { this.weakestKnowledgePoint = weakestKnowledgePoint; }
        }
    }

    /**
     * 薄弱知识点响应DTO
     */
    public static class WeakKnowledgePointResponse {
        private List<WeakKnowledgePoint> weakPoints;
        private List<String> improvementPlan;
        private List<ResourceRecommendation> recommendations;
        
        // Getters and Setters
        public List<WeakKnowledgePoint> getWeakPoints() { return weakPoints; }
        public void setWeakPoints(List<WeakKnowledgePoint> weakPoints) { this.weakPoints = weakPoints; }
        public List<String> getImprovementPlan() { return improvementPlan; }
        public void setImprovementPlan(List<String> improvementPlan) { this.improvementPlan = improvementPlan; }
        public List<ResourceRecommendation> getRecommendations() { return recommendations; }
        public void setRecommendations(List<ResourceRecommendation> recommendations) { this.recommendations = recommendations; }
        
        public static class WeakKnowledgePoint {
            private String knowledgePoint;
            private BigDecimal masteryLevel;
            private Integer wrongCount;
            private BigDecimal averageScore;
            private String weakness;
            private String priority;
            
            // Getters and Setters
            public String getKnowledgePoint() { return knowledgePoint; }
            public void setKnowledgePoint(String knowledgePoint) { this.knowledgePoint = knowledgePoint; }
            public BigDecimal getMasteryLevel() { return masteryLevel; }
            public void setMasteryLevel(BigDecimal masteryLevel) { this.masteryLevel = masteryLevel; }
            public Integer getWrongCount() { return wrongCount; }
            public void setWrongCount(Integer wrongCount) { this.wrongCount = wrongCount; }
            public BigDecimal getAverageScore() { return averageScore; }
            public void setAverageScore(BigDecimal averageScore) { this.averageScore = averageScore; }
            public String getWeakness() { return weakness; }
            public void setWeakness(String weakness) { this.weakness = weakness; }
            public String getPriority() { return priority; }
            public void setPriority(String priority) { this.priority = priority; }
        }
        
        public static class ResourceRecommendation {
            private String resourceType;
            private String resourceName;
            private String resourceUrl;
            private String description;
            
            // Getters and Setters
            public String getResourceType() { return resourceType; }
            public void setResourceType(String resourceType) { this.resourceType = resourceType; }
            public String getResourceName() { return resourceName; }
            public void setResourceName(String resourceName) { this.resourceName = resourceName; }
            public String getResourceUrl() { return resourceUrl; }
            public void setResourceUrl(String resourceUrl) { this.resourceUrl = resourceUrl; }
            public String getDescription() { return description; }
            public void setDescription(String description) { this.description = description; }
        }
    }

    /**
     * 学习建议响应DTO
     */
    public static class StudySuggestionResponse {
        private List<StudySuggestion> suggestions;
        private String overallAdvice;
        private StudyPlan studyPlan;
        
        // Getters and Setters
        public List<StudySuggestion> getSuggestions() { return suggestions; }
        public void setSuggestions(List<StudySuggestion> suggestions) { this.suggestions = suggestions; }
        public String getOverallAdvice() { return overallAdvice; }
        public void setOverallAdvice(String overallAdvice) { this.overallAdvice = overallAdvice; }
        public StudyPlan getStudyPlan() { return studyPlan; }
        public void setStudyPlan(StudyPlan studyPlan) { this.studyPlan = studyPlan; }
        public void setStudentId(Long studentId) { /* Implementation needed */ }
        
        public static class StudySuggestion {
            private String category;
            private String suggestion;
            private String priority;
            private String expectedEffect;
            private Integer estimatedTime;
            
            // Getters and Setters
            public String getCategory() { return category; }
            public void setCategory(String category) { this.category = category; }
            public String getSuggestion() { return suggestion; }
            public void setSuggestion(String suggestion) { this.suggestion = suggestion; }
            public String getPriority() { return priority; }
            public void setPriority(String priority) { this.priority = priority; }
            public String getExpectedEffect() { return expectedEffect; }
            public void setExpectedEffect(String expectedEffect) { this.expectedEffect = expectedEffect; }
            public Integer getEstimatedTime() { return estimatedTime; }
            public void setEstimatedTime(Integer estimatedTime) { this.estimatedTime = estimatedTime; }
        }
        
        public static class StudyPlan {
            private List<StudyTask> tasks;
            private Integer totalDuration;
            private String goal;
            
            // Getters and Setters
            public List<StudyTask> getTasks() { return tasks; }
            public void setTasks(List<StudyTask> tasks) { this.tasks = tasks; }
            public Integer getTotalDuration() { return totalDuration; }
            public void setTotalDuration(Integer totalDuration) { this.totalDuration = totalDuration; }
            public String getGoal() { return goal; }
            public void setGoal(String goal) { this.goal = goal; }
            
            public static class StudyTask {
                private String taskName;
                private String description;
                private Integer duration;
                private String priority;
                private List<String> resources;
                
                // Getters and Setters
                public String getTaskName() { return taskName; }
                public void setTaskName(String taskName) { this.taskName = taskName; }
                public String getDescription() { return description; }
                public void setDescription(String description) { this.description = description; }
                public Integer getDuration() { return duration; }
                public void setDuration(Integer duration) { this.duration = duration; }
                public String getPriority() { return priority; }
                public void setPriority(String priority) { this.priority = priority; }
                public List<String> getResources() { return resources; }
                public void setResources(List<String> resources) { this.resources = resources; }
            }
        }
    }

    /**
     * 成绩报告响应
     */
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class GradeReportResponse {
        private String reportId;
        private String title;
        private String content;
        private LocalDate generateDate;
        private String reportType;
        private String status;
        
        public void setStudentId(Long studentId) { /* Implementation needed */ }
        public void setTimeRange(String timeRange) { /* Implementation needed */ }
        public void setGeneratedTime(LocalDateTime generatedTime) { /* Implementation needed */ }
    }

    /**
     * 导出响应
     */
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class ExportResponse {
        private String exportId;
        private String fileName;
        private String downloadUrl;
        private LocalDateTime exportTime;
        private String status;
        private Long fileSize;
        
        public void setStudentId(Long studentId) { /* Implementation needed */ }
        public void setExportFormat(String exportFormat) { /* Implementation needed */ }
    }

    /**
     * 成绩数据导出请求
     */
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class GradeDataExportRequest {
        @NotNull(message = "学生ID不能为空")
        private Long studentId;
        
        @NotBlank(message = "导出格式不能为空")
        private String format; // PDF, EXCEL, CSV
        
        private LocalDate startDate;
        private LocalDate endDate;
        private List<String> includeFields;
    }

    /**
     * 成绩通知响应
     */
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class GradeNotificationResponse {
        private String notificationId;
        private String title;
        private String message;
        private LocalDateTime sendTime;
        private String type;
        private Boolean isRead;
        private String priority;
    }

    /**
     * 成绩申诉响应
     */
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class GradeAppealResponse {
        private String appealId;
        private String status;
        private String result;
        private String feedback;
        private LocalDateTime processTime;
        private String processorName;
    }

    /**
     * 成绩申诉请求
     */
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class GradeAppealRequest {
        @NotNull(message = "成绩ID不能为空")
        private Long gradeId;
        
        @NotBlank(message = "申诉原因不能为空")
        private String reason;
        
        private String description;
        private List<String> evidenceUrls;
    }

    /**
     * 申诉结果响应
     */
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class AppealResultResponse {
        private String appealId;
        private String originalGrade;
        private String finalGrade;
        private String status;
        private String result;
        private String feedback;
        private LocalDateTime submitTime;
        private LocalDateTime processTime;
        private String processorName;
        private Long studentId;
        
        // Getters and Setters
        public String getAppealId() { return appealId; }
        public void setAppealId(String appealId) { this.appealId = appealId; }
        public String getOriginalGrade() { return originalGrade; }
        public void setOriginalGrade(String originalGrade) { this.originalGrade = originalGrade; }
        public String getFinalGrade() { return finalGrade; }
        public void setFinalGrade(String finalGrade) { this.finalGrade = finalGrade; }
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
        public String getResult() { return result; }
        public void setResult(String result) { this.result = result; }
        public String getFeedback() { return feedback; }
        public void setFeedback(String feedback) { this.feedback = feedback; }
        public LocalDateTime getSubmitTime() { return submitTime; }
        public void setSubmitTime(LocalDateTime submitTime) { this.submitTime = submitTime; }
        public LocalDateTime getProcessTime() { return processTime; }
        public void setProcessTime(LocalDateTime processTime) { this.processTime = processTime; }
        public String getProcessorName() { return processorName; }
        public void setProcessorName(String processorName) { this.processorName = processorName; }
        public Long getStudentId() { return studentId; }
        public void setStudentId(Long studentId) { this.studentId = studentId; }
    }

    /**
     * 批量成绩响应DTO
     */
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class BatchGradeResponse {
        private Integer totalCount;
        private Integer successCount;
        private Integer failureCount;
        private List<GradeResponse> successGrades;
        private List<GradeError> errors;
        
        @Data
        @NoArgsConstructor
        @AllArgsConstructor
        public static class GradeError {
            private Long studentId;
            private Long taskId;
            private String errorMessage;
        }
    }

    /**
     * 学生成绩详情响应DTO
     */
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class StudentGradeDetailResponse {
        private Long studentId;
        private String studentName;
        private String studentNumber;
        private Long courseId;
        private String courseName;
        private BigDecimal totalScore;
        private String finalGrade;
        private List<TaskGradeDetail> taskGrades;
        private Double completionRate;
        private Integer rank;
        private LocalDateTime lastUpdateTime;
        
        @Data
        @NoArgsConstructor
        @AllArgsConstructor
        public static class TaskGradeDetail {
            private Long taskId;
            private String taskTitle;
            private String taskType;
            private BigDecimal score;
            private BigDecimal maxScore;
            private String grade;
            private String feedback;
            private LocalDateTime gradeTime;
        }
    }

    /**
     * 成绩导出请求DTO
     */
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class GradeExportRequest {
        private Long courseId;
        private List<Long> taskIds;
        private List<Long> studentIds;
        private String exportFormat; // EXCEL, CSV, PDF
        private Boolean includeStatistics;
        private String dateRange;
    }

    /**
     * 成绩导入响应DTO
     */
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class GradeImportResponse {
        private Integer totalCount;
        private Integer successCount;
        private Integer failureCount;
        private List<ImportError> errors;
        private String importId;
        private LocalDateTime importTime;
        
        @Data
        @NoArgsConstructor
        @AllArgsConstructor
        public static class ImportError {
            private Integer rowNumber;
            private String studentNumber;
            private String errorMessage;
        }
    }

    /**
     * 成绩导入请求DTO
     */
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class GradeImportRequest {
        private Long courseId;
        private Long taskId;
        private String fileContent;
        private String fileFormat; // EXCEL, CSV
        private Boolean overwriteExisting;
    }

    /**
     * 成绩排名响应DTO
     */
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class GradeRankingResponse {
        private Long courseId;
        private String courseName;
        private Long taskId;
        private String taskTitle;
        private Integer totalStudents;
        private List<StudentRanking> rankings;
        private RankingStatistics statistics;
        
        @Data
        @NoArgsConstructor
        @AllArgsConstructor
        public static class StudentRanking {
            private Integer rank;
            private Long studentId;
            private String studentName;
            private String studentNumber;
            private BigDecimal score;
            private String grade;
        }
        
        @Data
        @NoArgsConstructor
        @AllArgsConstructor
        public static class RankingStatistics {
            private BigDecimal averageScore;
            private BigDecimal medianScore;
            private BigDecimal topScore;
            private BigDecimal bottomScore;
            private Double standardDeviation;
        }
    }

    /**
     * 成绩权重请求DTO
     */
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class GradeWeightRequest {
        private Long courseId;
        private List<WeightConfig> weights;
        
        @Data
        @NoArgsConstructor
        @AllArgsConstructor
        public static class WeightConfig {
            private String taskType;
            private Double weight;
            private String description;
        }
    }

    /**
     * 成绩权重响应DTO
     */
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class GradeWeightResponse {
        private Long courseId;
        private String courseName;
        private List<WeightInfo> weights;
        private LocalDateTime lastUpdateTime;
        
        @Data
        @NoArgsConstructor
        @AllArgsConstructor
        public static class WeightInfo {
            private String taskType;
            private Double weight;
            private String description;
            private Integer taskCount;
        }
    }

    /**
     * 总成绩响应DTO
     */
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class TotalGradeResponse {
        private Long studentId;
        private String studentName;
        private String studentNumber;
        private Long courseId;
        private String courseName;
        private BigDecimal totalScore;
        private String finalGrade;
        private List<ComponentGrade> componentGrades;
        private Double completionRate;
        private Integer rank;
        private String status;
        
        @Data
        @NoArgsConstructor
        @AllArgsConstructor
        public static class ComponentGrade {
            private String taskType;
            private BigDecimal score;
            private BigDecimal maxScore;
            private Double weight;
            private BigDecimal weightedScore;
        }
    }

    /**
     * 成绩分析响应DTO
     */
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class GradeAnalysisResponse {
        private Long courseId;
        private String courseName;
        private OverallStatistics overallStats;
        private List<TaskAnalysis> taskAnalyses;
        private List<StudentPerformance> topPerformers;
        private List<StudentPerformance> needsImprovement;
        private GradeTrend trend;
        
        @Data
        @NoArgsConstructor
        @AllArgsConstructor
        public static class OverallStatistics {
            private Integer totalStudents;
            private BigDecimal averageScore;
            private BigDecimal medianScore;
            private Double passRate;
            private Double excellentRate;
        }
        
        @Data
        @NoArgsConstructor
        @AllArgsConstructor
        public static class TaskAnalysis {
            private Long taskId;
            private String taskTitle;
            private String taskType;
            private BigDecimal averageScore;
            private Double completionRate;
            private String difficulty;
        }
        
        @Data
        @NoArgsConstructor
        @AllArgsConstructor
        public static class StudentPerformance {
            private Long studentId;
            private String studentName;
            private String studentNumber;
            private BigDecimal averageScore;
            private String grade;
            private Integer rank;
        }
        
        @Data
        @NoArgsConstructor
        @AllArgsConstructor
        public static class GradeTrend {
            private String direction; // IMPROVING, DECLINING, STABLE
            private Double changeRate;
            private String analysis;
        }
    }

    /**
     * 成绩评论请求DTO
     */
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class GradeCommentRequest {
        @NotNull(message = "成绩ID不能为空")
        private Long gradeId;
        
        @NotBlank(message = "评论内容不能为空")
        private String comment;
        
        private String commentType; // TEACHER, PEER, SELF
        private Boolean isPublic;
        private List<String> tags;
    }

    /**
     * 成绩报告请求DTO
     */
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class GradeReportRequest {
        @NotNull(message = "课程ID不能为空")
        private Long courseId;
        
        private List<Long> studentIds;
        private String reportType; // INDIVIDUAL, CLASS, COURSE
        private LocalDate startDate;
        private LocalDate endDate;
        private List<String> includeComponents; // HOMEWORK, EXAM, PROJECT, etc.
        private String format; // PDF, EXCEL, JSON
        private Boolean includeAnalysis;
        private Boolean includeCharts;
    }
}