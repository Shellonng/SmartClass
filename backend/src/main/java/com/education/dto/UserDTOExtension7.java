package com.education.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import java.time.LocalDateTime;
import java.util.List;
import java.math.BigDecimal;

/**
 * 用户DTO扩展类 - 第7部分
 * 包含用户关注、学习报告、成就、积分等相关DTO
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
public class UserDTOExtension7 {

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
        private Boolean isOnline;
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
        public Boolean getIsOnline() { return isOnline; }
        public void setIsOnline(Boolean isOnline) { this.isOnline = isOnline; }
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
        private LocalDateTime generatedTime;
        private StudyStatistics studyStatistics;
        private List<CourseProgress> courseProgress;
        private List<PerformanceMetric> performanceMetrics;
        private List<String> recommendations;
        
        // Getters and Setters
        public String getReportType() { return reportType; }
        public void setReportType(String reportType) { this.reportType = reportType; }
        public String getTimeRange() { return timeRange; }
        public void setTimeRange(String timeRange) { this.timeRange = timeRange; }
        public LocalDateTime getGeneratedTime() { return generatedTime; }
        public void setGeneratedTime(LocalDateTime generatedTime) { this.generatedTime = generatedTime; }
        public StudyStatistics getStudyStatistics() { return studyStatistics; }
        public void setStudyStatistics(StudyStatistics studyStatistics) { this.studyStatistics = studyStatistics; }
        public List<CourseProgress> getCourseProgress() { return courseProgress; }
        public void setCourseProgress(List<CourseProgress> courseProgress) { this.courseProgress = courseProgress; }
        public List<PerformanceMetric> getPerformanceMetrics() { return performanceMetrics; }
        public void setPerformanceMetrics(List<PerformanceMetric> performanceMetrics) { this.performanceMetrics = performanceMetrics; }
        public List<String> getRecommendations() { return recommendations; }
        public void setRecommendations(List<String> recommendations) { this.recommendations = recommendations; }
        
        public static class StudyStatistics {
            private Integer totalHours;
            private Integer totalDays;
            private BigDecimal averageDaily;
            private Integer longestStreak;
            
            // Getters and Setters
            public Integer getTotalHours() { return totalHours; }
            public void setTotalHours(Integer totalHours) { this.totalHours = totalHours; }
            public Integer getTotalDays() { return totalDays; }
            public void setTotalDays(Integer totalDays) { this.totalDays = totalDays; }
            public BigDecimal getAverageDaily() { return averageDaily; }
            public void setAverageDaily(BigDecimal averageDaily) { this.averageDaily = averageDaily; }
            public Integer getLongestStreak() { return longestStreak; }
            public void setLongestStreak(Integer longestStreak) { this.longestStreak = longestStreak; }
        }
        
        public static class CourseProgress {
            private Long courseId;
            private String courseName;
            private BigDecimal progress;
            private BigDecimal grade;
            private Integer timeSpent;
            
            // Getters and Setters
            public Long getCourseId() { return courseId; }
            public void setCourseId(Long courseId) { this.courseId = courseId; }
            public String getCourseName() { return courseName; }
            public void setCourseName(String courseName) { this.courseName = courseName; }
            public BigDecimal getProgress() { return progress; }
            public void setProgress(BigDecimal progress) { this.progress = progress; }
            public BigDecimal getGrade() { return grade; }
            public void setGrade(BigDecimal grade) { this.grade = grade; }
            public Integer getTimeSpent() { return timeSpent; }
            public void setTimeSpent(Integer timeSpent) { this.timeSpent = timeSpent; }
        }
        
        public static class PerformanceMetric {
            private String metricName;
            private String value;
            private String trend;
            
            // Getters and Setters
            public String getMetricName() { return metricName; }
            public void setMetricName(String metricName) { this.metricName = metricName; }
            public String getValue() { return value; }
            public void setValue(String value) { this.value = value; }
            public String getTrend() { return trend; }
            public void setTrend(String trend) { this.trend = trend; }
        }
    }

    /**
     * 成就响应DTO
     */
    public static class AchievementResponse {
        private Long achievementId;
        private String name;
        private String description;
        private String icon;
        private String category;
        private String difficulty;
        private Integer points;
        private LocalDateTime unlockedTime;
        private BigDecimal progress;
        private Boolean isUnlocked;
        
        // Getters and Setters
        public Long getAchievementId() { return achievementId; }
        public void setAchievementId(Long achievementId) { this.achievementId = achievementId; }
        public String getName() { return name; }
        public void setName(String name) { this.name = name; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
        public String getIcon() { return icon; }
        public void setIcon(String icon) { this.icon = icon; }
        public String getCategory() { return category; }
        public void setCategory(String category) { this.category = category; }
        public String getDifficulty() { return difficulty; }
        public void setDifficulty(String difficulty) { this.difficulty = difficulty; }
        public Integer getPoints() { return points; }
        public void setPoints(Integer points) { this.points = points; }
        public LocalDateTime getUnlockedTime() { return unlockedTime; }
        public void setUnlockedTime(LocalDateTime unlockedTime) { this.unlockedTime = unlockedTime; }
        public BigDecimal getProgress() { return progress; }
        public void setProgress(BigDecimal progress) { this.progress = progress; }
        public Boolean getIsUnlocked() { return isUnlocked; }
        public void setIsUnlocked(Boolean isUnlocked) { this.isUnlocked = isUnlocked; }
    }

    /**
     * 用户积分响应DTO
     */
    public static class UserPointsResponse {
        private Integer totalPoints;
        private Integer availablePoints;
        private Integer usedPoints;
        private Integer currentLevel;
        private Integer pointsToNextLevel;
        private String levelName;
        private List<PointSource> recentEarnings;
        
        // Getters and Setters
        public Integer getTotalPoints() { return totalPoints; }
        public void setTotalPoints(Integer totalPoints) { this.totalPoints = totalPoints; }
        public Integer getAvailablePoints() { return availablePoints; }
        public void setAvailablePoints(Integer availablePoints) { this.availablePoints = availablePoints; }
        public Integer getUsedPoints() { return usedPoints; }
        public void setUsedPoints(Integer usedPoints) { this.usedPoints = usedPoints; }
        public Integer getCurrentLevel() { return currentLevel; }
        public void setCurrentLevel(Integer currentLevel) { this.currentLevel = currentLevel; }
        public Integer getPointsToNextLevel() { return pointsToNextLevel; }
        public void setPointsToNextLevel(Integer pointsToNextLevel) { this.pointsToNextLevel = pointsToNextLevel; }
        public String getLevelName() { return levelName; }
        public void setLevelName(String levelName) { this.levelName = levelName; }
        public List<PointSource> getRecentEarnings() { return recentEarnings; }
        public void setRecentEarnings(List<PointSource> recentEarnings) { this.recentEarnings = recentEarnings; }
        
        public static class PointSource {
            private String source;
            private Integer points;
            private LocalDateTime earnedTime;
            
            // Getters and Setters
            public String getSource() { return source; }
            public void setSource(String source) { this.source = source; }
            public Integer getPoints() { return points; }
            public void setPoints(Integer points) { this.points = points; }
            public LocalDateTime getEarnedTime() { return earnedTime; }
            public void setEarnedTime(LocalDateTime earnedTime) { this.earnedTime = earnedTime; }
        }
    }

    /**
     * 积分历史响应DTO
     */
    public static class PointsHistoryResponse {
        private Long historyId;
        private String action;
        private Integer points;
        private String description;
        private String source;
        private LocalDateTime timestamp;
        private Integer balanceAfter;
        
        // Getters and Setters
        public Long getHistoryId() { return historyId; }
        public void setHistoryId(Long historyId) { this.historyId = historyId; }
        public String getAction() { return action; }
        public void setAction(String action) { this.action = action; }
        public Integer getPoints() { return points; }
        public void setPoints(Integer points) { this.points = points; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
        public String getSource() { return source; }
        public void setSource(String source) { this.source = source; }
        public LocalDateTime getTimestamp() { return timestamp; }
        public void setTimestamp(LocalDateTime timestamp) { this.timestamp = timestamp; }
        public Integer getBalanceAfter() { return balanceAfter; }
        public void setBalanceAfter(Integer balanceAfter) { this.balanceAfter = balanceAfter; }
    }

    /**
     * 用户等级响应DTO
     */
    public static class UserLevelResponse {
        private Integer currentLevel;
        private String levelName;
        private String levelDescription;
        private Integer currentPoints;
        private Integer pointsRequired;
        private Integer pointsToNext;
        private BigDecimal progress;
        private List<String> privileges;
        private String badge;
        
        // Getters and Setters
        public Integer getCurrentLevel() { return currentLevel; }
        public void setCurrentLevel(Integer currentLevel) { this.currentLevel = currentLevel; }
        public String getLevelName() { return levelName; }
        public void setLevelName(String levelName) { this.levelName = levelName; }
        public String getLevelDescription() { return levelDescription; }
        public void setLevelDescription(String levelDescription) { this.levelDescription = levelDescription; }
        public Integer getCurrentPoints() { return currentPoints; }
        public void setCurrentPoints(Integer currentPoints) { this.currentPoints = currentPoints; }
        public Integer getPointsRequired() { return pointsRequired; }
        public void setPointsRequired(Integer pointsRequired) { this.pointsRequired = pointsRequired; }
        public Integer getPointsToNext() { return pointsToNext; }
        public void setPointsToNext(Integer pointsToNext) { this.pointsToNext = pointsToNext; }
        public BigDecimal getProgress() { return progress; }
        public void setProgress(BigDecimal progress) { this.progress = progress; }
        public List<String> getPrivileges() { return privileges; }
        public void setPrivileges(List<String> privileges) { this.privileges = privileges; }
        public String getBadge() { return badge; }
        public void setBadge(String badge) { this.badge = badge; }
    }

    /**
     * 徽章响应DTO
     */
    public static class BadgeResponse {
        private Long badgeId;
        private String name;
        private String description;
        private String icon;
        private String category;
        private String rarity;
        private LocalDateTime earnedTime;
        private String condition;
        
        // Getters and Setters
        public Long getBadgeId() { return badgeId; }
        public void setBadgeId(Long badgeId) { this.badgeId = badgeId; }
        public String getName() { return name; }
        public void setName(String name) { this.name = name; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
        public String getIcon() { return icon; }
        public void setIcon(String icon) { this.icon = icon; }
        public String getCategory() { return category; }
        public void setCategory(String category) { this.category = category; }
        public String getRarity() { return rarity; }
        public void setRarity(String rarity) { this.rarity = rarity; }
        public LocalDateTime getEarnedTime() { return earnedTime; }
        public void setEarnedTime(LocalDateTime earnedTime) { this.earnedTime = earnedTime; }
        public String getCondition() { return condition; }
        public void setCondition(String condition) { this.condition = condition; }
    }
}