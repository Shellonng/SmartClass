package com.education.dto.clazz;

import lombok.Data;
import java.time.LocalDateTime;
import java.util.List;

/**
 * 班级详情响应DTO
 */
@Data
public class ClassDetailResponse {
    
    private Long id;
    private String name;
    private String grade;
    private String major;
    private String description;
    private Integer capacity;
    private Integer studentCount;
    private String semester;
    private String status;
    private String teacherName;
    private LocalDateTime createTime;
    private LocalDateTime updateTime;
    
    // 班级统计信息
    private ClassStatistics statistics;
    
    // 最近活动
    private List<RecentActivity> recentActivities;
    
    @Data
    public static class ClassStatistics {
        private Integer totalStudents;
        private Integer activeStudents;
        private Integer totalCourses;
        private Integer totalAssignments;
        private Double averageScore;
        private Integer completedAssignments;
    }
    
    @Data
    public static class RecentActivity {
        private String type;
        private String description;
        private LocalDateTime time;
        private String studentName;
    }
} 