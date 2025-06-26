package com.education.dto.clazz;

import lombok.Data;
import java.time.LocalDateTime;

/**
 * 班级学生响应DTO
 */
@Data
public class ClassStudentResponse {
    
    private Long id;
    private String studentId;
    private String name;
    private String email;
    private String phone;
    private String gender;
    private LocalDateTime joinTime;
    private String status;
    
    // 学习统计
    private Integer completedAssignments;
    private Integer totalAssignments;
    private Double averageScore;
    private Integer attendance;
    private LocalDateTime lastActiveTime;
} 