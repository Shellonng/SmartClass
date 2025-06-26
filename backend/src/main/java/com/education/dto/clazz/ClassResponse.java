package com.education.dto.clazz;

import lombok.Data;
import java.time.LocalDateTime;

/**
 * 班级响应DTO
 */
@Data
public class ClassResponse {
    
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
} 