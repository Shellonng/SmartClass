package com.education.dto;

import lombok.Data;

import java.time.LocalDateTime;

@Data
public class LearningRecordDTO {
    private Long id;
    private Long studentId;
    private Long courseId;
    private Long sectionId;
    private Long resourceId;
    private String resourceType;
    private LocalDateTime startTime;
    private LocalDateTime endTime;
    private Integer duration;
    private Integer progress;
    private Boolean completed;
    private String deviceInfo;
    private String ipAddress;
    
    // 额外字段，用于前端展示
    private String courseName;
    private String sectionTitle;
    private String resourceName;
} 