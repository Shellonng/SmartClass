package com.education.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

import java.time.LocalDateTime;

@Data
@TableName("learning_records")
public class LearningRecord {
    @TableId(value = "id", type = IdType.AUTO)
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
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
} 