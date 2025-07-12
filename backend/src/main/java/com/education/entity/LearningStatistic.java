package com.education.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

import java.time.LocalDate;
import java.time.LocalDateTime;

@Data
@TableName("learning_statistics")
public class LearningStatistic {
    @TableId(value = "id", type = IdType.AUTO)
    private Long id;
    private Long studentId;
    private Long courseId;
    private LocalDate date;
    private Integer totalDuration;
    private Integer sectionsCompleted;
    private Integer resourcesViewed;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
} 