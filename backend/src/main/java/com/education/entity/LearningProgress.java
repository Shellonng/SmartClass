package com.education.entity;

import com.baomidou.mybatisplus.annotation.*;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.experimental.Accessors;

import java.io.Serializable;
import java.time.LocalDateTime;

/**
 * 学习进度实体类
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
@Data
@EqualsAndHashCode(callSuper = false)
@Accessors(chain = true)
@TableName("learning_progress")
public class LearningProgress implements Serializable {

    private static final long serialVersionUID = 1L;

    /**
     * 进度ID
     */
    @TableId(value = "id", type = IdType.AUTO)
    private Long id;

    /**
     * 学生ID
     */
    @TableField("student_id")
    private Long studentId;

    /**
     * 课程ID
     */
    @TableField("course_id")
    private Long courseId;

    /**
     * 章节ID
     */
    @TableField("chapter_id")
    private Long chapterId;

    /**
     * 学习状态：NOT_STARTED-未开始，IN_PROGRESS-学习中，COMPLETED-已完成
     */
    @TableField("status")
    private String status;

    /**
     * 学习进度百分比（0-100）
     */
    @TableField("progress_percentage")
    private Double progressPercentage;

    /**
     * 学习时长（分钟）
     */
    @TableField("study_duration")
    private Integer studyDuration;

    /**
     * 最后学习时间
     */
    @TableField("last_study_time")
    private LocalDateTime lastStudyTime;

    /**
     * 完成时间
     */
    @TableField("completed_time")
    private LocalDateTime completedTime;

    /**
     * 创建时间
     */
    @TableField(value = "create_time", fill = FieldFill.INSERT)
    private LocalDateTime createTime;

    /**
     * 更新时间
     */
    @TableField(value = "update_time", fill = FieldFill.INSERT_UPDATE)
    private LocalDateTime updateTime;
}