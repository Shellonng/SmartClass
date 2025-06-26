package com.education.entity;

import com.baomidou.mybatisplus.annotation.*;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.experimental.Accessors;

import java.io.Serializable;
import java.time.LocalDateTime;

/**
 * 课程章节实体类
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
@Data
@EqualsAndHashCode(callSuper = false)
@Accessors(chain = true)
@TableName("chapter")
public class Chapter implements Serializable {

    private static final long serialVersionUID = 1L;

    /**
     * 章节ID
     */
    @TableId(value = "id", type = IdType.AUTO)
    private Long id;

    /**
     * 课程ID
     */
    @TableField("course_id")
    private Long courseId;

    /**
     * 章节标题
     */
    @TableField("title")
    private String title;

    /**
     * 章节描述
     */
    @TableField("description")
    private String description;

    /**
     * 章节内容
     */
    @TableField("content")
    private String content;

    /**
     * 排序号
     */
    @TableField("sort_order")
    private Integer sortOrder;

    /**
     * 章节状态：DRAFT-草稿，PUBLISHED-已发布
     */
    @TableField("status")
    private String status;

    /**
     * 是否必修：0-选修，1-必修
     */
    @TableField("is_required")
    private Boolean isRequired;

    /**
     * 预计学习时长（分钟）
     */
    @TableField("estimated_duration")
    private Integer estimatedDuration;

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

    /**
     * 是否删除：0-未删除，1-已删除
     */
    @TableField("is_deleted")
    @TableLogic
    private Boolean isDeleted;
}