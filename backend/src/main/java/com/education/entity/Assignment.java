package com.education.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

import java.util.Date;

/**
 * 作业或考试实体类
 */
@Data
@TableName("assignment")
public class Assignment {
    /**
     * 主键ID
     */
    @TableId(value = "id", type = IdType.AUTO)
    private Long id;

    /**
     * 作业或考试标题
     */
    @TableField("title")
    private String title;

    /**
     * 所属课程ID
     */
    @TableField("course_id")
    private Long courseId;

    /**
     * 发布作业的用户ID
     */
    @TableField("user_id")
    private Long userId;

    /**
     * 类型：homework-作业，exam-考试
     */
    @TableField("type")
    private String type;

    /**
     * 描述
     */
    @TableField("description")
    private String description;

    /**
     * 开始时间
     */
    @TableField("start_time")
    private Date startTime;

    /**
     * 结束时间
     */
    @TableField("end_time")
    private Date endTime;

    /**
     * 创建时间
     */
    @TableField("create_time")
    private Date createTime;

    /**
     * 发布状态：0-未发布，1-已发布
     */
    @TableField("status")
    private Integer status;

    /**
     * 更新时间
     */
    @TableField("update_time")
    private Date updateTime;

    /**
     * 作业模式：question-答题型，file-上传型
     */
    @TableField("mode")
    private String mode;

    /**
     * 时间限制（分钟）
     */
    @TableField("time_limit")
    private Integer timeLimit;
} 