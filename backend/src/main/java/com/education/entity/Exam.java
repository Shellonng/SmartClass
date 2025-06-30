package com.education.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

import java.time.LocalDateTime;

/**
 * 考试实体类，对应assignment表中type为exam的数据
 */
@Data
@TableName("assignment")
public class Exam {
    
    @TableId(value = "id", type = IdType.AUTO)
    private Long id;
    
    /**
     * 考试标题
     */
    private String title;
    
    /**
     * 所属课程ID
     */
    private Long courseId;
    
    /**
     * 发布考试的用户ID（教师）
     */
    private Long userId;
    
    /**
     * 类型：固定为exam
     */
    private String type = "exam";
    
    /**
     * 考试说明
     */
    private String description;
    
    /**
     * 考试开始时间
     */
    private LocalDateTime startTime;
    
    /**
     * 考试结束时间
     */
    private LocalDateTime endTime;
    
    /**
     * 考试时长（分钟）
     */
    @TableField(exist = false)
    private Integer duration;
    
    /**
     * 考试总分
     */
    @TableField(exist = false)
    private Integer totalScore;
    
    /**
     * 发布状态：0 未发布，1 已发布
     */
    private Integer status;
    
    /**
     * 创建时间
     */
    private LocalDateTime createTime;
    
    /**
     * 更新时间
     */
    private LocalDateTime updateTime;
} 