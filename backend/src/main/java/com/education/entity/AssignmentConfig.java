package com.education.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

import java.time.LocalDateTime;

/**
 * 作业配置实体类
 */
@Data
@TableName("assignment_config")
public class AssignmentConfig {
    
    /**
     * 主键ID
     */
    @TableId(value = "id", type = IdType.AUTO)
    private Long id;
    
    /**
     * 作业ID
     */
    @TableField("assignment_id")
    private Long assignmentId;
    
    /**
     * 知识点范围（JSON格式）
     */
    @TableField("knowledge_points")
    private String knowledgePoints;
    
    /**
     * 难度级别：EASY-简单，MEDIUM-中等，HARD-困难
     */
    @TableField("difficulty")
    private String difficulty;
    
    /**
     * 题目总数
     */
    @TableField("question_count")
    private Integer questionCount;
    
    /**
     * 题型分布（JSON格式）
     */
    @TableField("question_types")
    private String questionTypes;
    
    /**
     * 额外要求
     */
    @TableField("additional_requirements")
    private String additionalRequirements;
    
    /**
     * 创建时间
     */
    @TableField("create_time")
    private LocalDateTime createTime;
    
    /**
     * 更新时间
     */
    @TableField("update_time")
    private LocalDateTime updateTime;
}