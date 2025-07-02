package com.education.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

/**
 * 作业题目关联实体类
 */
@Data
@TableName("assignment_question")
public class AssignmentQuestion {
    
    /**
     * 主键ID
     */
    @TableId(type = IdType.AUTO)
    private Long id;
    
    /**
     * 作业ID
     */
    private Long assignmentId;
    
    /**
     * 题目ID
     */
    private Long questionId;
    
    /**
     * 题目分值
     */
    private Integer score;
    
    /**
     * 题目顺序
     */
    private Integer sequence;
} 