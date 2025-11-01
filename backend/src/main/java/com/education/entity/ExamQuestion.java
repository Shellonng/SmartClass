package com.education.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

/**
 * 考试题目关联实体类
 */
@Data
@TableName("assignment_question")
public class ExamQuestion {
    
    @TableId(value = "id", type = IdType.AUTO)
    private Long id;
    
    /**
     * 考试ID，对应assignment表的id
     */
    private Long assignmentId;
    
    /**
     * 题目ID，对应question表的id
     */
    private Long questionId;
    
    /**
     * 该题满分
     */
    private Integer score;
    
    /**
     * 题目顺序
     */
    private Integer sequence;
} 