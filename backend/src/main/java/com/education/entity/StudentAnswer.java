package com.education.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

import java.util.Date;

/**
 * 学生答题记录实体类
 */
@Data
@TableName("student_answer")
public class StudentAnswer {
    
    /**
     * 主键ID
     */
    @TableId(type = IdType.AUTO)
    private Long id;
    
    /**
     * 学生ID
     */
    private Long studentId;
    
    /**
     * 作业ID
     */
    private Long assignmentId;
    
    /**
     * 题目ID
     */
    private Long questionId;
    
    /**
     * 学生答案内容
     */
    private String answerContent;
    
    /**
     * 是否正确
     */
    private Boolean isCorrect;
    
    /**
     * 得分
     */
    private Integer score;
    
    /**
     * 答题时间
     */
    private Date answerTime;
} 