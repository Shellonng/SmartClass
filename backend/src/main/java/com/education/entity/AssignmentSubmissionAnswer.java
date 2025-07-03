package com.education.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

import java.util.Date;

/**
 * 作业题目答案实体类
 */
@Data
@TableName("assignment_submission_answer")
public class AssignmentSubmissionAnswer {
    
    /**
     * 主键ID
     */
    @TableId(type = IdType.AUTO)
    private Long id;
    
    /**
     * 提交记录ID
     */
    private Long submissionId;
    
    /**
     * 题目ID
     */
    private Long questionId;
    
    /**
     * 学生答案
     */
    private String studentAnswer;
    
    /**
     * 是否正确
     */
    private Boolean isCorrect;
    
    /**
     * 得分
     */
    private Integer score;
    
    /**
     * 批阅评论
     */
    private String comment;
    
    /**
     * 创建时间
     */
    private Date createTime;
    
    /**
     * 更新时间
     */
    private Date updateTime;
} 