package com.education.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

import java.util.Date;

/**
 * 作业提交记录实体类
 */
@Data
@TableName("assignment_submission")
public class AssignmentSubmission {
    
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
     * 学生ID
     */
    private Long studentId;
    
    /**
     * 状态：0-未提交，1-已提交未批改，2-已批改
     */
    private Integer status;
    
    /**
     * 得分
     */
    private Integer score;
    
    /**
     * 教师反馈
     */
    private String feedback;
    
    /**
     * 提交时间
     */
    private Date submitTime;
    
    /**
     * 批改时间
     */
    private Date gradeTime;
    
    /**
     * 批改人ID
     */
    private Long gradedBy;
    
    /**
     * 提交内容
     */
    private String content;
    
    /**
     * 创建时间
     */
    private Date createTime;
    
    /**
     * 更新时间
     */
    private Date updateTime;
    
    /**
     * 文件名称
     */
    private String fileName;
    
    /**
     * 文件路径
     */
    private String filePath;
} 