package com.education.entity;

import com.baomidou.mybatisplus.annotation.*;
import lombok.Data;
import lombok.EqualsAndHashCode;

import java.time.LocalDateTime;

/**
 * 题库实体类
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
@Data
@EqualsAndHashCode(callSuper = false)
@TableName("question_bank")
public class QuestionBank {
    
    /**
     * 题目ID
     */
    @TableId(value = "id", type = IdType.AUTO)
    private Long id;
    
    /**
     * 题目标题
     */
    @TableField("title")
    private String title;
    
    /**
     * 题目内容
     */
    @TableField("content")
    private String content;
    
    /**
     * 题目类型：SINGLE_CHOICE-单选，MULTIPLE_CHOICE-多选，TRUE_FALSE-判断，FILL_BLANK-填空，ESSAY-问答，CODING-编程
     */
    @TableField("question_type")
    private String questionType;
    
    /**
     * 选项（JSON格式）
     */
    @TableField("options")
    private String options;
    
    /**
     * 正确答案
     */
    @TableField("correct_answer")
    private String correctAnswer;
    
    /**
     * 答案解析
     */
    @TableField("explanation")
    private String explanation;
    
    /**
     * 难度等级：1-简单，2-中等，3-困难
     */
    @TableField("difficulty")
    private Integer difficulty;
    
    /**
     * 分值
     */
    @TableField("score")
    private Integer score;
    
    /**
     * 课程ID
     */
    @TableField("course_id")
    private Long courseId;
    
    /**
     * 知识点ID
     */
    @TableField("knowledge_point_id")
    private Long knowledgePointId;
    
    /**
     * 创建者ID
     */
    @TableField("created_by")
    private Long createdBy;
    
    /**
     * 题目状态：DRAFT-草稿，PUBLISHED-已发布，ARCHIVED-已归档
     */
    @TableField("status")
    private String status;
    
    /**
     * 是否公开
     */
    @TableField("is_public")
    private Boolean isPublic;
    
    /**
     * 使用次数
     */
    @TableField("usage_count")
    private Integer usageCount;
    
    /**
     * 正确率
     */
    @TableField("accuracy_rate")
    private Double accuracyRate;
    
    /**
     * 标签
     */
    @TableField("tags")
    private String tags;
    
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
     * 是否删除
     */
    @TableLogic
    @TableField("is_deleted")
    private Boolean isDeleted;
}