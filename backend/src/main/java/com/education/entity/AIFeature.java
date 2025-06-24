package com.education.entity;

import com.baomidou.mybatisplus.annotation.*;
import lombok.Data;
import lombok.EqualsAndHashCode;

import java.time.LocalDateTime;

/**
 * AI功能实体类
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
@Data
@EqualsAndHashCode(callSuper = false)
@TableName("ai_feature")
public class AIFeature {
    
    /**
     * AI功能记录ID
     */
    @TableId(value = "id", type = IdType.AUTO)
    private Long id;
    
    /**
     * 功能类型：QUESTION_GENERATION-题目生成，HOMEWORK_GRADING-作业批改，TEACHING_SUGGESTION-教学建议，LEARNING_ANALYSIS-学习分析等
     */
    @TableField("feature_type")
    private String featureType;
    
    /**
     * 用户ID
     */
    @TableField("user_id")
    private Long userId;
    
    /**
     * 用户类型：TEACHER-教师，STUDENT-学生
     */
    @TableField("user_type")
    private String userType;
    
    /**
     * 输入数据（JSON格式）
     */
    @TableField("input_data")
    private String inputData;
    
    /**
     * 输出结果（JSON格式）
     */
    @TableField("output_result")
    private String outputResult;
    
    /**
     * 处理状态：PENDING-处理中，SUCCESS-成功，FAILED-失败
     */
    @TableField("status")
    private String status;
    
    /**
     * 错误信息
     */
    @TableField("error_message")
    private String errorMessage;
    
    /**
     * 处理耗时（毫秒）
     */
    @TableField("processing_time")
    private Long processingTime;
    
    /**
     * AI模型版本
     */
    @TableField("model_version")
    private String modelVersion;
    
    /**
     * 置信度
     */
    @TableField("confidence")
    private Double confidence;
    
    /**
     * 相关课程ID
     */
    @TableField("course_id")
    private Long courseId;
    
    /**
     * 相关任务ID
     */
    @TableField("task_id")
    private Long taskId;
    
    /**
     * 会话ID
     */
    @TableField("session_id")
    private String sessionId;
    
    /**
     * 是否收藏
     */
    @TableField("is_favorite")
    private Boolean isFavorite;
    
    /**
     * 用户评分（1-5星）
     */
    @TableField("user_rating")
    private Integer userRating;
    
    /**
     * 用户反馈
     */
    @TableField("user_feedback")
    private String userFeedback;
    
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