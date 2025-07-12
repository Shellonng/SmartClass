package com.education.dto;

import com.education.entity.Question;
import lombok.Data;

import java.time.LocalDateTime;
import java.util.List;

/**
 * 考试数据传输对象
 */
@Data
public class ExamDTO {
    
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
     * 所属课程名称
     */
    private String courseName;
    
    /**
     * 发布考试的用户ID（教师）
     */
    private Long userId;
    
    /**
     * 发布考试的用户名称
     */
    private String userName;
    
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
    private Integer duration;
    
    /**
     * 考试总分
     */
    private Integer totalScore;
    
    /**
     * 发布状态：0 未发布，1 已发布
     */
    private Integer status;
    
    /**
     * 状态描述
     */
    private String statusDesc;
    
    /**
     * 创建时间
     */
    private LocalDateTime createTime;
    
    /**
     * 类型：exam(考试) 或 homework(作业)
     */
    private String type;
    
    /**
     * 作业模式：question-答题型，file-上传型
     */
    private String mode;
    
    /**
     * 提交率（百分比，0-100）
     */
    private Double submissionRate;
    
    /**
     * 考试题目配置
     */
    private ExamPaperConfig paperConfig;
    
    /**
     * 考试题目列表
     */
    private List<ExamQuestionDTO> questions;
    
    /**
     * 参考答案（用于智能批改）
     */
    private String referenceAnswer;
    
    /**
     * 考试组卷配置
     */
    @Data
    public static class ExamPaperConfig {
        /**
         * 单选题数量
         */
        private Integer singleCount;
        
        /**
         * 单选题分值
         */
        private Integer singleScore;
        
        /**
         * 多选题数量
         */
        private Integer multipleCount;
        
        /**
         * 多选题分值
         */
        private Integer multipleScore;
        
        /**
         * 判断题数量
         */
        private Integer trueFalseCount;
        
        /**
         * 判断题分值
         */
        private Integer trueFalseScore;
        
        /**
         * 填空题数量
         */
        private Integer blankCount;
        
        /**
         * 填空题分值
         */
        private Integer blankScore;
        
        /**
         * 简答题数量
         */
        private Integer shortCount;
        
        /**
         * 简答题分值
         */
        private Integer shortScore;
        
        /**
         * 编程题数量
         */
        private Integer codeCount;
        
        /**
         * 编程题分值
         */
        private Integer codeScore;
        
        /**
         * 是否随机组卷
         */
        private Boolean isRandom;
        
        /**
         * 难度等级
         */
        private Integer difficulty;
        
        /**
         * 知识点
         */
        private String knowledgePoint;
    }
    
    /**
     * 考试题目DTO
     */
    @Data
    public static class ExamQuestionDTO {
        /**
         * 题目ID
         */
        private Long id;
        
        /**
         * 题干内容
         */
        private String title;
        
        /**
         * 题型
         */
        private String questionType;
        
        /**
         * 题型描述
         */
        private String questionTypeDesc;
        
        /**
         * 难度等级
         */
        private Integer difficulty;
        
        /**
         * 分值
         */
        private Integer score;
        
        /**
         * 顺序
         */
        private Integer sequence;
        
        /**
         * 选项列表（选择题）
         */
        private List<QuestionOptionDTO> options;
        
        /**
         * 标准答案
         */
        private String correctAnswer;
        
        /**
         * 答案解析
         */
        private String explanation;
        
        /**
         * 知识点
         */
        private String knowledgePoint;
    }
    
    /**
     * 题目选项DTO
     */
    @Data
    public static class QuestionOptionDTO {
        /**
         * 选项ID
         */
        private Long id;
        
        /**
         * 选项标识 A/B/C/D/T/F
         */
        private String optionLabel;
        
        /**
         * 选项内容
         */
        private String optionText;
    }
} 