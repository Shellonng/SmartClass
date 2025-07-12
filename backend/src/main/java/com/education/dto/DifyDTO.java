package com.education.dto;

import lombok.Data;
import lombok.Builder;
import lombok.AllArgsConstructor;
import lombok.NoArgsConstructor;

import java.util.List;
import java.util.Map;

/**
 * Dify相关DTO
 * @author Education Platform Team
 */
public class DifyDTO {

    /**
     * 组卷请求
     */
    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class PaperGenerationRequest {
        /**
         * 课程ID
         */
        private Long courseId;
        
        /**
         * 知识点范围
         */
        private List<String> knowledgePoints;
        
        /**
         * 难度级别 (EASY, MEDIUM, HARD)
         */
        private String difficulty;
        
        /**
         * 题目数量
         */
        private Integer questionCount;
        
        /**
         * 题型分布
         */
        private Map<String, Integer> questionTypes;
        
        /**
         * 考试时长(分钟)
         */
        private Integer duration;
        
        /**
         * 总分
         */
        private Integer totalScore;
        
        /**
         * 额外要求
         */
        private String additionalRequirements;
    }

    /**
     * 组卷响应
     */
    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class PaperGenerationResponse {
        /**
         * 试卷标题
         */
        private String title;
        
        /**
         * 生成的题目列表
         */
        private List<GeneratedQuestion> questions;
        
        /**
         * 生成状态
         */
        private String status;
        
        /**
         * 任务ID（用于异步查询）
         */
        private String taskId;
        
        /**
         * 错误信息（如果有）
         */
        private String errorMessage;
    }

    /**
     * 生成的题目
     */
    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class GeneratedQuestion {
        /**
         * 题目内容
         */
        private String questionText;
        
        /**
         * 题目类型
         */
        private String questionType;
        
        /**
         * 选项（选择题）
         */
        private List<String> options;
        
        /**
         * 正确答案
         */
        private String correctAnswer;
        
        /**
         * 分值
         */
        private Integer score;
        
        /**
         * 知识点
         */
        private String knowledgePoint;
        
        /**
         * 难度
         */
        private String difficulty;
        
        /**
         * 解析
         */
        private String explanation;
    }

    /**
     * 自动批改请求
     */
    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class AutoGradingRequest {
        /**
         * 提交ID
         */
        private Long submissionId;
        
        /**
         * 作业/考试ID
         */
        private Long assignmentId;
        
        /**
         * 学生ID
         */
        private Long studentId;
        
        /**
         * 题目列表
         */
        private List<Question> questions;
        
        /**
         * 题目和学生答案
         */
        private List<StudentAnswer> answers;
        
        /**
         * 学生答案（新版本）
         */
        private List<StudentAnswer> studentAnswers;
        
        /**
         * 批改类型 (OBJECTIVE, SUBJECTIVE, MIXED)
         */
        private String gradingType;
        
        /**
         * 评分标准
         */
        private String gradingCriteria;
        
        /**
         * 最大分数
         */
        private Double maxScore;
    }

    /**
     * 题目信息
     */
    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class Question {
        /**
         * 题目ID
         */
        private Long questionId;
        
        /**
         * 题目内容
         */
        private String questionText;
        
        /**
         * 题目类型
         */
        private String questionType;
        
        /**
         * 选项（选择题）
         */
        private List<String> options;
        
        /**
         * 标准答案
         */
        private String correctAnswer;
        
        /**
         * 题目分值
         */
        private Integer score;
        
        /**
         * 知识点
         */
        private String knowledgePoint;
    }

    /**
     * 学生答案
     */
    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class StudentAnswer {
        /**
         * 题目ID
         */
        private Long questionId;
        
        /**
         * 题目内容
         */
        private String questionText;
        
        /**
         * 题目类型
         */
        private String questionType;
        
        /**
         * 标准答案
         */
        private String correctAnswer;
        
        /**
         * 学生答案
         */
        private String studentAnswer;
        
        /**
         * 题目分值
         */
        private Integer totalScore;
    }

    /**
     * 自动批改响应
     */
    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class AutoGradingResponse {
        /**
         * 批改结果
         */
        private List<GradingResult> results;
        
        /**
         * 总分
         */
        private Integer totalScore;
        
        /**
         * 得分
         */
        private Integer earnedScore;
        
        /**
         * 百分比得分
         */
        private Double percentage;
        
        /**
         * 整体评价
         */
        private String overallComment;
        
        /**
         * 批改状态
         */
        private String status;
        
        /**
         * 任务ID
         */
        private String taskId;
        
        /**
         * 错误信息（如果有）
         */
        private String errorMessage;
    }

    /**
     * 单题批改结果
     */
    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class GradingResult {
        /**
         * 题目ID
         */
        private Long questionId;
        
        /**
         * 是否正确
         */
        private Boolean isCorrect;
        
        /**
         * 得分
         */
        private Integer score;
        
        /**
         * 总分
         */
        private Integer totalScore;
        
        /**
         * 批改意见
         */
        private String comment;
        
        /**
         * 错误类型（如果错误）
         */
        private String errorType;
        
        /**
         * 建议
         */
        private String suggestion;
    }

    /**
     * Dify API通用请求
     */
    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class DifyRequest {
        /**
         * 输入参数
         */
        private Map<String, Object> inputs;
        
        /**
         * 用户标识
         */
        private String user;
        
        /**
         * 是否流式响应
         */
        @Builder.Default
        private Boolean stream = false;
        
        /**
         * 响应模式 (blocking/streaming)
         */
        @Builder.Default
        private String responseMode = "blocking";
    }

    /**
     * Dify API通用响应
     */
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class DifyResponse {
        /**
         * 工作流执行ID
         */
        private String workflowRunId;
        
        /**
         * 任务ID
         */
        private String taskId;
        
        /**
         * 消息ID
         */
        private String messageId;
        
        /**
         * 对话ID
         */
        private String conversationId;
        
        /**
         * 执行状态
         */
        private String status;
        
        /**
         * 响应数据
         */
        private Map<String, Object> data;
        
        /**
         * 错误信息
         */
        private String error;
        
        /**
         * Token使用情况
         */
        private Map<String, Integer> usage;
        
        /**
         * 元数据
         */
        private Map<String, Object> metadata;
    }
} 