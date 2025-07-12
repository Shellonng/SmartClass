package com.education.dto;

import com.fasterxml.jackson.annotation.JsonFormat;
import lombok.Data;

import java.time.LocalDateTime;
import java.util.List;

/**
 * 作业数据传输对象
 * 用于前后端数据交互
 */
@Data
public class AssignmentDTO {
    
    private Long id;
    
    private String title;
    
    private Long courseId;
    
    private String courseName;
    
    private Long userId;
    
    private String teacherName;
    
    private String type; // homework 或 exam
    
    private String mode; // question（答题型）或 file（文件上传型）
    
    private String description;
    
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime startTime;
    
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime endTime;
    
    private Integer status; // 0-未发布，1-已发布
    
    private Integer timeLimit; // 时间限制（分钟）
    
    private Integer totalScore; // 总分
    
    private Integer duration; // 考试时长（分钟）
    
    private List<String> allowedFileTypes; // 允许的文件类型
    
    private Integer maxFileSize; // 最大文件大小（MB）
    
    private String referenceAnswer; // 参考答案
    
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime createTime;
    
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime updateTime;
    
    // 智能组卷配置
    private AssignmentConfig config;
    
    // 题目列表（用于答题型作业）
    private List<AssignmentQuestionDTO> questions;
    
    // 提交率（用于列表显示）
    private Double submissionRate;
    
    /**
     * 作业配置内部类
     */
    @Data
    public static class AssignmentConfig {
        private Long id;
        private Long assignmentId;
        private List<Long> knowledgePoints; // 知识点ID列表
        private String difficulty; // EASY, MEDIUM, HARD
        private Integer questionCount; // 题目总数
        private QuestionTypeConfig questionTypes; // 题型分布
        private String additionalRequirements; // 额外要求
    }
    
    /**
     * 题型配置内部类
     */
    @Data
    public static class QuestionTypeConfig {
        private Integer singleChoice; // 单选题数量
        private Integer multipleChoice; // 多选题数量
        private Integer trueFalse; // 判断题数量
        private Integer fillBlank; // 填空题数量
        private Integer shortAnswer; // 简答题数量
        private Integer essay; // 论述题数量
    }
    
    /**
     * 作业题目内部类
     */
    @Data
    public static class AssignmentQuestionDTO {
        private Long id;
        private Long questionId;
        private String content; // 题干
        private String type; // 题型
        private String difficulty; // 难度
        private Integer score; // 分值
        private Integer orderNum; // 题目顺序
        private List<QuestionOptionDTO> options; // 选项
        private String correctAnswer; // 标准答案
        private String explanation; // 解析
        private List<String> knowledgePoints; // 知识点
    }
    
    /**
     * 题目选项内部类
     */
    @Data
    public static class QuestionOptionDTO {
        private String label; // 选项标签（A、B、C、D）
        private String content; // 选项内容
        private Boolean isCorrect; // 是否为正确答案
    }
}