package com.education.dto;

import com.education.entity.QuestionImage;
import com.education.entity.QuestionOption;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.time.LocalDateTime;
import java.util.List;

/**
 * 题目数据传输对象
 */
@Data
@Schema(description = "题目数据传输对象")
public class QuestionDTO {

    @Schema(description = "题目ID")
    private Long id;

    @Schema(description = "题目标题")
    private String title;

    @Schema(description = "题目类型")
    private String questionType;  // single, multiple, true_false, blank, short, code

    @Schema(description = "题目类型描述")
    private String questionTypeDesc;

    @Schema(description = "难度等级")
    private Integer difficulty;  // 1-5

    @Schema(description = "标准答案")
    private String correctAnswer;

    @Schema(description = "答案解析")
    private String explanation;

    @Schema(description = "知识点")
    private String knowledgePoint;

    @Schema(description = "课程ID")
    private Long courseId;

    @Schema(description = "章节ID")
    private Long chapterId;

    @Schema(description = "创建者ID")
    private Long createdBy;

    @Schema(description = "创建时间")
    private LocalDateTime createTime;

    @Schema(description = "更新时间")
    private LocalDateTime updateTime;

    // 额外字段，用于前端显示
    @Schema(description = "课程名称")
    private String courseName;

    @Schema(description = "章节名称")
    private String chapterName;

    @Schema(description = "教师名称")
    private String teacherName;

    @Schema(description = "题目选项列表")
    private List<QuestionOption> options;

    @Schema(description = "题目图片列表")
    private List<QuestionImage> images;

    /**
     * 添加题目请求
     */
    @Data
    @Schema(description = "题目添加请求")
    public static class AddRequest {
        @Schema(description = "题目标题")
        private String title;

        @Schema(description = "题目类型")
        private String questionType;

        @Schema(description = "难度等级")
        private Integer difficulty;

        @Schema(description = "标准答案")
        private String correctAnswer;

        @Schema(description = "答案解析")
        private String explanation;

        @Schema(description = "知识点")
        private String knowledgePoint;

        @Schema(description = "课程ID")
        private Long courseId;

        @Schema(description = "章节ID")
        private Long chapterId;
        
        @Schema(description = "创建者ID")
        private Long createdBy;

        @Schema(description = "题目选项列表")
        private List<QuestionOption> options;

        @Schema(description = "题目图片列表")
        private List<QuestionImage> images;
    }

    /**
     * 更新题目请求
     */
    @Data
    @Schema(description = "题目更新请求")
    public static class UpdateRequest {
        @Schema(description = "题目ID")
        private Long id;

        @Schema(description = "题目标题")
        private String title;

        @Schema(description = "题目类型")
        private String questionType;

        @Schema(description = "难度等级")
        private Integer difficulty;

        @Schema(description = "标准答案")
        private String correctAnswer;

        @Schema(description = "答案解析")
        private String explanation;

        @Schema(description = "知识点")
        private String knowledgePoint;

        @Schema(description = "题目选项列表")
        private List<QuestionOption> options;

        @Schema(description = "题目图片列表")
        private List<QuestionImage> images;
    }

    /**
     * 查询题目请求
     */
    @Data
    @Schema(description = "题目查询请求")
    public static class QueryRequest {
        @Schema(description = "页码")
        private Integer pageNum = 1;

        @Schema(description = "每页数量")
        private Integer pageSize = 10;

        @Schema(description = "课程ID")
        private Long courseId;

        @Schema(description = "章节ID")
        private Long chapterId;

        @Schema(description = "题目类型")
        private String questionType;

        @Schema(description = "难度等级")
        private Integer difficulty;

        @Schema(description = "知识点")
        private String knowledgePoint;

        @Schema(description = "关键词")
        private String keyword;
    }
} 