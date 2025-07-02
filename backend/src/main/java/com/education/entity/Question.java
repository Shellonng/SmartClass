package com.education.entity;

import com.baomidou.mybatisplus.annotation.*;
import com.fasterxml.jackson.annotation.JsonFormat;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.experimental.Accessors;

import java.io.Serializable;
import java.time.LocalDateTime;
import java.util.List;

/**
 * 题目实体类
 */
@Data
@EqualsAndHashCode(callSuper = false)
@Accessors(chain = true)
@TableName("question")
@Schema(description = "题目信息")
public class Question implements Serializable {

    private static final long serialVersionUID = 1L;

    @Schema(description = "题目ID")
    @TableId(value = "id", type = IdType.AUTO)
    private Long id;

    @Schema(description = "题干内容")
    @TableField("title")
    private String title;

    @Schema(description = "题目类型")
    @TableField("question_type")
    private String questionType;

    @Schema(description = "难度等级，1~5整数")
    @TableField("difficulty")
    private Integer difficulty;

    @Schema(description = "标准答案")
    @TableField("correct_answer")
    private String correctAnswer;

    @Schema(description = "答案解析")
    @TableField("explanation")
    private String explanation;

    @Schema(description = "知识点")
    @TableField("knowledge_point")
    private String knowledgePoint;

    @Schema(description = "课程ID")
    @TableField("course_id")
    private Long courseId;

    @Schema(description = "章节ID")
    @TableField("chapter_id")
    private Long chapterId;

    @Schema(description = "创建者用户ID")
    @TableField("created_by")
    private Long createdBy;

    @Schema(description = "创建时间")
    @TableField(value = "create_time", fill = FieldFill.INSERT)
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime createTime;

    @Schema(description = "更新时间")
    @TableField(value = "update_time", fill = FieldFill.INSERT_UPDATE)
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime updateTime;

    // 非数据库字段
    @TableField(exist = false)
    private List<QuestionOption> options;

    @TableField(exist = false)
    private List<QuestionImage> images;

    @TableField(exist = false)
    private Chapter chapter;

    @TableField(exist = false)
    private Course course;

    @TableField(exist = false)
    private Teacher teacher;

    /**
     * 题目类型枚举
     */
    public enum QuestionType {
        SINGLE("single", "单选题"),
        MULTIPLE("multiple", "多选题"),
        TRUE_FALSE("true_false", "判断题"),
        BLANK("blank", "填空题"),
        SHORT("short", "简答题"),
        CODE("code", "编程题");

        private final String code;
        private final String description;

        QuestionType(String code, String description) {
            this.code = code;
            this.description = description;
        }

        public String getCode() {
            return code;
        }

        public String getDescription() {
            return description;
        }

        public static String getDescriptionByCode(String code) {
            for (QuestionType type : QuestionType.values()) {
                if (type.getCode().equals(code)) {
                    return type.getDescription();
                }
            }
            return code;
        }
    }
} 