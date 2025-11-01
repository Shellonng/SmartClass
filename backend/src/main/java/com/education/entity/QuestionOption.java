package com.education.entity;

import com.baomidou.mybatisplus.annotation.*;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.experimental.Accessors;

import java.io.Serializable;

/**
 * 题目选项实体类
 */
@Data
@EqualsAndHashCode(callSuper = false)
@Accessors(chain = true)
@TableName("question_option")
@Schema(description = "题目选项信息")
public class QuestionOption implements Serializable {

    private static final long serialVersionUID = 1L;

    @Schema(description = "选项ID")
    @TableId(value = "id", type = IdType.AUTO)
    private Long id;

    @Schema(description = "题目ID")
    @TableField("question_id")
    private Long questionId;

    @Schema(description = "选项标识 A/B/C/D/T/F")
    @TableField("option_label")
    private String optionLabel;

    @Schema(description = "选项内容")
    @TableField("option_text")
    private String optionText;
} 