package com.education.entity;

import com.baomidou.mybatisplus.annotation.*;
import com.fasterxml.jackson.annotation.JsonFormat;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.experimental.Accessors;

import java.io.Serializable;
import java.time.LocalDateTime;

/**
 * 题目图片实体类
 */
@Data
@EqualsAndHashCode(callSuper = false)
@Accessors(chain = true)
@TableName("question_image")
@Schema(description = "题目图片信息")
public class QuestionImage implements Serializable {

    private static final long serialVersionUID = 1L;

    @Schema(description = "图片ID")
    @TableId(value = "id", type = IdType.AUTO)
    private Long id;

    @Schema(description = "题目ID")
    @TableField("question_id")
    private Long questionId;

    @Schema(description = "图片URL或路径")
    @TableField("image_url")
    private String imageUrl;

    @Schema(description = "图片说明")
    @TableField("description")
    private String description;

    @Schema(description = "图片显示顺序")
    @TableField("sequence")
    private Integer sequence;

    @Schema(description = "上传时间")
    @TableField("upload_time")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime uploadTime;
} 