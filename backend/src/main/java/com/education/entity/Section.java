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
 * 课程小节实体类
 */
@Data
@EqualsAndHashCode(callSuper = false)
@Accessors(chain = true)
@TableName("section")
@Schema(description = "课程小节信息")
public class Section implements Serializable {

    private static final long serialVersionUID = 1L;

    @Schema(description = "小节ID")
    @TableId(value = "id", type = IdType.AUTO)
    private Long id;

    @Schema(description = "所属章节ID")
    @TableField("chapter_id")
    private Long chapterId;

    @Schema(description = "小节名称")
    @TableField("title")
    private String title;

    @Schema(description = "小节简介")
    @TableField("description")
    private String description;

    @Schema(description = "视频播放地址")
    @TableField("video_url")
    private String videoUrl;

    @Schema(description = "视频时长(秒)")
    @TableField("duration")
    private Integer duration;

    @Schema(description = "小节顺序")
    @TableField("sort_order")
    private Integer sortOrder;

    @Schema(description = "创建时间")
    @TableField(value = "create_time", fill = FieldFill.INSERT)
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime createTime;

    @Schema(description = "更新时间")
    @TableField(value = "update_time", fill = FieldFill.INSERT_UPDATE)
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime updateTime;
} 