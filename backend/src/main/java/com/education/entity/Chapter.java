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
 * 课程章节实体类
 */
@Data
@EqualsAndHashCode(callSuper = false)
@Accessors(chain = true)
@TableName("chapter")
@Schema(description = "课程章节信息")
public class Chapter implements Serializable {

    private static final long serialVersionUID = 1L;

    @Schema(description = "章节ID")
    @TableId(value = "id", type = IdType.AUTO)
    private Long id;

    @Schema(description = "所属课程ID")
    @TableField("course_id")
    private Long courseId;

    @Schema(description = "章节名称")
    @TableField("title")
    private String title;

    @Schema(description = "章节描述")
    @TableField("description")
    private String description;

    @Schema(description = "章节顺序")
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

    @TableField(exist = false)
    private List<Section> sections;
} 