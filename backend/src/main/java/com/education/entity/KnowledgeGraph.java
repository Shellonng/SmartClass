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
 * 知识图谱实体类
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
@Data
@EqualsAndHashCode(callSuper = false)
@Accessors(chain = true)
@TableName("knowledge_graph")
@Schema(description = "知识图谱信息")
public class KnowledgeGraph implements Serializable {

    private static final long serialVersionUID = 1L;

    @Schema(description = "图谱ID")
    @TableId(value = "id", type = IdType.AUTO)
    private Long id;

    @Schema(description = "关联课程ID")
    @TableField("course_id")
    private Long courseId;

    @Schema(description = "图谱标题")
    @TableField("title")
    private String title;

    @Schema(description = "图谱描述")
    @TableField("description")
    private String description;

    @Schema(description = "图谱类型", allowableValues = {"concept", "skill", "comprehensive"})
    @TableField("graph_type")
    private String graphType;

    @Schema(description = "图谱数据(JSON格式)", hidden = true)
    @TableField("graph_data")
    private String graphData;

    @Schema(description = "创建者ID")
    @TableField("creator_id")
    private Long creatorId;

    @Schema(description = "图谱状态", allowableValues = {"draft", "published", "archived"})
    @TableField("status")
    private String status;

    // 以下字段在数据库中不存在，注释掉
    /*
    @Schema(description = "版本号")
    @TableField("version")
    private Integer version;

    @Schema(description = "是否公开")
    @TableField("is_public")
    private Boolean isPublic;

    @Schema(description = "访问次数")
    @TableField("view_count")
    private Integer viewCount;
    */

    @Schema(description = "创建时间")
    @TableField(value = "create_time", fill = FieldFill.INSERT)
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime createTime;

    @Schema(description = "更新时间")
    @TableField(value = "update_time", fill = FieldFill.INSERT_UPDATE)
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime updateTime;

    // 非数据库字段
    @Schema(description = "课程名称")
    @TableField(exist = false)
    private String courseName;

    @Schema(description = "创建者姓名")
    @TableField(exist = false)
    private String creatorName;

    /**
     * 图谱类型枚举
     */
    public enum GraphType {
        CONCEPT("concept", "概念图谱"),
        SKILL("skill", "技能图谱"),
        COMPREHENSIVE("comprehensive", "综合图谱");

        private final String code;
        private final String description;

        GraphType(String code, String description) {
            this.code = code;
            this.description = description;
        }

        public String getCode() {
            return code;
        }

        public String getDescription() {
            return description;
        }
    }

    /**
     * 图谱状态枚举
     */
    public enum Status {
        DRAFT("draft", "草稿"),
        PUBLISHED("published", "已发布"),
        ARCHIVED("archived", "已归档");

        private final String code;
        private final String description;

        Status(String code, String description) {
            this.code = code;
            this.description = description;
        }

        public String getCode() {
            return code;
        }

        public String getDescription() {
            return description;
        }
    }
} 