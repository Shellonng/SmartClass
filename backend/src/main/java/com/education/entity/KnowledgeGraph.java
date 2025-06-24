package com.education.entity;

import com.baomidou.mybatisplus.annotation.*;
import lombok.Data;
import lombok.EqualsAndHashCode;

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
@TableName("knowledge_graph")
public class KnowledgeGraph {
    
    /**
     * 知识图谱ID
     */
    @TableId(value = "id", type = IdType.AUTO)
    private Long id;
    
    /**
     * 知识图谱名称
     */
    @TableField("name")
    private String name;
    
    /**
     * 知识图谱描述
     */
    @TableField("description")
    private String description;
    
    /**
     * 课程ID
     */
    @TableField("course_id")
    private Long courseId;
    
    /**
     * 创建者ID
     */
    @TableField("created_by")
    private Long createdBy;
    
    /**
     * 知识图谱数据（JSON格式）
     */
    @TableField("graph_data")
    private String graphData;
    
    /**
     * 版本号
     */
    @TableField("version")
    private String version;
    
    /**
     * 状态：DRAFT-草稿，PUBLISHED-已发布，ARCHIVED-已归档
     */
    @TableField("status")
    private String status;
    
    /**
     * 是否公开
     */
    @TableField("is_public")
    private Boolean isPublic;
    
    /**
     * 标签
     */
    @TableField("tags")
    private String tags;
    
    /**
     * 创建时间
     */
    @TableField(value = "create_time", fill = FieldFill.INSERT)
    private LocalDateTime createTime;
    
    /**
     * 更新时间
     */
    @TableField(value = "update_time", fill = FieldFill.INSERT_UPDATE)
    private LocalDateTime updateTime;
    
    /**
     * 是否删除
     */
    @TableLogic
    @TableField("is_deleted")
    private Boolean isDeleted;
}