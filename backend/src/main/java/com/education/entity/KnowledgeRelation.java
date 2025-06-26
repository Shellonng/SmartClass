package com.education.entity;

import com.baomidou.mybatisplus.annotation.*;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.experimental.Accessors;

import java.io.Serializable;
import java.time.LocalDateTime;

/**
 * 知识关系实体类
 */
@Data
@EqualsAndHashCode(callSuper = false)
@Accessors(chain = true)
@TableName("knowledge_relation")
public class KnowledgeRelation implements Serializable {

    private static final long serialVersionUID = 1L;

    /**
     * 主键ID
     */
    @TableId(value = "id", type = IdType.AUTO)
    private Long id;

    /**
     * 知识图谱ID
     */
    @TableField("knowledge_graph_id")
    private Long knowledgeGraphId;

    /**
     * 源节点ID
     */
    @TableField("source_node_id")
    private Long sourceNodeId;

    /**
     * 目标节点ID
     */
    @TableField("target_node_id")
    private Long targetNodeId;

    /**
     * 关系类型
     */
    @TableField("relation_type")
    private String relationType;

    /**
     * 关系名称
     */
    @TableField("relation_name")
    private String relationName;

    /**
     * 关系描述
     */
    @TableField("description")
    private String description;

    /**
     * 关系权重
     */
    @TableField("weight")
    private Double weight;

    /**
     * 关系强度
     */
    @TableField("strength")
    private Integer strength;

    /**
     * 关系状态
     */
    @TableField("status")
    private String status;

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
     * 创建者ID
     */
    @TableField("creator_id")
    private Long creatorId;

    /**
     * 更新者ID
     */
    @TableField("updater_id")
    private Long updaterId;

    /**
     * 是否删除
     */
    @TableLogic
    @TableField("is_deleted")
    private Boolean isDeleted;
}