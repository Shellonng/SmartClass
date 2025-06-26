package com.education.entity;

import com.baomidou.mybatisplus.annotation.*;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.experimental.Accessors;

import java.io.Serializable;
import java.time.LocalDateTime;

/**
 * 知识节点实体类
 */
@Data
@EqualsAndHashCode(callSuper = false)
@Accessors(chain = true)
@TableName("knowledge_node")
public class KnowledgeNode implements Serializable {

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
     * 节点名称
     */
    @TableField("node_name")
    private String nodeName;

    /**
     * 节点类型
     */
    @TableField("node_type")
    private String nodeType;

    /**
     * 节点描述
     */
    @TableField("description")
    private String description;

    /**
     * 重要程度
     */
    @TableField("importance_level")
    private Integer importanceLevel;

    /**
     * 难度等级
     */
    @TableField("difficulty_level")
    private Integer difficultyLevel;

    /**
     * 节点位置X坐标
     */
    @TableField("position_x")
    private Double positionX;

    /**
     * 节点位置Y坐标
     */
    @TableField("position_y")
    private Double positionY;

    /**
     * 节点状态
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