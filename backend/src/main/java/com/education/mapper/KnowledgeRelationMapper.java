package com.education.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.education.entity.KnowledgeRelation;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

import java.util.List;

/**
 * 知识关系Mapper接口
 */
@Mapper
public interface KnowledgeRelationMapper extends BaseMapper<KnowledgeRelation> {

    /**
     * 根据知识图谱ID查询关系列表
     */
    List<KnowledgeRelation> selectByKnowledgeGraphId(@Param("knowledgeGraphId") Long knowledgeGraphId);

    /**
     * 根据源节点ID查询关系列表
     */
    List<KnowledgeRelation> selectBySourceNodeId(@Param("sourceNodeId") Long sourceNodeId);

    /**
     * 根据目标节点ID查询关系列表
     */
    List<KnowledgeRelation> selectByTargetNodeId(@Param("targetNodeId") Long targetNodeId);

    /**
     * 根据关系类型查询关系列表
     */
    List<KnowledgeRelation> selectByRelationType(@Param("relationType") String relationType);

    /**
     * 查询两个节点之间的关系
     */
    List<KnowledgeRelation> selectBetweenNodes(@Param("sourceNodeId") Long sourceNodeId, 
                                               @Param("targetNodeId") Long targetNodeId);

    /**
     * 批量插入关系
     */
    int batchInsert(@Param("relations") List<KnowledgeRelation> relations);

    /**
     * 根据知识图谱ID删除关系
     */
    int deleteByKnowledgeGraphId(@Param("knowledgeGraphId") Long knowledgeGraphId);

    /**
     * 根据节点ID删除相关关系
     */
    int deleteByNodeId(@Param("nodeId") Long nodeId);
}