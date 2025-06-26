package com.education.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.education.entity.KnowledgeNode;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

import java.util.List;

/**
 * 知识节点Mapper接口
 */
@Mapper
public interface KnowledgeNodeMapper extends BaseMapper<KnowledgeNode> {

    /**
     * 根据知识图谱ID查询节点列表
     */
    List<KnowledgeNode> selectByKnowledgeGraphId(@Param("knowledgeGraphId") Long knowledgeGraphId);

    /**
     * 根据节点类型查询节点列表
     */
    List<KnowledgeNode> selectByNodeType(@Param("nodeType") String nodeType);

    /**
     * 根据重要程度查询节点列表
     */
    List<KnowledgeNode> selectByImportanceLevel(@Param("importanceLevel") Integer importanceLevel);

    /**
     * 根据难度等级查询节点列表
     */
    List<KnowledgeNode> selectByDifficultyLevel(@Param("difficultyLevel") Integer difficultyLevel);

    /**
     * 批量插入节点
     */
    int batchInsert(@Param("nodes") List<KnowledgeNode> nodes);

    /**
     * 根据知识图谱ID删除节点
     */
    int deleteByKnowledgeGraphId(@Param("knowledgeGraphId") Long knowledgeGraphId);
}