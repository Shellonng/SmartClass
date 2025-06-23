package com.education.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.education.entity.KnowledgeGraph;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

import java.util.List;
import java.util.Map;

/**
 * 知识图谱数据访问层
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
@Mapper
public interface KnowledgeGraphMapper extends BaseMapper<KnowledgeGraph> {
    
    /**
     * 根据课程ID查询知识图谱
     */
    List<KnowledgeGraph> selectByCourseId(@Param("courseId") Long courseId);
    
    /**
     * 根据创建者ID查询知识图谱
     */
    List<KnowledgeGraph> selectByCreatedBy(@Param("createdBy") Long createdBy);
    
    /**
     * 查询公开的知识图谱
     */
    List<KnowledgeGraph> selectPublicGraphs();
    
    /**
     * 根据标签查询知识图谱
     */
    List<KnowledgeGraph> selectByTags(@Param("tags") List<String> tags);
    
    /**
     * 搜索知识图谱
     */
    List<KnowledgeGraph> searchGraphs(@Param("keyword") String keyword);
    
    /**
     * 查询知识图谱版本历史
     */
    List<KnowledgeGraph> selectVersionHistory(@Param("name") String name, @Param("courseId") Long courseId);
    
    /**
     * 查询最新版本的知识图谱
     */
    KnowledgeGraph selectLatestVersion(@Param("name") String name, @Param("courseId") Long courseId);
    
    /**
     * 查询知识图谱统计信息
     */
    Map<String, Object> selectGraphStats(@Param("graphId") Long graphId);
    
    /**
     * 查询热门知识图谱
     */
    List<KnowledgeGraph> selectPopularGraphs(@Param("limit") Integer limit);
    
    /**
     * 查询推荐知识图谱
     */
    List<KnowledgeGraph> selectRecommendedGraphs(@Param("userId") Long userId, @Param("limit") Integer limit);
    
    /**
     * 复制知识图谱
     */
    int copyGraph(@Param("sourceId") Long sourceId, @Param("targetName") String targetName, @Param("createdBy") Long createdBy);
    
    /**
     * 合并知识图谱
     */
    int mergeGraphs(@Param("sourceIds") List<Long> sourceIds, @Param("targetName") String targetName, @Param("createdBy") Long createdBy);
    
    /**
     * 验证知识图谱完整性
     */
    Map<String, Object> validateGraphIntegrity(@Param("graphId") Long graphId);
    
    /**
     * 获取知识图谱可视化数据
     */
    Map<String, Object> selectVisualizationData(@Param("graphId") Long graphId);
    
    /**
     * 批量更新知识图谱状态
     */
    int batchUpdateStatus(@Param("graphIds") List<Long> graphIds, @Param("status") String status);
}