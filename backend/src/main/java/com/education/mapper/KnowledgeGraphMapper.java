package com.education.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.education.entity.KnowledgeGraph;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;
import org.apache.ibatis.annotations.Update;

import java.util.List;

/**
 * 知识图谱Mapper接口
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
@Mapper
public interface KnowledgeGraphMapper extends BaseMapper<KnowledgeGraph> {

    /**
     * 分页查询知识图谱（包含关联信息）
     */
    @Select("""
        SELECT kg.*, c.title as course_name, u.real_name as creator_name
        FROM knowledge_graph kg
        LEFT JOIN course c ON kg.course_id = c.id
        LEFT JOIN user u ON kg.creator_id = u.id
        ${ew.customSqlSegment}
    """)
    IPage<KnowledgeGraph> selectPageWithRelations(Page<KnowledgeGraph> page, @Param("ew") com.baomidou.mybatisplus.core.conditions.Wrapper<KnowledgeGraph> wrapper);

    /**
     * 根据课程ID查询知识图谱列表
     */
    @Select("""
        SELECT kg.*, c.title as course_name, u.real_name as creator_name
        FROM knowledge_graph kg
        LEFT JOIN course c ON kg.course_id = c.id
        LEFT JOIN user u ON kg.creator_id = u.id
        WHERE kg.course_id = #{courseId}
        AND kg.status != 'archived'
        ORDER BY kg.create_time DESC
    """)
    List<KnowledgeGraph> selectByCourseId(@Param("courseId") Long courseId);

    /**
     * 根据课程ID和图谱类型查询
     */
    @Select("""
        SELECT kg.*, c.title as course_name, u.real_name as creator_name
        FROM knowledge_graph kg
        LEFT JOIN course c ON kg.course_id = c.id
        LEFT JOIN user u ON kg.creator_id = u.id
        WHERE kg.course_id = #{courseId}
        AND kg.graph_type = #{graphType}
        AND kg.status = 'published'
        ORDER BY kg.version DESC
        LIMIT 1
    """)
    KnowledgeGraph selectLatestByTypeAndCourse(@Param("courseId") Long courseId, @Param("graphType") String graphType);

    /**
     * 查询用户创建的知识图谱
     */
    @Select("""
        SELECT kg.*, c.title as course_name
        FROM knowledge_graph kg
        LEFT JOIN course c ON kg.course_id = c.id
        WHERE kg.creator_id = #{creatorId}
        ORDER BY kg.update_time DESC
    """)
    List<KnowledgeGraph> selectByCreatorId(@Param("creatorId") Long creatorId);

    /**
     * 查询公开的知识图谱
     */
    @Select("""
        SELECT kg.*, c.title as course_name, u.real_name as creator_name
        FROM knowledge_graph kg
        LEFT JOIN course c ON kg.course_id = c.id
        LEFT JOIN user u ON kg.creator_id = u.id
        WHERE kg.status = 'published'
        ORDER BY kg.update_time DESC
        LIMIT #{limit}
    """)
    List<KnowledgeGraph> selectPublicGraphs(@Param("limit") Integer limit);

    /**
     * 查询已发布状态的知识图谱
     */
    @Select("""
        SELECT kg.*, c.title as course_name, u.real_name as creator_name
        FROM knowledge_graph kg
        LEFT JOIN course c ON kg.course_id = c.id
        LEFT JOIN user u ON kg.creator_id = u.id
        WHERE kg.status = 'published'
        ORDER BY kg.update_time DESC
        LIMIT #{limit}
    """)
    List<KnowledgeGraph> selectPublishedGraphs(@Param("limit") Integer limit);

    /**
     * 增加访问次数 - 注释掉，因为没有view_count列
     */
    // @Update("UPDATE knowledge_graph SET view_count = view_count + 1 WHERE id = #{id}")
    // int incrementViewCount(@Param("id") Long id);

    /**
     * 搜索知识图谱
     */
    @Select("""
        SELECT kg.*, c.title as course_name, u.real_name as creator_name
        FROM knowledge_graph kg
        LEFT JOIN course c ON kg.course_id = c.id
        LEFT JOIN user u ON kg.creator_id = u.id
        WHERE (kg.title LIKE CONCAT('%', #{keyword}, '%')
        OR kg.description LIKE CONCAT('%', #{keyword}, '%')
        OR c.title LIKE CONCAT('%', #{keyword}, '%'))
        AND kg.status = 'published'
        AND (kg.creator_id = #{userId})
        ORDER BY kg.update_time DESC
    """)
    List<KnowledgeGraph> searchGraphs(@Param("keyword") String keyword, @Param("userId") Long userId);

    /**
     * 统计用户创建的图谱数量
     */
    @Select("SELECT COUNT(*) FROM knowledge_graph WHERE creator_id = #{creatorId}")
    Long countByCreatorId(@Param("creatorId") Long creatorId);

    /**
     * 统计课程的图谱数量
     */
    @Select("SELECT COUNT(*) FROM knowledge_graph WHERE course_id = #{courseId} AND status != 'archived'")
    Long countByCourseId(@Param("courseId") Long courseId);
} 