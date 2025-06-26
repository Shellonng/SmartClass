package com.education.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.education.entity.Chapter;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

import java.util.List;

/**
 * 章节数据访问层
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
@Mapper
public interface ChapterMapper extends BaseMapper<Chapter> {

    /**
     * 根据课程ID查询章节列表
     * 
     * @param courseId 课程ID
     * @return 章节列表
     */
    @Select("SELECT * FROM chapter WHERE course_id = #{courseId} AND is_deleted = 0 ORDER BY sort_order ASC")
    List<Chapter> selectByCourseId(@Param("courseId") Long courseId);

    /**
     * 根据课程ID统计章节数量
     * 
     * @param courseId 课程ID
     * @return 章节数量
     */
    @Select("SELECT COUNT(*) FROM chapter WHERE course_id = #{courseId} AND is_deleted = 0")
    Integer countByCourseId(@Param("courseId") Long courseId);

    /**
     * 根据课程ID统计已发布章节数量
     * 
     * @param courseId 课程ID
     * @return 已发布章节数量
     */
    @Select("SELECT COUNT(*) FROM chapter WHERE course_id = #{courseId} AND status = 'PUBLISHED' AND is_deleted = 0")
    Integer countPublishedByCourseId(@Param("courseId") Long courseId);

    /**
     * 获取课程中最大的排序号
     * 
     * @param courseId 课程ID
     * @return 最大排序号
     */
    @Select("SELECT COALESCE(MAX(sort_order), 0) FROM chapter WHERE course_id = #{courseId} AND is_deleted = 0")
    Integer getMaxSortOrderByCourseId(@Param("courseId") Long courseId);
}