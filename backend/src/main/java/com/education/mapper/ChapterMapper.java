package com.education.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.education.entity.Chapter;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

import java.util.List;

/**
 * 章节Mapper接口
 */
@Mapper
public interface ChapterMapper extends BaseMapper<Chapter> {
    
    /**
     * 根据课程ID查询章节列表
     * 
     * @param courseId 课程ID
     * @return 章节列表
     */
    @Select("SELECT * FROM chapter WHERE course_id = #{courseId} ORDER BY sort_order ASC")
    List<Chapter> selectByCourseId(@Param("courseId") Long courseId);
} 