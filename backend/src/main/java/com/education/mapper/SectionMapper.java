package com.education.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.education.entity.Section;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

import java.util.List;

/**
 * 小节Mapper接口
 */
@Mapper
public interface SectionMapper extends BaseMapper<Section> {
    
    /**
     * 根据章节ID查询小节列表
     * 
     * @param chapterId 章节ID
     * @return 小节列表
     */
    @Select("SELECT * FROM section WHERE chapter_id = #{chapterId} ORDER BY sort_order ASC")
    List<Section> selectByChapterId(@Param("chapterId") Long chapterId);
    
    /**
     * 根据课程ID查询所有小节列表
     * 
     * @param courseId 课程ID
     * @return 小节列表
     */
    @Select("SELECT s.* FROM section s JOIN chapter c ON s.chapter_id = c.id WHERE c.course_id = #{courseId} ORDER BY c.sort_order ASC, s.sort_order ASC")
    List<Section> selectByCourseId(@Param("courseId") Long courseId);
} 