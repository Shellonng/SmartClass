package com.education.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.education.entity.Course;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

import java.util.List;

/**
 * 课程Mapper接口
 * 
 * @author Education Platform Team
 * @version 1.0.0
 */
@Mapper
public interface CourseMapper extends BaseMapper<Course> {
    
    /**
     * 根据教师ID查询课程列表
     * 
     * @param teacherId 教师ID
     * @return 课程列表
     */
    @Select("SELECT * FROM course WHERE teacher_id = #{teacherId}")
    List<Course> selectByTeacherId(@Param("teacherId") Long teacherId);
    
    /**
     * 根据学期查询课程列表
     * 
     * @param term 学期
     * @return 课程列表
     */
    @Select("SELECT * FROM course WHERE term = #{term}")
    List<Course> selectByTerm(@Param("term") String term);
    
    /**
     * 根据教师ID和学期查询课程列表
     * 
     * @param teacherId 教师ID
     * @param term 学期
     * @return 课程列表
     */
    @Select("SELECT * FROM course WHERE teacher_id = #{teacherId} AND term = #{term}")
    List<Course> selectByTeacherIdAndTerm(@Param("teacherId") Long teacherId, @Param("term") String term);
} 