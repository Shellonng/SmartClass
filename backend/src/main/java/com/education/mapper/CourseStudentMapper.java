package com.education.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.education.entity.CourseStudent;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

/**
 * 课程学生关联Mapper接口
 */
@Mapper
public interface CourseStudentMapper extends BaseMapper<CourseStudent> {

    /**
     * 统计课程学生数量
     *
     * @param courseId 课程ID
     * @return 学生数量
     */
    @Select("SELECT COUNT(*) FROM course_student WHERE course_id = #{courseId}")
    Integer countByCourseId(@Param("courseId") Long courseId);
} 