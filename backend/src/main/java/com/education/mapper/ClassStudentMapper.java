package com.education.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.education.entity.ClassStudent;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

/**
 * 班级学生关联数据访问层
 */
@Mapper
public interface ClassStudentMapper extends BaseMapper<ClassStudent> {

    /**
     * 统计班级学生数量
     * 
     * @param classId 班级ID
     * @return 学生数量
     */
    @Select("SELECT COUNT(1) FROM class_student WHERE class_id = #{classId}")
    Integer countByClassId(@Param("classId") Long classId);
    
    /**
     * 检查学生是否在班级中
     * 
     * @param classId 班级ID
     * @param studentId 学生ID
     * @return 是否存在
     */
    @Select("SELECT COUNT(1) FROM class_student WHERE class_id = #{classId} AND student_id = #{studentId}")
    Integer checkStudentInClass(@Param("classId") Long classId, @Param("studentId") Long studentId);
} 