package com.education.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.education.entity.Teacher;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

import java.util.List;
import java.util.Map;

/**
 * 教师数据访问层
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
@Mapper
public interface TeacherMapper extends BaseMapper<Teacher> {
    
    /**
     * 根据用户ID查询教师信息
     */
    Teacher selectByUserId(@Param("userId") Long userId);
    
    /**
     * 根据工号查询教师信息
     */
    Teacher selectByEmployeeId(@Param("employeeId") String employeeId);
    
    /**
     * 根据学院查询教师列表
     */
    List<Teacher> selectByCollege(@Param("college") String college);
    
    /**
     * 根据部门查询教师列表
     */
    List<Teacher> selectByDepartment(@Param("department") String department);
    
    /**
     * 根据职称查询教师列表
     */
    List<Teacher> selectByTitle(@Param("title") String title);
    
    /**
     * 搜索教师
     */
    List<Teacher> searchTeachers(@Param("keyword") String keyword);
    
    /**
     * 查询教师统计信息
     */
    Map<String, Object> selectTeacherStats(@Param("teacherId") Long teacherId);
    
    /**
     * 查询教师课程统计
     */
    Map<String, Object> selectCourseStats(@Param("teacherId") Long teacherId);
    
    /**
     * 查询教师学生统计
     */
    Map<String, Object> selectStudentStats(@Param("teacherId") Long teacherId);
    
    /**
     * 查询教师任务统计
     */
    Map<String, Object> selectTaskStats(@Param("teacherId") Long teacherId);
    
    /**
     * 查询教师评价统计
     */
    Map<String, Object> selectRatingStats(@Param("teacherId") Long teacherId);
    
    /**
     * 查询优秀教师列表
     */
    List<Teacher> selectExcellentTeachers(@Param("limit") Integer limit);
    
    /**
     * 查询活跃教师列表
     */
    List<Teacher> selectActiveTeachers(@Param("days") Integer days, @Param("limit") Integer limit);
    
    /**
     * 批量更新教师状态
     */
    int batchUpdateStatus(@Param("teacherIds") List<Long> teacherIds, @Param("status") String status);
}