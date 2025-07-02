package com.education.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.education.entity.Teacher;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

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
    @Select("SELECT * FROM teacher WHERE user_id = #{userId}")
    Teacher selectByUserId(@Param("userId") Long userId);
    
    /**
     * 根据用户名查询教师信息
     */
    @Select("SELECT t.* FROM teacher t JOIN user u ON t.user_id = u.id WHERE u.username = #{username}")
    Teacher selectByUsername(@Param("username") String username);
    
    /**
     * 根据部门查询教师列表
     */
    @Select("SELECT * FROM teacher WHERE department = #{department}")
    List<Teacher> selectByDepartment(@Param("department") String department);
    
    /**
     * 根据职称查询教师列表
     */
    @Select("SELECT * FROM teacher WHERE title = #{title}")
    List<Teacher> selectByTitle(@Param("title") String title);
    
    /**
     * 搜索教师
     */
    @Select("SELECT t.* FROM teacher t " +
            "LEFT JOIN user u ON t.user_id = u.id " +
            "WHERE u.real_name LIKE CONCAT('%', #{keyword}, '%') " +
            "OR t.department LIKE CONCAT('%', #{keyword}, '%') " +
            "OR t.title LIKE CONCAT('%', #{keyword}, '%')")
    List<Teacher> searchTeachers(@Param("keyword") String keyword);
    
    /**
     * 查询教师课程统计
     */
    @Select("SELECT " +
            "COUNT(*) as total_courses, " +
            "SUM(CASE WHEN c.status = '已发布' THEN 1 ELSE 0 END) as published_courses, " +
            "SUM(CASE WHEN c.status = '草稿' THEN 1 ELSE 0 END) as draft_courses, " +
            "SUM(c.student_count) as total_students " +
            "FROM course c WHERE c.teacher_id = #{teacherId}")
    Map<String, Object> selectCourseStats(@Param("teacherId") Long teacherId);
    
    /**
     * 查询教师学生统计
     */
    @Select("SELECT " +
            "COUNT(DISTINCT sc.student_id) as total_students, " +
            "COUNT(DISTINCT CASE WHEN sc.status = 'ENROLLED' THEN sc.student_id END) as active_students, " +
            "COUNT(DISTINCT CASE WHEN sc.status = 'COMPLETED' THEN sc.student_id END) as completed_students " +
            "FROM student_course sc " +
            "JOIN course c ON sc.course_id = c.id " +
            "WHERE c.teacher_id = #{teacherId}")
    Map<String, Object> selectStudentStats(@Param("teacherId") Long teacherId);
    
    /**
     * 查询教师评价统计
     */
    @Select("SELECT " +
            "COUNT(*) as total_reviews, " +
            "AVG(r.rating) as average_rating, " +
            "MAX(r.rating) as max_rating, " +
            "MIN(r.rating) as min_rating " +
            "FROM review r " +
            "JOIN course c ON r.course_id = c.id " +
            "WHERE c.teacher_id = #{teacherId}")
    Map<String, Object> selectRatingStats(@Param("teacherId") Long teacherId);
    
    /**
     * 查询优秀教师列表
     */
    @Select("SELECT t.*, AVG(r.rating) as avg_rating " +
            "FROM teacher t " +
            "LEFT JOIN user u ON t.user_id = u.id " +
            "LEFT JOIN course c ON t.id = c.teacher_id " +
            "LEFT JOIN review r ON c.id = r.course_id " +
            "GROUP BY t.id " +
            "ORDER BY avg_rating DESC " +
            "LIMIT #{limit}")
    List<Teacher> selectExcellentTeachers(@Param("limit") Integer limit);
    
    /**
     * 查询活跃教师列表
     */
    @Select("SELECT DISTINCT t.* " +
            "FROM teacher t " +
            "LEFT JOIN user u ON t.user_id = u.id " +
            "LEFT JOIN course c ON t.id = c.teacher_id " +
            "LEFT JOIN task ta ON c.id = ta.course_id " +
            "WHERE ta.created_time >= DATE_SUB(NOW(), INTERVAL #{days} DAY) " +
            "ORDER BY ta.created_time DESC " +
            "LIMIT #{limit}")
    List<Teacher> selectActiveTeachers(@Param("days") Integer days, @Param("limit") Integer limit);
    
    /**
     * 批量更新教师状态
     */
    int batchUpdateStatus(@Param("teacherIds") List<Long> teacherIds, @Param("status") String status);
    
    /**
     * 获取指定年份的最大教师工号
     * 
     * @param year 年份
     * @return 最大教师工号
     */
    @Select("SELECT MAX(teacher_id) FROM teacher WHERE teacher_id LIKE CONCAT('T', #{year}, '%')")
    String getMaxTeacherIdByYear(@Param("year") String year);
}