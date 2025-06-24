package com.education.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.education.entity.Course;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;
import org.apache.ibatis.annotations.Update;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

/**
 * 课程Mapper接口
 * 
 * @author Education Platform Team
 * @version 1.0.0
 */
@Mapper
public interface CourseMapper extends BaseMapper<Course> {

    /**
     * 根据课程代码查询课程信息
     * 
     * @param courseCode 课程代码
     * @return 课程信息
     */
    @Select("SELECT * FROM course WHERE course_code = #{courseCode} AND is_deleted = 0")
    Course selectByCourseCode(@Param("courseCode") String courseCode);

    /**
     * 根据教师ID查询课程列表
     * 
     * @param teacherId 教师ID
     * @return 课程列表
     */
    @Select("SELECT * FROM course WHERE teacher_id = #{teacherId} AND is_deleted = 0 ORDER BY create_time DESC")
    List<Course> selectByTeacherId(@Param("teacherId") Long teacherId);

    /**
     * 根据班级ID查询课程列表
     * 
     * @param classId 班级ID
     * @return 课程列表
     */
    @Select("SELECT * FROM course WHERE class_id = #{classId} AND is_deleted = 0 ORDER BY course_name")
    List<Course> selectByClassId(@Param("classId") Long classId);

    /**
     * 根据课程类型查询课程列表
     * 
     * @param courseType 课程类型
     * @return 课程列表
     */
    @Select("SELECT * FROM course WHERE course_type = #{courseType} AND is_deleted = 0 ORDER BY course_name")
    List<Course> selectByCourseType(@Param("courseType") String courseType);

    /**
     * 根据课程分类查询课程列表
     * 
     * @param category 课程分类
     * @return 课程列表
     */
    @Select("SELECT * FROM course WHERE category = #{category} AND is_deleted = 0 ORDER BY course_name")
    List<Course> selectByCategory(@Param("category") String category);

    /**
     * 根据学期查询课程列表
     * 
     * @param semester 学期
     * @return 课程列表
     */
    @Select("SELECT * FROM course WHERE semester = #{semester} AND is_deleted = 0 ORDER BY course_name")
    List<Course> selectBySemester(@Param("semester") String semester);

    /**
     * 根据课程状态查询课程列表
     * 
     * @param status 课程状态
     * @return 课程列表
     */
    @Select("SELECT * FROM course WHERE status = #{status} AND is_deleted = 0 ORDER BY create_time DESC")
    List<Course> selectByStatus(@Param("status") String status);

    /**
     * 根据难度等级查询课程列表
     * 
     * @param difficulty 难度等级
     * @return 课程列表
     */
    @Select("SELECT * FROM course WHERE difficulty = #{difficulty} AND is_deleted = 0 ORDER BY course_name")
    List<Course> selectByDifficulty(@Param("difficulty") String difficulty);

    /**
     * 查询公开课程
     * 
     * @return 课程列表
     */
    @Select("SELECT * FROM course WHERE is_public = 1 AND status = 'PUBLISHED' AND is_deleted = 0 ORDER BY create_time DESC")
    List<Course> selectPublicCourses();

    /**
     * 查询可选课程（状态为PUBLISHED且未满员）
     * 
     * @return 课程列表
     */
    @Select("SELECT * FROM course " +
            "WHERE status = 'PUBLISHED' AND is_deleted = 0 " +
            "AND (max_students IS NULL OR current_students < max_students) " +
            "ORDER BY course_name")
    List<Course> selectAvailableCourses();

    /**
     * 查询即将开始的课程
     * 
     * @param days 提前几天
     * @return 课程列表
     */
    @Select("SELECT * FROM course " +
            "WHERE status = 'PUBLISHED' AND is_deleted = 0 " +
            "AND start_date <= DATE_ADD(NOW(), INTERVAL #{days} DAY) " +
            "AND start_date >= NOW() " +
            "ORDER BY start_date ASC")
    List<Course> selectCoursesStartingSoon(@Param("days") int days);

    /**
     * 查询即将结束的课程
     * 
     * @param days 提前几天
     * @return 课程列表
     */
    @Select("SELECT * FROM course " +
            "WHERE status = 'PUBLISHED' AND is_deleted = 0 " +
            "AND end_date <= DATE_ADD(NOW(), INTERVAL #{days} DAY) " +
            "AND end_date >= NOW() " +
            "ORDER BY end_date ASC")
    List<Course> selectCoursesEndingSoon(@Param("days") int days);

    /**
     * 查询热门课程（按选课人数排序）
     * 
     * @param limit 限制数量
     * @return 课程列表
     */
    @Select("SELECT * FROM course " +
            "WHERE status = 'PUBLISHED' AND is_deleted = 0 " +
            "ORDER BY current_students DESC " +
            "LIMIT #{limit}")
    List<Course> selectPopularCourses(@Param("limit") int limit);

    /**
     * 统计课程总数
     * 
     * @return 课程总数
     */
    @Select("SELECT COUNT(*) FROM course WHERE is_deleted = 0")
    Long countTotalCourses();

    /**
     * 根据课程类型统计课程数
     * 
     * @return 类型统计结果
     */
    @Select("SELECT course_type, COUNT(*) as count FROM course WHERE is_deleted = 0 GROUP BY course_type ORDER BY count DESC")
    List<Map<String, Object>> countCoursesByType();

    /**
     * 根据课程分类统计课程数
     * 
     * @return 分类统计结果
     */
    @Select("SELECT category, COUNT(*) as count FROM course WHERE is_deleted = 0 GROUP BY category ORDER BY count DESC")
    List<Map<String, Object>> countCoursesByCategory();

    /**
     * 根据状态统计课程数
     * 
     * @return 状态统计结果
     */
    @Select("SELECT status, COUNT(*) as count FROM course WHERE is_deleted = 0 GROUP BY status")
    List<Map<String, Object>> countCoursesByStatus();

    /**
     * 根据难度等级统计课程数
     * 
     * @return 难度统计结果
     */
    @Select("SELECT difficulty, COUNT(*) as count FROM course WHERE is_deleted = 0 GROUP BY difficulty")
    List<Map<String, Object>> countCoursesByDifficulty();

    /**
     * 根据学期统计课程数
     * 
     * @return 学期统计结果
     */
    @Select("SELECT semester, COUNT(*) as count FROM course WHERE is_deleted = 0 GROUP BY semester ORDER BY semester DESC")
    List<Map<String, Object>> countCoursesBySemester();

    /**
     * 搜索课程（模糊查询）
     * 
     * @param keyword 关键词
     * @return 课程列表
     */
    @Select("SELECT c.*, u.real_name as teacher_name, cl.class_name " +
            "FROM course c " +
            "LEFT JOIN user u ON c.teacher_id = u.id " +
            "LEFT JOIN class cl ON c.class_id = cl.id " +
            "WHERE c.is_deleted = 0 " +
            "AND (c.course_name LIKE CONCAT('%', #{keyword}, '%') " +
            "OR c.course_code LIKE CONCAT('%', #{keyword}, '%') " +
            "OR c.description LIKE CONCAT('%', #{keyword}, '%') " +
            "OR c.category LIKE CONCAT('%', #{keyword}, '%') " +
            "OR u.real_name LIKE CONCAT('%', #{keyword}, '%')) " +
            "ORDER BY c.create_time DESC")
    List<Map<String, Object>> searchCourses(@Param("keyword") String keyword);

    /**
     * 检查课程代码是否存在
     * 
     * @param courseCode 课程代码
     * @param excludeCourseId 排除的课程ID
     * @return 是否存在
     */
    @Select("SELECT COUNT(*) FROM course " +
            "WHERE course_code = #{courseCode} AND is_deleted = 0 " +
            "AND (#{excludeCourseId} IS NULL OR id != #{excludeCourseId})")
    int checkCourseCodeExists(@Param("courseCode") String courseCode, @Param("excludeCourseId") Long excludeCourseId);

    /**
     * 更新课程选课人数
     * 
     * @param courseId 课程ID
     * @param currentStudents 当前选课人数
     * @return 更新结果
     */
    @Update("UPDATE course SET current_students = #{currentStudents}, update_time = NOW() WHERE id = #{courseId} AND is_deleted = 0")
    int updateCurrentStudents(@Param("courseId") Long courseId, @Param("currentStudents") Integer currentStudents);

    /**
     * 增加课程选课人数
     * 
     * @param courseId 课程ID
     * @param increment 增加数量
     * @return 更新结果
     */
    @Update("UPDATE course SET current_students = current_students + #{increment}, update_time = NOW() WHERE id = #{courseId} AND is_deleted = 0")
    int incrementCurrentStudents(@Param("courseId") Long courseId, @Param("increment") Integer increment);

    /**
     * 减少课程选课人数
     * 
     * @param courseId 课程ID
     * @param decrement 减少数量
     * @return 更新结果
     */
    @Update("UPDATE course SET current_students = GREATEST(0, current_students - #{decrement}), update_time = NOW() WHERE id = #{courseId} AND is_deleted = 0")
    int decrementCurrentStudents(@Param("courseId") Long courseId, @Param("decrement") Integer decrement);

    /**
     * 更新课程状态
     * 
     * @param courseId 课程ID
     * @param status 课程状态
     * @return 更新结果
     */
    @Update("UPDATE course SET status = #{status}, update_time = NOW() WHERE id = #{courseId} AND is_deleted = 0")
    int updateStatus(@Param("courseId") Long courseId, @Param("status") String status);

    /**
     * 批量更新课程状态
     * 
     * @param courseIds 课程ID列表
     * @param status 课程状态
     * @return 更新结果
     */
    int batchUpdateStatus(@Param("courseIds") List<Long> courseIds, @Param("status") String status);

    /**
     * 更新课程教师
     * 
     * @param courseId 课程ID
     * @param teacherId 教师ID
     * @return 更新结果
     */
    @Update("UPDATE course SET teacher_id = #{teacherId}, update_time = NOW() WHERE id = #{courseId} AND is_deleted = 0")
    int updateTeacher(@Param("courseId") Long courseId, @Param("teacherId") Long teacherId);

    /**
     * 获取课程详细统计信息
     * 
     * @return 统计信息
     */
    @Select("SELECT " +
            "COUNT(*) as total_courses, " +
            "SUM(CASE WHEN status = 'PUBLISHED' THEN 1 ELSE 0 END) as published_courses, " +
            "SUM(CASE WHEN status = 'DRAFT' THEN 1 ELSE 0 END) as draft_courses, " +
            "SUM(CASE WHEN status = 'ARCHIVED' THEN 1 ELSE 0 END) as archived_courses, " +
            "SUM(CASE WHEN is_public = 1 THEN 1 ELSE 0 END) as public_courses, " +
            "SUM(current_students) as total_enrollments, " +
            "AVG(current_students) as average_enrollments_per_course, " +
            "MAX(current_students) as max_enrollments_in_course, " +
            "MIN(current_students) as min_enrollments_in_course, " +
            "AVG(credits) as average_credits, " +
            "COUNT(DISTINCT teacher_id) as teacher_count, " +
            "COUNT(DISTINCT category) as category_count " +
            "FROM course WHERE is_deleted = 0")
    Map<String, Object> getCourseStatistics();

    /**
     * 获取课程的选课学生列表
     * 
     * @param courseId 课程ID
     * @return 学生列表
     */
    @Select("SELECT s.*, u.username, u.real_name, u.email, u.phone, sc.enrollment_date, sc.status as enrollment_status " +
            "FROM student_course sc " +
            "JOIN student s ON sc.student_id = s.id " +
            "JOIN user u ON s.user_id = u.id " +
            "WHERE sc.course_id = #{courseId} AND sc.is_deleted = 0 " +
            "ORDER BY sc.enrollment_date DESC")
    List<Map<String, Object>> getCourseStudents(@Param("courseId") Long courseId);

    /**
     * 获取课程的任务列表
     * 
     * @param courseId 课程ID
     * @return 任务列表
     */
    @Select("SELECT t.*, u.real_name as teacher_name " +
            "FROM task t " +
            "LEFT JOIN user u ON t.teacher_id = u.id " +
            "WHERE t.course_id = #{courseId} AND t.is_deleted = 0 " +
            "ORDER BY t.create_time DESC")
    List<Map<String, Object>> getCourseTasks(@Param("courseId") Long courseId);

    /**
     * 获取课程的资源列表
     * 
     * @param courseId 课程ID
     * @return 资源列表
     */
    @Select("SELECT r.*, u.real_name as uploader_name " +
            "FROM resource r " +
            "LEFT JOIN user u ON r.uploader_id = u.id " +
            "WHERE r.course_id = #{courseId} AND r.is_deleted = 0 " +
            "ORDER BY r.create_time DESC")
    List<Map<String, Object>> getCourseResources(@Param("courseId") Long courseId);

    /**
     * 获取课程选课率统计
     * 
     * @return 选课率统计
     */
    @Select("SELECT " +
            "id, course_name, course_code, " +
            "current_students, max_students, " +
            "ROUND((current_students * 100.0 / NULLIF(max_students, 0)), 2) as enrollment_rate " +
            "FROM course " +
            "WHERE is_deleted = 0 AND max_students > 0 " +
            "ORDER BY enrollment_rate DESC")
    List<Map<String, Object>> getCourseEnrollmentRates();

    /**
     * 查询满员的课程
     * 
     * @return 课程列表
     */
    @Select("SELECT * FROM course " +
            "WHERE is_deleted = 0 AND max_students > 0 " +
            "AND current_students >= max_students " +
            "ORDER BY current_students DESC")
    List<Course> selectFullCourses();

    /**
     * 根据时间范围查询课程
     * 
     * @param startTime 开始时间
     * @param endTime 结束时间
     * @return 课程列表
     */
    @Select("SELECT * FROM course " +
            "WHERE is_deleted = 0 " +
            "AND ((start_date <= #{endTime} AND end_date >= #{startTime}) " +
            "OR (start_date >= #{startTime} AND start_date <= #{endTime})) " +
            "ORDER BY start_date ASC")
    List<Course> selectCoursesByTimeRange(@Param("startTime") LocalDateTime startTime, @Param("endTime") LocalDateTime endTime);

    /**
     * 根据先修课程查询课程
     * 
     * @param prerequisiteCourseId 先修课程ID
     * @return 课程列表
     */
    @Select("SELECT * FROM course " +
            "WHERE is_deleted = 0 " +
            "AND FIND_IN_SET(#{prerequisiteCourseId}, prerequisites) > 0 " +
            "ORDER BY course_name")
    List<Course> selectCoursesByPrerequisite(@Param("prerequisiteCourseId") Long prerequisiteCourseId);

    /**
     * 根据标签查询课程
     * 
     * @param tag 标签
     * @return 课程列表
     */
    @Select("SELECT * FROM course " +
            "WHERE is_deleted = 0 " +
            "AND FIND_IN_SET(#{tag}, tags) > 0 " +
            "ORDER BY course_name")
    List<Course> selectCoursesByTag(@Param("tag") String tag);

    /**
     * 软删除课程
     * 
     * @param courseId 课程ID
     * @return 删除结果
     */
    @Update("UPDATE course SET is_deleted = 1, update_time = NOW() WHERE id = #{courseId}")
    int softDeleteCourse(@Param("courseId") Long courseId);

    /**
     * 批量软删除课程
     * 
     * @param courseIds 课程ID列表
     * @return 删除结果
     */
    int batchSoftDeleteCourses(@Param("courseIds") List<Long> courseIds);

    /**
     * 恢复已删除的课程
     * 
     * @param courseId 课程ID
     * @return 恢复结果
     */
    @Update("UPDATE course SET is_deleted = 0, update_time = NOW() WHERE id = #{courseId}")
    int restoreCourse(@Param("courseId") Long courseId);

    /**
     * 更新课程扩展字段
     * 
     * @param courseId 课程ID
     * @param extField1 扩展字段1
     * @param extField2 扩展字段2
     * @param extField3 扩展字段3
     * @return 更新结果
     */
    @Update("UPDATE course SET ext_field1 = #{extField1}, ext_field2 = #{extField2}, ext_field3 = #{extField3}, update_time = NOW() " +
            "WHERE id = #{courseId} AND is_deleted = 0")
    int updateExtFields(@Param("courseId") Long courseId, 
                       @Param("extField1") String extField1, 
                       @Param("extField2") String extField2, 
                       @Param("extField3") String extField3);

    /**
     * 获取课程进度信息
     * 
     * @param courseId 课程ID
     * @return 进度信息
     */
    @Select("SELECT " +
            "id, course_name, start_date, end_date, status, " +
            "DATEDIFF(NOW(), start_date) as running_days, " +
            "DATEDIFF(end_date, start_date) as total_days, " +
            "CASE " +
            "WHEN start_date > NOW() THEN 0 " +
            "WHEN end_date < NOW() THEN 100 " +
            "ELSE ROUND((DATEDIFF(NOW(), start_date) * 100.0 / NULLIF(DATEDIFF(end_date, start_date), 0)), 2) " +
            "END as progress_percentage " +
            "FROM course " +
            "WHERE id = #{courseId} AND is_deleted = 0")
    Map<String, Object> getCourseProgress(@Param("courseId") Long courseId);
}