package com.education.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.education.entity.Class;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;
import org.apache.ibatis.annotations.Update;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

/**
 * 班级Mapper接口
 * 
 * @author Education Platform Team
 * @version 1.0.0
 */
@Mapper
public interface ClassMapper extends BaseMapper<Class> {

    /**
     * 根据班级代码查询班级信息
     * 
     * @param classCode 班级代码
     * @return 班级信息
     */
    @Select("SELECT * FROM class WHERE class_code = #{classCode} AND is_deleted = 0")
    Class selectByClassCode(@Param("classCode") String classCode);

    /**
     * 根据班主任ID查询班级列表
     * 
     * @param teacherId 班主任ID
     * @return 班级列表
     */
    @Select("SELECT * FROM class WHERE teacher_id = #{teacherId} AND is_deleted = 0 ORDER BY create_time DESC")
    List<Class> selectByTeacherId(@Param("teacherId") Long teacherId);

    /**
     * 根据专业查询班级列表
     * 
     * @param major 专业
     * @return 班级列表
     */
    @Select("SELECT * FROM class WHERE major = #{major} AND is_deleted = 0 ORDER BY grade DESC, class_name")
    List<Class> selectByMajor(@Param("major") String major);

    /**
     * 根据年级查询班级列表
     * 
     * @param grade 年级
     * @return 班级列表
     */
    @Select("SELECT * FROM class WHERE grade = #{grade} AND is_deleted = 0 ORDER BY major, class_name")
    List<Class> selectByGrade(@Param("grade") Integer grade);

    /**
     * 根据学期查询班级列表
     * 
     * @param semester 学期
     * @return 班级列表
     */
    @Select("SELECT * FROM class WHERE semester = #{semester} AND is_deleted = 0 ORDER BY grade DESC, class_name")
    List<Class> selectBySemester(@Param("semester") String semester);

    /**
     * 根据班级状态查询班级列表
     * 
     * @param status 班级状态
     * @return 班级列表
     */
    @Select("SELECT * FROM class WHERE status = #{status} AND is_deleted = 0 ORDER BY create_time DESC")
    List<Class> selectByStatus(@Param("status") String status);

    /**
     * 查询活跃班级（状态为ACTIVE）
     * 
     * @return 班级列表
     */
    @Select("SELECT * FROM class WHERE status = 'ACTIVE' AND is_deleted = 0 ORDER BY grade DESC, class_name")
    List<Class> selectActiveClasses();

    /**
     * 查询即将开班的班级
     * 
     * @param days 提前几天
     * @return 班级列表
     */
    @Select("SELECT * FROM class " +
            "WHERE status = 'PREPARING' AND is_deleted = 0 " +
            "AND start_date <= DATE_ADD(NOW(), INTERVAL #{days} DAY) " +
            "AND start_date >= NOW() " +
            "ORDER BY start_date ASC")
    List<Class> selectClassesStartingSoon(@Param("days") int days);

    /**
     * 查询即将结班的班级
     * 
     * @param days 提前几天
     * @return 班级列表
     */
    @Select("SELECT * FROM class " +
            "WHERE status = 'ACTIVE' AND is_deleted = 0 " +
            "AND end_date <= DATE_ADD(NOW(), INTERVAL #{days} DAY) " +
            "AND end_date >= NOW() " +
            "ORDER BY end_date ASC")
    List<Class> selectClassesEndingSoon(@Param("days") int days);

    /**
     * 统计班级总数
     * 
     * @return 班级总数
     */
    @Select("SELECT COUNT(*) FROM class WHERE is_deleted = 0")
    Long countTotalClasses();

    /**
     * 根据专业统计班级数
     * 
     * @return 专业统计结果
     */
    @Select("SELECT major, COUNT(*) as count FROM class WHERE is_deleted = 0 GROUP BY major ORDER BY count DESC")
    List<Map<String, Object>> countClassesByMajor();

    /**
     * 根据年级统计班级数
     * 
     * @return 年级统计结果
     */
    @Select("SELECT grade, COUNT(*) as count FROM class WHERE is_deleted = 0 GROUP BY grade ORDER BY grade DESC")
    List<Map<String, Object>> countClassesByGrade();

    /**
     * 根据状态统计班级数
     * 
     * @return 状态统计结果
     */
    @Select("SELECT status, COUNT(*) as count FROM class WHERE is_deleted = 0 GROUP BY status")
    List<Map<String, Object>> countClassesByStatus();

    /**
     * 根据学期统计班级数
     * 
     * @return 学期统计结果
     */
    @Select("SELECT semester, COUNT(*) as count FROM class WHERE is_deleted = 0 GROUP BY semester ORDER BY semester DESC")
    List<Map<String, Object>> countClassesBySemester();

    /**
     * 搜索班级（模糊查询）
     * 
     * @param keyword 关键词
     * @return 班级列表
     */
    @Select("SELECT c.*, u.real_name as teacher_name " +
            "FROM class c " +
            "LEFT JOIN user u ON c.teacher_id = u.id " +
            "WHERE c.is_deleted = 0 " +
            "AND (c.class_name LIKE CONCAT('%', #{keyword}, '%') " +
            "OR c.class_code LIKE CONCAT('%', #{keyword}, '%') " +
            "OR c.major LIKE CONCAT('%', #{keyword}, '%') " +
            "OR c.description LIKE CONCAT('%', #{keyword}, '%') " +
            "OR u.real_name LIKE CONCAT('%', #{keyword}, '%')) " +
            "ORDER BY c.grade DESC, c.class_name")
    List<Map<String, Object>> searchClasses(@Param("keyword") String keyword);

    /**
     * 检查班级代码是否存在
     * 
     * @param classCode 班级代码
     * @param excludeClassId 排除的班级ID
     * @return 是否存在
     */
    @Select("SELECT COUNT(*) FROM class " +
            "WHERE class_code = #{classCode} AND is_deleted = 0 " +
            "AND (#{excludeClassId} IS NULL OR id != #{excludeClassId})")
    int checkClassCodeExists(@Param("classCode") String classCode, @Param("excludeClassId") Long excludeClassId);

    /**
     * 更新班级学生人数
     * 
     * @param classId 班级ID
     * @param studentCount 学生人数
     * @return 更新结果
     */
    @Update("UPDATE class SET student_count = #{studentCount}, update_time = NOW() WHERE id = #{classId} AND is_deleted = 0")
    int updateStudentCount(@Param("classId") Long classId, @Param("studentCount") Integer studentCount);

    /**
     * 增加班级学生人数
     * 
     * @param classId 班级ID
     * @param increment 增加数量
     * @return 更新结果
     */
    @Update("UPDATE class SET student_count = student_count + #{increment}, update_time = NOW() WHERE id = #{classId} AND is_deleted = 0")
    int incrementStudentCount(@Param("classId") Long classId, @Param("increment") Integer increment);

    /**
     * 减少班级学生人数
     * 
     * @param classId 班级ID
     * @param decrement 减少数量
     * @return 更新结果
     */
    @Update("UPDATE class SET student_count = GREATEST(0, student_count - #{decrement}), update_time = NOW() WHERE id = #{classId} AND is_deleted = 0")
    int decrementStudentCount(@Param("classId") Long classId, @Param("decrement") Integer decrement);

    /**
     * 更新班级状态
     * 
     * @param classId 班级ID
     * @param status 班级状态
     * @return 更新结果
     */
    @Update("UPDATE class SET status = #{status}, update_time = NOW() WHERE id = #{classId} AND is_deleted = 0")
    int updateStatus(@Param("classId") Long classId, @Param("status") String status);

    /**
     * 批量更新班级状态
     * 
     * @param classIds 班级ID列表
     * @param status 班级状态
     * @return 更新结果
     */
    int batchUpdateStatus(@Param("classIds") List<Long> classIds, @Param("status") String status);

    /**
     * 更新班级班主任
     * 
     * @param classId 班级ID
     * @param teacherId 班主任ID
     * @return 更新结果
     */
    @Update("UPDATE class SET teacher_id = #{teacherId}, update_time = NOW() WHERE id = #{classId} AND is_deleted = 0")
    int updateTeacher(@Param("classId") Long classId, @Param("teacherId") Long teacherId);

    /**
     * 获取班级详细统计信息
     * 
     * @return 统计信息
     */
    @Select("SELECT " +
            "COUNT(*) as total_classes, " +
            "SUM(CASE WHEN status = 'ACTIVE' THEN 1 ELSE 0 END) as active_classes, " +
            "SUM(CASE WHEN status = 'PREPARING' THEN 1 ELSE 0 END) as preparing_classes, " +
            "SUM(CASE WHEN status = 'COMPLETED' THEN 1 ELSE 0 END) as completed_classes, " +
            "SUM(CASE WHEN status = 'SUSPENDED' THEN 1 ELSE 0 END) as suspended_classes, " +
            "SUM(student_count) as total_students, " +
            "AVG(student_count) as average_students_per_class, " +
            "MAX(student_count) as max_students_in_class, " +
            "MIN(student_count) as min_students_in_class, " +
            "COUNT(DISTINCT major) as major_count, " +
            "COUNT(DISTINCT grade) as grade_count, " +
            "COUNT(DISTINCT teacher_id) as teacher_count " +
            "FROM class WHERE is_deleted = 0")
    Map<String, Object> getClassStatistics();

    /**
     * 获取班级的学生列表
     * 
     * @param classId 班级ID
     * @return 学生列表
     */
    @Select("SELECT s.*, u.username, u.real_name, u.email, u.phone " +
            "FROM student s " +
            "JOIN user u ON s.user_id = u.id " +
            "WHERE s.class_id = #{classId} AND s.is_deleted = 0 " +
            "ORDER BY s.student_number")
    List<Map<String, Object>> getClassStudents(@Param("classId") Long classId);

    /**
     * 获取班级的课程列表
     * 
     * @param classId 班级ID
     * @return 课程列表
     */
    @Select("SELECT c.*, u.real_name as teacher_name " +
            "FROM course c " +
            "LEFT JOIN user u ON c.teacher_id = u.id " +
            "WHERE c.class_id = #{classId} AND c.is_deleted = 0 " +
            "ORDER BY c.create_time DESC")
    List<Map<String, Object>> getClassCourses(@Param("classId") Long classId);

    /**
     * 获取班级容量使用率统计
     * 
     * @return 容量使用率统计
     */
    @Select("SELECT " +
            "id, class_name, class_code, " +
            "student_count, max_capacity, " +
            "ROUND((student_count * 100.0 / NULLIF(max_capacity, 0)), 2) as usage_rate " +
            "FROM class " +
            "WHERE is_deleted = 0 AND max_capacity > 0 " +
            "ORDER BY usage_rate DESC")
    List<Map<String, Object>> getClassCapacityUsage();

    /**
     * 查询超员的班级
     * 
     * @return 班级列表
     */
    @Select("SELECT * FROM class " +
            "WHERE is_deleted = 0 AND max_capacity > 0 " +
            "AND student_count > max_capacity " +
            "ORDER BY (student_count - max_capacity) DESC")
    List<Class> selectOvercrowdedClasses();

    /**
     * 查询可以添加学生的班级
     * 
     * @return 班级列表
     */
    @Select("SELECT * FROM class " +
            "WHERE is_deleted = 0 AND status = 'ACTIVE' " +
            "AND (max_capacity IS NULL OR student_count < max_capacity) " +
            "ORDER BY grade DESC, class_name")
    List<Class> selectAvailableClasses();

    /**
     * 根据时间范围查询班级
     * 
     * @param startTime 开始时间
     * @param endTime 结束时间
     * @return 班级列表
     */
    @Select("SELECT * FROM class " +
            "WHERE is_deleted = 0 " +
            "AND ((start_date <= #{endTime} AND end_date >= #{startTime}) " +
            "OR (start_date >= #{startTime} AND start_date <= #{endTime})) " +
            "ORDER BY start_date ASC")
    List<Class> selectClassesByTimeRange(@Param("startTime") LocalDateTime startTime, @Param("endTime") LocalDateTime endTime);

    /**
     * 软删除班级
     * 
     * @param classId 班级ID
     * @return 删除结果
     */
    @Update("UPDATE class SET is_deleted = 1, update_time = NOW() WHERE id = #{classId}")
    int softDeleteClass(@Param("classId") Long classId);

    /**
     * 批量软删除班级
     * 
     * @param classIds 班级ID列表
     * @return 删除结果
     */
    int batchSoftDeleteClasses(@Param("classIds") List<Long> classIds);

    /**
     * 恢复已删除的班级
     * 
     * @param classId 班级ID
     * @return 恢复结果
     */
    @Update("UPDATE class SET is_deleted = 0, update_time = NOW() WHERE id = #{classId}")
    int restoreClass(@Param("classId") Long classId);

    /**
     * 更新班级扩展字段
     * 
     * @param classId 班级ID
     * @param extField1 扩展字段1
     * @param extField2 扩展字段2
     * @param extField3 扩展字段3
     * @return 更新结果
     */
    @Update("UPDATE class SET ext_field1 = #{extField1}, ext_field2 = #{extField2}, ext_field3 = #{extField3}, update_time = NOW() " +
            "WHERE id = #{classId} AND is_deleted = 0")
    int updateExtFields(@Param("classId") Long classId, 
                       @Param("extField1") String extField1, 
                       @Param("extField2") String extField2, 
                       @Param("extField3") String extField3);

    /**
     * 获取班级进度信息
     * 
     * @param classId 班级ID
     * @return 进度信息
     */
    @Select("SELECT " +
            "id, class_name, start_date, end_date, status, " +
            "DATEDIFF(NOW(), start_date) as running_days, " +
            "DATEDIFF(end_date, start_date) as total_days, " +
            "CASE " +
            "WHEN start_date > NOW() THEN 0 " +
            "WHEN end_date < NOW() THEN 100 " +
            "ELSE ROUND((DATEDIFF(NOW(), start_date) * 100.0 / NULLIF(DATEDIFF(end_date, start_date), 0)), 2) " +
            "END as progress_percentage " +
            "FROM class " +
            "WHERE id = #{classId} AND is_deleted = 0")
    Map<String, Object> getClassProgress(@Param("classId") Long classId);
}