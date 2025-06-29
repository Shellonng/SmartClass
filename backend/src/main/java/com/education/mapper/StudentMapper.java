package com.education.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.education.entity.Student;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;
import org.apache.ibatis.annotations.Update;

import java.util.List;
import java.util.Map;

/**
 * 学生Mapper接口
 * 
 * @author Education Platform Team
 * @version 1.0.0
 */
@Mapper
public interface StudentMapper extends BaseMapper<Student> {

    /**
     * 根据用户ID查询学生信息
     * 
     * @param userId 用户ID
     * @return 学生信息
     */
    @Select("SELECT * FROM student WHERE user_id = #{userId} AND is_deleted = 0")
    Student selectByUserId(@Param("userId") Long userId);

    /**
     * 根据学号查询学生信息
     * 
     * @param studentNumber 学号
     * @return 学生信息
     */
    @Select("SELECT * FROM student WHERE student_number = #{studentNumber} AND is_deleted = 0")
    Student selectByStudentNumber(@Param("studentNumber") String studentNumber);

    /**
     * 根据班级ID查询学生列表
     * 
     * @param classId 班级ID
     * @return 学生列表
     */
    @Select("SELECT * FROM student WHERE class_id = #{classId} AND is_deleted = 0 ORDER BY student_number")
    List<Student> selectByClassId(@Param("classId") Long classId);

    /**
     * 根据专业查询学生列表
     * 
     * @param major 专业
     * @return 学生列表
     */
    @Select("SELECT * FROM student WHERE major = #{major} AND is_deleted = 0 ORDER BY grade DESC, student_number")
    List<Student> selectByMajor(@Param("major") String major);

    /**
     * 根据年级查询学生列表
     * 
     * @param grade 年级
     * @return 学生列表
     */
    @Select("SELECT * FROM student WHERE grade = #{grade} AND is_deleted = 0 ORDER BY major, student_number")
    List<Student> selectByGrade(@Param("grade") Integer grade);

    /**
     * 根据学籍状态查询学生列表
     * 
     * @param status 学籍状态
     * @return 学生列表
     */
    @Select("SELECT * FROM student WHERE status = #{status} AND is_deleted = 0 ORDER BY grade DESC, student_number")
    List<Student> selectByStatus(@Param("status") String status);

    /**
     * 根据导师ID查询学生列表
     * 
     * @param mentorId 导师ID
     * @return 学生列表
     */
    @Select("SELECT * FROM student WHERE mentor_id = #{mentorId} AND is_deleted = 0 ORDER BY grade DESC, student_number")
    List<Student> selectByMentorId(@Param("mentorId") Long mentorId);

    /**
     * 查询优秀学生（GPA >= 3.5）
     * 
     * @param minGpa 最低GPA
     * @return 学生列表
     */
    @Select("SELECT * FROM student WHERE gpa >= #{minGpa} AND is_deleted = 0 ORDER BY gpa DESC, student_number")
    List<Student> selectExcellentStudents(@Param("minGpa") Double minGpa);

    /**
     * 查询需要学业预警的学生（GPA < 2.0）
     * 
     * @param maxGpa 最高GPA
     * @return 学生列表
     */
    @Select("SELECT * FROM student WHERE gpa < #{maxGpa} AND status = 'ENROLLED' AND is_deleted = 0 ORDER BY gpa ASC, student_number")
    List<Student> selectStudentsNeedingWarning(@Param("maxGpa") Double maxGpa);

    /**
     * 统计学生总数
     * 
     * @return 学生总数
     */
    @Select("SELECT COUNT(*) FROM student WHERE is_deleted = 0")
    Long countTotalStudents();

    /**
     * 根据专业统计学生数
     * 
     * @return 专业统计结果
     */
    @Select("SELECT major, COUNT(*) as count FROM student WHERE is_deleted = 0 GROUP BY major ORDER BY count DESC")
    List<Map<String, Object>> countStudentsByMajor();

    /**
     * 根据年级统计学生数
     * 
     * @return 年级统计结果
     */
    @Select("SELECT grade, COUNT(*) as count FROM student WHERE is_deleted = 0 GROUP BY grade ORDER BY grade DESC")
    List<Map<String, Object>> countStudentsByGrade();

    /**
     * 根据学籍状态统计学生数
     * 
     * @return 状态统计结果
     */
    @Select("SELECT status, COUNT(*) as count FROM student WHERE is_deleted = 0 GROUP BY status")
    List<Map<String, Object>> countStudentsByStatus();

    /**
     * 根据学生类型统计学生数
     * 
     * @return 类型统计结果
     */
    @Select("SELECT student_type, COUNT(*) as count FROM student WHERE is_deleted = 0 GROUP BY student_type")
    List<Map<String, Object>> countStudentsByType();

    /**
     * 获取GPA分布统计
     * 
     * @return GPA分布
     */
    @Select("SELECT " +
            "CASE " +
            "WHEN gpa >= 3.7 THEN 'A (3.7-4.0)' " +
            "WHEN gpa >= 3.3 THEN 'B+ (3.3-3.6)' " +
            "WHEN gpa >= 3.0 THEN 'B (3.0-3.2)' " +
            "WHEN gpa >= 2.7 THEN 'B- (2.7-2.9)' " +
            "WHEN gpa >= 2.3 THEN 'C+ (2.3-2.6)' " +
            "WHEN gpa >= 2.0 THEN 'C (2.0-2.2)' " +
            "WHEN gpa >= 1.7 THEN 'C- (1.7-1.9)' " +
            "WHEN gpa >= 1.3 THEN 'D+ (1.3-1.6)' " +
            "WHEN gpa >= 1.0 THEN 'D (1.0-1.2)' " +
            "ELSE 'F (0.0-0.9)' " +
            "END as gpa_range, " +
            "COUNT(*) as count " +
            "FROM student " +
            "WHERE is_deleted = 0 AND gpa IS NOT NULL " +
            "GROUP BY " +
            "CASE " +
            "WHEN gpa >= 3.7 THEN 'A (3.7-4.0)' " +
            "WHEN gpa >= 3.3 THEN 'B+ (3.3-3.6)' " +
            "WHEN gpa >= 3.0 THEN 'B (3.0-3.2)' " +
            "WHEN gpa >= 2.7 THEN 'B- (2.7-2.9)' " +
            "WHEN gpa >= 2.3 THEN 'C+ (2.3-2.6)' " +
            "WHEN gpa >= 2.0 THEN 'C (2.0-2.2)' " +
            "WHEN gpa >= 1.7 THEN 'C- (1.7-1.9)' " +
            "WHEN gpa >= 1.3 THEN 'D+ (1.3-1.6)' " +
            "WHEN gpa >= 1.0 THEN 'D (1.0-1.2)' " +
            "ELSE 'F (0.0-0.9)' " +
            "END " +
            "ORDER BY MIN(gpa) DESC")
    List<Map<String, Object>> getGpaDistribution();

    /**
     * 搜索学生（模糊查询）
     * 
     * @param keyword 关键词
     * @return 学生列表
     */
    @Select("SELECT s.*, u.username, u.real_name, u.email, u.phone " +
            "FROM student s " +
            "LEFT JOIN user u ON s.user_id = u.id " +
            "WHERE s.is_deleted = 0 " +
            "AND (s.student_number LIKE CONCAT('%', #{keyword}, '%') " +
            "OR s.major LIKE CONCAT('%', #{keyword}, '%') " +
            "OR u.real_name LIKE CONCAT('%', #{keyword}, '%') " +
            "OR u.username LIKE CONCAT('%', #{keyword}, '%') " +
            "OR u.email LIKE CONCAT('%', #{keyword}, '%')) " +
            "ORDER BY s.grade DESC, s.student_number")
    List<Map<String, Object>> searchStudents(@Param("keyword") String keyword);

    /**
     * 检查学号是否存在
     * 
     * @param studentNumber 学号
     * @param excludeStudentId 排除的学生ID
     * @return 是否存在
     */
    @Select("SELECT COUNT(*) FROM student " +
            "WHERE student_number = #{studentNumber} AND is_deleted = 0 " +
            "AND (#{excludeStudentId} IS NULL OR id != #{excludeStudentId})")
    int checkStudentNumberExists(@Param("studentNumber") String studentNumber, @Param("excludeStudentId") Long excludeStudentId);

    /**
     * 更新学生GPA
     * 
     * @param studentId 学生ID
     * @param gpa GPA
     * @return 更新结果
     */
    @Update("UPDATE student SET gpa = #{gpa}, update_time = NOW() WHERE id = #{studentId} AND is_deleted = 0")
    int updateGpa(@Param("studentId") Long studentId, @Param("gpa") Double gpa);

    /**
     * 更新学生学籍状态
     * 
     * @param studentId 学生ID
     * @param status 学籍状态
     * @return 更新结果
     */
    @Update("UPDATE student SET status = #{status}, update_time = NOW() WHERE id = #{studentId} AND is_deleted = 0")
    int updateStatus(@Param("studentId") Long studentId, @Param("status") String status);

    /**
     * 批量更新学生班级
     * 
     * @param studentIds 学生ID列表
     * @param classId 班级ID
     * @return 更新结果
     */
    int batchUpdateClass(@Param("studentIds") List<Long> studentIds, @Param("classId") Long classId);

    /**
     * 批量更新学生导师
     * 
     * @param studentIds 学生ID列表
     * @param mentorId 导师ID
     * @return 更新结果
     */
    int batchUpdateMentor(@Param("studentIds") List<Long> studentIds, @Param("mentorId") Long mentorId);

    /**
     * 获取即将毕业的学生（根据预计毕业时间）
     * 
     * @param months 提前几个月
     * @return 学生列表
     */
    @Select("SELECT * FROM student " +
            "WHERE is_deleted = 0 AND status = 'ENROLLED' " +
            "AND expected_graduation_date <= DATE_ADD(NOW(), INTERVAL #{months} MONTH) " +
            "AND expected_graduation_date >= NOW() " +
            "ORDER BY expected_graduation_date ASC")
    List<Student> selectStudentsNearGraduation(@Param("months") int months);

    /**
     * 获取新入学的学生（根据入学年份）
     * 
     * @param year 入学年份
     * @return 学生列表
     */
    @Select("SELECT * FROM student WHERE enrollment_year = #{year} AND is_deleted = 0 ORDER BY student_number")
    List<Student> selectNewStudentsByYear(@Param("year") Integer year);

    /**
     * 获取学生详细统计信息
     * 
     * @return 统计信息
     */
    @Select("SELECT " +
            "COUNT(*) as total_students, " +
            "SUM(CASE WHEN status = 'ENROLLED' THEN 1 ELSE 0 END) as enrolled_students, " +
            "SUM(CASE WHEN status = 'GRADUATED' THEN 1 ELSE 0 END) as graduated_students, " +
            "SUM(CASE WHEN status = 'SUSPENDED' THEN 1 ELSE 0 END) as suspended_students, " +
            "SUM(CASE WHEN status = 'DROPPED_OUT' THEN 1 ELSE 0 END) as dropped_students, " +
            "SUM(CASE WHEN gpa >= 3.5 THEN 1 ELSE 0 END) as excellent_students, " +
            "SUM(CASE WHEN gpa < 2.0 AND status = 'ENROLLED' THEN 1 ELSE 0 END) as warning_students, " +
            "AVG(gpa) as average_gpa, " +
            "MAX(gpa) as highest_gpa, " +
            "MIN(gpa) as lowest_gpa, " +
            "COUNT(DISTINCT major) as major_count, " +
            "COUNT(DISTINCT grade) as grade_count, " +
            "COUNT(DISTINCT class_id) as class_count " +
            "FROM student WHERE is_deleted = 0")
    Map<String, Object> getStudentStatistics();

    /**
     * 根据班级ID统计学生数
     * 
     * @param classId 班级ID
     * @return 学生数
     */
    @Select("SELECT COUNT(*) FROM student WHERE class_id = #{classId} AND is_deleted = 0")
    Long countStudentsByClassId(@Param("classId") Long classId);

    /**
     * 获取学生的课程选修情况
     * 
     * @param studentId 学生ID
     * @return 课程列表
     */
    @Select("SELECT c.*, sc.enrollment_date, sc.status as enrollment_status " +
            "FROM student_course sc " +
            "JOIN course c ON sc.course_id = c.id " +
            "WHERE sc.student_id = #{studentId} AND sc.is_deleted = 0 " +
            "ORDER BY sc.enrollment_date DESC")
    List<Map<String, Object>> getStudentCourses(@Param("studentId") Long studentId);

    /**
     * 获取学生的成绩记录
     * 
     * @param studentId 学生ID
     * @return 成绩列表
     */
    @Select("SELECT g.*, c.course_name, c.credits, t.task_title " +
            "FROM grade g " +
            "LEFT JOIN course c ON g.course_id = c.id " +
            "LEFT JOIN task t ON g.task_id = t.id " +
            "WHERE g.student_id = #{studentId} AND g.is_deleted = 0 " +
            "ORDER BY g.create_time DESC")
    List<Map<String, Object>> getStudentGrades(@Param("studentId") Long studentId);

    /**
     * 软删除学生
     * 
     * @param studentId 学生ID
     * @return 删除结果
     */
    @Update("UPDATE student SET is_deleted = 1, update_time = NOW() WHERE id = #{studentId}")
    int softDeleteStudent(@Param("studentId") Long studentId);

    /**
     * 批量软删除学生
     * 
     * @param studentIds 学生ID列表
     * @return 删除结果
     */
    int batchSoftDeleteStudents(@Param("studentIds") List<Long> studentIds);

    /**
     * 恢复已删除的学生
     * 
     * @param studentId 学生ID
     * @return 恢复结果
     */
    @Update("UPDATE student SET is_deleted = 0, update_time = NOW() WHERE id = #{studentId}")
    int restoreStudent(@Param("studentId") Long studentId);

    /**
     * 更新学生扩展字段
     * 
     * @param studentId 学生ID
     * @param extField1 扩展字段1
     * @param extField2 扩展字段2
     * @param extField3 扩展字段3
     * @return 更新结果
     */
    @Update("UPDATE student SET ext_field1 = #{extField1}, ext_field2 = #{extField2}, ext_field3 = #{extField3}, update_time = NOW() " +
            "WHERE id = #{studentId} AND is_deleted = 0")
    int updateExtFields(@Param("studentId") Long studentId, 
                       @Param("extField1") String extField1, 
                       @Param("extField2") String extField2, 
                       @Param("extField3") String extField3);

    /**
     * 获取指定年份的最大学号
     * 
     * @param year 年份
     * @return 最大学号
     */
    @Select("SELECT MAX(student_id) FROM student WHERE student_id LIKE CONCAT(#{year}, '%')")
    String getMaxStudentIdByYear(@Param("year") String year);
}