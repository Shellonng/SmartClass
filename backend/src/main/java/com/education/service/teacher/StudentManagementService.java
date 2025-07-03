package com.education.service.teacher;

import com.education.dto.common.PageResponse;
import com.education.entity.CourseClass;
import com.education.entity.Student;
import com.education.entity.User;

import java.util.List;
import java.util.Map;

/**
 * 学生管理服务接口
 */
public interface StudentManagementService {

    /**
     * 分页获取学生列表
     *
     * @param page 页码
     * @param size 每页大小
     * @param keyword 关键词（学生姓名或学号）
     * @param classId 班级ID
     * @param courseId 课程ID
     * @return 分页学生列表
     */
    PageResponse<Student> getStudents(int page, int size, String keyword, Long classId, Long courseId);

    /**
     * 搜索学生
     *
     * @param keyword 关键词（学生姓名或学号）
     * @return 学生列表
     */
    List<Map<String, Object>> searchStudents(String keyword);

    /**
     * 获取学生详情
     *
     * @param id 学生ID
     * @return 学生详情
     */
    Student getStudentById(Long id);

    /**
     * 获取教师的班级列表
     *
     * @param teacherId 教师ID
     * @return 班级列表
     */
    List<CourseClass> getClassesByTeacherId(Long teacherId);

    /**
     * 添加学生到班级
     *
     * @param studentId 学生ID
     * @param classId 班级ID
     */
    void addStudentToClass(Long studentId, Long classId);

    /**
     * 从班级移除学生
     *
     * @param studentId 学生ID
     * @param classId 班级ID
     */
    void removeStudentFromClass(Long studentId, Long classId);

    /**
     * 添加学生到课程
     *
     * @param studentId 学生ID
     * @param courseId 课程ID
     */
    void addStudentToCourse(Long studentId, Long courseId);

    /**
     * 从课程移除学生
     *
     * @param studentId 学生ID
     * @param courseId 课程ID
     */
    void removeStudentFromCourse(Long studentId, Long courseId);

    /**
     * 处理选课申请
     *
     * @param requestId 申请ID
     * @param approved 是否通过
     * @param comment 评论
     */
    void processEnrollmentRequest(Long requestId, Boolean approved, String comment);

    /**
     * 获取选课申请列表
     *
     * @param page 页码
     * @param size 每页大小
     * @param courseId 课程ID
     * @return 分页选课申请列表
     */
    PageResponse<Map<String, Object>> getEnrollmentRequests(int page, int size, Long courseId);

    /**
     * 创建学生账户
     *
     * @param user 用户信息
     * @return 创建的用户
     */
    User createStudent(User user);

    /**
     * 更新学生信息
     *
     * @param student 学生信息
     * @return 更新后的学生
     */
    Student updateStudent(Student student);

    /**
     * 获取学生所属班级列表
     *
     * @param studentId 学生ID
     * @return 班级列表，包含班级基本信息和关联课程信息
     */
    List<Map<String, Object>> getStudentClasses(Long studentId);

    /**
     * 获取学生所属课程列表
     *
     * @param studentId 学生ID
     * @return 课程列表，包含课程基本信息和选课时间
     */
    List<Map<String, Object>> getStudentCourses(Long studentId);
} 