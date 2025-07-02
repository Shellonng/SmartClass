package com.education.service.teacher;

import com.education.dto.common.PageResponse;
import com.education.entity.CourseClass;
import com.education.entity.Student;

import java.util.List;

/**
 * 班级管理服务接口
 */
public interface ClassService {

    /**
     * 分页获取当前教师的班级列表
     *
     * @param page 页码
     * @param size 每页大小
     * @param keyword 关键词
     * @param courseId 课程ID
     * @return 分页班级列表
     */
    PageResponse<CourseClass> getClassesByTeacher(int page, int size, String keyword, Long courseId);

    /**
     * 获取班级详情
     *
     * @param id 班级ID
     * @return 班级详情
     */
    CourseClass getClassById(Long id);

    /**
     * 创建班级
     *
     * @param courseClass 班级信息
     * @return 创建后的班级
     */
    CourseClass createClass(CourseClass courseClass);

    /**
     * 更新班级信息
     *
     * @param courseClass 班级信息
     * @return 更新后的班级
     */
    CourseClass updateClass(CourseClass courseClass);

    /**
     * 删除班级
     *
     * @param id 班级ID
     */
    void deleteClass(Long id);

    /**
     * 获取班级学生列表
     *
     * @param classId 班级ID
     * @param page 页码
     * @param size 每页大小
     * @param keyword 关键词
     * @return 分页学生列表
     */
    PageResponse<Student> getStudentsByClassId(Long classId, int page, int size, String keyword);

    /**
     * 添加学生到班级
     *
     * @param classId 班级ID
     * @param studentIds 学生ID列表
     */
    void addStudentsToClass(Long classId, List<Long> studentIds);

    /**
     * 从班级移除学生
     *
     * @param classId 班级ID
     * @param studentId 学生ID
     */
    void removeStudentFromClass(Long classId, Long studentId);

    /**
     * 根据用户ID查询教师ID
     *
     * @param userId 用户ID
     * @return 教师ID，如果不存在则返回null
     */
    Long getTeacherIdByUserId(Long userId);
} 