package com.education.service.teacher;

import com.education.dto.common.PageRequest;
import com.education.dto.common.PageResponse;
import com.education.entity.Course;

import java.util.Map;

/**
 * 教师课程服务接口
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
public interface CourseService {

    /**
     * 获取教师的课程列表
     * 
     * @param username 用户名
     * @param pageRequest 分页请求
     * @param keyword 关键词
     * @param status 状态
     * @param term 学期
     * @return 课程分页数据
     */
    PageResponse<Course> getTeacherCourses(String username, PageRequest pageRequest, String keyword, String status, String term);

    /**
     * 创建课程
     * 
     * @param username 用户名
     * @param course 课程信息
     * @return 创建后的课程
     */
    Course createCourse(String username, Course course);

    /**
     * 获取课程详情
     * 
     * @param username 用户名
     * @param courseId 课程ID
     * @return 课程详情
     */
    Course getCourseDetail(String username, Long courseId);

    /**
     * 更新课程信息
     * 
     * @param username 用户名
     * @param course 课程信息
     * @return 更新后的课程
     */
    Course updateCourse(String username, Course course);

    /**
     * 删除课程
     * 
     * @param username 用户名
     * @param courseId 课程ID
     * @return 是否删除成功
     */
    boolean deleteCourse(String username, Long courseId);

    /**
     * 发布课程
     * 
     * @param username 用户名
     * @param courseId 课程ID
     * @return 更新后的课程
     */
    Course publishCourse(String username, Long courseId);

    /**
     * 取消发布课程
     * 
     * @param username 用户名
     * @param courseId 课程ID
     * @return 更新后的课程
     */
    Course unpublishCourse(String username, Long courseId);

    /**
     * 获取课程统计信息
     * 
     * @param username 用户名
     * @param courseId 课程ID
     * @return 统计信息
     */
    Map<String, Object> getCourseStatistics(String username, Long courseId);
} 