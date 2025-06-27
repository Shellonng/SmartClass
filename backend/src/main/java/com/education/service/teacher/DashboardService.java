package com.education.service.teacher;

import java.util.Map;

/**
 * 教师端仪表板服务接口
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
public interface DashboardService {

    /**
     * 获取教师仪表板数据
     * 
     * @param teacherId 教师ID
     * @return 仪表板数据
     */
    Map<String, Object> getDashboardData(Long teacherId);

    /**
     * 获取教学统计数据
     * 
     * @param teacherId 教师ID
     * @param timeRange 时间范围
     * @return 教学统计数据
     */
    Map<String, Object> getTeachingStatistics(Long teacherId, String timeRange);

    /**
     * 获取待处理任务
     * 
     * @param teacherId 教师ID
     * @return 待处理任务
     */
    Map<String, Object> getPendingTasks(Long teacherId);

    /**
     * 获取课程概览
     * 
     * @param teacherId 教师ID
     * @return 课程概览
     */
    Map<String, Object> getCourseOverview(Long teacherId);

    /**
     * 获取学生表现分析
     * 
     * @param teacherId 教师ID
     * @param courseId 课程ID
     * @param classId 班级ID
     * @return 学生表现分析
     */
    Map<String, Object> getStudentPerformance(Long teacherId, Long courseId, Long classId);

    /**
     * 获取教学建议
     * 
     * @param teacherId 教师ID
     * @return 教学建议
     */
    Map<String, Object> getTeachingSuggestions(Long teacherId);

    /**
     * 获取最近活动
     * 
     * @param teacherId 教师ID
     * @param limit 限制数量
     * @return 最近活动
     */
    Map<String, Object> getRecentActivities(Long teacherId, Integer limit);
} 