package com.education.service.student;

import java.util.Map;

/**
 * 学生端仪表板服务接口
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
public interface DashboardService {

    /**
     * 获取学生仪表板数据
     * 
     * @param studentId 学生ID
     * @return 仪表板数据
     */
    Map<String, Object> getDashboardData(Long studentId);

    /**
     * 获取学习统计数据
     * 
     * @param studentId 学生ID
     * @param timeRange 时间范围
     * @return 学习统计数据
     */
    Map<String, Object> getStudyStatistics(Long studentId, String timeRange);

    /**
     * 获取今日任务
     * 
     * @param studentId 学生ID
     * @return 今日任务
     */
    Map<String, Object> getTodayTasks(Long studentId);

    /**
     * 获取学习进度概览
     * 
     * @param studentId 学生ID
     * @return 学习进度概览
     */
    Map<String, Object> getProgressOverview(Long studentId);

    /**
     * 获取最近成绩
     * 
     * @param studentId 学生ID
     * @param limit 限制数量
     * @return 最近成绩
     */
    Map<String, Object> getRecentGrades(Long studentId, Integer limit);

    /**
     * 获取学习建议
     * 
     * @param studentId 学生ID
     * @return 学习建议
     */
    Map<String, Object> getLearningSuggestions(Long studentId);
} 