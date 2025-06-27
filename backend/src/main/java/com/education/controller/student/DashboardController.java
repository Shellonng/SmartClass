package com.education.controller.student;

import com.education.dto.common.Result;
import com.education.service.student.DashboardService;
import com.education.util.SecurityUtils;
import org.springframework.beans.factory.annotation.Qualifier;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

/**
 * 学生端仪表板控制器
 */
@Tag(name = "学生端-仪表板", description = "学生仪表板数据接口")
@RestController("studentDashboardController")
@RequestMapping("/api/student/dashboard")
@Slf4j
public class DashboardController {

    private final DashboardService dashboardService;
    
    public DashboardController(@Qualifier("studentDashboardServiceImpl") DashboardService dashboardService) {
        this.dashboardService = dashboardService;
    }

    /**
     * 获取学生仪表板数据
     */
    @Operation(summary = "获取学生仪表板数据")
    @GetMapping
    public Result<Map<String, Object>> getDashboardData() {
        log.info("获取学生仪表板数据");
        
        Long studentId = SecurityUtils.getCurrentUserId();
        Map<String, Object> dashboardData = dashboardService.getDashboardData(studentId);
        return Result.success(dashboardData);
    }

    /**
     * 获取学习统计数据
     */
    @Operation(summary = "获取学习统计数据")
    @GetMapping("/statistics")
    public Result<Map<String, Object>> getStudyStatistics(
            @RequestParam(required = false) String timeRange) {
        log.info("获取学习统计数据，时间范围：{}", timeRange);
        
        Long studentId = SecurityUtils.getCurrentUserId();
        Map<String, Object> statistics = dashboardService.getStudyStatistics(studentId, timeRange);
        return Result.success(statistics);
    }

    /**
     * 获取今日任务列表
     */
    @Operation(summary = "获取今日任务列表")
    @GetMapping("/today-tasks")
    public Result<Map<String, Object>> getTodayTasks() {
        log.info("获取今日任务列表");
        
        Long studentId = SecurityUtils.getCurrentUserId();
        Map<String, Object> todayTasks = dashboardService.getTodayTasks(studentId);
        return Result.success(todayTasks);
    }

    /**
     * 获取学习进度概览
     */
    @Operation(summary = "获取学习进度概览")
    @GetMapping("/progress-overview")
    public Result<Map<String, Object>> getProgressOverview() {
        log.info("获取学习进度概览");
        
        Long studentId = SecurityUtils.getCurrentUserId();
        Map<String, Object> progressOverview = dashboardService.getProgressOverview(studentId);
        return Result.success(progressOverview);
    }

    /**
     * 获取近期成绩
     */
    @Operation(summary = "获取近期成绩")
    @GetMapping("/recent-grades")
    public Result<Map<String, Object>> getRecentGrades(
            @RequestParam(defaultValue = "10") Integer limit) {
        log.info("获取近期成绩，限制数量：{}", limit);
        
        Long studentId = SecurityUtils.getCurrentUserId();
        Map<String, Object> recentGrades = dashboardService.getRecentGrades(studentId, limit);
        return Result.success(recentGrades);
    }

    /**
     * 获取学习建议
     */
    @Operation(summary = "获取学习建议")
    @GetMapping("/learning-suggestions")
    public Result<Map<String, Object>> getLearningSuggestions() {
        log.info("获取学习建议");
        
        Long studentId = SecurityUtils.getCurrentUserId();
        Map<String, Object> suggestions = dashboardService.getLearningSuggestions(studentId);
        return Result.success(suggestions);
    }
} 