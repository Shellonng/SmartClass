package com.education.controller.teacher;

import com.education.dto.common.Result;
import com.education.service.teacher.DashboardService;
import com.education.util.SecurityUtils;
import org.springframework.beans.factory.annotation.Qualifier;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

/**
 * 教师端仪表板控制器
 */
@Tag(name = "教师端-仪表板", description = "教师仪表板数据接口")
@RestController("teacherDashboardController")
@RequestMapping("/api/teacher/dashboard")
@Slf4j
public class DashboardController {

    private final DashboardService dashboardService;
    
    public DashboardController(@Qualifier("teacherDashboardServiceImpl") DashboardService dashboardService) {
        this.dashboardService = dashboardService;
    }

    /**
     * 获取教师仪表板数据
     */
    @Operation(summary = "获取教师仪表板数据")
    @GetMapping
    public Result<Map<String, Object>> getDashboardData() {
        log.info("获取教师仪表板数据");
        
        Long teacherId = SecurityUtils.getCurrentUserId();
        Map<String, Object> dashboardData = dashboardService.getDashboardData(teacherId);
        return Result.success(dashboardData);
    }

    /**
     * 获取教学统计数据
     */
    @Operation(summary = "获取教学统计数据")
    @GetMapping("/statistics")
    public Result<Map<String, Object>> getTeachingStatistics(
            @RequestParam(required = false) String timeRange) {
        log.info("获取教学统计数据，时间范围：{}", timeRange);
        
        Long teacherId = SecurityUtils.getCurrentUserId();
        Map<String, Object> statistics = dashboardService.getTeachingStatistics(teacherId, timeRange);
        return Result.success(statistics);
    }

    /**
     * 获取待处理任务
     */
    @Operation(summary = "获取待处理任务")
    @GetMapping("/pending-tasks")
    public Result<Map<String, Object>> getPendingTasks() {
        log.info("获取待处理任务");
        
        Long teacherId = SecurityUtils.getCurrentUserId();
        Map<String, Object> pendingTasks = dashboardService.getPendingTasks(teacherId);
        return Result.success(pendingTasks);
    }

    /**
     * 获取课程概览
     */
    @Operation(summary = "获取课程概览")
    @GetMapping("/course-overview")
    public Result<Map<String, Object>> getCourseOverview() {
        log.info("获取课程概览");
        
        Long teacherId = SecurityUtils.getCurrentUserId();
        Map<String, Object> courseOverview = dashboardService.getCourseOverview(teacherId);
        return Result.success(courseOverview);
    }

    /**
     * 获取学生表现分析
     */
    @Operation(summary = "获取学生表现分析")
    @GetMapping("/student-performance")
    public Result<Map<String, Object>> getStudentPerformance(
            @RequestParam(required = false) Long courseId,
            @RequestParam(required = false) Long classId) {
        log.info("获取学生表现分析，课程ID：{}，班级ID：{}", courseId, classId);
        
        Long teacherId = SecurityUtils.getCurrentUserId();
        Map<String, Object> performance = dashboardService.getStudentPerformance(teacherId, courseId, classId);
        return Result.success(performance);
    }

    /**
     * 获取教学建议
     */
    @Operation(summary = "获取教学建议")
    @GetMapping("/teaching-suggestions")
    public Result<Map<String, Object>> getTeachingSuggestions() {
        log.info("获取教学建议");
        
        Long teacherId = SecurityUtils.getCurrentUserId();
        Map<String, Object> suggestions = dashboardService.getTeachingSuggestions(teacherId);
        return Result.success(suggestions);
    }

    /**
     * 获取近期活动
     */
    @Operation(summary = "获取近期活动")
    @GetMapping("/recent-activities")
    public Result<Map<String, Object>> getRecentActivities(
            @RequestParam(defaultValue = "10") Integer limit) {
        log.info("获取近期活动，限制数量：{}", limit);
        
        Long teacherId = SecurityUtils.getCurrentUserId();
        Map<String, Object> activities = dashboardService.getRecentActivities(teacherId, limit);
        return Result.success(activities);
    }
} 