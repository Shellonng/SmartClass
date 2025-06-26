package com.education.controller.teacher;

import com.education.dto.common.Result;
import com.education.dto.common.PageRequest;
import com.education.dto.common.PageResponse;
import com.education.dto.task.TaskCommonDTOs.*;
import com.education.service.teacher.TaskService;
import com.education.utils.SecurityUtils;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.*;

import jakarta.validation.Valid;
import java.util.List;

/**
 * 教师端任务管理控制器
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
@Tag(name = "教师端任务管理", description = "任务的创建、发布、批阅、统计等功能")
@RestController
@RequestMapping("/api/teacher/tasks")
@RequiredArgsConstructor
@Slf4j
public class TaskController {

    private final TaskService taskService;

    /**
     * 分页查询任务列表
     */
    @Operation(summary = "分页查询任务列表")
    @GetMapping
    public Result<PageResponse<TaskListResponse>> getTaskList(
            @Valid PageRequest pageRequest,
            @RequestParam(required = false) String title,
            @RequestParam(required = false) String type,
            @RequestParam(required = false) String status,
            @RequestParam(required = false) Long courseId,
            @RequestParam(required = false) Long classId) {
        log.info("分页查询任务列表，页码：{}，页大小：{}，标题：{}，类型：{}，状态：{}", 
                pageRequest.getPageNum(), pageRequest.getPageSize(), title, type, status);
        
        PageResponse<TaskListResponse> response = taskService.getTaskList(pageRequest, title, status, type, courseId, classId);
        return Result.success(response);
    }

    /**
     * 创建任务
     */
    @Operation(summary = "创建任务")
    @PostMapping
    public Result<TaskResponse> createTask(@Valid @RequestBody TaskCreateRequest request) {
        log.info("创建任务，标题：{}，类型：{}，截止时间：{}", 
                request.getTitle(), request.getType(), request.getDeadline());
        
        TaskResponse response = taskService.createTask(request);
        return Result.success(response);
    }

    /**
     * 获取任务详情
     */
    @Operation(summary = "获取任务详情")
    @GetMapping("/{taskId}")
    public Result<TaskDetailResponse> getTaskDetail(@PathVariable Long taskId) {
        log.info("获取任务详情，任务ID：{}", taskId);
        
        TaskDetailResponse response = taskService.getTaskDetail(taskId);
        return Result.success(response);
    }

    /**
     * 更新任务信息
     */
    @Operation(summary = "更新任务信息")
    @PutMapping("/{taskId}")
    public Result<TaskResponse> updateTask(
            @PathVariable Long taskId,
            @Valid @RequestBody TaskUpdateRequest request) {
        log.info("更新任务信息，任务ID：{}，标题：{}", taskId, request.getTitle());
        
        TaskResponse response = taskService.updateTask(taskId, request);
        return Result.success(response);
    }

    /**
     * 删除任务
     */
    @Operation(summary = "删除任务")
    @DeleteMapping("/{taskId}")
    public Result<Void> deleteTask(@PathVariable Long taskId) {
        log.info("删除任务，任务ID：{}", taskId);
        
        taskService.deleteTask(taskId);
        return Result.success("任务删除成功");
    }

    /**
     * 发布任务
     */
    @Operation(summary = "发布任务")
    @PostMapping("/{taskId}/publish")
    public Result<Void> publishTask(@PathVariable Long taskId) {
        log.info("发布任务，任务ID：{}", taskId);
        
        taskService.publishTask(taskId);
        return Result.success("任务发布成功");
    }

    /**
     * 取消发布任务
     */
    @Operation(summary = "取消发布任务")  
    @PostMapping("/{taskId}/unpublish")
    public Result<Void> unpublishTask(@PathVariable Long taskId) {
        log.info("取消发布任务，任务ID：{}", taskId);
        
        taskService.unpublishTask(taskId);
        return Result.success("任务已取消发布");
    }

    /**
     * 获取任务提交列表
     */
    @Operation(summary = "获取任务提交列表")
    @GetMapping("/{taskId}/submissions")
    public Result<PageResponse<TaskSubmissionResponse>> getTaskSubmissions(
            @PathVariable Long taskId,
            @Valid PageRequest pageRequest,
            @RequestParam(required = false) String status,
            @RequestParam(required = false) String studentName) {
        log.info("获取任务提交列表，任务ID：{}，状态：{}，学生姓名：{}", taskId, status, studentName);
        
        PageResponse<TaskSubmissionResponse> response = taskService.getTaskSubmissions(taskId, pageRequest, studentName, status);
        return Result.success(response);
    }

    /**
     * 批阅任务提交
     */
    @Operation(summary = "批阅任务提交")
    @PostMapping("/{taskId}/submissions/{submissionId}/grade")
    public Result<Void> gradeSubmission(
            @PathVariable Long taskId,
            @PathVariable Long submissionId,
            @Valid @RequestBody TaskGradeRequest request) {
        log.info("批阅任务提交，任务ID：{}，提交ID：{}，分数：{}", taskId, submissionId, request.getScore());
        
        taskService.gradeSubmission(submissionId, request);
        return Result.success("批阅完成");
    }

    /**
     * 批量批阅
     */
    @Operation(summary = "批量批阅")
    @PostMapping("/{taskId}/submissions/batch-grade")
    public Result<Void> batchGradeSubmissions(
            @PathVariable Long taskId,
            @RequestBody List<TaskBatchGradeRequest> requests) {
        log.info("批量批阅，任务ID：{}，批阅数量：{}", taskId, requests.size());
        
        taskService.batchGradeSubmissions(requests);
        return Result.success("批量批阅完成");
    }

    /**
     * 获取任务统计信息
     */
    @Operation(summary = "获取任务统计信息")
    @GetMapping("/{taskId}/statistics")
    public Result<TaskStatisticsResponse> getTaskStatistics(@PathVariable Long taskId) {
        log.info("获取任务统计信息，任务ID：{}", taskId);
        
        TaskStatisticsResponse response = taskService.getTaskStatistics(taskId);
        return Result.success(response);
    }

    /**
     * 导出任务成绩
     */
    @Operation(summary = "导出任务成绩")
    @GetMapping("/{taskId}/export")
    public Result<String> exportTaskGrades(@PathVariable Long taskId) {
        log.info("导出任务成绩，任务ID：{}", taskId);
        
        String downloadUrl = taskService.exportTaskGrades(taskId);
        return Result.success(downloadUrl);
    }

    /**
     * 复制任务
     */
    @Operation(summary = "复制任务")
    @PostMapping("/{taskId}/copy")
    public Result<TaskResponse> copyTask(
            @PathVariable Long taskId,
            @RequestBody TaskCopyRequest request) {
        log.info("复制任务，原任务ID：{}，新标题：{}", taskId, request.getNewTitle());
        
        TaskResponse response = taskService.copyTask(taskId, request);
        return Result.success(response);
    }

    /**
     * 设置任务扩展时间
     */
    @Operation(summary = "设置任务扩展时间")
    @PostMapping("/{taskId}/extend")
    public Result<Void> extendTaskDeadline(
            @PathVariable Long taskId,
            @RequestBody TaskExtendRequest request) {
        log.info("扩展任务截止时间，任务ID：{}，新截止时间：{}", taskId, request.getNewDeadline());
        
        taskService.extendTaskDeadline(taskId, request);
        return Result.success("截止时间已扩展");
    }

    /**
     * 启用AI自动批阅
     */
    @Operation(summary = "启用AI自动批阅")
    @PostMapping("/{taskId}/ai-grade")
    public Result<Void> enableAIGrading(@PathVariable Long taskId) {
        log.info("启用AI自动批阅，任务ID：{}", taskId);
        
        taskService.enableAIGrading(taskId);
        return Result.success("AI自动批阅已启用");
    }

    /**
     * 获取任务模板列表
     */
    @Operation(summary = "获取任务模板列表")
    @GetMapping("/templates")
    public Result<PageResponse<TaskTemplateResponse>> getTaskTemplates(
            @RequestParam(required = false) String type,
            @RequestParam(required = false) String subject) {
        log.info("获取任务模板列表，类型：{}，学科：{}", type, subject);
        
        PageResponse<TaskTemplateResponse> response = taskService.getTaskTemplates(type, subject);
        return Result.success(response);
    }

    /**
     * 从模板创建任务
     */
    @Operation(summary = "从模板创建任务")
    @PostMapping("/from-template/{templateId}")
    public Result<TaskResponse> createTaskFromTemplate(
            @PathVariable Long templateId,
            @Valid @RequestBody TaskFromTemplateRequest request) {
        log.info("从模板创建任务，模板ID：{}，标题：{}", templateId, request.getTitle());
        
        TaskResponse response = taskService.createTaskFromTemplate(templateId, request);
        return Result.success(response);
    }
}