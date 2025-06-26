package com.education.controller.teacher;
import com.education.dto.common.Result;
import com.education.dto.common.PageRequest;
import com.education.dto.TaskDTO;
import com.education.service.teacher.TaskService;
import com.education.utils.JwtUtils;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.servlet.http.HttpServletRequest;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.List;

/**
 * 教师端任务管理控制器
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
@Tag(name = "教师端-任务管理", description = "教师任务创建、管理、批改等接口")
@RestController("teacherTaskController")
@RequestMapping("/api/teacher/tasks")
public class TaskController {

    @Autowired
    private TaskService taskService;

    @Operation(summary = "创建任务", description = "教师创建新任务")
    @PostMapping
    public Result<Object> createTask(@RequestBody TaskDTO.TaskCreateRequest createRequest) {
        try {
            // 1. 获取当前教师ID
            Long teacherId = getCurrentTeacherId();
            
            // 2. 调用服务层创建任务
            Object result = taskService.createTask(createRequest, teacherId);
            
            return Result.success(result);
        } catch (Exception e) {
            return Result.error("创建任务失败: " + e.getMessage());
        }
    }

    @Operation(summary = "获取任务列表", description = "获取教师创建的任务列表")
    @GetMapping
    public Result<Object> getTasks(
            @RequestParam(defaultValue = "1") Integer page,
            @RequestParam(defaultValue = "10") Integer size,
            @RequestParam(required = false) String keyword,
            @RequestParam(required = false) Long courseId,
            @RequestParam(required = false) String status) {
        try {
            // 1. 获取当前教师ID
            Long teacherId = getCurrentTeacherId();
            
            // 2. 构建分页请求
            PageRequest pageRequest = new PageRequest();
            pageRequest.setPageNum(page);
            pageRequest.setPageSize(size);
            pageRequest.setKeyword(keyword);
            
            // 3. 调用服务层查询任务列表
            Object result = taskService.getTaskList(teacherId, pageRequest);
            
            return Result.success(result);
        } catch (Exception e) {
            return Result.error("获取任务列表失败: " + e.getMessage());
        }
    }

    @Operation(summary = "获取任务详情", description = "获取指定任务的详细信息")
    @GetMapping("/{taskId}")
    public Result<Object> getTaskDetail(@PathVariable Long taskId) {
        try {
            // 1. 获取当前教师ID
            Long teacherId = getCurrentTeacherId();
            
            // 2. 调用服务层获取任务详情
            Object result = taskService.getTaskDetail(taskId, teacherId);
            
            return Result.success(result);
        } catch (Exception e) {
            return Result.error("获取任务详情失败: " + e.getMessage());
        }
    }

    @Operation(summary = "更新任务信息", description = "更新任务基本信息")
    @PutMapping("/{taskId}")
    public Result<Object> updateTask(@PathVariable Long taskId, @RequestBody TaskDTO.TaskUpdateRequest updateRequest) {
        try {
            // 1. 获取当前教师ID
            Long teacherId = getCurrentTeacherId();
            
            // 2. 调用服务层更新任务
            Object result = taskService.updateTask(taskId, updateRequest, teacherId);
            
            return Result.success(result);
        } catch (Exception e) {
            return Result.error("更新任务失败: " + e.getMessage());
        }
    }

    @Operation(summary = "删除任务", description = "删除指定任务")
    @DeleteMapping("/{taskId}")
    public Result<Void> deleteTask(@PathVariable Long taskId) {
        try {
            // 1. 获取当前教师ID
            Long teacherId = getCurrentTeacherId();
            
            // 2. 调用服务层删除任务
            Boolean result = taskService.deleteTask(taskId, teacherId);
            
            if (result) {
                return Result.success();
            } else {
                return Result.error("删除任务失败");
            }
        } catch (Exception e) {
            return Result.error("删除任务失败: " + e.getMessage());
        }
    }

    @Operation(summary = "发布任务", description = "发布任务供学生完成")
    @PostMapping("/{taskId}/publish")
    public Result<Void> publishTask(@PathVariable Long taskId) {
        try {
            // 1. 获取当前教师ID
            Long teacherId = getCurrentTeacherId();
            
            // 2. 调用服务层发布任务
            Boolean result = taskService.publishTask(taskId, teacherId);
            
            if (result) {
                return Result.success();
            } else {
                return Result.error("发布任务失败");
            }
        } catch (Exception e) {
            return Result.error("发布任务失败: " + e.getMessage());
        }
    }

    @Operation(summary = "关闭任务", description = "关闭任务，不再接受提交")
    @PostMapping("/{taskId}/close")
    public Result<Void> closeTask(@PathVariable Long taskId) {
        try {
            // 1. 获取当前教师ID
            Long teacherId = getCurrentTeacherId();
            
            // 2. 调用服务层关闭任务
            Boolean result = taskService.closeTask(taskId, teacherId);
            
            if (result) {
                return Result.success();
            } else {
                return Result.error("关闭任务失败");
            }
        } catch (Exception e) {
            return Result.error("关闭任务失败: " + e.getMessage());
        }
    }

    @Operation(summary = "获取任务提交列表", description = "获取任务的所有学生提交")
    @GetMapping("/{taskId}/submissions")
    public Result<Object> getTaskSubmissions(
            @PathVariable Long taskId,
            @RequestParam(defaultValue = "1") Integer page,
            @RequestParam(defaultValue = "10") Integer size,
            @RequestParam(required = false) String status,
            @RequestParam(required = false) String keyword) {
        try {
            // 1. 获取当前教师ID
            Long teacherId = getCurrentTeacherId();
            
            // 2. 构建分页请求
            PageRequest pageRequest = new PageRequest();
            pageRequest.setPageNum(page);
            pageRequest.setPageSize(size);
            pageRequest.setKeyword(keyword);
            
            // 3. 调用服务层获取提交列表
            Object result = taskService.getTaskSubmissions(taskId, teacherId, pageRequest);
            
            return Result.success(result);
        } catch (Exception e) {
            return Result.error("获取任务提交列表失败: " + e.getMessage());
        }
    }

    @Operation(summary = "批改任务提交", description = "对学生的任务提交进行批改")
    @PostMapping("/{taskId}/submissions/{submissionId}/grade")
    public Result<Object> gradeSubmission(
            @PathVariable Long taskId,
            @PathVariable Long submissionId,
            @RequestBody Object gradeRequest) {
        try {
            // 1. 获取当前教师ID
            Long teacherId = getCurrentTeacherId();
            
            // 2. 调用服务层批改提交
            Object result = taskService.gradeSubmission(submissionId, gradeRequest, teacherId);
            
            return Result.success(result);
        } catch (Exception e) {
            return Result.error("批改任务提交失败: " + e.getMessage());
        }
    }

    @Operation(summary = "批量批改", description = "批量批改多个学生的提交")
    @PostMapping("/{taskId}/submissions/batch-grade")
    public Result<Object> batchGradeSubmissions(
            @PathVariable Long taskId,
            @RequestBody List<Object> batchGradeRequest) {
        try {
            // 1. 获取当前教师ID
            Long teacherId = getCurrentTeacherId();
            
            // 2. 调用服务层批量批改
            Object result = taskService.batchGradeSubmissions(batchGradeRequest, teacherId);
            
            return Result.success(result);
        } catch (Exception e) {
            return Result.error("批量批改失败: " + e.getMessage());
        }
    }

    @Operation(summary = "获取任务统计", description = "获取任务的统计数据")
    @GetMapping("/{taskId}/statistics")
    public Result<Object> getTaskStatistics(@PathVariable Long taskId) {
        try {
            // 1. 获取当前教师ID
            Long teacherId = getCurrentTeacherId();
            
            // 2. 调用服务层获取任务统计
            Object result = taskService.getTaskStatistics(taskId, teacherId);
            
            return Result.success(result);
        } catch (Exception e) {
            return Result.error("获取任务统计失败: " + e.getMessage());
        }
    }

    @Operation(summary = "导出任务成绩", description = "导出任务成绩到Excel")
    @GetMapping("/{taskId}/export-grades")
    public Result<Object> exportTaskGrades(@PathVariable Long taskId) {
        try {
            // 1. 获取当前教师ID
            Long teacherId = getCurrentTeacherId();
            
            // 2. 调用服务层导出成绩
            Object result = taskService.exportTaskGrades(taskId, teacherId);
            
            return Result.success(result);
        } catch (Exception e) {
            return Result.error("导出任务成绩失败: " + e.getMessage());
        }
    }

    @Operation(summary = "复制任务", description = "复制现有任务创建新任务")
    @PostMapping("/{taskId}/copy")
    public Result<Object> copyTask(@PathVariable Long taskId, @RequestBody String newTaskTitle) {
        try {
            // 1. 获取当前教师ID
            Long teacherId = getCurrentTeacherId();
            
            // 2. 调用服务层复制任务
            Object result = taskService.copyTask(taskId, newTaskTitle, teacherId);
            
            return Result.success(result);
        } catch (Exception e) {
            return Result.error("复制任务失败: " + e.getMessage());
        }
    }

    @Operation(summary = "获取抄袭检测结果", description = "获取任务的抄袭检测结果")
    @GetMapping("/{taskId}/plagiarism")
    public Result<Object> getPlagiarismResults(@PathVariable Long taskId) {
        try {
            // 1. 获取当前教师ID
            Long teacherId = getCurrentTeacherId();
            
            // 2. 调用服务层获取抄袭检测结果
            Object result = taskService.getPlagiarismDetectionResult(taskId, teacherId);
            
            return Result.success(result);
        } catch (Exception e) {
            return Result.error("获取抄袭检测结果失败: " + e.getMessage());
        }
    }

    @Operation(summary = "启动抄袭检测", description = "对任务提交启动抄袭检测")
    @PostMapping("/{taskId}/plagiarism/start")
    public Result<Void> startPlagiarismDetection(@PathVariable Long taskId) {
        try {
            // 1. 获取当前教师ID
            Long teacherId = getCurrentTeacherId();
            
            // 2. 调用服务层启动抄袭检测
            Boolean result = taskService.startPlagiarismDetection(taskId, teacherId);
            
            if (result) {
                return Result.success();
            } else {
                return Result.error("启动抄袭检测失败");
            }
        } catch (Exception e) {
            return Result.error("启动抄袭检测失败: " + e.getMessage());
        }
    }

    /**
     * 获取当前教师ID
     * 从JWT token或session中获取当前登录教师的ID
     */
    private Long getCurrentTeacherId() {
        // TODO: 实际项目中应该从JWT token或session中获取
        // 这里暂时返回模拟数据
        return 1L;
    }


}