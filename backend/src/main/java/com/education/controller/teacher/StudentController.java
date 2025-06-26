package com.education.controller.teacher;

import com.education.dto.common.Result;
import com.education.dto.common.PageRequest;
import com.education.dto.StudentDTO;
import com.education.dto.StudentDTOExtension;
import com.education.service.teacher.StudentService;
import com.education.utils.JwtUtils;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.servlet.http.HttpServletRequest;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

/**
 * 教师端学生管理控制器
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
@Tag(name = "教师端-学生管理", description = "教师查看和管理学生信息相关接口")
@RestController
@RequestMapping("/api/teacher/students")
public class StudentController {

    @Autowired
    private StudentService studentService;
    
    @Autowired
    private JwtUtils jwtUtils;
    
    @Autowired
    private HttpServletRequest request;

    @Operation(summary = "获取学生列表", description = "获取教师所有班级的学生列表")
    @GetMapping
    public Result<Object> getStudents(
            @RequestParam(defaultValue = "1") Integer page,
            @RequestParam(defaultValue = "10") Integer size,
            @RequestParam(required = false) String keyword,
            @RequestParam(required = false) Long classId) {
        try {
            Long teacherId = getCurrentTeacherId();
            PageRequest pageRequest = buildPageRequest(page, size, classId, keyword);
            Object students = studentService.getStudentList(teacherId, pageRequest);
            return Result.success(students);
        } catch (Exception e) {
            return Result.error("获取学生列表失败: " + e.getMessage());
        }
    }

    @Operation(summary = "获取学生详情", description = "获取指定学生的详细信息")
    @GetMapping("/{studentId}")
    public Result<Object> getStudentDetail(@PathVariable Long studentId) {
        try {
            Long teacherId = getCurrentTeacherId();
            Object student = studentService.getStudentDetail(studentId, teacherId);
            return Result.success(student);
        } catch (Exception e) {
            return Result.error("获取学生详情失败: " + e.getMessage());
        }
    }

    @Operation(summary = "获取学生学习进度", description = "获取学生在各课程中的学习进度")
    @GetMapping("/{studentId}/progress")
    public Result<Object> getStudentProgress(@PathVariable Long studentId) {
        try {
            Long teacherId = getCurrentTeacherId();
            Object progress = studentService.getStudentProgress(studentId, teacherId, null);
            return Result.success(progress);
        } catch (Exception e) {
            return Result.error("获取学生学习进度失败: " + e.getMessage());
        }
    }

    @Operation(summary = "获取学生成绩统计", description = "获取学生的成绩统计信息")
    @GetMapping("/{studentId}/grades")
    public Result<Object> getStudentGrades(
            @PathVariable Long studentId,
            @RequestParam(required = false) Long courseId,
            @RequestParam(defaultValue = "1") Integer page,
            @RequestParam(defaultValue = "10") Integer size) {
        try {
            Long teacherId = getCurrentTeacherId();
            Object grades = studentService.getStudentGradeStatistics(studentId, teacherId);
            return Result.success(grades);
        } catch (Exception e) {
            return Result.error("获取学生成绩失败: " + e.getMessage());
        }
    }

    @Operation(summary = "获取学生任务提交记录", description = "获取学生的任务提交历史")
    @GetMapping("/{studentId}/submissions")
    public Result<Object> getStudentSubmissions(
            @PathVariable Long studentId,
            @RequestParam(required = false) Long taskId,
            @RequestParam(defaultValue = "1") Integer page,
            @RequestParam(defaultValue = "10") Integer size) {
        try {
            Long teacherId = getCurrentTeacherId();
            PageRequest pageRequest = buildSubmissionPageRequest(page, size, taskId);
            Object submissions = studentService.getStudentSubmissions(studentId, teacherId, pageRequest);
            return Result.success(submissions);
        } catch (Exception e) {
            return Result.error("获取学生提交记录失败: " + e.getMessage());
        }
    }

    @Operation(summary = "获取学生学习分析", description = "获取学生的学习行为分析数据")
    @GetMapping("/{studentId}/analytics")
    public Result<Object> getStudentAnalytics(
            @PathVariable Long studentId,
            @RequestParam(required = false) String timeRange) {
        try {
            Long teacherId = getCurrentTeacherId();
            Object analytics = studentService.getStudentAnalysis(studentId, teacherId, timeRange);
            return Result.success(analytics);
        } catch (Exception e) {
            return Result.error("获取学生学习分析失败: " + e.getMessage());
        }
    }

    @Operation(summary = "批量导入学生", description = "通过Excel批量导入学生信息")
    @PostMapping("/import")
    public Result<Object> importStudents(@RequestParam Long classId, @RequestParam String fileUrl) {
        try {
            Long teacherId = getCurrentTeacherId();
            StudentDTO.StudentImportRequest importRequest = buildImportRequest(classId, fileUrl);
            Object importResult = studentService.importStudents(importRequest, teacherId);
            return Result.success(importResult);
        } catch (Exception e) {
            return Result.error("批量导入学生失败: " + e.getMessage());
        }
    }

    @Operation(summary = "导出学生信息", description = "导出学生信息到Excel")
    @GetMapping("/export")
    public Result<Object> exportStudents(
            @RequestParam(required = false) Long classId,
            @RequestParam(required = false) String keyword) {
        try {
            Long teacherId = getCurrentTeacherId();
            StudentDTOExtension.StudentExportRequest exportRequest = buildExportRequest(classId, keyword);
            Object exportResult = studentService.exportStudents(exportRequest, teacherId);
            return Result.success(exportResult);
        } catch (Exception e) {
            return Result.error("导出学生信息失败: " + e.getMessage());
        }
    }

    @Operation(summary = "重置学生密码", description = "重置指定学生的登录密码")
    @PostMapping("/{studentId}/reset-password")
    public Result<Object> resetStudentPassword(@PathVariable Long studentId) {
        try {
            Long teacherId = getCurrentTeacherId();
            Object resetResult = studentService.resetStudentPassword(studentId, teacherId);
            return Result.success(resetResult);
        } catch (Exception e) {
            return Result.error("重置学生密码失败: " + e.getMessage());
        }
    }

    @Operation(summary = "获取学生排行榜", description = "获取班级或课程的学生排行榜")
    @GetMapping("/ranking")
    public Result<Object> getStudentRanking(
            @RequestParam(required = false) Long classId,
            @RequestParam(required = false) Long courseId,
            @RequestParam(defaultValue = "score") String rankBy,
            @RequestParam(defaultValue = "10") Integer limit) {
        try {
            Long teacherId = getCurrentTeacherId();
            Object ranking = studentService.getStudentRanking(teacherId, rankBy, classId, limit);
            return Result.success(ranking);
        } catch (Exception e) {
            return Result.error("获取学生排行榜失败: " + e.getMessage());
        }
    }

    // 辅助方法
    private Long getCurrentTeacherId() {
        try {
            String token = JwtUtils.getTokenFromRequest(request);
            return jwtUtils.getUserIdFromToken(token);
        } catch (Exception e) {
            throw new RuntimeException("获取当前教师ID失败: " + e.getMessage());
        }
    }

    private PageRequest buildPageRequest(Integer page, Integer size, Long classId, String keyword) {
        PageRequest pageRequest = new PageRequest();
        pageRequest.setPageNum(page);
        pageRequest.setPageSize(size);
        // 可以根据需要添加其他查询条件
        return pageRequest;
    }

    private PageRequest buildSubmissionPageRequest(Integer page, Integer size, Long taskId) {
        PageRequest pageRequest = new PageRequest();
        pageRequest.setPageNum(page);
        pageRequest.setPageSize(size);
        // 可以根据需要添加taskId等查询条件
        return pageRequest;
    }

    private StudentDTO.StudentImportRequest buildImportRequest(Long classId, String fileUrl) {
        StudentDTO.StudentImportRequest importRequest = new StudentDTO.StudentImportRequest();
        importRequest.setImportType("EXCEL");
        importRequest.setFileUrl(fileUrl);
        return importRequest;
    }

    private StudentDTOExtension.StudentExportRequest buildExportRequest(Long classId, String keyword) {
        StudentDTOExtension.StudentExportRequest exportRequest = new StudentDTOExtension.StudentExportRequest();
        exportRequest.setExportType("EXCEL");
        exportRequest.setIncludeGrades(true);
        exportRequest.setIncludeProgress(true);
        return exportRequest;
    }
}