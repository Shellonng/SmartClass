package com.education.controller.teacher;

import com.education.dto.GradeDTO;
import com.education.dto.common.PageRequest;
import com.education.dto.common.PageResponse;
import com.education.dto.common.Result;
import com.education.service.teacher.GradeService;
import com.education.utils.JwtUtils;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.servlet.http.HttpServletRequest;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;
import java.util.Arrays;

import java.util.List;

/**
 * 教师端成绩管理控制器
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
@Tag(name = "教师端-成绩管理", description = "教师成绩录入、查看、统计等接口")
@RestController("teacherGradeController")
@RequestMapping("/api/teacher/grades")
public class GradeController {

    @Autowired
    private GradeService gradeService;
    
    @Autowired
    private JwtUtils jwtUtils;
    
    @Autowired
    private HttpServletRequest request;

    @Operation(summary = "获取成绩列表", description = "获取班级或课程的成绩列表")
    @GetMapping
    public Result<Object> getGrades(
            @RequestParam(defaultValue = "1") Integer page,
            @RequestParam(defaultValue = "10") Integer size,
            @RequestParam(required = false) Long classId,
            @RequestParam(required = false) Long courseId,
            @RequestParam(required = false) Long taskId,
            @RequestParam(required = false) String keyword) {
        try {
            // 1. 获取当前教师ID
            Long teacherId = getCurrentTeacherId();
            
            // 2. 构建查询参数
            PageRequest pageRequest = new PageRequest();
            pageRequest.setPage(page);
        pageRequest.setSize(size);
        // PageRequest类没有setKeyword方法，需要在查询时处理关键字
        // pageRequest.setKeyword(keyword);
            
            // 3. 调用服务层获取成绩列表
            Object result = gradeService.getGradeList(teacherId, pageRequest);
            
            return Result.success(result);
        } catch (Exception e) {
            return Result.error("获取成绩列表失败: " + e.getMessage());
        }
    }

    @Operation(summary = "录入成绩", description = "为学生录入成绩")
    @PostMapping
    public Result<GradeDTO.GradeResponse> createGrade(@RequestBody GradeDTO.GradeCreateRequest createRequest) {
        try {
            // 1. 获取当前教师ID
            Long teacherId = getCurrentTeacherId();
            
            // 2. 调用服务层录入成绩
            GradeDTO.GradeResponse result = gradeService.createGrade(createRequest, teacherId);
            
            return Result.success(result);
        } catch (Exception e) {
            return Result.error("录入成绩失败: " + e.getMessage());
        }
    }

    @Operation(summary = "批量录入成绩", description = "批量为多个学生录入成绩")
    @PostMapping("/batch")
    public Result<GradeDTO.BatchGradeResponse> batchCreateGrades(@RequestBody List<GradeDTO.GradeCreateRequest> batchCreateRequest) {
        try {
            // 1. 获取当前教师ID
            Long teacherId = getCurrentTeacherId();
            
            // 2. 调用服务层批量录入成绩
            GradeDTO.BatchGradeResponse result = gradeService.batchCreateGrades(batchCreateRequest, teacherId);
            
            return Result.success(result);
        } catch (Exception e) {
            return Result.error("批量录入成绩失败: " + e.getMessage());
        }
    }

    @Operation(summary = "更新成绩", description = "更新已录入的成绩")
    @PutMapping("/{gradeId}")
    public Result<GradeDTO.GradeResponse> updateGrade(@PathVariable Long gradeId, @RequestBody GradeDTO.GradeUpdateRequest updateRequest) {
        try {
            // 1. 获取当前教师ID
            Long teacherId = getCurrentTeacherId();
            
            // 2. 调用服务层更新成绩
            GradeDTO.GradeResponse result = gradeService.updateGrade(gradeId, updateRequest, teacherId);
            
            return Result.success(result);
        } catch (Exception e) {
            return Result.error("更新成绩失败: " + e.getMessage());
        }
    }

    @Operation(summary = "删除成绩", description = "删除成绩记录")
    @DeleteMapping("/{gradeId}")
    public Result<Void> deleteGrade(@PathVariable Long gradeId) {
        try {
            // 1. 获取当前教师ID
            Long teacherId = getCurrentTeacherId();
            
            // 2. 调用服务层删除成绩
            Boolean result = gradeService.deleteGrade(gradeId, teacherId);
            
            if (result) {
                return Result.success();
            } else {
                return Result.error("删除成绩失败");
            }
        } catch (Exception e) {
            return Result.error("删除成绩失败: " + e.getMessage());
        }
    }

    @Operation(summary = "发布成绩", description = "发布指定成绩")
    @PostMapping("/{gradeId}/publish")
    public Result<Void> publishGrade(@PathVariable Long gradeId) {
        try {
            // 1. 获取当前教师ID
            Long teacherId = getCurrentTeacherId();
            
            // 2. 调用服务层发布成绩
            Boolean result = gradeService.publishGrades(Arrays.asList(gradeId), teacherId);
            
            if (result) {
                return Result.success();
            } else {
                return Result.error("发布成绩失败");
            }
        } catch (Exception e) {
            return Result.error("发布成绩失败: " + e.getMessage());
        }
    }

    @Operation(summary = "批量发布成绩", description = "批量发布多个成绩")
    @PostMapping("/batch-publish")
    public Result<Void> batchPublishGrades(
            @RequestParam Long courseId,
            @RequestParam(required = false) Long taskId) {
        try {
            // 1. 获取当前教师ID
            Long teacherId = getCurrentTeacherId();
            
            // 2. 调用服务层批量发布成绩
            Boolean result = gradeService.batchPublishGrades(courseId, taskId, teacherId);
            
            if (result) {
                return Result.success();
            } else {
                return Result.error("批量发布成绩失败");
            }
        } catch (Exception e) {
            return Result.error("批量发布成绩失败: " + e.getMessage());
        }
    }

    @Operation(summary = "获取成绩统计", description = "获取班级或课程的成绩统计")
    @GetMapping("/statistics")
    public Result<GradeDTO.GradeStatisticsResponse> getGradeStatistics(
            @RequestParam Long courseId,
            @RequestParam(required = false) Long taskId) {
        try {
            // 1. 获取当前教师ID
            Long teacherId = getCurrentTeacherId();
            
            // 2. 调用服务层获取成绩统计
            GradeDTO.GradeStatisticsResponse result = gradeService.getGradeStatistics(courseId, taskId, teacherId);
            
            return Result.success(result);
        } catch (Exception e) {
            return Result.error("获取成绩统计失败: " + e.getMessage());
        }
    }

    @Operation(summary = "获取学生成绩详情", description = "获取指定学生的成绩详情")
    @GetMapping("/student/{studentId}")
    public Result<GradeDTO.StudentGradeDetailResponse> getStudentGrades(
            @PathVariable Long studentId,
            @RequestParam Long courseId) {
        try {
            // 1. 获取当前教师ID
            Long teacherId = getCurrentTeacherId();
            
            // 2. 调用服务层获取学生成绩详情
            GradeDTO.StudentGradeDetailResponse result = gradeService.getStudentGradeDetail(studentId, courseId, teacherId);
            
            return Result.success(result);
        } catch (Exception e) {
            return Result.error("获取学生成绩详情失败: " + e.getMessage());
        }
    }

    @Operation(summary = "导出成绩", description = "导出成绩到Excel")
    @PostMapping("/export")
    public Result<String> exportGrades(@RequestBody GradeDTO.GradeExportRequest exportRequest) {
        try {
            // 1. 获取当前教师ID
            Long teacherId = getCurrentTeacherId();
            
            // 2. 调用服务层导出成绩
            String result = gradeService.exportGrades(exportRequest, teacherId);
            
            return Result.success(result);
        } catch (Exception e) {
            return Result.error("导出成绩失败: " + e.getMessage());
        }
    }

    @Operation(summary = "导入成绩", description = "通过Excel批量导入成绩")
    @PostMapping("/import")
    public Result<GradeDTO.GradeImportResponse> importGrades(@RequestBody GradeDTO.GradeImportRequest importRequest) {
        try {
            // 1. 获取当前教师ID
            Long teacherId = getCurrentTeacherId();
            
            // 2. 调用服务层导入成绩
            GradeDTO.GradeImportResponse result = gradeService.importGrades(importRequest, teacherId);
            
            return Result.success(result);
        } catch (Exception e) {
            return Result.error("导入成绩失败: " + e.getMessage());
        }
    }

    @Operation(summary = "获取成绩分布", description = "获取成绩分布图表数据")
    @GetMapping("/distribution")
    public Result<GradeDTO.GradeDistributionResponse> getGradeDistribution(
            @RequestParam Long courseId,
            @RequestParam(required = false) Long taskId) {
        try {
            // 1. 获取当前教师ID
            Long teacherId = getCurrentTeacherId();
            
            // 2. 调用服务层获取成绩分布
            GradeDTO.GradeDistributionResponse result = gradeService.getGradeDistribution(courseId, taskId, teacherId);
            
            return Result.success(result);
        } catch (Exception e) {
            return Result.error("获取成绩分布失败: " + e.getMessage());
        }
    }

    @Operation(summary = "获取成绩趋势", description = "获取学生成绩变化趋势")
    @GetMapping("/trend")
    public Result<GradeDTO.GradeTrendResponse> getGradeTrend(
            @RequestParam Long studentId,
            @RequestParam Long courseId,
            @RequestParam(required = false) String timeRange) {
        try {
            // 1. 获取当前教师ID
            Long teacherId = getCurrentTeacherId();
            
            // 2. 调用服务层获取成绩趋势
            GradeDTO.GradeTrendResponse result = gradeService.getGradeTrend(studentId, courseId, timeRange, teacherId);
            
            return Result.success(result);
        } catch (Exception e) {
            return Result.error("获取成绩趋势失败: " + e.getMessage());
        }
    }

    @Operation(summary = "成绩排名", description = "获取班级或课程的成绩排名")
    @GetMapping("/ranking")
    public Result<List<GradeDTO.GradeRankingResponse>> getGradeRanking(
            @RequestParam Long courseId,
            @RequestParam(required = false) Long taskId) {
        try {
            // 1. 获取当前教师ID
            Long teacherId = getCurrentTeacherId();
            
            // 2. 调用服务层获取成绩排名
            List<GradeDTO.GradeRankingResponse> result = gradeService.getGradeRanking(courseId, taskId, teacherId);
            
            return Result.success(result);
        } catch (Exception e) {
            return Result.error("获取成绩排名失败: " + e.getMessage());
        }
    }

    /**
     * 获取当前教师ID
     * 从JWT token或session中获取当前登录教师的ID
     */
    private Long getCurrentTeacherId() {
        try {
            String token = JwtUtils.getTokenFromRequest(request);
            if (token != null) {
                return jwtUtils.getUserIdFromToken(token);
            }
            throw new RuntimeException("未找到有效的认证令牌");
        } catch (Exception e) {
            throw new RuntimeException("获取当前用户信息失败: " + e.getMessage());
        }
    }


}