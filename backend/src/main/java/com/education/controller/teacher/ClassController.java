package com.education.controller.teacher;

import com.education.dto.ClassDTO;
import com.education.dto.common.PageRequest;
import com.education.dto.common.Result;
import com.education.service.teacher.ClassService;
import com.education.utils.JwtUtils;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.servlet.http.HttpServletRequest;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.List;

/**
 * 教师端班级管理控制器
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
@Tag(name = "教师端-班级管理", description = "教师班级创建、管理、学生管理等接口")
@RestController
@RequestMapping("/api/teacher/classes")
public class ClassController {

    @Autowired
    private ClassService classService;
    
    @Autowired
    private JwtUtils jwtUtils;
    
    @Autowired
    private HttpServletRequest request;

    @Operation(summary = "创建班级", description = "教师创建新班级")
    @PostMapping
    public Result<ClassDTO.ClassResponse> createClass(@RequestBody ClassDTO.ClassCreateRequest createRequest) {
        try {
            // 1. 获取当前教师ID
            Long teacherId = getCurrentTeacherId();
            
            // 2. 调用服务层创建班级
            ClassDTO.ClassResponse result = classService.createClass(createRequest, teacherId);
            
            return Result.success(result);
        } catch (Exception e) {
            return Result.error("创建班级失败: " + e.getMessage());
        }
    }

    @Operation(summary = "获取我的班级列表", description = "获取当前教师创建的所有班级")
    @GetMapping
    public Result<Object> getMyClasses(
            @RequestParam(defaultValue = "1") Integer page,
            @RequestParam(defaultValue = "10") Integer size) {
        try {
            // 1. 获取当前教师ID
            Long teacherId = getCurrentTeacherId();
            
            // 2. 构建查询参数
            PageRequest pageRequest = new PageRequest();
            pageRequest.setPageNum(page);
            pageRequest.setPageSize(size);
            
            // 3. 调用服务层获取班级列表
            Object result = classService.getClassList(teacherId, pageRequest);
            
            return Result.success(result);
        } catch (Exception e) {
            return Result.error("获取班级列表失败: " + e.getMessage());
        }
    }

    @Operation(summary = "获取班级详情", description = "获取指定班级的详细信息")
    @GetMapping("/{classId}")
    public Result<ClassDTO.ClassResponse> getClassDetail(@PathVariable Long classId) {
        try {
            // 1. 获取当前教师ID
            Long teacherId = getCurrentTeacherId();
            
            // 2. 调用服务层获取班级详情
            ClassDTO.ClassResponse result = classService.getClassDetail(classId, teacherId);
            
            return Result.success(result);
        } catch (Exception e) {
            return Result.error("获取班级详情失败: " + e.getMessage());
        }
    }

    @Operation(summary = "更新班级信息", description = "更新班级基本信息")
    @PutMapping("/{classId}")
    public Result<ClassDTO.ClassResponse> updateClass(@PathVariable Long classId, @RequestBody ClassDTO.ClassUpdateRequest updateRequest) {
        try {
            // 1. 获取当前教师ID
            Long teacherId = getCurrentTeacherId();
            
            // 2. 调用服务层更新班级
            ClassDTO.ClassResponse result = classService.updateClass(classId, updateRequest, teacherId);
            
            return Result.success(result);
        } catch (Exception e) {
            return Result.error("更新班级失败: " + e.getMessage());
        }
    }

    @Operation(summary = "删除班级", description = "删除指定班级")
    @DeleteMapping("/{classId}")
    public Result<Void> deleteClass(@PathVariable Long classId) {
        try {
            Long teacherId = getCurrentTeacherId();
            classService.deleteClass(classId, teacherId);
            return Result.success();
        } catch (Exception e) {
            return Result.error("删除班级失败: " + e.getMessage());
        }
    }

    @Operation(summary = "获取班级学生列表", description = "获取班级内所有学生信息")
    @GetMapping("/{classId}/students")
    public Result<Object> getClassStudents(
            @PathVariable Long classId,
            @RequestParam(defaultValue = "1") Integer page,
            @RequestParam(defaultValue = "10") Integer size,
            @RequestParam(required = false) String keyword) {
        try {
            Long teacherId = getCurrentTeacherId();
            PageRequest pageRequest = new PageRequest();
            pageRequest.setPageNum(page);
            pageRequest.setPageSize(size);
            pageRequest.setKeyword(keyword);
            Object students = classService.getClassStudents(classId, teacherId, pageRequest);
            return Result.success(students);
        } catch (Exception e) {
            return Result.error("获取班级学生列表失败: " + e.getMessage());
        }
    }

    @Operation(summary = "移除班级学生", description = "将学生从班级中移除")
    @DeleteMapping("/{classId}/students/{studentId}")
    public Result<Void> removeStudent(@PathVariable Long classId, @PathVariable Long studentId) {
        try {
            Long teacherId = getCurrentTeacherId();
            classService.removeStudent(classId, studentId, teacherId);
            return Result.success();
        } catch (Exception e) {
            return Result.error("移除学生失败: " + e.getMessage());
        }
    }

    @Operation(summary = "批量移除学生", description = "批量将学生从班级中移除")
    @DeleteMapping("/{classId}/students")
    public Result<Void> removeStudents(@PathVariable Long classId, @RequestBody List<Long> studentIds) {
        try {
            Long teacherId = getCurrentTeacherId();
            classService.removeStudents(classId, studentIds, teacherId);
            return Result.success();
        } catch (Exception e) {
            return Result.error("批量移除学生失败: " + e.getMessage());
        }
    }

    @Operation(summary = "生成班级邀请码", description = "重新生成班级邀请码")
    @PostMapping("/{classId}/invite-code")
    public Result<Object> generateInviteCode(@PathVariable Long classId) {
        try {
            Long teacherId = getCurrentTeacherId();
            Object inviteCode = classService.generateInviteCode(classId, teacherId, 24); // 默认24小时过期
            return Result.success(inviteCode);
        } catch (Exception e) {
            return Result.error("生成邀请码失败: " + e.getMessage());
        }
    }

    @Operation(summary = "获取班级统计信息", description = "获取班级的统计数据")
    @GetMapping("/{classId}/statistics")
    public Result<Object> getClassStatistics(@PathVariable Long classId) {
        try {
            Long teacherId = getCurrentTeacherId();
            Object statistics = classService.getClassStatistics(classId, teacherId);
            return Result.success(statistics);
        } catch (Exception e) {
            return Result.error("获取班级统计信息失败: " + e.getMessage());
        }
    }

    @Operation(summary = "导出班级学生信息", description = "导出班级学生信息到Excel")
    @GetMapping("/{classId}/export")
    public Result<Object> exportClassStudents(@PathVariable Long classId) {
        try {
            Long teacherId = getCurrentTeacherId();
            Object exportResult = classService.exportClassStudents(classId, teacherId);
            return Result.success(exportResult);
        } catch (Exception e) {
            return Result.error("导出学生信息失败: " + e.getMessage());
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