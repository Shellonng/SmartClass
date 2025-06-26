package com.education.controller.teacher;

import com.education.dto.common.Result;
import com.education.dto.common.PageRequest;
import com.education.dto.common.PageResponse;
import com.education.dto.clazz.ClassCreateRequest;
import com.education.dto.clazz.ClassUpdateRequest;
import com.education.dto.clazz.ClassResponse;
import com.education.dto.clazz.ClassDetailResponse;
import com.education.dto.clazz.ClassStudentResponse;
import com.education.service.teacher.ClassService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.*;

import jakarta.validation.Valid;
import java.util.List;

/**
 * 教师端班级管理控制器
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
@Tag(name = "教师端班级管理", description = "班级的创建、编辑、删除、查询等功能")
@RestController
@RequestMapping("/api/teacher/classes")
@RequiredArgsConstructor
@Slf4j
public class ClassController {

    private final ClassService classService;

    /**
     * 分页查询班级列表
     */
    @Operation(summary = "分页查询班级列表")
    @GetMapping
    public Result<PageResponse<ClassResponse>> getClassList(
            @Valid PageRequest pageRequest,
            @RequestParam(required = false) String name,
            @RequestParam(required = false) String grade,
            @RequestParam(required = false) String status) {
        log.info("分页查询班级列表，页码：{}，页大小：{}，班级名称：{}，年级：{}，状态：{}", 
                pageRequest.getCurrent(), pageRequest.getPageSize(), name, grade, status);
        
        PageResponse<ClassResponse> response = classService.getClassList(pageRequest, name, grade, status);
        return Result.success(response);
    }

    /**
     * 创建班级
     */
    @Operation(summary = "创建班级")
    @PostMapping
    public Result<ClassResponse> createClass(@Valid @RequestBody ClassCreateRequest request) {
        log.info("创建班级，班级名称：{}，年级：{}，专业：{}", 
                request.getName(), request.getGrade(), request.getMajor());
        
        ClassResponse response = classService.createClass(request);
        return Result.success("班级创建成功", response);
    }

    /**
     * 获取班级详情
     */
    @Operation(summary = "获取班级详情")
    @GetMapping("/{classId}")
    public Result<ClassDetailResponse> getClassDetail(@PathVariable Long classId) {
        log.info("获取班级详情，班级ID：{}", classId);
        
        ClassDetailResponse response = classService.getClassDetail(classId);
        return Result.success(response);
    }

    /**
     * 更新班级信息
     */
    @Operation(summary = "更新班级信息")
    @PutMapping("/{classId}")
    public Result<ClassResponse> updateClass(
            @PathVariable Long classId, 
            @Valid @RequestBody ClassUpdateRequest request) {
        log.info("更新班级信息，班级ID：{}，班级名称：{}", classId, request.getName());
        
        ClassResponse response = classService.updateClass(classId, request);
        return Result.success("班级信息更新成功", response);
    }

    /**
     * 删除班级
     */
    @Operation(summary = "删除班级")
    @DeleteMapping("/{classId}")
    public Result<Void> deleteClass(@PathVariable Long classId) {
        log.info("删除班级，班级ID：{}", classId);
        
        classService.deleteClass(classId);
        return Result.success("班级删除成功");
    }

    /**
     * 获取班级学生列表
     */
    @Operation(summary = "获取班级学生列表")
    @GetMapping("/{classId}/students")
    public Result<PageResponse<ClassStudentResponse>> getClassStudents(
            @PathVariable Long classId,
            @Valid PageRequest pageRequest,
            @RequestParam(required = false) String keyword) {
        log.info("获取班级学生列表，班级ID：{}，关键词：{}", classId, keyword);
        
        PageResponse<ClassStudentResponse> response = classService.getClassStudents(classId, pageRequest, keyword);
        return Result.success(response);
    }

    /**
     * 添加学生到班级
     */
    @Operation(summary = "添加学生到班级")
    @PostMapping("/{classId}/students")
    public Result<Void> addStudentsToClass(
            @PathVariable Long classId,
            @RequestBody List<Long> studentIds) {
        log.info("添加学生到班级，班级ID：{}，学生数量：{}", classId, studentIds.size());
        
        classService.addStudentsToClass(classId, studentIds);
        return Result.success("学生添加成功");
    }

    /**
     * 从班级移除学生
     */
    @Operation(summary = "从班级移除学生")
    @DeleteMapping("/{classId}/students/{studentId}")
    public Result<Void> removeStudentFromClass(
            @PathVariable Long classId,
            @PathVariable Long studentId) {
        log.info("从班级移除学生，班级ID：{}，学生ID：{}", classId, studentId);
        
        classService.removeStudentFromClass(classId, studentId);
        return Result.success("学生移除成功");
    }

    /**
     * 批量从班级移除学生
     */
    @Operation(summary = "批量从班级移除学生")
    @DeleteMapping("/{classId}/students")
    public Result<Void> removeStudentsFromClass(
            @PathVariable Long classId,
            @RequestBody List<Long> studentIds) {
        log.info("批量从班级移除学生，班级ID：{}，学生数量：{}", classId, studentIds.size());
        
        classService.removeStudentsFromClass(classId, studentIds);
        return Result.success("学生批量移除成功");
    }

    /**
     * 获取班级统计信息
     */
    @Operation(summary = "获取班级统计信息")
    @GetMapping("/{classId}/statistics")
    public Result<Object> getClassStatistics(@PathVariable Long classId) {
        log.info("获取班级统计信息，班级ID：{}", classId);
        
        Object statistics = classService.getClassStatistics(classId);
        return Result.success(statistics);
    }

    /**
     * 设置班级状态
     */
    @Operation(summary = "设置班级状态")
    @PutMapping("/{classId}/status")
    public Result<Void> updateClassStatus(
            @PathVariable Long classId,
            @RequestParam String status) {
        log.info("设置班级状态，班级ID：{}，状态：{}", classId, status);
        
        classService.updateClassStatus(classId, status);
        return Result.success("班级状态更新成功");
    }

    /**
     * 复制班级
     */
    @Operation(summary = "复制班级")
    @PostMapping("/{classId}/copy")
    public Result<ClassResponse> copyClass(
            @PathVariable Long classId,
            @RequestParam String newName) {
        log.info("复制班级，原班级ID：{}，新班级名称：{}", classId, newName);
        
        ClassResponse response = classService.copyClass(classId, newName);
        return Result.success("班级复制成功", response);
    }

    /**
     * 导出班级学生名单
     */
    @Operation(summary = "导出班级学生名单")
    @GetMapping("/{classId}/export")
    public Result<String> exportClassStudents(@PathVariable Long classId) {
        log.info("导出班级学生名单，班级ID：{}", classId);
        
        String downloadUrl = classService.exportClassStudents(classId);
        return Result.success("学生名单导出成功", downloadUrl);
    }
}