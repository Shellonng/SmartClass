package com.education.controller.teacher;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.education.dto.common.PageRequest;
import com.education.dto.common.PageResponse;
import com.education.dto.common.Result;
import com.education.entity.Course;
import com.education.entity.CourseClass;
import com.education.entity.Student;
import com.education.mapper.CourseMapper;
import com.education.security.SecurityUtil;
import com.education.service.teacher.ClassService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.*;

import java.util.Collections;
import java.util.List;

/**
 * 班级管理控制器
 */
@RestController
@RequestMapping("/api/teacher/classes")
@Tag(name = "班级管理", description = "教师班级管理相关接口")
@RequiredArgsConstructor
@Slf4j
public class ClassController {

    private final ClassService classService;
    private final CourseMapper courseMapper;
    private final SecurityUtil securityUtil;

    @GetMapping
    @Operation(summary = "获取班级列表", description = "分页获取当前教师的班级列表")
    public Result<PageResponse<CourseClass>> getClasses(
            @Parameter(description = "分页参数") PageRequest pageRequest,
            @Parameter(description = "搜索关键词") @RequestParam(required = false) String keyword,
            @Parameter(description = "课程ID") @RequestParam(required = false) Long courseId) {
        
        PageResponse<CourseClass> response = classService.getClassesByTeacher(
                pageRequest.getCurrent() - 1, // 前端是0-based索引
                pageRequest.getPageSize(),
                keyword,
                courseId
        );
        
        return Result.success(response);
    }

    @GetMapping("/{id}")
    @Operation(summary = "获取班级详情", description = "根据ID获取班级详细信息")
    public Result<CourseClass> getClassDetail(
            @Parameter(description = "班级ID") @PathVariable Long id) {
        
        CourseClass courseClass = classService.getClassById(id);
        return Result.success(courseClass);
    }

    @PostMapping
    @Operation(summary = "创建班级", description = "创建新的班级")
    public Result<CourseClass> createClass(
            @Parameter(description = "班级信息") @RequestBody CourseClass courseClass) {
        
        CourseClass createdClass = classService.createClass(courseClass);
        return Result.success(createdClass);
    }

    @PutMapping("/{id}")
    @Operation(summary = "更新班级", description = "更新班级信息")
    public Result<CourseClass> updateClass(
            @Parameter(description = "班级ID") @PathVariable Long id,
            @Parameter(description = "班级信息") @RequestBody CourseClass courseClass) {
        
        // 设置ID
        courseClass.setId(id);
        
        // 记录接收到的班级信息，特别关注课程ID
        log.info("接收到更新班级请求，班级ID: {}, 班级名称: {}, 课程ID: {}", 
            id, courseClass.getName(), courseClass.getCourseId());
        
        CourseClass updatedClass = classService.updateClass(courseClass);
        
        // 记录更新后的班级信息
        log.info("班级更新成功，ID: {}, 名称: {}, 课程ID: {}", 
            updatedClass.getId(), updatedClass.getName(), updatedClass.getCourseId());
        
        return Result.success(updatedClass);
    }

    @DeleteMapping("/{id}")
    @Operation(summary = "删除班级", description = "删除指定班级")
    public Result<Void> deleteClass(
            @Parameter(description = "班级ID") @PathVariable Long id) {
        
        classService.deleteClass(id);
        return Result.success();
    }

    @GetMapping("/{id}/students")
    @Operation(summary = "获取班级学生列表", description = "分页获取班级学生列表")
    public Result<PageResponse<Student>> getClassStudents(
            @Parameter(description = "班级ID") @PathVariable Long id,
            @Parameter(description = "分页参数") PageRequest pageRequest,
            @Parameter(description = "搜索关键词") @RequestParam(required = false) String keyword) {
        
        PageResponse<Student> response = classService.getStudentsByClassId(
                id,
                pageRequest.getCurrent() - 1, // 前端是0-based索引
                pageRequest.getPageSize(),
                keyword
        );
        
        return Result.success(response);
    }

    @PostMapping("/{id}/students")
    @Operation(summary = "添加学生到班级", description = "批量添加学生到班级")
    public Result<Void> addStudentsToClass(
            @Parameter(description = "班级ID") @PathVariable Long id,
            @Parameter(description = "学生ID列表") @RequestBody List<Long> studentIds) {
        
        classService.addStudentsToClass(id, studentIds);
        return Result.success();
    }

    @DeleteMapping("/{id}/students/{studentId}")
    @Operation(summary = "从班级移除学生", description = "从班级中移除指定学生")
    public Result<Void> removeStudentFromClass(
            @Parameter(description = "班级ID") @PathVariable Long id,
            @Parameter(description = "学生ID") @PathVariable Long studentId) {
        
        classService.removeStudentFromClass(id, studentId);
        return Result.success();
    }

    /**
     * 获取当前教师的课程列表（用于班级绑定）
     */
    @GetMapping("/courses")
    @Operation(summary = "获取当前教师的课程列表", description = "获取当前教师的课程列表，用于班级绑定")
    public Result<List<Course>> getTeacherCourses() {
        // 获取当前登录的用户ID
        Long userId = securityUtil.getCurrentUserId();
        
        // 打印日志，便于调试
        log.info("当前用户ID: {}", userId);
        
        // 查询用户ID对应的教师记录
        Long teacherId = classService.getTeacherIdByUserId(userId);
        
        // 打印日志，便于调试
        log.info("查询到的教师ID: {}", teacherId);
        
        if (teacherId == null) {
            log.warn("未找到当前用户对应的教师信息，用户ID: {}", userId);
            return Result.success(Collections.emptyList());
        }
        
        // 使用教师ID查询课程
        List<Course> courses = courseMapper.selectList(
            new LambdaQueryWrapper<Course>()
                .eq(Course::getTeacherId, teacherId)
                .orderByDesc(Course::getCreateTime)
        );
        
        log.info("查询到{}门课程", courses.size());
        return Result.success(courses);
    }
} 