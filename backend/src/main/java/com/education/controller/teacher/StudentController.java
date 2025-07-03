package com.education.controller.teacher;

import com.education.dto.common.PageRequest;
import com.education.dto.common.PageResponse;
import com.education.dto.common.Result;
import com.education.entity.ClassStudent;
import com.education.entity.CourseClass;
import com.education.entity.Student;
import com.education.entity.User;
import com.education.security.SecurityUtil;
import com.education.service.teacher.StudentManagementService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

/**
 * 学生管理控制器
 */
@RestController
@RequestMapping("/api/teacher/students")
@Tag(name = "学生管理", description = "教师学生管理相关接口")
@RequiredArgsConstructor
@Slf4j
public class StudentController {

    private final StudentManagementService studentManagementService;
    private final SecurityUtil securityUtil;

    @GetMapping
    @Operation(summary = "获取学生列表", description = "分页获取学生列表，可按班级、课程筛选")
    public Result<PageResponse<Student>> getStudents(
            @Parameter(description = "分页参数") PageRequest pageRequest,
            @Parameter(description = "搜索关键词(学生姓名或学号)") @RequestParam(required = false) String keyword,
            @Parameter(description = "班级ID") @RequestParam(required = false) Long classId,
            @Parameter(description = "课程ID") @RequestParam(required = false) Long courseId) {
        
        PageResponse<Student> response = studentManagementService.getStudents(
                pageRequest.getCurrent() - 1, // 前端是1-based索引
                pageRequest.getPageSize(),
                keyword,
                classId,
                courseId
        );
        
        return Result.success(response);
    }
    
    @GetMapping("/{id}")
    @Operation(summary = "获取学生详情", description = "根据ID获取学生详细信息")
    public Result<Student> getStudentDetail(
            @Parameter(description = "学生ID") @PathVariable Long id) {
        
        Student student = studentManagementService.getStudentById(id);
        return Result.success(student);
    }
    
    @GetMapping("/classes")
    @Operation(summary = "获取班级列表", description = "获取当前教师的班级列表，用于学生管理")
    public Result<List<CourseClass>> getTeacherClasses() {
        Long teacherId = securityUtil.getCurrentUserId();
        List<CourseClass> classes = studentManagementService.getClassesByTeacherId(teacherId);
        return Result.success(classes);
    }
    
    @GetMapping("/search")
    @Operation(summary = "搜索学生", description = "模糊搜索学生，用于添加学生到课程或班级")
    public Result<List<Map<String, Object>>> searchStudents(
            @Parameter(description = "搜索关键词(学生姓名或学号)") @RequestParam(required = false) String keyword) {
        
        List<Map<String, Object>> students = studentManagementService.searchStudents(keyword);
        return Result.success(students);
    }
    
    @PostMapping("/add-to-class")
    @Operation(summary = "添加学生到班级", description = "添加学生到指定班级")
    public Result<Void> addStudentToClass(
            @Parameter(description = "学生班级关联信息") @RequestBody ClassStudent classStudent) {
        
        studentManagementService.addStudentToClass(classStudent.getStudentId(), classStudent.getClassId());
        return Result.success();
    }
    
    @DeleteMapping("/remove-from-class")
    @Operation(summary = "从班级移除学生", description = "从指定班级移除学生")
    public Result<Void> removeStudentFromClass(
            @Parameter(description = "学生班级关联信息") @RequestBody ClassStudent classStudent) {
        
        studentManagementService.removeStudentFromClass(classStudent.getStudentId(), classStudent.getClassId());
        return Result.success();
    }
    
    @PostMapping("/add-to-course")
    @Operation(summary = "添加学生到课程", description = "添加学生到指定课程")
    public Result<Void> addStudentToCourse(
            @Parameter(description = "学生课程关联信息") @RequestBody Map<String, Object> params) {
        
        Long studentId = Long.valueOf(params.get("studentId").toString());
        Long courseId = Long.valueOf(params.get("courseId").toString());
        
        studentManagementService.addStudentToCourse(studentId, courseId);
        return Result.success();
    }
    
    @DeleteMapping("/remove-from-course")
    @Operation(summary = "从课程移除学生", description = "从指定课程移除学生")
    public Result<Void> removeStudentFromCourse(
            @Parameter(description = "学生课程关联信息") @RequestBody Map<String, Object> params) {
        
        Long studentId = Long.valueOf(params.get("studentId").toString());
        Long courseId = Long.valueOf(params.get("courseId").toString());
        
        studentManagementService.removeStudentFromCourse(studentId, courseId);
        return Result.success();
    }
    
    @PostMapping("/process-enrollment-request")
    @Operation(summary = "处理选课申请", description = "通过或拒绝学生的选课申请")
    public Result<Void> processEnrollmentRequest(
            @Parameter(description = "申请处理信息") @RequestBody Map<String, Object> params) {
        
        Long requestId = Long.valueOf(params.get("requestId").toString());
        Boolean approved = (Boolean) params.get("approved");
        String comment = (String) params.get("comment");
        
        studentManagementService.processEnrollmentRequest(requestId, approved, comment);
        return Result.success();
    }
    
    @GetMapping("/enrollment-requests")
    @Operation(summary = "获取选课申请列表", description = "获取待处理的选课申请")
    public Result<PageResponse<Map<String, Object>>> getEnrollmentRequests(
            @Parameter(description = "分页参数") PageRequest pageRequest,
            @Parameter(description = "课程ID") @RequestParam(required = false) Long courseId) {
        
        PageResponse<Map<String, Object>> response = studentManagementService.getEnrollmentRequests(
                pageRequest.getCurrent() - 1,
                pageRequest.getPageSize(),
                courseId
        );
        
        return Result.success(response);
    }
    
    @PostMapping("/create")
    @Operation(summary = "创建学生账户", description = "创建新的学生账户")
    public Result<User> createStudent(
            @Parameter(description = "学生信息") @RequestBody User user) {
        
        User createdUser = studentManagementService.createStudent(user);
        return Result.success(createdUser);
    }
    
    @PutMapping("/{id}")
    @Operation(summary = "更新学生信息", description = "更新学生基本信息")
    public Result<Student> updateStudent(
            @Parameter(description = "学生ID") @PathVariable Long id,
            @Parameter(description = "学生信息") @RequestBody Student student) {
        
        student.setId(id);
        Student updatedStudent = studentManagementService.updateStudent(student);
        return Result.success(updatedStudent);
    }
    
    @GetMapping("/{id}/classes")
    @Operation(summary = "获取学生所属班级", description = "获取指定学生所属的班级列表")
    public Result<List<Map<String, Object>>> getStudentClasses(
            @Parameter(description = "学生ID") @PathVariable Long id) {
        
        log.info("获取学生班级信息, 学生ID: {}", id);
        List<Map<String, Object>> classes = studentManagementService.getStudentClasses(id);
        return Result.success(classes);
    }
    
    @GetMapping("/{id}/courses")
    @Operation(summary = "获取学生所属课程", description = "获取指定学生所属的课程列表")
    public Result<List<Map<String, Object>>> getStudentCourses(
            @Parameter(description = "学生ID") @PathVariable Long id) {
        
        log.info("获取学生课程信息, 学生ID: {}", id);
        List<Map<String, Object>> courses = studentManagementService.getStudentCourses(id);
        return Result.success(courses);
    }
} 