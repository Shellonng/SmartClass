package com.education.controller.student;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.education.dto.common.Result;
import com.education.entity.*;
import com.education.mapper.*;
import com.education.security.SecurityUtil;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.*;
import java.util.stream.Collectors;

/**
 * 学生班级控制器
 */
@Tag(name = "学生-班级管理", description = "学生班级相关接口")
@RestController
@RequestMapping("/api/student/classes")
public class StudentClassController {

    private static final Logger logger = LoggerFactory.getLogger(StudentClassController.class);

    @Autowired
    private SecurityUtil securityUtil;

    @Autowired
    private UserMapper userMapper;

    @Autowired
    private StudentMapper studentMapper;

    @Autowired
    private ClassStudentMapper classStudentMapper;

    @Autowired
    private CourseClassMapper courseClassMapper;

    @Autowired
    private CourseMapper courseMapper;

    /**
     * 获取当前学生所在的班级信息
     * @return 班级信息
     */
    @Operation(summary = "获取班级信息", description = "获取当前登录学生所在的班级信息")
    @GetMapping("/info")
    public Result getClassInfo() {
        try {
            // 获取当前登录用户ID
            Long userId = securityUtil.getCurrentUserId();
            if (userId == null) {
                return Result.error("未登录");
            }

            // 获取学生信息
            Student student = studentMapper.selectOne(
                new LambdaQueryWrapper<Student>()
                    .eq(Student::getUserId, userId)
            );
            
            if (student == null) {
                return Result.error("学生信息不存在");
            }

            // 查询学生所在的班级
            List<ClassStudent> classStudents = classStudentMapper.selectList(
                new LambdaQueryWrapper<ClassStudent>()
                    .eq(ClassStudent::getStudentId, student.getId())
            );

            if (classStudents.isEmpty()) {
                return Result.error("未找到班级信息");
            }

            // 获取班级ID
            Long classId = classStudents.get(0).getClassId();

            // 查询班级详细信息
            CourseClass courseClass = courseClassMapper.selectById(classId);
            if (courseClass == null) {
                return Result.error("班级不存在");
            }

            // 获取关联课程信息
            if (courseClass.getCourseId() != null) {
                Course course = courseMapper.selectById(courseClass.getCourseId());
                courseClass.setCourse(course);
            }

            // 获取班级学生人数
            Integer studentCount = classStudentMapper.selectCount(
                new LambdaQueryWrapper<ClassStudent>()
                    .eq(ClassStudent::getClassId, classId)
            ).intValue();
            
            courseClass.setStudentCount(studentCount);

            return Result.success(courseClass);
        } catch (Exception e) {
            logger.error("获取班级信息失败", e);
            return Result.error("获取班级信息失败: " + e.getMessage());
        }
    }

    /**
     * 获取班级同学列表
     * @return 同学列表
     */
    @Operation(summary = "获取班级同学列表", description = "获取当前登录学生所在班级的同学列表")
    @GetMapping("/members")
    public Result getClassMembers() {
        try {
            // 获取当前登录用户ID
            Long userId = securityUtil.getCurrentUserId();
            if (userId == null) {
                return Result.error("未登录");
            }

            // 获取学生信息
            Student student = studentMapper.selectOne(
                new LambdaQueryWrapper<Student>()
                    .eq(Student::getUserId, userId)
            );
            
            if (student == null) {
                return Result.error("学生信息不存在");
            }

            // 查询学生所在的班级
            List<ClassStudent> classStudents = classStudentMapper.selectList(
                new LambdaQueryWrapper<ClassStudent>()
                    .eq(ClassStudent::getStudentId, student.getId())
            );

            if (classStudents.isEmpty()) {
                return Result.error("未找到班级信息");
            }

            // 获取班级ID
            Long classId = classStudents.get(0).getClassId();

            // 查询班级所有学生
            List<ClassStudent> allClassStudents = classStudentMapper.selectList(
                new LambdaQueryWrapper<ClassStudent>()
                    .eq(ClassStudent::getClassId, classId)
            );

            // 获取所有学生ID
            List<Long> studentIds = allClassStudents.stream()
                .map(ClassStudent::getStudentId)
                .collect(Collectors.toList());

            // 查询学生信息
            List<Student> students = new ArrayList<>();
            if (!studentIds.isEmpty()) {
                students = studentMapper.selectBatchIds(studentIds);
                
                // 获取所有用户ID
                List<Long> userIds = students.stream()
                    .map(Student::getUserId)
                    .collect(Collectors.toList());
                
                // 查询用户信息
                List<User> users = userMapper.selectBatchIds(userIds);
                Map<Long, User> userMap = users.stream()
                    .collect(Collectors.toMap(User::getId, user -> user));
                
                // 关联用户信息
                students.forEach(s -> s.setUser(userMap.get(s.getUserId())));
            }

            // 构建返回结果
            Map<String, Object> result = new HashMap<>();
            result.put("classId", classId);
            result.put("students", students);
            
            return Result.success(result);
        } catch (Exception e) {
            logger.error("获取班级同学列表失败", e);
            return Result.error("获取班级同学列表失败: " + e.getMessage());
        }
    }
} 