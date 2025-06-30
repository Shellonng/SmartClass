package com.education.controller.teacher;

import com.education.dto.common.PageRequest;
import com.education.dto.common.PageResponse;
import com.education.dto.common.Result;
import com.education.entity.Course;
import com.education.service.teacher.CourseService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

/**
 * 教师课程管理控制器
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
@Tag(name = "教师-课程管理", description = "教师课程管理相关接口")
@RestController
@RequestMapping("/api/teacher/courses")
public class CourseController {

    private static final Logger logger = LoggerFactory.getLogger(CourseController.class);

    @Autowired
    private CourseService courseService;

    @Operation(summary = "获取课程列表", description = "获取当前教师的课程列表")
    @GetMapping
    public Result<PageResponse<Course>> getCourses(
            @Parameter(description = "页码") @RequestParam(defaultValue = "1") Integer page,
            @Parameter(description = "每页大小") @RequestParam(defaultValue = "10") Integer size,
            @Parameter(description = "关键词搜索") @RequestParam(required = false) String keyword,
            @Parameter(description = "状态筛选") @RequestParam(required = false) String status,
            @Parameter(description = "学期筛选") @RequestParam(required = false) String term) {
        
        PageRequest pageRequest = new PageRequest(page, size);
        Authentication authentication = SecurityContextHolder.getContext().getAuthentication();
        String username = authentication.getName();
        
        PageResponse<Course> courses = courseService.getTeacherCourses(username, pageRequest, keyword, status, term);
        return Result.success(courses);
    }

    @Operation(summary = "创建课程", description = "创建新课程")
    @PostMapping
    public Result<Course> createCourse(@RequestBody Course course) {
        Authentication authentication = SecurityContextHolder.getContext().getAuthentication();
        String username = authentication.getName();
        
        // 记录请求详情
        logger.info("===============================================");
        logger.info("接收到创建课程请求");
        logger.info("请求路径: /api/teacher/courses");
        logger.info("认证信息: {}", authentication);
        logger.info("用户名: {}", username);
        logger.info("认证类型: {}", authentication.getClass().getName());
        logger.info("认证权限: {}", authentication.getAuthorities());
        
        // 检查是否为匿名用户
        if (username.equals("anonymousUser")) {
            logger.error("匿名用户尝试创建课程，请先登录");
            return Result.error("请先登录后再创建课程");
        }
        
        logger.info("课程信息: title={}, description={}, coverImage={}, credit={}, courseType={}, startTime={}, endTime={}, teacherId={}, status={}, term={}, studentCount={}",
                course.getTitle(), 
                (course.getDescription() != null && course.getDescription().length() > 20) ? 
                    course.getDescription().substring(0, 20) + "..." : course.getDescription(),
                course.getCoverImage(),
                course.getCredit(),
                course.getCourseType(),
                course.getStartTime(),
                course.getEndTime(),
                course.getTeacherId(),
                course.getStatus(),
                course.getTerm(),
                course.getStudentCount());
        
        try {
            // 确保前端传递的课程名称正确设置
            if (course.getTitle() == null && course.getCourseName() != null) {
                course.setTitle(course.getCourseName());
                logger.info("从courseName字段设置title: {}", course.getTitle());
            }
            
            // 确保前端传递的课程类型正确设置
            if (course.getCourseType() == null && course.getCategory() != null) {
                course.setCourseType(course.getCategory());
                logger.info("从category字段设置courseType: {}", course.getCourseType());
            }
            
            // 确保前端传递的学期正确设置
            if (course.getTerm() == null && course.getSemester() != null) {
                course.setTerm(course.getSemester());
                logger.info("从semester字段设置term: {}", course.getTerm());
            }
            
            Course createdCourse = courseService.createCourse(username, course);
            logger.info("课程创建成功，ID: {}, 教师ID: {}", createdCourse.getId(), createdCourse.getTeacherId());
            logger.info("===============================================");
            return Result.success(createdCourse);
        } catch (Exception e) {
            logger.error("创建课程失败: {}", e.getMessage(), e);
            logger.info("===============================================");
            return Result.error("创建课程失败: " + e.getMessage());
        }
    }

    @Operation(summary = "获取课程详情", description = "根据ID获取课程详情")
    @GetMapping("/{id}")
    public Result<Course> getCourseDetail(@PathVariable Long id) {
        Authentication authentication = SecurityContextHolder.getContext().getAuthentication();
        String username = authentication.getName();
        
        Course course = courseService.getCourseDetail(username, id);
        return Result.success(course);
    }

    @Operation(summary = "更新课程", description = "更新课程信息")
    @PutMapping("/{id}")
    public Result<Course> updateCourse(@PathVariable Long id, @RequestBody Course course) {
        Authentication authentication = SecurityContextHolder.getContext().getAuthentication();
        String username = authentication.getName();
        
        course.setId(id);
        Course updatedCourse = courseService.updateCourse(username, course);
        return Result.success(updatedCourse);
    }

    @Operation(summary = "删除课程", description = "删除指定课程")
    @DeleteMapping("/{id}")
    public Result<Boolean> deleteCourse(@PathVariable String id) {
        Authentication authentication = SecurityContextHolder.getContext().getAuthentication();
        String username = authentication.getName();
        
        logger.info("===============================================");
        logger.info("接收到删除课程请求");
        logger.info("请求路径: /api/teacher/courses/{}", id);
        logger.info("认证信息: {}", authentication);
        logger.info("用户名: {}", username);
        
        try {
            // 将字符串ID转换为Long类型
            Long courseId = Long.parseLong(id);
            logger.info("解析课程ID: {} -> {}", id, courseId);
            
            boolean result = courseService.deleteCourse(username, courseId);
            logger.info("课程删除结果: {}", result);
            logger.info("===============================================");
            return Result.success(result);
        } catch (NumberFormatException e) {
            logger.error("课程ID格式错误: {}", id, e);
            logger.info("===============================================");
            return Result.error("课程ID格式错误");
        } catch (Exception e) {
            logger.error("删除课程失败: {}", e.getMessage(), e);
            logger.info("===============================================");
            return Result.error("删除课程失败: " + e.getMessage());
        }
    }

    @Operation(summary = "发布课程", description = "将课程状态改为已发布")
    @PostMapping("/{id}/publish")
    public Result<Course> publishCourse(@PathVariable Long id) {
        Authentication authentication = SecurityContextHolder.getContext().getAuthentication();
        String username = authentication.getName();
        
        Course course = courseService.publishCourse(username, id);
        return Result.success(course);
    }

    @Operation(summary = "取消发布课程", description = "将课程状态改为未发布")
    @PostMapping("/{id}/unpublish")
    public Result<Course> unpublishCourse(@PathVariable Long id) {
        Authentication authentication = SecurityContextHolder.getContext().getAuthentication();
        String username = authentication.getName();
        
        Course course = courseService.unpublishCourse(username, id);
        return Result.success(course);
    }

    @Operation(summary = "获取课程统计信息", description = "获取课程的统计数据")
    @GetMapping("/{id}/statistics")
    public Result<Map<String, Object>> getCourseStatistics(@PathVariable Long id) {
        Authentication authentication = SecurityContextHolder.getContext().getAuthentication();
        String username = authentication.getName();
        
        Map<String, Object> statistics = courseService.getCourseStatistics(username, id);
        return Result.success(statistics);
    }
} 