package com.education.controller.teacher;

import com.education.dto.CourseDTO;
import com.education.dto.common.PageRequest;
import com.education.dto.common.PageResponse;
import com.education.dto.common.Result;
import com.education.service.teacher.CourseService;
import com.education.utils.JwtUtils;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.servlet.http.HttpServletRequest;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

/**
 * 教师端课程管理控制器
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
@Tag(name = "教师端-课程管理", description = "教师课程创建、管理、内容编辑等接口")
@RestController("teacherCourseController")
@RequestMapping("/api/teacher/courses")
public class CourseController {

    @Autowired
    private CourseService courseService;
    
    @Autowired
    private JwtUtils jwtUtils;
    
    @Autowired
    private HttpServletRequest request;

    @Operation(summary = "创建课程", description = "教师创建新课程")
    @PostMapping
    public Result<CourseDTO.CourseResponse> createCourse(@RequestBody CourseDTO.CourseCreateRequest createRequest) {
        try {
            // 1. 获取当前教师ID（从JWT token或session中获取）
            Long teacherId = getCurrentTeacherId();
            
            // 2. 调用服务层创建课程
            CourseDTO.CourseResponse result = courseService.createCourse(createRequest, teacherId);
            
            return Result.success(result);
        } catch (Exception e) {
            return Result.error("创建课程失败: " + e.getMessage());
        }
    }

    @Operation(summary = "获取我的课程列表", description = "获取当前教师创建的所有课程")
    @GetMapping
    public Result<PageResponse<CourseDTO.CourseResponse>> getMyCourses(
            @RequestParam(defaultValue = "1") Integer page,
            @RequestParam(defaultValue = "10") Integer size,
            @RequestParam(required = false) String keyword,
            @RequestParam(required = false) String status) {
        try {
            // 1. 获取当前教师ID
            Long teacherId = getCurrentTeacherId();
            
            // 2. 构建分页参数
            PageRequest pageRequest = new PageRequest();
            pageRequest.setPage(page);
        pageRequest.setSize(size);
        // PageRequest类没有setKeyword方法，需要在查询时处理关键字
        // pageRequest.setKeyword(keyword);
            
            // 3. 调用服务层查询课程列表
            PageResponse<CourseDTO.CourseResponse> result = courseService.getCourseList(teacherId, pageRequest);
            
            return Result.success(result);
        } catch (Exception e) {
            return Result.error("获取课程列表失败: " + e.getMessage());
        }
    }

    @Operation(summary = "获取课程详情", description = "获取指定课程的详细信息")
    @GetMapping("/{courseId}")
    public Result<CourseDTO.CourseDetailResponse> getCourseDetail(@PathVariable Long courseId) {
        try {
            // 1. 获取当前教师ID
            Long teacherId = getCurrentTeacherId();
            
            // 2. 调用服务层获取课程详情
            CourseDTO.CourseDetailResponse result = courseService.getCourseDetail(courseId, teacherId);
            
            return Result.success(result);
        } catch (Exception e) {
            return Result.error("获取课程详情失败: " + e.getMessage());
        }
    }

    @Operation(summary = "更新课程信息", description = "更新课程基本信息")
    @PutMapping("/{courseId}")
    public Result<CourseDTO.CourseResponse> updateCourse(@PathVariable Long courseId, @RequestBody CourseDTO.CourseUpdateRequest updateRequest) {
        try {
            // 1. 获取当前教师ID
            Long teacherId = getCurrentTeacherId();
            
            // 2. 调用服务层更新课程
            CourseDTO.CourseResponse result = courseService.updateCourse(courseId, updateRequest, teacherId);
            
            return Result.success(result);
        } catch (Exception e) {
            return Result.error("更新课程失败: " + e.getMessage());
        }
    }

    @Operation(summary = "删除课程", description = "删除指定课程")
    @DeleteMapping("/{courseId}")
    public Result<Void> deleteCourse(@PathVariable Long courseId) {
        try {
            // 1. 获取当前教师ID
            Long teacherId = getCurrentTeacherId();
            
            // 2. 调用服务层删除课程
            Boolean result = courseService.deleteCourse(courseId, teacherId);
            
            if (result) {
                return Result.success();
            } else {
                return Result.error("删除课程失败");
            }
        } catch (Exception e) {
            return Result.error("删除课程失败: " + e.getMessage());
        }
    }

    @Operation(summary = "发布课程", description = "发布课程供学生学习")
    @PostMapping("/{courseId}/publish")
    public Result<Void> publishCourse(@PathVariable Long courseId) {
        try {
            // 1. 获取当前教师ID
            Long teacherId = getCurrentTeacherId();
            
            // 2. 调用服务层发布课程
            Boolean result = courseService.publishCourse(courseId, teacherId);
            
            if (result) {
                return Result.success();
            } else {
                return Result.error("发布课程失败");
            }
        } catch (Exception e) {
            return Result.error("发布课程失败: " + e.getMessage());
        }
    }

    @Operation(summary = "下架课程", description = "将已发布的课程下架")
    @PostMapping("/{courseId}/unpublish")
    public Result<Void> unpublishCourse(@PathVariable Long courseId) {
        try {
            // 1. 获取当前教师ID
            Long teacherId = getCurrentTeacherId();
            
            // 2. 调用服务层下架课程
            Boolean result = courseService.unpublishCourse(courseId, teacherId);
            
            if (result) {
                return Result.success();
            } else {
                return Result.error("下架课程失败");
            }
        } catch (Exception e) {
            return Result.error("下架课程失败: " + e.getMessage());
        }
    }

    @Operation(summary = "获取课程章节", description = "获取课程的章节结构")
    @GetMapping("/{courseId}/chapters")
    public Result<Object> getCourseChapters(@PathVariable Long courseId) {
        try {
            // 1. 获取当前教师ID
            Long teacherId = getCurrentTeacherId();
            
            // 2. 调用服务层获取课程章节
            Object result = courseService.getCourseChapters(courseId, teacherId);
            
            return Result.success(result);
        } catch (Exception e) {
            return Result.error("获取课程章节失败: " + e.getMessage());
        }
    }

    @Operation(summary = "创建课程章节", description = "为课程创建新章节")
    @PostMapping("/{courseId}/chapters")
    public Result<CourseDTO.ChapterResponse> createChapter(@PathVariable Long courseId, @RequestBody CourseDTO.ChapterCreateRequest createRequest) {
        try {
            // 1. 获取当前教师ID
            Long teacherId = getCurrentTeacherId();
            
            // 2. 调用服务层创建章节
            CourseDTO.ChapterResponse result = courseService.createChapter(courseId, createRequest, teacherId);
            
            return Result.success(result);
        } catch (Exception e) {
            return Result.error("创建章节失败: " + e.getMessage());
        }
    }

    @Operation(summary = "更新章节信息", description = "更新指定章节的信息")
    @PutMapping("/{courseId}/chapters/{chapterId}")
    public Result<CourseDTO.ChapterResponse> updateChapter(
            @PathVariable Long courseId,
            @PathVariable Long chapterId,
            @RequestBody CourseDTO.ChapterUpdateRequest updateRequest) {
        try {
            // 1. 获取当前教师ID
            Long teacherId = getCurrentTeacherId();
            
            // 2. 调用服务层更新章节
            CourseDTO.ChapterResponse result = courseService.updateChapter(chapterId, updateRequest, teacherId);
            
            return Result.success(result);
        } catch (Exception e) {
            return Result.error("更新章节失败: " + e.getMessage());
        }
    }

    @Operation(summary = "删除章节", description = "删除指定章节")
    @DeleteMapping("/{courseId}/chapters/{chapterId}")
    public Result<Void> deleteChapter(@PathVariable Long courseId, @PathVariable Long chapterId) {
        try {
            // 1. 获取当前教师ID
            Long teacherId = getCurrentTeacherId();
            
            // 2. 调用服务层删除章节
            Boolean result = courseService.deleteChapter(chapterId, teacherId);
            
            if (result) {
                return Result.success();
            } else {
                return Result.error("删除章节失败");
            }
        } catch (Exception e) {
            return Result.error("删除章节失败: " + e.getMessage());
        }
    }

    @Operation(summary = "获取课程统计", description = "获取课程的统计数据")
    @GetMapping("/{courseId}/statistics")
    public Result<Object> getCourseStatistics(@PathVariable Long courseId) {
        try {
            // 1. 获取当前教师ID
            Long teacherId = getCurrentTeacherId();
            
            // 2. 调用服务层获取课程统计
            Object result = courseService.getCourseStatistics(courseId, teacherId);
            
            return Result.success(result);
        } catch (Exception e) {
            return Result.error("获取课程统计失败: " + e.getMessage());
        }
    }

    @Operation(summary = "复制课程", description = "复制现有课程创建新课程")
    @PostMapping("/{courseId}/copy")
    public Result<Object> copyCourse(@PathVariable Long courseId, @RequestBody CourseDTO.CourseCopyRequest copyRequest) {
        try {
            // 1. 获取当前教师ID
            Long teacherId = getCurrentTeacherId();
            
            // 2. 调用服务层复制课程
            Object result = courseService.copyCourse(courseId, copyRequest.getNewCourseName(), teacherId);
            
            return Result.success(result);
        } catch (Exception e) {
            return Result.error("复制课程失败: " + e.getMessage());
        }
    }

    @Operation(summary = "导出课程", description = "导出课程内容")
    @GetMapping("/{courseId}/export")
    public Result<Object> exportCourse(@PathVariable Long courseId) {
        try {
            // 1. 获取当前教师ID
            Long teacherId = getCurrentTeacherId();
            
            // 2. 调用服务层导出课程
            Object result = courseService.exportCourse(courseId, teacherId);
            
            return Result.success(result);
        } catch (Exception e) {
            return Result.error("导出课程失败: " + e.getMessage());
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