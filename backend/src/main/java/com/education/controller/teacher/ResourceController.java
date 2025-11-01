package com.education.controller.teacher;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.education.dto.common.PageRequest;
import com.education.dto.common.Result;
import com.education.entity.CourseResource;
import com.education.entity.Course;
import com.education.mapper.CourseMapper;
import com.education.mapper.CourseResourceMapper;
import com.education.security.SecurityUtil;
import com.education.service.teacher.CourseResourceService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/api/teacher/resources")
public class ResourceController {

    @Autowired
    private CourseResourceMapper resourceMapper;
    
    @Autowired
    private CourseMapper courseMapper;
    
    @Autowired
    private CourseResourceService courseResourceService;
    
    @Autowired
    private SecurityUtil securityUtil;

    /**
     * 获取当前用户上传的所有资源
     * @param pageRequest 分页请求
     * @return 资源列表
     */
    @GetMapping("/user")
    public Result<Page<CourseResource>> getUserResources(PageRequest pageRequest) {
        // 获取当前登录用户ID
        Long userId = securityUtil.getCurrentUserId();
        if (userId == null) {
            return Result.error("未登录");
        }
        
        // 创建分页对象
        Page<CourseResource> page = new Page<>(pageRequest.getPageNum(), pageRequest.getPageSize());
        
        // 查询条件：上传者ID等于当前用户ID
        LambdaQueryWrapper<CourseResource> queryWrapper = new LambdaQueryWrapper<>();
        queryWrapper.eq(CourseResource::getUploadUserId, userId);
        queryWrapper.orderByDesc(CourseResource::getCreateTime);
        
        // 执行查询
        Page<CourseResource> resultPage = resourceMapper.selectPage(page, queryWrapper);
        
        // 获取所有涉及的课程ID
        List<Long> courseIds = resultPage.getRecords().stream()
                .map(CourseResource::getCourseId)
                .distinct()
                .collect(Collectors.toList());
        
        // 批量查询课程信息
        Map<Long, String> courseNameMap = courseMapper.selectBatchIds(courseIds).stream()
                .collect(Collectors.toMap(Course::getId, Course::getTitle));
        
        // 设置课程名称
        resultPage.getRecords().forEach(resource -> {
            if (resource.getCourseId() != null) {
                String courseName = courseNameMap.get(resource.getCourseId());
                resource.setCourseName(courseName);
            }
        });
        
        return Result.success(resultPage);
    }
    
    /**
     * 获取所有资源列表
     * @param pageRequest 分页请求
     * @return 资源列表
     */
    @GetMapping
    public Result<Page<CourseResource>> getAllResources(PageRequest pageRequest) {
        return courseResourceService.getAllResources(pageRequest);
    }
} 