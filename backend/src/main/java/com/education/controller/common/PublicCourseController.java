package com.education.controller.common;

import com.education.dto.common.PageRequest;
import com.education.dto.common.PageResponse;
import com.education.dto.common.Result;
import com.education.entity.Course;
import com.education.mapper.CourseMapper;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.util.StringUtils;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;
import java.util.stream.Collectors;

/**
 * 公共课程控制器
 * 提供不需要认证的课程相关接口
 */
@Tag(name = "公共-课程管理", description = "公共课程查询相关接口")
@RestController
@RequestMapping("/api/courses")
public class PublicCourseController {

    private static final Logger logger = LoggerFactory.getLogger(PublicCourseController.class);

    @Autowired
    private CourseMapper courseMapper;

    @Operation(summary = "获取公开课程列表", description = "获取公开的课程列表，不需要认证")
    @GetMapping("/public")
    public Result<PageResponse<Course>> getPublicCourses(
            @Parameter(description = "页码") @RequestParam(defaultValue = "0") Integer page,
            @Parameter(description = "每页大小") @RequestParam(defaultValue = "10") Integer size,
            @Parameter(description = "关键词搜索") @RequestParam(required = false) String keyword,
            @Parameter(description = "课程类型") @RequestParam(required = false) String courseType,
            @Parameter(description = "排序方式") @RequestParam(defaultValue = "latest") String sortBy) {
        
        logger.info("获取公开课程列表 - 页码: {}, 每页大小: {}, 关键词: {}, 课程类型: {}, 排序: {}", 
                page, size, keyword, courseType, sortBy);
        
        try {
            // 构建查询条件
            LambdaQueryWrapper<Course> queryWrapper = new LambdaQueryWrapper<>();
            
            // 移除仅查询"开始"状态的课程限制
            // queryWrapper.eq(Course::getStatus, "开始");
            
            // 添加关键词搜索
            if (StringUtils.hasText(keyword)) {
                queryWrapper.like(Course::getTitle, keyword)
                        .or()
                        .like(Course::getDescription, keyword);
                logger.info("添加关键词搜索条件: {}", keyword);
            }
            
            // 添加课程类型筛选
            if (StringUtils.hasText(courseType)) {
                queryWrapper.eq(Course::getCourseType, courseType);
                logger.info("添加课程类型筛选条件: {}", courseType);
            }
            
            // 排序
            switch (sortBy) {
                case "popular":
                    queryWrapper.orderByDesc(Course::getStudentCount);
                    break;
                case "latest":
                    queryWrapper.orderByDesc(Course::getCreateTime);
                    break;
                default:
                    queryWrapper.orderByDesc(Course::getCreateTime);
                    break;
            }
            
            // 分页查询
            Page<Course> coursePage = new Page<>(page + 1, size);  // MyBatisPlus分页从1开始，前端从0开始
            Page<Course> result = courseMapper.selectPage(coursePage, queryWrapper);
            
            // 构建响应
            PageResponse<Course> response = new PageResponse<>();
            response.setRecords(result.getRecords());
            response.setTotal(result.getTotal());
            response.setCurrent((int) result.getCurrent() - 1);  // 返回给前端的页码从0开始
            response.setPageSize((int) result.getSize());
            response.setPages(result.getPages());
            
            logger.info("查询到课程总数: {}, 当前页课程数: {}", result.getTotal(), result.getRecords().size());
            
            return Result.success(response);
        } catch (Exception e) {
            logger.error("获取公开课程列表异常", e);
            return Result.error("获取课程列表失败: " + e.getMessage());
        }
    }

    @Operation(summary = "获取课程分类列表", description = "获取系统中所有的课程分类")
    @GetMapping("/categories")
    public Result<List<String>> getCourseCategories() {
        logger.info("获取课程分类列表");
        
        try {
            // 从数据库中查询所有不同的课程类型
            LambdaQueryWrapper<Course> queryWrapper = new LambdaQueryWrapper<>();
            queryWrapper.select(Course::getCourseType).groupBy(Course::getCourseType);
            
            List<String> categories = courseMapper.selectList(queryWrapper)
                    .stream()
                    .map(Course::getCourseType)
                    .collect(Collectors.toList());
            
            logger.info("查询到课程分类: {}", categories);
            
            return Result.success(categories);
        } catch (Exception e) {
            logger.error("获取课程分类列表异常", e);
            return Result.error("获取课程分类失败: " + e.getMessage());
        }
    }
} 