package com.education.service.teacher.impl;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.education.dto.CourseDTO;
import com.education.dto.common.PageRequest;
import com.education.dto.common.PageResponse;
import com.education.entity.Course;
import com.education.entity.Teacher;
import com.education.entity.User;
import com.education.mapper.CourseMapper;
import com.education.mapper.TeacherMapper;
import com.education.mapper.UserMapper;
import com.education.service.teacher.CourseService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.BeanUtils;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.util.StringUtils;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

/**
 * 教师端课程服务实现类
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class CourseServiceImpl implements CourseService {
    
    private final CourseMapper courseMapper;
    private final TeacherMapper teacherMapper;
    private final UserMapper userMapper;
    
    @Override
    @Transactional
    public CourseDTO.CourseResponse createCourse(CourseDTO.CourseCreateRequest createRequest, Long teacherId) {
        log.info("创建课程，教师ID: {}, 课程名称: {}", teacherId, createRequest.getCourseName());
        
        // 验证教师是否存在
        Teacher teacher = teacherMapper.selectById(teacherId);
        if (teacher == null) {
            throw new RuntimeException("教师不存在");
        }
        
        // 检查课程代码是否重复（如果有课程代码）
        QueryWrapper<Course> wrapper = new QueryWrapper<>();
        // 注意：CourseCreateRequest没有courseCode字段，这里先注释掉
        // wrapper.eq("course_code", createRequest.getCourseCode())
        //        .eq("is_deleted", false);
        Course existingCourse = courseMapper.selectOne(wrapper);
        if (existingCourse != null) {
            throw new RuntimeException("课程代码已存在");
        }
        
        // 创建课程实体
        Course course = new Course();
        BeanUtils.copyProperties(createRequest, course);
        course.setTeacherId(teacherId);
        course.setStatus("DRAFT"); // 默认为草稿状态
        course.setCreateTime(LocalDateTime.now());
        course.setUpdateTime(LocalDateTime.now());
        course.setIsDeleted(false);
        
        // 保存课程
        courseMapper.insert(course);
        
        // 返回响应
        return convertToCourseResponse(course);
    }
    
    @Override
    public PageResponse<CourseDTO.CourseResponse> getCourseList(Long teacherId, PageRequest pageRequest) {
        log.info("获取教师课程列表，教师ID: {}, 页码: {}, 页大小: {}", teacherId, pageRequest.getPageNum(), pageRequest.getPageSize());
        
        // 构建分页对象
        Page<Course> page = new Page<>(pageRequest.getPageNum(), pageRequest.getPageSize());
        
        // 构建查询条件
        QueryWrapper<Course> wrapper = new QueryWrapper<>();
        wrapper.eq("teacher_id", teacherId)
               .eq("is_deleted", false)
               .orderByDesc("create_time");
        
        // 注意：PageRequest没有keyword字段，这里先注释掉搜索功能
        // if (StringUtils.hasText(pageRequest.getKeyword())) {
        //     wrapper.and(w -> w.like("course_name", pageRequest.getKeyword())
        //                    .or().like("course_code", pageRequest.getKeyword())
        //                    .or().like("description", pageRequest.getKeyword()));
        // }
        
        // 执行分页查询
        IPage<Course> coursePage = courseMapper.selectPage(page, wrapper);
        
        // 转换为响应对象
        List<CourseDTO.CourseResponse> courseResponses = coursePage.getRecords().stream()
                .map(this::convertToCourseResponse)
                .collect(Collectors.toList());
        
        return new PageResponse<>(
                coursePage.getCurrent(),
                coursePage.getSize(),
                coursePage.getTotal(),
                courseResponses
        );
    }
    
    @Override
    public CourseDTO.CourseDetailResponse getCourseDetail(Long courseId, Long teacherId) {
        log.info("获取课程详情，课程ID: {}, 教师ID: {}", courseId, teacherId);
        
        // 查询课程
        Course course = courseMapper.selectById(courseId);
        if (course == null || course.getIsDeleted()) {
            throw new RuntimeException("课程不存在");
        }
        
        // 验证权限
        if (!course.getTeacherId().equals(teacherId)) {
            throw new RuntimeException("无权限访问该课程");
        }
        
        // 转换为详情响应对象
        CourseDTO.CourseDetailResponse response = new CourseDTO.CourseDetailResponse();
        BeanUtils.copyProperties(course, response);
        
        // 获取教师信息
        Teacher teacher = teacherMapper.selectById(course.getTeacherId());
        if (teacher != null && teacher.getUser() != null) {
            response.setTeacherName(teacher.getUser().getRealName());
        }
        
        // TODO: 添加章节信息、学生统计等
        // 注意：CourseDetailResponse没有setChapterCount和setTaskCount方法
        response.setStudentCount(0);
        
        return response;
    }
    
    @Override
    @Transactional
    public CourseDTO.CourseResponse updateCourse(Long courseId, CourseDTO.CourseUpdateRequest updateRequest, Long teacherId) {
        log.info("更新课程，课程ID: {}, 教师ID: {}", courseId, teacherId);
        
        // 查询课程
        Course course = courseMapper.selectById(courseId);
        if (course == null || course.getIsDeleted()) {
            throw new RuntimeException("课程不存在");
        }
        
        // 验证权限
        if (!course.getTeacherId().equals(teacherId)) {
            throw new RuntimeException("无权限修改该课程");
        }
        
        // 注意：CourseUpdateRequest没有courseCode字段，这里先注释掉
        // if (!course.getCourseCode().equals(updateRequest.getCourseCode())) {
        //     QueryWrapper<Course> wrapper = new QueryWrapper<>();
        //     wrapper.eq("course_code", updateRequest.getCourseCode())
        //            .eq("is_deleted", false)
        //            .ne("id", courseId);
        //     Course existingCourse = courseMapper.selectOne(wrapper);
        //     if (existingCourse != null) {
        //         throw new RuntimeException("课程代码已存在");
        //     }
        // }
        
        // 更新课程信息
        BeanUtils.copyProperties(updateRequest, course);
        course.setUpdateTime(LocalDateTime.now());
        
        // 保存更新
        courseMapper.updateById(course);
        
        return convertToCourseResponse(course);
    }
    
    @Override
    @Transactional
    public Boolean deleteCourse(Long courseId, Long teacherId) {
        log.info("删除课程，课程ID: {}, 教师ID: {}", courseId, teacherId);
        
        // 查询课程
        Course course = courseMapper.selectById(courseId);
        if (course == null || course.getIsDeleted()) {
            throw new RuntimeException("课程不存在");
        }
        
        // 验证权限
        if (!course.getTeacherId().equals(teacherId)) {
            throw new RuntimeException("无权限删除该课程");
        }
        
        // 软删除
        course.setIsDeleted(true);
        course.setUpdateTime(LocalDateTime.now());
        
        return courseMapper.updateById(course) > 0;
    }
    
    @Override
    @Transactional
    public Boolean publishCourse(Long courseId, Long teacherId) {
        log.info("发布课程，课程ID: {}, 教师ID: {}", courseId, teacherId);
        
        return updateCourseStatus(courseId, teacherId, "PUBLISHED");
    }
    
    @Override
    @Transactional
    public Boolean unpublishCourse(Long courseId, Long teacherId) {
        log.info("下架课程，课程ID: {}, 教师ID: {}", courseId, teacherId);
        
        return updateCourseStatus(courseId, teacherId, "DRAFT");
    }
    
    @Override
    public List<CourseDTO.ChapterResponse> getCourseChapters(Long courseId, Long teacherId) {
        log.info("获取课程章节列表，课程ID: {}, 教师ID: {}", courseId, teacherId);
        
        // 验证课程权限
        validateCourseAccess(courseId, teacherId);
        
        // TODO: 实现章节查询逻辑
        // 这里返回空列表，实际应该查询章节表
        return new ArrayList<>();
    }
    
    @Override
    @Transactional
    public CourseDTO.ChapterResponse createChapter(Long courseId, CourseDTO.ChapterCreateRequest chapterRequest, Long teacherId) {
        log.info("创建课程章节，课程ID: {}, 教师ID: {}, 章节标题: {}", courseId, teacherId, chapterRequest.getChapterName());
        
        // 验证课程权限
        validateCourseAccess(courseId, teacherId);
        
        // TODO: 实现章节创建逻辑
        // 这里返回null，实际应该创建章节并返回响应对象
        return null;
    }
    
    @Override
    @Transactional
    public CourseDTO.ChapterResponse updateChapter(Long chapterId, CourseDTO.ChapterUpdateRequest chapterRequest, Long teacherId) {
        log.info("更新课程章节，章节ID: {}, 教师ID: {}", chapterId, teacherId);
        
        // TODO: 实现章节更新逻辑
        return null;
    }
    
    @Override
    @Transactional
    public Boolean deleteChapter(Long chapterId, Long teacherId) {
        log.info("删除课程章节，章节ID: {}, 教师ID: {}", chapterId, teacherId);
        
        // TODO: 实现章节删除逻辑
        return false;
    }
    
    // 私有辅助方法
    
    /**
     * 转换课程实体为响应对象
     */
    private CourseDTO.CourseResponse convertToCourseResponse(Course course) {
        CourseDTO.CourseResponse response = new CourseDTO.CourseResponse();
        BeanUtils.copyProperties(course, response);
        
        // 获取教师姓名
        Teacher teacher = teacherMapper.selectById(course.getTeacherId());
        if (teacher != null && teacher.getUser() != null) {
            response.setTeacherName(teacher.getUser().getRealName());
        }
        
        return response;
    }
    
    /**
     * 更新课程状态
     */
    private Boolean updateCourseStatus(Long courseId, Long teacherId, String status) {
        Course course = courseMapper.selectById(courseId);
        if (course == null || course.getIsDeleted()) {
            throw new RuntimeException("课程不存在");
        }
        
        // 验证权限
        if (!course.getTeacherId().equals(teacherId)) {
            throw new RuntimeException("无权限修改该课程");
        }
        
        course.setStatus(status);
        course.setUpdateTime(LocalDateTime.now());
        
        return courseMapper.updateById(course) > 0;
    }
    
    /**
     * 验证课程访问权限
     */
    private void validateCourseAccess(Long courseId, Long teacherId) {
        Course course = courseMapper.selectById(courseId);
        if (course == null || course.getIsDeleted()) {
            throw new RuntimeException("课程不存在");
        }
        
        if (!course.getTeacherId().equals(teacherId)) {
            throw new RuntimeException("无权限访问该课程");
        }
    }
    
    // 以下方法暂时返回默认值，待后续完善
    
    @Override
    public Boolean reorderChapters(Long courseId, List<CourseDTO.ChapterOrderRequest> chapterOrders, Long teacherId) {
        log.info("调整章节顺序，课程ID: {}, 教师ID: {}", courseId, teacherId);
        validateCourseAccess(courseId, teacherId);
        // TODO: 实现章节排序逻辑
        return true;
    }
    
    @Override
    public CourseDTO.CourseStatisticsResponse getCourseStatistics(Long courseId, Long teacherId) {
        log.info("获取课程统计信息，课程ID: {}, 教师ID: {}", courseId, teacherId);
        validateCourseAccess(courseId, teacherId);
        // TODO: 实现统计信息查询
        return new CourseDTO.CourseStatisticsResponse();
    }
    
    @Override
    public PageResponse<Object> getCourseStudents(Long courseId, Long teacherId, PageRequest pageRequest) {
        log.info("获取课程学生列表，课程ID: {}, 教师ID: {}", courseId, teacherId);
        validateCourseAccess(courseId, teacherId);
        // TODO: 实现学生列表查询
        return new PageResponse<>(
                (long) pageRequest.getPageNum(),
                (long) pageRequest.getPageSize(),
                0L,
                new ArrayList<Object>()
        );
    }
    
    @Override
    @Transactional
    public CourseDTO.CourseResponse copyCourse(Long courseId, String newCourseName, Long teacherId) {
        log.info("复制课程，课程ID: {}, 新课程名称: {}, 教师ID: {}", courseId, newCourseName, teacherId);
        validateCourseAccess(courseId, teacherId);
        // TODO: 实现课程复制逻辑
        return null;
    }
    
    @Override
    public String exportCourse(Long courseId, Long teacherId) {
        log.info("导出课程内容，课程ID: {}, 教师ID: {}", courseId, teacherId);
        validateCourseAccess(courseId, teacherId);
        // TODO: 实现课程导出逻辑
        return null;
    }
    
    @Override
    @Transactional
    public Object importCourse(CourseDTO.CourseImportRequest importRequest, Long teacherId) {
        log.info("导入课程内容，教师ID: {}", teacherId);
        // TODO: 实现课程导入逻辑
        return null;
    }
    
    @Override
    public Object getCourseProgressStatistics(Long courseId, Long teacherId) {
        log.info("获取课程学习进度统计，课程ID: {}, 教师ID: {}", courseId, teacherId);
        validateCourseAccess(courseId, teacherId);
        // TODO: 实现学习进度统计
        return null;
    }
    
    @Override
    @Transactional
    public Boolean setCoursePermissions(Long courseId, Object permissions, Long teacherId) {
        log.info("设置课程权限，课程ID: {}, 教师ID: {}", courseId, teacherId);
        validateCourseAccess(courseId, teacherId);
        // TODO: 实现权限设置逻辑
        return true;
    }
    
    @Override
    public PageResponse<Object> getCourseReviews(Long courseId, Long teacherId, PageRequest pageRequest) {
        log.info("获取课程评价列表，课程ID: {}, 教师ID: {}", courseId, teacherId);
        validateCourseAccess(courseId, teacherId);
        // TODO: 实现课程评价查询逻辑
        return new PageResponse<>(
                (long) pageRequest.getPageNum(),
                (long) pageRequest.getPageSize(),
                0L,
                new ArrayList<Object>()
        );
    }
    
    @Override
    @Transactional
    public Boolean replyCourseReview(Long reviewId, String reply, Long teacherId) {
        log.info("回复课程评价，评价ID: {}, 教师ID: {}", reviewId, teacherId);
        // TODO: 实现课程评价回复逻辑
        return true;
    }
    
    @Override
    public PageResponse<Object> getCourseResources(Long courseId, Long teacherId, PageRequest pageRequest) {
        log.info("获取课程资源列表，课程ID: {}, 教师ID: {}", courseId, teacherId);
        validateCourseAccess(courseId, teacherId);
        // TODO: 实现课程资源查询逻辑
        return new PageResponse<>(
                (long) pageRequest.getPageNum(),
                (long) pageRequest.getPageSize(),
                0L,
                new ArrayList<Object>()
        );
    }
    
    @Override
    @Transactional
    public Boolean addCourseResources(Long courseId, List<Long> resourceIds, Long teacherId) {
        log.info("添加课程资源，课程ID: {}, 教师ID: {}", courseId, teacherId);
        validateCourseAccess(courseId, teacherId);
        // TODO: 实现添加课程资源逻辑
        return true;
    }
    
    @Override
    @Transactional
    public Boolean removeCourseResource(Long courseId, Long resourceId, Long teacherId) {
        log.info("移除课程资源，课程ID: {}, 资源ID: {}, 教师ID: {}", courseId, resourceId, teacherId);
        validateCourseAccess(courseId, teacherId);
        // TODO: 实现移除课程资源逻辑
        return true;
    }
    
    @Override
    public PageResponse<Object> getCourseTasks(Long courseId, Long teacherId, PageRequest pageRequest) {
        log.info("获取课程任务列表，课程ID: {}, 教师ID: {}", courseId, teacherId);
        validateCourseAccess(courseId, teacherId);
        // TODO: 实现课程任务查询逻辑
        return new PageResponse<>(
                (long) pageRequest.getPageNum(),
                (long) pageRequest.getPageSize(),
                0L,
                new ArrayList<Object>()
        );
    }
    
    @Override
    @Transactional
    public Object createCourseAnnouncement(Long courseId, Object announcement, Long teacherId) {
        log.info("创建课程公告，课程ID: {}, 教师ID: {}", courseId, teacherId);
        validateCourseAccess(courseId, teacherId);
        // TODO: 实现创建课程公告逻辑
        return null;
    }
    
    @Override
    public PageResponse<Object> getCourseAnnouncements(Long courseId, Long teacherId, PageRequest pageRequest) {
        log.info("获取课程公告列表，课程ID: {}, 教师ID: {}", courseId, teacherId);
        validateCourseAccess(courseId, teacherId);
        // TODO: 实现课程公告查询逻辑
        return new PageResponse<>(
                (long) pageRequest.getPageNum(),
                (long) pageRequest.getPageSize(),
                0L,
                new ArrayList<Object>()
        );
    }
    
    @Override
    @Transactional
    public Boolean updateCourseAnnouncement(Long announcementId, Object announcement, Long teacherId) {
        log.info("更新课程公告，公告ID: {}, 教师ID: {}", announcementId, teacherId);
        // TODO: 实现更新课程公告逻辑
        return true;
    }
    
    @Override
    @Transactional
    public Boolean deleteCourseAnnouncement(Long announcementId, Long teacherId) {
        log.info("删除课程公告，公告ID: {}, 教师ID: {}", announcementId, teacherId);
        // TODO: 实现删除课程公告逻辑
        return true;
    }
    
    @Override
    @Transactional
    public Boolean setCourseTags(Long courseId, List<String> tags, Long teacherId) {
        log.info("设置课程标签，课程ID: {}, 教师ID: {}", courseId, teacherId);
        validateCourseAccess(courseId, teacherId);
        // TODO: 实现设置课程标签逻辑
        return true;
    }
    
    @Override
    public List<String> getCourseTags(Long courseId, Long teacherId) {
        log.info("获取课程标签，课程ID: {}, 教师ID: {}", courseId, teacherId);
        validateCourseAccess(courseId, teacherId);
        // TODO: 实现获取课程标签逻辑
        return new ArrayList<>();
    }
    
    @Override
    @Transactional
    public Boolean archiveCourse(Long courseId, Long teacherId) {
        log.info("归档课程，课程ID: {}, 教师ID: {}", courseId, teacherId);
        return updateCourseStatus(courseId, teacherId, "ARCHIVED");
    }
    
    @Override
    @Transactional
    public Boolean restoreCourse(Long courseId, Long teacherId) {
        log.info("恢复归档课程，课程ID: {}, 教师ID: {}", courseId, teacherId);
        return updateCourseStatus(courseId, teacherId, "DRAFT");
    }
    
    @Override
    public List<Object> getCourseTemplates(Long teacherId) {
        log.info("获取课程模板列表，教师ID: {}", teacherId);
        // TODO: 实现获取课程模板逻辑
        return new ArrayList<>();
    }
    
    @Override
    @Transactional
    public CourseDTO.CourseResponse createCourseFromTemplate(Long templateId, String courseName, Long teacherId) {
        log.info("从模板创建课程，模板ID: {}, 课程名称: {}, 教师ID: {}", templateId, courseName, teacherId);
        // TODO: 实现从模板创建课程逻辑
        return null;
    }
    
    @Override
    @Transactional
    public Boolean saveCourseAsTemplate(Long courseId, String templateName, Long teacherId) {
        log.info("保存课程为模板，课程ID: {}, 模板名称: {}, 教师ID: {}", courseId, templateName, teacherId);
        validateCourseAccess(courseId, teacherId);
        // TODO: 实现保存课程为模板逻辑
        return true;
    }
}