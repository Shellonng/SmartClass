package com.education.service.teacher.impl;

import com.education.dto.CourseDTO;
import com.education.dto.common.PageRequest;
import com.education.dto.common.PageResponse;
import com.education.entity.Course;
import com.education.entity.Teacher;
import com.education.entity.Chapter;
import com.education.entity.User;
import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.education.mapper.CourseMapper;
import com.education.mapper.TeacherMapper;
import com.education.mapper.UserMapper;
import com.education.mapper.ChapterMapper;
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
    private final ChapterMapper chapterMapper;
    
    /**
     * 转换课程实体为响应对象
     */
    private CourseDTO.CourseResponse convertToCourseResponse(Course course) {
        CourseDTO.CourseResponse response = new CourseDTO.CourseResponse();
        BeanUtils.copyProperties(course, response);
        
        // 获取教师信息
        Teacher teacher = teacherMapper.selectById(course.getTeacherId());
        if (teacher != null) {
            User user = userMapper.selectById(teacher.getUserId());
            if (user != null) {
                response.setTeacherName(user.getRealName());
            }
        }
        
        return response;
    }
    
    @Override
    @Transactional
    public CourseDTO.CourseResponse createCourse(Object createRequest, Long teacherId) {
        CourseDTO.CourseCreateRequest request = (CourseDTO.CourseCreateRequest) createRequest;
        log.info("创建课程，教师ID: {}, 课程名称: {}", teacherId, request.getCourseName());
        
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
        BeanUtils.copyProperties(request, course);
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
        
        return new PageResponse<CourseDTO.CourseResponse>(
                (int) coursePage.getCurrent(),
                (int) coursePage.getSize(),
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
        
        // 查询课程
        Course course = courseMapper.selectById(courseId);
        if (course == null || course.getIsDeleted()) {
            throw new RuntimeException("课程不存在");
        }
        
        // 验证权限
        if (!course.getTeacherId().equals(teacherId)) {
            throw new RuntimeException("无权限发布该课程");
        }
        
        // 检查课程是否可以发布（至少有基本信息）
        if (!StringUtils.hasText(course.getCourseName()) || 
            !StringUtils.hasText(course.getDescription())) {
            throw new RuntimeException("课程信息不完整，无法发布");
        }
        
        // 更新课程状态为已发布
        course.setStatus("PUBLISHED");
        course.setUpdateTime(LocalDateTime.now());
        courseMapper.updateById(course);
        
        return true;
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
        
        // 查询课程章节
        List<Chapter> chapters = chapterMapper.selectByCourseId(courseId);
        
        // 转换为响应对象
        return chapters.stream()
                .map(this::convertToChapterResponse)
                .collect(Collectors.toList());
    }
    
    @Override
    @Transactional
    public CourseDTO.ChapterResponse createChapter(Long courseId, CourseDTO.ChapterCreateRequest createRequest, Long teacherId) {
        CourseDTO.ChapterCreateRequest chapterRequest = createRequest;
        log.info("创建课程章节，课程ID: {}, 教师ID: {}, 章节标题: {}", courseId, teacherId, chapterRequest.getChapterName());
        
        // 验证课程权限
        validateCourseAccess(courseId, teacherId);
        
        // 获取下一个排序号
        Integer maxSortOrder = chapterMapper.getMaxSortOrderByCourseId(courseId);
        int nextSortOrder = (maxSortOrder != null ? maxSortOrder : 0) + 1;
        
        // 创建章节
        Chapter chapter = new Chapter();
        chapter.setCourseId(courseId);
        chapter.setTitle(chapterRequest.getChapterName());
        chapter.setDescription(chapterRequest.getDescription());
        chapter.setContent(chapterRequest.getContent());
        chapter.setSortOrder(nextSortOrder);
        chapter.setStatus("DRAFT"); // 默认为草稿状态
        chapter.setIsRequired(chapterRequest.getIsRequired() != null ? chapterRequest.getIsRequired() : true);
        chapter.setEstimatedDuration(chapterRequest.getEstimatedDuration());
        chapter.setCreateTime(LocalDateTime.now());
        chapter.setUpdateTime(LocalDateTime.now());
        chapter.setIsDeleted(false);
        
        chapterMapper.insert(chapter);
        
        return convertToChapterResponse(chapter);
    }
    
    @Override
    @Transactional
    public CourseDTO.ChapterResponse updateChapter(Long chapterId, CourseDTO.ChapterUpdateRequest chapterRequest, Long teacherId) {
        log.info("更新课程章节，章节ID: {}, 教师ID: {}", chapterId, teacherId);
        
        // 查询章节
        Chapter chapter = chapterMapper.selectById(chapterId);
        if (chapter == null || chapter.getIsDeleted()) {
            throw new RuntimeException("章节不存在");
        }
        
        // 验证课程权限
        validateCourseAccess(chapter.getCourseId(), teacherId);
        
        // 更新章节信息
        if (chapterRequest.getChapterName() != null) {
            chapter.setTitle(chapterRequest.getChapterName());
        }
        if (chapterRequest.getDescription() != null) {
            chapter.setDescription(chapterRequest.getDescription());
        }
        if (chapterRequest.getContent() != null) {
            chapter.setContent(chapterRequest.getContent());
        }
        if (chapterRequest.getIsRequired() != null) {
            chapter.setIsRequired(chapterRequest.getIsRequired());
        }
        if (chapterRequest.getEstimatedDuration() != null) {
            chapter.setEstimatedDuration(chapterRequest.getEstimatedDuration());
        }
        chapter.setUpdateTime(LocalDateTime.now());
        
        chapterMapper.updateById(chapter);
        
        return convertToChapterResponse(chapter);
    }
    
    @Override
    @Transactional
    public Boolean deleteChapter(Long chapterId, Long teacherId) {
        log.info("删除课程章节，章节ID: {}, 教师ID: {}", chapterId, teacherId);
        
        // 查询章节
        Chapter chapter = chapterMapper.selectById(chapterId);
        if (chapter == null || chapter.getIsDeleted()) {
            throw new RuntimeException("章节不存在");
        }
        
        // 验证课程权限
        validateCourseAccess(chapter.getCourseId(), teacherId);
        
        // 软删除章节
        chapter.setIsDeleted(true);
        chapter.setUpdateTime(LocalDateTime.now());
        
        return chapterMapper.updateById(chapter) > 0;
    }
    
    // 私有辅助方法
    

    
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
    
    /**
     * 转换为章节响应对象
     */
    private CourseDTO.ChapterResponse convertToChapterResponse(Chapter chapter) {
        CourseDTO.ChapterResponse response = new CourseDTO.ChapterResponse();
        response.setChapterId(chapter.getId());
        response.setCourseId(chapter.getCourseId());
        response.setTitle(chapter.getTitle());
        response.setDescription(chapter.getDescription());
        response.setSortOrder(chapter.getSortOrder());
        response.setEstimatedDuration(chapter.getEstimatedDuration());
        response.setCreatedTime(chapter.getCreateTime());
        response.setStatus(chapter.getStatus());
        response.setIsRequired(chapter.getIsRequired());
        return response;
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
        
        CourseDTO.CourseStatisticsResponse response = new CourseDTO.CourseStatisticsResponse();
        response.setCourseId(courseId);
        
        // TODO: 统计学生数量、章节数量、任务数量等
        response.setStudentCount(0);
        response.setChapterCount(0);
        response.setTaskCount(0);
        response.setCompletionRate(0.0);
        response.setAverageScore(0.0);
        
        return response;
    }
    
    @Override
    public PageResponse<Object> getCourseStudents(Long courseId, Long teacherId, PageRequest pageRequest) {
        validateCourseAccess(courseId, teacherId);
        
        // 获取课程学生列表
        // 这里假设有一个方法可以获取课程学生列表
        List<Object> students = courseMapper.selectCourseStudents(courseId, pageRequest);
        Long total = courseMapper.countCourseStudents(courseId);
        
        return PageResponse.<Object>builder()
                .records(students)
                .total(total)
                .current(pageRequest.getCurrent())
                .pageSize(pageRequest.getPageSize())
                .<Object>build();
    }
    
    @Override
    @Transactional
    public CourseDTO.CourseResponse copyCourse(Long courseId, String newCourseName, Long teacherId) {
        log.info("复制课程，课程ID: {}, 新课程名称: {}, 教师ID: {}", courseId, newCourseName, teacherId);
        
        // 查询原课程
        Course originalCourse = courseMapper.selectById(courseId);
        if (originalCourse == null || originalCourse.getIsDeleted()) {
            throw new RuntimeException("原课程不存在");
        }
        
        // 验证权限
        if (!originalCourse.getTeacherId().equals(teacherId)) {
            throw new RuntimeException("无权限复制该课程");
        }
        
        // 创建新课程
        Course newCourse = new Course();
        BeanUtils.copyProperties(originalCourse, newCourse);
        newCourse.setCourseId(null); // 清空ID，让数据库自动生成
        newCourse.setCourseName(newCourseName);
        newCourse.setStatus("DRAFT"); // 复制的课程默认为草稿状态
        newCourse.setCreateTime(LocalDateTime.now());
        newCourse.setUpdateTime(LocalDateTime.now());
        
        // 保存新课程
        courseMapper.insert(newCourse);
        
        // TODO: 复制章节、任务等相关数据
        
        return convertToCourseResponse(newCourse);
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
        validateCourseAccess(courseId, teacherId);
        
        // 获取课程评价列表
        List<Object> reviews = courseMapper.selectCourseReviews(courseId, pageRequest);
        Long total = courseMapper.countCourseReviews(courseId);
        
        return PageResponse.<Object>builder()
                .records(reviews)
                .total(total)
                .current(pageRequest.getCurrent())
                .pageSize(pageRequest.getPageSize())
                .<Object>build();
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
        validateCourseAccess(courseId, teacherId);
        
        // 获取课程资源列表
        List<Object> resources = courseMapper.selectCourseResources(courseId, pageRequest);
        Long total = courseMapper.countCourseResources(courseId);
        
        return PageResponse.<Object>builder()
                .records(resources)
                .total(total)
                .current(pageRequest.getCurrent())
                .pageSize(pageRequest.getPageSize())
                .<Object>build();
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
        validateCourseAccess(courseId, teacherId);
        
        // 获取课程任务列表
        List<Object> tasks = courseMapper.selectCourseTasks(courseId, pageRequest);
        Long total = courseMapper.countCourseTasks(courseId);
        
        return PageResponse.<Object>builder()
                .records(tasks)
                .total(total)
                .current(pageRequest.getCurrent())
                .pageSize(pageRequest.getPageSize())
                .<Object>build();
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
        validateCourseAccess(courseId, teacherId);
        
        // 获取课程公告列表
        List<Object> announcements = courseMapper.selectCourseAnnouncements(courseId, pageRequest);
        Long total = courseMapper.countCourseAnnouncements(courseId);
        
        return PageResponse.<Object>builder()
                .records(announcements)
                .total(total)
                .current(pageRequest.getCurrent())
                .pageSize(pageRequest.getPageSize())
                .<Object>build();
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
    

    

    
    public Object getMyCourses(Object queryParams) {
        log.info("获取我的课程列表");
        // TODO: 实现获取我的课程列表逻辑
        return null;
    }
}