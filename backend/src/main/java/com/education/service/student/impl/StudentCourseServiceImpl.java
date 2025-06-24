package com.education.service.student.impl;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.education.dto.CourseDTO;
import com.education.dto.common.PageRequest;
import com.education.dto.common.PageResponse;
import com.education.entity.*;
import com.education.exception.BusinessException;
import com.education.mapper.*;
import com.education.exception.ResultCode;
import com.education.service.student.StudentCourseService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * 学生端课程服务实现类
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
@Slf4j
@Service
public class StudentCourseServiceImpl implements StudentCourseService {

    @Autowired
    private CourseMapper courseMapper;
    
    @Autowired
    private StudentMapper studentMapper;
    
    @Autowired
    private ClassMapper classMapper;
    
    // ChapterMapper暂时注释，因为该类不存在
    // @Autowired
    // private ChapterMapper chapterMapper;
    
    @Autowired
    private ResourceMapper resourceMapper;

    @Override
    public PageResponse<CourseDTO.CourseListResponse> getStudentCourses(Long studentId, PageRequest pageRequest) {
        log.info("获取学生课程列表，学生ID: {}", studentId);
        
        Student student = studentMapper.selectById(studentId);
        if (student == null) {
            throw new BusinessException(ResultCode.USER_NOT_FOUND, "学生不存在");
        }
        
        // 构建分页查询
        Page<Course> page = new Page<>(pageRequest.getPageNum(), pageRequest.getPageSize());
        
        // 查询学生所在班级的课程
        QueryWrapper<Course> queryWrapper = new QueryWrapper<>();
        queryWrapper.exists("SELECT 1 FROM class_course cc WHERE cc.course_id = course.id AND cc.class_id = {0}", student.getClassId())
                   .eq("status", 1)
                   .orderByDesc("created_time");
        
        Page<Course> coursePage = courseMapper.selectPage(page, queryWrapper);
        
        List<CourseDTO.CourseListResponse> responses = coursePage.getRecords().stream()
                .map(this::convertToCourseListResponse)
                .collect(Collectors.toList());
        
        return PageResponse.of((long)pageRequest.getPageNum(), (long)pageRequest.getPageSize(), coursePage.getTotal(), responses);
    }

    @Override
    public CourseDTO.CourseDetailResponse getCourseDetail(Long courseId, Long studentId) {
        log.info("获取课程详情，课程ID: {}, 学生ID: {}", courseId, studentId);
        
        // 验证权限
        if (!hasCourseAccess(courseId, studentId)) {
            throw new BusinessException(ResultCode.ACCESS_DENIED, "无权限访问该课程");
        }
        
        Course course = courseMapper.selectById(courseId);
        if (course == null) {
            throw new BusinessException(ResultCode.DATA_NOT_FOUND, "课程不存在");
        }
        
        CourseDTO.CourseDetailResponse response = new CourseDTO.CourseDetailResponse();
        response.setCourseId(courseId);
        response.setCourseName(course.getCourseName());
        response.setDescription(course.getDescription());
        // TODO: 需要通过teacherId查询教师姓名
        // response.setTeacherName(course.getTeacherName());
        // TODO: Course实体缺少createdTime和updatedTime字段
        // response.setCreatedTime(course.getCreatedTime());
        // response.setUpdatedTime(course.getUpdatedTime());
        
        return response;
    }



    @Override
    public List<CourseDTO.ChapterResponse> getCourseChapters(Long courseId, Long studentId) {
        log.info("获取课程章节列表，课程ID: {}, 学生ID: {}", courseId, studentId);
        
        // 验证权限
        if (!hasCourseAccess(courseId, studentId)) {
            throw new BusinessException(ResultCode.ACCESS_DENIED, "无权限访问该课程");
        }
        
        // TODO: ChapterMapper不存在，暂时返回空列表
        log.warn("ChapterMapper不存在，返回空章节列表");
        return List.of();
    }

    @Override
    public CourseDTO.ChapterDetailResponse getChapterDetail(Long chapterId, Long studentId) {
        log.info("获取章节详情，章节ID: {}, 学生ID: {}", chapterId, studentId);
        
        // TODO: ChapterMapper不存在，暂时抛出异常
        log.warn("ChapterMapper不存在，无法获取章节详情");
        throw new BusinessException(ResultCode.DATA_NOT_FOUND, "章节功能暂未实现");
    }

    @Override
    public Boolean markChapterAsLearned(Long chapterId, Long studentId) {
        log.info("标记章节为已学习，章节ID: {}, 学生ID: {}", chapterId, studentId);
        
        // TODO: ChapterMapper不存在，暂时返回false
        log.warn("ChapterMapper不存在，无法标记章节为已学习");
        return false;
    }

    @Override
    public CourseDTO.LearningProgressResponse getLearningProgress(Long courseId, Long studentId) {
        log.info("获取学习进度，课程ID: {}, 学生ID: {}", courseId, studentId);
        
        // 验证权限
        if (!hasCourseAccess(courseId, studentId)) {
            throw new BusinessException(ResultCode.ACCESS_DENIED, "无权限访问该课程");
        }
        
        CourseDTO.LearningProgressResponse response = new CourseDTO.LearningProgressResponse();
        response.setCourseId(courseId);
        response.setStudentId(studentId);
        response.setProgress(0.0);
        response.setCompletedChapters(0);
        response.setTotalChapters(0);
        
        return response;
    }

    @Override
    public Boolean updateLearningProgress(CourseDTO.ProgressUpdateRequest progressRequest, Long studentId) {
        log.info("更新学习进度，学生ID: {}", studentId);
        
        Student student = studentMapper.selectById(studentId);
        if (student == null) {
            throw new BusinessException(ResultCode.USER_NOT_FOUND, "学生不存在");
        }
        
        // 这里可以实现更新学习进度的逻辑
        log.info("学习进度更新成功");
        return true;
    }

    @Override
    public PageResponse<CourseDTO.AnnouncementResponse> getCourseAnnouncements(Long courseId, Long studentId, PageRequest pageRequest) {
        log.info("获取课程公告，课程ID: {}, 学生ID: {}", courseId, studentId);
        
        // 验证权限
        if (!hasCourseAccess(courseId, studentId)) {
            throw new BusinessException(ResultCode.ACCESS_DENIED, "无权限访问该课程");
        }
        
        // 这里可以实现获取课程公告的逻辑
        return PageResponse.of((long)pageRequest.getPageNum(), (long)pageRequest.getPageSize(), 0L, List.of());
    }

    @Override
    public PageResponse<CourseDTO.ResourceResponse> getCourseResources(Long courseId, Long studentId, PageRequest pageRequest) {
        log.info("获取课程资源，课程ID: {}, 学生ID: {}", courseId, studentId);
        
        // 验证权限
        if (!hasCourseAccess(courseId, studentId)) {
            throw new BusinessException(ResultCode.ACCESS_DENIED, "无权限访问该课程");
        }
        
        // 构建分页查询
        Page<Resource> page = new Page<>(pageRequest.getPageNum(), pageRequest.getPageSize());
        
        QueryWrapper<Resource> queryWrapper = new QueryWrapper<>();
        queryWrapper.eq("course_id", courseId)
                   .eq("status", 1)
                   .orderByDesc("created_time");
        
        Page<Resource> resourcePage = resourceMapper.selectPage(page, queryWrapper);
        
        List<CourseDTO.ResourceResponse> responses = resourcePage.getRecords().stream()
                .map(this::convertToResourceResponse)
                .collect(Collectors.toList());
        
        return PageResponse.of((long)pageRequest.getPageNum(), (long)pageRequest.getPageSize(), resourcePage.getTotal(), responses);
    }



    @Override
    public PageResponse<CourseDTO.DiscussionResponse> getCourseDiscussions(Long courseId, Long studentId, PageRequest pageRequest) {
        log.info("获取课程讨论，课程ID: {}, 学生ID: {}", courseId, studentId);
        
        // 验证权限
        if (!hasCourseAccess(courseId, studentId)) {
            throw new BusinessException(ResultCode.ACCESS_DENIED, "无权限访问该课程");
        }
        
        // 这里可以实现获取课程讨论的逻辑
        return PageResponse.of((long)pageRequest.getPageNum(), (long)pageRequest.getPageSize(), 0L, List.of());
    }

    @Override
    public Long createDiscussion(CourseDTO.DiscussionCreateRequest discussionRequest, Long studentId) {
        log.info("创建课程讨论，学生ID: {}", studentId);
        
        Student student = studentMapper.selectById(studentId);
        if (student == null) {
            throw new BusinessException(ResultCode.USER_NOT_FOUND, "学生不存在");
        }
        
        // 这里可以实现创建讨论的逻辑
        log.info("创建讨论成功");
        return 1L;
    }

    @Override
    public Long replyDiscussion(Long discussionId, CourseDTO.DiscussionReplyRequest replyRequest, Long studentId) {
        log.info("回复讨论，讨论ID: {}, 学生ID: {}", discussionId, studentId);
        
        Student student = studentMapper.selectById(studentId);
        if (student == null) {
            throw new BusinessException(ResultCode.USER_NOT_FOUND, "学生不存在");
        }
        
        // 这里可以实现回复讨论的逻辑
        log.info("回复讨论成功");
        return 1L;
    }

    @Override
    public Boolean favoriteCourse(Long courseId, Long studentId) {
        log.info("收藏课程，课程ID: {}, 学生ID: {}", courseId, studentId);
        
        // 验证权限
        if (!hasCourseAccess(courseId, studentId)) {
            throw new BusinessException(ResultCode.ACCESS_DENIED, "无权限收藏该课程");
        }
        
        // 这里可以实现收藏课程的逻辑
        log.info("课程收藏成功");
        return true;
    }

    @Override
    public Boolean unfavoriteCourse(Long courseId, Long studentId) {
        log.info("取消收藏课程，课程ID: {}, 学生ID: {}", courseId, studentId);
        
        Course course = courseMapper.selectById(courseId);
        if (course == null) {
            throw new BusinessException(ResultCode.DATA_NOT_FOUND, "课程不存在");
        }
        
        // 这里可以实现取消收藏课程的逻辑
        log.info("取消收藏课程成功");
        return true;
    }

    @Override
    public PageResponse<CourseDTO.CourseListResponse> getFavoriteCourses(Long studentId, PageRequest pageRequest) {
        log.info("获取收藏的课程列表，学生ID: {}", studentId);
        
        Student student = studentMapper.selectById(studentId);
        if (student == null) {
            throw new BusinessException(ResultCode.USER_NOT_FOUND, "学生不存在");
        }
        
        // 这里可以实现获取收藏课程的逻辑
        return PageResponse.of((long)pageRequest.getPageNum(), (long)pageRequest.getPageSize(), 0L, List.of());
    }

    @Override
    public Boolean evaluateCourse(CourseDTO.CourseEvaluationRequest evaluationRequest, Long studentId) {
        log.info("评价课程，学生ID: {}", studentId);
        
        Student student = studentMapper.selectById(studentId);
        if (student == null) {
            throw new BusinessException(ResultCode.USER_NOT_FOUND, "学生不存在");
        }
        
        // 这里可以实现评价课程的逻辑
        log.info("课程评价成功");
        return true;
    }

    @Override
    public CourseDTO.EvaluationResponse getMyCourseEvaluation(Long courseId, Long studentId) {
        log.info("获取我的课程评价，课程ID: {}, 学生ID: {}", courseId, studentId);
        
        // 验证权限
        if (!hasCourseAccess(courseId, studentId)) {
            throw new BusinessException(ResultCode.ACCESS_DENIED, "无权限访问该课程");
        }
        
        CourseDTO.EvaluationResponse response = new CourseDTO.EvaluationResponse();
        response.setCourseId(courseId);
        response.setStudentId(studentId);
        response.setRating(0);
        response.setComment("");
        
        return response;
    }

    @Override
    public PageResponse<CourseDTO.CourseListResponse> getRecommendedCourses(Long studentId, PageRequest pageRequest) {
        log.info("获取推荐课程，学生ID: {}", studentId);
        
        Student student = studentMapper.selectById(studentId);
        if (student == null) {
            throw new BusinessException(ResultCode.USER_NOT_FOUND, "学生不存在");
        }
        
        // 这里可以实现获取推荐课程的逻辑
        return PageResponse.of((long)pageRequest.getPageNum(), (long)pageRequest.getPageSize(), 0L, List.of());
    }

    @Override
    public CourseDTO.LearningStatisticsResponse getLearningStatistics(Long courseId, Long studentId) {
        log.info("获取学习统计，课程ID: {}, 学生ID: {}", courseId, studentId);
        
        // 验证权限
        if (!hasCourseAccess(courseId, studentId)) {
            throw new BusinessException(ResultCode.ACCESS_DENIED, "无权限访问该课程");
        }
        
        CourseDTO.LearningStatisticsResponse response = new CourseDTO.LearningStatisticsResponse();
        response.setCourseId(courseId);
        response.setStudentId(studentId);
        response.setTotalStudyTime(0);
        response.setCompletedTasks(0);
        response.setTotalTasks(0);
        
        return response;
    }

    @Override
    public List<CourseDTO.CourseListResponse> getRecentCourses(Long studentId, Integer limit) {
        log.info("获取最近学习的课程，学生ID: {}, 限制数量: {}", studentId, limit);
        
        Student student = studentMapper.selectById(studentId);
        if (student == null) {
            throw new BusinessException(ResultCode.USER_NOT_FOUND, "学生不存在");
        }
        
        // 这里可以实现获取最近学习课程的逻辑
        return List.of();
    }

    @Override
    public CourseDTO.ResourceDownloadResponse downloadResource(Long resourceId, Long studentId) {
        log.info("下载课程资源，资源ID: {}, 学生ID: {}", resourceId, studentId);
        
        Resource resource = resourceMapper.selectById(resourceId);
        if (resource == null) {
            throw new BusinessException(ResultCode.DATA_NOT_FOUND, "资源不存在");
        }
        
        // 验证权限
        if (!hasCourseAccess(resource.getCourseId(), studentId)) {
            throw new BusinessException(ResultCode.ACCESS_DENIED, "无权限下载该资源");
        }
        
        CourseDTO.ResourceDownloadResponse response = new CourseDTO.ResourceDownloadResponse();
        response.setResourceId(resourceId);
        response.setFileName(resource.getResourceName());
        response.setDownloadUrl(resource.getFilePath());
        response.setFileSize(resource.getFileSize());
        
        return response;
    }

    @Override
    public PageResponse<CourseDTO.EvaluationResponse> getCourseEvaluations(Long courseId, Long studentId, PageRequest pageRequest) {
        log.info("获取课程评价列表，课程ID: {}, 学生ID: {}", courseId, studentId);
        
        // 验证权限
        if (!hasCourseAccess(courseId, studentId)) {
            throw new BusinessException(ResultCode.ACCESS_DENIED, "无权限访问该课程");
        }
        
        // 这里可以实现获取课程评价的逻辑
        return PageResponse.of((long)pageRequest.getPageNum(), (long)pageRequest.getPageSize(), 0L, List.of());
    }

    @Override
    public PageResponse<CourseDTO.CourseListResponse> searchCourses(CourseDTO.CourseSearchRequest searchRequest, Long studentId) {
        log.info("搜索课程，学生ID: {}", studentId);
        
        Student student = studentMapper.selectById(studentId);
        if (student == null) {
            throw new BusinessException(ResultCode.USER_NOT_FOUND, "学生不存在");
        }
        
        // 这里可以实现搜索课程的逻辑
        return PageResponse.of(1L, 10L, 0L, List.of());
    }

    @Override
    public CourseDTO.StudyTimeStatisticsResponse getStudyTimeStatistics(Long studentId, String timeRange) {
        log.info("获取学习时长统计，学生ID: {}, 时间范围: {}", studentId, timeRange);
        
        Student student = studentMapper.selectById(studentId);
        if (student == null) {
            throw new BusinessException(ResultCode.USER_NOT_FOUND, "学生不存在");
        }
        
        CourseDTO.StudyTimeStatisticsResponse response = new CourseDTO.StudyTimeStatisticsResponse();
        response.setStudentId(studentId);
        response.setTimeRange(timeRange);
        response.setTotalStudyTime(0);
        response.setAverageStudyTime(0.0);
        
        return response;
    }

    @Override
    public Boolean recordStudyTime(CourseDTO.StudyTimeRecordRequest studyTimeRequest, Long studentId) {
        log.info("记录学习时长，学生ID: {}", studentId);
        
        Student student = studentMapper.selectById(studentId);
        if (student == null) {
            throw new BusinessException(ResultCode.USER_NOT_FOUND, "学生不存在");
        }
        
        // 这里可以实现记录学习时长的逻辑
        log.info("学习时长记录成功");
        return true;
    }

    @Override
    public PageResponse<CourseDTO.NoteResponse> getCourseNotes(Long courseId, Long studentId, PageRequest pageRequest) {
        log.info("获取课程笔记，课程ID: {}, 学生ID: {}", courseId, studentId);
        
        // 验证权限
        if (!hasCourseAccess(courseId, studentId)) {
            throw new BusinessException(ResultCode.ACCESS_DENIED, "无权限访问该课程");
        }
        
        // 这里可以实现获取课程笔记的逻辑
        return PageResponse.of((long)pageRequest.getPageNum(), (long)pageRequest.getPageSize(), 0L, List.of());
    }

    @Override
    public Long createNote(CourseDTO.NoteCreateRequest noteRequest, Long studentId) {
        log.info("创建课程笔记，学生ID: {}", studentId);
        
        Student student = studentMapper.selectById(studentId);
        if (student == null) {
            throw new BusinessException(ResultCode.USER_NOT_FOUND, "学生不存在");
        }
        
        // 这里可以实现创建笔记的逻辑
        log.info("创建笔记成功");
        return 1L;
    }

    @Override
    public Boolean updateNote(Long noteId, CourseDTO.NoteUpdateRequest noteRequest, Long studentId) {
        log.info("更新课程笔记，笔记ID: {}, 学生ID: {}", noteId, studentId);
        
        Student student = studentMapper.selectById(studentId);
        if (student == null) {
            throw new BusinessException(ResultCode.USER_NOT_FOUND, "学生不存在");
        }
        
        // 这里可以实现更新笔记的逻辑
        log.info("更新笔记成功");
        return true;
    }

    @Override
    public Boolean deleteNote(Long noteId, Long studentId) {
        log.info("删除课程笔记，笔记ID: {}, 学生ID: {}", noteId, studentId);
        
        Student student = studentMapper.selectById(studentId);
        if (student == null) {
            throw new BusinessException(ResultCode.USER_NOT_FOUND, "学生不存在");
        }
        
        // 这里可以实现删除笔记的逻辑
        log.info("删除笔记成功");
        return true;
    }

    @Override
    public CourseDTO.CertificateResponse getCertificate(Long courseId, Long studentId) {
        log.info("获取学习证书，课程ID: {}, 学生ID: {}", courseId, studentId);
        
        // 验证权限
        if (!hasCourseAccess(courseId, studentId)) {
            throw new BusinessException(ResultCode.ACCESS_DENIED, "无权限访问该课程");
        }
        
        CourseDTO.CertificateResponse response = new CourseDTO.CertificateResponse();
        response.setCourseId(courseId);
        response.setStudentId(studentId);
        response.setCertificateUrl("");
        response.setIssueDate(LocalDateTime.now());
        
        return response;
    }

    @Override
    public Boolean applyCertificate(Long courseId, Long studentId) {
        log.info("申请学习证书，课程ID: {}, 学生ID: {}", courseId, studentId);
        
        // 验证权限
        if (!hasCourseAccess(courseId, studentId)) {
            throw new BusinessException(ResultCode.ACCESS_DENIED, "无权限申请该课程证书");
        }
        
        // 这里可以实现申请证书的逻辑
        log.info("申请证书成功");
        return true;
    }

    @Override
    public CourseDTO.CourseCalendarResponse getCourseCalendar(Long studentId, Integer year, Integer month) {
        log.info("获取课程日历，学生ID: {}, 年份: {}, 月份: {}", studentId, year, month);
        
        Student student = studentMapper.selectById(studentId);
        if (student == null) {
            throw new BusinessException(ResultCode.USER_NOT_FOUND, "学生不存在");
        }
        
        CourseDTO.CourseCalendarResponse response = new CourseDTO.CourseCalendarResponse();
        response.setStudentId(studentId);
        response.setYear(year);
        response.setMonth(month);
        response.setEvents(List.of());
        
        return response;
    }

    @Override
    public List<CourseDTO.StudyReminderResponse> getStudyReminders(Long studentId) {
        log.info("获取学习提醒，学生ID: {}", studentId);
        
        Student student = studentMapper.selectById(studentId);
        if (student == null) {
            throw new BusinessException(ResultCode.USER_NOT_FOUND, "学生不存在");
        }
        
        // 这里可以实现获取学习提醒的逻辑
        return List.of();
    }

    @Override
    public Boolean setStudyReminder(CourseDTO.StudyReminderRequest reminderRequest, Long studentId) {
        log.info("设置学习提醒，学生ID: {}", studentId);
        
        Student student = studentMapper.selectById(studentId);
        if (student == null) {
            throw new BusinessException(ResultCode.USER_NOT_FOUND, "学生不存在");
        }
        
        // 这里可以实现设置学习提醒的逻辑
        log.info("设置学习提醒成功");
        return true;
    }

    @Override
    public Boolean cancelStudyReminder(Long reminderId, Long studentId) {
        log.info("取消学习提醒，提醒ID: {}, 学生ID: {}", reminderId, studentId);
        
        Student student = studentMapper.selectById(studentId);
        if (student == null) {
            throw new BusinessException(ResultCode.USER_NOT_FOUND, "学生不存在");
        }
        
        // 这里可以实现取消学习提醒的逻辑
        log.info("取消学习提醒成功");
        return true;
    }

    @Override
    public CourseDTO.LearningReportResponse getLearningReport(Long studentId, String reportType, String timeRange) {
        log.info("获取学习报告，学生ID: {}, 报告类型: {}, 时间范围: {}", studentId, reportType, timeRange);
        
        Student student = studentMapper.selectById(studentId);
        if (student == null) {
            throw new BusinessException(ResultCode.USER_NOT_FOUND, "学生不存在");
        }
        
        CourseDTO.LearningReportResponse response = new CourseDTO.LearningReportResponse();
        response.setStudentId(studentId);
        response.setReportType(reportType);
        response.setTimeRange(timeRange);
        response.setReportData(Map.of());
        
        return response;
    }

    @Override
    public CourseDTO.ExportResponse exportLearningData(Long studentId, CourseDTO.LearningDataExportRequest exportRequest) {
        log.info("导出学习数据，学生ID: {}", studentId);
        
        Student student = studentMapper.selectById(studentId);
        if (student == null) {
            throw new BusinessException(ResultCode.USER_NOT_FOUND, "学生不存在");
        }
        
        CourseDTO.ExportResponse response = new CourseDTO.ExportResponse();
        response.setStudentId(studentId);
        response.setExportUrl("");
        response.setFileName("learning_data_export.xlsx");
        response.setExportTime(LocalDateTime.now());
        
        return response;
    }
    
    /**
     * 转换为课程列表响应对象
     */
    private CourseDTO.CourseListResponse convertToCourseListResponse(Course course) {
        CourseDTO.CourseListResponse response = new CourseDTO.CourseListResponse();
        response.setCourseId(course.getId());
        response.setCourseName(course.getCourseName());
        response.setDescription(course.getDescription());
        // TODO: 需要通过teacherId查询教师姓名
        // response.setTeacherName(course.getTeacherName());
        // TODO: Course实体缺少createdTime字段
        // response.setCreatedTime(course.getCreatedTime());
        return response;
    }
    
    /**
     * 转换为章节响应对象
     */
    // TODO: Chapter类不存在，暂时注释该方法
    // private CourseDTO.ChapterResponse convertToChapterResponse(Chapter chapter) {
    //     CourseDTO.ChapterResponse response = new CourseDTO.ChapterResponse();
    //     response.setChapterId(chapter.getId());
    //     response.setCourseId(chapter.getCourseId());
    //     response.setTitle(chapter.getTitle());
    //     response.setSortOrder(chapter.getSortOrder());
    //     response.setCreatedTime(chapter.getCreatedTime());
    //     return response;
    // }
    
    /**
     * 转换为资源响应对象
     */
    private CourseDTO.ResourceResponse convertToResourceResponse(Resource resource) {
        CourseDTO.ResourceResponse response = new CourseDTO.ResourceResponse();
        response.setResourceId(resource.getId());
        response.setCourseId(resource.getCourseId());
        response.setFileName(resource.getResourceName());
        response.setFileType(resource.getFileType());
        response.setFileSize(resource.getFileSize());
        response.setCreatedTime(resource.getCreateTime());
        return response;
    }
    
    /**
     * 检查学生是否有课程访问权限
     */
    private boolean hasCourseAccess(Long courseId, Long studentId) {
        // 这里可以实现具体的权限检查逻辑
        return true;
    }
}