package com.education.service.teacher;

import com.education.dto.CourseDTO;
import com.education.dto.common.PageRequest;
import com.education.dto.common.PageResponse;

import java.util.List;

/**
 * 教师端课程服务接口
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
public interface CourseService {

    /**
     * 创建课程
     * 
     * @param createRequest 创建课程请求
     * @param teacherId 教师ID
     * @return 课程信息
     */
    CourseDTO.CourseResponse createCourse(Object createRequest, Long teacherId);

    /**
     * 获取教师的课程列表
     * 
     * @param teacherId 教师ID
     * @param pageRequest 分页请求
     * @return 课程列表
     */
    PageResponse<CourseDTO.CourseResponse> getCourseList(Long teacherId, PageRequest pageRequest);

    /**
     * 获取课程详情
     * 
     * @param courseId 课程ID
     * @param teacherId 教师ID
     * @return 课程详情
     */
    CourseDTO.CourseDetailResponse getCourseDetail(Long courseId, Long teacherId);

    /**
     * 更新课程信息
     * 
     * @param courseId 课程ID
     * @param updateRequest 更新请求
     * @param teacherId 教师ID
     * @return 更新后的课程信息
     */
    CourseDTO.CourseResponse updateCourse(Long courseId, CourseDTO.CourseUpdateRequest updateRequest, Long teacherId);

    /**
     * 删除课程
     * 
     * @param courseId 课程ID
     * @param teacherId 教师ID
     * @return 操作结果
     */
    Boolean deleteCourse(Long courseId, Long teacherId);

    /**
     * 发布课程
     * 
     * @param courseId 课程ID
     * @param teacherId 教师ID
     * @return 操作结果
     */
    Boolean publishCourse(Long courseId, Long teacherId);

    /**
     * 下架课程
     * 
     * @param courseId 课程ID
     * @param teacherId 教师ID
     * @return 操作结果
     */
    Boolean unpublishCourse(Long courseId, Long teacherId);

    /**
     * 获取课程章节列表
     * 
     * @param courseId 课程ID
     * @param teacherId 教师ID
     * @return 章节列表
     */
    List<CourseDTO.ChapterResponse> getCourseChapters(Long courseId, Long teacherId);

    /**
     * 创建课程章节
     * 
     * @param courseId 课程ID
     * @param createRequest 章节请求
     * @param teacherId 教师ID
     * @return 章节信息
     */
    CourseDTO.ChapterResponse createChapter(Long courseId, CourseDTO.ChapterCreateRequest createRequest, Long teacherId);

    /**
     * 更新课程章节
     * 
     * @param chapterId 章节ID
     * @param chapterRequest 章节请求
     * @param teacherId 教师ID
     * @return 更新后的章节信息
     */
    CourseDTO.ChapterResponse updateChapter(Long chapterId, CourseDTO.ChapterUpdateRequest chapterRequest, Long teacherId);

    /**
     * 删除课程章节
     * 
     * @param chapterId 章节ID
     * @param teacherId 教师ID
     * @return 操作结果
     */
    Boolean deleteChapter(Long chapterId, Long teacherId);

    /**
     * 调整章节顺序
     * 
     * @param courseId 课程ID
     * @param chapterOrders 章节顺序列表
     * @param teacherId 教师ID
     * @return 操作结果
     */
    Boolean reorderChapters(Long courseId, List<CourseDTO.ChapterOrderRequest> chapterOrders, Long teacherId);

    /**
     * 获取课程统计信息
     * 
     * @param courseId 课程ID
     * @param teacherId 教师ID
     * @return 统计信息
     */
    CourseDTO.CourseStatisticsResponse getCourseStatistics(Long courseId, Long teacherId);

    /**
     * 获取课程学生列表
     * 
     * @param courseId 课程ID
     * @param teacherId 教师ID
     * @param pageRequest 分页请求
     * @return 学生列表
     */
    PageResponse<Object> getCourseStudents(Long courseId, Long teacherId, PageRequest pageRequest);

    /**
     * 复制课程
     * 
     * @param courseId 课程ID
     * @param newCourseName 新课程名称
     * @param teacherId 教师ID
     * @return 新课程信息
     */
    CourseDTO.CourseResponse copyCourse(Long courseId, String newCourseName, Long teacherId);

    /**
     * 导出课程内容
     * 
     * @param courseId 课程ID
     * @param teacherId 教师ID
     * @return 导出文件路径
     */
    String exportCourse(Long courseId, Long teacherId);

    /**
     * 导入课程内容
     * 
     * @param importRequest 导入请求
     * @param teacherId 教师ID
     * @return 导入结果
     */
    Object importCourse(CourseDTO.CourseImportRequest importRequest, Long teacherId);

    /**
     * 获取课程学习进度统计
     * 
     * @param courseId 课程ID
     * @param teacherId 教师ID
     * @return 学习进度统计
     */
    Object getCourseProgressStatistics(Long courseId, Long teacherId);

    /**
     * 设置课程权限
     * 
     * @param courseId 课程ID
     * @param permissions 权限设置
     * @param teacherId 教师ID
     * @return 操作结果
     */
    Boolean setCoursePermissions(Long courseId, Object permissions, Long teacherId);

    /**
     * 获取课程评价列表
     * 
     * @param courseId 课程ID
     * @param teacherId 教师ID
     * @param pageRequest 分页请求
     * @return 评价列表
     */
    PageResponse<Object> getCourseReviews(Long courseId, Long teacherId, PageRequest pageRequest);

    /**
     * 回复课程评价
     * 
     * @param reviewId 评价ID
     * @param reply 回复内容
     * @param teacherId 教师ID
     * @return 操作结果
     */
    Boolean replyCourseReview(Long reviewId, String reply, Long teacherId);

    /**
     * 获取课程资源列表
     * 
     * @param courseId 课程ID
     * @param teacherId 教师ID
     * @param pageRequest 分页请求
     * @return 资源列表
     */
    PageResponse<Object> getCourseResources(Long courseId, Long teacherId, PageRequest pageRequest);

    /**
     * 添加课程资源
     * 
     * @param courseId 课程ID
     * @param resourceIds 资源ID列表
     * @param teacherId 教师ID
     * @return 操作结果
     */
    Boolean addCourseResources(Long courseId, List<Long> resourceIds, Long teacherId);

    /**
     * 移除课程资源
     * 
     * @param courseId 课程ID
     * @param resourceId 资源ID
     * @param teacherId 教师ID
     * @return 操作结果
     */
    Boolean removeCourseResource(Long courseId, Long resourceId, Long teacherId);

    /**
     * 获取课程任务列表
     * 
     * @param courseId 课程ID
     * @param teacherId 教师ID
     * @param pageRequest 分页请求
     * @return 任务列表
     */
    PageResponse<Object> getCourseTasks(Long courseId, Long teacherId, PageRequest pageRequest);

    /**
     * 创建课程公告
     * 
     * @param courseId 课程ID
     * @param announcement 公告内容
     * @param teacherId 教师ID
     * @return 公告信息
     */
    Object createCourseAnnouncement(Long courseId, Object announcement, Long teacherId);

    /**
     * 获取课程公告列表
     * 
     * @param courseId 课程ID
     * @param teacherId 教师ID
     * @param pageRequest 分页请求
     * @return 公告列表
     */
    PageResponse<Object> getCourseAnnouncements(Long courseId, Long teacherId, PageRequest pageRequest);

    /**
     * 更新课程公告
     * 
     * @param announcementId 公告ID
     * @param announcement 公告内容
     * @param teacherId 教师ID
     * @return 操作结果
     */
    Boolean updateCourseAnnouncement(Long announcementId, Object announcement, Long teacherId);

    /**
     * 删除课程公告
     * 
     * @param announcementId 公告ID
     * @param teacherId 教师ID
     * @return 操作结果
     */
    Boolean deleteCourseAnnouncement(Long announcementId, Long teacherId);

    /**
     * 设置课程标签
     * 
     * @param courseId 课程ID
     * @param tags 标签列表
     * @param teacherId 教师ID
     * @return 操作结果
     */
    Boolean setCourseTags(Long courseId, List<String> tags, Long teacherId);

    /**
     * 获取课程标签
     * 
     * @param courseId 课程ID
     * @param teacherId 教师ID
     * @return 标签列表
     */
    List<String> getCourseTags(Long courseId, Long teacherId);

    /**
     * 归档课程
     * 
     * @param courseId 课程ID
     * @param teacherId 教师ID
     * @return 操作结果
     */
    Boolean archiveCourse(Long courseId, Long teacherId);

    /**
     * 恢复归档课程
     * 
     * @param courseId 课程ID
     * @param teacherId 教师ID
     * @return 操作结果
     */
    Boolean restoreCourse(Long courseId, Long teacherId);

    /**
     * 获取课程模板列表
     * 
     * @param teacherId 教师ID
     * @return 模板列表
     */
    List<Object> getCourseTemplates(Long teacherId);

    /**
     * 从模板创建课程
     * 
     * @param templateId 模板ID
     * @param courseName 课程名称
     * @param teacherId 教师ID
     * @return 课程信息
     */
    CourseDTO.CourseResponse createCourseFromTemplate(Long templateId, String courseName, Long teacherId);

    /**
     * 保存课程为模板
     * 
     * @param courseId 课程ID
     * @param templateName 模板名称
     * @param teacherId 教师ID
     * @return 操作结果
     */
    Boolean saveCourseAsTemplate(Long courseId, String templateName, Long teacherId);
}