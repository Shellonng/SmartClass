package com.education.service.student;

import com.education.dto.CourseDTO;
import com.education.dto.common.PageRequest;
import com.education.dto.common.PageResponse;

import java.util.List;

/**
 * 学生端课程服务接口
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
public interface StudentCourseService {

    /**
     * 获取学生课程列表
     * 
     * @param studentId 学生ID
     * @param pageRequest 分页请求
     * @return 课程列表
     */
    PageResponse<CourseDTO.CourseListResponse> getStudentCourses(Long studentId, PageRequest pageRequest);

    /**
     * 获取课程详情
     * 
     * @param courseId 课程ID
     * @param studentId 学生ID
     * @return 课程详情
     */
    CourseDTO.CourseDetailResponse getCourseDetail(Long courseId, Long studentId);

    /**
     * 获取课程章节列表
     * 
     * @param courseId 课程ID
     * @param studentId 学生ID
     * @return 章节列表
     */
    List<CourseDTO.ChapterResponse> getCourseChapters(Long courseId, Long studentId);

    /**
     * 获取章节详情
     * 
     * @param chapterId 章节ID
     * @param studentId 学生ID
     * @return 章节详情
     */
    CourseDTO.ChapterDetailResponse getChapterDetail(Long chapterId, Long studentId);

    /**
     * 标记章节为已学习
     * 
     * @param chapterId 章节ID
     * @param studentId 学生ID
     * @return 操作结果
     */
    Boolean markChapterAsLearned(Long chapterId, Long studentId);

    /**
     * 获取学习进度
     * 
     * @param courseId 课程ID
     * @param studentId 学生ID
     * @return 学习进度
     */
    CourseDTO.LearningProgressResponse getLearningProgress(Long courseId, Long studentId);

    /**
     * 更新学习进度
     * 
     * @param progressRequest 进度更新请求
     * @param studentId 学生ID
     * @return 操作结果
     */
    Boolean updateLearningProgress(CourseDTO.ProgressUpdateRequest progressRequest, Long studentId);

    /**
     * 获取课程公告
     * 
     * @param courseId 课程ID
     * @param studentId 学生ID
     * @param pageRequest 分页请求
     * @return 公告列表
     */
    PageResponse<CourseDTO.AnnouncementResponse> getCourseAnnouncements(Long courseId, Long studentId, PageRequest pageRequest);

    /**
     * 获取课程资源
     * 
     * @param courseId 课程ID
     * @param studentId 学生ID
     * @param pageRequest 分页请求
     * @return 资源列表
     */
    PageResponse<CourseDTO.ResourceResponse> getCourseResources(Long courseId, Long studentId, PageRequest pageRequest);

    /**
     * 下载课程资源
     * 
     * @param resourceId 资源ID
     * @param studentId 学生ID
     * @return 下载信息
     */
    CourseDTO.ResourceDownloadResponse downloadResource(Long resourceId, Long studentId);

    /**
     * 收藏课程
     * 
     * @param courseId 课程ID
     * @param studentId 学生ID
     * @return 操作结果
     */
    Boolean favoriteCourse(Long courseId, Long studentId);

    /**
     * 取消收藏课程
     * 
     * @param courseId 课程ID
     * @param studentId 学生ID
     * @return 操作结果
     */
    Boolean unfavoriteCourse(Long courseId, Long studentId);

    /**
     * 获取收藏的课程列表
     * 
     * @param studentId 学生ID
     * @param pageRequest 分页请求
     * @return 收藏课程列表
     */
    PageResponse<CourseDTO.CourseListResponse> getFavoriteCourses(Long studentId, PageRequest pageRequest);

    /**
     * 评价课程
     * 
     * @param evaluationRequest 评价请求
     * @param studentId 学生ID
     * @return 操作结果
     */
    Boolean evaluateCourse(CourseDTO.CourseEvaluationRequest evaluationRequest, Long studentId);

    /**
     * 获取课程评价
     * 
     * @param courseId 课程ID
     * @param studentId 学生ID
     * @param pageRequest 分页请求
     * @return 评价列表
     */
    PageResponse<CourseDTO.EvaluationResponse> getCourseEvaluations(Long courseId, Long studentId, PageRequest pageRequest);

    /**
     * 获取我的课程评价
     * 
     * @param courseId 课程ID
     * @param studentId 学生ID
     * @return 我的评价
     */
    CourseDTO.EvaluationResponse getMyCourseEvaluation(Long courseId, Long studentId);

    /**
     * 搜索课程
     * 
     * @param searchRequest 搜索请求
     * @param studentId 学生ID
     * @return 搜索结果
     */
    PageResponse<CourseDTO.CourseListResponse> searchCourses(CourseDTO.CourseSearchRequest searchRequest, Long studentId);

    /**
     * 获取推荐课程
     * 
     * @param studentId 学生ID
     * @param pageRequest 分页请求
     * @return 推荐课程列表
     */
    PageResponse<CourseDTO.CourseListResponse> getRecommendedCourses(Long studentId, PageRequest pageRequest);

    /**
     * 获取最近学习的课程
     * 
     * @param studentId 学生ID
     * @param limit 限制数量
     * @return 最近学习课程列表
     */
    List<CourseDTO.CourseListResponse> getRecentCourses(Long studentId, Integer limit);

    /**
     * 获取课程学习统计
     * 
     * @param courseId 课程ID
     * @param studentId 学生ID
     * @return 学习统计
     */
    CourseDTO.LearningStatisticsResponse getLearningStatistics(Long courseId, Long studentId);

    /**
     * 获取学习时长统计
     * 
     * @param studentId 学生ID
     * @param timeRange 时间范围
     * @return 学习时长统计
     */
    CourseDTO.StudyTimeStatisticsResponse getStudyTimeStatistics(Long studentId, String timeRange);

    /**
     * 记录学习时长
     * 
     * @param studyTimeRequest 学习时长记录请求
     * @param studentId 学生ID
     * @return 操作结果
     */
    Boolean recordStudyTime(CourseDTO.StudyTimeRecordRequest studyTimeRequest, Long studentId);

    /**
     * 获取课程笔记
     * 
     * @param courseId 课程ID
     * @param studentId 学生ID
     * @param pageRequest 分页请求
     * @return 笔记列表
     */
    PageResponse<CourseDTO.NoteResponse> getCourseNotes(Long courseId, Long studentId, PageRequest pageRequest);

    /**
     * 创建课程笔记
     * 
     * @param noteRequest 笔记创建请求
     * @param studentId 学生ID
     * @return 笔记ID
     */
    Long createNote(CourseDTO.NoteCreateRequest noteRequest, Long studentId);

    /**
     * 更新课程笔记
     * 
     * @param noteId 笔记ID
     * @param noteRequest 笔记更新请求
     * @param studentId 学生ID
     * @return 操作结果
     */
    Boolean updateNote(Long noteId, CourseDTO.NoteUpdateRequest noteRequest, Long studentId);

    /**
     * 删除课程笔记
     * 
     * @param noteId 笔记ID
     * @param studentId 学生ID
     * @return 操作结果
     */
    Boolean deleteNote(Long noteId, Long studentId);

    /**
     * 获取课程讨论
     * 
     * @param courseId 课程ID
     * @param studentId 学生ID
     * @param pageRequest 分页请求
     * @return 讨论列表
     */
    PageResponse<CourseDTO.DiscussionResponse> getCourseDiscussions(Long courseId, Long studentId, PageRequest pageRequest);

    /**
     * 创建课程讨论
     * 
     * @param discussionRequest 讨论创建请求
     * @param studentId 学生ID
     * @return 讨论ID
     */
    Long createDiscussion(CourseDTO.DiscussionCreateRequest discussionRequest, Long studentId);

    /**
     * 回复课程讨论
     * 
     * @param discussionId 讨论ID
     * @param replyRequest 回复请求
     * @param studentId 学生ID
     * @return 回复ID
     */
    Long replyDiscussion(Long discussionId, CourseDTO.DiscussionReplyRequest replyRequest, Long studentId);

    /**
     * 获取学习证书
     * 
     * @param courseId 课程ID
     * @param studentId 学生ID
     * @return 证书信息
     */
    CourseDTO.CertificateResponse getCertificate(Long courseId, Long studentId);

    /**
     * 申请学习证书
     * 
     * @param courseId 课程ID
     * @param studentId 学生ID
     * @return 操作结果
     */
    Boolean applyCertificate(Long courseId, Long studentId);

    /**
     * 获取课程日历
     * 
     * @param studentId 学生ID
     * @param year 年份
     * @param month 月份
     * @return 课程日历
     */
    CourseDTO.CourseCalendarResponse getCourseCalendar(Long studentId, Integer year, Integer month);

    /**
     * 获取学习提醒
     * 
     * @param studentId 学生ID
     * @return 学习提醒列表
     */
    List<CourseDTO.StudyReminderResponse> getStudyReminders(Long studentId);

    /**
     * 设置学习提醒
     * 
     * @param reminderRequest 提醒设置请求
     * @param studentId 学生ID
     * @return 操作结果
     */
    Boolean setStudyReminder(CourseDTO.StudyReminderRequest reminderRequest, Long studentId);

    /**
     * 取消学习提醒
     * 
     * @param reminderId 提醒ID
     * @param studentId 学生ID
     * @return 操作结果
     */
    Boolean cancelStudyReminder(Long reminderId, Long studentId);

    /**
     * 获取学习报告
     * 
     * @param studentId 学生ID
     * @param reportType 报告类型
     * @param timeRange 时间范围
     * @return 学习报告
     */
    CourseDTO.LearningReportResponse getLearningReport(Long studentId, String reportType, String timeRange);

    /**
     * 导出学习数据
     * 
     * @param studentId 学生ID
     * @param exportRequest 导出请求
     * @return 导出文件信息
     */
    CourseDTO.ExportResponse exportLearningData(Long studentId, CourseDTO.LearningDataExportRequest exportRequest);
}