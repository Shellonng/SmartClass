package com.education.service.student;

import com.education.dto.TaskDTO;
import com.education.dto.common.PageRequest;
import com.education.dto.common.PageResponse;

import java.util.List;

/**
 * 学生端任务服务接口
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
public interface StudentTaskService {

    /**
     * 获取学生任务列表
     * 
     * @param studentId 学生ID
     * @param pageRequest 分页请求
     * @return 任务列表
     */
    PageResponse<TaskDTO.TaskListResponse> getStudentTasks(Long studentId, PageRequest pageRequest);

    /**
     * 获取任务详情
     * 
     * @param taskId 任务ID
     * @param studentId 学生ID
     * @return 任务详情
     */
    TaskDTO.TaskDetailResponse getTaskDetail(Long taskId, Long studentId);

    /**
     * 提交任务
     * 
     * @param submissionRequest 提交请求
     * @param studentId 学生ID
     * @return 提交ID
     */
    Long submitTask(TaskDTO.TaskSubmissionRequest submissionRequest, Long studentId);

    /**
     * 保存任务草稿
     * 
     * @param draftRequest 草稿保存请求
     * @param studentId 学生ID
     * @return 操作结果
     */
    Boolean saveDraft(TaskDTO.TaskDraftRequest draftRequest, Long studentId);

    /**
     * 获取任务草稿
     * 
     * @param taskId 任务ID
     * @param studentId 学生ID
     * @return 草稿内容
     */
    TaskDTO.TaskDraftResponse getDraft(Long taskId, Long studentId);

    /**
     * 获取任务提交记录
     * 
     * @param taskId 任务ID
     * @param studentId 学生ID
     * @return 提交记录
     */
    TaskDTO.SubmissionResponse getSubmission(Long taskId, Long studentId);

    /**
     * 更新任务提交
     * 
     * @param submissionId 提交ID
     * @param updateRequest 更新请求
     * @param studentId 学生ID
     * @return 操作结果
     */
    Boolean updateSubmission(Long submissionId, TaskDTO.SubmissionUpdateRequest updateRequest, Long studentId);

    /**
     * 撤回任务提交
     * 
     * @param submissionId 提交ID
     * @param studentId 学生ID
     * @return 操作结果
     */
    Boolean withdrawSubmission(Long submissionId, Long studentId);

    /**
     * 获取任务成绩
     * 
     * @param taskId 任务ID
     * @param studentId 学生ID
     * @return 任务成绩
     */
    TaskDTO.TaskGradeResponse getTaskGrade(Long taskId, Long studentId);

    /**
     * 获取任务反馈
     * 
     * @param taskId 任务ID
     * @param studentId 学生ID
     * @return 任务反馈
     */
    TaskDTO.TaskFeedbackResponse getTaskFeedback(Long taskId, Long studentId);

    /**
     * 获取待完成任务列表
     * 
     * @param studentId 学生ID
     * @param pageRequest 分页请求
     * @return 待完成任务列表
     */
    PageResponse<TaskDTO.TaskListResponse> getPendingTasks(Long studentId, PageRequest pageRequest);

    /**
     * 获取已完成任务列表
     * 
     * @param studentId 学生ID
     * @param pageRequest 分页请求
     * @return 已完成任务列表
     */
    PageResponse<TaskDTO.TaskListResponse> getCompletedTasks(Long studentId, PageRequest pageRequest);

    /**
     * 获取逾期任务列表
     * 
     * @param studentId 学生ID
     * @param pageRequest 分页请求
     * @return 逾期任务列表
     */
    PageResponse<TaskDTO.TaskListResponse> getOverdueTasks(Long studentId, PageRequest pageRequest);

    /**
     * 搜索任务
     * 
     * @param searchRequest 搜索请求
     * @param studentId 学生ID
     * @return 搜索结果
     */
    PageResponse<TaskDTO.TaskListResponse> searchTasks(TaskDTO.TaskSearchRequest searchRequest, Long studentId);

    /**
     * 获取任务统计
     * 
     * @param studentId 学生ID
     * @return 任务统计
     */
    TaskDTO.TaskStatisticsResponse getTaskStatistics(Long studentId);

    /**
     * 获取任务日历
     * 
     * @param studentId 学生ID
     * @param year 年份
     * @param month 月份
     * @return 任务日历
     */
    TaskDTO.TaskCalendarResponse getTaskCalendar(Long studentId, Integer year, Integer month);

    /**
     * 收藏任务
     * 
     * @param taskId 任务ID
     * @param studentId 学生ID
     * @return 操作结果
     */
    Boolean favoriteTask(Long taskId, Long studentId);

    /**
     * 取消收藏任务
     * 
     * @param taskId 任务ID
     * @param studentId 学生ID
     * @return 操作结果
     */
    Boolean unfavoriteTask(Long taskId, Long studentId);

    /**
     * 获取收藏的任务列表
     * 
     * @param studentId 学生ID
     * @param pageRequest 分页请求
     * @return 收藏任务列表
     */
    PageResponse<TaskDTO.TaskListResponse> getFavoriteTasks(Long studentId, PageRequest pageRequest);

    /**
     * 获取任务提醒
     * 
     * @param studentId 学生ID
     * @return 任务提醒列表
     */
    List<TaskDTO.TaskReminderResponse> getTaskReminders(Long studentId);

    /**
     * 设置任务提醒
     * 
     * @param reminderRequest 提醒设置请求
     * @param studentId 学生ID
     * @return 操作结果
     */
    Boolean setTaskReminder(TaskDTO.TaskReminderRequest reminderRequest, Long studentId);

    /**
     * 取消任务提醒
     * 
     * @param reminderId 提醒ID
     * @param studentId 学生ID
     * @return 操作结果
     */
    Boolean cancelTaskReminder(Long reminderId, Long studentId);

    /**
     * 获取任务讨论
     * 
     * @param taskId 任务ID
     * @param studentId 学生ID
     * @param pageRequest 分页请求
     * @return 讨论列表
     */
    PageResponse<TaskDTO.TaskDiscussionResponse> getTaskDiscussions(Long taskId, Long studentId, PageRequest pageRequest);

    /**
     * 创建任务讨论
     * 
     * @param discussionRequest 讨论创建请求
     * @param studentId 学生ID
     * @return 讨论ID
     */
    Long createTaskDiscussion(TaskDTO.TaskDiscussionCreateRequest discussionRequest, Long studentId);

    /**
     * 回复任务讨论
     * 
     * @param discussionId 讨论ID
     * @param replyRequest 回复请求
     * @param studentId 学生ID
     * @return 回复ID
     */
    Long replyTaskDiscussion(Long discussionId, TaskDTO.TaskDiscussionReplyRequest replyRequest, Long studentId);

    /**
     * 获取任务资源
     * 
     * @param taskId 任务ID
     * @param studentId 学生ID
     * @return 任务资源列表
     */
    List<TaskDTO.TaskResourceResponse> getTaskResources(Long taskId, Long studentId);

    /**
     * 下载任务资源
     * 
     * @param resourceId 资源ID
     * @param studentId 学生ID
     * @return 下载信息
     */
    TaskDTO.ResourceDownloadResponse downloadTaskResource(Long resourceId, Long studentId);

    /**
     * 获取任务模板
     * 
     * @param taskId 任务ID
     * @param studentId 学生ID
     * @return 任务模板
     */
    TaskDTO.TaskTemplateResponse getTaskTemplate(Long taskId, Long studentId);

    /**
     * 获取任务评分标准
     * 
     * @param taskId 任务ID
     * @param studentId 学生ID
     * @return 评分标准
     */
    TaskDTO.GradingCriteriaResponse getGradingCriteria(Long taskId, Long studentId);

    /**
     * 获取同伴评价任务
     * 
     * @param studentId 学生ID
     * @param pageRequest 分页请求
     * @return 同伴评价任务列表
     */
    PageResponse<TaskDTO.PeerReviewTaskResponse> getPeerReviewTasks(Long studentId, PageRequest pageRequest);

    /**
     * 提交同伴评价
     * 
     * @param reviewRequest 同伴评价请求
     * @param studentId 学生ID
     * @return 操作结果
     */
    Boolean submitPeerReview(TaskDTO.PeerReviewRequest reviewRequest, Long studentId);

    /**
     * 获取我的同伴评价
     * 
     * @param taskId 任务ID
     * @param studentId 学生ID
     * @return 同伴评价列表
     */
    List<TaskDTO.PeerReviewResponse> getMyPeerReviews(Long taskId, Long studentId);

    /**
     * 申请任务延期
     * 
     * @param extensionRequest 延期申请请求
     * @param studentId 学生ID
     * @return 操作结果
     */
    Boolean requestExtension(TaskDTO.TaskExtensionRequest extensionRequest, Long studentId);

    /**
     * 获取任务延期申请状态
     * 
     * @param taskId 任务ID
     * @param studentId 学生ID
     * @return 延期申请状态
     */
    TaskDTO.ExtensionStatusResponse getExtensionStatus(Long taskId, Long studentId);

    /**
     * 获取任务进度
     * 
     * @param taskId 任务ID
     * @param studentId 学生ID
     * @return 任务进度
     */
    TaskDTO.TaskProgressResponse getTaskProgress(Long taskId, Long studentId);

    /**
     * 更新任务进度
     * 
     * @param progressRequest 进度更新请求
     * @param studentId 学生ID
     * @return 操作结果
     */
    Boolean updateTaskProgress(TaskDTO.TaskProgressUpdateRequest progressRequest, Long studentId);

    /**
     * 获取任务学习建议
     * 
     * @param taskId 任务ID
     * @param studentId 学生ID
     * @return 学习建议
     */
    TaskDTO.TaskSuggestionResponse getTaskSuggestion(Long taskId, Long studentId);

    /**
     * 获取任务相关资料推荐
     * 
     * @param taskId 任务ID
     * @param studentId 学生ID
     * @return 推荐资料列表
     */
    List<TaskDTO.RecommendedMaterialResponse> getRecommendedMaterials(Long taskId, Long studentId);

    /**
     * 获取任务完成报告
     * 
     * @param studentId 学生ID
     * @param timeRange 时间范围
     * @return 完成报告
     */
    TaskDTO.TaskCompletionReportResponse getTaskCompletionReport(Long studentId, String timeRange);

    /**
     * 导出任务数据
     * 
     * @param studentId 学生ID
     * @param exportRequest 导出请求
     * @return 导出文件信息
     */
    TaskDTO.ExportResponse exportTaskData(Long studentId, TaskDTO.TaskDataExportRequest exportRequest);
}