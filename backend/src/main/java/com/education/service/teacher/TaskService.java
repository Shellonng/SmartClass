package com.education.service.teacher;

import com.education.dto.task.TaskCommonDTOs.*;
import com.education.dto.common.PageRequest;
import com.education.dto.common.PageResponse;

import java.util.List;

/**
 * 教师端任务服务接口
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
public interface TaskService {

    /**
     * 获取任务列表（支持多条件查询）
     */
    PageResponse<TaskListResponse> getTaskList(
        PageRequest pageRequest, 
        String title, 
        String status, 
        String type, 
        Long courseId, 
        Long classId
    );

    /**
     * 创建任务
     */
    TaskResponse createTask(TaskCreateRequest createRequest);

    /**
     * 获取任务详情
     */
    TaskDetailResponse getTaskDetail(Long taskId);

    /**
     * 更新任务
     */
    TaskResponse updateTask(Long taskId, TaskUpdateRequest updateRequest);

    /**
     * 删除任务
     */
    Boolean deleteTask(Long taskId);

    /**
     * 发布任务
     */
    Boolean publishTask(Long taskId);

    /**
     * 取消发布任务
     */
    Boolean unpublishTask(Long taskId);

    /**
     * 获取任务提交列表
     */
    PageResponse<TaskSubmissionResponse> getTaskSubmissions(
        Long taskId, 
        PageRequest pageRequest, 
        String studentName, 
        String submissionStatus
    );

    /**
     * 批改提交
     */
    Boolean gradeSubmission(Long submissionId, TaskGradeRequest gradeRequest);

    /**
     * 批量批改
     */
    Boolean batchGradeSubmissions(List<TaskBatchGradeRequest> gradeRequests);

    /**
     * 获取任务统计
     */
    TaskStatisticsResponse getTaskStatistics(Long taskId);

    /**
     * 导出任务成绩
     */
    String exportTaskGrades(Long taskId);

    /**
     * 复制任务
     */
    TaskResponse copyTask(Long taskId, TaskCopyRequest copyRequest);

    /**
     * 延长任务截止时间
     */
    Boolean extendTaskDeadline(Long taskId, TaskExtendRequest extendRequest);

    /**
     * 启用AI批改
     */
    Boolean enableAIGrading(Long taskId);

    /**
     * 获取任务模板
     */
    PageResponse<TaskTemplateResponse> getTaskTemplates(String category, String keyword);

    /**
     * 从模板创建任务
     */
    TaskResponse createTaskFromTemplate(Long templateId, TaskFromTemplateRequest fromTemplateRequest);

    /**
     * 获取任务详情
     * 
     * @param taskId 任务ID
     * @param teacherId 教师ID
     * @return 任务详情
     */
    TaskDTO.TaskDetailResponse getTaskDetail(Long taskId, Long teacherId);

    /**
     * 更新任务
     * 
     * @param taskId 任务ID
     * @param updateRequest 更新请求
     * @param teacherId 教师ID
     * @return 更新后的任务信息
     */
    TaskDTO.TaskResponse updateTask(Long taskId, TaskDTO.TaskUpdateRequest updateRequest, Long teacherId);

    /**
     * 删除任务
     * 
     * @param taskId 任务ID
     * @param teacherId 教师ID
     * @return 操作结果
     */
    Boolean deleteTask(Long taskId, Long teacherId);

    /**
     * 发布任务
     * 
     * @param taskId 任务ID
     * @param teacherId 教师ID
     * @return 操作结果
     */
    Boolean publishTask(Long taskId, Long teacherId);

    /**
     * 关闭任务
     * 
     * @param taskId 任务ID
     * @param teacherId 教师ID
     * @return 操作结果
     */
    Boolean closeTask(Long taskId, Long teacherId);

    /**
     * 获取任务提交列表
     * 
     * @param taskId 任务ID
     * @param teacherId 教师ID
     * @param pageRequest 分页请求
     * @return 提交列表
     */
    PageResponse<Object> getTaskSubmissions(Long taskId, Long teacherId, PageRequest pageRequest);

    /**
     * 批改任务提交
     * 
     * @param submissionId 提交ID
     * @param gradeRequest 批改请求
     * @param teacherId 教师ID
     * @return 操作结果
     */
    Boolean gradeSubmission(Long submissionId, Object gradeRequest, Long teacherId);

    /**
     * 批量批改任务提交
     * 
     * @param gradeRequests 批改请求列表
     * @param teacherId 教师ID
     * @return 操作结果
     */
    Boolean batchGradeSubmissions(List<Object> gradeRequests, Long teacherId);

    /**
     * 获取任务统计
     * 
     * @param taskId 任务ID
     * @param teacherId 教师ID
     * @return 任务统计
     */
    Object getTaskStatistics(Long taskId, Long teacherId);

    /**
     * 导出任务成绩
     * 
     * @param taskId 任务ID
     * @param teacherId 教师ID
     * @return 导出文件路径
     */
    String exportTaskGrades(Long taskId, Long teacherId);

    /**
     * 复制任务
     * 
     * @param taskId 任务ID
     * @param newTaskTitle 新任务标题
     * @param teacherId 教师ID
     * @return 新任务信息
     */
    TaskDTO.TaskResponse copyTask(Long taskId, String newTaskTitle, Long teacherId);

    /**
     * 获取抄袭检测结果
     * 
     * @param taskId 任务ID
     * @param teacherId 教师ID
     * @return 抄袭检测结果
     */
    Object getPlagiarismDetectionResult(Long taskId, Long teacherId);

    /**
     * 启动抄袭检测
     * 
     * @param taskId 任务ID
     * @param teacherId 教师ID
     * @return 操作结果
     */
    Boolean startPlagiarismDetection(Long taskId, Long teacherId);

    /**
     * 获取任务模板列表
     * 
     * @param teacherId 教师ID
     * @return 模板列表
     */
    List<Object> getTaskTemplates(Long teacherId);

    /**
     * 从模板创建任务
     * 
     * @param templateId 模板ID
     * @param taskTitle 任务标题
     * @param teacherId 教师ID
     * @return 任务信息
     */
    TaskDTO.TaskResponse createTaskFromTemplate(Long templateId, TaskDTO.TaskCreateRequest createRequest, Long teacherId);

    /**
     * 保存任务为模板
     * 
     * @param taskId 任务ID
     * @param templateName 模板名称
     * @param teacherId 教师ID
     * @return 操作结果
     */
    Boolean saveTaskAsTemplate(Long taskId, String templateName, Long teacherId);

    /**
     * 设置任务权重
     * 
     * @param taskId 任务ID
     * @param weight 权重
     * @param teacherId 教师ID
     * @return 操作结果
     */
    Boolean setTaskWeight(Long taskId, Double weight, Long teacherId);

    /**
     * 获取任务提交统计
     * 
     * @param taskId 任务ID
     * @param teacherId 教师ID
     * @return 提交统计
     */
    Object getTaskSubmissionStatistics(Long taskId, Long teacherId);

    /**
     * 发送任务提醒
     * 
     * @param taskId 任务ID
     * @param reminderType 提醒类型
     * @param teacherId 教师ID
     * @return 操作结果
     */
    Boolean sendTaskReminder(Long taskId, String reminderType, Long teacherId);

    /**
     * 获取任务评论列表
     * 
     * @param taskId 任务ID
     * @param teacherId 教师ID
     * @param pageRequest 分页请求
     * @return 评论列表
     */
    PageResponse<Object> getTaskComments(Long taskId, Long teacherId, PageRequest pageRequest);

    /**
     * 回复任务评论
     * 
     * @param commentId 评论ID
     * @param reply 回复内容
     * @param teacherId 教师ID
     * @return 操作结果
     */
    Boolean replyTaskComment(Long commentId, String reply, Long teacherId);

    /**
     * 设置任务可见性
     * 
     * @param taskId 任务ID
     * @param visibility 可见性设置
     * @param teacherId 教师ID
     * @return 操作结果
     */
    Boolean setTaskVisibility(Long taskId, Object visibility, Long teacherId);

    /**
     * 获取任务分析报告
     * 
     * @param taskId 任务ID
     * @param teacherId 教师ID
     * @return 分析报告
     */
    Object getTaskAnalysisReport(Long taskId, Long teacherId);

    /**
     * 导入任务
     * 
     * @param importRequest 导入请求
     * @param teacherId 教师ID
     * @return 导入结果
     */
    Object importTask(Object importRequest, Long teacherId);

    /**
     * 导出任务
     * 
     * @param taskId 任务ID
     * @param teacherId 教师ID
     * @return 导出文件路径
     */
    String exportTask(Long taskId, Long teacherId);

    /**
     * 设置任务评分标准
     * 
     * @param taskId 任务ID
     * @param gradingCriteria 评分标准
     * @param teacherId 教师ID
     * @return 操作结果
     */
    Boolean setTaskGradingCriteria(Long taskId, Object gradingCriteria, Long teacherId);

    /**
     * 获取任务评分标准
     * 
     * @param taskId 任务ID
     * @param teacherId 教师ID
     * @return 评分标准
     */
    Object getTaskGradingCriteria(Long taskId, Long teacherId);

    /**
     * 自动批改任务
     * 
     * @param taskId 任务ID
     * @param teacherId 教师ID
     * @return 操作结果
     */
    Boolean autoGradeTask(Long taskId, Long teacherId);

    /**
     * 获取任务难度分析
     * 
     * @param taskId 任务ID
     * @param teacherId 教师ID
     * @return 难度分析
     */
    Object getTaskDifficultyAnalysis(Long taskId, Long teacherId);

    /**
     * 设置任务标签
     * 
     * @param taskId 任务ID
     * @param tags 标签列表
     * @param teacherId 教师ID
     * @return 操作结果
     */
    Boolean setTaskTags(Long taskId, List<String> tags, Long teacherId);

    /**
     * 获取任务标签
     * 
     * @param taskId 任务ID
     * @param teacherId 教师ID
     * @return 标签列表
     */
    List<String> getTaskTags(Long taskId, Long teacherId);

    /**
     * 归档任务
     * 
     * @param taskId 任务ID
     * @param teacherId 教师ID
     * @return 操作结果
     */
    Boolean archiveTask(Long taskId, Long teacherId);

    /**
     * 恢复归档任务
     * 
     * @param taskId 任务ID
     * @param teacherId 教师ID
     * @return 操作结果
     */
    Boolean restoreTask(Long taskId, Long teacherId);

    /**
     * 导出任务数据
     * 
     * @param taskId 任务ID
     * @param teacherId 教师ID
     * @return 导出文件路径
     */
    String exportTasks(Long taskId, Long teacherId);

    /**
     * 设置评分标准
     * 
     * @param taskId 任务ID
     * @param gradingCriteria 评分标准
     * @param teacherId 教师ID
     * @return 操作结果
     */
    Boolean setGradingCriteria(Long taskId, TaskDTO.GradingCriteria gradingCriteria, Long teacherId);

    /**
     * 获取评分标准
     * 
     * @param taskId 任务ID
     * @param teacherId 教师ID
     * @return 评分标准
     */
    TaskDTO.GradingCriteria getGradingCriteria(Long taskId, Long teacherId);
}