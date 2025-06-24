package com.education.service.student;

import com.education.dto.GradeDTO;
import com.education.dto.common.PageRequest;
import com.education.dto.common.PageResponse;

import java.util.List;

/**
 * 学生端成绩服务接口
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
public interface StudentGradeService {

    /**
     * 获取学生成绩列表
     * 
     * @param studentId 学生ID
     * @param pageRequest 分页请求
     * @return 成绩列表
     */
    PageResponse<GradeDTO.GradeListResponse> getStudentGrades(Long studentId, PageRequest pageRequest);

    /**
     * 获取课程成绩
     * 
     * @param courseId 课程ID
     * @param studentId 学生ID
     * @return 课程成绩
     */
    GradeDTO.CourseGradeResponse getCourseGrade(Long courseId, Long studentId);

    /**
     * 获取任务成绩详情
     * 
     * @param taskId 任务ID
     * @param studentId 学生ID
     * @return 任务成绩详情
     */
    GradeDTO.TaskGradeDetailResponse getTaskGradeDetail(Long taskId, Long studentId);

    /**
     * 获取成绩统计
     * 
     * @param studentId 学生ID
     * @return 成绩统计
     */
    GradeDTO.GradeStatisticsResponse getGradeStatistics(Long studentId);

    /**
     * 获取成绩趋势
     * 
     * @param studentId 学生ID
     * @param timeRange 时间范围
     * @return 成绩趋势
     */
    GradeDTO.GradeTrendResponse getGradeTrend(Long studentId, String timeRange);

    /**
     * 获取成绩分布
     * 
     * @param studentId 学生ID
     * @param courseId 课程ID（可选）
     * @return 成绩分布
     */
    GradeDTO.GradeDistributionResponse getGradeDistribution(Long studentId, Long courseId);

    /**
     * 获取班级排名
     * 
     * @param studentId 学生ID
     * @param classId 班级ID
     * @return 班级排名
     */
    GradeDTO.ClassRankingResponse getClassRanking(Long studentId, Long classId);

    /**
     * 获取课程排名
     * 
     * @param studentId 学生ID
     * @param courseId 课程ID
     * @return 课程排名
     */
    GradeDTO.CourseRankingResponse getCourseRanking(Long studentId, Long courseId);

    /**
     * 获取成绩对比分析
     * 
     * @param studentId 学生ID
     * @param compareRequest 对比请求
     * @return 对比分析结果
     */
    GradeDTO.GradeComparisonResponse getGradeComparison(Long studentId, GradeDTO.GradeComparisonRequest compareRequest);

    /**
     * 获取学期成绩汇总
     * 
     * @param studentId 学生ID
     * @param semester 学期
     * @return 学期成绩汇总
     */
    GradeDTO.SemesterGradeSummaryResponse getSemesterGradeSummary(Long studentId, String semester);

    /**
     * 获取年度成绩汇总
     * 
     * @param studentId 学生ID
     * @param year 年份
     * @return 年度成绩汇总
     */
    GradeDTO.YearlyGradeSummaryResponse getYearlyGradeSummary(Long studentId, Integer year);

    /**
     * 获取成绩预警信息
     * 
     * @param studentId 学生ID
     * @return 成绩预警信息
     */
    List<GradeDTO.GradeWarningResponse> getGradeWarnings(Long studentId);

    /**
     * 获取成绩改进建议
     * 
     * @param studentId 学生ID
     * @param courseId 课程ID（可选）
     * @return 改进建议
     */
    GradeDTO.ImprovementSuggestionResponse getImprovementSuggestions(Long studentId, Long courseId);

    /**
     * 获取学习目标完成情况
     * 
     * @param studentId 学生ID
     * @return 目标完成情况
     */
    GradeDTO.LearningGoalProgressResponse getLearningGoalProgress(Long studentId);

    /**
     * 设置学习目标
     * 
     * @param goalRequest 目标设置请求
     * @param studentId 学生ID
     * @return 操作结果
     */
    Boolean setLearningGoal(GradeDTO.LearningGoalRequest goalRequest, Long studentId);

    /**
     * 更新学习目标
     * 
     * @param goalId 目标ID
     * @param goalRequest 目标更新请求
     * @param studentId 学生ID
     * @return 操作结果
     */
    Boolean updateLearningGoal(Long goalId, GradeDTO.LearningGoalUpdateRequest goalRequest, Long studentId);

    /**
     * 删除学习目标
     * 
     * @param goalId 目标ID
     * @param studentId 学生ID
     * @return 操作结果
     */
    Boolean deleteLearningGoal(Long goalId, Long studentId);

    /**
     * 获取成绩证书
     * 
     * @param studentId 学生ID
     * @param courseId 课程ID
     * @return 成绩证书
     */
    GradeDTO.GradeCertificateResponse getGradeCertificate(Long studentId, Long courseId);

    /**
     * 申请成绩证书
     * 
     * @param certificateRequest 证书申请请求
     * @param studentId 学生ID
     * @return 操作结果
     */
    Boolean applyCertificate(GradeDTO.CertificateApplicationRequest certificateRequest, Long studentId);

    /**
     * 获取成绩历史记录
     * 
     * @param studentId 学生ID
     * @param taskId 任务ID
     * @return 成绩历史记录
     */
    List<GradeDTO.GradeHistoryResponse> getGradeHistory(Long studentId, Long taskId);

    /**
     * 获取成绩详细反馈
     * 
     * @param studentId 学生ID
     * @param taskId 任务ID
     * @return 详细反馈
     */
    GradeDTO.DetailedFeedbackResponse getDetailedFeedback(Long studentId, Long taskId);

    /**
     * 获取同伴评价结果
     * 
     * @param studentId 学生ID
     * @param taskId 任务ID
     * @return 同伴评价结果
     */
    List<GradeDTO.PeerEvaluationResponse> getPeerEvaluationResults(Long studentId, Long taskId);

    /**
     * 获取自我评价记录
     * 
     * @param studentId 学生ID
     * @param taskId 任务ID
     * @return 自我评价记录
     */
    GradeDTO.SelfEvaluationResponse getSelfEvaluation(Long studentId, Long taskId);

    /**
     * 提交自我评价
     * 
     * @param evaluationRequest 自我评价请求
     * @param studentId 学生ID
     * @return 操作结果
     */
    Boolean submitSelfEvaluation(GradeDTO.SelfEvaluationRequest evaluationRequest, Long studentId);

    /**
     * 获取能力雷达图
     * 
     * @param studentId 学生ID
     * @param courseId 课程ID（可选）
     * @return 能力雷达图数据
     */
    GradeDTO.AbilityRadarResponse getAbilityRadar(Long studentId, Long courseId);

    /**
     * 获取学习效率分析
     * 
     * @param studentId 学生ID
     * @param timeRange 时间范围
     * @return 学习效率分析
     */
    GradeDTO.LearningEfficiencyResponse getLearningEfficiency(Long studentId, String timeRange);

    /**
     * 获取知识点掌握情况
     * 
     * @param studentId 学生ID
     * @param courseId 课程ID
     * @return 知识点掌握情况
     */
    GradeDTO.KnowledgePointMasteryResponse getKnowledgePointMastery(Long studentId, Long courseId);

    /**
     * 获取错题分析
     * 
     * @param studentId 学生ID
     * @param pageRequest 分页请求
     * @return 错题分析
     */
    PageResponse<GradeDTO.WrongQuestionAnalysisResponse> getWrongQuestionAnalysis(Long studentId, PageRequest pageRequest);

    /**
     * 获取薄弱知识点
     * 
     * @param studentId 学生ID
     * @param courseId 课程ID（可选）
     * @return 薄弱知识点列表
     */
    List<GradeDTO.WeakKnowledgePointResponse> getWeakKnowledgePoints(Long studentId, Long courseId);

    /**
     * 获取学习建议
     * 
     * @param studentId 学生ID
     * @return 学习建议
     */
    GradeDTO.StudySuggestionResponse getStudySuggestions(Long studentId);

    /**
     * 获取成绩报告
     * 
     * @param studentId 学生ID
     * @param reportType 报告类型
     * @param timeRange 时间范围
     * @return 成绩报告
     */
    GradeDTO.GradeReportResponse getGradeReport(Long studentId, String reportType, String timeRange);

    /**
     * 导出成绩数据
     * 
     * @param studentId 学生ID
     * @param exportRequest 导出请求
     * @return 导出文件信息
     */
    GradeDTO.ExportResponse exportGradeData(Long studentId, GradeDTO.GradeDataExportRequest exportRequest);

    /**
     * 获取成绩通知
     * 
     * @param studentId 学生ID
     * @param pageRequest 分页请求
     * @return 成绩通知列表
     */
    PageResponse<GradeDTO.GradeNotificationResponse> getGradeNotifications(Long studentId, PageRequest pageRequest);

    /**
     * 标记成绩通知为已读
     * 
     * @param notificationId 通知ID
     * @param studentId 学生ID
     * @return 操作结果
     */
    Boolean markNotificationAsRead(Long notificationId, Long studentId);

    /**
     * 获取成绩申诉记录
     * 
     * @param studentId 学生ID
     * @param pageRequest 分页请求
     * @return 申诉记录列表
     */
    PageResponse<GradeDTO.GradeAppealResponse> getGradeAppeals(Long studentId, PageRequest pageRequest);

    /**
     * 提交成绩申诉
     * 
     * @param appealRequest 申诉请求
     * @param studentId 学生ID
     * @return 申诉ID
     */
    Long submitGradeAppeal(GradeDTO.GradeAppealRequest appealRequest, Long studentId);

    /**
     * 获取申诉处理结果
     * 
     * @param appealId 申诉ID
     * @param studentId 学生ID
     * @return 处理结果
     */
    GradeDTO.AppealResultResponse getAppealResult(Long appealId, Long studentId);
}