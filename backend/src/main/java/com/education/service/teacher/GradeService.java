package com.education.service.teacher;

import com.education.dto.GradeDTO;
import com.education.dto.common.PageRequest;
import com.education.dto.common.PageResponse;

import java.util.List;

/**
 * 教师端成绩服务接口
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
public interface GradeService {

    /**
     * 获取成绩列表
     * 
     * @param teacherId 教师ID
     * @param pageRequest 分页请求
     * @return 成绩列表
     */
    PageResponse<GradeDTO.GradeResponse> getGradeList(Long teacherId, PageRequest pageRequest);

    /**
     * 录入成绩
     * 
     * @param gradeRequest 成绩录入请求
     * @param teacherId 教师ID
     * @return 成绩信息
     */
    GradeDTO.GradeResponse createGrade(GradeDTO.GradeCreateRequest gradeRequest, Long teacherId);

    /**
     * 批量录入成绩
     * 
     * @param gradeRequests 批量成绩录入请求
     * @param teacherId 教师ID
     * @return 录入结果
     */
    GradeDTO.BatchGradeResponse batchCreateGrades(List<GradeDTO.GradeCreateRequest> gradeRequests, Long teacherId);

    /**
     * 更新成绩
     * 
     * @param gradeId 成绩ID
     * @param updateRequest 更新请求
     * @param teacherId 教师ID
     * @return 更新后的成绩信息
     */
    GradeDTO.GradeResponse updateGrade(Long gradeId, GradeDTO.GradeUpdateRequest updateRequest, Long teacherId);

    /**
     * 删除成绩
     * 
     * @param gradeId 成绩ID
     * @param teacherId 教师ID
     * @return 操作结果
     */
    Boolean deleteGrade(Long gradeId, Long teacherId);

    /**
     * 发布成绩
     * 
     * @param gradeIds 成绩ID列表
     * @param teacherId 教师ID
     * @return 操作结果
     */
    Boolean publishGrades(List<Long> gradeIds, Long teacherId);

    /**
     * 批量发布成绩
     * 
     * @param courseId 课程ID
     * @param taskId 任务ID（可选）
     * @param teacherId 教师ID
     * @return 操作结果
     */
    Boolean batchPublishGrades(Long courseId, Long taskId, Long teacherId);

    /**
     * 获取成绩统计
     * 
     * @param courseId 课程ID
     * @param taskId 任务ID（可选）
     * @param teacherId 教师ID
     * @return 成绩统计
     */
    GradeDTO.GradeStatisticsResponse getGradeStatistics(Long courseId, Long taskId, Long teacherId);

    /**
     * 获取学生成绩详情
     * 
     * @param studentId 学生ID
     * @param courseId 课程ID
     * @param teacherId 教师ID
     * @return 学生成绩详情
     */
    GradeDTO.StudentGradeDetailResponse getStudentGradeDetail(Long studentId, Long courseId, Long teacherId);

    /**
     * 导出成绩
     * 
     * @param exportRequest 导出请求
     * @param teacherId 教师ID
     * @return 导出文件路径
     */
    String exportGrades(GradeDTO.GradeExportRequest exportRequest, Long teacherId);

    /**
     * 导入成绩
     * 
     * @param importRequest 导入请求
     * @param teacherId 教师ID
     * @return 导入结果
     */
    GradeDTO.GradeImportResponse importGrades(GradeDTO.GradeImportRequest importRequest, Long teacherId);

    /**
     * 获取成绩分布
     * 
     * @param courseId 课程ID
     * @param taskId 任务ID（可选）
     * @param teacherId 教师ID
     * @return 成绩分布
     */
    GradeDTO.GradeDistributionResponse getGradeDistribution(Long courseId, Long taskId, Long teacherId);

    /**
     * 获取成绩趋势
     * 
     * @param studentId 学生ID
     * @param courseId 课程ID
     * @param timeRange 时间范围
     * @param teacherId 教师ID
     * @return 成绩趋势
     */
    GradeDTO.GradeTrendResponse getGradeTrend(Long studentId, Long courseId, String timeRange, Long teacherId);

    /**
     * 获取成绩排名
     * 
     * @param courseId 课程ID
     * @param taskId 任务ID（可选）
     * @param teacherId 教师ID
     * @return 成绩排名
     */
    List<GradeDTO.GradeRankingResponse> getGradeRanking(Long courseId, Long taskId, Long teacherId);

    /**
     * 设置成绩权重
     * 
     * @param courseId 课程ID
     * @param gradeWeights 成绩权重设置
     * @param teacherId 教师ID
     * @return 操作结果
     */
    Boolean setGradeWeights(Long courseId, List<GradeDTO.GradeWeightRequest> gradeWeights, Long teacherId);

    /**
     * 获取成绩权重
     * 
     * @param courseId 课程ID
     * @param teacherId 教师ID
     * @return 成绩权重
     */
    List<GradeDTO.GradeWeightResponse> getGradeWeights(Long courseId, Long teacherId);

    /**
     * 计算总成绩
     * 
     * @param studentId 学生ID
     * @param courseId 课程ID
     * @param teacherId 教师ID
     * @return 总成绩
     */
    GradeDTO.TotalGradeResponse calculateTotalGrade(Long studentId, Long courseId, Long teacherId);

    /**
     * 批量计算总成绩
     * 
     * @param courseId 课程ID
     * @param teacherId 教师ID
     * @return 操作结果
     */
    Boolean batchCalculateTotalGrades(Long courseId, Long teacherId);

    /**
     * 获取成绩分析报告
     * 
     * @param courseId 课程ID
     * @param teacherId 教师ID
     * @return 分析报告
     */
    GradeDTO.GradeAnalysisResponse getGradeAnalysis(Long courseId, Long teacherId);

    /**
     * 设置及格线
     * 
     * @param courseId 课程ID
     * @param passingGrade 及格分数
     * @param teacherId 教师ID
     * @return 操作结果
     */
    Boolean setPassingGrade(Long courseId, Double passingGrade, Long teacherId);

    /**
     * 获取不及格学生列表
     * 
     * @param courseId 课程ID
     * @param teacherId 教师ID
     * @param pageRequest 分页请求
     * @return 不及格学生列表
     */
    PageResponse<Object> getFailingStudents(Long courseId, Long teacherId, PageRequest pageRequest);

    /**
     * 发送成绩通知
     * 
     * @param gradeIds 成绩ID列表
     * @param message 通知消息
     * @param teacherId 教师ID
     * @return 操作结果
     */
    Boolean sendGradeNotification(List<Long> gradeIds, String message, Long teacherId);

    /**
     * 获取成绩修改历史
     * 
     * @param gradeId 成绩ID
     * @param teacherId 教师ID
     * @return 修改历史
     */
    List<Object> getGradeHistory(Long gradeId, Long teacherId);

    /**
     * 恢复成绩版本
     * 
     * @param gradeId 成绩ID
     * @param versionId 版本ID
     * @param teacherId 教师ID
     * @return 操作结果
     */
    Boolean restoreGradeVersion(Long gradeId, Long versionId, Long teacherId);

    /**
     * 设置成绩评语
     * 
     * @param gradeId 成绩ID
     * @param comment 评语
     * @param teacherId 教师ID
     * @return 操作结果
     */
    Boolean setGradeComment(Long gradeId, String comment, Long teacherId);

    /**
     * 获取成绩评语
     * 
     * @param gradeId 成绩ID
     * @param teacherId 教师ID
     * @return 评语
     */
    String getGradeComment(Long gradeId, Long teacherId);

    /**
     * 批量设置成绩评语
     * 
     * @param gradeComments 成绩评语列表
     * @param teacherId 教师ID
     * @return 操作结果
     */
    Boolean batchSetGradeComments(List<GradeDTO.GradeCommentRequest> gradeComments, Long teacherId);

    /**
     * 获取成绩对比分析
     * 
     * @param courseId1 课程ID1
     * @param courseId2 课程ID2
     * @param teacherId 教师ID
     * @return 对比分析结果
     */
    Object compareGrades(Long courseId1, Long courseId2, Long teacherId);

    /**
     * 生成成绩报告
     * 
     * @param reportRequest 报告请求
     * @param teacherId 教师ID
     * @return 报告文件路径
     */
    String generateGradeReport(GradeDTO.GradeReportRequest reportRequest, Long teacherId);

    /**
     * 设置成绩公开性
     * 
     * @param gradeIds 成绩ID列表
     * @param isPublic 是否公开
     * @param teacherId 教师ID
     * @return 操作结果
     */
    Boolean setGradeVisibility(List<Long> gradeIds, Boolean isPublic, Long teacherId);

    /**
     * 获取成绩预警列表
     * 
     * @param courseId 课程ID
     * @param teacherId 教师ID
     * @return 预警列表
     */
    List<Object> getGradeWarnings(Long courseId, Long teacherId);

    /**
     * 设置成绩预警规则
     * 
     * @param courseId 课程ID
     * @param warningRules 预警规则
     * @param teacherId 教师ID
     * @return 操作结果
     */
    Boolean setGradeWarningRules(Long courseId, Object warningRules, Long teacherId);

    /**
     * 归档成绩
     * 
     * @param courseId 课程ID
     * @param teacherId 教师ID
     * @return 操作结果
     */
    Boolean archiveGrades(Long courseId, Long teacherId);

    /**
     * 恢复归档成绩
     * 
     * @param courseId 课程ID
     * @param teacherId 教师ID
     * @return 操作结果
     */
    Boolean restoreGrades(Long courseId, Long teacherId);
}