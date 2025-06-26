package com.education.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.education.entity.TaskSubmission;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;
import org.apache.ibatis.annotations.Update;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

/**
 * 任务提交Mapper接口
 * 
 * @author Education Platform Team
 * @version 1.0.0
 */
@Mapper
public interface TaskSubmissionMapper extends BaseMapper<TaskSubmission> {

    /**
     * 根据任务ID查询提交列表
     * 
     * @param taskId 任务ID
     * @return 提交列表
     */
    @Select("SELECT * FROM task_submission WHERE task_id = #{taskId} AND is_deleted = 0 ORDER BY submit_time DESC")
    List<TaskSubmission> selectByTaskId(@Param("taskId") Long taskId);

    /**
     * 根据学生ID查询提交列表
     * 
     * @param studentId 学生ID
     * @return 提交列表
     */
    @Select("SELECT * FROM task_submission WHERE student_id = #{studentId} AND is_deleted = 0 ORDER BY submit_time DESC")
    List<TaskSubmission> selectByStudentId(@Param("studentId") Long studentId);

    /**
     * 根据任务ID和学生ID查询提交记录
     * 
     * @param taskId 任务ID
     * @param studentId 学生ID
     * @return 提交列表
     */
    @Select("SELECT * FROM task_submission WHERE task_id = #{taskId} AND student_id = #{studentId} AND is_deleted = 0 ORDER BY submit_time DESC")
    List<TaskSubmission> selectByTaskAndStudent(@Param("taskId") Long taskId, @Param("studentId") Long studentId);

    /**
     * 统计学生的提交总数
     * 
     * @param studentId 学生ID
     * @return 提交总数
     */
    @Select("SELECT COUNT(*) FROM task_submission WHERE student_id = #{studentId} AND is_deleted = 0")
    Integer countSubmissionsByStudent(@Param("studentId") Long studentId);

    /**
     * 统计学生已评分的提交数
     * 
     * @param studentId 学生ID
     * @return 已评分提交数
     */
    @Select("SELECT COUNT(*) FROM task_submission WHERE student_id = #{studentId} AND is_deleted = 0 AND score IS NOT NULL")
    Integer countGradedSubmissionsByStudent(@Param("studentId") Long studentId);

    /**
     * 计算学生的平均分
     * 
     * @param studentId 学生ID
     * @return 平均分
     */
    @Select("SELECT AVG(score) FROM task_submission WHERE student_id = #{studentId} AND is_deleted = 0 AND score IS NOT NULL")
    Double calculateAverageScoreByStudent(@Param("studentId") Long studentId);

    /**
     * 获取学生的最高分
     * 
     * @param studentId 学生ID
     * @return 最高分
     */
    @Select("SELECT MAX(score) FROM task_submission WHERE student_id = #{studentId} AND is_deleted = 0 AND score IS NOT NULL")
    Integer getHighestScoreByStudent(@Param("studentId") Long studentId);

    /**
     * 获取学生的最低分
     * 
     * @param studentId 学生ID
     * @return 最低分
     */
    @Select("SELECT MIN(score) FROM task_submission WHERE student_id = #{studentId} AND is_deleted = 0 AND score IS NOT NULL")
    Integer getLowestScoreByStudent(@Param("studentId") Long studentId);

    /**
     * 统计学生最近指定天数内的提交数
     * 
     * @param studentId 学生ID
     * @param days 天数
     * @return 最近提交数
     */
    @Select("SELECT COUNT(*) FROM task_submission WHERE student_id = #{studentId} AND is_deleted = 0 AND submit_time >= DATE_SUB(NOW(), INTERVAL #{days} DAY)")
    Integer countRecentSubmissionsByStudent(@Param("studentId") Long studentId, @Param("days") Integer days);

    /**
     * 根据任务ID和学生ID查询最新提交记录
     * 
     * @param taskId 任务ID
     * @param studentId 学生ID
     * @return 最新提交记录
     */
    @Select("SELECT * FROM task_submission WHERE task_id = #{taskId} AND student_id = #{studentId} AND is_deleted = 0 ORDER BY submit_time DESC LIMIT 1")
    TaskSubmission selectLatestByTaskAndStudent(@Param("taskId") Long taskId, @Param("studentId") Long studentId);

    /**
     * 根据任务ID和学生ID查询最终提交记录
     * 
     * @param taskId 任务ID
     * @param studentId 学生ID
     * @return 最终提交记录
     */
    @Select("SELECT * FROM task_submission WHERE task_id = #{taskId} AND student_id = #{studentId} AND is_final_submission = 1 AND is_deleted = 0")
    TaskSubmission selectFinalByTaskAndStudent(@Param("taskId") Long taskId, @Param("studentId") Long studentId);

    /**
     * 根据提交状态查询提交列表
     * 
     * @param status 提交状态
     * @return 提交列表
     */
    @Select("SELECT * FROM task_submission WHERE status = #{status} AND is_deleted = 0 ORDER BY submit_time DESC")
    List<TaskSubmission> selectByStatus(@Param("status") String status);

    /**
     * 根据评分教师ID查询提交列表
     * 
     * @param teacherId 评分教师ID
     * @return 提交列表
     */
    @Select("SELECT * FROM task_submission WHERE graded_by = #{teacherId} AND is_deleted = 0 ORDER BY submit_time DESC")
    List<TaskSubmission> selectByGradedTeacher(@Param("teacherId") Long teacherId);

    /**
     * 查询待评分的提交
     * 
     * @return 提交列表
     */
    @Select("SELECT * FROM task_submission WHERE status = 'SUBMITTED' AND score IS NULL AND is_deleted = 0 ORDER BY submit_time ASC")
    List<TaskSubmission> selectPendingGrading();

    /**
     * 查询已评分的提交
     * 
     * @return 提交列表
     */
    @Select("SELECT * FROM task_submission WHERE status = 'GRADED' AND score IS NOT NULL AND is_deleted = 0 ORDER BY submit_time DESC")
    List<TaskSubmission> selectGraded();

    /**
     * 查询迟交的提交
     * 
     * @return 提交列表
     */
    @Select("SELECT * FROM task_submission WHERE is_late = 1 AND is_deleted = 0 ORDER BY submit_time DESC")
    List<TaskSubmission> selectLateSubmissions();

    /**
     * 查询最终提交
     * 
     * @return 提交列表
     */
    @Select("SELECT * FROM task_submission WHERE is_final_submission = 1 AND is_deleted = 0 ORDER BY submit_time DESC")
    List<TaskSubmission> selectFinalSubmissions();

    /**
     * 统计提交总数
     * 
     * @return 提交总数
     */
    @Select("SELECT COUNT(*) FROM task_submission WHERE is_deleted = 0")
    Long countTotalSubmissions();

    /**
     * 根据状态统计提交数
     * 
     * @return 状态统计结果
     */
    @Select("SELECT status, COUNT(*) as count FROM task_submission WHERE is_deleted = 0 GROUP BY status")
    List<Map<String, Object>> countSubmissionsByStatus();

    /**
     * 根据任务统计提交数
     * 
     * @return 任务统计结果
     */
    @Select("SELECT t.title as task_title, COUNT(ts.id) as submission_count " +
            "FROM task_submission ts " +
            "JOIN task t ON ts.task_id = t.id " +
            "WHERE ts.is_deleted = 0 " +
            "GROUP BY ts.task_id, t.title " +
            "ORDER BY submission_count DESC")
    List<Map<String, Object>> countSubmissionsByTask();

    /**
     * 搜索提交（模糊查询）
     * 
     * @param keyword 关键词
     * @return 提交列表
     */
    @Select("SELECT ts.*, t.title as task_title, s.student_number, u.real_name as student_name " +
            "FROM task_submission ts " +
            "JOIN task t ON ts.task_id = t.id " +
            "JOIN student s ON ts.student_id = s.id " +
            "JOIN user u ON s.user_id = u.id " +
            "WHERE ts.is_deleted = 0 " +
            "AND (ts.content LIKE CONCAT('%', #{keyword}, '%') " +
            "OR ts.teacher_comment LIKE CONCAT('%', #{keyword}, '%') " +
            "OR t.title LIKE CONCAT('%', #{keyword}, '%') " +
            "OR s.student_number LIKE CONCAT('%', #{keyword}, '%') " +
            "OR u.real_name LIKE CONCAT('%', #{keyword}, '%')) " +
            "ORDER BY ts.submit_time DESC")
    List<Map<String, Object>> searchSubmissions(@Param("keyword") String keyword);

    /**
     * 更新提交状态
     * 
     * @param submissionId 提交ID
     * @param status 提交状态
     * @return 更新结果
     */
    @Update("UPDATE task_submission SET status = #{status}, update_time = NOW() WHERE id = #{submissionId} AND is_deleted = 0")
    int updateStatus(@Param("submissionId") Long submissionId, @Param("status") String status);

    /**
     * 更新提交评分信息
     * 
     * @param submissionId 提交ID
     * @param score 分数
     * @param gradedBy 评分教师ID
     * @param teacherComment 教师评语
     * @return 更新结果
     */
    @Update("UPDATE task_submission SET score = #{score}, graded_by = #{gradedBy}, teacher_comment = #{teacherComment}, " +
            "status = 'GRADED', update_time = NOW() " +
            "WHERE id = #{submissionId} AND is_deleted = 0")
    int updateGrading(@Param("submissionId") Long submissionId, 
                     @Param("score") Double score, 
                     @Param("gradedBy") Long gradedBy, 
                     @Param("teacherComment") String teacherComment);

    /**
     * 标记为最终提交
     * 
     * @param submissionId 提交ID
     * @return 更新结果
     */
    @Update("UPDATE task_submission SET is_final_submission = 1, update_time = NOW() WHERE id = #{submissionId} AND is_deleted = 0")
    int markAsFinal(@Param("submissionId") Long submissionId);

    /**
     * 取消最终提交标记（同一任务同一学生的其他提交）
     * 
     * @param taskId 任务ID
     * @param studentId 学生ID
     * @param excludeSubmissionId 排除的提交ID
     * @return 更新结果
     */
    @Update("UPDATE task_submission SET is_final_submission = 0, update_time = NOW() " +
            "WHERE task_id = #{taskId} AND student_id = #{studentId} " +
            "AND id != #{excludeSubmissionId} AND is_deleted = 0")
    int unmarkOtherFinalSubmissions(@Param("taskId") Long taskId, 
                                   @Param("studentId") Long studentId, 
                                   @Param("excludeSubmissionId") Long excludeSubmissionId);

    /**
     * 批量更新提交状态
     * 
     * @param submissionIds 提交ID列表
     * @param status 提交状态
     * @return 更新结果
     */
    int batchUpdateStatus(@Param("submissionIds") List<Long> submissionIds, @Param("status") String status);

    /**
     * 获取提交详细统计信息
     * 
     * @return 统计信息
     */
    @Select("SELECT " +
            "COUNT(*) as total_submissions, " +
            "SUM(CASE WHEN status = 'SUBMITTED' THEN 1 ELSE 0 END) as submitted_count, " +
            "SUM(CASE WHEN status = 'GRADED' THEN 1 ELSE 0 END) as graded_count, " +
            "SUM(CASE WHEN status = 'RETURNED' THEN 1 ELSE 0 END) as returned_count, " +
            "SUM(CASE WHEN is_late = 1 THEN 1 ELSE 0 END) as late_submissions, " +
            "SUM(CASE WHEN is_final_submission = 1 THEN 1 ELSE 0 END) as final_submissions, " +
            "AVG(score) as average_score, " +
            "MAX(score) as highest_score, " +
            "MIN(score) as lowest_score, " +
            "AVG(submission_count) as average_submission_count, " +
            "MAX(submission_count) as max_submission_count, " +
            "COUNT(DISTINCT task_id) as task_count, " +
            "COUNT(DISTINCT student_id) as student_count " +
            "FROM task_submission WHERE is_deleted = 0")
    Map<String, Object> getSubmissionStatistics();

    /**
     * 获取任务的提交统计
     * 
     * @param taskId 任务ID
     * @return 统计信息
     */
    @Select("SELECT " +
            "COUNT(*) as total_submissions, " +
            "COUNT(DISTINCT student_id) as unique_students, " +
            "SUM(CASE WHEN status = 'GRADED' THEN 1 ELSE 0 END) as graded_count, " +
            "SUM(CASE WHEN is_late = 1 THEN 1 ELSE 0 END) as late_count, " +
            "SUM(CASE WHEN is_final_submission = 1 THEN 1 ELSE 0 END) as final_count, " +
            "AVG(score) as average_score, " +
            "MAX(score) as highest_score, " +
            "MIN(score) as lowest_score " +
            "FROM task_submission " +
            "WHERE task_id = #{taskId} AND is_deleted = 0")
    Map<String, Object> getTaskSubmissionStatistics(@Param("taskId") Long taskId);

    /**
     * 获取学生的提交统计
     * 
     * @param studentId 学生ID
     * @return 统计信息
     */
    @Select("SELECT " +
            "COUNT(*) as total_submissions, " +
            "COUNT(DISTINCT task_id) as unique_tasks, " +
            "SUM(CASE WHEN status = 'GRADED' THEN 1 ELSE 0 END) as graded_count, " +
            "SUM(CASE WHEN is_late = 1 THEN 1 ELSE 0 END) as late_count, " +
            "SUM(CASE WHEN is_final_submission = 1 THEN 1 ELSE 0 END) as final_count, " +
            "AVG(score) as average_score, " +
            "MAX(score) as highest_score, " +
            "MIN(score) as lowest_score " +
            "FROM task_submission " +
            "WHERE student_id = #{studentId} AND is_deleted = 0")
    Map<String, Object> getStudentSubmissionStatistics(@Param("studentId") Long studentId);

    /**
     * 根据时间范围查询提交
     * 
     * @param startTime 开始时间
     * @param endTime 结束时间
     * @return 提交列表
     */
    @Select("SELECT * FROM task_submission " +
            "WHERE is_deleted = 0 " +
            "AND submit_time >= #{startTime} AND submit_time <= #{endTime} " +
            "ORDER BY submit_time DESC")
    List<TaskSubmission> selectSubmissionsByTimeRange(@Param("startTime") LocalDateTime startTime, @Param("endTime") LocalDateTime endTime);

    /**
     * 获取提交趋势（按天）
     * 
     * @param days 天数
     * @return 提交趋势
     */
    @Select("SELECT DATE(submit_time) as date, COUNT(*) as count " +
            "FROM task_submission " +
            "WHERE is_deleted = 0 AND submit_time >= DATE_SUB(NOW(), INTERVAL #{days} DAY) " +
            "GROUP BY DATE(submit_time) " +
            "ORDER BY date")
    List<Map<String, Object>> getSubmissionTrend(@Param("days") int days);

    /**
     * 获取分数分布统计
     * 
     * @return 分数分布
     */
    @Select("SELECT " +
            "CASE " +
            "WHEN score >= 90 THEN 'A (90-100)' " +
            "WHEN score >= 80 THEN 'B (80-89)' " +
            "WHEN score >= 70 THEN 'C (70-79)' " +
            "WHEN score >= 60 THEN 'D (60-69)' " +
            "ELSE 'F (0-59)' " +
            "END as score_range, " +
            "COUNT(*) as count " +
            "FROM task_submission " +
            "WHERE is_deleted = 0 AND score IS NOT NULL " +
            "GROUP BY " +
            "CASE " +
            "WHEN score >= 90 THEN 'A (90-100)' " +
            "WHEN score >= 80 THEN 'B (80-89)' " +
            "WHEN score >= 70 THEN 'C (70-79)' " +
            "WHEN score >= 60 THEN 'D (60-69)' " +
            "ELSE 'F (0-59)' " +
            "END " +
            "ORDER BY MIN(score) DESC")
    List<Map<String, Object>> getScoreDistribution();

    /**
     * 获取查重结果统计
     * 
     * @return 查重统计
     */
    @Select("SELECT " +
            "COUNT(*) as total_checked, " +
            "SUM(CASE WHEN similarity_percentage > 80 THEN 1 ELSE 0 END) as high_similarity, " +
            "SUM(CASE WHEN similarity_percentage BETWEEN 50 AND 80 THEN 1 ELSE 0 END) as medium_similarity, " +
            "SUM(CASE WHEN similarity_percentage < 50 THEN 1 ELSE 0 END) as low_similarity, " +
            "AVG(similarity_percentage) as average_similarity " +
            "FROM task_submission " +
            "WHERE is_deleted = 0 AND plagiarism_result IS NOT NULL")
    Map<String, Object> getPlagiarismStatistics();

    /**
     * 根据相似度查询提交
     * 
     * @param minSimilarity 最小相似度
     * @param maxSimilarity 最大相似度
     * @return 提交列表
     */
    @Select("SELECT * FROM task_submission " +
            "WHERE is_deleted = 0 " +
            "AND similarity_percentage >= #{minSimilarity} " +
            "AND similarity_percentage <= #{maxSimilarity} " +
            "ORDER BY similarity_percentage DESC")
    List<TaskSubmission> selectSubmissionsBySimilarity(@Param("minSimilarity") Double minSimilarity, @Param("maxSimilarity") Double maxSimilarity);

    /**
     * 软删除提交
     * 
     * @param submissionId 提交ID
     * @return 删除结果
     */
    @Update("UPDATE task_submission SET is_deleted = 1, update_time = NOW() WHERE id = #{submissionId}")
    int softDeleteSubmission(@Param("submissionId") Long submissionId);

    /**
     * 批量软删除提交
     * 
     * @param submissionIds 提交ID列表
     * @return 删除结果
     */
    int batchSoftDeleteSubmissions(@Param("submissionIds") List<Long> submissionIds);

    /**
     * 恢复已删除的提交
     * 
     * @param submissionId 提交ID
     * @return 恢复结果
     */
    @Update("UPDATE task_submission SET is_deleted = 0, update_time = NOW() WHERE id = #{submissionId}")
    int restoreSubmission(@Param("submissionId") Long submissionId);

    /**
     * 更新提交扩展字段
     * 
     * @param submissionId 提交ID
     * @param extField1 扩展字段1
     * @param extField2 扩展字段2
     * @param extField3 扩展字段3
     * @return 更新结果
     */
    @Update("UPDATE task_submission SET ext_field1 = #{extField1}, ext_field2 = #{extField2}, ext_field3 = #{extField3}, update_time = NOW() " +
            "WHERE id = #{submissionId} AND is_deleted = 0")
    int updateExtFields(@Param("submissionId") Long submissionId, 
                       @Param("extField1") String extField1, 
                       @Param("extField2") String extField2, 
                       @Param("extField3") String extField3);

    /**
     * 统计学生提交次数
     * 
     * @param taskId 任务ID
     * @param studentId 学生ID
     * @return 提交次数
     */
    @Select("SELECT COUNT(*) FROM task_submission WHERE task_id = #{taskId} AND student_id = #{studentId} AND is_deleted = 0")
    int countSubmissionsByTaskAndStudent(@Param("taskId") Long taskId, @Param("studentId") Long studentId);

    /**
     * 根据任务ID统计提交总数
     * 
     * @param taskId 任务ID
     * @return 提交总数
     */
    @Select("SELECT COUNT(*) FROM task_submission WHERE task_id = #{taskId} AND is_deleted = 0")
    Integer countByTaskId(@Param("taskId") Long taskId);

    /**
     * 根据任务ID统计已评分提交数
     * 
     * @param taskId 任务ID
     * @return 已评分提交数
     */
    @Select("SELECT COUNT(*) FROM task_submission WHERE task_id = #{taskId} AND status = 'GRADED' AND is_deleted = 0")
    Integer countGradedByTaskId(@Param("taskId") Long taskId);

    /**
     * 根据任务ID统计迟交提交数
     * 
     * @param taskId 任务ID
     * @return 迟交提交数
     */
    @Select("SELECT COUNT(*) FROM task_submission WHERE task_id = #{taskId} AND is_late = 1 AND is_deleted = 0")
    Integer countLateByTaskId(@Param("taskId") Long taskId);

    /**
     * 根据任务ID获取平均分
     * 
     * @param taskId 任务ID
     * @return 平均分
     */
    @Select("SELECT AVG(score) FROM task_submission WHERE task_id = #{taskId} AND score IS NOT NULL AND is_deleted = 0")
    Double getAverageScoreByTaskId(@Param("taskId") Long taskId);

    /**
     * 根据任务ID获取最高分
     * 
     * @param taskId 任务ID
     * @return 最高分
     */
    @Select("SELECT MAX(score) FROM task_submission WHERE task_id = #{taskId} AND score IS NOT NULL AND is_deleted = 0")
    Double getMaxScoreByTaskId(@Param("taskId") Long taskId);

    /**
     * 根据任务ID获取最低分
     * 
     * @param taskId 任务ID
     * @return 最低分
     */
    @Select("SELECT MIN(score) FROM task_submission WHERE task_id = #{taskId} AND score IS NOT NULL AND is_deleted = 0")
    Double getMinScoreByTaskId(@Param("taskId") Long taskId);


    /**
     * 检查是否可以重新提交
     * 
     * @param taskId 任务ID
     * @param studentId 学生ID
     * @return 是否可以重新提交
     */
    @Select("SELECT " +
            "CASE " +
            "WHEN t.max_submissions IS NULL THEN 1 " +
            "WHEN (SELECT COUNT(*) FROM task_submission ts WHERE ts.task_id = #{taskId} AND ts.student_id = #{studentId} AND ts.is_deleted = 0) < t.max_submissions THEN 1 " +
            "ELSE 0 " +
            "END as can_resubmit " +
            "FROM task t " +
            "WHERE t.id = #{taskId} AND t.is_deleted = 0")
    int checkCanResubmit(@Param("taskId") Long taskId, @Param("studentId") Long studentId);
}