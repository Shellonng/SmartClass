package com.education.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.education.entity.Task;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;
import org.apache.ibatis.annotations.Update;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

/**
 * 任务Mapper接口
 * 
 * @author Education Platform Team
 * @version 1.0.0
 */
@Mapper
public interface TaskMapper extends BaseMapper<Task> {

    /**
     * 根据课程ID查询任务列表
     * 
     * @param courseId 课程ID
     * @return 任务列表
     */
    @Select("SELECT * FROM task WHERE course_id = #{courseId} AND is_deleted = 0 ORDER BY create_time DESC")
    List<Task> selectByCourseId(@Param("courseId") Long courseId);

    /**
     * 根据教师ID查询任务列表
     * 
     * @param teacherId 教师ID
     * @return 任务列表
     */
    @Select("SELECT * FROM task WHERE teacher_id = #{teacherId} AND is_deleted = 0 ORDER BY create_time DESC")
    List<Task> selectByTeacherId(@Param("teacherId") Long teacherId);

    /**
     * 根据任务类型查询任务列表
     * 
     * @param taskType 任务类型
     * @return 任务列表
     */
    @Select("SELECT * FROM task WHERE task_type = #{taskType} AND is_deleted = 0 ORDER BY create_time DESC")
    List<Task> selectByTaskType(@Param("taskType") String taskType);

    /**
     * 根据任务状态查询任务列表
     * 
     * @param status 任务状态
     * @return 任务列表
     */
    @Select("SELECT * FROM task WHERE status = #{status} AND is_deleted = 0 ORDER BY create_time DESC")
    List<Task> selectByStatus(@Param("status") String status);

    /**
     * 根据优先级查询任务列表
     * 
     * @param priority 优先级
     * @return 任务列表
     */
    @Select("SELECT * FROM task WHERE priority = #{priority} AND is_deleted = 0 ORDER BY due_date ASC")
    List<Task> selectByPriority(@Param("priority") String priority);

    /**
     * 根据难度等级查询任务列表
     * 
     * @param difficulty 难度等级
     * @return 任务列表
     */
    @Select("SELECT * FROM task WHERE difficulty = #{difficulty} AND is_deleted = 0 ORDER BY create_time DESC")
    List<Task> selectByDifficulty(@Param("difficulty") String difficulty);

    /**
     * 查询已发布的任务
     * 
     * @return 任务列表
     */
    @Select("SELECT * FROM task WHERE status = 'PUBLISHED' AND is_deleted = 0 ORDER BY publish_date DESC")
    List<Task> selectPublishedTasks();

    /**
     * 查询进行中的任务
     * 
     * @return 任务列表
     */
    @Select("SELECT * FROM task " +
            "WHERE status = 'PUBLISHED' AND is_deleted = 0 " +
            "AND start_date <= NOW() AND due_date >= NOW() " +
            "ORDER BY due_date ASC")
    List<Task> selectActiveTasks();

    /**
     * 查询即将到期的任务
     * 
     * @param hours 提前几小时
     * @return 任务列表
     */
    @Select("SELECT * FROM task " +
            "WHERE status = 'PUBLISHED' AND is_deleted = 0 " +
            "AND due_date <= DATE_ADD(NOW(), INTERVAL #{hours} HOUR) " +
            "AND due_date >= NOW() " +
            "ORDER BY due_date ASC")
    List<Task> selectTasksDueSoon(@Param("hours") int hours);

    /**
     * 查询已过期的任务
     * 
     * @return 任务列表
     */
    @Select("SELECT * FROM task " +
            "WHERE status = 'PUBLISHED' AND is_deleted = 0 " +
            "AND due_date < NOW() " +
            "ORDER BY due_date DESC")
    List<Task> selectOverdueTasks();

    /**
     * 查询高优先级任务
     * 
     * @return 任务列表
     */
    @Select("SELECT * FROM task " +
            "WHERE priority = 'HIGH' AND status = 'PUBLISHED' AND is_deleted = 0 " +
            "ORDER BY due_date ASC")
    List<Task> selectHighPriorityTasks();

    /**
     * 统计任务总数
     * 
     * @return 任务总数
     */
    @Select("SELECT COUNT(*) FROM task WHERE is_deleted = 0")
    Long countTotalTasks();

    /**
     * 根据任务类型统计任务数
     * 
     * @return 类型统计结果
     */
    @Select("SELECT task_type, COUNT(*) as count FROM task WHERE is_deleted = 0 GROUP BY task_type ORDER BY count DESC")
    List<Map<String, Object>> countTasksByType();

    /**
     * 根据状态统计任务数
     * 
     * @return 状态统计结果
     */
    @Select("SELECT status, COUNT(*) as count FROM task WHERE is_deleted = 0 GROUP BY status")
    List<Map<String, Object>> countTasksByStatus();

    /**
     * 根据优先级统计任务数
     * 
     * @return 优先级统计结果
     */
    @Select("SELECT priority, COUNT(*) as count FROM task WHERE is_deleted = 0 GROUP BY priority")
    List<Map<String, Object>> countTasksByPriority();

    /**
     * 根据难度等级统计任务数
     * 
     * @return 难度统计结果
     */
    @Select("SELECT difficulty, COUNT(*) as count FROM task WHERE is_deleted = 0 GROUP BY difficulty")
    List<Map<String, Object>> countTasksByDifficulty();

    /**
     * 搜索任务（模糊查询）
     * 
     * @param keyword 关键词
     * @return 任务列表
     */
    @Select("SELECT t.*, c.course_name, u.real_name as teacher_name " +
            "FROM task t " +
            "LEFT JOIN course c ON t.course_id = c.id " +
            "LEFT JOIN user u ON t.teacher_id = u.id " +
            "WHERE t.is_deleted = 0 " +
            "AND (t.title LIKE CONCAT('%', #{keyword}, '%') " +
            "OR t.description LIKE CONCAT('%', #{keyword}, '%') " +
            "OR t.requirements LIKE CONCAT('%', #{keyword}, '%') " +
            "OR c.course_name LIKE CONCAT('%', #{keyword}, '%') " +
            "OR u.real_name LIKE CONCAT('%', #{keyword}, '%')) " +
            "ORDER BY t.create_time DESC")
    List<Map<String, Object>> searchTasks(@Param("keyword") String keyword);

    /**
     * 更新任务状态
     * 
     * @param taskId 任务ID
     * @param status 任务状态
     * @return 更新结果
     */
    @Update("UPDATE task SET status = #{status}, update_time = NOW() WHERE id = #{taskId} AND is_deleted = 0")
    int updateStatus(@Param("taskId") Long taskId, @Param("status") String status);

    /**
     * 批量更新任务状态
     * 
     * @param taskIds 任务ID列表
     * @param status 任务状态
     * @return 更新结果
     */
    int batchUpdateStatus(@Param("taskIds") List<Long> taskIds, @Param("status") String status);

    /**
     * 更新任务统计信息
     * 
     * @param taskId 任务ID
     * @param completedCount 完成人数
     * @param submittedCount 提交人数
     * @param averageScore 平均分
     * @return 更新结果
     */
    @Update("UPDATE task SET completed_count = #{completedCount}, submitted_count = #{submittedCount}, " +
            "average_score = #{averageScore}, update_time = NOW() " +
            "WHERE id = #{taskId} AND is_deleted = 0")
    int updateStatistics(@Param("taskId") Long taskId, 
                        @Param("completedCount") Integer completedCount, 
                        @Param("submittedCount") Integer submittedCount, 
                        @Param("averageScore") Double averageScore);

    /**
     * 增加任务提交人数
     * 
     * @param taskId 任务ID
     * @param increment 增加数量
     * @return 更新结果
     */
    @Update("UPDATE task SET submitted_count = submitted_count + #{increment}, update_time = NOW() WHERE id = #{taskId} AND is_deleted = 0")
    int incrementSubmittedCount(@Param("taskId") Long taskId, @Param("increment") Integer increment);

    /**
     * 增加任务完成人数
     * 
     * @param taskId 任务ID
     * @param increment 增加数量
     * @return 更新结果
     */
    @Update("UPDATE task SET completed_count = completed_count + #{increment}, update_time = NOW() WHERE id = #{taskId} AND is_deleted = 0")
    int incrementCompletedCount(@Param("taskId") Long taskId, @Param("increment") Integer increment);

    /**
     * 获取任务详细统计信息
     * 
     * @return 统计信息
     */
    @Select("SELECT " +
            "COUNT(*) as total_tasks, " +
            "SUM(CASE WHEN status = 'PUBLISHED' THEN 1 ELSE 0 END) as published_tasks, " +
            "SUM(CASE WHEN status = 'DRAFT' THEN 1 ELSE 0 END) as draft_tasks, " +
            "SUM(CASE WHEN status = 'COMPLETED' THEN 1 ELSE 0 END) as completed_tasks, " +
            "SUM(CASE WHEN status = 'CANCELLED' THEN 1 ELSE 0 END) as cancelled_tasks, " +
            "SUM(CASE WHEN due_date < NOW() AND status = 'PUBLISHED' THEN 1 ELSE 0 END) as overdue_tasks, " +
            "SUM(CASE WHEN priority = 'HIGH' THEN 1 ELSE 0 END) as high_priority_tasks, " +
            "SUM(submitted_count) as total_submissions, " +
            "SUM(completed_count) as total_completions, " +
            "AVG(average_score) as overall_average_score, " +
            "AVG(max_score) as average_max_score, " +
            "COUNT(DISTINCT course_id) as course_count, " +
            "COUNT(DISTINCT teacher_id) as teacher_count " +
            "FROM task WHERE is_deleted = 0")
    Map<String, Object> getTaskStatistics();

    /**
     * 获取任务的提交列表
     * 
     * @param taskId 任务ID
     * @return 提交列表
     */
    @Select("SELECT ts.*, s.student_number, u.real_name as student_name, u.email " +
            "FROM task_submission ts " +
            "JOIN student s ON ts.student_id = s.id " +
            "JOIN user u ON s.user_id = u.id " +
            "WHERE ts.task_id = #{taskId} AND ts.is_deleted = 0 " +
            "ORDER BY ts.submit_time DESC")
    List<Map<String, Object>> getTaskSubmissions(@Param("taskId") Long taskId);

    /**
     * 获取任务的成绩列表
     * 
     * @param taskId 任务ID
     * @return 成绩列表
     */
    @Select("SELECT g.*, s.student_number, u.real_name as student_name " +
            "FROM grade g " +
            "JOIN student s ON g.student_id = s.id " +
            "JOIN user u ON s.user_id = u.id " +
            "WHERE g.task_id = #{taskId} AND g.is_deleted = 0 " +
            "ORDER BY g.score DESC")
    List<Map<String, Object>> getTaskGrades(@Param("taskId") Long taskId);

    /**
     * 根据时间范围查询任务
     * 
     * @param startTime 开始时间
     * @param endTime 结束时间
     * @return 任务列表
     */
    @Select("SELECT * FROM task " +
            "WHERE is_deleted = 0 " +
            "AND ((start_date <= #{endTime} AND due_date >= #{startTime}) " +
            "OR (start_date >= #{startTime} AND start_date <= #{endTime})) " +
            "ORDER BY start_date ASC")
    List<Task> selectTasksByTimeRange(@Param("startTime") LocalDateTime startTime, @Param("endTime") LocalDateTime endTime);

    /**
     * 根据标签查询任务
     * 
     * @param tag 标签
     * @return 任务列表
     */
    @Select("SELECT * FROM task " +
            "WHERE is_deleted = 0 " +
            "AND FIND_IN_SET(#{tag}, tags) > 0 " +
            "ORDER BY create_time DESC")
    List<Task> selectTasksByTag(@Param("tag") String tag);

    /**
     * 获取学生的任务列表（通过选课关系）
     * 
     * @param studentId 学生ID
     * @return 任务列表
     */
    @Select("SELECT DISTINCT t.*, c.course_name " +
            "FROM task t " +
            "JOIN course c ON t.course_id = c.id " +
            "JOIN student_course sc ON c.id = sc.course_id " +
            "WHERE sc.student_id = #{studentId} AND t.status = 'PUBLISHED' " +
            "AND t.is_deleted = 0 AND sc.is_deleted = 0 " +
            "ORDER BY t.due_date ASC")
    List<Map<String, Object>> getStudentTasks(@Param("studentId") Long studentId);

    /**
     * 获取学生未完成的任务
     * 
     * @param studentId 学生ID
     * @return 任务列表
     */
    @Select("SELECT DISTINCT t.*, c.course_name " +
            "FROM task t " +
            "JOIN course c ON t.course_id = c.id " +
            "JOIN student_course sc ON c.id = sc.course_id " +
            "LEFT JOIN task_submission ts ON t.id = ts.task_id AND ts.student_id = #{studentId} AND ts.is_deleted = 0 " +
            "WHERE sc.student_id = #{studentId} AND t.status = 'PUBLISHED' " +
            "AND t.is_deleted = 0 AND sc.is_deleted = 0 " +
            "AND (ts.id IS NULL OR ts.is_final_submission = 0) " +
            "ORDER BY t.due_date ASC")
    List<Map<String, Object>> getStudentUncompletedTasks(@Param("studentId") Long studentId);

    /**
     * 获取学生已完成的任务
     * 
     * @param studentId 学生ID
     * @return 任务列表
     */
    @Select("SELECT DISTINCT t.*, c.course_name, ts.submit_time, ts.score " +
            "FROM task t " +
            "JOIN course c ON t.course_id = c.id " +
            "JOIN student_course sc ON c.id = sc.course_id " +
            "JOIN task_submission ts ON t.id = ts.task_id AND ts.student_id = #{studentId} " +
            "WHERE sc.student_id = #{studentId} AND t.status = 'PUBLISHED' " +
            "AND t.is_deleted = 0 AND sc.is_deleted = 0 AND ts.is_deleted = 0 " +
            "AND ts.is_final_submission = 1 " +
            "ORDER BY ts.submit_time DESC")
    List<Map<String, Object>> getStudentCompletedTasks(@Param("studentId") Long studentId);

    /**
     * 获取任务完成率统计
     * 
     * @return 完成率统计
     */
    @Select("SELECT " +
            "id, title, " +
            "submitted_count, completed_count, " +
            "(SELECT COUNT(*) FROM student_course sc JOIN course c ON sc.course_id = c.id WHERE c.id = task.course_id AND sc.is_deleted = 0) as total_students, " +
            "ROUND((submitted_count * 100.0 / NULLIF((SELECT COUNT(*) FROM student_course sc JOIN course c ON sc.course_id = c.id WHERE c.id = task.course_id AND sc.is_deleted = 0), 0)), 2) as submission_rate, " +
            "ROUND((completed_count * 100.0 / NULLIF((SELECT COUNT(*) FROM student_course sc JOIN course c ON sc.course_id = c.id WHERE c.id = task.course_id AND sc.is_deleted = 0), 0)), 2) as completion_rate " +
            "FROM task " +
            "WHERE is_deleted = 0 AND status = 'PUBLISHED' " +
            "ORDER BY completion_rate DESC")
    List<Map<String, Object>> getTaskCompletionRates();

    /**
     * 软删除任务
     * 
     * @param taskId 任务ID
     * @return 删除结果
     */
    @Update("UPDATE task SET is_deleted = 1, update_time = NOW() WHERE id = #{taskId}")
    int softDeleteTask(@Param("taskId") Long taskId);

    /**
     * 批量软删除任务
     * 
     * @param taskIds 任务ID列表
     * @return 删除结果
     */
    int batchSoftDeleteTasks(@Param("taskIds") List<Long> taskIds);

    /**
     * 恢复已删除的任务
     * 
     * @param taskId 任务ID
     * @return 恢复结果
     */
    @Update("UPDATE task SET is_deleted = 0, update_time = NOW() WHERE id = #{taskId}")
    int restoreTask(@Param("taskId") Long taskId);

    /**
     * 更新任务扩展字段
     * 
     * @param taskId 任务ID
     * @param extField1 扩展字段1
     * @param extField2 扩展字段2
     * @param extField3 扩展字段3
     * @return 更新结果
     */
    @Update("UPDATE task SET ext_field1 = #{extField1}, ext_field2 = #{extField2}, ext_field3 = #{extField3}, update_time = NOW() " +
            "WHERE id = #{taskId} AND is_deleted = 0")
    int updateExtFields(@Param("taskId") Long taskId, 
                       @Param("extField1") String extField1, 
                       @Param("extField2") String extField2, 
                       @Param("extField3") String extField3);

    /**
     * 获取任务进度信息
     * 
     * @param taskId 任务ID
     * @return 进度信息
     */
    @Select("SELECT " +
            "id, title, start_date, due_date, status, " +
            "DATEDIFF(NOW(), start_date) as running_days, " +
            "DATEDIFF(due_date, start_date) as total_days, " +
            "CASE " +
            "WHEN start_date > NOW() THEN 0 " +
            "WHEN due_date < NOW() THEN 100 " +
            "ELSE ROUND((DATEDIFF(NOW(), start_date) * 100.0 / NULLIF(DATEDIFF(due_date, start_date), 0)), 2) " +
            "END as progress_percentage " +
            "FROM task " +
            "WHERE id = #{taskId} AND is_deleted = 0")
    Map<String, Object> getTaskProgress(@Param("taskId") Long taskId);
}