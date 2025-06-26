package com.education.service.teacher;

import com.education.dto.StudentDTO;
import com.education.dto.StudentDTOExtension;
import com.education.dto.common.PageRequest;
import com.education.dto.common.PageResponse;

import java.util.List;

/**
 * 教师端学生服务接口
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
public interface StudentService {

    /**
     * 获取学生列表
     * 
     * @param teacherId 教师ID
     * @param pageRequest 分页请求
     * @return 学生列表
     */
    PageResponse<StudentDTO.StudentResponse> getStudentList(Long teacherId, PageRequest pageRequest);

    /**
     * 获取学生详情
     * 
     * @param studentId 学生ID
     * @param teacherId 教师ID
     * @return 学生详情
     */
    StudentDTO.StudentDetailResponse getStudentDetail(Long studentId, Long teacherId);

    /**
     * 获取学生学习进度
     * 
     * @param studentId 学生ID
     * @param teacherId 教师ID
     * @param courseId 课程ID（可选）
     * @return 学习进度
     */
    StudentDTO.StudentProgressResponse getStudentProgress(Long studentId, Long teacherId, Long courseId);

    /**
     * 获取学生成绩统计
     * 
     * @param studentId 学生ID
     * @param teacherId 教师ID
     * @return 成绩统计
     */
    StudentDTO.StudentGradeStatisticsResponse getStudentGradeStatistics(Long studentId, Long teacherId);

    /**
     * 获取学生任务提交记录
     * 
     * @param studentId 学生ID
     * @param teacherId 教师ID
     * @param pageRequest 分页请求
     * @return 提交记录列表
     */
    PageResponse<StudentDTO.StudentSubmissionResponse> getStudentSubmissions(Long studentId, Long teacherId, PageRequest pageRequest);

    /**
     * 获取学生学习分析
     * 
     * @param studentId 学生ID
     * @param teacherId 教师ID
     * @param timeRange 时间范围
     * @return 学习分析报告
     */
    StudentDTO.StudentAnalysisResponse getStudentAnalysis(Long studentId, Long teacherId, String timeRange);

    /**
     * 批量导入学生
     * 
     * @param importRequest 导入请求
     * @param teacherId 教师ID
     * @return 导入结果
     */
    StudentDTO.StudentImportResponse importStudents(StudentDTO.StudentImportRequest importRequest, Long teacherId);

    /**
     * 导出学生信息
     * 
     * @param exportRequest 导出请求
     * @param teacherId 教师ID
     * @return 导出文件路径
     */
    String exportStudents(StudentDTOExtension.StudentExportRequest exportRequest, Long teacherId);

    /**
     * 重置学生密码
     * 
     * @param studentId 学生ID
     * @param teacherId 教师ID
     * @return 新密码
     */
    String resetStudentPassword(Long studentId, Long teacherId);

    /**
     * 获取学生排行榜
     * 
     * @param teacherId 教师ID
     * @param rankType 排行类型
     * @param classId 班级ID（可选）
     * @param limit 限制数量
     * @return 排行榜
     */
    List<StudentDTOExtension.StudentRankResponse> getStudentRanking(Long teacherId, String rankType, Long classId, Integer limit);

    /**
     * 搜索学生
     * 
     * @param keyword 关键词
     * @param teacherId 教师ID
     * @param pageRequest 分页请求
     * @return 搜索结果
     */
    PageResponse<StudentDTO.StudentResponse> searchStudents(String keyword, Long teacherId, PageRequest pageRequest);

    /**
     * 获取学生课程列表
     * 
     * @param studentId 学生ID
     * @param teacherId 教师ID
     * @return 课程列表
     */
    List<Object> getStudentCourses(Long studentId, Long teacherId);

    /**
     * 获取学生班级信息
     * 
     * @param studentId 学生ID
     * @param teacherId 教师ID
     * @return 班级信息
     */
    List<Object> getStudentClasses(Long studentId, Long teacherId);

    /**
     * 设置学生状态
     * 
     * @param studentId 学生ID
     * @param status 状态
     * @param teacherId 教师ID
     * @return 操作结果
     */
    Boolean setStudentStatus(Long studentId, String status, Long teacherId);

    /**
     * 获取学生学习时长统计
     * 
     * @param studentId 学生ID
     * @param teacherId 教师ID
     * @param timeRange 时间范围
     * @return 学习时长统计
     */
    Object getStudentLearningTime(Long studentId, Long teacherId, String timeRange);

    /**
     * 获取学生错题统计
     * 
     * @param studentId 学生ID
     * @param teacherId 教师ID
     * @return 错题统计
     */
    Object getStudentWrongQuestions(Long studentId, Long teacherId);

    /**
     * 获取学生学习轨迹
     * 
     * @param studentId 学生ID
     * @param teacherId 教师ID
     * @param pageRequest 分页请求
     * @return 学习轨迹
     */
    PageResponse<Object> getStudentLearningTrack(Long studentId, Long teacherId, PageRequest pageRequest);

    /**
     * 为学生添加标签
     * 
     * @param studentId 学生ID
     * @param tags 标签列表
     * @param teacherId 教师ID
     * @return 操作结果
     */
    Boolean addStudentTags(Long studentId, List<String> tags, Long teacherId);

    /**
     * 移除学生标签
     * 
     * @param studentId 学生ID
     * @param tags 标签列表
     * @param teacherId 教师ID
     * @return 操作结果
     */
    Boolean removeStudentTags(Long studentId, List<String> tags, Long teacherId);

    /**
     * 获取学生标签
     * 
     * @param studentId 学生ID
     * @param teacherId 教师ID
     * @return 标签列表
     */
    List<String> getStudentTags(Long studentId, Long teacherId);

    /**
     * 发送消息给学生
     * 
     * @param studentId 学生ID
     * @param message 消息内容
     * @param teacherId 教师ID
     * @return 操作结果
     */
    Boolean sendMessageToStudent(Long studentId, String message, Long teacherId);

    /**
     * 获取学生消息记录
     * 
     * @param studentId 学生ID
     * @param teacherId 教师ID
     * @param pageRequest 分页请求
     * @return 消息记录
     */
    PageResponse<Object> getStudentMessages(Long studentId, Long teacherId, PageRequest pageRequest);

    /**
     * 获取学生出勤统计
     * 
     * @param studentId 学生ID
     * @param teacherId 教师ID
     * @param timeRange 时间范围
     * @return 出勤统计
     */
    Object getStudentAttendance(Long studentId, Long teacherId, String timeRange);

    /**
     * 记录学生出勤
     * 
     * @param studentId 学生ID
     * @param attendanceType 出勤类型
     * @param courseId 课程ID
     * @param teacherId 教师ID
     * @return 操作结果
     */
    Boolean recordStudentAttendance(Long studentId, String attendanceType, Long courseId, Long teacherId);

    /**
     * 获取学生家长联系信息
     * 
     * @param studentId 学生ID
     * @param teacherId 教师ID
     * @return 家长联系信息
     */
    Object getStudentParentContact(Long studentId, Long teacherId);

    /**
     * 生成学生学习报告
     * 
     * @param studentId 学生ID
     * @param teacherId 教师ID
     * @param reportType 报告类型
     * @param timeRange 时间范围
     * @return 报告文件路径
     */
    String generateStudentReport(Long studentId, Long teacherId, String reportType, String timeRange);

    /**
     * 获取学生能力评估
     * 
     * @param studentId 学生ID
     * @param teacherId 教师ID
     * @return 能力评估结果
     */
    Object getStudentAbilityAssessment(Long studentId, Long teacherId);

    /**
     * 更新学生能力评估
     * 
     * @param studentId 学生ID
     * @param assessment 评估数据
     * @param teacherId 教师ID
     * @return 操作结果
     */
    Boolean updateStudentAbilityAssessment(Long studentId, Object assessment, Long teacherId);

    Object getStudentRanking(Long teacherId, Object rankingParams);
}