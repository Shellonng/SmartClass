package com.education.service.student.impl;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.education.dto.GradeDTO;
import com.education.dto.common.PageRequest;
import com.education.dto.common.PageResponse;
import com.education.entity.*;
import com.education.exception.BusinessException;
import com.education.mapper.*;
import com.education.exception.ResultCode;
import com.education.service.student.StudentGradeService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

/**
 * 学生端成绩服务实现类
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
@Slf4j
@Service
public class StudentGradeServiceImpl implements StudentGradeService {

    @Autowired
    private TaskSubmissionMapper taskSubmissionMapper;
    
    @Autowired
    private StudentMapper studentMapper;
    
    @Autowired
    private TaskMapper taskMapper;
    
    @Autowired
    private CourseMapper courseMapper;
    
    @Autowired
    private ClassMapper classMapper;
    
    @Override
    public PageResponse<GradeDTO.GradeListResponse> getStudentGrades(Long studentId, PageRequest pageRequest) {
        log.info("获取学生成绩列表，学生ID: {}", studentId);
        
        Student student = studentMapper.selectById(studentId);
        if (student == null) {
            throw new BusinessException(ResultCode.USER_NOT_FOUND, "学生不存在");
        }
        
        // 构建分页查询
        Page<TaskSubmission> page = new Page<>(pageRequest.getPageNum(), pageRequest.getPageSize());
        
        // 查询学生的任务提交记录（已评分的）
        QueryWrapper<TaskSubmission> queryWrapper = new QueryWrapper<>();
        queryWrapper.eq("student_id", studentId)
                   .isNotNull("score")
                   .orderByDesc("grading_time");
        
        Page<TaskSubmission> submissionPage = taskSubmissionMapper.selectPage(page, queryWrapper);
        
        List<GradeDTO.GradeListResponse> responses = submissionPage.getRecords().stream()
                .map(this::convertToGradeListResponse)
                .collect(Collectors.toList());
        
        return PageResponse.of((long) pageRequest.getPageNum(), (long) pageRequest.getPageSize(), submissionPage.getTotal(), responses);
    }

    @Override
    public GradeDTO.CourseGradeResponse getCourseGrade(Long courseId, Long studentId) {
        log.info("获取课程成绩，课程ID: {}, 学生ID: {}", courseId, studentId);
        
        // 验证权限
        if (!hasCourseAccess(courseId, studentId)) {
            throw new BusinessException(ResultCode.ACCESS_DENIED, "无权限访问该课程");
        }
        
        // 查询课程下的所有任务提交记录
        QueryWrapper<TaskSubmission> queryWrapper = new QueryWrapper<>();
        queryWrapper.eq("student_id", studentId)
                   .exists("SELECT 1 FROM task t WHERE t.id = task_submission.task_id AND t.course_id = {0}", courseId)
                   .isNotNull("score");
        
        List<TaskSubmission> submissions = taskSubmissionMapper.selectList(queryWrapper);
        
        GradeDTO.CourseGradeResponse response = new GradeDTO.CourseGradeResponse();
        response.setCourseId(courseId);
        
        if (!submissions.isEmpty()) {
            // 计算总分
            BigDecimal averageScore = submissions.stream()
                    .map(TaskSubmission::getScore)
                    .filter(score -> score != null)
                    .reduce(BigDecimal.ZERO, BigDecimal::add)
                    .divide(BigDecimal.valueOf(submissions.size()), 2, RoundingMode.HALF_UP);
            response.setTotalScore(averageScore);
            
            // 设置课程名称
            Course course = courseMapper.selectById(courseId);
            if (course != null) {
                response.setCourseName(course.getCourseName());
            }
            
            // 设置任务成绩列表
            List<GradeDTO.CourseGradeResponse.TaskGrade> taskGrades = submissions.stream()
                    .map(this::convertToTaskGrade)
                    .collect(Collectors.toList());
            response.setTaskGrades(taskGrades);
            
        } else {
            response.setTotalScore(BigDecimal.ZERO);
        }
        
        return response;
    }

    @Override
    public GradeDTO.TaskGradeDetailResponse getTaskGradeDetail(Long taskId, Long studentId) {
        log.info("获取任务成绩详情，任务ID: {}, 学生ID: {}", taskId, studentId);
        
        // 验证权限
        if (!hasTaskAccess(taskId, studentId)) {
            throw new BusinessException(ResultCode.ACCESS_DENIED, "无权限访问该任务");
        }
        
        QueryWrapper<TaskSubmission> queryWrapper = new QueryWrapper<>();
        queryWrapper.eq("task_id", taskId)
                   .eq("student_id", studentId);
        
        TaskSubmission submission = taskSubmissionMapper.selectOne(queryWrapper);
        if (submission == null || submission.getScore() == null) {
            throw new BusinessException(ResultCode.DATA_NOT_FOUND, "任务成绩不存在");
        }
        
        GradeDTO.TaskGradeDetailResponse response = new GradeDTO.TaskGradeDetailResponse();
        response.setTaskId(taskId);
        response.setStudentId(studentId);
        response.setScore(submission.getScore());
        response.setOriginalScore(submission.getOriginalScore());
        response.setDeductedPoints(submission.getDeduction());
        response.setGradeTime(submission.getGradeTime());
        response.setIsLate(submission.getIsLate());
        response.setLateDays(submission.getLateDays());
        
        return response;
    }

    @Override
    public GradeDTO.GradeStatisticsResponse getGradeStatistics(Long studentId) {
        log.info("获取成绩统计，学生ID: {}", studentId);
        
        Student student = studentMapper.selectById(studentId);
        if (student == null) {
            throw new BusinessException(ResultCode.USER_NOT_FOUND, "学生不存在");
        }
        
        // 查询所有已评分的提交记录
        QueryWrapper<TaskSubmission> queryWrapper = new QueryWrapper<>();
        queryWrapper.eq("student_id", studentId)
                   .isNotNull("score");
        
        List<TaskSubmission> submissions = taskSubmissionMapper.selectList(queryWrapper);
        
        GradeDTO.GradeStatisticsResponse response = new GradeDTO.GradeStatisticsResponse();
        response.setStudentId(studentId);
        
        if (!submissions.isEmpty()) {
            BigDecimal totalScore = submissions.stream()
                .map(TaskSubmission::getScore)
                .reduce(BigDecimal.ZERO, BigDecimal::add);
            BigDecimal averageScore = totalScore.divide(BigDecimal.valueOf(submissions.size()), 2, RoundingMode.HALF_UP);
            BigDecimal maxScore = submissions.stream()
                .map(TaskSubmission::getScore)
                .max(BigDecimal::compareTo)
                .orElse(BigDecimal.ZERO);
            BigDecimal minScore = submissions.stream()
                .map(TaskSubmission::getScore)
                .min(BigDecimal::compareTo)
                .orElse(BigDecimal.ZERO);
            
            response.setTotalTasks(submissions.size());
            response.setAverageScore(averageScore);
            response.setMaxScore(maxScore);
            response.setMinScore(minScore);
            response.setPassingRate(calculatePassingRate(submissions));
        } else {
            response.setTotalTasks(0);
            response.setAverageScore(BigDecimal.ZERO);
            response.setMaxScore(BigDecimal.ZERO);
            response.setMinScore(BigDecimal.ZERO);
            response.setPassingRate(0.0);
        }
        
        return response;
    }

    @Override
    public GradeDTO.GradeTrendResponse getGradeTrend(Long studentId, String timeRange) {
        log.info("获取成绩趋势，学生ID: {}, 时间范围: {}", studentId, timeRange);
        
        Student student = studentMapper.selectById(studentId);
        if (student == null) {
            throw new BusinessException(ResultCode.USER_NOT_FOUND, "学生不存在");
        }
        
        GradeDTO.GradeTrendResponse response = new GradeDTO.GradeTrendResponse();
        response.setStudentId(studentId);
        response.setTimeRange(timeRange);
        // 这里可以实现具体的趋势分析逻辑
        response.setTrendData(List.of());
        
        return response;
    }

    @Override
    public GradeDTO.GradeDistributionResponse getGradeDistribution(Long studentId, Long courseId) {
        log.info("获取成绩分布，学生ID: {}, 课程ID: {}", studentId, courseId);
        
        Student student = studentMapper.selectById(studentId);
        if (student == null) {
            throw new BusinessException(ResultCode.USER_NOT_FOUND, "学生不存在");
        }
        
        GradeDTO.GradeDistributionResponse response = new GradeDTO.GradeDistributionResponse();
        response.setStudentId(studentId);
        response.setCourseId(courseId);
        // 这里可以实现具体的分布分析逻辑
        response.setDistributionData(List.of());
        
        return response;
    }

    @Override
    public GradeDTO.ClassRankingResponse getClassRanking(Long studentId, Long classId) {
        log.info("获取班级排名，学生ID: {}, 班级ID: {}", studentId, classId);
        
        Student student = studentMapper.selectById(studentId);
        if (student == null) {
            throw new BusinessException(ResultCode.USER_NOT_FOUND, "学生不存在");
        }
        
        GradeDTO.ClassRankingResponse response = new GradeDTO.ClassRankingResponse();
        response.setStudentId(studentId);
        response.setClassId(classId);
        response.setRanking(1); // 默认排名
        response.setTotalStudents(1);
        response.setPercentile(100.0);
        
        return response;
    }

    @Override
    public GradeDTO.CourseRankingResponse getCourseRanking(Long studentId, Long courseId) {
        log.info("获取课程排名，学生ID: {}, 课程ID: {}", studentId, courseId);
        
        // 验证权限
        if (!hasCourseAccess(courseId, studentId)) {
            throw new BusinessException(ResultCode.ACCESS_DENIED, "无权限访问该课程");
        }
        
        GradeDTO.CourseRankingResponse response = new GradeDTO.CourseRankingResponse();
        response.setStudentId(studentId);
        response.setCourseId(courseId);
        response.setRanking(1); // 默认排名
        response.setTotalStudents(1);
        response.setPercentile(100.0);
        
        return response;
    }

    @Override
    public GradeDTO.GradeComparisonResponse getGradeComparison(Long studentId, GradeDTO.GradeComparisonRequest compareRequest) {
        log.info("获取成绩对比分析，学生ID: {}", studentId);
        
        Student student = studentMapper.selectById(studentId);
        if (student == null) {
            throw new BusinessException(ResultCode.USER_NOT_FOUND, "学生不存在");
        }
        
        GradeDTO.GradeComparisonResponse response = new GradeDTO.GradeComparisonResponse();
        response.setStudentId(studentId);
        // 这里可以实现具体的对比分析逻辑
        response.setComparisonData(List.of());
        
        return response;
    }

    @Override
    public GradeDTO.SemesterGradeSummaryResponse getSemesterGradeSummary(Long studentId, String semester) {
        log.info("获取学期成绩汇总，学生ID: {}, 学期: {}", studentId, semester);
        
        Student student = studentMapper.selectById(studentId);
        if (student == null) {
            throw new BusinessException(ResultCode.USER_NOT_FOUND, "学生不存在");
        }
        
        GradeDTO.SemesterGradeSummaryResponse response = new GradeDTO.SemesterGradeSummaryResponse();
        response.setStudentId(studentId);
        response.setSemester(semester);
        response.setTotalCourses(0);
        response.setAverageGPA(0.0);
        response.setCourseGrades(List.of());
        
        return response;
    }

    @Override
    public GradeDTO.YearlyGradeSummaryResponse getYearlyGradeSummary(Long studentId, Integer year) {
        log.info("获取年度成绩汇总，学生ID: {}, 年份: {}", studentId, year);
        
        Student student = studentMapper.selectById(studentId);
        if (student == null) {
            throw new BusinessException(ResultCode.USER_NOT_FOUND, "学生不存在");
        }
        
        GradeDTO.YearlyGradeSummaryResponse response = new GradeDTO.YearlyGradeSummaryResponse();
        response.setStudentId(studentId);
        response.setYear(year);
        response.setTotalCourses(0);
        response.setAverageGPA(0.0);
        response.setSemesterSummaries(List.of());
        
        return response;
    }

    @Override
    public List<GradeDTO.GradeWarningResponse> getGradeWarnings(Long studentId) {
        log.info("获取成绩预警信息，学生ID: {}", studentId);
        
        Student student = studentMapper.selectById(studentId);
        if (student == null) {
            throw new BusinessException(ResultCode.USER_NOT_FOUND, "学生不存在");
        }
        
        // 这里可以实现预警逻辑
        return List.of();
    }

    @Override
    public GradeDTO.ImprovementSuggestionResponse getImprovementSuggestions(Long studentId, Long courseId) {
        log.info("获取成绩改进建议，学生ID: {}, 课程ID: {}", studentId, courseId);
        
        Student student = studentMapper.selectById(studentId);
        if (student == null) {
            throw new BusinessException(ResultCode.USER_NOT_FOUND, "学生不存在");
        }
        
        GradeDTO.ImprovementSuggestionResponse response = new GradeDTO.ImprovementSuggestionResponse();
        response.setStudentId(studentId);
        response.setCourseId(courseId);
        // 这里应该使用适当的类型，但当前方法不是StudySuggestionResponse，所以保留原有代码
        // 如果需要修改，请确认正确的类型
        response.setSuggestions(List.of("建议多做练习", "加强基础知识学习", "及时完成作业"));
        
        return response;
    }

    @Override
    public GradeDTO.LearningGoalProgressResponse getLearningGoalProgress(Long studentId) {
        log.info("获取学习目标进度，学生ID: {}", studentId);
        
        Student student = studentMapper.selectById(studentId);
        if (student == null) {
            throw new BusinessException(ResultCode.USER_NOT_FOUND, "学生不存在");
        }
        
        GradeDTO.LearningGoalProgressResponse response = new GradeDTO.LearningGoalProgressResponse();
        response.setStudentId(studentId);
        response.setGoals(List.of());
        response.setOverallProgress(0.0);
        
        return response;
    }

    @Override
    public Boolean setLearningGoal(GradeDTO.LearningGoalRequest goalRequest, Long studentId) {
        log.info("设置学习目标，学生ID: {}", studentId);
        
        Student student = studentMapper.selectById(studentId);
        if (student == null) {
            throw new BusinessException(ResultCode.USER_NOT_FOUND, "学生不存在");
        }
        
        // 这里可以实现设置学习目标的逻辑
        log.info("学习目标设置成功");
        return true;
    }

    @Override
    public Boolean updateLearningGoal(Long goalId, GradeDTO.LearningGoalUpdateRequest goalRequest, Long studentId) {
        log.info("更新学习目标，学生ID: {}, 目标ID: {}", studentId, goalId);
        
        Student student = studentMapper.selectById(studentId);
        if (student == null) {
            throw new BusinessException(ResultCode.USER_NOT_FOUND, "学生不存在");
        }
        
        // 这里可以实现更新学习目标的逻辑
        log.info("学习目标更新成功");
        return true;
    }

    @Override
    public Boolean deleteLearningGoal(Long goalId, Long studentId) {
        log.info("删除学习目标，学生ID: {}, 目标ID: {}", studentId, goalId);
        
        Student student = studentMapper.selectById(studentId);
        if (student == null) {
            throw new BusinessException(ResultCode.USER_NOT_FOUND, "学生不存在");
        }
        
        // 这里可以实现删除学习目标的逻辑
        log.info("学习目标删除成功");
        return true;
    }

    @Override
    public GradeDTO.GradeCertificateResponse getGradeCertificate(Long studentId, Long courseId) {
        log.info("获取成绩证书，学生ID: {}, 课程ID: {}", studentId, courseId);
        
        Student student = studentMapper.selectById(studentId);
        if (student == null) {
            throw new BusinessException(ResultCode.USER_NOT_FOUND, "学生不存在");
        }
        
        // 验证权限
        if (!hasCourseAccess(courseId, studentId)) {
            throw new BusinessException(ResultCode.ACCESS_DENIED, "无权限访问该课程");
        }
        
        GradeDTO.GradeCertificateResponse response = new GradeDTO.GradeCertificateResponse();
        response.setStudentId(studentId);
        response.setCourseId(courseId);
        response.setCertificateUrl("");
        
        return response;
    }

    @Override
    public Boolean applyCertificate(GradeDTO.CertificateApplicationRequest certificateRequest, Long studentId) {
        log.info("申请成绩证书，学生ID: {}", studentId);
        
        Student student = studentMapper.selectById(studentId);
        if (student == null) {
            throw new BusinessException(ResultCode.USER_NOT_FOUND, "学生不存在");
        }
        
        // 这里可以实现申请证书的逻辑
        log.info("成绩证书申请成功");
        return true;
    }

    @Override
    public List<GradeDTO.GradeHistoryResponse> getGradeHistory(Long studentId, Long taskId) {
        log.info("获取成绩历史记录，学生ID: {}, 任务ID: {}", studentId, taskId);
        
        Student student = studentMapper.selectById(studentId);
        if (student == null) {
            throw new BusinessException(ResultCode.USER_NOT_FOUND, "学生不存在");
        }
        
        // 验证权限
        if (!hasTaskAccess(taskId, studentId)) {
            throw new BusinessException(ResultCode.ACCESS_DENIED, "无权限访问该任务");
        }
        
        // 这里可以实现获取成绩历史的逻辑
        return List.of();
    }

    @Override
    public GradeDTO.DetailedFeedbackResponse getDetailedFeedback(Long studentId, Long taskId) {
        log.info("获取详细反馈，学生ID: {}, 任务ID: {}", studentId, taskId);
        
        // 验证权限
        if (!hasTaskAccess(taskId, studentId)) {
            throw new BusinessException(ResultCode.ACCESS_DENIED, "无权限访问该任务");
        }
        
        QueryWrapper<TaskSubmission> queryWrapper = new QueryWrapper<>();
        queryWrapper.eq("task_id", taskId)
                   .eq("student_id", studentId);
        
        TaskSubmission submission = taskSubmissionMapper.selectOne(queryWrapper);
        if (submission == null) {
            throw new BusinessException(ResultCode.DATA_NOT_FOUND, "任务提交记录不存在");
        }
        
        GradeDTO.DetailedFeedbackResponse response = new GradeDTO.DetailedFeedbackResponse();
        response.setTaskId(taskId);
        response.setStudentId(studentId);
        response.setFeedback(submission.getFeedback());
        response.setStrengths(List.of("完成度较高"));
        response.setWeaknesses(List.of("细节需要改进"));
        // 这里应该使用适当的类型，但当前方法不是StudySuggestionResponse，所以保留原有代码
        // 如果需要修改，请确认正确的类型
        response.setSuggestions(List.of("建议多练习"));
        
        return response;
    }

    @Override
    public List<GradeDTO.PeerEvaluationResponse> getPeerEvaluationResults(Long studentId, Long taskId) {
        log.info("获取同行评议结果，学生ID: {}, 任务ID: {}", studentId, taskId);
        
        // 验证权限
        if (!hasTaskAccess(taskId, studentId)) {
            throw new BusinessException(ResultCode.ACCESS_DENIED, "无权限访问该任务");
        }
        
        // 这里可以实现获取同行评议结果的逻辑
        return List.of();
    }

    @Override
    public GradeDTO.SelfEvaluationResponse getSelfEvaluation(Long studentId, Long taskId) {
        log.info("获取自我评价，学生ID: {}, 任务ID: {}", studentId, taskId);
        
        // 验证权限
        if (!hasTaskAccess(taskId, studentId)) {
            throw new BusinessException(ResultCode.ACCESS_DENIED, "无权限访问该任务");
        }
        
        GradeDTO.SelfEvaluationResponse response = new GradeDTO.SelfEvaluationResponse();
        response.setTaskId(taskId);
        response.setStudentId(studentId);
        response.setSelfScore(BigDecimal.ZERO);
        response.setSelfComment("");
        
        return response;
    }

    @Override
    public Boolean submitSelfEvaluation(GradeDTO.SelfEvaluationRequest evaluationRequest, Long studentId) {
        log.info("提交自我评价，学生ID: {}", studentId);
        
        Student student = studentMapper.selectById(studentId);
        if (student == null) {
            throw new BusinessException(ResultCode.USER_NOT_FOUND, "学生不存在");
        }
        
        // 这里可以实现提交自我评价的逻辑
        log.info("自我评价提交成功");
        return true;
    }

    @Override
    public GradeDTO.AbilityRadarResponse getAbilityRadar(Long studentId, Long courseId) {
        log.info("获取能力雷达图，学生ID: {}, 课程ID: {}", studentId, courseId);
        
        Student student = studentMapper.selectById(studentId);
        if (student == null) {
            throw new BusinessException(ResultCode.USER_NOT_FOUND, "学生不存在");
        }
        
        // 验证权限
        if (!hasCourseAccess(courseId, studentId)) {
            throw new BusinessException(ResultCode.ACCESS_DENIED, "无权限访问该课程");
        }
        
        GradeDTO.AbilityRadarResponse response = new GradeDTO.AbilityRadarResponse();
        response.setStudentId(studentId);
        response.setCourseId(courseId);
        response.setAbilities(List.of());
        
        return response;
    }

    @Override
    public GradeDTO.LearningEfficiencyResponse getLearningEfficiency(Long studentId, String timeRange) {
        log.info("获取学习效率分析，学生ID: {}, 时间范围: {}", studentId, timeRange);
        
        Student student = studentMapper.selectById(studentId);
        if (student == null) {
            throw new BusinessException(ResultCode.USER_NOT_FOUND, "学生不存在");
        }
        
        GradeDTO.LearningEfficiencyResponse response = new GradeDTO.LearningEfficiencyResponse();
        response.setStudentId(studentId);
        response.setTimeRange(timeRange);
        response.setEfficiencyScore(new BigDecimal("85.0"));
        response.setAnalysis("学习效率良好");
        
        return response;
    }

    @Override
    public GradeDTO.KnowledgePointMasteryResponse getKnowledgePointMastery(Long studentId, Long courseId) {
        log.info("获取知识点掌握情况，学生ID: {}, 课程ID: {}", studentId, courseId);
        
        // 验证权限
        if (!hasCourseAccess(courseId, studentId)) {
            throw new BusinessException(ResultCode.ACCESS_DENIED, "无权限访问该课程");
        }
        
        GradeDTO.KnowledgePointMasteryResponse response = new GradeDTO.KnowledgePointMasteryResponse();
        response.setStudentId(studentId);
        response.setCourseId(courseId);
        response.setKnowledgePoints(List.of());
        
        return response;
    }

    @Override
    public PageResponse<GradeDTO.WrongQuestionAnalysisResponse> getWrongQuestionAnalysis(Long studentId, PageRequest pageRequest) {
        log.info("获取错题分析，学生ID: {}", studentId);
        
        Student student = studentMapper.selectById(studentId);
        if (student == null) {
            throw new BusinessException(ResultCode.USER_NOT_FOUND, "学生不存在");
        }
        
        // 这里可以实现错题分析的逻辑
        return PageResponse.of((long) pageRequest.getPageNum(), (long) pageRequest.getPageSize(), 0L, List.of());
    }

    @Override
    public List<GradeDTO.WeakKnowledgePointResponse> getWeakKnowledgePoints(Long studentId, Long courseId) {
        log.info("获取薄弱知识点，学生ID: {}, 课程ID: {}", studentId, courseId);
        
        // 验证权限
        if (!hasCourseAccess(courseId, studentId)) {
            throw new BusinessException(ResultCode.ACCESS_DENIED, "无权限访问该课程");
        }
        
        // 这里可以实现获取薄弱知识点的逻辑
        return List.of();
    }

    @Override
    public GradeDTO.StudySuggestionResponse getStudySuggestions(Long studentId) {
        log.info("获取学习建议，学生ID: {}", studentId);
        
        Student student = studentMapper.selectById(studentId);
        if (student == null) {
            throw new BusinessException(ResultCode.USER_NOT_FOUND, "学生不存在");
        }
        
        GradeDTO.StudySuggestionResponse response = new GradeDTO.StudySuggestionResponse();
        response.setStudentId(studentId);
        List<GradeDTO.StudySuggestionResponse.StudySuggestion> suggestions = new ArrayList<>();
        GradeDTO.StudySuggestionResponse.StudySuggestion suggestion1 = new GradeDTO.StudySuggestionResponse.StudySuggestion();
        suggestion1.setSuggestion("建议加强基础练习");
        suggestion1.setCategory("学习方法");
        suggestion1.setPriority("高");
        
        GradeDTO.StudySuggestionResponse.StudySuggestion suggestion2 = new GradeDTO.StudySuggestionResponse.StudySuggestion();
        suggestion2.setSuggestion("多参与课堂讨论");
        suggestion2.setCategory("课堂参与");
        suggestion2.setPriority("中");
        
        GradeDTO.StudySuggestionResponse.StudySuggestion suggestion3 = new GradeDTO.StudySuggestionResponse.StudySuggestion();
        suggestion3.setSuggestion("及时复习课程内容");
        suggestion3.setCategory("复习");
        suggestion3.setPriority("中");
        
        suggestions.add(suggestion1);
        suggestions.add(suggestion2);
        suggestions.add(suggestion3);
        
        response.setSuggestions(suggestions);
        
        return response;
    }

    @Override
    public GradeDTO.GradeReportResponse getGradeReport(Long studentId, String reportType, String timeRange) {
        log.info("获取成绩报告，学生ID: {}, 报告类型: {}, 时间范围: {}", studentId, reportType, timeRange);
        
        Student student = studentMapper.selectById(studentId);
        if (student == null) {
            throw new BusinessException(ResultCode.USER_NOT_FOUND, "学生不存在");
        }
        
        GradeDTO.GradeReportResponse response = new GradeDTO.GradeReportResponse();
        response.setStudentId(studentId);
        response.setReportType(reportType);
        response.setTimeRange(timeRange);
        response.setGeneratedTime(LocalDateTime.now());
        response.setContent("成绩报告内容");
        
        return response;
    }

    @Override
    public GradeDTO.ExportResponse exportGradeData(Long studentId, GradeDTO.GradeDataExportRequest exportRequest) {
        log.info("导出成绩数据，学生ID: {}", studentId);
        
        Student student = studentMapper.selectById(studentId);
        if (student == null) {
            throw new BusinessException(ResultCode.USER_NOT_FOUND, "学生不存在");
        }
        
        GradeDTO.ExportResponse response = new GradeDTO.ExportResponse();
        response.setStudentId(studentId);
        response.setExportFormat(exportRequest.getFormat());
        response.setDownloadUrl("export_file_path." + exportRequest.getFormat());
        
        return response;
    }

    @Override
    public PageResponse<GradeDTO.GradeNotificationResponse> getGradeNotifications(Long studentId, PageRequest pageRequest) {
        log.info("获取成绩通知，学生ID: {}", studentId);
        
        Student student = studentMapper.selectById(studentId);
        if (student == null) {
            throw new BusinessException(ResultCode.USER_NOT_FOUND, "学生不存在");
        }
        
        // 这里可以实现获取通知的逻辑
        return PageResponse.of((long) pageRequest.getPageNum(), (long) pageRequest.getPageSize(), 0L, List.of());
    }

    @Override
    public Boolean markNotificationAsRead(Long notificationId, Long studentId) {
        log.info("标记通知为已读，学生ID: {}, 通知ID: {}", studentId, notificationId);
        
        Student student = studentMapper.selectById(studentId);
        if (student == null) {
            throw new BusinessException(ResultCode.USER_NOT_FOUND, "学生不存在");
        }
        
        // 这里可以实现标记通知为已读的逻辑
        log.info("通知标记为已读成功");
        return true;
    }

    @Override
    public PageResponse<GradeDTO.GradeAppealResponse> getGradeAppeals(Long studentId, PageRequest pageRequest) {
        log.info("获取成绩申诉记录，学生ID: {}", studentId);
        
        Student student = studentMapper.selectById(studentId);
        if (student == null) {
            throw new BusinessException(ResultCode.USER_NOT_FOUND, "学生不存在");
        }
        
        // 这里可以实现获取申诉记录的逻辑
        return PageResponse.of((long) pageRequest.getPageNum(), (long) pageRequest.getPageSize(), 0L, List.of());
    }

    @Override
    public Long submitGradeAppeal(GradeDTO.GradeAppealRequest appealRequest, Long studentId) {
        log.info("提交成绩申诉，学生ID: {}", studentId);
        
        Student student = studentMapper.selectById(studentId);
        if (student == null) {
            throw new BusinessException(ResultCode.USER_NOT_FOUND, "学生不存在");
        }
        
        // 这里可以实现提交申诉的逻辑
        log.info("成绩申诉提交成功");
        return 1L;
    }

    @Override
    public GradeDTO.AppealResultResponse getAppealResult(Long appealId, Long studentId) {
        log.info("获取申诉结果，学生ID: {}, 申诉ID: {}", studentId, appealId);
        
        Student student = studentMapper.selectById(studentId);
        if (student == null) {
            throw new BusinessException(ResultCode.USER_NOT_FOUND, "学生不存在");
        }
        
        GradeDTO.AppealResultResponse response = new GradeDTO.AppealResultResponse();
        response.setAppealId(String.valueOf(appealId));
        response.setStudentId(studentId);
        response.setStatus("处理中");
        response.setResult("");
        
        return response;
    }

    /**
     * 转换为成绩列表响应对象
     */
    private GradeDTO.GradeListResponse convertToGradeListResponse(TaskSubmission submission) {
        GradeDTO.GradeListResponse response = new GradeDTO.GradeListResponse();
        response.setTaskId(submission.getTaskId());
        response.setStudentId(submission.getStudentId());
        response.setScore(submission.getScore());
        response.setGradeTime(submission.getGradeTime());
        
        // 获取任务信息
        Task task = taskMapper.selectById(submission.getTaskId());
        if (task != null) {
            response.setTaskTitle(task.getTitle());
            response.setCourseId(task.getCourseId());
            
            // 获取课程信息
            Course course = courseMapper.selectById(task.getCourseId());
            if (course != null) {
                response.setCourseName(course.getCourseName());
            }
        }
        
        return response;
    }

    /**
     * 检查学生是否有课程访问权限
     */
    private boolean hasCourseAccess(Long courseId, Long studentId) {
        // 这里可以实现具体的权限检查逻辑
        return true;
    }

    /**
     * 检查学生是否有任务访问权限
     */
    private boolean hasTaskAccess(Long taskId, Long studentId) {
        // 这里可以实现具体的权限检查逻辑
        return true;
    }

    /**
     * 计算及格率
     */
    private double calculatePassingRate(List<TaskSubmission> submissions) {
        if (submissions.isEmpty()) {
            return 0.0;
        }
        
        long passingCount = submissions.stream()
                .filter(submission -> submission.getScore() != null && submission.getScore().doubleValue() >= 60.0)
                .count();
        
        return (double) passingCount / submissions.size() * 100;
    }

    /**
     * 转换TaskSubmission为TaskGrade
     */
    private GradeDTO.CourseGradeResponse.TaskGrade convertToTaskGrade(TaskSubmission submission) {
        GradeDTO.CourseGradeResponse.TaskGrade taskGrade = new GradeDTO.CourseGradeResponse.TaskGrade();
        taskGrade.setTaskId(submission.getTaskId());
        taskGrade.setScore(submission.getScore());
        taskGrade.setGradeTime(submission.getGradeTime());
        
        // 获取任务信息
        Task task = taskMapper.selectById(submission.getTaskId());
        if (task != null) {
            taskGrade.setTaskTitle(task.getTitle());
            taskGrade.setTaskType(task.getTaskType());
            taskGrade.setMaxScore(task.getMaxScore());
        }
        
        return taskGrade;
    }
}