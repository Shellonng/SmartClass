package com.education.service.teacher.impl;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.education.dto.StudentDTO;
import com.education.dto.StudentDTOExtension;
import com.education.dto.common.PageRequest;
import com.education.dto.common.PageResponse;
import com.education.entity.Student;
import com.education.entity.User;
import com.education.entity.LearningProgress;
import com.education.mapper.StudentMapper;
import com.education.mapper.UserMapper;
import com.education.mapper.LearningProgressMapper;
import com.education.mapper.TaskSubmissionMapper;
import com.education.service.teacher.StudentService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.util.StringUtils;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

/**
 * 教师端学生服务实现类
 */
@Service
@Slf4j
public class StudentServiceImpl implements StudentService {
    
    @Autowired
    private StudentMapper studentMapper;
    
    @Autowired
    private UserMapper userMapper;
    
    @Autowired
    private LearningProgressMapper learningProgressMapper;
    
    @Autowired
    private TaskSubmissionMapper taskSubmissionMapper;

    @Override
    public PageResponse<StudentDTO.StudentResponse> getStudentList(Long teacherId, PageRequest pageRequest) {
        log.info("获取学生列表，教师ID: {}", teacherId);
        
        // 构建查询条件
        QueryWrapper<Student> queryWrapper = new QueryWrapper<>();
        queryWrapper.eq("is_deleted", false);
        
        // 添加关键词搜索
        if (StringUtils.hasText(pageRequest.getKeyword())) {
            queryWrapper.and(wrapper -> wrapper
                .like("student_name", pageRequest.getKeyword())
                .or()
                .like("student_number", pageRequest.getKeyword()));
        }
        
        queryWrapper.orderByDesc("create_time");
        
        // 分页查询
        Page<Student> page = new Page<>(pageRequest.getPageNum(), pageRequest.getPageSize());
        Page<Student> studentPage = studentMapper.selectPage(page, queryWrapper);
        
        // 转换为响应对象
        List<StudentDTO.StudentResponse> studentResponses = studentPage.getRecords().stream()
                .map(this::convertToStudentResponse)
                .collect(Collectors.toList());
        
        return PageResponse.<StudentDTO.StudentResponse>builder()
                .records(studentResponses)
                .total(studentPage.getTotal())
                .current(pageRequest.getCurrent())
                .pageSize(pageRequest.getPageSize())
                .build();
    }

    @Override
    public StudentDTO.StudentDetailResponse getStudentDetail(Long studentId, Long teacherId) {
        log.info("获取学生详情，学生ID: {}, 教师ID: {}", studentId, teacherId);
        
        // 查询学生信息
        Student student = studentMapper.selectById(studentId);
        if (student == null || student.getIsDeleted()) {
            throw new RuntimeException("学生不存在");
        }
        
        // 查询用户信息
        User user = userMapper.selectById(student.getUserId());
        
        // 构建详情响应
        StudentDTO.StudentDetailResponse response = new StudentDTO.StudentDetailResponse();
        response.setStudentId(studentId);
        // 从User实体获取姓名
        response.setRealName(user != null ? user.getRealName() : "");
        response.setStudentNumber(student.getStudentNo());
        response.setGrade(student.getGrade() != null ? student.getGrade().toString() : "");
        response.setMajor(student.getMajor());
        response.setStatus(student.getEnrollmentStatus());
        response.setCreateTime(student.getCreateTime());
        
        if (user != null) {
            response.setEmail(user.getEmail());
            response.setPhone(user.getPhone());
            response.setAvatar(user.getAvatar());
        }
        
        return response;
    }

    @Override
    public StudentDTO.StudentProgressResponse getStudentProgress(Long studentId, Long teacherId, Long courseId) {
        log.info("获取学生学习进度，学生ID: {}, 教师ID: {}, 课程ID: {}", studentId, teacherId, courseId);
        
        StudentDTO.StudentProgressResponse response = new StudentDTO.StudentProgressResponse();
        response.setStudentId(studentId);
        response.setCourseId(courseId);
        
        if (courseId != null) {
            // 查询特定课程的学习进度
            Double averageProgress = learningProgressMapper.calculateAverageProgress(studentId, courseId);
            Integer completedChapters = learningProgressMapper.countCompletedChapters(studentId, courseId);
            // 记录但不使用，避免编译警告
            @SuppressWarnings("unused")
            Integer totalStudyDuration = learningProgressMapper.calculateTotalStudyDuration(studentId, courseId);
            
            response.setProgressPercentage(averageProgress != null ? averageProgress : 0.0);
            response.setCompletedLessons(completedChapters != null ? completedChapters : 0);
        } else {
            // 查询所有课程的学习进度概览
            response.setProgressPercentage(0.0);
            response.setCompletedLessons(0);
        }
        
        return response;
    }

    @Override
    public StudentDTO.StudentGradeStatisticsResponse getStudentGradeStatistics(Long studentId, Long teacherId) {
        log.info("获取学生成绩统计，学生ID: {}, 教师ID: {}", studentId, teacherId);
        
        StudentDTO.StudentGradeStatisticsResponse response = new StudentDTO.StudentGradeStatisticsResponse();
        response.setStudentId(studentId);
        
        // 查询学生的作业提交统计
        Integer totalSubmissions = taskSubmissionMapper.countSubmissionsByStudent(studentId);
        Integer gradedSubmissions = taskSubmissionMapper.countGradedSubmissionsByStudent(studentId);
        Double averageScore = taskSubmissionMapper.calculateAverageScoreByStudent(studentId);
        Integer highestScore = taskSubmissionMapper.getHighestScoreByStudent(studentId);
        Integer lowestScore = taskSubmissionMapper.getLowestScoreByStudent(studentId);
        
        response.setTotalAssignments(totalSubmissions != null ? totalSubmissions : 0);
        response.setGradedAssignments(gradedSubmissions != null ? gradedSubmissions : 0);
        response.setOverallAverage(averageScore != null ? averageScore : 0.0);
        response.setHighestGrade(highestScore != null ? highestScore.doubleValue() : null);
        response.setLowestGrade(lowestScore != null ? lowestScore.doubleValue() : null);
        
        return response;
    }

    @Override
    public PageResponse<StudentDTO.StudentSubmissionResponse> getStudentSubmissions(Long studentId, Long teacherId, PageRequest pageRequest) {
        log.info("获取学生提交记录，学生ID: {}, 教师ID: {}", studentId, teacherId);
        
        // 构建查询条件
        QueryWrapper<com.education.entity.TaskSubmission> queryWrapper = new QueryWrapper<>();
        queryWrapper.eq("student_id", studentId)
                   .eq("is_deleted", false)
                   .orderByDesc("submit_time");
        
        // 分页查询
        Page<com.education.entity.TaskSubmission> page = new Page<>(pageRequest.getPageNum(), pageRequest.getPageSize());
        Page<com.education.entity.TaskSubmission> submissionPage = taskSubmissionMapper.selectPage(page, queryWrapper);
        
        // 转换为响应对象
        List<StudentDTO.StudentSubmissionResponse> submissionResponses = submissionPage.getRecords().stream()
                .map(this::convertToSubmissionResponse)
                .collect(Collectors.toList());
        
        return PageResponse.<StudentDTO.StudentSubmissionResponse>builder()
                .records(submissionResponses)
                .total(submissionPage.getTotal())
                .current(pageRequest.getCurrent())
                .pageSize(pageRequest.getPageSize())
                .build();
    }

    @Override
    public StudentDTO.StudentAnalysisResponse getStudentAnalysis(Long studentId, Long teacherId, String timeRange) {
        log.info("获取学生学习分析，学生ID: {}, 教师ID: {}", studentId, teacherId);
        
        StudentDTO.StudentAnalysisResponse response = new StudentDTO.StudentAnalysisResponse();
        response.setStudentId(studentId);
        
        // 学习活跃度分析
        Integer recentSubmissions = taskSubmissionMapper.countRecentSubmissionsByStudent(studentId, 30); // 最近30天
        Integer totalLearningTime = learningProgressMapper.calculateTotalStudyDurationForStudent(studentId);
        
        // 设置参与度和表现分数（基于提交数和学习时长）
        Double engagementScore = recentSubmissions != null ? Math.min(recentSubmissions * 10.0, 100.0) : 0.0;
        Double performanceScore = totalLearningTime != null ? Math.min(totalLearningTime / 60.0 * 5, 100.0) : 0.0;
        
        response.setEngagementScore(engagementScore);
        response.setPerformanceScore(performanceScore);
        
        // 设置学习风险等级
        String riskLevel = "低风险";
        if (engagementScore < 30 || performanceScore < 30) {
            riskLevel = "高风险";
        } else if (engagementScore < 60 || performanceScore < 60) {
            riskLevel = "中风险";
        }
        response.setRiskLevel(riskLevel);
        
        // 设置推荐建议
        List<String> recommendations = new ArrayList<>();
        if (recentSubmissions != null && recentSubmissions < 5) {
            recommendations.add("建议增加作业提交频率");
        }
        if (totalLearningTime != null && totalLearningTime < 300) {
            recommendations.add("建议增加每日学习时间");
        }
        recommendations.add("多参与课堂讨论");
        response.setRecommendations(recommendations);
        
        response.setAnalysisType(timeRange);
        response.setLastAnalysis(LocalDateTime.now());
        
        return response;
    }

    @Override
    public StudentDTO.StudentImportResponse importStudents(StudentDTO.StudentImportRequest importRequest, Long teacherId) {
        // TODO: 实现批量导入学生逻辑
        throw new UnsupportedOperationException("Method not implemented yet");
    }

    @Override
    public String exportStudents(StudentDTOExtension.StudentExportRequest exportRequest, Long teacherId) {
        // TODO: 实现导出学生信息逻辑
        throw new UnsupportedOperationException("Method not implemented yet");
    }

    @Override
    public String resetStudentPassword(Long studentId, Long teacherId) {
        log.info("重置学生密码，学生ID: {}, 教师ID: {}", studentId, teacherId);
        
        // 查询学生信息
        Student student = studentMapper.selectById(studentId);
        if (student == null || student.getIsDeleted()) {
            throw new RuntimeException("学生不存在");
        }
        
        // 查询用户信息
        User user = userMapper.selectById(student.getUserId());
        if (user == null) {
            throw new RuntimeException("用户信息不存在");
        }
        
        // 重置密码为默认密码（通常是学号）
        String defaultPassword = student.getStudentNo();
        user.setPassword(defaultPassword); // 实际应用中需要加密
        user.setUpdateTime(LocalDateTime.now());
        
        userMapper.updateById(user);
        log.info("学生密码重置成功，学生ID: {}", studentId);
        
        return "密码重置成功，新密码为: " + defaultPassword;
    }

    @Override
    public List<StudentDTOExtension.StudentRankResponse> getStudentRanking(Long teacherId, String rankType, Long classId, Integer limit) {
        // TODO: 实现获取学生排行榜逻辑
        throw new UnsupportedOperationException("Method not implemented yet");
    }

    @Override
    public PageResponse<StudentDTO.StudentResponse> searchStudents(String keyword, Long teacherId, PageRequest pageRequest) {
        // TODO: 实现搜索学生逻辑
        throw new UnsupportedOperationException("Method not implemented yet");
    }

    @Override
    public List<Object> getStudentCourses(Long studentId, Long teacherId) {
        // TODO: 实现获取学生课程列表逻辑
        throw new UnsupportedOperationException("Method not implemented yet");
    }

    @Override
    public List<Object> getStudentClasses(Long studentId, Long teacherId) {
        // TODO: 实现获取学生班级信息逻辑
        throw new UnsupportedOperationException("Method not implemented yet");
    }

    @Override
    public Boolean setStudentStatus(Long studentId, String status, Long teacherId) {
        log.info("设置学生状态，学生ID: {}, 状态: {}, 教师ID: {}", studentId, status, teacherId);
        
        // 查询学生信息
        Student student = studentMapper.selectById(studentId);
        if (student == null || student.getIsDeleted()) {
            throw new RuntimeException("学生不存在");
        }
        
        // 更新学生状态
        student.setStatus(status);
        student.setUpdateTime(LocalDateTime.now());
        
        studentMapper.updateById(student);
        log.info("学生状态更新成功，学生ID: {}, 新状态: {}", studentId, status);
        return true;
    }
    
    /**
     * 转换学生实体为响应对象
     */
    private StudentDTO.StudentResponse convertToStudentResponse(Student student) {
        StudentDTO.StudentResponse response = new StudentDTO.StudentResponse();
        response.setStudentId(student.getId());
        // 从关联用户获取真实姓名，如果没有则使用学号
        if (student.getUser() != null) {
            response.setRealName(student.getUser().getRealName());
        }
        response.setStudentNumber(student.getStudentNo());
        response.setGrade(student.getGrade() != null ? student.getGrade().toString() : "");
        response.setMajor(student.getMajor());
        response.setClassName(student.getClassId() != null ? student.getClassId().toString() : "");
        response.setStatus(student.getStatus());
        response.setCreateTime(student.getCreateTime());
        return response;
    }
    
    /**
     * 转换提交记录实体为响应对象
     */
    private StudentDTO.StudentSubmissionResponse convertToSubmissionResponse(com.education.entity.TaskSubmission submission) {
        StudentDTO.StudentSubmissionResponse response = new StudentDTO.StudentSubmissionResponse();
        response.setSubmissionId(submission.getId());
        response.setStudentId(submission.getStudentId());
        response.setAssignmentId(submission.getTaskId());
        response.setSubmissionContent(submission.getContent());
        response.setAttachmentUrl(submission.getFiles());
        response.setStatus(submission.getStatus());
        response.setGrade(submission.getScore() != null ? submission.getScore().doubleValue() : null);
        response.setFeedback(submission.getFeedback());
        response.setSubmitTime(submission.getSubmitTime());
        response.setGradeTime(submission.getGradeTime());
        
        // TODO: 设置学生姓名、作业标题和评分者姓名
        // response.setStudentName(...);
        // response.setAssignmentTitle(...);
        // response.setGraderName(...);
        return response;
    }

    @Override
    public Object getStudentLearningTime(Long studentId, Long teacherId, String timeRange) {
        // TODO: 实现获取学生学习时长统计逻辑
        throw new UnsupportedOperationException("Method not implemented yet");
    }

    @Override
    public Object getStudentWrongQuestions(Long studentId, Long teacherId) {
        // TODO: 实现获取学生错题统计逻辑
        throw new UnsupportedOperationException("Method not implemented yet");
    }

    @Override
    public PageResponse<Object> getStudentLearningTrack(Long studentId, Long teacherId, PageRequest pageRequest) {
        // TODO: 实现获取学生学习轨迹逻辑
        throw new UnsupportedOperationException("Method not implemented yet");
    }

    @Override
    public Boolean addStudentTags(Long studentId, List<String> tags, Long teacherId) {
        // TODO: 实现为学生添加标签逻辑
        throw new UnsupportedOperationException("Method not implemented yet");
    }

    @Override
    public Boolean removeStudentTags(Long studentId, List<String> tags, Long teacherId) {
        // TODO: 实现移除学生标签逻辑
        throw new UnsupportedOperationException("Method not implemented yet");
    }

    @Override
    public List<String> getStudentTags(Long studentId, Long teacherId) {
        // TODO: 实现获取学生标签逻辑
        throw new UnsupportedOperationException("Method not implemented yet");
    }

    @Override
    public Boolean sendMessageToStudent(Long studentId, String message, Long teacherId) {
        // TODO: 实现发送消息给学生逻辑
        throw new UnsupportedOperationException("Method not implemented yet");
    }

    @Override
    public PageResponse<Object> getStudentMessages(Long studentId, Long teacherId, PageRequest pageRequest) {
        // TODO: 实现获取学生消息记录逻辑
        throw new UnsupportedOperationException("Method not implemented yet");
    }

    @Override
    public Object getStudentAttendance(Long studentId, Long teacherId, String timeRange) {
        // TODO: 实现获取学生出勤统计逻辑
        throw new UnsupportedOperationException("Method not implemented yet");
    }

    @Override
    public Boolean recordStudentAttendance(Long studentId, String attendanceType, Long courseId, Long teacherId) {
        // TODO: 实现记录学生出勤逻辑
        throw new UnsupportedOperationException("Method not implemented yet");
    }

    @Override
    public Object getStudentParentContact(Long studentId, Long teacherId) {
        // TODO: 实现获取学生家长联系信息逻辑
        throw new UnsupportedOperationException("Method not implemented yet");
    }

    @Override
    public String generateStudentReport(Long studentId, Long teacherId, String reportType, String timeRange) {
        // TODO: 实现生成学生学习报告逻辑
        throw new UnsupportedOperationException("Method not implemented yet");
    }

    @Override
    public Object getStudentAbilityAssessment(Long studentId, Long teacherId) {
        // TODO: 实现获取学生能力评估逻辑
        throw new UnsupportedOperationException("Method not implemented yet");
    }

    @Override
    public Boolean updateStudentAbilityAssessment(Long studentId, Object assessment, Long teacherId) {
        // TODO: 实现更新学生能力评估逻辑
        throw new UnsupportedOperationException("Method not implemented yet");
    }

    @Override
    public Object getStudentRanking(Long teacherId, Object rankingParams) {
        log.info("获取学生排名，教师ID: {}, 排名参数: {}", teacherId, rankingParams);
        
        // TODO: 实现学生排名逻辑
        return null;
    }
}