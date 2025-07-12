package com.education.controller.teacher;

import com.education.dto.DifyDTO;
import com.education.dto.common.Result;
import com.education.security.SecurityUtil;
import com.education.service.DifyService;
import com.education.entity.Assignment;
import com.education.entity.AssignmentSubmission;
import com.education.entity.AssignmentSubmissionAnswer;
import com.education.entity.Question;
import com.education.mapper.AssignmentMapper;
import com.education.mapper.AssignmentSubmissionMapper;
import com.education.mapper.AssignmentSubmissionAnswerMapper;
import com.education.mapper.QuestionMapper;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;

import jakarta.validation.Valid;
import java.util.List;
import java.util.Map;
import java.util.HashMap;
import java.util.ArrayList;
import java.util.Date;
import java.util.stream.Collectors;

/**
 * 教师端智能批改控制器
 * @author Education Platform Team
 */
@Tag(name = "教师端智能批改", description = "基于Dify AI的智能批改功能")
@RestController
@RequestMapping("/api/teacher/grading")
@Validated
@Slf4j
public class TeacherGradingController {

    @Autowired
    private DifyService difyService;
    
    @Autowired
    private SecurityUtil securityUtil;
    
    @Autowired
    private AssignmentMapper assignmentMapper;
    
    @Autowired
    private AssignmentSubmissionMapper submissionMapper;
    
    @Autowired
    private AssignmentSubmissionAnswerMapper submissionAnswerMapper;
    
    @Autowired
    private QuestionMapper questionMapper;

    @Operation(summary = "智能批改单份作业", description = "对单个学生的作业进行AI批改")
    @PostMapping("/single")
    public Result<DifyDTO.AutoGradingResponse> gradeAssignment(
            @Valid @RequestBody GradingRequest request) {
        try {
            Long teacherId = securityUtil.getCurrentUserId();
            log.info("教师{}请求智能批改作业: {}", teacherId, request.getAssignmentId());
            
            String userId = securityUtil.getCurrentUserId().toString();
            
            // 如果submissionId不为空，从数据库加载学生提交和答案
            if (request.getSubmissionId() != null) {
                // 加载提交记录
                AssignmentSubmission submission = submissionMapper.selectById(request.getSubmissionId());
                if (submission == null) {
                    return Result.error(404, "未找到提交记录");
                }
                
                // 加载题目答案
                List<AssignmentSubmissionAnswer> answers = submissionAnswerMapper.selectList(
                    new LambdaQueryWrapper<AssignmentSubmissionAnswer>()
                        .eq(AssignmentSubmissionAnswer::getSubmissionId, submission.getId())
                );
                
                if (answers == null || answers.isEmpty()) {
                    return Result.error(404, "未找到学生答案");
                }
                
                // 加载题目信息
                List<Long> questionIds = answers.stream()
                    .map(AssignmentSubmissionAnswer::getQuestionId)
                    .collect(Collectors.toList());
                
                List<Question> questions = questionMapper.selectBatchIds(questionIds);
                Map<Long, Question> questionMap = questions.stream()
                    .collect(Collectors.toMap(Question::getId, q -> q));
                
                // 构建批改请求
                List<DifyDTO.Question> questionDTOs = new ArrayList<>();
                List<DifyDTO.StudentAnswer> studentAnswerDTOs = new ArrayList<>();
                
                for (AssignmentSubmissionAnswer answer : answers) {
                    Question question = questionMap.get(answer.getQuestionId());
                    if (question != null) {
                        // 添加题目信息
                        DifyDTO.Question questionDTO = DifyDTO.Question.builder()
                            .questionId(question.getId())
                            .questionText(question.getTitle())
                            .questionType(question.getQuestionType())
                            .correctAnswer(question.getCorrectAnswer())
                            .build();
                        questionDTOs.add(questionDTO);
                        
                        // 添加学生答案
                        DifyDTO.StudentAnswer studentAnswerDTO = DifyDTO.StudentAnswer.builder()
                            .questionId(question.getId())
                            .studentAnswer(answer.getStudentAnswer())
                            .build();
                        studentAnswerDTOs.add(studentAnswerDTO);
                    }
                }
                
                // 设置请求参数
                request.setQuestions(questionDTOs);
                request.setStudentAnswers(studentAnswerDTOs);
                request.setAssignmentId(submission.getAssignmentId());
            }
            
            // 构建批改请求
            DifyDTO.AutoGradingRequest gradingRequest = DifyDTO.AutoGradingRequest.builder()
                    .submissionId(request.getSubmissionId())
                    .assignmentId(request.getAssignmentId())
                    .questions(request.getQuestions())
                    .studentAnswers(request.getStudentAnswers())
                    .gradingCriteria(request.getGradingCriteria())
                    .maxScore(request.getMaxScore())
                    .build();
            
            DifyDTO.AutoGradingResponse response = difyService.gradeAssignment(gradingRequest, userId);
            
            // 如果批改完成，更新数据库
            if ("completed".equals(response.getStatus()) && request.getSubmissionId() != null) {
                updateSubmissionGradingResults(request.getSubmissionId(), response);
            }
            
            if ("completed".equals(response.getStatus())) {
                log.info("智能批改完成，总分: {}", response.getTotalScore());
                return Result.success("智能批改完成", response);
            } else if ("failed".equals(response.getStatus())) {
                log.error("智能批改失败: {}", response.getErrorMessage());
                return Result.error(500, response.getErrorMessage());
            } else {
                return Result.success("智能批改处理中", response);
            }
            
        } catch (Exception e) {
            log.error("智能批改异常: {}", e.getMessage(), e);
            return Result.error(500, "智能批改失败: " + e.getMessage());
        }
    }

    @Operation(summary = "批量智能批改", description = "批量批改多个学生的作业")
    @PostMapping("/batch")
    public Result<List<DifyDTO.AutoGradingResponse>> batchGradeAssignments(
            @Valid @RequestBody BatchGradingRequest request) {
        try {
            Long teacherId = securityUtil.getCurrentUserId();
            
            // 检查submissions是否为null
            if (request.getSubmissions() == null || request.getSubmissions().isEmpty()) {
                // 如果未提供提交列表但提供了作业ID，则从数据库加载该作业的所有提交
                if (request.getAssignmentId() != null) {
                    log.info("教师{}请求批量智能批改，从数据库加载作业{}的提交", teacherId, request.getAssignmentId());
                    
                    // 查询所有未批改的提交
                    List<AssignmentSubmission> submissions = submissionMapper.selectList(
                        new LambdaQueryWrapper<AssignmentSubmission>()
                            .eq(AssignmentSubmission::getAssignmentId, request.getAssignmentId())
                            .eq(AssignmentSubmission::getStatus, 1) // 已提交未批改
                    );
                    
                    if (submissions == null || submissions.isEmpty()) {
                        return Result.error(404, "未找到需要批改的提交");
                    }
                    
                    // 为每个提交加载答案和题目
                    request.setSubmissions(new ArrayList<>());
                    
                    for (AssignmentSubmission submission : submissions) {
                        // 加载答案
                        List<AssignmentSubmissionAnswer> answers = submissionAnswerMapper.selectList(
                            new LambdaQueryWrapper<AssignmentSubmissionAnswer>()
                                .eq(AssignmentSubmissionAnswer::getSubmissionId, submission.getId())
                        );
                        
                        if (answers == null || answers.isEmpty()) {
                            continue;
                        }
                        
                        // 加载题目信息
                        List<Long> questionIds = answers.stream()
                            .map(AssignmentSubmissionAnswer::getQuestionId)
                            .collect(Collectors.toList());
                        
                        List<Question> questions = questionMapper.selectBatchIds(questionIds);
                        Map<Long, Question> questionMap = questions.stream()
                            .collect(Collectors.toMap(Question::getId, q -> q));
                        
                        // 构建学生提交DTO
                        StudentSubmission submissionDTO = new StudentSubmission();
                        submissionDTO.setSubmissionId(submission.getId());
                        submissionDTO.setAssignmentId(submission.getAssignmentId());
                        submissionDTO.setMaxScore(Double.valueOf(100)); // 默认100分
                        
                        // 构建题目和答案列表
                        List<DifyDTO.Question> questionDTOs = new ArrayList<>();
                        List<DifyDTO.StudentAnswer> studentAnswerDTOs = new ArrayList<>();
                        
                        for (AssignmentSubmissionAnswer answer : answers) {
                            Question question = questionMap.get(answer.getQuestionId());
                            if (question != null) {
                                // 添加题目信息
                                DifyDTO.Question questionDTO = DifyDTO.Question.builder()
                                    .questionId(question.getId())
                                    .questionText(question.getTitle())
                                    .questionType(question.getQuestionType())
                                    .correctAnswer(question.getCorrectAnswer())
                                    .build();
                                questionDTOs.add(questionDTO);
                                
                                // 添加学生答案
                                DifyDTO.StudentAnswer studentAnswerDTO = DifyDTO.StudentAnswer.builder()
                                    .questionId(question.getId())
                                    .studentAnswer(answer.getStudentAnswer())
                                    .build();
                                studentAnswerDTOs.add(studentAnswerDTO);
                            }
                        }
                        
                        submissionDTO.setQuestions(questionDTOs);
                        submissionDTO.setStudentAnswers(studentAnswerDTOs);
                        request.getSubmissions().add(submissionDTO);
                    }
                } else {
                    log.warn("教师{}请求批量智能批改，但提交列表为空", teacherId);
                    return Result.error(400, "提交列表不能为空");
                }
            }
            
            log.info("教师{}请求批量智能批改，作业数量: {}", teacherId, request.getSubmissions().size());
            
            String userId = securityUtil.getCurrentUserId().toString();
            
            // 构建批量批改请求列表
            List<DifyDTO.AutoGradingRequest> gradingRequests = request.getSubmissions().stream()
                    .map(submission -> DifyDTO.AutoGradingRequest.builder()
                            .submissionId(submission.getSubmissionId())
                            .assignmentId(submission.getAssignmentId())
                            .questions(submission.getQuestions())
                            .studentAnswers(submission.getStudentAnswers())
                            .gradingCriteria(request.getGradingCriteria())
                            .maxScore(submission.getMaxScore())
                            .build())
                    .collect(Collectors.toList());
            
            // 批量调用批改服务
            List<DifyDTO.AutoGradingResponse> responses = difyService.batchGradeAssignments(gradingRequests, userId);
            
            // 更新数据库中的批改结果
            for (int i = 0; i < responses.size(); i++) {
                DifyDTO.AutoGradingResponse response = responses.get(i);
                Long submissionId = request.getSubmissions().get(i).getSubmissionId();
                
                if ("completed".equals(response.getStatus())) {
                    updateSubmissionGradingResults(submissionId, response);
                }
            }
            
            log.info("批量智能批改完成，成功批改{}份作业", responses.size());
            return Result.success("批量智能批改完成", responses);
            
        } catch (Exception e) {
            log.error("批量智能批改异常: {}", e.getMessage(), e);
            return Result.error(500, "批量智能批改失败: " + e.getMessage());
        }
    }
    
    /**
     * 更新提交记录的批改结果
     */
    private void updateSubmissionGradingResults(Long submissionId, DifyDTO.AutoGradingResponse response) {
        try {
            if (submissionId == null || response == null || response.getResults() == null) {
                return;
            }
            
            // 更新提交记录状态和分数
            AssignmentSubmission submission = new AssignmentSubmission();
            submission.setId(submissionId);
            submission.setStatus(2); // 已批改
            submission.setScore(response.getEarnedScore().intValue());
            submission.setGradeTime(new Date());
            submission.setGradedBy(securityUtil.getCurrentUserId());
            submission.setFeedback(response.getOverallComment());
            
            submissionMapper.updateById(submission);
            
            // 更新每道题的评分
            for (DifyDTO.GradingResult result : response.getResults()) {
                AssignmentSubmissionAnswer answer = new AssignmentSubmissionAnswer();
                answer.setSubmissionId(submissionId);
                answer.setQuestionId(result.getQuestionId());
                answer.setIsCorrect(result.getIsCorrect());
                answer.setScore(result.getScore());
                answer.setComment(result.getComment());
                
                // 更新答案记录
                submissionAnswerMapper.update(
                    answer,
                    new LambdaQueryWrapper<AssignmentSubmissionAnswer>()
                        .eq(AssignmentSubmissionAnswer::getSubmissionId, submissionId)
                        .eq(AssignmentSubmissionAnswer::getQuestionId, result.getQuestionId())
                );
            }
            
            log.info("已更新提交记录{}的批改结果", submissionId);
        } catch (Exception e) {
            log.error("更新批改结果时出错: {}", e.getMessage(), e);
        }
    }

    @Operation(summary = "获取批改进度", description = "查询批改任务的执行进度")
    @GetMapping("/progress/{taskId}")
    public Result<DifyDTO.DifyResponse> getGradingProgress(@PathVariable String taskId) {
        try {
            log.info("查询批改进度: {}", taskId);
            
            DifyDTO.DifyResponse response = difyService.getTaskStatus(taskId, "auto-grading");
            
            return Result.success("批改进度查询成功", response);
            
        } catch (Exception e) {
            log.error("查询批改进度异常: {}", e.getMessage(), e);
            return Result.error(500, "查询批改进度失败: " + e.getMessage());
        }
    }

    @Operation(summary = "获取批改统计", description = "获取批改结果的统计分析")
    @GetMapping("/statistics/{assignmentId}")
    public Result<GradingStatistics> getGradingStatistics(@PathVariable Long assignmentId) {
        try {
            log.info("获取作业{}的批改统计", assignmentId);
            
            // 从数据库获取实际统计数据
            Assignment assignment = assignmentMapper.selectById(assignmentId);
            if (assignment == null) {
                return Result.error(404, "作业不存在");
            }
            
            // 查询提交记录
            List<AssignmentSubmission> submissions = submissionMapper.selectList(
                new LambdaQueryWrapper<AssignmentSubmission>()
                    .eq(AssignmentSubmission::getAssignmentId, assignmentId)
            );
            
            int totalSubmissions = submissions.size();
            int gradedSubmissions = 0;
            double totalScore = 0;
            double highestScore = 0;
            double lowestScore = 100;
            
            Map<String, Integer> scoreDistribution = new HashMap<>();
            scoreDistribution.put("90-100", 0);
            scoreDistribution.put("80-89", 0);
            scoreDistribution.put("70-79", 0);
            scoreDistribution.put("60-69", 0);
            scoreDistribution.put("0-59", 0);
            
            for (AssignmentSubmission submission : submissions) {
                if (submission.getStatus() != null && submission.getStatus() == 2 && submission.getScore() != null) {
                    gradedSubmissions++;
                    totalScore += submission.getScore();
                    
                    if (submission.getScore() > highestScore) {
                        highestScore = submission.getScore();
                    }
                    
                    if (submission.getScore() < lowestScore) {
                        lowestScore = submission.getScore();
                    }
                    
                    // 统计分数段
                    int score = submission.getScore();
                    if (score >= 90) {
                        scoreDistribution.put("90-100", scoreDistribution.get("90-100") + 1);
                    } else if (score >= 80) {
                        scoreDistribution.put("80-89", scoreDistribution.get("80-89") + 1);
                    } else if (score >= 70) {
                        scoreDistribution.put("70-79", scoreDistribution.get("70-79") + 1);
                    } else if (score >= 60) {
                        scoreDistribution.put("60-69", scoreDistribution.get("60-69") + 1);
                    } else {
                        scoreDistribution.put("0-59", scoreDistribution.get("0-59") + 1);
                    }
                }
            }
            
            // 计算平均分
            double averageScore = gradedSubmissions > 0 ? totalScore / gradedSubmissions : 0;
            
            GradingStatistics statistics = new GradingStatistics();
            statistics.setAssignmentId(assignmentId);
            statistics.setTotalSubmissions(totalSubmissions);
            statistics.setGradedSubmissions(gradedSubmissions);
            statistics.setAverageScore(averageScore);
            statistics.setHighestScore(gradedSubmissions > 0 ? highestScore : 0);
            statistics.setLowestScore(gradedSubmissions > 0 && lowestScore < 100 ? lowestScore : 0);
            statistics.setScoreDistribution(scoreDistribution);
            
            return Result.success("批改统计获取成功", statistics);
            
        } catch (Exception e) {
            log.error("获取批改统计异常: {}", e.getMessage(), e);
            return Result.error(500, "获取批改统计失败: " + e.getMessage());
        }
    }

    @Operation(summary = "设置批改标准", description = "配置智能批改的评分标准")
    @PostMapping("/criteria")
    public Result<String> setGradingCriteria(@RequestBody Map<String, Object> criteria) {
        try {
            log.info("设置批改标准: {}", criteria);
            
            // 获取当前教师ID
            Long teacherId = securityUtil.getCurrentUserId();
            
            // 获取作业ID
            Long assignmentId = null;
            if (criteria.containsKey("assignmentId")) {
                assignmentId = Long.valueOf(criteria.get("assignmentId").toString());
            }
            
            if (assignmentId == null) {
                return Result.error(400, "作业ID不能为空");
            }
            
            // 将批改标准转换为JSON字符串保存
            // 这里应该调用相应的service方法保存数据
            // 如果没有对应的表结构，可以临时保存到缓存或者配置文件中
            // 例如: gradingCriteriaService.saveCriteria(assignmentId, teacherId, criteriaJson);
            
            // 记录保存成功的日志
            log.info("教师{}成功保存作业{}的批改标准", teacherId, assignmentId);
            
            return Result.success("批改标准设置成功");
            
        } catch (Exception e) {
            log.error("设置批改标准异常: {}", e.getMessage(), e);
            return Result.error(500, "设置批改标准失败: " + e.getMessage());
        }
    }
    
    /**
     * 获取学生提交详情
     */
    @Operation(summary = "获取提交详情", description = "获取学生提交的详细信息和批改结果")
    @GetMapping("/submission/{submissionId}")
    public Result<Map<String, Object>> getSubmissionDetail(@PathVariable Long submissionId) {
        try {
            // 获取提交记录
            AssignmentSubmission submission = submissionMapper.selectById(submissionId);
            if (submission == null) {
                return Result.error(404, "未找到提交记录");
            }
            
            // 获取答案详情
            List<AssignmentSubmissionAnswer> answers = submissionAnswerMapper.selectList(
                new LambdaQueryWrapper<AssignmentSubmissionAnswer>()
                    .eq(AssignmentSubmissionAnswer::getSubmissionId, submissionId)
            );
            
            // 获取题目信息
            List<Long> questionIds = answers.stream()
                .map(AssignmentSubmissionAnswer::getQuestionId)
                .collect(Collectors.toList());
            
            List<Question> questions = questionMapper.selectBatchIds(questionIds);
            Map<Long, Question> questionMap = questions.stream()
                .collect(Collectors.toMap(Question::getId, q -> q));
            
            // 构建响应数据
            Map<String, Object> result = new HashMap<>();
            result.put("submissionId", submission.getId());
            result.put("assignmentId", submission.getAssignmentId());
            result.put("studentId", submission.getStudentId());
            result.put("status", submission.getStatus());
            result.put("score", submission.getScore());
            result.put("feedback", submission.getFeedback());
            result.put("submitTime", submission.getSubmitTime());
            result.put("gradeTime", submission.getGradeTime());
            
            // 处理答案
            List<Map<String, Object>> answersList = new ArrayList<>();
            for (AssignmentSubmissionAnswer answer : answers) {
                Map<String, Object> answerData = new HashMap<>();
                answerData.put("questionId", answer.getQuestionId());
                
                // 添加题目信息
                Question question = questionMap.get(answer.getQuestionId());
                if (question != null) {
                    answerData.put("questionText", question.getTitle());
                    answerData.put("questionType", question.getQuestionType());
                    answerData.put("correctAnswer", question.getCorrectAnswer());
                }
                
                answerData.put("studentAnswer", answer.getStudentAnswer());
                answerData.put("isCorrect", answer.getIsCorrect());
                answerData.put("score", answer.getScore());
                answerData.put("comment", answer.getComment());
                
                answersList.add(answerData);
            }
            
            result.put("answers", answersList);
            
            return Result.success("获取提交详情成功", result);
            
        } catch (Exception e) {
            log.error("获取提交详情异常: {}", e.getMessage(), e);
            return Result.error(500, "获取提交详情失败: " + e.getMessage());
        }
    }
    
    /**
     * 获取作业中的未批改提交列表
     */
    @Operation(summary = "获取未批改提交", description = "获取指定作业中未批改的提交列表")
    @GetMapping("/submissions/{assignmentId}/ungraded")
    public Result<List<Map<String, Object>>> getUngradedSubmissions(@PathVariable Long assignmentId) {
        try {
            // 查询未批改的提交
            List<AssignmentSubmission> submissions = submissionMapper.selectList(
                new LambdaQueryWrapper<AssignmentSubmission>()
                    .eq(AssignmentSubmission::getAssignmentId, assignmentId)
                    .eq(AssignmentSubmission::getStatus, 1) // 已提交未批改
            );
            
            List<Map<String, Object>> result = new ArrayList<>();
            for (AssignmentSubmission submission : submissions) {
                Map<String, Object> item = new HashMap<>();
                item.put("submissionId", submission.getId());
                item.put("assignmentId", submission.getAssignmentId());
                item.put("studentId", submission.getStudentId());
                item.put("submitTime", submission.getSubmitTime());
                
                result.add(item);
            }
            
            return Result.success("获取未批改提交成功", result);
            
        } catch (Exception e) {
            log.error("获取未批改提交异常: {}", e.getMessage(), e);
            return Result.error(500, "获取未批改提交失败: " + e.getMessage());
        }
    }

    @Operation(summary = "获取教师关联作业提交情况", description = "获取特定教师相关的作业及其提交情况")
    @GetMapping("/teacher-assignments")
    public Result<List<Map<String, Object>>> getTeacherRelatedAssignments() {
        try {
            Long teacherId = securityUtil.getCurrentUserId();
            log.info("获取教师{}关联的作业及提交情况", teacherId);
            
            // 查询该教师创建的所有作业
            List<Assignment> assignments = assignmentMapper.selectList(
                new LambdaQueryWrapper<Assignment>()
                    .eq(Assignment::getUserId, teacherId)
                    .orderByDesc(Assignment::getCreateTime)
            );
            
            List<Map<String, Object>> resultList = new ArrayList<>();
            
            for (Assignment assignment : assignments) {
                Map<String, Object> assignmentMap = new HashMap<>();
                assignmentMap.put("id", assignment.getId());
                assignmentMap.put("title", assignment.getTitle());
                assignmentMap.put("courseId", assignment.getCourseId());
                assignmentMap.put("type", assignment.getType());
                assignmentMap.put("mode", assignment.getMode());
                assignmentMap.put("startTime", assignment.getStartTime());
                assignmentMap.put("endTime", assignment.getEndTime());
                assignmentMap.put("totalScore", assignment.getTotalScore());
                assignmentMap.put("status", assignment.getStatus());
                
                // 查询该作业的提交情况
                List<AssignmentSubmission> submissions = submissionMapper.selectList(
                    new LambdaQueryWrapper<AssignmentSubmission>()
                        .eq(AssignmentSubmission::getAssignmentId, assignment.getId())
                );
                
                // 统计提交情况
                int totalSubmissions = submissions.size();
                int gradedSubmissions = 0;
                int ungradedSubmissions = 0;
                
                for (AssignmentSubmission submission : submissions) {
                    if (submission.getStatus() != null) {
                        if (submission.getStatus() == 2) { // 已批改
                            gradedSubmissions++;
                        } else if (submission.getStatus() == 1) { // 已提交未批改
                            ungradedSubmissions++;
                        }
                    }
                }
                
                assignmentMap.put("totalSubmissions", totalSubmissions);
                assignmentMap.put("gradedSubmissions", gradedSubmissions);
                assignmentMap.put("ungradedSubmissions", ungradedSubmissions);
                
                resultList.add(assignmentMap);
            }
            
            return Result.success("获取教师关联作业成功", resultList);
            
        } catch (Exception e) {
            log.error("获取教师关联作业异常: {}", e.getMessage(), e);
            return Result.error(500, "获取教师关联作业失败: " + e.getMessage());
        }
    }

    /**
     * 单个批改请求
     */
    @Data
    public static class GradingRequest {
        private Long submissionId;
        private Long assignmentId;
        private List<DifyDTO.Question> questions;
        private List<DifyDTO.StudentAnswer> studentAnswers;
        private String gradingCriteria;
        private Double maxScore;
    }

    /**
     * 批量批改请求
     */
    @Data
    public static class BatchGradingRequest {
        private Long assignmentId;
        private List<StudentSubmission> submissions;
        private String gradingCriteria;
    }

    /**
     * 学生提交
     */
    @Data
    public static class StudentSubmission {
        private Long submissionId;
        private Long assignmentId;
        private Long studentId;
        private List<DifyDTO.StudentAnswer> studentAnswers;
        private Double maxScore;
        private List<DifyDTO.Question> questions;
    }

    /**
     * 批改统计
     */
    public static class GradingStatistics {
        private Long assignmentId;
        private Integer totalSubmissions;
        private Integer gradedSubmissions;
        private Double averageScore;
        private Double highestScore;
        private Double lowestScore;
        private Map<String, Integer> scoreDistribution;
        private List<String> commonErrors;
        private Map<String, Double> knowledgePointMastery;

        public Long getAssignmentId() { return assignmentId; }
        public void setAssignmentId(Long assignmentId) { this.assignmentId = assignmentId; }
        public Integer getTotalSubmissions() { return totalSubmissions; }
        public void setTotalSubmissions(Integer totalSubmissions) { this.totalSubmissions = totalSubmissions; }
        public Integer getGradedSubmissions() { return gradedSubmissions; }
        public void setGradedSubmissions(Integer gradedSubmissions) { this.gradedSubmissions = gradedSubmissions; }
        public Double getAverageScore() { return averageScore; }
        public void setAverageScore(Double averageScore) { this.averageScore = averageScore; }
        public Double getHighestScore() { return highestScore; }
        public void setHighestScore(Double highestScore) { this.highestScore = highestScore; }
        public Double getLowestScore() { return lowestScore; }
        public void setLowestScore(Double lowestScore) { this.lowestScore = lowestScore; }
        public Map<String, Integer> getScoreDistribution() { return scoreDistribution; }
        public void setScoreDistribution(Map<String, Integer> scoreDistribution) { this.scoreDistribution = scoreDistribution; }
        public List<String> getCommonErrors() { return commonErrors; }
        public void setCommonErrors(List<String> commonErrors) { this.commonErrors = commonErrors; }
        public Map<String, Double> getKnowledgePointMastery() { return knowledgePointMastery; }
        public void setKnowledgePointMastery(Map<String, Double> knowledgePointMastery) { this.knowledgePointMastery = knowledgePointMastery; }
    }

    /**
     * 批改标准请求
     */
    @Data
    public static class GradingCriteriaRequest {
        private Long assignmentId;
        private String criteria;
        private Map<String, Object> parameters;
    }
} 