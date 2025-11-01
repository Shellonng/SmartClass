package com.education.controller.student;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.education.dto.common.Result;
import com.education.entity.Assignment;
import com.education.entity.AssignmentQuestion;
import com.education.entity.AssignmentSubmission;
import com.education.entity.AssignmentSubmissionAnswer;
import com.education.entity.Question;
import com.education.entity.Student;
import com.education.mapper.AssignmentMapper;
import com.education.mapper.AssignmentQuestionMapper;
import com.education.mapper.AssignmentSubmissionAnswerMapper;
import com.education.mapper.AssignmentSubmissionMapper;
import com.education.mapper.QuestionMapper;
import com.education.mapper.StudentMapper;
import com.education.security.SecurityUtil;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import javax.sql.DataSource;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.io.File;
import java.util.UUID;

@Tag(name = "学生任务接口", description = "学生任务相关接口")
@RestController
@RequestMapping("/api/student/assignments")
public class StudentAssignmentController {
    
    private static final Logger logger = LoggerFactory.getLogger(StudentAssignmentController.class);
    
    @Autowired
    private AssignmentMapper assignmentMapper;
    
    @Autowired
    private AssignmentQuestionMapper assignmentQuestionMapper;
    
    @Autowired
    private AssignmentSubmissionMapper assignmentSubmissionMapper;
    
    @Autowired
    private AssignmentSubmissionAnswerMapper assignmentSubmissionAnswerMapper;
    
    @Autowired
    private QuestionMapper questionMapper;
    
    @Autowired
    private StudentMapper studentMapper;
    
    @Autowired
    private DataSource dataSource;
    
    @Autowired
    private SecurityUtil securityUtil;
    
    /**
     * 获取学生任务列表
     * @param status 状态筛选（可选）：pending-待完成，completed-已完成，overdue-已逾期，不传则查询全部
     * @param courseId 课程ID筛选（可选）
     * @param type 任务类型筛选（可选）：homework-课后任务，exam-考试，project-项目任务，report-实验报告
     * @param keyword 关键词搜索（可选）
     * @return 任务列表
     */
    @Operation(summary = "获取学生任务列表", description = "获取学生的任务列表，支持多种筛选条件")
    @GetMapping
    public Result<List<Map<String, Object>>> getAssignmentList(
            @Parameter(description = "状态筛选") @RequestParam(required = false) String status,
            @Parameter(description = "课程ID筛选") @RequestParam(required = false) Long courseId,
            @Parameter(description = "任务类型筛选") @RequestParam(required = false) String type,
            @Parameter(description = "关键词搜索") @RequestParam(required = false) String keyword) {
        
        logger.info("获取学生任务列表，状态: {}, 课程ID: {}, 类型: {}, 关键词: {}", status, courseId, type, keyword);
        
        try {
            // 获取当前登录用户ID
            Long currentUserId = securityUtil.getCurrentUserId();
            if (currentUserId == null) {
                return Result.error("未登录或登录已过期");
            }
            
            // 查询学生信息 - 仅用于验证学生身份
            Student student = null;
            try {
                student = studentMapper.selectOne(
                    new LambdaQueryWrapper<Student>()
                        .eq(Student::getUserId, currentUserId)
                        .last("LIMIT 1") // 限制只返回一条记录
                );
                
                if (student == null) {
                    logger.error("学生信息不存在，用户ID: {}", currentUserId);
                    return Result.error("学生信息不存在");
                }
                
                logger.info("查询到学生信息: {}", student);
            } catch (Exception e) {
                logger.error("查询学生信息异常: {}", e.getMessage(), e);
                
                // 如果是因为找到多条记录导致的错误，则查询列表并使用第一条
                if (e.getMessage() != null && e.getMessage().contains("Expected one result")) {
                    List<Student> students = studentMapper.selectList(
                        new LambdaQueryWrapper<Student>()
                            .eq(Student::getUserId, currentUserId)
                    );
                    
                    if (students != null && !students.isEmpty()) {
                        student = students.get(0);
                        logger.warn("发现多条学生记录，使用第一条: {}", student);
                    } else {
                        return Result.error("学生信息不存在");
                    }
                } else {
                    return Result.error("查询学生信息失败: " + e.getMessage());
                }
            }
            
            // 查询学生可见的所有任务（已发布的）
            String sql = "SELECT a.*, c.title AS course_name, u.username AS teacher_name, " +
                         "s.status AS submission_status, s.score, s.submit_time " +
                         "FROM assignment a " +
                         "JOIN course c ON a.course_id = c.id " +
                         "JOIN user u ON a.user_id = u.id " +
                         "JOIN course_student cs ON c.id = cs.course_id " +
                         "LEFT JOIN assignment_submission s ON a.id = s.assignment_id AND s.student_id = ? " +
                         "WHERE a.status = 1 AND cs.student_id = ? ";
            
            // 根据筛选条件添加WHERE子句
            List<Object> params = new ArrayList<>();
            params.add(student.getId());
            params.add(student.getId());
            
            if (courseId != null) {
                sql += "AND a.course_id = ? ";
                params.add(courseId);
            }
            
            if (type != null && !type.isEmpty()) {
                sql += "AND a.type = ? ";
                params.add(type);
            }
            
            if (keyword != null && !keyword.isEmpty()) {
                sql += "AND (a.title LIKE ? OR a.description LIKE ? OR c.title LIKE ?) ";
                String likeKeyword = "%" + keyword + "%";
                params.add(likeKeyword);
                params.add(likeKeyword);
                params.add(likeKeyword);
            }
            
            // 根据状态筛选
            if (status != null && !status.isEmpty()) {
                Date now = new Date();
                
                if ("pending".equals(status)) {
                    // 待完成：submission状态为0且未过期
                    sql += "AND (s.status = 0 OR s.status IS NULL) AND (a.end_time IS NULL OR a.end_time > ?) ";
                    params.add(now);
                } else if ("completed".equals(status)) {
                    // 已完成：submission状态为1（已提交未批改）或2（已批改）
                    sql += "AND (s.status = 1 OR s.status = 2) ";
                } else if ("overdue".equals(status)) {
                    // 已逾期：submission状态为0且已过期
                    sql += "AND (s.status = 0 OR s.status IS NULL) AND a.end_time < ? ";
                    params.add(now);
                }
            }
            
            // 按截止时间升序排序
            sql += "ORDER BY a.end_time ASC";
            
            List<Map<String, Object>> assignments = new ArrayList<>();
            
            try (Connection conn = dataSource.getConnection();
                 PreparedStatement stmt = conn.prepareStatement(sql)) {
                
                // 设置参数
                for (int i = 0; i < params.size(); i++) {
                    stmt.setObject(i + 1, params.get(i));
                }
                
                try (ResultSet rs = stmt.executeQuery()) {
                    while (rs.next()) {
                        Map<String, Object> assignment = new HashMap<>();
                        
                        // 基本信息
                        assignment.put("id", rs.getLong("id"));
                        assignment.put("title", rs.getString("title"));
                        assignment.put("description", rs.getString("description"));
                        assignment.put("type", rs.getString("type"));
                        assignment.put("courseId", rs.getLong("course_id"));
                        assignment.put("courseName", rs.getString("course_name"));
                        assignment.put("teacherName", rs.getString("teacher_name"));
                        assignment.put("createTime", rs.getTimestamp("create_time"));
                        assignment.put("deadline", rs.getTimestamp("end_time"));
                        assignment.put("startTime", rs.getTimestamp("start_time"));
                        
                        // 计算优先级
                        Date deadline = rs.getTimestamp("end_time");
                        Date now = new Date();
                        String priority = "low";
                        
                        if (deadline != null) {
                            long diff = deadline.getTime() - now.getTime();
                            long diffHours = diff / (60 * 60 * 1000);
                            
                            if (diff < 0) {
                                priority = "overdue";
                            } else if (diffHours <= 24) {
                                priority = "high";
                            } else if (diffHours <= 72) {
                                priority = "medium";
                            }
                        }
                        
                        assignment.put("priority", priority);
                        
                        // 提交状态
                        Integer submissionStatus = rs.getObject("submission_status") != null ? rs.getInt("submission_status") : null;
                        Date endTime = rs.getTimestamp("end_time");
                        
                        String statusValue;
                        if (submissionStatus != null && (submissionStatus == 1 || submissionStatus == 2)) {
                            // 状态为1（已提交未批改）或2（已批改）时，显示为已完成
                            statusValue = "completed";
                            assignment.put("submissionTime", rs.getTimestamp("submit_time"));
                            assignment.put("score", rs.getObject("score"));
                            
                            // 如果是已批改状态，添加标记
                            if (submissionStatus == 2) {
                                assignment.put("isGraded", true);
                            }
                        } else if (endTime != null && now.after(endTime)) {
                            // 未提交且已过期
                            statusValue = "overdue";
                        } else {
                            // 未提交且未过期
                            statusValue = "pending";
                        }
                        
                        assignment.put("status", statusValue);
                        
                        // 总分
                        String scoreSql = "SELECT SUM(score) FROM assignment_question WHERE assignment_id = ?";
                        try (PreparedStatement scoreStmt = conn.prepareStatement(scoreSql)) {
                            scoreStmt.setLong(1, rs.getLong("id"));
                            try (ResultSet scoreRs = scoreStmt.executeQuery()) {
                                if (scoreRs.next()) {
                                    assignment.put("totalScore", scoreRs.getInt(1));
                                } else {
                                    assignment.put("totalScore", 100);
                                }
                            }
                        }
                        
                        assignments.add(assignment);
                    }
                }
            }
            
            return Result.success(assignments);
            
        } catch (Exception e) {
            logger.error("获取学生任务列表失败: {}", e.getMessage(), e);
            return Result.error("获取任务列表失败: " + e.getMessage());
        }
    }
    
    /**
     * 获取任务详情
     * @param id 任务ID
     * @return 任务详情
     */
    @Operation(summary = "获取任务详情", description = "获取任务的详细信息")
    @GetMapping("/{id}")
    public Result<Map<String, Object>> getAssignmentDetail(
            @Parameter(description = "任务ID") @PathVariable Long id) {
        
        logger.info("获取学生任务详情，任务ID: {}", id);
        
        try {
            // 查询任务信息
            Assignment assignment = assignmentMapper.selectById(id);
            
            if (assignment == null) {
                logger.error("任务不存在，任务ID: {}", id);
                return Result.error("任务不存在");
            }
            
            logger.info("成功查询到任务信息: {}", assignment);
            
            // 获取当前登录用户ID
            Long currentUserId = securityUtil.getCurrentUserId();
            if (currentUserId == null) {
                return Result.error("未登录或登录已过期");
            }
            
            // 查询学生信息 - 仅用于验证学生身份
            Student student = null;
            try {
                student = studentMapper.selectOne(
                new LambdaQueryWrapper<Student>()
                    .eq(Student::getUserId, currentUserId)
                        .last("LIMIT 1") // 限制只返回一条记录
            );
            
            if (student == null) {
                logger.error("学生信息不存在，用户ID: {}", currentUserId);
                return Result.error("学生信息不存在");
                }
                
                logger.info("查询到学生信息: {}", student);
            } catch (Exception e) {
                logger.error("查询学生信息异常: {}", e.getMessage(), e);
                
                // 如果是因为找到多条记录导致的错误，则查询列表并使用第一条
                if (e.getMessage() != null && e.getMessage().contains("Expected one result")) {
                    List<Student> students = studentMapper.selectList(
                        new LambdaQueryWrapper<Student>()
                            .eq(Student::getUserId, currentUserId)
                    );
                    
                    if (students != null && !students.isEmpty()) {
                        student = students.get(0);
                        logger.warn("发现多条学生记录，使用第一条: {}", student);
                    } else {
                        return Result.error("学生信息不存在");
                    }
                } else {
                    return Result.error("查询学生信息失败: " + e.getMessage());
                }
            }
            
            // 获取课程和教师信息
            String sql = "SELECT c.title AS course_name, u.username AS teacher_name " +
                         "FROM course c " +
                         "JOIN user u ON c.teacher_id = u.id " +
                         "WHERE c.id = ?";
            
            try (Connection conn = dataSource.getConnection();
                 PreparedStatement stmt = conn.prepareStatement(sql)) {
                stmt.setLong(1, assignment.getCourseId());
                
                try (ResultSet rs = stmt.executeQuery()) {
                    if (rs.next()) {
                        assignment.setCourseName(rs.getString("course_name"));
                        assignment.setTeacherName(rs.getString("teacher_name"));
                    }
                }
            }
            
            // 检查是否已经有提交记录，如果没有则创建一个状态为未提交的记录
            AssignmentSubmission submission = assignmentSubmissionMapper.selectOne(
                new LambdaQueryWrapper<AssignmentSubmission>()
                    .eq(AssignmentSubmission::getAssignmentId, id)
                    .eq(AssignmentSubmission::getStudentId, currentUserId)
            );
            
            if (submission == null) {
                submission = new AssignmentSubmission();
                submission.setAssignmentId(id);
                submission.setStudentId(currentUserId);
                submission.setStatus(0); // 未提交
                submission.setCreateTime(new Date());
                assignmentSubmissionMapper.insert(submission);
                
                logger.info("为学生创建任务提交记录，学生ID: {}, 任务ID: {}", currentUserId, id);
            }
            
            // 返回任务详情和提交状态
            Map<String, Object> result = new HashMap<String, Object>();
            result.put("assignment", assignment);
            result.put("submission", submission);
            
            return Result.success(result);
            
        } catch (Exception e) {
            logger.error("获取任务详情失败: {}", e.getMessage(), e);
            return Result.error("获取任务详情失败: " + e.getMessage());
        }
    }
    
    /**
     * 获取任务题目
     * @param id 任务ID
     * @return 任务题目列表
     */
    @Operation(summary = "获取任务题目", description = "获取任务的题目列表")
    @GetMapping("/{id}/questions")
    public Result<List<Map<String, Object>>> getAssignmentQuestions(
            @Parameter(description = "任务ID") @PathVariable Long id) {
        
        logger.info("获取学生任务题目，任务ID: {}", id);
        
        try {
            // 查询任务信息
            Assignment assignment = assignmentMapper.selectById(id);
            
            if (assignment == null) {
                logger.error("任务不存在，任务ID: {}", id);
                return Result.error("任务不存在");
            }
            
            // 检查任务是否已发布
            if (assignment.getStatus() != 1) {
                logger.error("任务未发布，不能查看题目，任务ID: {}", id);
                return Result.error("任务未发布，不能查看题目");
            }
            
            // 检查是否是考试类型，且考试未开始
            if ("exam".equals(assignment.getType()) && 
                assignment.getStartTime() != null && 
                new Date().before(assignment.getStartTime())) {
                logger.error("考试未开始，不能查看题目，任务ID: {}", id);
                return Result.error("考试未开始，不能查看题目");
            }
            
            // 查询任务题目
            List<Map<String, Object>> questions = new ArrayList<>();
            
            // 1. 查询任务-题目关联表
            String sql = "SELECT aq.question_id, aq.score, aq.sequence, " +
                         "q.title, q.question_type, q.difficulty, q.correct_answer, q.explanation, q.knowledge_point " +
                         "FROM assignment_question aq " +
                         "JOIN question q ON aq.question_id = q.id " +
                         "WHERE aq.assignment_id = ? " +
                         "ORDER BY aq.sequence";
            
            try (Connection conn = dataSource.getConnection();
                 PreparedStatement stmt = conn.prepareStatement(sql)) {
                stmt.setLong(1, id);
                
                try (ResultSet rs = stmt.executeQuery()) {
                    while (rs.next()) {
                        Map<String, Object> question = new HashMap<>();
                        
                        Long questionId = rs.getLong("question_id");
                        String questionType = rs.getString("question_type");
                        
                        question.put("id", questionId);
                        question.put("title", rs.getString("title"));
                        question.put("questionType", questionType);
                        question.put("difficulty", rs.getInt("difficulty"));
                        question.put("score", rs.getInt("score"));
                        question.put("sequence", rs.getInt("sequence"));
                        question.put("knowledgePoint", rs.getString("knowledge_point"));
                        
                        // 对于学生端，不返回正确答案和解析
                        // question.put("correctAnswer", rs.getString("correct_answer"));
                        // question.put("explanation", rs.getString("explanation"));
                        
                        // 如果是选择题或判断题，需要获取选项
                        if ("single".equals(questionType) || "multiple".equals(questionType) || "true_false".equals(questionType)) {
                            List<Map<String, Object>> options = getQuestionOptions(questionId);
                            question.put("options", options);
                        }
                        
                        questions.add(question);
                    }
                }
            }
            
            // 按照题型分组
            Map<String, List<Map<String, Object>>> groupedQuestions = new HashMap<>();
            
            for (Map<String, Object> question : questions) {
                String type = (String) question.get("questionType");
                if (!groupedQuestions.containsKey(type)) {
                    groupedQuestions.put(type, new ArrayList<>());
                }
                groupedQuestions.get(type).add(question);
            }
            
            // 构建返回数据
            List<Map<String, Object>> result = new ArrayList<>();
            
            // 题型顺序：单选题、多选题、判断题、填空题、简答题
            String[] typeOrder = {"single", "multiple", "true_false", "blank", "short", "code"};
            
            for (String type : typeOrder) {
                if (groupedQuestions.containsKey(type)) {
                    Map<String, Object> group = new HashMap<>();
                    List<Map<String, Object>> typeQuestions = groupedQuestions.get(type);
                    
                    group.put("type", type);
                    group.put("questions", typeQuestions);
                    
                    // 计算总分
                    int totalScore = 0;
                    for (Map<String, Object> q : typeQuestions) {
                        totalScore += (int) q.get("score");
                    }
                    group.put("totalScore", totalScore);
                    
                    result.add(group);
                }
            }
            
            return Result.success(result);
            
        } catch (Exception e) {
            logger.error("获取任务题目失败: {}", e.getMessage(), e);
            return Result.error("获取任务题目失败: " + e.getMessage());
        }
    }
    
    /**
     * 获取题目选项
     * @param questionId 题目ID
     * @return 选项列表
     */
    private List<Map<String, Object>> getQuestionOptions(Long questionId) throws SQLException {
        List<Map<String, Object>> options = new ArrayList<>();
        
        String sql = "SELECT id, option_label as optionKey, option_text as content " +
                     "FROM question_option WHERE question_id = ? ORDER BY option_label";
        
        try (Connection conn = dataSource.getConnection();
             PreparedStatement stmt = conn.prepareStatement(sql)) {
            stmt.setLong(1, questionId);
            
            try (ResultSet rs = stmt.executeQuery()) {
                while (rs.next()) {
                    Map<String, Object> option = new HashMap<>();
                    option.put("id", rs.getLong("id"));
                    option.put("optionKey", rs.getString("optionKey"));
                    option.put("content", rs.getString("content"));
                    options.add(option);
                }
            }
        }
        
        return options;
    }
    
    /**
     * 保存单题答案
     * @param id 任务ID
     * @param questionId 题目ID
     * @param request 答案数据
     * @return 保存结果
     */
    @Operation(summary = "保存单题答案", description = "保存单个题目的答案")
    @PostMapping("/{id}/questions/{questionId}/save")
    @Transactional
    public Result<Map<String, Object>> saveQuestionAnswer(
            @Parameter(description = "任务ID") @PathVariable Long id,
            @Parameter(description = "题目ID") @PathVariable Long questionId,
            @RequestBody Map<String, Object> request) {
        
        logger.info("保存单题答案，任务ID: {}, 题目ID: {}", id, questionId);
        
        try {
            // 获取当前登录用户ID
            Long currentUserId = securityUtil.getCurrentUserId();
            if (currentUserId == null) {
                return Result.error("未登录或登录已过期");
            }
            
            // 查询学生ID
            Student student = studentMapper.selectOne(
                new LambdaQueryWrapper<Student>()
                    .eq(Student::getUserId, currentUserId)
            );
            
            if (student == null) {
                logger.error("学生信息不存在，用户ID: {}", currentUserId);
                return Result.error("学生信息不存在");
            }
            
            // 查询任务信息
            Assignment assignment = assignmentMapper.selectById(id);
            
            if (assignment == null) {
                logger.error("任务不存在，任务ID: {}", id);
                return Result.error("任务不存在");
            }
            
            // 检查任务是否已发布
            if (assignment.getStatus() != 1) {
                logger.error("任务未发布，不能保存答案，任务ID: {}", id);
                return Result.error("任务未发布，不能保存答案");
            }
            
            // 检查是否已过截止时间
            if (assignment.getEndTime() != null && new Date().after(assignment.getEndTime())) {
                logger.error("任务已截止，不能保存答案，任务ID: {}", id);
                return Result.error("任务已截止，不能保存答案");
            }
            
            // 检查题目是否属于该任务
            AssignmentQuestion assignmentQuestion = assignmentQuestionMapper.selectOne(
                new LambdaQueryWrapper<AssignmentQuestion>()
                    .eq(AssignmentQuestion::getAssignmentId, id)
                    .eq(AssignmentQuestion::getQuestionId, questionId)
            );
            
            if (assignmentQuestion == null) {
                logger.error("题目不属于该任务，任务ID: {}, 题目ID: {}", id, questionId);
                return Result.error("题目不属于该任务");
            }
            
            // 查询提交记录
            AssignmentSubmission submission = assignmentSubmissionMapper.selectOne(
                new LambdaQueryWrapper<AssignmentSubmission>()
                    .eq(AssignmentSubmission::getAssignmentId, id)
                    .eq(AssignmentSubmission::getStudentId, currentUserId)
            );
            
            // 如果没有提交记录，则创建一个
            if (submission == null) {
                submission = new AssignmentSubmission();
                submission.setAssignmentId(id);
                submission.setStudentId(currentUserId);
                submission.setStatus(0); // 未提交
                submission.setCreateTime(new Date());
                assignmentSubmissionMapper.insert(submission);
                
                logger.info("为学生创建任务提交记录，学生ID: {}, 任务ID: {}", currentUserId, id);
            } else if (submission.getStatus() == 1) {
                // 如果已提交，不能再保存答案
                logger.error("任务已提交，不能再保存答案，任务ID: {}", id);
                return Result.error("任务已提交，不能再保存答案");
            }
            
            // 获取答案内容
            Object answerObj = request.get("answer");
            String answerContent;
            
            // 处理不同类型的答案
            if (answerObj instanceof List) {
                // 多选题答案是数组，转为逗号分隔的字符串
                // Add SuppressWarnings annotation to fix the unchecked cast warning
                @SuppressWarnings("unchecked")
                List<String> answerList = (List<String>) answerObj;
                answerContent = String.join(",", answerList);
            } else {
                answerContent = String.valueOf(answerObj);
            }
            
            // 查询题目信息
            Question question = questionMapper.selectById(questionId);
            
            // 判断答案是否正确
            boolean isCorrect = false;
            if (question != null && question.getCorrectAnswer() != null) {
                if ("multiple".equals(question.getQuestionType())) {
                    // 多选题答案可能顺序不同，需要特殊处理
                    String[] correctOptions = question.getCorrectAnswer().split(",");
                    String[] studentOptions = answerContent.split(",");
                    
                    // 排序后比较
                    Arrays.sort(correctOptions);
                    Arrays.sort(studentOptions);
                    
                    isCorrect = Arrays.equals(correctOptions, studentOptions);
                } else {
                    // 其他题型直接比较
                    isCorrect = question.getCorrectAnswer().equalsIgnoreCase(answerContent);
                }
            }
            
            // 查询是否已有该题目的答案记录
            AssignmentSubmissionAnswer existingAnswer = assignmentSubmissionAnswerMapper.selectOne(
                new LambdaQueryWrapper<AssignmentSubmissionAnswer>()
                    .eq(AssignmentSubmissionAnswer::getSubmissionId, submission.getId())
                    .eq(AssignmentSubmissionAnswer::getQuestionId, questionId)
            );
            
            if (existingAnswer != null) {
                // 更新已有答案
                existingAnswer.setStudentAnswer(answerContent);
                existingAnswer.setIsCorrect(isCorrect);
                existingAnswer.setScore(isCorrect ? assignmentQuestion.getScore() : 0);
                existingAnswer.setUpdateTime(new Date());
                assignmentSubmissionAnswerMapper.updateById(existingAnswer);
                
                logger.info("更新题目答案，提交ID: {}, 题目ID: {}", submission.getId(), questionId);
            } else {
                // 创建新答案记录
                AssignmentSubmissionAnswer answer = new AssignmentSubmissionAnswer();
                answer.setSubmissionId(submission.getId());
                answer.setQuestionId(questionId);
                answer.setStudentAnswer(answerContent);
                answer.setIsCorrect(isCorrect);
                answer.setScore(isCorrect ? assignmentQuestion.getScore() : 0);
                answer.setCreateTime(new Date());
                answer.setUpdateTime(new Date());
                assignmentSubmissionAnswerMapper.insert(answer);
                
                logger.info("保存题目答案，提交ID: {}, 题目ID: {}", submission.getId(), questionId);
            }
            
            return Result.success("保存成功");
            
        } catch (Exception e) {
            logger.error("保存题目答案失败: {}", e.getMessage(), e);
            return Result.error("保存题目答案失败: " + e.getMessage());
        }
    }
    
    /**
     * 提交任务/考试答案
     * @param id 任务/考试ID
     * @return 提交结果
     */
    @Operation(summary = "提交任务/考试答案", description = "提交任务/考试的答案")
    @PostMapping("/{id}/submit")
    @Transactional
    public Result<Map<String, Object>> submitAssignment(
            @Parameter(description = "任务ID") @PathVariable Long id) {
        
        logger.info("提交任务/考试答案，任务ID: {}", id);
        
        try {
            // 获取当前登录用户ID
            Long currentUserId = securityUtil.getCurrentUserId();
            if (currentUserId == null) {
                return Result.error("未登录或登录已过期");
            }
            
            // 查询学生ID
            Student student = studentMapper.selectOne(
                new LambdaQueryWrapper<Student>()
                    .eq(Student::getUserId, currentUserId)
            );
            
            if (student == null) {
                logger.error("学生信息不存在，用户ID: {}", currentUserId);
                return Result.error("学生信息不存在");
            }
            
            // 查询任务信息
            Assignment assignment = assignmentMapper.selectById(id);
            
            if (assignment == null) {
                logger.error("任务不存在，任务ID: {}", id);
                return Result.error("任务不存在");
            }
            
            // 检查任务是否已发布
            if (assignment.getStatus() != 1) {
                logger.error("任务未发布，不能提交，任务ID: {}", id);
                return Result.error("任务未发布，不能提交");
            }
            
            // 检查是否已过截止时间
            if (assignment.getEndTime() != null && new Date().after(assignment.getEndTime())) {
                logger.error("任务已截止，不能提交，任务ID: {}", id);
                return Result.error("任务已截止，不能提交");
            }
            
            // 查询提交记录
            AssignmentSubmission submission = assignmentSubmissionMapper.selectOne(
                new LambdaQueryWrapper<AssignmentSubmission>()
                    .eq(AssignmentSubmission::getAssignmentId, id)
                    .eq(AssignmentSubmission::getStudentId, currentUserId)
            );
            
            if (submission == null) {
                logger.error("未找到提交记录，任务ID: {}, 学生ID: {}", id, currentUserId);
                return Result.error("未找到提交记录");
            }
            
            // 检查是否已经提交过
            if (submission.getStatus() == 1) {
                logger.error("已经提交过，不能重复提交，任务ID: {}", id);
                return Result.error("已经提交过，不能重复提交");
            }
            
            // 获取该作业的所有题目
            List<AssignmentQuestion> assignmentQuestions = assignmentQuestionMapper.selectList(
                new LambdaQueryWrapper<AssignmentQuestion>()
                    .eq(AssignmentQuestion::getAssignmentId, id)
            );
            
            // 获取该提交的所有答案
            List<AssignmentSubmissionAnswer> answers = assignmentSubmissionAnswerMapper.selectList(
                new LambdaQueryWrapper<AssignmentSubmissionAnswer>()
                    .eq(AssignmentSubmissionAnswer::getSubmissionId, submission.getId())
            );
            
            // 为选择题、判断题和填空题进行自动评分
            int totalScore = 0;
            
            for (AssignmentSubmissionAnswer answer : answers) {
                // 获取题目信息
                Question question = questionMapper.selectById(answer.getQuestionId());
                
                if (question != null) {
                    // 找到该题在assignmentQuestions中的配置
                    AssignmentQuestion assignmentQuestion = null;
                    for (AssignmentQuestion aq : assignmentQuestions) {
                        if (aq.getQuestionId().equals(question.getId())) {
                            assignmentQuestion = aq;
                            break;
                        }
                    }
                    
                    if (assignmentQuestion == null) {
                        logger.warn("题目不属于该任务，跳过评分，题目ID: {}", question.getId());
                        continue;
                    }
                    
                    // 获取该题在当前作业中的分值
                    Integer questionScore = assignmentQuestion.getScore();
                    
                    // 对选择题、判断题和填空题自动评分
                    String questionType = question.getQuestionType();
                    if ("single".equals(questionType) || "true_false".equals(questionType) || "blank".equals(questionType) || "multiple".equals(questionType)) {
                        String correctAnswer = question.getCorrectAnswer();
                        String studentAnswer = answer.getStudentAnswer();
                        
                        boolean isCorrect = false;
                        
                        if (correctAnswer != null && studentAnswer != null) {
                            if ("multiple".equals(questionType)) {
                                // 多选题答案可能顺序不同，需要特殊处理
                                String[] correctOptions = correctAnswer.split(",");
                                String[] studentOptions = studentAnswer.split(",");
                                
                                // 排序后比较
                                Arrays.sort(correctOptions);
                                Arrays.sort(studentOptions);
                                
                                isCorrect = Arrays.equals(correctOptions, studentOptions);
                            } else {
                                // 其他题型直接比较
                                isCorrect = correctAnswer.equalsIgnoreCase(studentAnswer);
                            }
                        }
                        
                        // 更新答案的正确性和得分
                        answer.setIsCorrect(isCorrect);
                        answer.setScore(isCorrect ? questionScore : 0);
                        assignmentSubmissionAnswerMapper.updateById(answer);
                        
                        // 累计总分
                        if (isCorrect && questionScore != null) {
                            totalScore += questionScore;
                        }
                        
                        logger.info("自动评分 - 题目ID: {}, 类型: {}, 正确答案: {}, 学生答案: {}, 是否正确: {}, 得分: {}",
                                question.getId(), questionType, correctAnswer, studentAnswer, isCorrect, 
                                isCorrect ? questionScore : 0);
                    }
                }
            }
            
            // 更新提交状态
            submission.setStatus(1); // 已提交未批改
            submission.setSubmitTime(new Date());
            submission.setUpdateTime(new Date());
            submission.setScore(totalScore); // 更新自动评分的分数
            assignmentSubmissionMapper.updateById(submission);
            
            logger.info("提交任务成功，任务ID: {}, 学生ID: {}, 自动评分得分: {}", id, currentUserId, totalScore);
            
            Map<String, Object> result = new HashMap<>();
            result.put("assignmentId", id);
            result.put("studentId", currentUserId);
            result.put("score", totalScore);
            result.put("submitTime", submission.getSubmitTime());
            
            return Result.success(result);
            
        } catch (Exception e) {
            logger.error("提交任务答案失败: {}", e.getMessage(), e);
            return Result.error("提交任务答案失败: " + e.getMessage());
        }
    }
    
    /**
     * 获取已保存的答案
     * @param id 任务ID
     * @param questionId 题目ID
     * @return 已保存的答案
     */
    @Operation(summary = "获取已保存答案", description = "获取学生已保存的答案")
    @GetMapping("/{id}/questions/{questionId}/answer")
    public Result<Map<String, Object>> getSavedAnswer(
            @Parameter(description = "任务ID") @PathVariable Long id,
            @Parameter(description = "题目ID") @PathVariable Long questionId) {
        
        logger.info("获取已保存答案，任务ID: {}, 题目ID: {}", id, questionId);
        
        try {
            // 获取当前登录用户ID
            Long currentUserId = securityUtil.getCurrentUserId();
            if (currentUserId == null) {
                return Result.error("未登录或登录已过期");
            }
            
            // 查询提交记录
            AssignmentSubmission submission = assignmentSubmissionMapper.selectOne(
                new LambdaQueryWrapper<AssignmentSubmission>()
                    .eq(AssignmentSubmission::getAssignmentId, id)
                    .eq(AssignmentSubmission::getStudentId, currentUserId)
            );
            
            if (submission == null) {
                logger.info("未找到提交记录，任务ID: {}, 学生ID: {}", id, currentUserId);
                return Result.success(null); // 返回空答案
            }
            
            // 查询已保存的答案
            AssignmentSubmissionAnswer answer = assignmentSubmissionAnswerMapper.selectOne(
                new LambdaQueryWrapper<AssignmentSubmissionAnswer>()
                    .eq(AssignmentSubmissionAnswer::getSubmissionId, submission.getId())
                    .eq(AssignmentSubmissionAnswer::getQuestionId, questionId)
            );
            
            if (answer == null) {
                logger.info("未找到已保存答案，提交ID: {}, 题目ID: {}", submission.getId(), questionId);
                return Result.success(null); // 返回空答案
            }
            
            // 构造返回数据
            Map<String, Object> result = new HashMap<>();
            result.put("answer", answer.getStudentAnswer());
            result.put("isCorrect", answer.getIsCorrect());
            result.put("score", answer.getScore());
            
            return Result.success(result);
            
        } catch (Exception e) {
            logger.error("获取已保存答案失败: {}", e.getMessage(), e);
            return Result.error("获取已保存答案失败: " + e.getMessage());
        }
    }
    
    /**
     * 提交文件作业
     * @param id 任务ID
     * @param file 上传的文件
     * @return 提交结果
     */
    @Operation(summary = "提交文件作业", description = "提交文件类型的作业")
    @PostMapping("/{id}/submit-file")
    @Transactional
    public Result<Map<String, Object>> submitAssignmentFile(
            @Parameter(description = "任务ID") @PathVariable Long id,
            @RequestParam("file") MultipartFile file) {
        
        logger.info("提交文件作业，任务ID: {}, 文件名: {}", id, file.getOriginalFilename());
        
        try {
            // 获取当前登录用户ID
            Long currentUserId = securityUtil.getCurrentUserId();
            if (currentUserId == null) {
                return Result.error("未登录或登录已过期");
            }
            
            logger.info("当前用户ID: {}", currentUserId);
            
            // 查询学生信息 - 仅用于验证学生身份
            Student student = null;
            try {
                student = studentMapper.selectOne(
                    new LambdaQueryWrapper<Student>()
                        .eq(Student::getUserId, currentUserId)
                        .last("LIMIT 1") // 限制只返回一条记录
                );
                
                if (student == null) {
                    logger.error("学生信息不存在，用户ID: {}", currentUserId);
                    return Result.error("学生信息不存在");
                }
                
                logger.info("查询到学生信息: {}", student);
            } catch (Exception e) {
                logger.error("查询学生信息异常: {}", e.getMessage(), e);
                
                // 如果是因为找到多条记录导致的错误，则查询列表并使用第一条
                if (e.getMessage() != null && e.getMessage().contains("Expected one result")) {
                    List<Student> students = studentMapper.selectList(
                        new LambdaQueryWrapper<Student>()
                            .eq(Student::getUserId, currentUserId)
                    );
                    
                    if (students != null && !students.isEmpty()) {
                        student = students.get(0);
                        logger.warn("发现多条学生记录，使用第一条: {}", student);
                    } else {
                        return Result.error("学生信息不存在");
                    }
                } else {
                    return Result.error("查询学生信息失败: " + e.getMessage());
                }
            }
            
            // 查询任务信息
            Assignment assignment = assignmentMapper.selectById(id);
            
            if (assignment == null) {
                logger.error("任务不存在，任务ID: {}", id);
                return Result.error("任务不存在");
            }
            
            // 检查任务是否为文件提交类型
            if (!"file".equals(assignment.getMode())) {
                logger.error("任务不是文件提交类型，任务ID: {}, 类型: {}", id, assignment.getMode());
                return Result.error("该任务不是文件提交类型");
            }
            
            // 检查任务是否已发布
            if (assignment.getStatus() != 1) {
                logger.error("任务未发布，不能提交，任务ID: {}", id);
                return Result.error("任务未发布，不能提交");
            }
            
            // 检查截止时间
            if (assignment.getEndTime() != null && new Date().after(assignment.getEndTime())) {
                logger.error("任务已截止，不能提交，任务ID: {}", id);
                return Result.error("任务已截止，不能提交");
            }
            
            // 保存文件
            String originalFilename = file.getOriginalFilename();
            String fileExtension = originalFilename.substring(originalFilename.lastIndexOf("."));
            String newFilename = UUID.randomUUID().toString() + fileExtension;

            // 修改文件保存路径为项目根目录下的resource/file
            String uploadDir = "D:/my_git_code/SmartClass/resource/file/assignments/" + id + "/" + currentUserId + "/";
            File dir = new File(uploadDir);
            if (!dir.exists()) {
                boolean created = dir.mkdirs();
                if (!created) {
                    logger.error("创建目录失败: {}", uploadDir);
                    return Result.error("创建上传目录失败，请联系管理员");
                }
                logger.info("成功创建目录: {}", uploadDir);
            }

            File destFile = new File(dir, newFilename);
            logger.info("保存文件路径: {}", destFile.getAbsolutePath());
            file.transferTo(destFile);
            
            // 更新提交记录
            AssignmentSubmission submission = assignmentSubmissionMapper.selectOne(
                new LambdaQueryWrapper<AssignmentSubmission>()
                    .eq(AssignmentSubmission::getAssignmentId, id)
                    .eq(AssignmentSubmission::getStudentId, currentUserId)
            );
            
            if (submission == null) {
                submission = new AssignmentSubmission();
                submission.setAssignmentId(id);
                submission.setStudentId(currentUserId);
                submission.setCreateTime(new Date());
            }
            
            submission.setStatus(1); // 已提交
            submission.setSubmitTime(new Date());
            submission.setFileName(originalFilename);
            submission.setFilePath(uploadDir + newFilename);
            
            if (submission.getId() == null) {
                assignmentSubmissionMapper.insert(submission);
                logger.info("新增文件作业提交记录，任务ID: {}, 学生ID: {}", id, currentUserId);
            } else {
                assignmentSubmissionMapper.updateById(submission);
                logger.info("更新文件作业提交记录，提交ID: {}, 任务ID: {}, 学生ID: {}", submission.getId(), id, currentUserId);
            }
            
            logger.info("文件作业提交成功，任务ID: {}, 学生ID: {}, 文件名: {}", id, currentUserId, originalFilename);
            
            Map<String, Object> result = new HashMap<>();
            result.put("assignmentId", id);
            result.put("studentId", currentUserId);
            result.put("fileName", originalFilename);
            result.put("filePath", uploadDir + newFilename);
            result.put("submitTime", submission.getSubmitTime());
            
            return Result.success(result);
            
        } catch (Exception e) {
            logger.error("提交文件作业失败: {}", e.getMessage(), e);
            return Result.error("提交作业失败: " + e.getMessage());
        }
    }

    /**
     * 获取作业批改结果
     * @param id 作业提交ID
     * @return 批改结果详情
     */
    @Operation(summary = "获取作业批改结果", description = "获取学生作业的批改结果和教师反馈")
    @GetMapping("/submissions/{submissionId}/results")
    public Result<Map<String, Object>> getAssignmentResults(
            @Parameter(description = "作业提交ID") @PathVariable Long submissionId) {
        
        logger.info("获取作业批改结果，提交ID: {}", submissionId);
        
        try {
            // 获取当前登录用户ID
            Long currentUserId = securityUtil.getCurrentUserId();
            if (currentUserId == null) {
                return Result.error("未登录或登录已过期");
            }
            
            // 查询提交记录
            AssignmentSubmission submission = assignmentSubmissionMapper.selectById(submissionId);
            if (submission == null) {
                return Result.error(404, "未找到提交记录");
            }
            
            // 查询学生ID
            Student student = studentMapper.selectOne(
                new LambdaQueryWrapper<Student>()
                    .eq(Student::getUserId, currentUserId)
                    .last("LIMIT 1")
            );
            
            if (student == null) {
                return Result.error(401, "无权访问");
            }
            
            // 验证提交记录是否属于当前学生
            if (!submission.getStudentId().equals(student.getId())) {
                return Result.error(403, "无权查看该提交记录");
            }
            
            // 查询作业详情
            Assignment assignment = assignmentMapper.selectById(submission.getAssignmentId());
            if (assignment == null) {
                return Result.error(404, "未找到作业信息");
            }
            
            // 查询答案详情
            List<AssignmentSubmissionAnswer> answers = assignmentSubmissionAnswerMapper.selectList(
                new LambdaQueryWrapper<AssignmentSubmissionAnswer>()
                    .eq(AssignmentSubmissionAnswer::getSubmissionId, submission.getId())
            );
            
            // 查询题目信息
            List<Long> questionIds = new ArrayList<>();
            for (AssignmentSubmissionAnswer answer : answers) {
                questionIds.add(answer.getQuestionId());
            }
            
            Map<Long, Question> questionMap = new HashMap<>();
            if (!questionIds.isEmpty()) {
                List<Question> questions = questionMapper.selectBatchIds(questionIds);
                for (Question question : questions) {
                    questionMap.put(question.getId(), question);
                }
            }
            
            // 构建返回结果
            Map<String, Object> result = new HashMap<>();
            result.put("submissionId", submission.getId());
            result.put("assignmentId", submission.getAssignmentId());
            result.put("assignmentTitle", assignment.getTitle());
            result.put("status", submission.getStatus());
            result.put("score", submission.getScore());
            result.put("totalScore", assignment.getTotalScore());
            result.put("feedback", submission.getFeedback());
            result.put("submitTime", submission.getSubmitTime());
            result.put("gradeTime", submission.getGradeTime());
            
            // 答案详情
            List<Map<String, Object>> answerDetails = new ArrayList<>();
            for (AssignmentSubmissionAnswer answer : answers) {
                Map<String, Object> answerDetail = new HashMap<>();
                answerDetail.put("questionId", answer.getQuestionId());
                
                Question question = questionMap.get(answer.getQuestionId());
                if (question != null) {
                    answerDetail.put("questionTitle", question.getTitle());
                    answerDetail.put("questionType", question.getQuestionType());
                    answerDetail.put("correctAnswer", question.getCorrectAnswer());
                    answerDetail.put("explanation", question.getExplanation());
                }
                
                answerDetail.put("studentAnswer", answer.getStudentAnswer());
                answerDetail.put("isCorrect", answer.getIsCorrect());
                answerDetail.put("score", answer.getScore());
                answerDetail.put("comment", answer.getComment());
                
                answerDetails.add(answerDetail);
            }
            
            result.put("answers", answerDetails);
            
            // 计算正确率
            int correctCount = 0;
            for (AssignmentSubmissionAnswer answer : answers) {
                if (answer.getIsCorrect() != null && answer.getIsCorrect()) {
                    correctCount++;
                }
            }
            
            double correctRate = answers.isEmpty() ? 0 : (double) correctCount / answers.size() * 100;
            result.put("correctRate", Math.round(correctRate * 100) / 100.0); // 保留两位小数
            
            // 返回结果
            return Result.success("获取批改结果成功", result);
            
        } catch (Exception e) {
            logger.error("获取作业批改结果异常: {}", e.getMessage(), e);
            return Result.error(500, "获取作业批改结果失败: " + e.getMessage());
        }
    }
    
    /**
     * 获取学生的所有已批改作业
     */
    @Operation(summary = "获取已批改作业列表", description = "获取当前学生的所有已批改作业列表")
    @GetMapping("/graded")
    public Result<List<Map<String, Object>>> getGradedAssignments() {
        
        logger.info("获取学生已批改作业列表");
        
        try {
            // 获取当前登录用户ID
            Long currentUserId = securityUtil.getCurrentUserId();
            if (currentUserId == null) {
                return Result.error("未登录或登录已过期");
            }
            
            // 查询学生信息
            Student student = studentMapper.selectOne(
                new LambdaQueryWrapper<Student>()
                    .eq(Student::getUserId, currentUserId)
                    .last("LIMIT 1")
            );
            
            if (student == null) {
                return Result.error("学生信息不存在");
            }
            
            // 查询已批改的提交
            List<AssignmentSubmission> submissions = assignmentSubmissionMapper.selectList(
                new LambdaQueryWrapper<AssignmentSubmission>()
                    .eq(AssignmentSubmission::getStudentId, student.getId())
                    .eq(AssignmentSubmission::getStatus, 2) // 已批改
            );
            
            // 获取作业ID列表
            List<Long> assignmentIds = new ArrayList<>();
            for (AssignmentSubmission submission : submissions) {
                assignmentIds.add(submission.getAssignmentId());
            }
            
            // 查询作业信息
            Map<Long, Assignment> assignmentMap = new HashMap<>();
            if (!assignmentIds.isEmpty()) {
                List<Assignment> assignments = assignmentMapper.selectBatchIds(assignmentIds);
                for (Assignment assignment : assignments) {
                    assignmentMap.put(assignment.getId(), assignment);
                }
            }
            
            // 构建返回结果
            List<Map<String, Object>> result = new ArrayList<>();
            for (AssignmentSubmission submission : submissions) {
                Map<String, Object> item = new HashMap<>();
                
                Assignment assignment = assignmentMap.get(submission.getAssignmentId());
                if (assignment != null) {
                    item.put("submissionId", submission.getId());
                    item.put("assignmentId", submission.getAssignmentId());
                    item.put("title", assignment.getTitle());
                    item.put("courseId", assignment.getCourseId());
                    item.put("type", assignment.getType());
                    item.put("score", submission.getScore());
                    item.put("totalScore", assignment.getTotalScore());
                    item.put("submitTime", submission.getSubmitTime());
                    item.put("gradeTime", submission.getGradeTime());
                    
                    // 计算得分率
                    double scoreRate = 0;
                    if (assignment.getTotalScore() != null && assignment.getTotalScore() > 0) {
                        scoreRate = (double) submission.getScore() / assignment.getTotalScore() * 100;
                    }
                    item.put("scoreRate", Math.round(scoreRate * 100) / 100.0); // 保留两位小数
                    
                    result.add(item);
                }
            }
            
            return Result.success("获取已批改作业成功", result);
            
        } catch (Exception e) {
            logger.error("获取已批改作业列表异常: {}", e.getMessage(), e);
            return Result.error(500, "获取已批改作业列表失败: " + e.getMessage());
        }
    }
    
    /**
     * 获取作业批改统计
     */
    @Operation(summary = "获取作业批改统计", description = "获取当前学生的作业批改统计信息")
    @GetMapping("/statistics")
    public Result<Map<String, Object>> getAssignmentStatistics() {
        
        logger.info("获取学生作业批改统计");
        
        try {
            // 获取当前登录用户ID
            Long currentUserId = securityUtil.getCurrentUserId();
            if (currentUserId == null) {
                return Result.error("未登录或登录已过期");
            }
            
            // 查询学生信息
            Student student = studentMapper.selectOne(
                new LambdaQueryWrapper<Student>()
                    .eq(Student::getUserId, currentUserId)
                    .last("LIMIT 1")
            );
            
            if (student == null) {
                return Result.error("学生信息不存在");
            }
            
            // 查询所有提交记录
            List<AssignmentSubmission> submissions = assignmentSubmissionMapper.selectList(
                new LambdaQueryWrapper<AssignmentSubmission>()
                    .eq(AssignmentSubmission::getStudentId, student.getId())
            );
            
            // 统计数据
            int totalCount = submissions.size();
            int gradedCount = 0;
            int pendingCount = 0;
            int overdueCount = 0;
            int excellentCount = 0; // 优秀（90分以上）
            int goodCount = 0;      // 良好（80-89分）
            int passCount = 0;      // 及格（60-79分）
            int failCount = 0;      // 不及格（60分以下）
            
            double totalScore = 0;
            int scoreCount = 0;
            
            Date now = new Date();
            
            for (AssignmentSubmission submission : submissions) {
                if (submission.getStatus() != null) {
                    if (submission.getStatus() == 2) { // 已批改
                        gradedCount++;
                        
                        // 统计分数段
                        if (submission.getScore() != null) {
                            totalScore += submission.getScore();
                            scoreCount++;
                            
                            if (submission.getScore() >= 90) {
                                excellentCount++;
                            } else if (submission.getScore() >= 80) {
                                goodCount++;
                            } else if (submission.getScore() >= 60) {
                                passCount++;
                            } else {
                                failCount++;
                            }
                        }
                    } else if (submission.getStatus() == 1) { // 已提交未批改
                        pendingCount++;
                    } else if (submission.getStatus() == 0) { // 未提交
                        // 检查是否已过期
                        Assignment assignment = assignmentMapper.selectById(submission.getAssignmentId());
                        if (assignment != null && assignment.getEndTime() != null && now.after(assignment.getEndTime())) {
                            overdueCount++;
                        }
                    }
                }
            }
            
            // 计算平均分
            double averageScore = scoreCount > 0 ? totalScore / scoreCount : 0;
            
            // 构建返回结果
            Map<String, Object> result = new HashMap<>();
            result.put("totalCount", totalCount);
            result.put("gradedCount", gradedCount);
            result.put("pendingCount", pendingCount);
            result.put("overdueCount", overdueCount);
            result.put("averageScore", Math.round(averageScore * 100) / 100.0);
            
            // 分数分布
            Map<String, Integer> scoreDistribution = new HashMap<>();
            scoreDistribution.put("excellent", excellentCount);
            scoreDistribution.put("good", goodCount);
            scoreDistribution.put("pass", passCount);
            scoreDistribution.put("fail", failCount);
            result.put("scoreDistribution", scoreDistribution);
            
            return Result.success("获取批改统计成功", result);
            
        } catch (Exception e) {
            logger.error("获取作业批改统计异常: {}", e.getMessage(), e);
            return Result.error(500, "获取作业批改统计失败: " + e.getMessage());
        }
    }
} 