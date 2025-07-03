package com.education.controller.student;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.education.dto.common.Result;
import com.education.entity.Assignment;
import com.education.entity.AssignmentQuestion;
import com.education.entity.AssignmentSubmission;
import com.education.entity.AssignmentSubmissionAnswer;
import com.education.entity.Question;
import com.education.entity.Student;
import com.education.entity.StudentAnswer;
import com.education.mapper.AssignmentMapper;
import com.education.mapper.AssignmentQuestionMapper;
import com.education.mapper.AssignmentSubmissionAnswerMapper;
import com.education.mapper.AssignmentSubmissionMapper;
import com.education.mapper.QuestionMapper;
import com.education.mapper.StudentAnswerMapper;
import com.education.mapper.StudentMapper;
import com.education.security.SecurityUtil;
import com.fasterxml.jackson.databind.ObjectMapper;
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
    private StudentAnswerMapper studentAnswerMapper;
    
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
    public Result getAssignmentList(
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
                    // 待完成：未提交且未过期
                    sql += "AND (s.status IS NULL OR s.status = 0) AND (a.end_time IS NULL OR a.end_time > ?) ";
                    params.add(now);
                } else if ("completed".equals(status)) {
                    // 已完成：已提交
                    sql += "AND s.status = 1 ";
                } else if ("overdue".equals(status)) {
                    // 已逾期：未提交且已过期
                    sql += "AND (s.status IS NULL OR s.status = 0) AND a.end_time < ? ";
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
                        int submissionStatus = rs.getInt("submission_status");
                        Date endTime = rs.getTimestamp("end_time");
                        
                        String statusValue;
                        if (submissionStatus == 1) {
                            statusValue = "completed";
                            assignment.put("submissionTime", rs.getTimestamp("submit_time"));
                            assignment.put("score", rs.getObject("score"));
                        } else if (endTime != null && now.after(endTime)) {
                            statusValue = "overdue";
                        } else {
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
    public Result getAssignmentDetail(
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
    public Result getAssignmentQuestions(
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
    public Result saveQuestionAnswer(
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
                answerContent = String.join(",", (List<String>) answerObj);
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
    public Result submitAssignment(
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
            
            // 更新提交状态
            submission.setStatus(1); // 已提交未批改
            submission.setSubmitTime(new Date());
            submission.setUpdateTime(new Date());
            assignmentSubmissionMapper.updateById(submission);
            
            // 计算总分
            List<AssignmentSubmissionAnswer> answers = assignmentSubmissionAnswerMapper.selectList(
                new LambdaQueryWrapper<AssignmentSubmissionAnswer>()
                    .eq(AssignmentSubmissionAnswer::getSubmissionId, submission.getId())
            );
            
            int totalScore = 0;
            for (AssignmentSubmissionAnswer answer : answers) {
                if (answer.getScore() != null) {
                    totalScore += answer.getScore();
                }
            }
            
            // 更新总分
            submission.setScore(totalScore);
            assignmentSubmissionMapper.updateById(submission);
            
            logger.info("提交任务成功，任务ID: {}, 学生ID: {}, 得分: {}", id, currentUserId, totalScore);
            
            return Result.success("提交成功");
            
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
    public Result getSavedAnswer(
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
    public Result submitAssignmentFile(
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
            return Result.success("作业提交成功");
            
        } catch (Exception e) {
            logger.error("提交文件作业失败: {}", e.getMessage(), e);
            return Result.error("提交作业失败: " + e.getMessage());
        }
    }
} 