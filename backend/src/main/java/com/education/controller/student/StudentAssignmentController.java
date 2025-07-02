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

@Tag(name = "学生作业接口", description = "学生作业相关接口")
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
     * 获取作业详情
     * @param id 作业ID
     * @return 作业详情
     */
    @Operation(summary = "获取作业详情", description = "获取作业的详细信息")
    @GetMapping("/{id}")
    public Result getAssignmentDetail(
            @Parameter(description = "作业ID") @PathVariable Long id) {
        
        logger.info("获取学生作业详情，作业ID: {}", id);
        
        try {
            // 查询作业信息
            Assignment assignment = assignmentMapper.selectById(id);
            
            if (assignment == null) {
                logger.error("作业不存在，作业ID: {}", id);
                return Result.error("作业不存在");
            }
            
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
            
            // 检查是否已经有提交记录，如果没有则创建一个状态为未提交的记录
            AssignmentSubmission submission = assignmentSubmissionMapper.selectOne(
                new LambdaQueryWrapper<AssignmentSubmission>()
                    .eq(AssignmentSubmission::getAssignmentId, id)
                    .eq(AssignmentSubmission::getStudentId, currentUserId)
            );
            
            if (submission == null) {
                submission = new AssignmentSubmission();
                submission.setAssignmentId(id);
                submission.setStudentId(student.getId());
                submission.setStatus(0); // 未提交
                submission.setCreateTime(new Date());
                assignmentSubmissionMapper.insert(submission);
                
                logger.info("为学生创建作业提交记录，学生ID: {}, 作业ID: {}", student.getId(), id);
            }
            
            // 返回作业详情和提交状态
            Map<String, Object> result = new HashMap<>();
            result.put("assignment", assignment);
            result.put("submission", submission);
            
            return Result.success(result);
            
        } catch (Exception e) {
            logger.error("获取作业详情失败: {}", e.getMessage(), e);
            return Result.error("获取作业详情失败: " + e.getMessage());
        }
    }
    
    /**
     * 获取作业题目
     * @param id 作业ID
     * @return 作业题目列表
     */
    @Operation(summary = "获取作业题目", description = "获取作业的题目列表")
    @GetMapping("/{id}/questions")
    public Result getAssignmentQuestions(
            @Parameter(description = "作业ID") @PathVariable Long id) {
        
        logger.info("获取学生作业题目，作业ID: {}", id);
        
        try {
            // 查询作业信息
            Assignment assignment = assignmentMapper.selectById(id);
            
            if (assignment == null) {
                logger.error("作业不存在，作业ID: {}", id);
                return Result.error("作业不存在");
            }
            
            // 检查作业是否已发布
            if (assignment.getStatus() != 1) {
                logger.error("作业未发布，不能查看题目，作业ID: {}", id);
                return Result.error("作业未发布，不能查看题目");
            }
            
            // 检查是否是考试类型，且考试未开始
            if ("exam".equals(assignment.getType()) && 
                assignment.getStartTime() != null && 
                new Date().before(assignment.getStartTime())) {
                logger.error("考试未开始，不能查看题目，作业ID: {}", id);
                return Result.error("考试未开始，不能查看题目");
            }
            
            // 查询作业题目
            List<Map<String, Object>> questions = new ArrayList<>();
            
            // 1. 查询作业-题目关联表
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
            logger.error("获取作业题目失败: {}", e.getMessage(), e);
            return Result.error("获取作业题目失败: " + e.getMessage());
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
     * @param id 作业ID
     * @param questionId 题目ID
     * @param request 答案数据
     * @return 保存结果
     */
    @Operation(summary = "保存单题答案", description = "保存单个题目的答案")
    @PostMapping("/{id}/questions/{questionId}/save")
    @Transactional
    public Result saveQuestionAnswer(
            @Parameter(description = "作业ID") @PathVariable Long id,
            @Parameter(description = "题目ID") @PathVariable Long questionId,
            @RequestBody Map<String, Object> request) {
        
        logger.info("保存单题答案，作业ID: {}, 题目ID: {}", id, questionId);
        
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
            
            // 查询作业信息
            Assignment assignment = assignmentMapper.selectById(id);
            
            if (assignment == null) {
                logger.error("作业不存在，作业ID: {}", id);
                return Result.error("作业不存在");
            }
            
            // 检查作业是否已发布
            if (assignment.getStatus() != 1) {
                logger.error("作业未发布，不能保存答案，作业ID: {}", id);
                return Result.error("作业未发布，不能保存答案");
            }
            
            // 检查是否已过截止时间
            if (assignment.getEndTime() != null && new Date().after(assignment.getEndTime())) {
                logger.error("作业已截止，不能保存答案，作业ID: {}", id);
                return Result.error("作业已截止，不能保存答案");
            }
            
            // 检查题目是否属于该作业
            AssignmentQuestion assignmentQuestion = assignmentQuestionMapper.selectOne(
                new LambdaQueryWrapper<AssignmentQuestion>()
                    .eq(AssignmentQuestion::getAssignmentId, id)
                    .eq(AssignmentQuestion::getQuestionId, questionId)
            );
            
            if (assignmentQuestion == null) {
                logger.error("题目不属于该作业，作业ID: {}, 题目ID: {}", id, questionId);
                return Result.error("题目不属于该作业");
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
                
                logger.info("为学生创建作业提交记录，学生ID: {}, 作业ID: {}", currentUserId, id);
            } else if (submission.getStatus() == 1) {
                // 如果已提交，不能再保存答案
                logger.error("作业已提交，不能再保存答案，作业ID: {}", id);
                return Result.error("作业已提交，不能再保存答案");
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
     * 提交作业/考试答案
     * @param id 作业/考试ID
     * @return 提交结果
     */
    @Operation(summary = "提交作业/考试答案", description = "提交作业/考试的答案")
    @PostMapping("/{id}/submit")
    @Transactional
    public Result submitAssignment(
            @Parameter(description = "作业ID") @PathVariable Long id) {
        
        logger.info("提交作业/考试答案，作业ID: {}", id);
        
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
            
            // 查询作业信息
            Assignment assignment = assignmentMapper.selectById(id);
            
            if (assignment == null) {
                logger.error("作业不存在，作业ID: {}", id);
                return Result.error("作业不存在");
            }
            
            // 检查作业是否已发布
            if (assignment.getStatus() != 1) {
                logger.error("作业未发布，不能提交，作业ID: {}", id);
                return Result.error("作业未发布，不能提交");
            }
            
            // 检查是否已过截止时间
            if (assignment.getEndTime() != null && new Date().after(assignment.getEndTime())) {
                logger.error("作业已截止，不能提交，作业ID: {}", id);
                return Result.error("作业已截止，不能提交");
            }
            
            // 查询提交记录
            AssignmentSubmission submission = assignmentSubmissionMapper.selectOne(
                new LambdaQueryWrapper<AssignmentSubmission>()
                    .eq(AssignmentSubmission::getAssignmentId, id)
                    .eq(AssignmentSubmission::getStudentId, currentUserId)
            );
            
            if (submission == null) {
                logger.error("未找到提交记录，作业ID: {}, 学生ID: {}", id, currentUserId);
                return Result.error("未找到提交记录");
            }
            
            // 检查是否已经提交过
            if (submission.getStatus() == 1) {
                logger.error("已经提交过，不能重复提交，作业ID: {}", id);
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
            
            logger.info("提交作业成功，作业ID: {}, 学生ID: {}, 得分: {}", id, currentUserId, totalScore);
            
            return Result.success("提交成功");
            
        } catch (Exception e) {
            logger.error("提交作业答案失败: {}", e.getMessage(), e);
            return Result.error("提交作业答案失败: " + e.getMessage());
        }
    }
    
    /**
     * 获取已保存的答案
     * @param id 作业ID
     * @param questionId 题目ID
     * @return 已保存的答案
     */
    @Operation(summary = "获取已保存答案", description = "获取学生已保存的答案")
    @GetMapping("/{id}/questions/{questionId}/answer")
    public Result getSavedAnswer(
            @Parameter(description = "作业ID") @PathVariable Long id,
            @Parameter(description = "题目ID") @PathVariable Long questionId) {
        
        logger.info("获取已保存答案，作业ID: {}, 题目ID: {}", id, questionId);
        
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
                logger.info("未找到提交记录，作业ID: {}, 学生ID: {}", id, currentUserId);
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
} 