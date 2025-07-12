package com.education.controller.teacher;

import com.education.dto.common.Result;
import com.education.entity.AssignmentSubmissionAnswer;
import com.education.entity.Course;
import com.education.entity.Question;
import com.education.mapper.AssignmentSubmissionAnswerMapper;
import com.education.mapper.CourseMapper;
import com.education.mapper.QuestionMapper;
import com.education.security.SecurityUtil;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.*;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/api/teacher/analytics")
@Tag(name = "教师数据分析接口", description = "提供教师数据分析相关的API")
public class TeacherAnalyticsController {

    private final Logger logger = LoggerFactory.getLogger(TeacherAnalyticsController.class);
    
    @Autowired
    private AssignmentSubmissionAnswerMapper submissionAnswerMapper;
    
    @Autowired
    private QuestionMapper questionMapper;
    
    @Autowired
    private CourseMapper courseMapper;
    
    @Autowired
    private SecurityUtil securityUtil;

    @Operation(summary = "获取教师课程列表", description = "获取当前登录教师的所有课程")
    @GetMapping("/courses")
    public Result<List<Course>> getTeacherCourses() {
        Long currentUserId = securityUtil.getCurrentUserId();
        if (currentUserId == null) {
            return Result.error("未登录或登录已过期");
        }
        
        try {
            List<Course> courses = courseMapper.selectByTeacherId(currentUserId);
            return Result.success(courses);
        } catch (Exception e) {
            logger.error("获取教师课程列表失败", e);
            return Result.error("获取教师课程列表失败: " + e.getMessage());
        }
    }

    @Operation(summary = "获取题目正确率统计", description = "获取指定课程或所有课程中题目的正确率统计")
    @GetMapping("/question-stats")
    public Result<Map<String, Object>> getQuestionStats(
            @Parameter(description = "课程ID") @RequestParam(required = false) Long courseId) {
        
        Long currentUserId = securityUtil.getCurrentUserId();
        if (currentUserId == null) {
            return Result.error("未登录或登录已过期");
        }
        
        try {
            // 获取该教师的题目（可根据课程ID过滤）
            List<Question> questions;
            if (courseId != null) {
                questions = questionMapper.selectByTeacherAndCourse(currentUserId, courseId);
            } else {
                questions = questionMapper.selectByTeacher(currentUserId);
            }
            
            if (questions.isEmpty()) {
                return Result.success(Collections.singletonMap("message", "没有找到相关题目"));
            }
            
            // 获取题目ID列表
            List<Long> questionIds = questions.stream()
                    .map(Question::getId)
                    .collect(Collectors.toList());
            
            // 获取这些题目的所有回答
            List<AssignmentSubmissionAnswer> answers = submissionAnswerMapper.selectByQuestionIds(questionIds);
            
            // 统计每个题目的正确率
            Map<Long, Integer> totalAnswers = new HashMap<>();
            Map<Long, Integer> correctAnswers = new HashMap<>();
            
            for (AssignmentSubmissionAnswer answer : answers) {
                Long questionId = answer.getQuestionId();
                totalAnswers.put(questionId, totalAnswers.getOrDefault(questionId, 0) + 1);
                
                if (answer.getIsCorrect() != null && answer.getIsCorrect()) {
                    correctAnswers.put(questionId, correctAnswers.getOrDefault(questionId, 0) + 1);
                }
            }
            
            // 整理返回数据
            List<Map<String, Object>> questionStats = new ArrayList<>();
            
            for (Question question : questions) {
                Long questionId = question.getId();
                int total = totalAnswers.getOrDefault(questionId, 0);
                int correct = correctAnswers.getOrDefault(questionId, 0);
                double correctRate = total > 0 ? (double) correct / total * 100 : 0;
                
                Map<String, Object> stats = new HashMap<>();
                stats.put("questionId", questionId);
                stats.put("title", question.getTitle());
                stats.put("type", question.getQuestionType());
                stats.put("difficulty", question.getDifficulty());
                stats.put("totalAnswers", total);
                stats.put("correctAnswers", correct);
                stats.put("correctRate", Math.round(correctRate * 10) / 10.0); // 保留一位小数
                
                questionStats.add(stats);
            }
            
            // 计算各题型的平均正确率
            Map<String, Double> typeStats = calculateTypeStats(questions, totalAnswers, correctAnswers);
            
            // 计算各难度级别的平均正确率
            Map<Integer, Double> difficultyStats = calculateDifficultyStats(questions, totalAnswers, correctAnswers);
            
            Map<String, Object> result = new HashMap<>();
            result.put("questions", questionStats);
            result.put("typeStats", typeStats);
            result.put("difficultyStats", difficultyStats);
            result.put("totalQuestions", questions.size());
            
            return Result.success(result);
            
        } catch (Exception e) {
            logger.error("获取题目正确率统计失败", e);
            return Result.error("获取题目正确率统计失败: " + e.getMessage());
        }
    }
    
    private Map<String, Double> calculateTypeStats(
            List<Question> questions,
            Map<Long, Integer> totalAnswers,
            Map<Long, Integer> correctAnswers) {
        
        Map<String, Integer> typeTotalAnswers = new HashMap<>();
        Map<String, Integer> typeCorrectAnswers = new HashMap<>();
        
        for (Question question : questions) {
            String type = question.getQuestionType().toString();
            Long questionId = question.getId();
            
            int total = totalAnswers.getOrDefault(questionId, 0);
            int correct = correctAnswers.getOrDefault(questionId, 0);
            
            typeTotalAnswers.put(type, typeTotalAnswers.getOrDefault(type, 0) + total);
            typeCorrectAnswers.put(type, typeCorrectAnswers.getOrDefault(type, 0) + correct);
        }
        
        Map<String, Double> typeStats = new HashMap<>();
        for (String type : typeTotalAnswers.keySet()) {
            int total = typeTotalAnswers.get(type);
            int correct = typeCorrectAnswers.getOrDefault(type, 0);
            double correctRate = total > 0 ? (double) correct / total * 100 : 0;
            typeStats.put(type, Math.round(correctRate * 10) / 10.0); // 保留一位小数
        }
        
        return typeStats;
    }
    
    private Map<Integer, Double> calculateDifficultyStats(
            List<Question> questions,
            Map<Long, Integer> totalAnswers,
            Map<Long, Integer> correctAnswers) {
        
        Map<Integer, Integer> difficultyTotalAnswers = new HashMap<>();
        Map<Integer, Integer> difficultyCorrectAnswers = new HashMap<>();
        
        for (Question question : questions) {
            int difficulty = question.getDifficulty();
            Long questionId = question.getId();
            
            int total = totalAnswers.getOrDefault(questionId, 0);
            int correct = correctAnswers.getOrDefault(questionId, 0);
            
            difficultyTotalAnswers.put(difficulty, difficultyTotalAnswers.getOrDefault(difficulty, 0) + total);
            difficultyCorrectAnswers.put(difficulty, difficultyCorrectAnswers.getOrDefault(difficulty, 0) + correct);
        }
        
        Map<Integer, Double> difficultyStats = new HashMap<>();
        for (Integer difficulty : difficultyTotalAnswers.keySet()) {
            int total = difficultyTotalAnswers.get(difficulty);
            int correct = difficultyCorrectAnswers.getOrDefault(difficulty, 0);
            double correctRate = total > 0 ? (double) correct / total * 100 : 0;
            difficultyStats.put(difficulty, Math.round(correctRate * 10) / 10.0); // 保留一位小数
        }
        
        return difficultyStats;
    }
} 