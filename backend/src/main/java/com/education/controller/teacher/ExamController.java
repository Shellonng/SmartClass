package com.education.controller.teacher;

import com.education.dto.ExamDTO;
import com.education.dto.common.PageRequest;
import com.education.dto.common.PageResponse;
import com.education.dto.common.Result;
import com.education.service.ExamService;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

/**
 * 考试控制器
 */
@RestController
@RequestMapping("/api/teacher/exams")
@RequiredArgsConstructor
public class ExamController {
    
    private final ExamService examService;
    
    /**
     * 分页查询考试列表
     * @param pageRequest 分页请求
     * @param courseId 课程ID
     * @param userId 用户ID
     * @param keyword 关键词
     * @param status 状态
     * @return 分页结果
     */
    @GetMapping
    public Result<PageResponse<ExamDTO>> pageExams(PageRequest pageRequest,
                                                  @RequestParam(required = false) Long courseId,
                                                  @RequestParam(required = false) Long userId,
                                                  @RequestParam(required = false) String keyword,
                                                  @RequestParam(required = false) Integer status) {
        PageResponse<ExamDTO> pageResponse = examService.pageExams(pageRequest, courseId, userId, keyword, status);
        return Result.success(pageResponse);
    }
    
    /**
     * 获取考试详情
     * @param id 考试ID
     * @return 考试详情
     */
    @GetMapping("/{id}")
    public Result<ExamDTO> getExamDetail(@PathVariable Long id) {
        ExamDTO examDTO = examService.getExamDetail(id);
        return Result.success(examDTO);
    }
    
    /**
     * 创建考试
     * @param examDTO 考试信息
     * @return 创建的考试ID
     */
    @PostMapping
    public Result<Long> createExam(@RequestBody ExamDTO examDTO) {
        Long id = examService.createExam(examDTO);
        return Result.success(id);
    }
    
    /**
     * 更新考试
     * @param id 考试ID
     * @param examDTO 考试信息
     * @return 是否成功
     */
    @PutMapping("/{id}")
    public Result<Boolean> updateExam(@PathVariable Long id, @RequestBody ExamDTO examDTO) {
        examDTO.setId(id);
        boolean success = examService.updateExam(examDTO);
        return Result.success(success);
    }
    
    /**
     * 删除考试
     * @param id 考试ID
     * @return 是否成功
     */
    @DeleteMapping("/{id}")
    public Result<Boolean> deleteExam(@PathVariable Long id) {
        boolean success = examService.deleteExam(id);
        return Result.success(success);
    }
    
    /**
     * 发布考试
     * @param id 考试ID
     * @return 是否成功
     */
    @PutMapping("/{id}/publish")
    public Result<Boolean> publishExam(@PathVariable Long id) {
        boolean success = examService.publishExam(id);
        return Result.success(success);
    }
    
    /**
     * 组卷（自动生成试卷）
     * @param examDTO 考试信息（包含组卷配置）
     * @return 组卷后的考试信息
     */
    @PostMapping("/generate-paper")
    public Result<ExamDTO> generateExamPaper(@RequestBody ExamDTO examDTO) {
        ExamDTO result = examService.generateExamPaper(examDTO);
        return Result.success(result);
    }
    
    /**
     * 手动选题
     * @param examId 考试ID
     * @param questionIds 题目ID列表
     * @param scores 分值列表
     * @return 是否成功
     */
    @PostMapping("/{examId}/select-questions")
    public Result<Boolean> selectQuestions(@PathVariable Long examId,
                                          @RequestParam List<Long> questionIds,
                                          @RequestParam List<Integer> scores) {
        boolean success = examService.selectQuestions(examId, questionIds, scores);
        return Result.success(success);
    }
    
    /**
     * 获取知识点列表
     * @param courseId 课程ID
     * @param createdBy 创建者ID
     * @return 知识点列表
     */
    @GetMapping("/questions/knowledge-points")
    public Result<List<String>> getKnowledgePoints(@RequestParam(required = false) Long courseId,
                                                  @RequestParam(required = false) Long createdBy) {
        List<String> knowledgePoints = examService.getKnowledgePoints(courseId, createdBy);
        return Result.success(knowledgePoints);
    }
    
    /**
     * 获取题目列表（按题型分类）
     * @param courseId 课程ID
     * @param questionType 题目类型
     * @param difficulty 难度
     * @param knowledgePoint 知识点
     * @param createdBy 创建者ID
     * @return 题目列表
     */
    @GetMapping("/questions")
    public Result<Map<String, List<Map<String, Object>>>> getQuestionsByType(
            @RequestParam Long courseId,
            @RequestParam(required = false) String questionType,
            @RequestParam(required = false) Integer difficulty,
            @RequestParam(required = false) String knowledgePoint,
            @RequestParam(required = false) Long createdBy) {
        Map<String, List<Map<String, Object>>> questions = examService.getQuestionsByType(
                courseId, questionType, difficulty, knowledgePoint, createdBy);
        return Result.success(questions);
    }
} 