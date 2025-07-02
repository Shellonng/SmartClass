package com.education.controller.teacher;

import com.education.dto.ExamDTO;
import com.education.dto.common.PageRequest;
import com.education.dto.common.PageResponse;
import com.education.dto.common.Result;
import com.education.entity.Course;
import com.education.mapper.CourseMapper;
import com.education.security.SecurityUtil;
import com.education.service.ExamService;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

/**
 * 作业控制器
 */
@RestController
@RequestMapping("/api/teacher/assignments")
@RequiredArgsConstructor
public class AssignmentController {
    
    private final ExamService examService;
    private final CourseMapper courseMapper;
    private final SecurityUtil securityUtil;
    
    /**
     * 获取当前教师的课程列表
     * @return 课程列表
     */
    @GetMapping("/courses")
    public Result<List<Course>> getTeacherCourses() {
        Long userId = securityUtil.getCurrentUserId();
        if (userId == null) {
            return Result.error("未登录");
        }
        
        List<Course> courses = courseMapper.selectByTeacherId(userId);
        return Result.success(courses);
    }
    
    /**
     * 分页查询作业列表
     * @param pageRequest 分页请求
     * @param courseId 课程ID
     * @param keyword 关键词
     * @param status 状态
     * @return 分页结果
     */
    @GetMapping
    public Result<PageResponse<ExamDTO>> pageAssignments(PageRequest pageRequest,
                                                  @RequestParam(required = false) Long courseId,
                                                  @RequestParam(required = false) String keyword,
                                                  @RequestParam(required = false) Integer status) {
        // 获取当前登录用户ID
        Long userId = securityUtil.getCurrentUserId();
        if (userId == null) {
            return Result.error("未登录");
        }
        
        // 指定type为homework，并使用当前用户ID进行筛选
        PageResponse<ExamDTO> pageResponse = examService.pageExamsByType(pageRequest, courseId, userId, keyword, status, "homework");
        return Result.success(pageResponse);
    }
    
    /**
     * 获取作业详情
     * @param id 作业ID
     * @return 作业详情
     */
    @GetMapping("/{id}")
    public Result<ExamDTO> getAssignmentDetail(@PathVariable Long id) {
        ExamDTO examDTO = examService.getExamDetail(id);
        return Result.success(examDTO);
    }
    
    /**
     * 获取作业提交率
     * @param id 作业ID
     * @return 提交率（百分比，0-100）
     */
    @GetMapping("/{id}/submission-rate")
    public Result<Double> getSubmissionRate(@PathVariable Long id) {
        double submissionRate = examService.getSubmissionRate(id);
        return Result.success(submissionRate);
    }
    
    /**
     * 创建作业
     * @param examDTO 作业信息
     * @return 创建的作业ID
     */
    @PostMapping
    public Result<Long> createAssignment(@RequestBody ExamDTO examDTO) {
        // 获取当前登录用户ID
        Long userId = securityUtil.getCurrentUserId();
        if (userId == null) {
            return Result.error("未登录");
        }
        
        // 设置type为homework和userId
        examDTO.setType("homework");
        examDTO.setUserId(userId);
        
        Long id = examService.createExam(examDTO);
        return Result.success(id);
    }
    
    /**
     * 更新作业
     * @param id 作业ID
     * @param examDTO 作业信息
     * @return 是否成功
     */
    @PutMapping("/{id}")
    public Result<Boolean> updateAssignment(@PathVariable Long id, @RequestBody ExamDTO examDTO) {
        examDTO.setId(id);
        examDTO.setType("homework");
        boolean success = examService.updateExam(examDTO);
        return Result.success(success);
    }
    
    /**
     * 删除作业
     * @param id 作业ID
     * @return 是否成功
     */
    @DeleteMapping("/{id}")
    public Result<Boolean> deleteAssignment(@PathVariable Long id) {
        boolean success = examService.deleteExam(id);
        return Result.success(success);
    }
    
    /**
     * 发布作业
     * @param id 作业ID
     * @return 是否成功
     */
    @PutMapping("/{id}/publish")
    public Result<Boolean> publishAssignment(@PathVariable Long id) {
        boolean success = examService.publishExam(id);
        return Result.success(success);
    }
    
    /**
     * 组卷（自动生成作业）
     * @param examDTO 作业信息（包含组卷配置）
     * @return 组卷后的作业信息
     */
    @PostMapping("/generate-paper")
    public Result<ExamDTO> generateAssignmentPaper(@RequestBody ExamDTO examDTO) {
        examDTO.setType("homework");
        ExamDTO result = examService.generateExamPaper(examDTO);
        return Result.success(result);
    }
    
    /**
     * 手动选题
     * @param assignmentId 作业ID
     * @param requestBody 包含题目ID和分值的请求体
     * @return 是否成功
     */
    @PostMapping("/{assignmentId}/select-questions")
    public Result<Boolean> selectQuestions(
            @PathVariable Long assignmentId,
            @RequestBody Map<String, Object> requestBody) {
        
        @SuppressWarnings("unchecked")
        List<Object> questionIdObjects = (List<Object>) requestBody.get("questionIds");
        List<Long> questionIds = questionIdObjects.stream()
                .map(obj -> {
                    if (obj instanceof Integer) {
                        return ((Integer) obj).longValue();
                    } else if (obj instanceof Long) {
                        return (Long) obj;
                    } else if (obj instanceof Double) {
                        return ((Double) obj).longValue();
                    } else if (obj instanceof String) {
                        return Long.parseLong((String) obj);
                    }
                    return 0L;
                })
                .toList();
        
        @SuppressWarnings("unchecked")
        List<Object> scoreObjects = (List<Object>) requestBody.get("scores");
        List<Integer> scores = scoreObjects.stream()
                .map(obj -> {
                    if (obj instanceof Integer) {
                        return (Integer) obj;
                    } else if (obj instanceof Double) {
                        return ((Double) obj).intValue();
                    } else if (obj instanceof String) {
                        return Integer.parseInt((String) obj);
                    }
                    return 0;
                })
                .toList();
        
        boolean success = examService.selectQuestions(assignmentId, questionIds, scores);
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
     * @param keyword 关键词
     * @return 题目列表
     */
    @GetMapping("/questions")
    public Result<Map<String, List<Map<String, Object>>>> getQuestionsByType(
            @RequestParam Long courseId,
            @RequestParam(required = false) String questionType,
            @RequestParam(required = false) Integer difficulty,
            @RequestParam(required = false) String knowledgePoint,
            @RequestParam(required = false) Long createdBy,
            @RequestParam(required = false) String keyword) {
        Map<String, List<Map<String, Object>>> questions = examService.getQuestionsByType(
                courseId, questionType, difficulty, knowledgePoint, createdBy, keyword);
        return Result.success(questions);
    }
    
    /**
     * 获取作业提交记录列表
     * @param assignmentId 作业ID
     * @param pageRequest 分页请求
     * @param status 状态筛选
     * @return 提交记录分页结果
     */
    @GetMapping("/{assignmentId}/submissions")
    public Result<PageResponse<Map<String, Object>>> getAssignmentSubmissions(
            @PathVariable Long assignmentId,
            PageRequest pageRequest,
            @RequestParam(required = false) Integer status) {
        PageResponse<Map<String, Object>> pageResponse = examService.getAssignmentSubmissions(
                assignmentId, pageRequest, status);
        return Result.success(pageResponse);
    }
} 