package com.education.controller.teacher;

import com.education.dto.AssignmentDTO;
import com.education.dto.GradingResultDTO;
import com.education.dto.common.PageRequest;
import com.education.dto.common.PageResponse;
import com.education.dto.common.Result;
import com.education.entity.Course;
import com.education.mapper.CourseMapper;
import com.education.security.SecurityUtil;
import com.education.service.AssignmentService;
import com.education.service.DifyGradingService;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.core.metadata.IPage;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.context.request.RequestContextHolder;
import org.springframework.web.context.request.ServletRequestAttributes;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import lombok.extern.slf4j.Slf4j;
import io.swagger.v3.oas.annotations.Operation;

/**
 * 作业控制器
 */
@RestController
@RequestMapping("/api/teacher/assignments")
@RequiredArgsConstructor
@Slf4j
public class AssignmentController {
    
    private final AssignmentService assignmentService;
    private final CourseMapper courseMapper;
    private final SecurityUtil securityUtil;
    private final DifyGradingService difyGradingService;
    
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
    public Result<PageResponse<AssignmentDTO>> pageAssignments(PageRequest pageRequest,
                                                  @RequestParam(required = false) Long courseId,
                                                  @RequestParam(required = false) String keyword,
                                                  @RequestParam(required = false) Integer status) {
        try {
            // 获取当前登录用户ID
            Long userId = securityUtil.getCurrentUserId();
            
            // 如果userId为null，尝试从请求头中获取token
            if (userId == null) {
                // 从请求头中获取Authorization
                String authHeader = ((ServletRequestAttributes) RequestContextHolder.getRequestAttributes())
                    .getRequest().getHeader("Authorization");
                
                if (authHeader != null && authHeader.startsWith("Bearer ")) {
                    String token = authHeader.substring(7);
                    // 处理简化的token格式: token-{userId}
                    if (token.startsWith("token-")) {
                        try {
                            userId = Long.parseLong(token.substring(6)); // 提取userId部分
                            System.out.println("从token中提取的userId: " + userId);
                        } catch (NumberFormatException e) {
                            return Result.error("无效的token格式");
                        }
                    }
                }
                
                // 如果仍然为null，返回未登录错误
                if (userId == null) {
                    return Result.error("未登录");
                }
            }
            
            // 确保分页参数有效
            if (pageRequest == null) {
                pageRequest = new PageRequest(1, 10);
            }
            
            // 创建分页对象
            Page<com.education.entity.Assignment> page = new Page<>(
                pageRequest.getPageNum() != null ? pageRequest.getPageNum() : 1, 
                pageRequest.getPageSize() != null ? pageRequest.getPageSize() : 10
            );
            
            IPage<AssignmentDTO> assignmentPage = assignmentService.pageAssignments(page, courseId, userId, keyword, status);
            
            // 转换为PageResponse
            PageResponse<AssignmentDTO> pageResponse = new PageResponse<>();
            pageResponse.setList(assignmentPage.getRecords());
            pageResponse.setTotal(assignmentPage.getTotal());
            pageResponse.setPageNum(pageRequest.getPageNum() != null ? pageRequest.getPageNum() : 1);
            pageResponse.setPageSize(pageRequest.getPageSize() != null ? pageRequest.getPageSize() : 10);
            
            return Result.success(pageResponse);
        } catch (Exception e) {
            e.printStackTrace();
            System.err.println("获取作业列表失败: " + e.getMessage());
            return Result.error("获取作业列表失败: " + e.getMessage());
        }
    }
    
    /**
     * 获取作业详情
     * @param id 作业ID
     * @return 作业详情
     */
    @GetMapping("/{id}")
    public Result<AssignmentDTO> getAssignmentDetail(@PathVariable Long id) {
        AssignmentDTO assignmentDTO = assignmentService.getAssignmentById(id);
        return Result.success(assignmentDTO);
    }
    
    /**
     * 获取作业提交率
     * @param id 作业ID
     * @return 提交率（百分比，0-100）
     */
    @GetMapping("/{id}/submission-rate")
    public Result<Double> getSubmissionRate(@PathVariable Long id) {
        double submissionRate = assignmentService.getSubmissionRate(id);
        return Result.success(submissionRate);
    }
    
    /**
     * 创建作业
     * @param assignmentDTO 作业信息
     * @return 创建的作业ID
     */
    @PostMapping
    public Result<Long> createAssignment(@RequestBody AssignmentDTO assignmentDTO) {
        // 获取当前登录用户ID
        Long userId = securityUtil.getCurrentUserId();
        if (userId == null) {
            return Result.error("未登录");
        }
        
        // 设置type为homework和userId
        assignmentDTO.setType("homework");
        assignmentDTO.setUserId(userId);
        
        Long id = assignmentService.createAssignment(assignmentDTO);
        return Result.success(id);
    }
    
    /**
     * 更新作业
     * @param id 作业ID
     * @param assignmentDTO 作业信息
     * @return 是否成功
     */
    @PutMapping("/{id}")
    public Result<Boolean> updateAssignment(@PathVariable Long id, @RequestBody AssignmentDTO assignmentDTO) {
        // 获取当前登录用户ID
        Long userId = securityUtil.getCurrentUserId();
        if (userId == null) {
            return Result.error("未登录");
        }
        
        assignmentDTO.setId(id);
        assignmentDTO.setType("homework");
        assignmentDTO.setUserId(userId); // 确保使用当前登录用户的ID
        
        boolean success = assignmentService.updateAssignment(id, assignmentDTO);
        return Result.success(success);
    }
    
    /**
     * 删除作业
     * @param id 作业ID
     * @return 是否成功
     */
    @DeleteMapping("/{id}")
    public Result<Boolean> deleteAssignment(@PathVariable Long id) {
        boolean success = assignmentService.deleteAssignment(id);
        return Result.success(success);
    }
    
    /**
     * 发布作业
     * @param id 作业ID
     * @return 是否成功
     */
    @PutMapping("/{id}/publish")
    public Result<Boolean> publishAssignment(@PathVariable Long id) {
        boolean success = assignmentService.publishAssignment(id);
        return Result.success(success);
    }
    
    /**
     * 取消发布作业
     * @param id 作业ID
     * @return 是否成功
     */
    @PutMapping("/{id}/unpublish")
    public Result<Boolean> unpublishAssignment(@PathVariable Long id) {
        boolean success = assignmentService.unpublishAssignment(id);
        return Result.success(success);
    }
    
    /**
     * 组卷（自动生成作业）
     * @param assignmentDTO 作业信息（包含组卷配置）
     * @return 组卷后的题目列表
     */
    @PostMapping("/generate-paper")
    public Result<List<AssignmentDTO.AssignmentQuestionDTO>> generateAssignmentPaper(@RequestBody AssignmentDTO assignmentDTO) {
        assignmentDTO.setType("homework");
        List<AssignmentDTO.AssignmentQuestionDTO> questions = assignmentService.generatePaper(assignmentDTO);
        return Result.success(questions);
    }
    
    /**
     * 手动选题
     * @param assignmentId 作业ID
     * @param requestBody 包含题目ID和分值的请求体
     * @return 是否成功
     */
    @Operation(summary = "选择题目", description = "为作业选择题目")
    @PostMapping("/{assignmentId}/questions/select")
    public Result<Boolean> selectQuestions(
            @PathVariable Long assignmentId,
            @RequestBody Map<String, Object> requestBody) {
        try {
            // 获取题目ID列表
            List<Object> questionIdObjects = (List<Object>) requestBody.get("questionIds");
            List<Long> questionIds = questionIdObjects.stream()
                    .map(obj -> Long.valueOf(obj.toString()))
                    .collect(Collectors.toList());
            
            // 获取分值列表（不使用）
            List<Object> scoreObjects = (List<Object>) requestBody.get("scores");
            
            // 调用服务层方法选择题目
            boolean success = assignmentService.selectQuestions(assignmentId, questionIds);
            
            if (success) {
                return Result.success("题目选择成功", true);
            } else {
                return Result.error("题目选择失败");
            }
        } catch (Exception e) {
            log.error("选择题目异常: {}", e.getMessage(), e);
            return Result.error("选择题目失败: " + e.getMessage());
        }
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
        try {
            // 获取当前登录用户ID
            Long userId = securityUtil.getCurrentUserId();
            if (userId == null) {
                return Result.error("未登录");
            }
            
            // 如果未指定创建者ID，则使用当前用户ID
            if (createdBy == null) {
                createdBy = userId;
            }
            
            // 从题目表中获取所有知识点
            List<String> knowledgePoints = assignmentService.getKnowledgePoints(courseId, createdBy);
            
            return Result.success(knowledgePoints);
        } catch (Exception e) {
            log.error("获取知识点列表异常: {}", e.getMessage(), e);
            return Result.error("获取知识点列表失败: " + e.getMessage());
        }
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
        try {
            // 获取当前登录用户ID
            Long userId = securityUtil.getCurrentUserId();
            if (userId == null) {
                return Result.error("未登录");
            }
            
            // 如果未指定创建者ID，则使用当前用户ID
            if (createdBy == null) {
                createdBy = userId;
            }
            
            // 查询题目列表并按题型分类
            Map<String, List<Map<String, Object>>> questionsByType = assignmentService.getQuestionsByType(
                    courseId, questionType, difficulty, knowledgePoint, createdBy, keyword);
            
            return Result.success(questionsByType);
        } catch (Exception e) {
            log.error("获取题目列表异常: {}", e.getMessage(), e);
            return Result.error("获取题目列表失败: " + e.getMessage());
        }
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
        Page<Object> page = new Page<>(pageRequest.getPageNum(), pageRequest.getPageSize());
        IPage<Map<String, Object>> submissionPage = assignmentService.getAssignmentSubmissions(assignmentId, page, status);
        
        // 转换为PageResponse
        PageResponse<Map<String, Object>> pageResponse = new PageResponse<>();
        pageResponse.setList(submissionPage.getRecords());
        pageResponse.setTotal(submissionPage.getTotal());
        pageResponse.setPageNum(pageRequest.getPageNum());
        pageResponse.setPageSize(pageRequest.getPageSize());
        
        return Result.success(pageResponse);
    }
    
    /**
     * 智能批改单个作业提交
     * @param submissionId 提交记录ID
     * @param referenceAnswer 参考答案
     * @param submittedFile 提交的文件
     * @return 批改结果
     */
    @PostMapping("/submissions/{submissionId}/ai-grade")
    public Result<GradingResultDTO> aiGradeSubmission(
            @PathVariable Long submissionId,
            @RequestParam String referenceAnswer,
            @RequestParam MultipartFile submittedFile) {
        
        try {
            GradingResultDTO result = difyGradingService.gradeAssignmentSubmission(submissionId, referenceAnswer, submittedFile);
            return Result.success(result);
        } catch (Exception e) {
            return Result.error("智能批改失败：" + e.getMessage());
        }
    }
    
    /**
     * 批量智能批改作业
     * @param assignmentId 作业ID
     * @param referenceAnswer 参考答案
     * @return 批改结果汇总
     */
    @PostMapping("/{assignmentId}/ai-grade-batch")
    public Result<Map<String, Object>> aiGradeBatch(
            @PathVariable Long assignmentId,
            @RequestParam String referenceAnswer) {
        
        try {
            Map<String, Object> result = difyGradingService.batchGradeAssignments(assignmentId, referenceAnswer);
            return Result.success(result);
        } catch (Exception e) {
            return Result.error("批量智能批改失败：" + e.getMessage());
        }
    }
    
    /**
     * 上传文件到 Dify 平台
     * @param file 要上传的文件
     * @return 上传结果
     */
    @PostMapping("/upload-file")
    public Result<Map<String, Object>> uploadFile(@RequestParam MultipartFile file) {
        Long userId = securityUtil.getCurrentUserId();
        if (userId == null) {
            return Result.error("未登录");
        }
        
        try {
            Map<String, Object> result = difyGradingService.uploadFile(file, "teacher-" + userId);
            if (result != null) {
                return Result.success(result);
            } else {
                return Result.error("文件上传失败");
            }
        } catch (Exception e) {
            return Result.error("文件上传失败：" + e.getMessage());
        }
    }
    
    /**
     * 获取批改工作流状态
     * @param workflowRunId 工作流运行ID
     * @return 工作流状态
     */
    @GetMapping("/grading-status/{workflowRunId}")
    public Result<Map<String, Object>> getGradingStatus(@PathVariable String workflowRunId) {
        try {
            Map<String, Object> result = difyGradingService.getWorkflowStatus(workflowRunId);
            if (result != null) {
                return Result.success(result);
            } else {
                return Result.error("获取状态失败");
            }
        } catch (Exception e) {
            return Result.error("获取状态失败：" + e.getMessage());
        }
    }
    
    /**
     * 设置作业参考答案
     * @param assignmentId 作业ID
     * @param requestBody 包含参考答案的请求体
     * @return 是否成功
     */
    @PutMapping("/{assignmentId}/reference-answer")
    public Result<Boolean> setReferenceAnswer(
            @PathVariable Long assignmentId,
            @RequestBody Map<String, Object> requestBody) {
        
        try {
            String referenceAnswer = (String) requestBody.get("referenceAnswer");
            if (referenceAnswer == null || referenceAnswer.trim().isEmpty()) {
                return Result.error("参考答案不能为空");
            }
            
            // 设置参考答案
            boolean success = assignmentService.setReferenceAnswer(assignmentId, referenceAnswer);
            return Result.success(success);
        } catch (Exception e) {
            return Result.error("设置参考答案失败：" + e.getMessage());
        }
    }
}