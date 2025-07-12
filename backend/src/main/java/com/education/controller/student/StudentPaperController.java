package com.education.controller.student;

import com.education.dto.DifyDTO;
import com.education.dto.common.Result;
import com.education.service.DifyService;
import com.education.security.SecurityUtil;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;

import jakarta.validation.Valid;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.ArrayList;

/**
 * 学生端智能组卷控制器
 * @author Education Platform Team
 */
@Tag(name = "学生端智能组卷", description = "学生个性化练习题生成")
@RestController
@RequestMapping("/api/student/paper")
@Validated
@Slf4j
public class StudentPaperController {

    @Autowired
    private DifyService difyService;
    
    @Autowired
    private SecurityUtil securityUtil;

    @Operation(summary = "生成个性化练习", description = "基于学生学习情况生成个性化练习题")
    @PostMapping("/generate-practice")
    public Result<DifyDTO.PaperGenerationResponse> generatePractice(
            @Valid @RequestBody PersonalizedPracticeRequest request) {
        try {
            Long studentId = securityUtil.getCurrentUserId();
            log.info("学生{}请求生成个性化练习: {}", studentId, request);
            
            // 转换为标准的组卷请求
            DifyDTO.PaperGenerationRequest paperRequest = DifyDTO.PaperGenerationRequest.builder()
                    .courseId(request.getCourseId())
                    .knowledgePoints(request.getWeakKnowledgePoints())
                    .difficulty(determineDifficultyByAbility(request.getAbilityLevel()))
                    .questionCount(request.getQuestionCount())
                    .questionTypes(request.getPreferredQuestionTypes())
                    .totalScore(request.getQuestionCount() * 5) // 默认每题5分
                    .additionalRequirements("个性化练习 - 针对薄弱知识点: " + String.join(",", request.getWeakKnowledgePoints()))
                    .build();
            
            String userId = studentId.toString();
            DifyDTO.PaperGenerationResponse response = difyService.generatePaper(paperRequest, userId);
            
            if ("completed".equals(response.getStatus())) {
                log.info("个性化练习生成成功，为学生{}生成{}道题目", studentId, response.getQuestions().size());
                return Result.success("个性化练习生成成功", response);
            } else if ("failed".equals(response.getStatus())) {
                return Result.error(500, response.getErrorMessage());
            } else {
                return Result.success("练习题生成中，请稍后查询结果", response);
            }
            
        } catch (Exception e) {
            log.error("生成个性化练习异常: {}", e.getMessage(), e);
            return Result.error(500, "生成个性化练习失败: " + e.getMessage());
        }
    }

    @Operation(summary = "生成错题重练", description = "基于错题记录生成重练题目")
    @PostMapping("/generate-retry")
    public Result<DifyDTO.PaperGenerationResponse> generateRetryPractice(
            @RequestBody RetryPracticeRequest request) {
        try {
            Long studentId = securityUtil.getCurrentUserId();
            log.info("学生{}请求生成错题重练: {}", studentId, request);
            
            // 构建基于错题的组卷请求
            DifyDTO.PaperGenerationRequest paperRequest = DifyDTO.PaperGenerationRequest.builder()
                    .courseId(request.getCourseId())
                    .knowledgePoints(request.getErrorKnowledgePoints())
                    .difficulty("MEDIUM") // 错题重练使用中等难度
                    .questionCount(request.getRetryCount())
                    .questionTypes(determineQuestionTypesFromErrors(request.getErrorTypes()))
                    .totalScore(request.getRetryCount() * 5)
                    .additionalRequirements("错题重练 - 基于历史错题类型生成相似题目")
                    .build();
            
            String userId = studentId.toString();
            DifyDTO.PaperGenerationResponse response = difyService.generatePaper(paperRequest, userId);
            
            if ("completed".equals(response.getStatus())) {
                log.info("错题重练生成成功，为学生{}生成{}道题目", studentId, response.getQuestions().size());
                return Result.success("错题重练生成成功", response);
            } else {
                return Result.success("错题重练生成中", response);
            }
            
        } catch (Exception e) {
            log.error("生成错题重练异常: {}", e.getMessage(), e);
            return Result.error(500, "生成错题重练失败: " + e.getMessage());
        }
    }

    @Operation(summary = "智能推荐练习", description = "基于学习进度智能推荐练习题目")
    @GetMapping("/recommend/{courseId}")
    public Result<DifyDTO.PaperGenerationResponse> recommendPractice(
            @PathVariable Long courseId,
            @RequestParam(defaultValue = "10") Integer count) {
        try {
            Long studentId = securityUtil.getCurrentUserId();
            log.info("为学生{}推荐课程{}的练习题目，数量: {}", studentId, courseId, count);
            
            // 这里应该根据学生的学习记录和能力分析推荐
            // 暂时使用模拟数据
            DifyDTO.PaperGenerationRequest paperRequest = DifyDTO.PaperGenerationRequest.builder()
                    .courseId(courseId)
                    .knowledgePoints(Arrays.asList("基础概念", "核心知识点"))
                    .difficulty("MEDIUM")
                    .questionCount(count)
                    .questionTypes(Map.of("SINGLE_CHOICE", count/2, "MULTIPLE_CHOICE", count/2))
                    .totalScore(count * 5)
                    .additionalRequirements("智能推荐练习 - 基于学习进度和能力评估")
                    .build();
            
            String userId = studentId.toString();
            DifyDTO.PaperGenerationResponse response = difyService.generatePaper(paperRequest, userId);
            
            return Result.success("智能推荐练习生成成功", response);
            
        } catch (Exception e) {
            log.error("智能推荐练习异常: {}", e.getMessage(), e);
            return Result.error(500, "智能推荐练习失败: " + e.getMessage());
        }
    }

    @Operation(summary = "获取练习历史", description = "获取学生的练习历史记录")
    @GetMapping("/history")
    public Result<Object> getPracticeHistory(
            @RequestParam(defaultValue = "1") Integer page,
            @RequestParam(defaultValue = "10") Integer size) {
        try {
            Long studentId = securityUtil.getCurrentUserId();
            log.info("获取学生{}的练习历史，页码: {}, 大小: {}", studentId, page, size);
            
            // 这里应该从数据库查询练习历史
            // 暂时返回模拟数据
            Map<String, Object> history = buildMockPracticeHistory(studentId, page, size);
            
            return Result.success("练习历史获取成功", history);
            
        } catch (Exception e) {
            log.error("获取练习历史异常: {}", e.getMessage(), e);
            return Result.error(500, "获取练习历史失败: " + e.getMessage());
        }
    }

    /**
     * 个性化练习请求
     */
    public static class PersonalizedPracticeRequest {
        private Long courseId;
        private List<String> weakKnowledgePoints;
        private String abilityLevel; // LOW, MEDIUM, HIGH
        private Integer questionCount;
        private Map<String, Integer> preferredQuestionTypes;
        
        // getters and setters
        public Long getCourseId() { return courseId; }
        public void setCourseId(Long courseId) { this.courseId = courseId; }
        public List<String> getWeakKnowledgePoints() { return weakKnowledgePoints; }
        public void setWeakKnowledgePoints(List<String> weakKnowledgePoints) { this.weakKnowledgePoints = weakKnowledgePoints; }
        public String getAbilityLevel() { return abilityLevel; }
        public void setAbilityLevel(String abilityLevel) { this.abilityLevel = abilityLevel; }
        public Integer getQuestionCount() { return questionCount; }
        public void setQuestionCount(Integer questionCount) { this.questionCount = questionCount; }
        public Map<String, Integer> getPreferredQuestionTypes() { return preferredQuestionTypes; }
        public void setPreferredQuestionTypes(Map<String, Integer> preferredQuestionTypes) { this.preferredQuestionTypes = preferredQuestionTypes; }
    }

    /**
     * 错题重练请求
     */
    public static class RetryPracticeRequest {
        private Long courseId;
        private List<String> errorKnowledgePoints;
        private List<String> errorTypes;
        private Integer retryCount;
        
        // getters and setters
        public Long getCourseId() { return courseId; }
        public void setCourseId(Long courseId) { this.courseId = courseId; }
        public List<String> getErrorKnowledgePoints() { return errorKnowledgePoints; }
        public void setErrorKnowledgePoints(List<String> errorKnowledgePoints) { this.errorKnowledgePoints = errorKnowledgePoints; }
        public List<String> getErrorTypes() { return errorTypes; }
        public void setErrorTypes(List<String> errorTypes) { this.errorTypes = errorTypes; }
        public Integer getRetryCount() { return retryCount; }
        public void setRetryCount(Integer retryCount) { this.retryCount = retryCount; }
    }

    /**
     * 根据能力水平确定难度
     */
    private String determineDifficultyByAbility(String abilityLevel) {
        switch (abilityLevel.toUpperCase()) {
            case "HIGH": return "HARD";
            case "LOW": return "EASY";
            default: return "MEDIUM";
        }
    }

    /**
     * 根据错题类型确定题型分布
     */
    private Map<String, Integer> determineQuestionTypesFromErrors(List<String> errorTypes) {
        Map<String, Integer> questionTypes = new HashMap<>();
        if (errorTypes.contains("SINGLE_CHOICE")) {
            questionTypes.put("SINGLE_CHOICE", 5);
        }
        if (errorTypes.contains("MULTIPLE_CHOICE")) {
            questionTypes.put("MULTIPLE_CHOICE", 3);
        }
        if (errorTypes.contains("TRUE_FALSE")) {
            questionTypes.put("TRUE_FALSE", 2);
        }
        if (questionTypes.isEmpty()) {
            // 默认题型分布
            questionTypes.put("SINGLE_CHOICE", 3);
            questionTypes.put("MULTIPLE_CHOICE", 2);
        }
        return questionTypes;
    }

    /**
     * 构建模拟练习历史
     */
    private Map<String, Object> buildMockPracticeHistory(Long studentId, Integer page, Integer size) {
        Map<String, Object> result = new HashMap<>();
        result.put("total", 15);
        result.put("page", page);
        result.put("size", size);
        
        List<Map<String, Object>> records = new ArrayList<>();
        for (int i = 0; i < size; i++) {
            Map<String, Object> record = new HashMap<>();
            record.put("id", i + 1);
            record.put("title", "个性化练习 #" + (i + 1));
            record.put("courseId", 1L);
            record.put("courseName", "Java编程基础");
            record.put("questionCount", 10);
            record.put("score", 85 + (int)(Math.random() * 15));
            record.put("createTime", "2025-06-" + (10 + i));
            records.add(record);
        }
        
        result.put("records", records);
        return result;
    }
} 