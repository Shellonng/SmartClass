package com.education.controller.teacher;

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
import java.util.concurrent.CompletableFuture;
import java.util.HashMap;
import java.util.Map;
import java.util.Arrays;

/**
 * 教师端智能组卷控制器
 * @author Education Platform Team
 */
@Tag(name = "教师端智能组卷", description = "基于Dify AI的智能组卷功能")
@RestController
@RequestMapping("/api/teacher/paper")
@Validated
@Slf4j
public class TeacherPaperController {

    @Autowired
    private DifyService difyService;
    
    @Autowired
    private SecurityUtil securityUtil;

    @Operation(summary = "智能组卷", description = "基于AI生成试卷")
    @PostMapping("/generate")
    public Result<DifyDTO.PaperGenerationResponse> generatePaper(
            @Valid @RequestBody DifyDTO.PaperGenerationRequest request) {
        try {
            log.info("教师{}请求智能组卷: {}", securityUtil.getCurrentUserId(), request);
            
            // 获取当前用户ID作为Dify的用户标识
            String userId = securityUtil.getCurrentUserId().toString();
            
            // 调用Dify生成试卷
            DifyDTO.PaperGenerationResponse response = difyService.generatePaper(request, userId);
            
            // 检查响应状态
            if (response != null && "completed".equals(response.getStatus())) {
                // 成功生成试卷
                log.info("智能组卷成功，生成{}道题目", 
                        response.getQuestions() != null ? response.getQuestions().size() : 0);
                return Result.success(response);
            } else {
                // 如果AI生成失败，但已经使用了本地模板生成试卷
                if (response != null && response.getQuestions() != null && !response.getQuestions().isEmpty()) {
                    log.info("使用本地模板成功生成试卷，共{}道题目", response.getQuestions().size());
                    return Result.success(response);
                }
                
                // 完全失败的情况，尝试使用本地模板
                log.info("AI生成失败，尝试使用本地模板");
                try {
                    DifyDTO.PaperGenerationResponse localResponse = difyService.generateLocalPaperTemplate(request);
                    log.info("使用本地模板成功生成试卷，共{}道题目", 
                            localResponse.getQuestions() != null ? localResponse.getQuestions().size() : 0);
                    return Result.success(localResponse);
                } catch (Exception e) {
                    log.error("本地模板生成也失败: {}", e.getMessage());
                    String errorMsg = response != null && response.getErrorMessage() != null ? 
                            response.getErrorMessage() : "智能组卷失败";
                    return Result.error(errorMsg);
                }
            }
        } catch (Exception e) {
            log.error("智能组卷异常: {}", e.getMessage(), e);
            // 发生异常时也尝试使用本地模板
            try {
                log.info("发生异常，尝试使用本地模板");
                DifyDTO.PaperGenerationResponse localResponse = difyService.generateLocalPaperTemplate(request);
                log.info("使用本地模板成功生成试卷，共{}道题目", localResponse.getQuestions().size());
                return Result.success(localResponse);
            } catch (Exception ex) {
                log.error("本地模板生成也失败: {}", ex.getMessage());
                return Result.error("智能组卷异常: " + e.getMessage());
            }
        }
    }

    @Operation(summary = "异步智能组卷", description = "异步生成试卷，立即返回任务ID")
    @PostMapping("/generate-async")
    public Result<String> generatePaperAsync(
            @Valid @RequestBody DifyDTO.PaperGenerationRequest request) {
        try {
            log.info("教师{}请求异步智能组卷: {}", securityUtil.getCurrentUserId(), request);
            
            String userId = securityUtil.getCurrentUserId().toString();
            
            // 生成任务ID
            String taskId = "paper-task-" + System.currentTimeMillis();
            
            // 异步调用
            CompletableFuture.runAsync(() -> {
                try {
                    // 调用Dify API
                    DifyDTO.DifyResponse response = difyService.callWorkflowApi(
                            "paper-generation", 
                            convertToInputs(request), 
                            userId
                    );
                    
                    // 记录任务完成
                    log.info("异步组卷任务完成，任务ID: {}, 状态: {}", taskId, 
                            response != null ? response.getStatus() : "unknown");
                    
                    // 这里可以更新任务状态到数据库
                    // 例如: taskService.updateTaskStatus(taskId, response.getStatus(), response);
                } catch (Exception e) {
                    log.error("异步组卷任务执行异常，任务ID: {}, 错误: {}", taskId, e.getMessage(), e);
                    // 记录任务失败
                    // 例如: taskService.updateTaskStatus(taskId, "failed", null);
                }
            });
            
            // 保存任务信息到数据库或缓存
            // 如果有TaskService，可以调用: taskService.saveTask(taskId, userId, "paper-generation", request);
            
            log.info("异步组卷任务已提交，任务ID: {}", taskId);
            
            return Result.success("异步组卷任务已提交", taskId);
            
        } catch (Exception e) {
            log.error("异步智能组卷异常: {}", e.getMessage(), e);
            return Result.error(500, "异步组卷任务提交失败: " + e.getMessage());
        }
    }

    @Operation(summary = "查询组卷任务状态", description = "查询异步组卷任务的执行状态")
    @GetMapping("/task/{taskId}")
    public Result<DifyDTO.DifyResponse> getTaskStatus(@PathVariable String taskId) {
        try {
            log.info("查询组卷任务状态: {}", taskId);
            
            DifyDTO.DifyResponse response = difyService.getTaskStatus(taskId, "paper-generation");
            
            return Result.success("任务状态查询成功", response);
            
        } catch (Exception e) {
            log.error("查询任务状态异常: {}", e.getMessage(), e);
            return Result.error(500, "查询任务状态失败: " + e.getMessage());
        }
    }

    @Operation(summary = "预览组卷参数", description = "根据课程和知识点预览可生成的题目类型和数量")
    @PostMapping("/preview")
    public Result<Object> previewPaper(@RequestBody DifyDTO.PaperGenerationRequest request) {
        try {
            log.info("预览组卷参数: {}", request);
            
            // 这里可以调用轻量级的预览API，或者从本地题库统计
            // 暂时返回模拟数据
            return Result.success("参数预览成功", buildPreviewResponse(request));
            
        } catch (Exception e) {
            log.error("预览组卷参数异常: {}", e.getMessage(), e);
            return Result.error(500, "参数预览失败: " + e.getMessage());
        }
    }

    /**
     * 将请求转换为Dify输入参数
     */
    private Map<String, Object> convertToInputs(DifyDTO.PaperGenerationRequest request) {
        Map<String, Object> inputs = new HashMap<>();
        inputs.put("course_id", request.getCourseId());
        inputs.put("knowledge_points", String.join(",", request.getKnowledgePoints()));
        inputs.put("difficulty", request.getDifficulty());
        inputs.put("question_count", request.getQuestionCount());
        inputs.put("duration", request.getDuration());
        inputs.put("total_score", request.getTotalScore());
        inputs.put("additional_requirements", request.getAdditionalRequirements());
        return inputs;
    }

    /**
     * 构建预览响应
     */
    private Object buildPreviewResponse(DifyDTO.PaperGenerationRequest request) {
        Map<String, Object> preview = new HashMap<>();
        preview.put("course_id", request.getCourseId());
        preview.put("estimated_questions", request.getQuestionCount());
        preview.put("estimated_time", "预计生成时间: 2-5分钟");
        preview.put("available_types", Arrays.asList("单选题", "多选题", "判断题", "填空题", "简答题"));
        preview.put("difficulty_distribution", Map.of(
                "EASY", "30%",
                "MEDIUM", "50%", 
                "HARD", "20%"
        ));
        return preview;
    }
} 