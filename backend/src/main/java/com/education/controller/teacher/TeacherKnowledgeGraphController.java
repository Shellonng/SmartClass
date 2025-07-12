package com.education.controller.teacher;

import com.education.dto.KnowledgeGraphDTO;
import com.education.dto.common.Result;
import com.education.entity.KnowledgeGraph;
import com.education.security.SecurityUtil;
import com.education.service.KnowledgeGraphService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.List;

/**
 * 教师端-知识图谱构建控制器
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
@Slf4j
@RestController
@RequestMapping("/api/teacher/knowledge-graph")
@Tag(name = "教师端-知识图谱构建", description = "提供知识图谱的生成、管理和可视化功能")
public class TeacherKnowledgeGraphController {

    @Autowired
    private KnowledgeGraphService knowledgeGraphService;

    @Autowired
    private SecurityUtil securityUtil;

    /**
     * 生成知识图谱
     */
    @PostMapping("/create")
    @Operation(summary = "生成知识图谱", description = "基于课程章节内容使用AI生成知识图谱")
    public Result<com.education.dto.KnowledgeGraphDTO.GenerationResponse> generateKnowledgeGraph(
            @Parameter(description = "生成请求") @RequestBody com.education.dto.KnowledgeGraphDTO.GenerationRequest request) {
        
        log.info("教师生成知识图谱请求: {}", request);
        Long userId = securityUtil.getCurrentUserId();
        
        com.education.dto.KnowledgeGraphDTO.GenerationResponse response = knowledgeGraphService.generateKnowledgeGraph(request, userId);
        
        if ("completed".equals(response.getStatus())) {
            return Result.success("知识图谱生成成功", response);
        } else if ("failed".equals(response.getStatus())) {
            return Result.error(response.getErrorMessage() != null ? response.getErrorMessage() : "知识图谱生成失败");
        } else {
            return Result.success("知识图谱生成中，请稍候查看结果", response);
        }
    }

    /**
     * 获取课程的知识图谱列表
     */
    @GetMapping("/course/{courseId}/graphs")
    @Operation(summary = "获取课程知识图谱", description = "获取指定课程的所有知识图谱")
    public Result<List<KnowledgeGraph>> getCourseKnowledgeGraphs(
            @Parameter(description = "课程ID") @PathVariable Long courseId) {
        
        List<KnowledgeGraph> graphs = knowledgeGraphService.getCourseKnowledgeGraphs(courseId);
        return Result.success(graphs);
    }

    /**
     * 获取知识图谱详情
     */
    @GetMapping("/detail/{graphId}")
    @Operation(summary = "获取知识图谱详情", description = "获取知识图谱的完整数据")
    public Result<com.education.dto.KnowledgeGraphDTO.GraphData> getKnowledgeGraphDetail(
            @Parameter(description = "图谱ID") @PathVariable Long graphId) {
        
        Long userId = securityUtil.getCurrentUserId();
        com.education.dto.KnowledgeGraphDTO.GraphData graphData = knowledgeGraphService.getKnowledgeGraphDetail(graphId, userId);
        return Result.success(graphData);
    }

    /**
     * 更新知识图谱
     */
    @PutMapping("/update/{graphId}")
    @Operation(summary = "更新知识图谱", description = "更新知识图谱的内容和布局")
    public Result<Void> updateKnowledgeGraph(
            @Parameter(description = "图谱ID") @PathVariable Long graphId,
            @Parameter(description = "图谱数据") @RequestBody com.education.dto.KnowledgeGraphDTO.GraphData graphData) {
        
        Long userId = securityUtil.getCurrentUserId();
        knowledgeGraphService.updateKnowledgeGraph(graphId, graphData, userId);
        return Result.success("知识图谱更新成功");
    }

    /**
     * 删除知识图谱
     */
    @DeleteMapping("/delete/{graphId}")
    @Operation(summary = "删除知识图谱", description = "删除指定的知识图谱")
    public Result<Void> deleteKnowledgeGraph(
            @Parameter(description = "图谱ID") @PathVariable Long graphId) {
        
        Long userId = securityUtil.getCurrentUserId();
        knowledgeGraphService.deleteKnowledgeGraph(graphId, userId);
        return Result.success("知识图谱删除成功");
    }

    /**
     * 发布知识图谱（设置为公开）
     */
    @PutMapping("/{graphId}/publish")
    @Operation(summary = "发布知识图谱", description = "将知识图谱设置为公开可见")
    public Result<Void> publishKnowledgeGraph(
            @Parameter(description = "图谱ID") @PathVariable Long graphId) {
        
        Long userId = securityUtil.getCurrentUserId();
        knowledgeGraphService.publishKnowledgeGraph(graphId, userId);
        return Result.success("知识图谱已发布");
    }

    /**
     * 取消发布知识图谱
     */
    @PutMapping("/{graphId}/unpublish")
    @Operation(summary = "取消发布知识图谱", description = "将知识图谱设置为私有")
    public Result<Void> unpublishKnowledgeGraph(
            @Parameter(description = "图谱ID") @PathVariable Long graphId) {
        
        Long userId = securityUtil.getCurrentUserId();
        knowledgeGraphService.unpublishKnowledgeGraph(graphId, userId);
        return Result.success("知识图谱已取消发布");
    }

    /**
     * 获取知识图谱生成任务状态
     */
    @GetMapping("/task-status/{taskId}")
    @Operation(summary = "获取任务状态", description = "获取知识图谱生成任务的状态")
    public Result<com.education.dto.KnowledgeGraphDTO.GenerationResponse> getTaskStatus(
            @Parameter(description = "任务ID") @PathVariable String taskId) {
        
        com.education.dto.KnowledgeGraphDTO.GenerationResponse response = knowledgeGraphService.getTaskStatus(taskId);
        return Result.success(response);
    }
} 