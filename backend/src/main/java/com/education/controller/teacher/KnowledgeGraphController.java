package com.education.controller.teacher;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
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
 * 教师端-知识图谱控制器
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
@Slf4j
@RestController
@RequestMapping("/api/teacher/knowledge-graph")
@Tag(name = "教师端-知识图谱管理", description = "提供知识图谱的生成、查看、编辑等功能")
public class KnowledgeGraphController {

    @Autowired
    private KnowledgeGraphService knowledgeGraphService;

    @Autowired
    private SecurityUtil securityUtil;

    /**
     * 生成知识图谱
     */
    @PostMapping("/generate")
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
    @GetMapping("/course/{courseId}")
    @Operation(summary = "获取课程知识图谱", description = "获取指定课程的所有知识图谱")
    public Result<List<KnowledgeGraph>> getCourseKnowledgeGraphs(
            @Parameter(description = "课程ID") @PathVariable Long courseId) {
        
        List<KnowledgeGraph> graphs = knowledgeGraphService.getCourseKnowledgeGraphs(courseId);
        return Result.success(graphs);
    }

    /**
     * 获取知识图谱详情
     */
    @GetMapping("/{graphId}")
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
    @PutMapping("/{graphId}")
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
    @DeleteMapping("/{graphId}")
    @Operation(summary = "删除知识图谱", description = "删除指定的知识图谱")
    public Result<Void> deleteKnowledgeGraph(
            @Parameter(description = "图谱ID") @PathVariable Long graphId) {
        
        Long userId = securityUtil.getCurrentUserId();
        knowledgeGraphService.deleteKnowledgeGraph(graphId, userId);
        return Result.success("知识图谱删除成功");
    }

    /**
     * 分页查询知识图谱
     */
    @PostMapping("/page")
    @Operation(summary = "分页查询知识图谱", description = "分页查询知识图谱列表")
    public Result<IPage<KnowledgeGraph>> getKnowledgeGraphsPage(
            @Parameter(description = "页码") @RequestParam(defaultValue = "1") Integer page,
            @Parameter(description = "每页大小") @RequestParam(defaultValue = "10") Integer size,
            @Parameter(description = "查询条件") @RequestBody(required = false) com.education.dto.KnowledgeGraphDTO.QueryRequest query) {
        
        Page<KnowledgeGraph> pageParam = new Page<>(page, size);
        if (query == null) {
            query = new com.education.dto.KnowledgeGraphDTO.QueryRequest();
        }
        
        IPage<KnowledgeGraph> result = knowledgeGraphService.getKnowledgeGraphsPage(pageParam, query);
        return Result.success(result);
    }

    /**
     * 搜索知识图谱
     */
    @GetMapping("/search")
    @Operation(summary = "搜索知识图谱", description = "根据关键词搜索知识图谱")
    public Result<List<KnowledgeGraph>> searchKnowledgeGraphs(
            @Parameter(description = "搜索关键词") @RequestParam String keyword) {
        
        Long userId = securityUtil.getCurrentUserId();
        List<KnowledgeGraph> graphs = knowledgeGraphService.searchKnowledgeGraphs(keyword, userId);
        return Result.success(graphs);
    }

    /**
     * 获取我创建的知识图谱
     */
    @GetMapping("/my")
    @Operation(summary = "我的知识图谱", description = "获取当前教师创建的所有知识图谱")
    public Result<List<KnowledgeGraph>> getMyKnowledgeGraphs() {
        Long userId = securityUtil.getCurrentUserId();
        // 直接使用selectByCreatorId方法获取当前用户创建的知识图谱
        List<KnowledgeGraph> graphs = knowledgeGraphService.selectByCreatorId(userId);
        return Result.success(graphs);
    }

    /**
     * 知识点分析
     */
    @PostMapping("/analyze")
    @Operation(summary = "知识点分析", description = "分析知识图谱的结构和学习路径")
    public Result<com.education.dto.KnowledgeGraphDTO.AnalysisResponse> analyzeKnowledgeGraph(
            @Parameter(description = "分析请求") @RequestBody com.education.dto.KnowledgeGraphDTO.AnalysisRequest request) {
        
        com.education.dto.KnowledgeGraphDTO.AnalysisResponse response = knowledgeGraphService.analyzeKnowledgeGraph(request);
        return Result.success(response);
    }

    /**
     * 获取图谱生成任务状态
     */
    @GetMapping("/task/{taskId}")
    @Operation(summary = "查询生成任务状态", description = "查询知识图谱生成任务的进度")
    public Result<String> getTaskStatus(
            @Parameter(description = "任务ID") @PathVariable String taskId) {
        
        // 这里可以实现任务状态查询逻辑
        return Result.success("任务已完成", "completed");
    }
} 