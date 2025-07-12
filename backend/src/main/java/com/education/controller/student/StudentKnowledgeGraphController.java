package com.education.controller.student;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.education.dto.KnowledgeGraphDTO;
import com.education.dto.common.Result;
import com.education.entity.KnowledgeGraph;
import com.education.service.KnowledgeGraphService;
import com.education.mapper.ChapterMapper;
import com.education.mapper.SectionMapper;
import com.education.entity.Chapter;
import com.education.entity.Section;
import com.education.entity.Course;
import com.education.mapper.CourseMapper;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import jakarta.servlet.http.HttpServletRequest;
import java.util.*;
import java.util.stream.Collectors;

/**
 * 学生端知识图谱控制器
 */
@Tag(name = "学生-知识图谱", description = "学生端知识图谱相关接口")
@RestController
@RequestMapping("/api/student/knowledge-graph")
@Slf4j
@CrossOrigin
public class StudentKnowledgeGraphController {

    @Autowired
    private KnowledgeGraphService knowledgeGraphService;

    @Autowired
    private ChapterMapper chapterMapper;
    
    @Autowired
    private SectionMapper sectionMapper;
    
    @Autowired
    private CourseMapper courseMapper;

    @GetMapping("/course/{courseId}")
    @Operation(summary = "获取课程知识图谱", description = "获取指定课程的所有公开知识图谱")
    public Result<List<KnowledgeGraph>> getCourseKnowledgeGraphs(
            @Parameter(description = "课程ID") @PathVariable Long courseId) {
        
        try {
        List<KnowledgeGraph> graphs = knowledgeGraphService.getCourseKnowledgeGraphs(courseId);
        // 只返回已发布的图谱
        List<KnowledgeGraph> publicGraphs = graphs.stream()
                .filter(graph -> "published".equals(graph.getStatus()))
                    .collect(Collectors.toList());
            
            // 如果没有找到任何图谱，返回一个临时的图谱ID
            if (publicGraphs.isEmpty()) {
                KnowledgeGraph tempGraph = new KnowledgeGraph();
                tempGraph.setId(-1L);
                tempGraph.setTitle("临时知识图谱");
                tempGraph.setDescription("系统自动生成的知识图谱");
                tempGraph.setCourseId(courseId);
                tempGraph.setStatus("published");
                publicGraphs.add(tempGraph);
            }
        
        return Result.success(publicGraphs);
        } catch (Exception e) {
            log.error("获取课程知识图谱失败", e);
            return Result.error("获取课程知识图谱失败: " + e.getMessage());
        }
    }
    
    @GetMapping("/{id}")
    @Operation(summary = "获取知识图谱详情", description = "获取指定知识图谱的详细信息")
    public Result<KnowledgeGraphDTO.GraphData> getKnowledgeGraph(
            @Parameter(description = "知识图谱ID") @PathVariable Long id,
            @Parameter(description = "课程ID") @RequestParam(required = false) Long courseId) {
        
        try {
            // 如果是临时知识图谱（id=-1），则根据课程ID生成临时图谱
            if (id == -1 && courseId != null) {
                // 获取课程信息
                Course course = courseMapper.selectById(courseId);
                if (course == null) {
                    return Result.error("课程不存在");
                }
                
                // 获取课程章节
                List<Chapter> chapters = chapterMapper.selectByCourseId(courseId);
                if (chapters.isEmpty()) {
                    return Result.error("课程没有章节");
                }
                
                // 获取章节ID列表
                List<Long> chapterIds = chapters.stream()
                        .map(Chapter::getId)
                        .collect(Collectors.toList());
                
                // 创建临时知识图谱生成请求
                KnowledgeGraphDTO.GenerationRequest request = new KnowledgeGraphDTO.GenerationRequest();
                request.setCourseId(courseId);
                request.setChapterIds(chapterIds);
                request.setTitle(course.getTitle() + " - 知识图谱");
                request.setDescription("自动生成的知识图谱");
                
                // 获取章节和小节信息，构建课程内容
                StringBuilder courseContent = new StringBuilder();
                for (Chapter chapter : chapters) {
                    courseContent.append("# ").append(chapter.getTitle()).append("\n");
                    if (chapter.getDescription() != null) {
                        courseContent.append(chapter.getDescription()).append("\n\n");
                    }
                    
                    List<Section> sections = sectionMapper.selectByChapterId(chapter.getId());
                    for (Section section : sections) {
                        courseContent.append("## ").append(section.getTitle()).append("\n");
                        if (section.getDescription() != null) {
                            courseContent.append(section.getDescription()).append("\n\n");
                        }
                    }
                }
                request.setCourseContent(courseContent.toString());
                
                // 生成临时知识图谱
                KnowledgeGraphDTO.GenerationResponse response = knowledgeGraphService.generateTempKnowledgeGraph(request);
                
                if (response != null && response.getGraphData() != null) {
                    return Result.success(response.getGraphData());
                } else {
                    return Result.error("生成知识图谱失败");
                }
            }
            
            // 正常获取已有知识图谱
            KnowledgeGraphDTO.GraphData graphData = knowledgeGraphService.getKnowledgeGraphData(id);
            if (graphData == null) {
                return Result.error("知识图谱不存在");
            }
            
        return Result.success(graphData);
        } catch (Exception e) {
            log.error("获取知识图谱详情失败", e);
            return Result.error("获取知识图谱详情失败: " + e.getMessage());
        }
    }

    /**
     * 获取公开的知识图谱
     */
    @GetMapping("/public")
    @Operation(summary = "获取公开知识图谱", description = "获取所有公开的优质知识图谱")
    public Result<List<KnowledgeGraph>> getPublicKnowledgeGraphs(
            @Parameter(description = "返回数量限制") @RequestParam(defaultValue = "10") Integer limit) {
        
        List<KnowledgeGraph> graphs = knowledgeGraphService.getPublicKnowledgeGraphs(limit);
        return Result.success(graphs);
    }

    /**
     * 搜索知识图谱
     */
    @GetMapping("/search")
    @Operation(summary = "搜索知识图谱", description = "根据关键词搜索公开的知识图谱")
    public Result<List<KnowledgeGraph>> searchKnowledgeGraphs(
            @Parameter(description = "搜索关键词") @RequestParam String keyword,
            HttpServletRequest request) {
        
        // 从session获取用户ID
        Long userId = (Long) request.getSession().getAttribute("userId");
        if (userId == null) {
            userId = 0L; // 匿名用户
        }
        
        List<KnowledgeGraph> graphs = knowledgeGraphService.searchKnowledgeGraphs(keyword, userId);
        return Result.success(graphs);
    }

    /**
     * 分页查询知识图谱
     */
    @PostMapping("/page")
    @Operation(summary = "分页查询知识图谱", description = "分页查询公开的知识图谱列表")
    public Result<IPage<KnowledgeGraph>> getKnowledgeGraphsPage(
            @Parameter(description = "页码") @RequestParam(defaultValue = "1") Integer page,
            @Parameter(description = "每页大小") @RequestParam(defaultValue = "10") Integer size,
            @Parameter(description = "查询条件") @RequestBody(required = false) com.education.dto.KnowledgeGraphDTO.QueryRequest query) {
        
        Page<KnowledgeGraph> pageParam = new Page<>(page, size);
        if (query == null) {
            query = new com.education.dto.KnowledgeGraphDTO.QueryRequest();
        }
        
        IPage<KnowledgeGraph> result = knowledgeGraphService.getKnowledgeGraphsPage(pageParam, query);
        
        // 过滤只返回已发布的图谱
        List<KnowledgeGraph> publicRecords = result.getRecords().stream()
                .filter(graph -> "published".equals(graph.getStatus()))
                .toList();
        result.setRecords(publicRecords);
        
        return Result.success(result);
    }

    /**
     * 知识点学习路径分析
     */
    @PostMapping("/analyze")
    @Operation(summary = "知识点学习分析", description = "根据学生学习情况分析知识图谱和推荐学习路径")
    public Result<com.education.dto.KnowledgeGraphDTO.AnalysisResponse> analyzeKnowledgeGraph(
            @Parameter(description = "分析请求") @RequestBody com.education.dto.KnowledgeGraphDTO.AnalysisRequest request,
            HttpServletRequest httpRequest) {
        
        // 从session获取学生ID
        Long studentId = (Long) httpRequest.getSession().getAttribute("userId");
        if (studentId != null && request.getStudentId() == null) {
            request.setStudentId(studentId);
        }
        
        com.education.dto.KnowledgeGraphDTO.AnalysisResponse response = knowledgeGraphService.analyzeKnowledgeGraph(request);
        return Result.success(response);
    }

    /**
     * 获取推荐的知识图谱
     */
    @GetMapping("/recommended")
    @Operation(summary = "获取推荐图谱", description = "根据学生的学习情况推荐合适的知识图谱")
    public Result<List<KnowledgeGraph>> getRecommendedKnowledgeGraphs(
            HttpServletRequest request) {
        
        // 目前返回热门的公开图谱，后续可以根据学生学习数据进行个性化推荐
        List<KnowledgeGraph> graphs = knowledgeGraphService.getPublicKnowledgeGraphs(5);
        return Result.success("为您推荐以下知识图谱", graphs);
    }

    /**
     * 标记图谱学习进度
     */
    @PostMapping("/{graphId}/progress")
    @Operation(summary = "更新学习进度", description = "标记知识点的学习进度")
    public Result<Void> updateLearningProgress(
            @Parameter(description = "图谱ID") @PathVariable Long graphId,
            @Parameter(description = "节点ID") @RequestParam String nodeId,
            @Parameter(description = "是否完成") @RequestParam Boolean completed,
            HttpServletRequest request) {
        
        // 从session获取学生ID
        Long studentId = (Long) request.getSession().getAttribute("userId");
        if (studentId == null) {
            return Result.error("请先登录");
        }
        
        // 这里可以实现学习进度记录逻辑
        log.info("学生 {} 更新知识图谱 {} 节点 {} 学习进度: {}", studentId, graphId, nodeId, completed);
        
        return Result.success("学习进度更新成功", null);
    }

    /**
     * 获取学习统计
     */
    @GetMapping("/statistics")
    @Operation(summary = "学习统计", description = "获取学生的知识图谱学习统计数据")
    public Result<Object> getLearningStatistics(HttpServletRequest request) {
        
        // 从session获取学生ID
        Long studentId = (Long) request.getSession().getAttribute("userId");
        if (studentId == null) {
            return Result.error("请先登录");
        }
        
        // 这里可以实现学习统计逻辑
        return Result.success("暂无统计数据", null);
    }
} 