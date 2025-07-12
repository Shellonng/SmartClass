package com.education.service;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.education.dto.DifyDTO;
import com.education.dto.KnowledgeGraphDTO;
import com.education.entity.*;
import com.education.exception.BusinessException;
import com.education.mapper.*;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.*;
import java.util.stream.Collectors;

/**
 * 知识图谱服务类
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
@Slf4j
@Service
public class KnowledgeGraphService {

    @Autowired
    private KnowledgeGraphMapper knowledgeGraphMapper;

    @Autowired
    private CourseMapper courseMapper;

    @Autowired
    private ChapterMapper chapterMapper;

    @Autowired
    private SectionMapper sectionMapper;

    @Autowired
    private DifyService difyService;

    @Autowired
    private ObjectMapper objectMapper;

    /**
     * 生成知识图谱
     */
    @Transactional
    public KnowledgeGraphDTO.GenerationResponse generateKnowledgeGraph(
            KnowledgeGraphDTO.GenerationRequest request, Long userId) {
        
        log.info("开始生成知识图谱，用户ID: {}, 请求: {}", userId, request);

        try {
            // 1. 验证请求参数
            validateGenerationRequest(request, userId);

            // 2. 获取课程和章节信息
            Course course = courseMapper.selectById(request.getCourseId());
            if (course == null) {
                throw new BusinessException("课程不存在");
            }

            List<Chapter> chapters = getChaptersByIds(request.getChapterIds());
            if (chapters.isEmpty()) {
                throw new BusinessException("未找到有效的章节信息");
            }

            // 3. 构建课程内容信息
            enrichRequestWithCourseContent(request, course, chapters);

            // 4. 调用Dify生成图谱
            KnowledgeGraphDTO.GenerationResponse response = difyService.generateKnowledgeGraph(request, userId.toString());

            // 5. 如果生成成功，保存到数据库
            if ("completed".equals(response.getStatus()) && response.getGraphData() != null) {
                KnowledgeGraph savedGraph = saveKnowledgeGraph(response.getGraphData(), request, userId);
                response.getGraphData().setId(savedGraph.getId());
                log.info("知识图谱生成并保存成功，图谱ID: {}", savedGraph.getId());
            }

            return response;

        } catch (Exception e) {
            log.error("生成知识图谱失败: {}", e.getMessage(), e);
            return KnowledgeGraphDTO.GenerationResponse.builder()
                    .status("failed")
                    .errorMessage(e.getMessage())
                    .build();
        }
    }

    /**
     * 生成临时知识图谱（不保存到数据库）
     */
    public KnowledgeGraphDTO.GenerationResponse generateTempKnowledgeGraph(
            KnowledgeGraphDTO.GenerationRequest request) {
        
        log.info("开始生成临时知识图谱，请求: {}", request);

        try {
            // 1. 验证请求参数
            if (request.getCourseId() == null) {
                throw new BusinessException("课程ID不能为空");
            }

            // 2. 调用Dify生成图谱
            KnowledgeGraphDTO.GenerationResponse response = difyService.generateKnowledgeGraph(request, "system");

            // 3. 不保存到数据库，直接返回结果
            log.info("临时知识图谱生成成功");
            return response;

        } catch (Exception e) {
            log.error("生成临时知识图谱失败: {}", e.getMessage(), e);
            return KnowledgeGraphDTO.GenerationResponse.builder()
                    .status("failed")
                    .errorMessage(e.getMessage())
                    .build();
        }
    }

    /**
     * 获取课程的知识图谱列表
     */
    public List<KnowledgeGraph> getCourseKnowledgeGraphs(Long courseId) {
        return knowledgeGraphMapper.selectByCourseId(courseId);
    }

    /**
     * 获取知识图谱详情
     */
    public KnowledgeGraphDTO.GraphData getKnowledgeGraphDetail(Long graphId, Long userId) {
        KnowledgeGraph graph = knowledgeGraphMapper.selectById(graphId);
        if (graph == null) {
            throw new BusinessException("知识图谱不存在");
        }

        // 检查权限 - 移除对isPublic的检查
        if (!graph.getCreatorId().equals(userId)) {
            throw new BusinessException("无权访问此知识图谱");
        }

        // 移除增加访问次数的代码
        // knowledgeGraphMapper.incrementViewCount(graphId);

        try {
            return objectMapper.readValue(graph.getGraphData(), KnowledgeGraphDTO.GraphData.class);
        } catch (Exception e) {
            log.error("解析知识图谱数据失败: {}", e.getMessage(), e);
            throw new BusinessException("知识图谱数据格式错误");
        }
    }

    /**
     * 获取知识图谱数据
     */
    public KnowledgeGraphDTO.GraphData getKnowledgeGraphData(Long graphId) {
        KnowledgeGraph graph = knowledgeGraphMapper.selectById(graphId);
        if (graph == null) {
            throw new BusinessException("知识图谱不存在");
        }

        try {
            return objectMapper.readValue(graph.getGraphData(), KnowledgeGraphDTO.GraphData.class);
        } catch (Exception e) {
            log.error("解析知识图谱数据失败: {}", e.getMessage(), e);
            throw new BusinessException("知识图谱数据格式错误");
        }
    }

    /**
     * 更新知识图谱
     */
    @Transactional
    public void updateKnowledgeGraph(Long graphId, KnowledgeGraphDTO.GraphData graphData, Long userId) {
        KnowledgeGraph graph = knowledgeGraphMapper.selectById(graphId);
        if (graph == null) {
            throw new BusinessException("知识图谱不存在");
        }

        // 检查权限
        if (!graph.getCreatorId().equals(userId)) {
            throw new BusinessException("无权修改此知识图谱");
        }

        try {
            graph.setGraphData(objectMapper.writeValueAsString(graphData));
            graph.setUpdateTime(LocalDateTime.now());
            knowledgeGraphMapper.updateById(graph);
            log.info("知识图谱更新成功，图谱ID: {}", graphId);
        } catch (Exception e) {
            log.error("更新知识图谱失败: {}", e.getMessage(), e);
            throw new BusinessException("更新知识图谱失败: " + e.getMessage());
        }
    }

    /**
     * 删除知识图谱
     */
    @Transactional
    public void deleteKnowledgeGraph(Long graphId, Long userId) {
        KnowledgeGraph graph = knowledgeGraphMapper.selectById(graphId);
        if (graph == null) {
            throw new BusinessException("知识图谱不存在");
        }

        // 检查权限
        if (!graph.getCreatorId().equals(userId)) {
            throw new BusinessException("无权删除此知识图谱");
        }

        knowledgeGraphMapper.deleteById(graphId);
        log.info("知识图谱删除成功，图谱ID: {}", graphId);
    }

    /**
     * 搜索知识图谱
     */
    public List<KnowledgeGraph> searchKnowledgeGraphs(String keyword, Long userId) {
        if (keyword == null || keyword.trim().isEmpty()) {
            return Collections.emptyList();
        }
        return knowledgeGraphMapper.searchGraphs(keyword.trim(), userId);
    }

    /**
     * 分页查询知识图谱
     */
    public IPage<KnowledgeGraph> getKnowledgeGraphsPage(Page<KnowledgeGraph> page, KnowledgeGraphDTO.QueryRequest query) {
        LambdaQueryWrapper<KnowledgeGraph> wrapper = new LambdaQueryWrapper<>();
        
        // 添加基础条件，确保SQL语句正确
        wrapper.ne(KnowledgeGraph::getStatus, "archived"); // 使用实体的字段引用，避免列名歧义
        
        if (query.getCourseId() != null) {
            wrapper.eq(KnowledgeGraph::getCourseId, query.getCourseId());
        }
        
        if (query.getGraphType() != null && !query.getGraphType().isEmpty()) {
            wrapper.eq(KnowledgeGraph::getGraphType, query.getGraphType());
        }
        
        if (query.getKeyword() != null && !query.getKeyword().isEmpty()) {
            wrapper.and(w -> w.like(KnowledgeGraph::getTitle, query.getKeyword())
                    .or().like(KnowledgeGraph::getDescription, query.getKeyword()));
        }
        
        wrapper.orderByDesc(KnowledgeGraph::getUpdateTime);

        return knowledgeGraphMapper.selectPageWithRelations(page, wrapper);
    }

    /**
     * 查询用户创建的知识图谱
     */
    public List<KnowledgeGraph> selectByCreatorId(Long creatorId) {
        return knowledgeGraphMapper.selectByCreatorId(creatorId);
    }

    /**
     * 获取公开的知识图谱
     */
    public List<KnowledgeGraph> getPublicKnowledgeGraphs(Integer limit) {
        // 由于没有is_public列，改为获取已发布状态的知识图谱
        return knowledgeGraphMapper.selectPublishedGraphs(limit != null ? limit : 10);
    }

    /**
     * 发布知识图谱（设置为公开）
     */
    @Transactional
    public void publishKnowledgeGraph(Long graphId, Long userId) {
        KnowledgeGraph graph = knowledgeGraphMapper.selectById(graphId);
        if (graph == null) {
            throw new BusinessException("知识图谱不存在");
        }

        // 检查权限
        if (!graph.getCreatorId().equals(userId)) {
            throw new BusinessException("无权操作此知识图谱");
        }

        // 移除设置isPublic的代码
        // graph.setIsPublic(true);
        graph.setStatus("published");
        graph.setUpdateTime(LocalDateTime.now());
        knowledgeGraphMapper.updateById(graph);
        log.info("知识图谱已发布，图谱ID: {}", graphId);
    }

    /**
     * 取消发布知识图谱
     */
    @Transactional
    public void unpublishKnowledgeGraph(Long graphId, Long userId) {
        KnowledgeGraph graph = knowledgeGraphMapper.selectById(graphId);
        if (graph == null) {
            throw new BusinessException("知识图谱不存在");
        }

        // 检查权限
        if (!graph.getCreatorId().equals(userId)) {
            throw new BusinessException("无权操作此知识图谱");
        }

        // 移除设置isPublic的代码
        // graph.setIsPublic(false);
        graph.setStatus("draft");  // 改为草稿状态
        graph.setUpdateTime(LocalDateTime.now());
        knowledgeGraphMapper.updateById(graph);
        log.info("知识图谱已取消发布，图谱ID: {}", graphId);
    }

    /**
     * 获取知识图谱生成任务状态
     */
    public KnowledgeGraphDTO.GenerationResponse getTaskStatus(String taskId) {
        log.info("查询知识图谱生成任务状态: {}", taskId);
        
        // 调用DifyService获取任务状态
        try {
            // 调用Dify API获取任务状态
            DifyDTO.DifyResponse difyResponse = difyService.getTaskStatus(taskId, "knowledge-graph");
            
            if ("completed".equals(difyResponse.getStatus())) {
                // 任务完成，解析结果
                try {
                    // 从DifyResponse中提取数据并转换为GraphData
                    Map<String, Object> data = difyResponse.getData();
                    if (data == null) {
                        throw new RuntimeException("响应数据为空");
                    }
                    
                    // 构建GraphData对象
                    KnowledgeGraphDTO.GraphData graphData = parseGraphDataFromDifyResponse(data);
                    
                    return KnowledgeGraphDTO.GenerationResponse.builder()
                            .status("completed")
                            .taskId(taskId)
                            .graphData(graphData)
                            .suggestions((String) data.get("suggestions"))
                            .build();
                } catch (Exception e) {
                    log.error("解析知识图谱数据失败: {}", e.getMessage(), e);
                    return KnowledgeGraphDTO.GenerationResponse.builder()
                            .status("failed")
                            .errorMessage("解析知识图谱数据失败: " + e.getMessage())
                            .taskId(taskId)
                            .build();
                }
            } else if ("failed".equals(difyResponse.getStatus())) {
                // 任务失败
                return KnowledgeGraphDTO.GenerationResponse.builder()
                        .status("failed")
                        .errorMessage("生成任务失败: " + difyResponse.getError())
                        .taskId(taskId)
                        .build();
            } else {
                // 任务仍在处理中
                return KnowledgeGraphDTO.GenerationResponse.builder()
                        .status(difyResponse.getStatus())
                        .taskId(taskId)
                        .build();
            }
        } catch (Exception e) {
            log.error("获取任务状态失败: {}", e.getMessage(), e);
            return KnowledgeGraphDTO.GenerationResponse.builder()
                    .status("error")
                    .errorMessage("获取任务状态失败: " + e.getMessage())
                    .taskId(taskId)
                    .build();
        }
    }

    /**
     * 知识点分析
     */
    public KnowledgeGraphDTO.AnalysisResponse analyzeKnowledgeGraph(KnowledgeGraphDTO.AnalysisRequest request) {
        // 这里可以根据学生的学习数据进行个性化分析
        // 目前返回基础分析结果
        Map<String, Object> analysis = new HashMap<>();
        analysis.put("node_count", 0);
        analysis.put("edge_count", 0);
        analysis.put("complexity", "medium");

        return KnowledgeGraphDTO.AnalysisResponse.builder()
                .analysis(analysis)
                .learningPath(Arrays.asList("基础概念", "核心理论", "实践应用"))
                .keyPoints(Arrays.asList("重点1", "重点2", "重点3"))
                .difficultPoints(Arrays.asList("难点1", "难点2"))
                .build();
    }

    // ======================== 私有方法 ========================

    /**
     * 验证生成请求
     */
    private void validateGenerationRequest(KnowledgeGraphDTO.GenerationRequest request, Long userId) {
        if (request.getCourseId() == null) {
            throw new BusinessException("课程ID不能为空");
        }
        
        if (request.getChapterIds() == null || request.getChapterIds().isEmpty()) {
            throw new BusinessException("至少需要选择一个章节");
        }

        // 验证用户是否有权限访问该课程
        Course course = courseMapper.selectById(request.getCourseId());
        if (course == null) {
            throw new BusinessException("课程不存在");
        }
        
        // 这里可以添加更多权限验证逻辑
    }

    /**
     * 根据ID列表获取章节
     */
    private List<Chapter> getChaptersByIds(List<Long> chapterIds) {
        if (chapterIds == null || chapterIds.isEmpty()) {
            return Collections.emptyList();
        }
        
        LambdaQueryWrapper<Chapter> wrapper = new LambdaQueryWrapper<>();
        wrapper.in(Chapter::getId, chapterIds);
        return chapterMapper.selectList(wrapper);
    }

    /**
     * 丰富请求内容 - 从数据库提取完整的课程结构化信息
     */
    private void enrichRequestWithCourseContent(KnowledgeGraphDTO.GenerationRequest request, 
                                               Course course, List<Chapter> chapters) {
        try {
            // 获取章节下的小节信息
            for (Chapter chapter : chapters) {
                LambdaQueryWrapper<Section> sectionWrapper = new LambdaQueryWrapper<>();
                sectionWrapper.eq(Section::getChapterId, chapter.getId())
                            .orderByAsc(Section::getSortOrder);
                List<Section> sections = sectionMapper.selectList(sectionWrapper);
                chapter.setSections(sections);
            }
            
            // 构建结构化的课程数据，用于Dify智能体分析
            Map<String, Object> courseStructure = new HashMap<>();
            
            // 课程基本信息
            Map<String, Object> courseInfo = new HashMap<>();
            courseInfo.put("id", course.getId());
            courseInfo.put("title", course.getTitle());
            courseInfo.put("description", course.getDescription());
            courseInfo.put("courseType", course.getCourseType());
            courseInfo.put("credit", course.getCredit());
            courseStructure.put("course", courseInfo);
            
            // 章节和小节的详细内容
            List<Map<String, Object>> chaptersData = new ArrayList<>();
            for (Chapter chapter : chapters) {
                Map<String, Object> chapterData = new HashMap<>();
                chapterData.put("id", chapter.getId());
                chapterData.put("title", chapter.getTitle());
                chapterData.put("description", chapter.getDescription());
                chapterData.put("sortOrder", chapter.getSortOrder());
                
                // 小节信息
                List<Map<String, Object>> sectionsData = new ArrayList<>();
                if (chapter.getSections() != null) {
                    for (Section section : chapter.getSections()) {
                        Map<String, Object> sectionData = new HashMap<>();
                        sectionData.put("id", section.getId());
                        sectionData.put("title", section.getTitle());
                        sectionData.put("description", section.getDescription());
                        sectionData.put("duration", section.getDuration());
                        sectionData.put("sortOrder", section.getSortOrder());
                        sectionsData.add(sectionData);
                    }
                }
                chapterData.put("sections", sectionsData);
                chaptersData.add(chapterData);
            }
            courseStructure.put("chapters", chaptersData);
            
            // 生成知识图谱的指导信息
            String guidanceText = generateGraphGuidance(request.getGraphType(), request.getDepth());
            
            // 将结构化数据序列化为JSON字符串传递给Dify
            String structuredContent = objectMapper.writeValueAsString(courseStructure);
            
            // 构建完整的提示信息
            StringBuilder promptBuilder = new StringBuilder();
            promptBuilder.append("请根据以下课程结构化数据生成知识图谱：\n\n");
            promptBuilder.append("=== 课程数据 ===\n");
            promptBuilder.append(structuredContent).append("\n\n");
            promptBuilder.append("=== 生成要求 ===\n");
            promptBuilder.append(guidanceText).append("\n");
            
            if (request.getIncludePrerequisites()) {
                promptBuilder.append("- 需要包含知识点的先修关系\n");
            }
            if (request.getIncludeApplications()) {
                promptBuilder.append("- 需要包含知识点的应用关系\n");
            }
            
            if (request.getAdditionalRequirements() != null && !request.getAdditionalRequirements().isEmpty()) {
                promptBuilder.append("- 额外要求: ").append(request.getAdditionalRequirements()).append("\n");
            }
            
            // 设置完整的生成提示
            request.setAdditionalRequirements(promptBuilder.toString());
            
            log.info("已构建结构化课程数据，包含{}个章节，共{}个小节", 
                    chapters.size(), 
                    chapters.stream().mapToInt(c -> c.getSections() != null ? c.getSections().size() : 0).sum());
                    
        } catch (Exception e) {
            log.error("构建课程结构化数据失败: {}", e.getMessage(), e);
            // 降级为文本描述
            buildTextBasedContent(request, course, chapters);
        }
    }
    
    /**
     * 生成知识图谱的指导文本
     */
    private String generateGraphGuidance(String graphType, Integer depth) {
        StringBuilder guidance = new StringBuilder();
        
        guidance.append("图谱类型要求：\n");
        switch (graphType) {
            case "concept":
                guidance.append("- 重点提取和连接概念性知识点\n");
                guidance.append("- 突出概念间的层次关系和逻辑关系\n");
                break;
            case "skill":
                guidance.append("- 重点提取技能型知识点\n");
                guidance.append("- 突出技能的递进关系和依赖关系\n");
                break;
            case "comprehensive":
                guidance.append("- 全面提取概念、技能、应用等各类知识点\n");
                guidance.append("- 构建完整的知识体系结构\n");
                break;
        }
        
        guidance.append("\n深度级别要求：\n");
        guidance.append("- 当前深度级别：").append(depth).append("/5\n");
        if (depth <= 2) {
            guidance.append("- 提取主要的核心知识点，保持结构简洁\n");
        } else if (depth <= 3) {
            guidance.append("- 提取核心和重要的知识点，包含适当的细节\n");
        } else {
            guidance.append("- 提取详细的知识点，包含深层次的关联关系\n");
        }
        
        guidance.append("\n输出格式要求：\n");
        guidance.append("- 必须严格按照ECharts图表所需的JSON格式输出\n");
        guidance.append("- 直接输出可用于ECharts的option配置，不要包含任何思考过程或其他文本\n");
        guidance.append("- 禁止使用<think>标签或输出任何思考过程\n");
        guidance.append("- 只输出一个完整的JSON对象，不要有任何额外的解释或注释\n");
        guidance.append("- 使用graph系列类型，适合知识图谱展示\n");
        guidance.append("- 节点ID必须唯一，建议使用 'node_' + 序号格式\n");
        guidance.append("- 每个节点必须定义name、symbolSize、category等属性\n");
        guidance.append("- 每个关系必须定义source、target等属性\n");
        guidance.append("- 必须定义categories数组，对应节点类型\n");
        guidance.append("- JSON结构示例：\n");
        guidance.append("{\n");
        guidance.append("  \"title\": {\n");
        guidance.append("    \"text\": \"知识图谱标题\"\n");
        guidance.append("  },\n");
        guidance.append("  \"tooltip\": {},\n");
        guidance.append("  \"legend\": {\n");
        guidance.append("    \"data\": [\"主题\", \"章节\", \"概念\", \"技能\"]\n");
        guidance.append("  },\n");
        guidance.append("  \"series\": [{\n");
        guidance.append("    \"name\": \"知识图谱\",\n");
        guidance.append("    \"type\": \"graph\",\n");
        guidance.append("    \"layout\": \"force\",\n");
        guidance.append("    \"data\": [\n");
        guidance.append("      {\n");
        guidance.append("        \"id\": \"node_1\",\n");
        guidance.append("        \"name\": \"节点名称\",\n");
        guidance.append("        \"symbolSize\": 50,\n");
        guidance.append("        \"category\": 0,\n");
        guidance.append("        \"value\": 1,\n");
        guidance.append("        \"draggable\": true,\n");
        guidance.append("        \"tooltip\": { \"formatter\": \"节点详细描述\" }\n");
        guidance.append("      }\n");
        guidance.append("    ],\n");
        guidance.append("    \"links\": [\n");
        guidance.append("      {\n");
        guidance.append("        \"source\": \"node_1\",\n");
        guidance.append("        \"target\": \"node_2\",\n");
        guidance.append("        \"value\": 1,\n");
        guidance.append("        \"tooltip\": { \"formatter\": \"关系描述\" }\n");
        guidance.append("      }\n");
        guidance.append("    ],\n");
        guidance.append("    \"categories\": [\n");
        guidance.append("      { \"name\": \"主题\" },\n");
        guidance.append("      { \"name\": \"章节\" },\n");
        guidance.append("      { \"name\": \"概念\" },\n");
        guidance.append("      { \"name\": \"技能\" }\n");
        guidance.append("    ],\n");
        guidance.append("    \"roam\": true,\n");
        guidance.append("    \"label\": {\n");
        guidance.append("      \"show\": true,\n");
        guidance.append("      \"position\": \"right\"\n");
        guidance.append("    },\n");
        guidance.append("    \"force\": {\n");
        guidance.append("      \"repulsion\": 100,\n");
        guidance.append("      \"edgeLength\": 100\n");
        guidance.append("    }\n");
        guidance.append("  }]\n");
        guidance.append("}\n");
        
        guidance.append("\n关于节点映射：\n");
        guidance.append("- 将topic(主题)类型的节点映射为category=0\n");
        guidance.append("- 将chapter(章节)类型的节点映射为category=1\n");
        guidance.append("- 将concept(概念)类型的节点映射为category=2\n");
        guidance.append("- 将skill(技能)类型的节点映射为category=3\n");
        guidance.append("- 将节点level属性映射为symbolSize，公式为: level * 10 + 30\n");
        
        guidance.append("\n关于关系映射：\n");
        guidance.append("- 将contains(包含)关系映射为实线，value=1\n");
        guidance.append("- 将prerequisite(先修)关系映射为虚线，value=2\n");
        guidance.append("- 将application(应用)关系映射为点线，value=3\n");
        guidance.append("- 将similar(相似)关系映射为细线，value=1\n");
        
        guidance.append("\n最终输出要求：\n");
        guidance.append("- 只输出一个完整的JSON对象，不要包含任何额外的文本、解释或思考过程\n");
        guidance.append("- 确保JSON格式正确，可以直接被解析\n");
        guidance.append("- 禁止使用markdown代码块或其他格式标记\n");
        
        return guidance.toString();
    }
    
    /**
     * 降级方案：构建文本形式的课程内容
     */
    private void buildTextBasedContent(KnowledgeGraphDTO.GenerationRequest request, 
                                     Course course, List<Chapter> chapters) {
        StringBuilder contentBuilder = new StringBuilder();
        contentBuilder.append("课程: ").append(course.getTitle()).append("\n");
        contentBuilder.append("课程描述: ").append(course.getDescription()).append("\n\n");
        
        for (Chapter chapter : chapters) {
            contentBuilder.append("章节: ").append(chapter.getTitle()).append("\n");
            contentBuilder.append("章节描述: ").append(chapter.getDescription()).append("\n");
            
            if (chapter.getSections() != null) {
                for (Section section : chapter.getSections()) {
                    contentBuilder.append("  小节: ").append(section.getTitle()).append("\n");
                    contentBuilder.append("  小节描述: ").append(section.getDescription()).append("\n");
                }
            }
            contentBuilder.append("\n");
        }
        
        request.setAdditionalRequirements(contentBuilder.toString());
    }

    /**
     * 保存知识图谱到数据库
     */
    private KnowledgeGraph saveKnowledgeGraph(KnowledgeGraphDTO.GraphData graphData, 
                                            KnowledgeGraphDTO.GenerationRequest request, Long userId) {
        try {
            KnowledgeGraph graph = new KnowledgeGraph();
            graph.setCourseId(request.getCourseId());
            graph.setTitle(graphData.getTitle() != null ? graphData.getTitle() : "知识图谱");
            graph.setDescription(graphData.getDescription());
            graph.setGraphType(request.getGraphType());
            graph.setGraphData(objectMapper.writeValueAsString(graphData));
            graph.setCreatorId(userId);
            graph.setStatus("published");
            // 移除以下三行代码，因为数据库中没有这些列
            // graph.setVersion(1);
            // graph.setIsPublic(false);
            // graph.setViewCount(0);
            graph.setCreateTime(LocalDateTime.now());
            graph.setUpdateTime(LocalDateTime.now());

            knowledgeGraphMapper.insert(graph);
            return graph;
            
        } catch (Exception e) {
            log.error("保存知识图谱失败: {}", e.getMessage(), e);
            throw new BusinessException("保存知识图谱失败: " + e.getMessage());
        }
    }
    
    /**
     * 从Dify响应中解析知识图谱数据
     */
    @SuppressWarnings("unchecked")
    private KnowledgeGraphDTO.GraphData parseGraphDataFromDifyResponse(Map<String, Object> data) {
        try {
            // 解析节点数据
            List<Map<String, Object>> nodesList = (List<Map<String, Object>>) data.get("nodes");
            List<KnowledgeGraphDTO.GraphNode> nodes = new ArrayList<>();
            
            if (nodesList != null) {
                for (Map<String, Object> nodeData : nodesList) {
                    KnowledgeGraphDTO.GraphNode node = KnowledgeGraphDTO.GraphNode.builder()
                            .id((String) nodeData.get("id"))
                            .name((String) nodeData.get("name"))
                            .type((String) nodeData.get("type"))
                            .level(nodeData.get("level") instanceof Integer ? (Integer) nodeData.get("level") : 1)
                            .description((String) nodeData.get("description"))
                            .chapterId(nodeData.get("chapter_id") != null ? 
                                    Long.valueOf(nodeData.get("chapter_id").toString()) : null)
                            .sectionId(nodeData.get("section_id") != null ? 
                                    Long.valueOf(nodeData.get("section_id").toString()) : null)
                            .style(parseNodeStyle((Map<String, Object>) nodeData.get("style")))
                            .position(parseNodePosition((Map<String, Object>) nodeData.get("position")))
                            .properties((Map<String, Object>) nodeData.get("properties"))
                            .build();
                    nodes.add(node);
                }
            }
            
            // 解析边数据
            List<Map<String, Object>> edgesList = (List<Map<String, Object>>) data.get("edges");
            List<KnowledgeGraphDTO.GraphEdge> edges = new ArrayList<>();
            
            if (edgesList != null) {
                for (Map<String, Object> edgeData : edgesList) {
                    KnowledgeGraphDTO.GraphEdge edge = KnowledgeGraphDTO.GraphEdge.builder()
                            .id((String) edgeData.get("id"))
                            .source((String) edgeData.get("source"))
                            .target((String) edgeData.get("target"))
                            .type((String) edgeData.get("type"))
                            .description((String) edgeData.get("description"))
                            .weight(edgeData.get("weight") != null ? 
                                    Double.valueOf(edgeData.get("weight").toString()) : null)
                            .style(parseEdgeStyle((Map<String, Object>) edgeData.get("style")))
                            .properties((Map<String, Object>) edgeData.get("properties"))
                            .build();
                    edges.add(edge);
                }
            }
            
            return KnowledgeGraphDTO.GraphData.builder()
                    .title((String) data.get("title"))
                    .description((String) data.get("description"))
                    .nodes(nodes)
                    .edges(edges)
                    .metadata((Map<String, Object>) data.get("metadata"))
                    .build();
                    
        } catch (Exception e) {
            log.error("解析图谱数据失败: {}", e.getMessage(), e);
            throw new RuntimeException("解析图谱数据失败", e);
        }
    }
    
    /**
     * 解析节点样式
     */
    private KnowledgeGraphDTO.NodeStyle parseNodeStyle(Map<String, Object> styleData) {
        if (styleData == null) {
            return KnowledgeGraphDTO.NodeStyle.builder().build();
        }
        
        return KnowledgeGraphDTO.NodeStyle.builder()
                .color((String) styleData.get("color"))
                .size(styleData.get("size") instanceof Integer ? (Integer) styleData.get("size") : null)
                .shape((String) styleData.get("shape"))
                .fontSize(styleData.get("fontSize") instanceof Integer ? (Integer) styleData.get("fontSize") : null)
                .highlighted(styleData.get("highlighted") instanceof Boolean ? (Boolean) styleData.get("highlighted") : null)
                .build();
    }
    
    /**
     * 解析边样式
     */
    private KnowledgeGraphDTO.EdgeStyle parseEdgeStyle(Map<String, Object> styleData) {
        if (styleData == null) {
            return KnowledgeGraphDTO.EdgeStyle.builder().build();
        }
        
        return KnowledgeGraphDTO.EdgeStyle.builder()
                .color((String) styleData.get("color"))
                .width(styleData.get("width") instanceof Integer ? (Integer) styleData.get("width") : null)
                .lineType((String) styleData.get("lineType"))
                .showArrow(styleData.get("showArrow") instanceof Boolean ? (Boolean) styleData.get("showArrow") : null)
                .build();
    }
    
    /**
     * 解析节点位置
     */
    private KnowledgeGraphDTO.NodePosition parseNodePosition(Map<String, Object> positionData) {
        if (positionData == null) {
            return KnowledgeGraphDTO.NodePosition.builder().build();
        }
        
        return KnowledgeGraphDTO.NodePosition.builder()
                .x(positionData.get("x") != null ? Double.valueOf(positionData.get("x").toString()) : null)
                .y(positionData.get("y") != null ? Double.valueOf(positionData.get("y").toString()) : null)
                .fixed(positionData.get("fixed") instanceof Boolean ? (Boolean) positionData.get("fixed") : null)
                .build();
    }
} 