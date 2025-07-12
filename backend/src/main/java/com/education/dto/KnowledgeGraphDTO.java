package com.education.dto;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;
import java.util.Map;

/**
 * 知识图谱相关DTO
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
public class KnowledgeGraphDTO {

    /**
     * 知识图谱生成请求
     */
    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    @Schema(description = "知识图谱生成请求")
    public static class GenerationRequest {
        
        @Schema(description = "课程ID")
        private Long courseId;
        
        @Schema(description = "包含的章节ID列表")
        private List<Long> chapterIds;
        
        @Schema(description = "图谱标题")
        private String title;
        
        @Schema(description = "图谱描述")
        private String description;
        
        @Schema(description = "课程内容文本")
        private String courseContent;
        
        @Schema(description = "图谱类型", allowableValues = {"concept", "skill", "comprehensive"})
        @Builder.Default
        private String graphType = "comprehensive";
        
        @Schema(description = "节点深度级别", example = "3")
        @Builder.Default
        private Integer depth = 3;
        
        @Schema(description = "是否包含先修关系", example = "true")
        @Builder.Default
        private Boolean includePrerequisites = true;
        
        @Schema(description = "是否包含应用关系", example = "true")
        @Builder.Default
        private Boolean includeApplications = true;
        
        @Schema(description = "附加要求或说明")
        private String additionalRequirements;
    }

    /**
     * 知识图谱生成响应
     */
    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    @Schema(description = "知识图谱生成响应")
    public static class GenerationResponse {
        
        @Schema(description = "生成状态", allowableValues = {"pending", "processing", "completed", "failed"})
        private String status;
        
        @Schema(description = "任务ID")
        private String taskId;
        
        @Schema(description = "知识图谱数据")
        private GraphData graphData;
        
        @Schema(description = "错误信息")
        private String errorMessage;
        
        @Schema(description = "生成建议")
        private String suggestions;
    }

    /**
     * 知识图谱数据
     */
    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    @Schema(description = "知识图谱数据")
    public static class GraphData {
        
        @Schema(description = "图谱ID")
        private Long id;
        
        @Schema(description = "图谱标题")
        private String title;
        
        @Schema(description = "图谱描述")
        private String description;
        
        @Schema(description = "节点列表")
        private List<GraphNode> nodes;
        
        @Schema(description = "边（关系）列表")
        private List<GraphEdge> edges;
        
        @Schema(description = "图谱元数据")
        private Map<String, Object> metadata;
    }

    /**
     * 知识图谱节点
     */
    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    @Schema(description = "知识图谱节点")
    public static class GraphNode {
        
        @Schema(description = "节点ID")
        private String id;
        
        @Schema(description = "节点名称")
        private String name;
        
        @Schema(description = "节点类型", allowableValues = {"concept", "skill", "topic", "chapter"})
        private String type;
        
        @Schema(description = "节点级别/重要性", example = "1")
        private Integer level;
        
        @Schema(description = "节点描述")
        private String description;
        
        @Schema(description = "相关章节ID")
        private Long chapterId;
        
        @Schema(description = "相关小节ID")
        private Long sectionId;
        
        @Schema(description = "节点样式信息")
        private NodeStyle style;
        
        @Schema(description = "节点位置信息")
        private NodePosition position;
        
        @Schema(description = "扩展属性")
        private Map<String, Object> properties;
    }

    /**
     * 知识图谱边（关系）
     */
    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    @Schema(description = "知识图谱边")
    public static class GraphEdge {
        
        @Schema(description = "边ID")
        private String id;
        
        @Schema(description = "源节点ID")
        private String source;
        
        @Schema(description = "目标节点ID")
        private String target;
        
        @Schema(description = "关系类型", allowableValues = {"prerequisite", "application", "contains", "similar", "extends"})
        private String type;
        
        @Schema(description = "关系描述")
        private String description;
        
        @Schema(description = "关系强度", example = "0.8")
        private Double weight;
        
        @Schema(description = "边样式信息")
        private EdgeStyle style;
        
        @Schema(description = "扩展属性")
        private Map<String, Object> properties;
    }

    /**
     * 节点样式
     */
    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    @Schema(description = "节点样式")
    public static class NodeStyle {
        
        @Schema(description = "节点颜色", example = "#3498db")
        private String color;
        
        @Schema(description = "节点大小", example = "20")
        private Integer size;
        
        @Schema(description = "节点形状", allowableValues = {"circle", "rect", "diamond", "ellipse"})
        private String shape;
        
        @Schema(description = "字体大小", example = "14")
        private Integer fontSize;
        
        @Schema(description = "是否高亮")
        private Boolean highlighted;
    }

    /**
     * 边样式
     */
    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    @Schema(description = "边样式")
    public static class EdgeStyle {
        
        @Schema(description = "边颜色", example = "#7f8c8d")
        private String color;
        
        @Schema(description = "边宽度", example = "2")
        private Integer width;
        
        @Schema(description = "边类型", allowableValues = {"solid", "dashed", "dotted"})
        private String lineType;
        
        @Schema(description = "是否显示箭头")
        private Boolean showArrow;
    }

    /**
     * 节点位置
     */
    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    @Schema(description = "节点位置")
    public static class NodePosition {
        
        @Schema(description = "X坐标")
        private Double x;
        
        @Schema(description = "Y坐标")
        private Double y;
        
        @Schema(description = "是否固定位置")
        private Boolean fixed;
    }

    /**
     * 知识图谱查询请求
     */
    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    @Schema(description = "知识图谱查询请求")
    public static class QueryRequest {
        
        @Schema(description = "课程ID")
        private Long courseId;
        
        @Schema(description = "图谱类型过滤")
        private String graphType;
        
        @Schema(description = "关键词搜索")
        private String keyword;
        
        @Schema(description = "是否包含样式信息")
        @Builder.Default
        private Boolean includeStyle = true;
    }

    /**
     * 知识点分析请求
     */
    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    @Schema(description = "知识点分析请求")
    public static class AnalysisRequest {
        
        @Schema(description = "图谱ID")
        private Long graphId;
        
        @Schema(description = "学生ID（可选，用于个性化分析）")
        private Long studentId;
        
        @Schema(description = "分析类型", allowableValues = {"mastery", "difficulty", "path"})
        private String analysisType;
    }

    /**
     * 知识点分析响应
     */
    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    @Schema(description = "知识点分析响应")
    public static class AnalysisResponse {
        
        @Schema(description = "分析结果")
        private Map<String, Object> analysis;
        
        @Schema(description = "学习路径建议")
        private List<String> learningPath;
        
        @Schema(description = "重点知识点")
        private List<String> keyPoints;
        
        @Schema(description = "难点知识点")
        private List<String> difficultPoints;
    }
} 