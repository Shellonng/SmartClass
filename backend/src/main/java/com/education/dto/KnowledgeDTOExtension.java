package com.education.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

/**
 * 知识点相关DTO扩展类
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
public class KnowledgeDTOExtension {

    /**
     * 知识图谱创建请求DTO
     */
    public static class KnowledgeGraphCreateRequest {
        @NotBlank(message = "知识图谱名称不能为空")
        private String graphName;
        private String description;
        private Long courseId;
        private String graphType; // TREE, NETWORK, HIERARCHY
        private Map<String, Object> metadata;
        private String status;
        private Long creatorId;
        
        // Getters and Setters
        public String getGraphName() { return graphName; }
        public void setGraphName(String graphName) { this.graphName = graphName; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }

        public String getGraphType() { return graphType; }
        public void setGraphType(String graphType) { this.graphType = graphType; }
        public Map<String, Object> getMetadata() { return metadata; }
        public void setMetadata(Map<String, Object> metadata) { this.metadata = metadata; }
        public Long getCourseId() { return courseId; }
        public void setCourseId(Long courseId) { this.courseId = courseId; }
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
        public Long getCreatorId() { return creatorId; }
        public void setCreatorId(Long creatorId) { this.creatorId = creatorId; }
    }

    /**
     * 知识图谱响应DTO
     */
    public static class KnowledgeGraphResponse {
        private Long graphId;
        private String graphName;
        private String description;
        private Long courseId;
        private Long creatorId;
        private String courseName;
        private String graphType;
        private Integer nodeCount;
        private Integer relationCount;
        private String status;
        private LocalDateTime createTime;
        private LocalDateTime updateTime;
        private Map<String, Object> metadata;
        
        // Getters and Setters
        public Long getGraphId() { return graphId; }
        public void setGraphId(Long graphId) { this.graphId = graphId; }
        public String getGraphName() { return graphName; }
        public void setGraphName(String graphName) { this.graphName = graphName; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }

        public String getCourseName() { return courseName; }
        public void setCourseName(String courseName) { this.courseName = courseName; }
        public String getGraphType() { return graphType; }
        public void setGraphType(String graphType) { this.graphType = graphType; }
        public Integer getNodeCount() { return nodeCount; }
        public void setNodeCount(Integer nodeCount) { this.nodeCount = nodeCount; }
        public Integer getRelationCount() { return relationCount; }
        public void setRelationCount(Integer relationCount) { this.relationCount = relationCount; }

        public LocalDateTime getCreateTime() { return createTime; }
        public void setCreateTime(LocalDateTime createTime) { this.createTime = createTime; }
        public LocalDateTime getUpdateTime() { return updateTime; }
        public void setUpdateTime(LocalDateTime updateTime) { this.updateTime = updateTime; }
        public Map<String, Object> getMetadata() { return metadata; }
        public void setMetadata(Map<String, Object> metadata) { this.metadata = metadata; }
        public Long getCourseId() { return courseId; }
        public void setCourseId(Long courseId) { this.courseId = courseId; }
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
        public Long getCreatorId() { return creatorId; }
        public void setCreatorId(Long creatorId) { this.creatorId = creatorId; }
    }

    /**
     * 知识图谱详情响应DTO
     */
    public static class KnowledgeGraphDetailResponse {
        private Long graphId;
        private String graphName;
        private String description;
        private Long courseId;
        private Long creatorId;
        private String courseName;
        private String graphType;
        private List<KnowledgeNodeResponse> nodes;
        private List<KnowledgeRelationResponse> relations;
        private String status;
        private LocalDateTime createTime;
        private LocalDateTime updateTime;
        private Map<String, Object> metadata;
        
        // Getters and Setters
        public Long getGraphId() { return graphId; }
        public void setGraphId(Long graphId) { this.graphId = graphId; }
        public String getGraphName() { return graphName; }
        public void setGraphName(String graphName) { this.graphName = graphName; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }

        public String getCourseName() { return courseName; }
        public void setCourseName(String courseName) { this.courseName = courseName; }
        public String getGraphType() { return graphType; }
        public void setGraphType(String graphType) { this.graphType = graphType; }
        public List<KnowledgeNodeResponse> getNodes() { return nodes; }
        public void setNodes(List<KnowledgeNodeResponse> nodes) { this.nodes = nodes; }
        public List<KnowledgeRelationResponse> getRelations() { return relations; }
        public void setRelations(List<KnowledgeRelationResponse> relations) { this.relations = relations; }

        public LocalDateTime getCreateTime() { return createTime; }
        public void setCreateTime(LocalDateTime createTime) { this.createTime = createTime; }
        public LocalDateTime getUpdateTime() { return updateTime; }
        public void setUpdateTime(LocalDateTime updateTime) { this.updateTime = updateTime; }
        public Map<String, Object> getMetadata() { return metadata; }
        public void setMetadata(Map<String, Object> metadata) { this.metadata = metadata; }
        public Long getCourseId() { return courseId; }
        public void setCourseId(Long courseId) { this.courseId = courseId; }
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
        public Long getCreatorId() { return creatorId; }
        public void setCreatorId(Long creatorId) { this.creatorId = creatorId; }
    }

    /**
     * 知识图谱更新请求DTO
     */
    public static class KnowledgeGraphUpdateRequest {
        private String graphName;
        private String description;
        private String graphType;
        private Long courseId;
        private String status;
        private Map<String, Object> metadata;
        private Long creatorId;
        
        // Getters and Setters
        public String getGraphName() { return graphName; }
        public void setGraphName(String graphName) { this.graphName = graphName; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
        public String getGraphType() { return graphType; }
        public void setGraphType(String graphType) { this.graphType = graphType; }
        public Map<String, Object> getMetadata() { return metadata; }
        public void setMetadata(Map<String, Object> metadata) { this.metadata = metadata; }
        public Long getCourseId() { return courseId; }
        public void setCourseId(Long courseId) { this.courseId = courseId; }
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
        public Long getCreatorId() { return creatorId; }
        public void setCreatorId(Long creatorId) { this.creatorId = creatorId; }
    }

    /**
     * 知识节点创建请求DTO
     */
    public static class KnowledgeNodeCreateRequest {
        @NotBlank(message = "节点名称不能为空")
        private String nodeName;
        private String description;
        private String content;
        private String nodeType; // CONCEPT, SKILL, FACT, PROCEDURE
        private String difficulty;
        private Double positionX;
        private Double positionY;
        private Map<String, Object> properties;
        private Integer difficultyLevel;
        private Integer importanceLevel;
        
        // Getters and Setters
        public String getNodeName() { return nodeName; }
        public void setNodeName(String nodeName) { this.nodeName = nodeName; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
        public String getContent() { return content; }
        public void setContent(String content) { this.content = content; }
        public String getNodeType() { return nodeType; }
        public void setNodeType(String nodeType) { this.nodeType = nodeType; }
        public String getDifficulty() { return difficulty; }
        public void setDifficulty(String difficulty) { this.difficulty = difficulty; }
        public Double getPositionX() { return positionX; }
        public void setPositionX(Double positionX) { this.positionX = positionX; }
        public Double getPositionY() { return positionY; }
        public void setPositionY(Double positionY) { this.positionY = positionY; }
        public Map<String, Object> getProperties() { return properties; }
        public void setProperties(Map<String, Object> properties) { this.properties = properties; }
        public Integer getDifficultyLevel() { return difficultyLevel; }
        public void setDifficultyLevel(Integer difficultyLevel) { this.difficultyLevel = difficultyLevel; }
        public Integer getImportanceLevel() { return importanceLevel; }
        public void setImportanceLevel(Integer importanceLevel) { this.importanceLevel = importanceLevel; }
    }

    /**
     * 知识节点响应DTO
     */
    public static class KnowledgeNodeResponse {
        private Long nodeId;
        private String nodeName;
        private String description;
        private String content;
        private String nodeType;
        private String difficulty;
        private Integer difficultyLevel;
        private Integer importanceLevel;
        private Double positionX;
        private Double positionY;
        private Long graphId;
        private String status;
        private LocalDateTime createTime;
        private LocalDateTime updateTime;
        private Map<String, Object> properties;
        
        // Getters and Setters
        public Long getNodeId() { return nodeId; }
        public void setNodeId(Long nodeId) { this.nodeId = nodeId; }
        public String getNodeName() { return nodeName; }
        public void setNodeName(String nodeName) { this.nodeName = nodeName; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
        public String getContent() { return content; }
        public void setContent(String content) { this.content = content; }
        public String getNodeType() { return nodeType; }
        public void setNodeType(String nodeType) { this.nodeType = nodeType; }
        public String getDifficulty() { return difficulty; }
        public void setDifficulty(String difficulty) { this.difficulty = difficulty; }
        public Double getPositionX() { return positionX; }
        public void setPositionX(Double positionX) { this.positionX = positionX; }
        public Double getPositionY() { return positionY; }
        public void setPositionY(Double positionY) { this.positionY = positionY; }
        public Long getGraphId() { return graphId; }
        public void setGraphId(Long graphId) { this.graphId = graphId; }
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
        public LocalDateTime getCreateTime() { return createTime; }
        public void setCreateTime(LocalDateTime createTime) { this.createTime = createTime; }
        public LocalDateTime getUpdateTime() { return updateTime; }
        public void setUpdateTime(LocalDateTime updateTime) { this.updateTime = updateTime; }
        public Map<String, Object> getProperties() { return properties; }
        public void setProperties(Map<String, Object> properties) { this.properties = properties; }
        public Integer getDifficultyLevel() { return difficultyLevel; }
        public void setDifficultyLevel(Integer difficultyLevel) { this.difficultyLevel = difficultyLevel; }
        public Integer getImportanceLevel() { return importanceLevel; }
        public void setImportanceLevel(Integer importanceLevel) { this.importanceLevel = importanceLevel; }
    }

    /**
     * 知识节点更新请求DTO
     */
    public static class KnowledgeNodeUpdateRequest {
        private String nodeName;
        private String description;
        private String content;
        private String nodeType;
        private String difficulty;
        private Double positionX;
        private Double positionY;
        private Map<String, Object> properties;
        private Integer difficultyLevel;
        private Integer importanceLevel;
        
        // Getters and Setters
        public String getNodeName() { return nodeName; }
        public void setNodeName(String nodeName) { this.nodeName = nodeName; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
        public String getContent() { return content; }
        public void setContent(String content) { this.content = content; }
        public String getNodeType() { return nodeType; }
        public void setNodeType(String nodeType) { this.nodeType = nodeType; }
        public String getDifficulty() { return difficulty; }
        public void setDifficulty(String difficulty) { this.difficulty = difficulty; }
        public Double getPositionX() { return positionX; }
        public void setPositionX(Double positionX) { this.positionX = positionX; }
        public Double getPositionY() { return positionY; }
        public void setPositionY(Double positionY) { this.positionY = positionY; }
        public Map<String, Object> getProperties() { return properties; }
        public void setProperties(Map<String, Object> properties) { this.properties = properties; }
        public Integer getDifficultyLevel() { return difficultyLevel; }
        public void setDifficultyLevel(Integer difficultyLevel) { this.difficultyLevel = difficultyLevel; }
        public Integer getImportanceLevel() { return importanceLevel; }
        public void setImportanceLevel(Integer importanceLevel) { this.importanceLevel = importanceLevel; }
    }

    /**
     * 知识关系创建请求DTO
     */
    public static class KnowledgeRelationCreateRequest {
        @NotNull(message = "源节点ID不能为空")
        private Long sourceNodeId;
        @NotNull(message = "目标节点ID不能为空")
        private Long targetNodeId;
        @NotBlank(message = "关系类型不能为空")
        private String relationType; // PREREQUISITE, RELATED, CONTAINS, EXTENDS
        private String description;
        private Double weight;
        private Map<String, Object> properties;
        private Integer difficultyLevel;
        private Integer importanceLevel;
        
        // Getters and Setters
        public Long getSourceNodeId() { return sourceNodeId; }
        public void setSourceNodeId(Long sourceNodeId) { this.sourceNodeId = sourceNodeId; }
        public Long getTargetNodeId() { return targetNodeId; }
        public void setTargetNodeId(Long targetNodeId) { this.targetNodeId = targetNodeId; }
        public String getRelationType() { return relationType; }
        public void setRelationType(String relationType) { this.relationType = relationType; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
        public Double getWeight() { return weight; }
        public void setWeight(Double weight) { this.weight = weight; }
        public Map<String, Object> getProperties() { return properties; }
        public void setProperties(Map<String, Object> properties) { this.properties = properties; }
        public Integer getDifficultyLevel() { return difficultyLevel; }
        public void setDifficultyLevel(Integer difficultyLevel) { this.difficultyLevel = difficultyLevel; }
        public Integer getImportanceLevel() { return importanceLevel; }
        public void setImportanceLevel(Integer importanceLevel) { this.importanceLevel = importanceLevel; }
    }

    /**
     * 知识关系响应DTO
     */
    public static class KnowledgeRelationResponse {
        private Long relationId;
        private Long sourceNodeId;
        private String sourceNodeName;
        private Long targetNodeId;
        private String targetNodeName;
        private String relationType;
        private String description;
        private Double weight;
        private Long graphId;
        private String status;
        private LocalDateTime createTime;
        private LocalDateTime updateTime;
        private Map<String, Object> properties;
        private Integer difficultyLevel;
        private Integer importanceLevel;
        
        // Getters and Setters
        public Long getRelationId() { return relationId; }
        public void setRelationId(Long relationId) { this.relationId = relationId; }
        public Long getSourceNodeId() { return sourceNodeId; }
        public void setSourceNodeId(Long sourceNodeId) { this.sourceNodeId = sourceNodeId; }
        public String getSourceNodeName() { return sourceNodeName; }
        public void setSourceNodeName(String sourceNodeName) { this.sourceNodeName = sourceNodeName; }
        public Long getTargetNodeId() { return targetNodeId; }
        public void setTargetNodeId(Long targetNodeId) { this.targetNodeId = targetNodeId; }
        public String getTargetNodeName() { return targetNodeName; }
        public void setTargetNodeName(String targetNodeName) { this.targetNodeName = targetNodeName; }
        public String getRelationType() { return relationType; }
        public void setRelationType(String relationType) { this.relationType = relationType; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
        public Double getWeight() { return weight; }
        public void setWeight(Double weight) { this.weight = weight; }
        public Long getGraphId() { return graphId; }
        public void setGraphId(Long graphId) { this.graphId = graphId; }
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
        public LocalDateTime getCreateTime() { return createTime; }
        public void setCreateTime(LocalDateTime createTime) { this.createTime = createTime; }
        public LocalDateTime getUpdateTime() { return updateTime; }
        public void setUpdateTime(LocalDateTime updateTime) { this.updateTime = updateTime; }
        public Map<String, Object> getProperties() { return properties; }
        public void setProperties(Map<String, Object> properties) { this.properties = properties; }
        public Integer getDifficultyLevel() { return difficultyLevel; }
        public void setDifficultyLevel(Integer difficultyLevel) { this.difficultyLevel = difficultyLevel; }
        public Integer getImportanceLevel() { return importanceLevel; }
        public void setImportanceLevel(Integer importanceLevel) { this.importanceLevel = importanceLevel; }
    }

    /**
     * 知识关系更新请求DTO
     */
    public static class KnowledgeRelationUpdateRequest {
        private String relationType;
        private String description;
        private Double weight;
        private Map<String, Object> properties;
        private Integer difficultyLevel;
        private Integer importanceLevel;
        
        // Getters and Setters
        public String getRelationType() { return relationType; }
        public void setRelationType(String relationType) { this.relationType = relationType; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
        public Double getWeight() { return weight; }
        public void setWeight(Double weight) { this.weight = weight; }
        public Map<String, Object> getProperties() { return properties; }
        public void setProperties(Map<String, Object> properties) { this.properties = properties; }
        public Integer getDifficultyLevel() { return difficultyLevel; }
        public void setDifficultyLevel(Integer difficultyLevel) { this.difficultyLevel = difficultyLevel; }
        public Integer getImportanceLevel() { return importanceLevel; }
        public void setImportanceLevel(Integer importanceLevel) { this.importanceLevel = importanceLevel; }
    }
}