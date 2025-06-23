package com.education.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import java.time.LocalDateTime;
import java.util.List;

/**
 * 知识点相关DTO
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
public class KnowledgeDTO {

    /**
     * 知识点创建请求DTO
     */
    public static class KnowledgeCreateRequest {
        @NotBlank(message = "知识点名称不能为空")
        private String knowledgeName;
        
        private String description;
        private String content;
        private String difficulty; // EASY, MEDIUM, HARD
        private String category;
        private List<String> tags;
        private Long courseId;
        private Long parentId; // 父知识点ID
        private Integer sortOrder;
        
        // Getters and Setters
        public String getKnowledgeName() { return knowledgeName; }
        public void setKnowledgeName(String knowledgeName) { this.knowledgeName = knowledgeName; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
        public String getContent() { return content; }
        public void setContent(String content) { this.content = content; }
        public String getDifficulty() { return difficulty; }
        public void setDifficulty(String difficulty) { this.difficulty = difficulty; }
        public String getCategory() { return category; }
        public void setCategory(String category) { this.category = category; }
        public List<String> getTags() { return tags; }
        public void setTags(List<String> tags) { this.tags = tags; }
        public Long getCourseId() { return courseId; }
        public void setCourseId(Long courseId) { this.courseId = courseId; }
        public Long getParentId() { return parentId; }
        public void setParentId(Long parentId) { this.parentId = parentId; }
        public Integer getSortOrder() { return sortOrder; }
        public void setSortOrder(Integer sortOrder) { this.sortOrder = sortOrder; }
    }

    /**
     * 知识点响应DTO
     */
    public static class KnowledgeResponse {
        private Long knowledgeId;
        private String knowledgeName;
        private String description;
        private String content;
        private String difficulty;
        private String category;
        private List<String> tags;
        private Long courseId;
        private String courseName;
        private Long parentId;
        private String parentName;
        private Integer sortOrder;
        private String status;
        private LocalDateTime createTime;
        private LocalDateTime updateTime;
        private List<KnowledgeResponse> children;
        
        // Getters and Setters
        public Long getKnowledgeId() { return knowledgeId; }
        public void setKnowledgeId(Long knowledgeId) { this.knowledgeId = knowledgeId; }
        public String getKnowledgeName() { return knowledgeName; }
        public void setKnowledgeName(String knowledgeName) { this.knowledgeName = knowledgeName; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
        public String getContent() { return content; }
        public void setContent(String content) { this.content = content; }
        public String getDifficulty() { return difficulty; }
        public void setDifficulty(String difficulty) { this.difficulty = difficulty; }
        public String getCategory() { return category; }
        public void setCategory(String category) { this.category = category; }
        public List<String> getTags() { return tags; }
        public void setTags(List<String> tags) { this.tags = tags; }
        public Long getCourseId() { return courseId; }
        public void setCourseId(Long courseId) { this.courseId = courseId; }
        public String getCourseName() { return courseName; }
        public void setCourseName(String courseName) { this.courseName = courseName; }
        public Long getParentId() { return parentId; }
        public void setParentId(Long parentId) { this.parentId = parentId; }
        public String getParentName() { return parentName; }
        public void setParentName(String parentName) { this.parentName = parentName; }
        public Integer getSortOrder() { return sortOrder; }
        public void setSortOrder(Integer sortOrder) { this.sortOrder = sortOrder; }
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
        public LocalDateTime getCreateTime() { return createTime; }
        public void setCreateTime(LocalDateTime createTime) { this.createTime = createTime; }
        public LocalDateTime getUpdateTime() { return updateTime; }
        public void setUpdateTime(LocalDateTime updateTime) { this.updateTime = updateTime; }
        public List<KnowledgeResponse> getChildren() { return children; }
        public void setChildren(List<KnowledgeResponse> children) { this.children = children; }
    }

    /**
     * 知识图谱导入请求DTO
     */
    public static class KnowledgeGraphImportRequest {
        private String importType; // FILE, URL, TEXT
        private String fileUrl;
        private String graphData;
        private Long courseId;
        
        // Getters and Setters
        public String getImportType() { return importType; }
        public void setImportType(String importType) { this.importType = importType; }
        public String getFileUrl() { return fileUrl; }
        public void setFileUrl(String fileUrl) { this.fileUrl = fileUrl; }
        public String getGraphData() { return graphData; }
        public void setGraphData(String graphData) { this.graphData = graphData; }
        public Long getCourseId() { return courseId; }
        public void setCourseId(Long courseId) { this.courseId = courseId; }
    }

    /**
     * 知识图谱导入响应DTO
     */
    public static class KnowledgeGraphImportResponse {
        private Boolean success;
        private String message;
        private Integer importedCount;
        private List<String> errors;
        
        // Getters and Setters
        public Boolean getSuccess() { return success; }
        public void setSuccess(Boolean success) { this.success = success; }
        public String getMessage() { return message; }
        public void setMessage(String message) { this.message = message; }
        public Integer getImportedCount() { return importedCount; }
        public void setImportedCount(Integer importedCount) { this.importedCount = importedCount; }
        public List<String> getErrors() { return errors; }
        public void setErrors(List<String> errors) { this.errors = errors; }
    }
}