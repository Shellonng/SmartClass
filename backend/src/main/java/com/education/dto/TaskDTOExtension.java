package com.education.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;
import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

/**
 * 任务相关DTO扩展类
 * 包含更多任务相关的DTO定义
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
public class TaskDTOExtension {

    /**
     * 任务讨论响应DTO
     */
    public static class TaskDiscussionResponse {
        private Long discussionId;
        private Long taskId;
        private String content;
        private String authorName;
        private LocalDateTime createTime;
        private List<TaskDiscussionReply> replies;
        
        // Getters and Setters
        public Long getDiscussionId() { return discussionId; }
        public void setDiscussionId(Long discussionId) { this.discussionId = discussionId; }
        public Long getTaskId() { return taskId; }
        public void setTaskId(Long taskId) { this.taskId = taskId; }
        public String getContent() { return content; }
        public void setContent(String content) { this.content = content; }
        public String getAuthorName() { return authorName; }
        public void setAuthorName(String authorName) { this.authorName = authorName; }
        public LocalDateTime getCreateTime() { return createTime; }
        public void setCreateTime(LocalDateTime createTime) { this.createTime = createTime; }
        public List<TaskDiscussionReply> getReplies() { return replies; }
        public void setReplies(List<TaskDiscussionReply> replies) { this.replies = replies; }
        
        public static class TaskDiscussionReply {
            private Long replyId;
            private String content;
            private String authorName;
            private LocalDateTime createTime;
            
            // Getters and Setters
            public Long getReplyId() { return replyId; }
            public void setReplyId(Long replyId) { this.replyId = replyId; }
            public String getContent() { return content; }
            public void setContent(String content) { this.content = content; }
            public String getAuthorName() { return authorName; }
            public void setAuthorName(String authorName) { this.authorName = authorName; }
            public LocalDateTime getCreateTime() { return createTime; }
            public void setCreateTime(LocalDateTime createTime) { this.createTime = createTime; }
        }
    }

    /**
     * 任务讨论创建请求DTO
     */
    public static class TaskDiscussionCreateRequest {
        @NotNull(message = "任务ID不能为空")
        private Long taskId;
        
        @NotBlank(message = "讨论内容不能为空")
        private String content;
        
        // Getters and Setters
        public Long getTaskId() { return taskId; }
        public void setTaskId(Long taskId) { this.taskId = taskId; }
        public String getContent() { return content; }
        public void setContent(String content) { this.content = content; }
    }

    /**
     * 任务讨论回复请求DTO
     */
    public static class TaskDiscussionReplyRequest {
        @NotBlank(message = "回复内容不能为空")
        private String content;
        
        // Getters and Setters
        public String getContent() { return content; }
        public void setContent(String content) { this.content = content; }
    }

    /**
     * 任务资源响应DTO
     */
    public static class TaskResourceResponse {
        private Long resourceId;
        private String resourceName;
        private String resourceType;
        private String resourceUrl;
        private Long fileSize;
        private LocalDateTime uploadTime;
        
        // Getters and Setters
        public Long getResourceId() { return resourceId; }
        public void setResourceId(Long resourceId) { this.resourceId = resourceId; }
        public String getResourceName() { return resourceName; }
        public void setResourceName(String resourceName) { this.resourceName = resourceName; }
        public String getResourceType() { return resourceType; }
        public void setResourceType(String resourceType) { this.resourceType = resourceType; }
        public String getResourceUrl() { return resourceUrl; }
        public void setResourceUrl(String resourceUrl) { this.resourceUrl = resourceUrl; }
        public Long getFileSize() { return fileSize; }
        public void setFileSize(Long fileSize) { this.fileSize = fileSize; }
        public LocalDateTime getUploadTime() { return uploadTime; }
        public void setUploadTime(LocalDateTime uploadTime) { this.uploadTime = uploadTime; }
    }

    /**
     * 资源下载响应DTO
     */
    public static class ResourceDownloadResponse {
        private String downloadUrl;
        private String fileName;
        private Long fileSize;
        private String contentType;
        private LocalDateTime expiryTime;
        
        // Getters and Setters
        public String getDownloadUrl() { return downloadUrl; }
        public void setDownloadUrl(String downloadUrl) { this.downloadUrl = downloadUrl; }
        public String getFileName() { return fileName; }
        public void setFileName(String fileName) { this.fileName = fileName; }
        public Long getFileSize() { return fileSize; }
        public void setFileSize(Long fileSize) { this.fileSize = fileSize; }
        public String getContentType() { return contentType; }
        public void setContentType(String contentType) { this.contentType = contentType; }
        public LocalDateTime getExpiryTime() { return expiryTime; }
        public void setExpiryTime(LocalDateTime expiryTime) { this.expiryTime = expiryTime; }
    }

    /**
     * 任务模板响应DTO
     */
    public static class TaskTemplateResponse {
        private Long templateId;
        private String templateName;
        private String description;
        private String taskType;
        private Map<String, Object> templateData;
        
        // Getters and Setters
        public Long getTemplateId() { return templateId; }
        public void setTemplateId(Long templateId) { this.templateId = templateId; }
        public String getTemplateName() { return templateName; }
        public void setTemplateName(String templateName) { this.templateName = templateName; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
        public String getTaskType() { return taskType; }
        public void setTaskType(String taskType) { this.taskType = taskType; }
        public Map<String, Object> getTemplateData() { return templateData; }
        public void setTemplateData(Map<String, Object> templateData) { this.templateData = templateData; }
    }

    /**
     * 评分标准响应DTO
     */
    public static class GradingCriteriaResponse {
        private Long criteriaId;
        private String criteriaName;
        private String description;
        private BigDecimal maxScore;
        private List<GradingCriteriaItem> items;
        
        // Getters and Setters
        public Long getCriteriaId() { return criteriaId; }
        public void setCriteriaId(Long criteriaId) { this.criteriaId = criteriaId; }
        public String getCriteriaName() { return criteriaName; }
        public void setCriteriaName(String criteriaName) { this.criteriaName = criteriaName; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
        public BigDecimal getMaxScore() { return maxScore; }
        public void setMaxScore(BigDecimal maxScore) { this.maxScore = maxScore; }
        public List<GradingCriteriaItem> getItems() { return items; }
        public void setItems(List<GradingCriteriaItem> items) { this.items = items; }
        
        public static class GradingCriteriaItem {
            private String itemName;
            private String description;
            private BigDecimal weight;
            
            // Getters and Setters
            public String getItemName() { return itemName; }
            public void setItemName(String itemName) { this.itemName = itemName; }
            public String getDescription() { return description; }
            public void setDescription(String description) { this.description = description; }
            public BigDecimal getWeight() { return weight; }
            public void setWeight(BigDecimal weight) { this.weight = weight; }
        }
    }
}