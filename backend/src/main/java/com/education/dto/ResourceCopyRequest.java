package com.education.dto;

import jakarta.validation.constraints.NotNull;

/**
 * 资源复制请求DTO
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
public class ResourceCopyRequest {
    
    @NotNull(message = "目标课程ID不能为空")
    private Long targetCourseId;
    
    private Long targetFolderId;
    private String newResourceName;
    private String description;
    private Boolean copyMetadata; // 是否复制元数据
    private Boolean copyPermissions; // 是否复制权限设置
    
    // Getters and Setters
    public Long getTargetCourseId() { return targetCourseId; }
    public void setTargetCourseId(Long targetCourseId) { this.targetCourseId = targetCourseId; }
    
    public Long getTargetFolderId() { return targetFolderId; }
    public void setTargetFolderId(Long targetFolderId) { this.targetFolderId = targetFolderId; }
    
    public String getNewResourceName() { return newResourceName; }
    public void setNewResourceName(String newResourceName) { this.newResourceName = newResourceName; }
    
    public String getDescription() { return description; }
    public void setDescription(String description) { this.description = description; }
    
    public Boolean getCopyMetadata() { return copyMetadata; }
    public void setCopyMetadata(Boolean copyMetadata) { this.copyMetadata = copyMetadata; }
    
    public Boolean getCopyPermissions() { return copyPermissions; }
    public void setCopyPermissions(Boolean copyPermissions) { this.copyPermissions = copyPermissions; }
}