package com.education.dto;

import jakarta.validation.constraints.NotNull;

/**
 * 资源移动请求DTO
 */
public class ResourceMoveRequest {
    @NotNull(message = "目标文件夹ID不能为空")
    private Long targetFolderId;
    
    // Getters and Setters
    public Long getTargetFolderId() { return targetFolderId; }
    public void setTargetFolderId(Long targetFolderId) { this.targetFolderId = targetFolderId; }
}