package com.education.dto;

import jakarta.validation.constraints.NotBlank;

/**
 * 文件夹创建请求DTO
 */
public class FolderCreateRequest {
    @NotBlank(message = "文件夹名称不能为空")
    private String folderName;
    
    private String description;
    private Long parentFolderId;
    private String visibility;
    
    // Getters and Setters
    public String getFolderName() { return folderName; }
    public void setFolderName(String folderName) { this.folderName = folderName; }
    public String getDescription() { return description; }
    public void setDescription(String description) { this.description = description; }
    public Long getParentFolderId() { return parentFolderId; }
    public void setParentFolderId(Long parentFolderId) { this.parentFolderId = parentFolderId; }
    public String getVisibility() { return visibility; }
    public void setVisibility(String visibility) { this.visibility = visibility; }
}