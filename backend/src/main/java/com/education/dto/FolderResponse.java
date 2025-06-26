package com.education.dto;

import java.time.LocalDateTime;

/**
 * 文件夹响应DTO
 */
public class FolderResponse {
    private Long folderId;
    private String folderName;
    private String description;
    private Long parentFolderId;
    private String folderPath;
    private Integer resourceCount;
    private Integer subFolderCount;
    private LocalDateTime createTime;
    private LocalDateTime updateTime;
    private String creatorName;
    
    // Getters and Setters
    public Long getFolderId() { return folderId; }
    public void setFolderId(Long folderId) { this.folderId = folderId; }
    public String getFolderName() { return folderName; }
    public void setFolderName(String folderName) { this.folderName = folderName; }
    public String getDescription() { return description; }
    public void setDescription(String description) { this.description = description; }
    public Long getParentFolderId() { return parentFolderId; }
    public void setParentFolderId(Long parentFolderId) { this.parentFolderId = parentFolderId; }
    public String getFolderPath() { return folderPath; }
    public void setFolderPath(String folderPath) { this.folderPath = folderPath; }
    public Integer getResourceCount() { return resourceCount; }
    public void setResourceCount(Integer resourceCount) { this.resourceCount = resourceCount; }
    public Integer getSubFolderCount() { return subFolderCount; }
    public void setSubFolderCount(Integer subFolderCount) { this.subFolderCount = subFolderCount; }
    public LocalDateTime getCreateTime() { return createTime; }
    public void setCreateTime(LocalDateTime createTime) { this.createTime = createTime; }
    public LocalDateTime getUpdateTime() { return updateTime; }
    public void setUpdateTime(LocalDateTime updateTime) { this.updateTime = updateTime; }
    public String getCreatorName() { return creatorName; }
    public void setCreatorName(String creatorName) { this.creatorName = creatorName; }
}