package com.education.dto;

import java.time.LocalDateTime;
import java.util.List;

/**
 * 资源详情响应DTO
 */
public class ResourceDetailResponse {
    private Long resourceId;
    private Long courseId;
    private String title;
    private String description;
    private String fileType;
    private Long fileSize;
    private String filePath;
    private Integer downloadCount;
    private LocalDateTime createdTime;
    private ResourceResponse resource;
    private List<String> relatedResources;
    private Boolean canDownload;
    private Boolean canPreview;
    private String previewUrl;
    
    // Getters and Setters
    public Long getResourceId() { return resourceId; }
    public void setResourceId(Long resourceId) { this.resourceId = resourceId; }
    public Long getCourseId() { return courseId; }
    public void setCourseId(Long courseId) { this.courseId = courseId; }
    public String getTitle() { return title; }
    public void setTitle(String title) { this.title = title; }
    public String getDescription() { return description; }
    public void setDescription(String description) { this.description = description; }
    public String getFileType() { return fileType; }
    public void setFileType(String fileType) { this.fileType = fileType; }
    public Long getFileSize() { return fileSize; }
    public void setFileSize(Long fileSize) { this.fileSize = fileSize; }
    public String getFilePath() { return filePath; }
    public void setFilePath(String filePath) { this.filePath = filePath; }
    public Integer getDownloadCount() { return downloadCount; }
    public void setDownloadCount(Integer downloadCount) { this.downloadCount = downloadCount; }
    public LocalDateTime getCreatedTime() { return createdTime; }
    public void setCreatedTime(LocalDateTime createdTime) { this.createdTime = createdTime; }
    public ResourceResponse getResource() { return resource; }
    public void setResource(ResourceResponse resource) { this.resource = resource; }
    public List<String> getRelatedResources() { return relatedResources; }
    public void setRelatedResources(List<String> relatedResources) { this.relatedResources = relatedResources; }
    public Boolean getCanDownload() { return canDownload; }
    public void setCanDownload(Boolean canDownload) { this.canDownload = canDownload; }
    public Boolean getCanPreview() { return canPreview; }
    public void setCanPreview(Boolean canPreview) { this.canPreview = canPreview; }
    public String getPreviewUrl() { return previewUrl; }
    public void setPreviewUrl(String previewUrl) { this.previewUrl = previewUrl; }
}