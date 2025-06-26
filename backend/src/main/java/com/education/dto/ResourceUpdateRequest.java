package com.education.dto;

/**
 * 资源更新请求DTO
 */
public class ResourceUpdateRequest {
    private String resourceName;
    private String description;
    private String tags;
    private String visibility;
    
    // Getters and Setters
    public String getResourceName() { return resourceName; }
    public void setResourceName(String resourceName) { this.resourceName = resourceName; }
    public String getDescription() { return description; }
    public void setDescription(String description) { this.description = description; }
    public String getTags() { return tags; }
    public void setTags(String tags) { this.tags = tags; }
    public String getVisibility() { return visibility; }
    public void setVisibility(String visibility) { this.visibility = visibility; }
}