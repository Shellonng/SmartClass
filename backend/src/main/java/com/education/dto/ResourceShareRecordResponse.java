package com.education.dto;

import java.time.LocalDateTime;

/**
 * 资源分享记录响应DTO
 */
public class ResourceShareRecordResponse {
    private String shareId;
    private Long resourceId;
    private String resourceName;
    private String shareUrl;
    private LocalDateTime shareTime;
    private LocalDateTime expireTime;
    private Integer accessCount;
    private String status;
    
    // Getters and Setters
    public String getShareId() { return shareId; }
    public void setShareId(String shareId) { this.shareId = shareId; }
    public Long getResourceId() { return resourceId; }
    public void setResourceId(Long resourceId) { this.resourceId = resourceId; }
    public String getResourceName() { return resourceName; }
    public void setResourceName(String resourceName) { this.resourceName = resourceName; }
    public String getShareUrl() { return shareUrl; }
    public void setShareUrl(String shareUrl) { this.shareUrl = shareUrl; }
    public LocalDateTime getShareTime() { return shareTime; }
    public void setShareTime(LocalDateTime shareTime) { this.shareTime = shareTime; }
    public LocalDateTime getExpireTime() { return expireTime; }
    public void setExpireTime(LocalDateTime expireTime) { this.expireTime = expireTime; }
    public Integer getAccessCount() { return accessCount; }
    public void setAccessCount(Integer accessCount) { this.accessCount = accessCount; }
    public String getStatus() { return status; }
    public void setStatus(String status) { this.status = status; }
}