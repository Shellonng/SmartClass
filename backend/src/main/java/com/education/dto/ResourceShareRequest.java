package com.education.dto;

import jakarta.validation.constraints.NotNull;
import java.time.LocalDateTime;

/**
 * 资源分享请求DTO
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
public class ResourceShareRequest {
    
    @NotNull(message = "资源ID不能为空")
    private Long resourceId;
    
    private Boolean isPublic;
    private LocalDateTime expireTime;
    private String description;
    
    // Getters and Setters
    public Long getResourceId() { return resourceId; }
    public void setResourceId(Long resourceId) { this.resourceId = resourceId; }
    
    public Boolean getIsPublic() { return isPublic; }
    public void setIsPublic(Boolean isPublic) { this.isPublic = isPublic; }
    
    public LocalDateTime getExpireTime() { return expireTime; }
    public void setExpireTime(LocalDateTime expireTime) { this.expireTime = expireTime; }
    
    public String getDescription() { return description; }
    public void setDescription(String description) { this.description = description; }
}