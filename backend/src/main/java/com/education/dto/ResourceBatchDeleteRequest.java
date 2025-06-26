package com.education.dto;

import jakarta.validation.constraints.NotEmpty;
import java.util.List;

/**
 * 资源批量删除请求DTO
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
public class ResourceBatchDeleteRequest {
    
    @NotEmpty(message = "资源ID列表不能为空")
    private List<Long> resourceIds;
    
    private Boolean forceDelete; // 是否强制删除
    private String reason; // 删除原因
    
    // Getters and Setters
    public List<Long> getResourceIds() { return resourceIds; }
    public void setResourceIds(List<Long> resourceIds) { this.resourceIds = resourceIds; }
    
    public Boolean getForceDelete() { return forceDelete; }
    public void setForceDelete(Boolean forceDelete) { this.forceDelete = forceDelete; }
    
    public String getReason() { return reason; }
    public void setReason(String reason) { this.reason = reason; }
}