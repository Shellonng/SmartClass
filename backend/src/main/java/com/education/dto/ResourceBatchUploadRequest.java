package com.education.dto;

import jakarta.validation.constraints.NotEmpty;
import org.springframework.web.multipart.MultipartFile;
import java.util.List;

/**
 * 资源批量上传请求DTO
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
public class ResourceBatchUploadRequest {
    
    @NotEmpty(message = "上传文件列表不能为空")
    private List<ResourceUploadRequest> uploadRequests;
    
    private Long courseId;
    private String category;
    private String visibility;
    private String batchDescription;
    
    // Getters and Setters
    public List<ResourceUploadRequest> getUploadRequests() { return uploadRequests; }
    public void setUploadRequests(List<ResourceUploadRequest> uploadRequests) { this.uploadRequests = uploadRequests; }
    
    public Long getCourseId() { return courseId; }
    public void setCourseId(Long courseId) { this.courseId = courseId; }
    
    public String getCategory() { return category; }
    public void setCategory(String category) { this.category = category; }
    
    public String getVisibility() { return visibility; }
    public void setVisibility(String visibility) { this.visibility = visibility; }
    
    public String getBatchDescription() { return batchDescription; }
    public void setBatchDescription(String batchDescription) { this.batchDescription = batchDescription; }
}