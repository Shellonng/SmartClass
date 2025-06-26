package com.education.dto;

import java.util.List;

/**
 * 批量上传响应DTO
 */
public class BatchUploadResponse {
    private List<ResourceResponse> successfulUploads;
    private List<String> failedUploads;
    private Integer totalCount;
    private Integer successCount;
    private Integer failureCount;
    private String batchId;
    
    // Getters and Setters
    public List<ResourceResponse> getSuccessfulUploads() { return successfulUploads; }
    public void setSuccessfulUploads(List<ResourceResponse> successfulUploads) { this.successfulUploads = successfulUploads; }
    public List<String> getFailedUploads() { return failedUploads; }
    public void setFailedUploads(List<String> failedUploads) { this.failedUploads = failedUploads; }
    public Integer getTotalCount() { return totalCount; }
    public void setTotalCount(Integer totalCount) { this.totalCount = totalCount; }
    public Integer getSuccessCount() { return successCount; }
    public void setSuccessCount(Integer successCount) { this.successCount = successCount; }
    public Integer getFailureCount() { return failureCount; }
    public void setFailureCount(Integer failureCount) { this.failureCount = failureCount; }
    public String getBatchId() { return batchId; }
    public void setBatchId(String batchId) { this.batchId = batchId; }
}