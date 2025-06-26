package com.education.dto;

import java.util.Map;

/**
 * 资源统计响应DTO
 */
public class ResourceStatisticsResponse {
    private Long totalResources;
    private Long totalDownloads;
    private Long totalViews;
    private Long totalShares;
    private Double averageRating;
    private Map<String, Long> resourceTypeCount;
    private Map<String, Long> monthlyUsage;
    
    // Getters and Setters
    public Long getTotalResources() { return totalResources; }
    public void setTotalResources(Long totalResources) { this.totalResources = totalResources; }
    public Long getTotalDownloads() { return totalDownloads; }
    public void setTotalDownloads(Long totalDownloads) { this.totalDownloads = totalDownloads; }
    public Long getTotalViews() { return totalViews; }
    public void setTotalViews(Long totalViews) { this.totalViews = totalViews; }
    public Long getTotalShares() { return totalShares; }
    public void setTotalShares(Long totalShares) { this.totalShares = totalShares; }
    public Double getAverageRating() { return averageRating; }
    public void setAverageRating(Double averageRating) { this.averageRating = averageRating; }
    public Map<String, Long> getResourceTypeCount() { return resourceTypeCount; }
    public void setResourceTypeCount(Map<String, Long> resourceTypeCount) { this.resourceTypeCount = resourceTypeCount; }
    public Map<String, Long> getMonthlyUsage() { return monthlyUsage; }
    public void setMonthlyUsage(Map<String, Long> monthlyUsage) { this.monthlyUsage = monthlyUsage; }
}