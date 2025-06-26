package com.education.dto;

/**
 * 资源查询参数DTO
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
public class ResourceQueryParams {
    
    private int page;
    private int size;
    private String keyword;
    private Long courseId;
    private String category;
    private String fileType;
    private String sortBy; // NAME, CREATE_TIME, DOWNLOAD_COUNT, FILE_SIZE
    private String sortOrder; // ASC, DESC
    private String visibility; // PUBLIC, PRIVATE, COURSE_ONLY
    private String resourceType; // VIDEO, DOCUMENT, IMAGE, AUDIO, OTHER
    
    // Constructors
    public ResourceQueryParams() {}
    
    public ResourceQueryParams(int page, int size, String keyword, Long courseId, String category, String fileType) {
        this.page = page;
        this.size = size;
        this.keyword = keyword;
        this.courseId = courseId;
        this.category = category;
        this.fileType = fileType;
    }
    
    // Getters and Setters
    public int getPage() { return page; }
    public void setPage(int page) { this.page = page; }
    
    public int getSize() { return size; }
    public void setSize(int size) { this.size = size; }
    
    public String getKeyword() { return keyword; }
    public void setKeyword(String keyword) { this.keyword = keyword; }
    
    public Long getCourseId() { return courseId; }
    public void setCourseId(Long courseId) { this.courseId = courseId; }
    
    public String getCategory() { return category; }
    public void setCategory(String category) { this.category = category; }
    
    public String getFileType() { return fileType; }
    public void setFileType(String fileType) { this.fileType = fileType; }
    
    public String getSortBy() { return sortBy; }
    public void setSortBy(String sortBy) { this.sortBy = sortBy; }
    
    public String getSortOrder() { return sortOrder; }
    public void setSortOrder(String sortOrder) { this.sortOrder = sortOrder; }
    
    public String getVisibility() { return visibility; }
    public void setVisibility(String visibility) { this.visibility = visibility; }
    
    public String getResourceType() { return resourceType; }
    public void setResourceType(String resourceType) { this.resourceType = resourceType; }
}