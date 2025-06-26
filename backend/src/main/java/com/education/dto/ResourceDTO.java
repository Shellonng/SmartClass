package com.education.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotEmpty;
import jakarta.validation.constraints.NotNull;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

/**
 * 资源相关DTO
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
public class ResourceDTO {

    /**
     * 资源创建请求DTO
     */
    public static class ResourceCreateRequest {
        @NotBlank(message = "资源名称不能为空")
        private String resourceName;
        
        @NotBlank(message = "资源类型不能为空")
        private String resourceType; // VIDEO, DOCUMENT, IMAGE, AUDIO, OTHER
        
        private String description;
        private String fileUrl;
        private String fileName;
        private Long fileSize;
        private String tags;
        private Long courseId;
        private String visibility; // PUBLIC, PRIVATE, COURSE_ONLY
        
        // Getters and Setters
        public String getResourceName() { return resourceName; }
        public void setResourceName(String resourceName) { this.resourceName = resourceName; }
        public String getResourceType() { return resourceType; }
        public void setResourceType(String resourceType) { this.resourceType = resourceType; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
        public String getFileUrl() { return fileUrl; }
        public void setFileUrl(String fileUrl) { this.fileUrl = fileUrl; }
        public String getFileName() { return fileName; }
        public void setFileName(String fileName) { this.fileName = fileName; }
        public Long getFileSize() { return fileSize; }
        public void setFileSize(Long fileSize) { this.fileSize = fileSize; }
        public String getTags() { return tags; }
        public void setTags(String tags) { this.tags = tags; }
        public Long getCourseId() { return courseId; }
        public void setCourseId(Long courseId) { this.courseId = courseId; }
        public String getVisibility() { return visibility; }
        public void setVisibility(String visibility) { this.visibility = visibility; }
    }

    /**
     * 资源响应DTO
     */
    public static class ResourceResponse {
        private Long resourceId;
        private String resourceName;
        private String resourceType;
        private String description;
        private String fileUrl;
        private String fileName;
        private Long fileSize;
        private String tags;
        private String visibility;
        private Long downloadCount;
        private LocalDateTime createTime;
        private LocalDateTime updateTime;
        private String uploaderName;
        private String courseName;
        
        // Getters and Setters
        public Long getResourceId() { return resourceId; }
        public void setResourceId(Long resourceId) { this.resourceId = resourceId; }
        public String getResourceName() { return resourceName; }
        public void setResourceName(String resourceName) { this.resourceName = resourceName; }
        public String getResourceType() { return resourceType; }
        public void setResourceType(String resourceType) { this.resourceType = resourceType; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
        public String getFileUrl() { return fileUrl; }
        public void setFileUrl(String fileUrl) { this.fileUrl = fileUrl; }
        public String getFileName() { return fileName; }
        public void setFileName(String fileName) { this.fileName = fileName; }
        public Long getFileSize() { return fileSize; }
        public void setFileSize(Long fileSize) { this.fileSize = fileSize; }
        public String getTags() { return tags; }
        public void setTags(String tags) { this.tags = tags; }
        public String getVisibility() { return visibility; }
        public void setVisibility(String visibility) { this.visibility = visibility; }
        public Long getDownloadCount() { return downloadCount; }
        public void setDownloadCount(Integer downloadCount) { this.downloadCount = Long.valueOf(downloadCount); }
        public LocalDateTime getCreateTime() { return createTime; }
        public void setCreateTime(LocalDateTime createTime) { this.createTime = createTime; }
        public LocalDateTime getUpdateTime() { return updateTime; }
        public void setUpdateTime(LocalDateTime updateTime) { this.updateTime = updateTime; }
        public String getUploaderName() { return uploaderName; }
        public void setUploaderName(String uploaderName) { this.uploaderName = uploaderName; }
        public String getCourseName() { return courseName; }
        public void setCourseName(String courseName) { this.courseName = courseName; }
    }

    /**
     * 资源更新请求DTO
     */
    public static class ResourceUpdateRequest {
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

    /**
     * 资源列表响应DTO
     */
    public static class ResourceListResponse {
        private List<ResourceResponse> resources;
        private Integer totalCount;
        private Integer currentPage;
        private Integer pageSize;
        private Integer totalPages;
        
        // Getters and Setters
        public List<ResourceResponse> getResources() { return resources; }
        public void setResources(List<ResourceResponse> resources) { this.resources = resources; }
        public Integer getTotalCount() { return totalCount; }
        public void setTotalCount(Integer totalCount) { this.totalCount = totalCount; }
        public Integer getCurrentPage() { return currentPage; }
        public void setCurrentPage(Integer currentPage) { this.currentPage = currentPage; }
        public Integer getPageSize() { return pageSize; }
        public void setPageSize(Integer pageSize) { this.pageSize = pageSize; }
        public Integer getTotalPages() { return totalPages; }
        public void setTotalPages(Integer totalPages) { this.totalPages = totalPages; }
    }

    /**
     * 资源详情响应DTO
     */
    public static class ResourceDetailResponse {
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

    /**
     * 资源下载响应DTO
     */
    public static class ResourceDownloadResponse {
        private Long resourceId;
        private String downloadUrl;
        private String fileName;
        private String fileType;
        private Long fileSize;
        private String contentType;
        private LocalDateTime expireTime;
        private LocalDateTime downloadTime;
        
        // Getters and Setters
        public Long getResourceId() { return resourceId; }
        public void setResourceId(Long resourceId) { this.resourceId = resourceId; }
        public String getDownloadUrl() { return downloadUrl; }
        public void setDownloadUrl(String downloadUrl) { this.downloadUrl = downloadUrl; }
        public String getFileName() { return fileName; }
        public void setFileName(String fileName) { this.fileName = fileName; }
        public String getFileType() { return fileType; }
        public void setFileType(String fileType) { this.fileType = fileType; }
        public Long getFileSize() { return fileSize; }
        public void setFileSize(Long fileSize) { this.fileSize = fileSize; }
        public String getContentType() { return contentType; }
        public void setContentType(String contentType) { this.contentType = contentType; }
        public LocalDateTime getExpireTime() { return expireTime; }
        public void setExpireTime(LocalDateTime expireTime) { this.expireTime = expireTime; }
        public LocalDateTime getDownloadTime() { return downloadTime; }
        public void setDownloadTime(LocalDateTime downloadTime) { this.downloadTime = downloadTime; }
    }

    /**
     * 资源预览响应DTO
     */
    public static class ResourcePreviewResponse {
        private String previewUrl;
        private String previewType; // IMAGE, VIDEO, DOCUMENT, AUDIO
        private String thumbnailUrl;
        private Integer duration; // for video/audio
        private Integer pageCount; // for documents
        
        // Getters and Setters
        public String getPreviewUrl() { return previewUrl; }
        public void setPreviewUrl(String previewUrl) { this.previewUrl = previewUrl; }
        public String getPreviewType() { return previewType; }
        public void setPreviewType(String previewType) { this.previewType = previewType; }
        public String getThumbnailUrl() { return thumbnailUrl; }
        public void setThumbnailUrl(String thumbnailUrl) { this.thumbnailUrl = thumbnailUrl; }
        public Integer getDuration() { return duration; }
        public void setDuration(Integer duration) { this.duration = duration; }
        public Integer getPageCount() { return pageCount; }
        public void setPageCount(Integer pageCount) { this.pageCount = pageCount; }
    }

    /**
     * 资源搜索请求DTO
     */
    public static class ResourceSearchRequest {
        private String keyword;
        private String resourceType;
        private Long courseId;
        private String tags;
        private String visibility;
        private String sortBy; // NAME, CREATE_TIME, DOWNLOAD_COUNT
        private String sortOrder; // ASC, DESC
        private Integer page;
        private Integer size;
        
        // Getters and Setters
        public String getKeyword() { return keyword; }
        public void setKeyword(String keyword) { this.keyword = keyword; }
        public String getResourceType() { return resourceType; }
        public void setResourceType(String resourceType) { this.resourceType = resourceType; }
        public Long getCourseId() { return courseId; }
        public void setCourseId(Long courseId) { this.courseId = courseId; }
        public String getTags() { return tags; }
        public void setTags(String tags) { this.tags = tags; }
        public String getVisibility() { return visibility; }
        public void setVisibility(String visibility) { this.visibility = visibility; }
        public String getSortBy() { return sortBy; }
        public void setSortBy(String sortBy) { this.sortBy = sortBy; }
        public String getSortOrder() { return sortOrder; }
        public void setSortOrder(String sortOrder) { this.sortOrder = sortOrder; }
        public Integer getPage() { return page; }
        public void setPage(Integer page) { this.page = page; }
        public Integer getSize() { return size; }
        public void setSize(Integer size) { this.size = size; }
    }

    /**
     * 资源访问记录响应DTO
     */
    public static class ResourceAccessRecordResponse {
        private Long recordId;
        private Long resourceId;
        private String resourceName;
        private String accessType; // VIEW, DOWNLOAD, PREVIEW
        private LocalDateTime accessTime;
        private String userAgent;
        private String ipAddress;
        
        // Getters and Setters
        public Long getRecordId() { return recordId; }
        public void setRecordId(Long recordId) { this.recordId = recordId; }
        public Long getResourceId() { return resourceId; }
        public void setResourceId(Long resourceId) { this.resourceId = resourceId; }
        public String getResourceName() { return resourceName; }
        public void setResourceName(String resourceName) { this.resourceName = resourceName; }
        public String getAccessType() { return accessType; }
        public void setAccessType(String accessType) { this.accessType = accessType; }
        public LocalDateTime getAccessTime() { return accessTime; }
        public void setAccessTime(LocalDateTime accessTime) { this.accessTime = accessTime; }
        public String getUserAgent() { return userAgent; }
        public void setUserAgent(String userAgent) { this.userAgent = userAgent; }
        public String getIpAddress() { return ipAddress; }
         public void setIpAddress(String ipAddress) { this.ipAddress = ipAddress; }
     }

    /**
     * 资源访问请求DTO
     */
    public static class ResourceAccessRequest {
        @NotNull(message = "资源ID不能为空")
        private Long resourceId;
        
        private String accessType; // VIEW, DOWNLOAD, PREVIEW
        private String userAgent;
        private String ipAddress;
        
        // Getters and Setters
        public Long getResourceId() { return resourceId; }
        public void setResourceId(Long resourceId) { this.resourceId = resourceId; }
        public String getAccessType() { return accessType; }
        public void setAccessType(String accessType) { this.accessType = accessType; }
        public String getUserAgent() { return userAgent; }
        public void setUserAgent(String userAgent) { this.userAgent = userAgent; }
        public String getIpAddress() { return ipAddress; }
        public void setIpAddress(String ipAddress) { this.ipAddress = ipAddress; }
    }

    /**
     * 资源评价请求DTO
     */
    public static class ResourceEvaluationRequest {
        @NotNull(message = "资源ID不能为空")
        private Long resourceId;
        
        @NotNull(message = "评分不能为空")
        private Integer rating; // 1-5
        
        private String comment;
        private List<String> tags;
        
        // Getters and Setters
        public Long getResourceId() { return resourceId; }
        public void setResourceId(Long resourceId) { this.resourceId = resourceId; }
        public Integer getRating() { return rating; }
        public void setRating(Integer rating) { this.rating = rating; }
        public String getComment() { return comment; }
        public void setComment(String comment) { this.comment = comment; }
        public List<String> getTags() { return tags; }
        public void setTags(List<String> tags) { this.tags = tags; }
    }

    /**
     * 资源评价响应DTO
     */
    public static class ResourceEvaluationResponse {
        private Long evaluationId;
        private Long resourceId;
        private String resourceName;
        private Integer rating;
        private String comment;
        private List<String> tags;
        private LocalDateTime createTime;
        private String evaluatorName;
        
        // Getters and Setters
        public Long getEvaluationId() { return evaluationId; }
        public void setEvaluationId(Long evaluationId) { this.evaluationId = evaluationId; }
        public Long getResourceId() { return resourceId; }
        public void setResourceId(Long resourceId) { this.resourceId = resourceId; }
        public String getResourceName() { return resourceName; }
        public void setResourceName(String resourceName) { this.resourceName = resourceName; }
        public Integer getRating() { return rating; }
        public void setRating(Integer rating) { this.rating = rating; }
        public String getComment() { return comment; }
        public void setComment(String comment) { this.comment = comment; }
        public List<String> getTags() { return tags; }
        public void setTags(List<String> tags) { this.tags = tags; }
        public LocalDateTime getCreateTime() { return createTime; }
        public void setCreateTime(LocalDateTime createTime) { this.createTime = createTime; }
        public String getEvaluatorName() { return evaluatorName; }
        public void setEvaluatorName(String evaluatorName) { this.evaluatorName = evaluatorName; }
    }

    /**
     * 资源分享响应DTO
     */
    public static class ResourceShareResponse {
        private String shareId;
        private String shareUrl;
        private String shareCode;
        private LocalDateTime expireTime;
        private Boolean isPublic;
        private Integer accessCount;
        
        // Getters and Setters
        public String getShareId() { return shareId; }
        public void setShareId(String shareId) { this.shareId = shareId; }
        public String getShareUrl() { return shareUrl; }
        public void setShareUrl(String shareUrl) { this.shareUrl = shareUrl; }
        public String getShareCode() { return shareCode; }
        public void setShareCode(String shareCode) { this.shareCode = shareCode; }
        public LocalDateTime getExpireTime() { return expireTime; }
        public void setExpireTime(LocalDateTime expireTime) { this.expireTime = expireTime; }
        public Boolean getIsPublic() { return isPublic; }
        public void setIsPublic(Boolean isPublic) { this.isPublic = isPublic; }
        public Integer getAccessCount() { return accessCount; }
        public void setAccessCount(Integer accessCount) { this.accessCount = accessCount; }
    }

    /**
     * 资源分享请求DTO
     */
    public static class ResourceShareRequest {
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

    /**
     * 资源分享记录响应DTO
     */
    public static class ResourceShareRecordResponse {
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

    /**
     * 共享资源响应DTO
     */
    public static class SharedResourceResponse {
        private String shareId;
        private ResourceResponse resource;
        private String sharerName;
        private LocalDateTime shareTime;
        private String description;
        private Boolean canDownload;
        
        // Getters and Setters
        public String getShareId() { return shareId; }
        public void setShareId(String shareId) { this.shareId = shareId; }
        public ResourceResponse getResource() { return resource; }
        public void setResource(ResourceResponse resource) { this.resource = resource; }
        public String getSharerName() { return sharerName; }
        public void setSharerName(String sharerName) { this.sharerName = sharerName; }
        public LocalDateTime getShareTime() { return shareTime; }
        public void setShareTime(LocalDateTime shareTime) { this.shareTime = shareTime; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
        public Boolean getCanDownload() { return canDownload; }
        public void setCanDownload(Boolean canDownload) { this.canDownload = canDownload; }
    }

    /**
     * 资源笔记响应DTO
     */
    public static class ResourceNoteResponse {
        private Long noteId;
        private Long resourceId;
        private String title;
        private String content;
        private String noteType; // TEXT, HIGHLIGHT, BOOKMARK
        private Integer position; // for video/audio timestamp or document page
        private LocalDateTime createTime;
        private LocalDateTime updateTime;
        
        // Getters and Setters
        public Long getNoteId() { return noteId; }
        public void setNoteId(Long noteId) { this.noteId = noteId; }
        public Long getResourceId() { return resourceId; }
        public void setResourceId(Long resourceId) { this.resourceId = resourceId; }
        public String getTitle() { return title; }
        public void setTitle(String title) { this.title = title; }
        public String getContent() { return content; }
        public void setContent(String content) { this.content = content; }
        public String getNoteType() { return noteType; }
        public void setNoteType(String noteType) { this.noteType = noteType; }
        public Integer getPosition() { return position; }
        public void setPosition(Integer position) { this.position = position; }
        public LocalDateTime getCreateTime() { return createTime; }
        public void setCreateTime(LocalDateTime createTime) { this.createTime = createTime; }
        public LocalDateTime getUpdateTime() { return updateTime; }
        public void setUpdateTime(LocalDateTime updateTime) { this.updateTime = updateTime; }
    }

    /**
     * 资源笔记创建请求DTO
     */
    public static class ResourceNoteCreateRequest {
        @NotNull(message = "资源ID不能为空")
        private Long resourceId;
        
        @NotBlank(message = "笔记标题不能为空")
        private String title;
        
        private String content;
        private String noteType;
        private Integer position;
        
        // Getters and Setters
        public Long getResourceId() { return resourceId; }
        public void setResourceId(Long resourceId) { this.resourceId = resourceId; }
        public String getTitle() { return title; }
        public void setTitle(String title) { this.title = title; }
        public String getContent() { return content; }
        public void setContent(String content) { this.content = content; }
        public String getNoteType() { return noteType; }
        public void setNoteType(String noteType) { this.noteType = noteType; }
        public Integer getPosition() { return position; }
        public void setPosition(Integer position) { this.position = position; }
    }

    /**
     * 资源笔记更新请求DTO
     */
    public static class ResourceNoteUpdateRequest {
        private String title;
        private String content;
        private String noteType;
        private Integer position;
        
        // Getters and Setters
        public String getTitle() { return title; }
        public void setTitle(String title) { this.title = title; }
        public String getContent() { return content; }
        public void setContent(String content) { this.content = content; }
        public String getNoteType() { return noteType; }
        public void setNoteType(String noteType) { this.noteType = noteType; }
        public Integer getPosition() { return position; }
        public void setPosition(Integer position) { this.position = position; }
    }

    /**
     * 资源讨论响应DTO
     */
    public static class ResourceDiscussionResponse {
        private Long discussionId;
        private Long resourceId;
        private String title;
        private String content;
        private String authorName;
        private LocalDateTime createTime;
        private Integer replyCount;
        private Integer likeCount;
        private Boolean isLiked;
        
        // Getters and Setters
        public Long getDiscussionId() { return discussionId; }
        public void setDiscussionId(Long discussionId) { this.discussionId = discussionId; }
        public Long getResourceId() { return resourceId; }
        public void setResourceId(Long resourceId) { this.resourceId = resourceId; }
        public String getTitle() { return title; }
        public void setTitle(String title) { this.title = title; }
        public String getContent() { return content; }
        public void setContent(String content) { this.content = content; }
        public String getAuthorName() { return authorName; }
        public void setAuthorName(String authorName) { this.authorName = authorName; }
        public LocalDateTime getCreateTime() { return createTime; }
        public void setCreateTime(LocalDateTime createTime) { this.createTime = createTime; }
        public Integer getReplyCount() { return replyCount; }
        public void setReplyCount(Integer replyCount) { this.replyCount = replyCount; }
        public Integer getLikeCount() { return likeCount; }
        public void setLikeCount(Integer likeCount) { this.likeCount = likeCount; }
        public Boolean getIsLiked() { return isLiked; }
        public void setIsLiked(Boolean isLiked) { this.isLiked = isLiked; }
    }

    /**
     * 资源讨论创建请求DTO
     */
    public static class ResourceDiscussionCreateRequest {
        @NotNull(message = "资源ID不能为空")
        private Long resourceId;
        
        @NotBlank(message = "讨论标题不能为空")
        private String title;
        
        @NotBlank(message = "讨论内容不能为空")
        private String content;
        
        // Getters and Setters
        public Long getResourceId() { return resourceId; }
        public void setResourceId(Long resourceId) { this.resourceId = resourceId; }
        public String getTitle() { return title; }
        public void setTitle(String title) { this.title = title; }
        public String getContent() { return content; }
        public void setContent(String content) { this.content = content; }
    }

    /**
     * 资源讨论回复请求DTO
     */
    public static class ResourceDiscussionReplyRequest {
        @NotNull(message = "讨论ID不能为空")
        private Long discussionId;
        
        @NotBlank(message = "回复内容不能为空")
        private String content;
        
        private Long parentReplyId; // for nested replies
        
        // Getters and Setters
        public Long getDiscussionId() { return discussionId; }
        public void setDiscussionId(Long discussionId) { this.discussionId = discussionId; }
        public String getContent() { return content; }
        public void setContent(String content) { this.content = content; }
        public Long getParentReplyId() { return parentReplyId; }
         public void setParentReplyId(Long parentReplyId) { this.parentReplyId = parentReplyId; }
     }

    /**
     * 资源版本响应DTO
     */
    public static class ResourceVersionResponse {
        private Long versionId;
        private Long resourceId;
        private String versionNumber;
        private String description;
        private String filePath;
        private Long fileSize;
        private LocalDateTime createTime;
        private String creatorName;
        private Boolean isActive;
        
        // Getters and Setters
        public Long getVersionId() { return versionId; }
        public void setVersionId(Long versionId) { this.versionId = versionId; }
        public Long getResourceId() { return resourceId; }
        public void setResourceId(Long resourceId) { this.resourceId = resourceId; }
        public String getVersionNumber() { return versionNumber; }
        public void setVersionNumber(String versionNumber) { this.versionNumber = versionNumber; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
        public String getFilePath() { return filePath; }
        public void setFilePath(String filePath) { this.filePath = filePath; }
        public Long getFileSize() { return fileSize; }
        public void setFileSize(Long fileSize) { this.fileSize = fileSize; }
        public LocalDateTime getCreateTime() { return createTime; }
        public void setCreateTime(LocalDateTime createTime) { this.createTime = createTime; }
        public String getCreatorName() { return creatorName; }
        public void setCreatorName(String creatorName) { this.creatorName = creatorName; }
        public Boolean getIsActive() { return isActive; }
        public void setIsActive(Boolean isActive) { this.isActive = isActive; }
    }

    /**
     * 资源统计响应DTO
     */
    public static class ResourceStatisticsResponse {
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

    /**
     * 学习资源推荐响应DTO
     */
    public static class LearningResourceRecommendationResponse {
        private List<ResourceResponse> recommendedResources;
        private String recommendationReason;
        private Double relevanceScore;
        private String category;
        
        // Getters and Setters
        public List<ResourceResponse> getRecommendedResources() { return recommendedResources; }
        public void setRecommendedResources(List<ResourceResponse> recommendedResources) { this.recommendedResources = recommendedResources; }
        public String getRecommendationReason() { return recommendationReason; }
        public void setRecommendationReason(String recommendationReason) { this.recommendationReason = recommendationReason; }
        public Double getRelevanceScore() { return relevanceScore; }
        public void setRelevanceScore(Double relevanceScore) { this.relevanceScore = relevanceScore; }
        public String getCategory() { return category; }
        public void setCategory(String category) { this.category = category; }
    }

    /**
     * 资源学习进度响应DTO
     */
    public static class ResourceLearningProgressResponse {
        private Long resourceId;
        private String resourceName;
        private Integer progressPercentage;
        private Long timeSpent; // in minutes
        private LocalDateTime lastAccessTime;
        private String status; // NOT_STARTED, IN_PROGRESS, COMPLETED
        private List<String> completedSections;
        
        // Getters and Setters
        public Long getResourceId() { return resourceId; }
        public void setResourceId(Long resourceId) { this.resourceId = resourceId; }
        public String getResourceName() { return resourceName; }
        public void setResourceName(String resourceName) { this.resourceName = resourceName; }
        public Integer getProgressPercentage() { return progressPercentage; }
        public void setProgressPercentage(Integer progressPercentage) { this.progressPercentage = progressPercentage; }
        public Long getTimeSpent() { return timeSpent; }
        public void setTimeSpent(Long timeSpent) { this.timeSpent = timeSpent; }
        public LocalDateTime getLastAccessTime() { return lastAccessTime; }
        public void setLastAccessTime(LocalDateTime lastAccessTime) { this.lastAccessTime = lastAccessTime; }
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
        public List<String> getCompletedSections() { return completedSections; }
        public void setCompletedSections(List<String> completedSections) { this.completedSections = completedSections; }
    }

    /**
     * 资源进度更新请求DTO
     */
    public static class ResourceProgressUpdateRequest {
        @NotNull(message = "资源ID不能为空")
        private Long resourceId;
        
        private Integer progressPercentage;
        private Long timeSpent;
        private String currentSection;
        private Boolean isCompleted;
        
        // Getters and Setters
        public Long getResourceId() { return resourceId; }
        public void setResourceId(Long resourceId) { this.resourceId = resourceId; }
        public Integer getProgressPercentage() { return progressPercentage; }
        public void setProgressPercentage(Integer progressPercentage) { this.progressPercentage = progressPercentage; }
        public Long getTimeSpent() { return timeSpent; }
        public void setTimeSpent(Long timeSpent) { this.timeSpent = timeSpent; }
        public String getCurrentSection() { return currentSection; }
        public void setCurrentSection(String currentSection) { this.currentSection = currentSection; }
        public Boolean getIsCompleted() { return isCompleted; }
        public void setIsCompleted(Boolean isCompleted) { this.isCompleted = isCompleted; }
    }

    /**
     * 资源标签响应DTO
     */
    public static class ResourceTagResponse {
        private Long tagId;
        private String tagName;
        private String color;
        private Integer resourceCount;
        private String description;
        
        // Getters and Setters
        public Long getTagId() { return tagId; }
        public void setTagId(Long tagId) { this.tagId = tagId; }
        public String getTagName() { return tagName; }
        public void setTagName(String tagName) { this.tagName = tagName; }
        public String getColor() { return color; }
        public void setColor(String color) { this.color = color; }
        public Integer getResourceCount() { return resourceCount; }
        public void setResourceCount(Integer resourceCount) { this.resourceCount = resourceCount; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
    }

    /**
     * 资源类型响应DTO
     */
    public static class ResourceTypeResponse {
        private Long typeId;
        private String typeName;
        private String icon;
        private Integer resourceCount;
        private String description;
        private List<String> supportedFormats;
        
        // Getters and Setters
        public Long getTypeId() { return typeId; }
        public void setTypeId(Long typeId) { this.typeId = typeId; }
        public String getTypeName() { return typeName; }
        public void setTypeName(String typeName) { this.typeName = typeName; }
        public String getIcon() { return icon; }
        public void setIcon(String icon) { this.icon = icon; }
        public Integer getResourceCount() { return resourceCount; }
        public void setResourceCount(Integer resourceCount) { this.resourceCount = resourceCount; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
        public List<String> getSupportedFormats() { return supportedFormats; }
        public void setSupportedFormats(List<String> supportedFormats) { this.supportedFormats = supportedFormats; }
    }

    /**
     * 资源使用报告响应DTO
     */
    public static class ResourceUsageReportResponse {
        private String reportId;
        private LocalDateTime reportDate;
        private Map<String, Object> usageData;
        private List<ResourceResponse> topResources;
        private Map<String, Long> userActivity;
        
        // Getters and Setters
        public String getReportId() { return reportId; }
        public void setReportId(String reportId) { this.reportId = reportId; }
        public LocalDateTime getReportDate() { return reportDate; }
        public void setReportDate(LocalDateTime reportDate) { this.reportDate = reportDate; }
        public Map<String, Object> getUsageData() { return usageData; }
        public void setUsageData(Map<String, Object> usageData) { this.usageData = usageData; }
        public List<ResourceResponse> getTopResources() { return topResources; }
        public void setTopResources(List<ResourceResponse> topResources) { this.topResources = topResources; }
        public Map<String, Long> getUserActivity() { return userActivity; }
        public void setUserActivity(Map<String, Long> userActivity) { this.userActivity = userActivity; }
    }

    /**
     * 资源数据导出请求DTO
     */
    public static class ResourceDataExportRequest {
        private List<Long> resourceIds;
        private String exportFormat; // CSV, EXCEL, PDF
        private Boolean includeMetadata;
        private LocalDateTime startDate;
        private LocalDateTime endDate;
        
        // Getters and Setters
        public List<Long> getResourceIds() { return resourceIds; }
        public void setResourceIds(List<Long> resourceIds) { this.resourceIds = resourceIds; }
        public String getExportFormat() { return exportFormat; }
        public void setExportFormat(String exportFormat) { this.exportFormat = exportFormat; }
        public Boolean getIncludeMetadata() { return includeMetadata; }
        public void setIncludeMetadata(Boolean includeMetadata) { this.includeMetadata = includeMetadata; }
        public LocalDateTime getStartDate() { return startDate; }
        public void setStartDate(LocalDateTime startDate) { this.startDate = startDate; }
        public LocalDateTime getEndDate() { return endDate; }
        public void setEndDate(LocalDateTime endDate) { this.endDate = endDate; }
    }

    /**
     * 离线资源响应DTO
     */
    public static class OfflineResourceResponse {
        private Long resourceId;
        private String resourceName;
        private String localPath;
        private Long fileSize;
        private LocalDateTime downloadTime;
        private LocalDateTime expireTime;
        private String status; // DOWNLOADING, AVAILABLE, EXPIRED
        
        // Getters and Setters
        public Long getResourceId() { return resourceId; }
        public void setResourceId(Long resourceId) { this.resourceId = resourceId; }
        public String getResourceName() { return resourceName; }
        public void setResourceName(String resourceName) { this.resourceName = resourceName; }
        public String getLocalPath() { return localPath; }
        public void setLocalPath(String localPath) { this.localPath = localPath; }
        public Long getFileSize() { return fileSize; }
        public void setFileSize(Long fileSize) { this.fileSize = fileSize; }
        public LocalDateTime getDownloadTime() { return downloadTime; }
        public void setDownloadTime(LocalDateTime downloadTime) { this.downloadTime = downloadTime; }
        public LocalDateTime getExpireTime() { return expireTime; }
        public void setExpireTime(LocalDateTime expireTime) { this.expireTime = expireTime; }
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
    }

    /**
     * 离线包响应DTO
     */
    public static class OfflinePackageResponse {
        private String packageId;
        private String packageName;
        private List<OfflineResourceResponse> resources;
        private Long totalSize;
        private LocalDateTime createTime;
        private LocalDateTime expireTime;
        private String downloadUrl;
        
        // Getters and Setters
        public String getPackageId() { return packageId; }
        public void setPackageId(String packageId) { this.packageId = packageId; }
        public String getPackageName() { return packageName; }
        public void setPackageName(String packageName) { this.packageName = packageName; }
        public List<OfflineResourceResponse> getResources() { return resources; }
        public void setResources(List<OfflineResourceResponse> resources) { this.resources = resources; }
        public Long getTotalSize() { return totalSize; }
        public void setTotalSize(Long totalSize) { this.totalSize = totalSize; }
        public LocalDateTime getCreateTime() { return createTime; }
        public void setCreateTime(LocalDateTime createTime) { this.createTime = createTime; }
        public LocalDateTime getExpireTime() { return expireTime; }
        public void setExpireTime(LocalDateTime expireTime) { this.expireTime = expireTime; }
        public String getDownloadUrl() { return downloadUrl; }
        public void setDownloadUrl(String downloadUrl) { this.downloadUrl = downloadUrl; }
    }

    /**
     * 离线包请求DTO
     */
    public static class OfflinePackageRequest {
        @NotBlank(message = "包名不能为空")
        private String packageName;
        
        @NotEmpty(message = "资源列表不能为空")
        private List<Long> resourceIds;
        
        private LocalDateTime expireTime;
        private String description;
        
        // Getters and Setters
        public String getPackageName() { return packageName; }
        public void setPackageName(String packageName) { this.packageName = packageName; }
        public List<Long> getResourceIds() { return resourceIds; }
        public void setResourceIds(List<Long> resourceIds) { this.resourceIds = resourceIds; }
        public LocalDateTime getExpireTime() { return expireTime; }
        public void setExpireTime(LocalDateTime expireTime) { this.expireTime = expireTime; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
    }

    /**
     * 资源同步响应DTO
     */
    public static class ResourceSyncResponse {
        private String syncId;
        private LocalDateTime syncTime;
        private Integer totalResources;
        private Integer syncedResources;
        private Integer failedResources;
        private String status; // IN_PROGRESS, COMPLETED, FAILED
        private List<String> errorMessages;
        
        // Getters and Setters
        public String getSyncId() { return syncId; }
        public void setSyncId(String syncId) { this.syncId = syncId; }
        public LocalDateTime getSyncTime() { return syncTime; }
        public void setSyncTime(LocalDateTime syncTime) { this.syncTime = syncTime; }
        public Integer getTotalResources() { return totalResources; }
        public void setTotalResources(Integer totalResources) { this.totalResources = totalResources; }
        public Integer getSyncedResources() { return syncedResources; }
        public void setSyncedResources(Integer syncedResources) { this.syncedResources = syncedResources; }
        public Integer getFailedResources() { return failedResources; }
        public void setFailedResources(Integer failedResources) { this.failedResources = failedResources; }
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
        public List<String> getErrorMessages() { return errorMessages; }
        public void setErrorMessages(List<String> errorMessages) { this.errorMessages = errorMessages; }
    }

    /**
     * 导出响应DTO
     */
    public static class ExportResponse {
        private String exportId; // 导出ID
        private String fileName; // 文件名
        private String fileUrl; // 文件URL
        private String status; // 导出状态：PROCESSING, COMPLETED, FAILED
        private LocalDateTime exportTime; // 导出时间
        private Long fileSize; // 文件大小
        private String format; // 导出格式：EXCEL, PDF, CSV
        private String message; // 状态消息
        
        // Getters and Setters
        public String getExportId() { return exportId; }
        public void setExportId(String exportId) { this.exportId = exportId; }
        public String getFileName() { return fileName; }
        public void setFileName(String fileName) { this.fileName = fileName; }
        public String getFileUrl() { return fileUrl; }
        public void setFileUrl(String fileUrl) { this.fileUrl = fileUrl; }
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
        public LocalDateTime getExportTime() { return exportTime; }
        public void setExportTime(LocalDateTime exportTime) { this.exportTime = exportTime; }
        public Long getFileSize() { return fileSize; }
        public void setFileSize(Long fileSize) { this.fileSize = fileSize; }
        public String getFormat() { return format; }
        public void setFormat(String format) { this.format = format; }
        public String getMessage() { return message; }
        public void setMessage(String message) { this.message = message; }
    }

    /**
     * 资源同步请求DTO
     */
    public static class ResourceSyncRequest {
        @NotNull(message = "同步类型不能为空")
        private String syncType; // 同步类型：FULL, INCREMENTAL
        
        private LocalDateTime lastSyncTime; // 上次同步时间
        private List<Long> resourceIds; // 指定资源ID列表（可选）
        private List<String> resourceTypes; // 指定资源类型列表（可选）
        private Long courseId; // 课程ID（可选）
        private String targetSystem; // 目标系统
        private Boolean forceSync; // 是否强制同步
        
        // Getters and Setters
        public String getSyncType() { return syncType; }
        public void setSyncType(String syncType) { this.syncType = syncType; }
        public LocalDateTime getLastSyncTime() { return lastSyncTime; }
        public void setLastSyncTime(LocalDateTime lastSyncTime) { this.lastSyncTime = lastSyncTime; }
        public List<Long> getResourceIds() { return resourceIds; }
        public void setResourceIds(List<Long> resourceIds) { this.resourceIds = resourceIds; }
        public List<String> getResourceTypes() { return resourceTypes; }
        public void setResourceTypes(List<String> resourceTypes) { this.resourceTypes = resourceTypes; }
        public Long getCourseId() { return courseId; }
        public void setCourseId(Long courseId) { this.courseId = courseId; }
        public String getTargetSystem() { return targetSystem; }
        public void setTargetSystem(String targetSystem) { this.targetSystem = targetSystem; }
        public Boolean getForceSync() { return forceSync; }
        public void setForceSync(Boolean forceSync) { this.forceSync = forceSync; }
    }

    /**
     * 资源上传请求DTO
     */
    public static class ResourceUploadRequest {
        @NotBlank(message = "资源名称不能为空")
        private String resourceName;
        
        @NotBlank(message = "资源类型不能为空")
        private String resourceType;
        
        private String description;
        private String fileUrl;
        private String fileName;
        private Long fileSize;
        private String tags;
        private Long courseId;
        private String visibility;
        private String mimeType;
        private String checksum;
        
        // Getters and Setters
        public String getResourceName() { return resourceName; }
        public void setResourceName(String resourceName) { this.resourceName = resourceName; }
        public String getResourceType() { return resourceType; }
        public void setResourceType(String resourceType) { this.resourceType = resourceType; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
        public String getFileUrl() { return fileUrl; }
        public void setFileUrl(String fileUrl) { this.fileUrl = fileUrl; }
        public String getFileName() { return fileName; }
        public void setFileName(String fileName) { this.fileName = fileName; }
        public Long getFileSize() { return fileSize; }
        public void setFileSize(Long fileSize) { this.fileSize = fileSize; }
        public String getTags() { return tags; }
        public void setTags(String tags) { this.tags = tags; }
        public Long getCourseId() { return courseId; }
        public void setCourseId(Long courseId) { this.courseId = courseId; }
        public String getVisibility() { return visibility; }
        public void setVisibility(String visibility) { this.visibility = visibility; }
        public String getMimeType() { return mimeType; }
        public void setMimeType(String mimeType) { this.mimeType = mimeType; }
        public String getChecksum() { return checksum; }
        public void setChecksum(String checksum) { this.checksum = checksum; }
    }

    /**
     * 批量上传响应DTO
     */
    public static class BatchUploadResponse {
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

    /**
     * 文件夹响应DTO
     */
    public static class FolderResponse {
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

    /**
     * 文件夹创建请求DTO
     */
    public static class FolderCreateRequest {
        @NotBlank(message = "文件夹名称不能为空")
        private String folderName;
        
        private String description;
        private Long parentFolderId;
        private String visibility;
        
        // Getters and Setters
        public String getFolderName() { return folderName; }
        public void setFolderName(String folderName) { this.folderName = folderName; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
        public Long getParentFolderId() { return parentFolderId; }
        public void setParentFolderId(Long parentFolderId) { this.parentFolderId = parentFolderId; }
        public String getVisibility() { return visibility; }
        public void setVisibility(String visibility) { this.visibility = visibility; }
    }

    /**
     * 资源权限请求DTO
     */
    public static class ResourcePermissionRequest {
        @NotNull(message = "资源ID不能为空")
        private Long resourceId;
        
        @NotNull(message = "用户ID不能为空")
        private Long userId;
        
        @NotBlank(message = "权限类型不能为空")
        private String permissionType; // READ, WRITE, DELETE, SHARE
        
        private Boolean granted;
        private LocalDateTime expireTime;
        
        // Getters and Setters
        public Long getResourceId() { return resourceId; }
        public void setResourceId(Long resourceId) { this.resourceId = resourceId; }
        public Long getUserId() { return userId; }
        public void setUserId(Long userId) { this.userId = userId; }
        public String getPermissionType() { return permissionType; }
        public void setPermissionType(String permissionType) { this.permissionType = permissionType; }
        public Boolean getGranted() { return granted; }
        public void setGranted(Boolean granted) { this.granted = granted; }
        public LocalDateTime getExpireTime() { return expireTime; }
        public void setExpireTime(LocalDateTime expireTime) { this.expireTime = expireTime; }
    }

    /**
     * 资源权限响应DTO
     */
    public static class ResourcePermissionResponse {
        private Long permissionId;
        private Long resourceId;
        private String resourceName;
        private Long userId;
        private String userName;
        private String permissionType;
        private Boolean granted;
        private LocalDateTime grantTime;
        private LocalDateTime expireTime;
        private String granterName;
        
        // Getters and Setters
        public Long getPermissionId() { return permissionId; }
        public void setPermissionId(Long permissionId) { this.permissionId = permissionId; }
        public Long getResourceId() { return resourceId; }
        public void setResourceId(Long resourceId) { this.resourceId = resourceId; }
        public String getResourceName() { return resourceName; }
        public void setResourceName(String resourceName) { this.resourceName = resourceName; }
        public Long getUserId() { return userId; }
        public void setUserId(Long userId) { this.userId = userId; }
        public String getUserName() { return userName; }
        public void setUserName(String userName) { this.userName = userName; }
        public String getPermissionType() { return permissionType; }
        public void setPermissionType(String permissionType) { this.permissionType = permissionType; }
        public Boolean getGranted() { return granted; }
        public void setGranted(Boolean granted) { this.granted = granted; }
        public LocalDateTime getGrantTime() { return grantTime; }
        public void setGrantTime(LocalDateTime grantTime) { this.grantTime = grantTime; }
        public LocalDateTime getExpireTime() { return expireTime; }
        public void setExpireTime(LocalDateTime expireTime) { this.expireTime = expireTime; }
        public String getGranterName() { return granterName; }
        public void setGranterName(String granterName) { this.granterName = granterName; }
    }

    /**
     * 存储使用情况响应DTO
     */
    public static class StorageUsageResponse {
        private Long totalStorage;
        private Long usedStorage;
        private Long availableStorage;
        private Double usagePercentage;
        private Map<String, Long> storageByType;
        private List<ResourceResponse> largestFiles;
        private LocalDateTime lastUpdated;
        
        // Getters and Setters
        public Long getTotalStorage() { return totalStorage; }
        public void setTotalStorage(Long totalStorage) { this.totalStorage = totalStorage; }
        public Long getUsedStorage() { return usedStorage; }
        public void setUsedStorage(Long usedStorage) { this.usedStorage = usedStorage; }
        public Long getAvailableStorage() { return availableStorage; }
        public void setAvailableStorage(Long availableStorage) { this.availableStorage = availableStorage; }
        public Double getUsagePercentage() { return usagePercentage; }
        public void setUsagePercentage(Double usagePercentage) { this.usagePercentage = usagePercentage; }
        public Map<String, Long> getStorageByType() { return storageByType; }
        public void setStorageByType(Map<String, Long> storageByType) { this.storageByType = storageByType; }
        public List<ResourceResponse> getLargestFiles() { return largestFiles; }
        public void setLargestFiles(List<ResourceResponse> largestFiles) { this.largestFiles = largestFiles; }
        public LocalDateTime getLastUpdated() { return lastUpdated; }
        public void setLastUpdated(LocalDateTime lastUpdated) { this.lastUpdated = lastUpdated; }
    }

    /**
     * 资源移动请求DTO
     */
    public static class ResourceMoveRequest {
        @NotNull(message = "目标文件夹ID不能为空")
        private Long targetFolderId;
        
        // Getters and Setters
        public Long getTargetFolderId() { return targetFolderId; }
        public void setTargetFolderId(Long targetFolderId) { this.targetFolderId = targetFolderId; }
    }
  }