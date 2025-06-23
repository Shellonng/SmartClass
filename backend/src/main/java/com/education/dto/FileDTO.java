package com.education.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

/**
 * 文件相关DTO
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
public class FileDTO {

    /**
     * 文件上传请求DTO
     */
    public static class FileUploadRequest {
        @NotBlank(message = "文件名不能为空")
        private String fileName;
        
        @NotBlank(message = "文件类型不能为空")
        private String fileType;
        
        @NotNull(message = "文件大小不能为空")
        private Long fileSize;
        
        private String description;
        private String category; // DOCUMENT, IMAGE, VIDEO, AUDIO, OTHER
        private Boolean isPublic;
        private String tags;
        
        // Getters and Setters
        public String getFileName() { return fileName; }
        public void setFileName(String fileName) { this.fileName = fileName; }
        public String getFileType() { return fileType; }
        public void setFileType(String fileType) { this.fileType = fileType; }
        public Long getFileSize() { return fileSize; }
        public void setFileSize(Long fileSize) { this.fileSize = fileSize; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
        public String getCategory() { return category; }
        public void setCategory(String category) { this.category = category; }
        public Boolean getIsPublic() { return isPublic; }
        public void setIsPublic(Boolean isPublic) { this.isPublic = isPublic; }
        public String getTags() { return tags; }
        public void setTags(String tags) { this.tags = tags; }
    }

    /**
     * 文件响应DTO
     */
    public static class FileResponse {
        private Long fileId;
        private String fileName;
        private String originalName;
        private String fileType;
        private Long fileSize;
        private String filePath;
        private String fileUrl;
        private String description;
        private String category;
        private Boolean isPublic;
        private String tags;
        private String status; // UPLOADING, UPLOADED, PROCESSING, FAILED
        private Long uploaderId;
        private String uploaderName;
        private LocalDateTime uploadTime;
        private LocalDateTime updateTime;
        private String md5Hash;
        private Integer downloadCount;
        
        // Getters and Setters
        public Long getFileId() { return fileId; }
        public void setFileId(Long fileId) { this.fileId = fileId; }
        public String getFileName() { return fileName; }
        public void setFileName(String fileName) { this.fileName = fileName; }
        public String getOriginalName() { return originalName; }
        public void setOriginalName(String originalName) { this.originalName = originalName; }
        public String getFileType() { return fileType; }
        public void setFileType(String fileType) { this.fileType = fileType; }
        public Long getFileSize() { return fileSize; }
        public void setFileSize(Long fileSize) { this.fileSize = fileSize; }
        public String getFilePath() { return filePath; }
        public void setFilePath(String filePath) { this.filePath = filePath; }
        public String getFileUrl() { return fileUrl; }
        public void setFileUrl(String fileUrl) { this.fileUrl = fileUrl; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
        public String getCategory() { return category; }
        public void setCategory(String category) { this.category = category; }
        public Boolean getIsPublic() { return isPublic; }
        public void setIsPublic(Boolean isPublic) { this.isPublic = isPublic; }
        public String getTags() { return tags; }
        public void setTags(String tags) { this.tags = tags; }
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
        public Long getUploaderId() { return uploaderId; }
        public void setUploaderId(Long uploaderId) { this.uploaderId = uploaderId; }
        public String getUploaderName() { return uploaderName; }
        public void setUploaderName(String uploaderName) { this.uploaderName = uploaderName; }
        public LocalDateTime getUploadTime() { return uploadTime; }
        public void setUploadTime(LocalDateTime uploadTime) { this.uploadTime = uploadTime; }
        public LocalDateTime getUpdateTime() { return updateTime; }
        public void setUpdateTime(LocalDateTime updateTime) { this.updateTime = updateTime; }
        public String getMd5Hash() { return md5Hash; }
        public void setMd5Hash(String md5Hash) { this.md5Hash = md5Hash; }
        public Integer getDownloadCount() { return downloadCount; }
        public void setDownloadCount(Integer downloadCount) { this.downloadCount = downloadCount; }
    }

    /**
     * 文件更新请求DTO
     */
    public static class FileUpdateRequest {
        private String fileName;
        private String description;
        private String category;
        private Boolean isPublic;
        private String tags;
        
        // Getters and Setters
        public String getFileName() { return fileName; }
        public void setFileName(String fileName) { this.fileName = fileName; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
        public String getCategory() { return category; }
        public void setCategory(String category) { this.category = category; }
        public Boolean getIsPublic() { return isPublic; }
        public void setIsPublic(Boolean isPublic) { this.isPublic = isPublic; }
        public String getTags() { return tags; }
        public void setTags(String tags) { this.tags = tags; }
    }

    /**
     * 文件上传响应DTO
     */
    public static class FileUploadResponse {
        private Boolean success;
        private String message;
        private FileResponse fileInfo;
        private String uploadId;
        private String fileUrl;
        
        // Getters and Setters
        public Boolean getSuccess() { return success; }
        public void setSuccess(Boolean success) { this.success = success; }
        public String getMessage() { return message; }
        public void setMessage(String message) { this.message = message; }
        public FileResponse getFileInfo() { return fileInfo; }
        public void setFileInfo(FileResponse fileInfo) { this.fileInfo = fileInfo; }
        public String getUploadId() { return uploadId; }
        public void setUploadId(String uploadId) { this.uploadId = uploadId; }
        public String getFileUrl() { return fileUrl; }
        public void setFileUrl(String fileUrl) { this.fileUrl = fileUrl; }
    }

    /**
     * 批量文件上传请求DTO
     */
    public static class BatchFileUploadRequest {
        private List<FileUploadRequest> files;
        private String category;
        private Boolean isPublic;
        
        // Getters and Setters
        public List<FileUploadRequest> getFiles() { return files; }
        public void setFiles(List<FileUploadRequest> files) { this.files = files; }
        public String getCategory() { return category; }
        public void setCategory(String category) { this.category = category; }
        public Boolean getIsPublic() { return isPublic; }
        public void setIsPublic(Boolean isPublic) { this.isPublic = isPublic; }
    }

    /**
     * 批量文件上传响应DTO
     */
    public static class BatchFileUploadResponse {
        private Boolean success;
        private String message;
        private Integer totalCount;
        private Integer successCount;
        private Integer failCount;
        private List<FileResponse> successFiles;
        private List<String> errors;
        private LocalDateTime syncTime;
        private Long userId;
        private String syncStatus;
        
        // Getters and Setters
        public Boolean getSuccess() { return success; }
        public void setSuccess(Boolean success) { this.success = success; }
        public String getMessage() { return message; }
        public void setMessage(String message) { this.message = message; }
        public Integer getTotalCount() { return totalCount; }
        public void setTotalCount(Integer totalCount) { this.totalCount = totalCount; }
        public Integer getSuccessCount() { return successCount; }
        public void setSuccessCount(Integer successCount) { this.successCount = successCount; }
        public Integer getFailCount() { return failCount; }
        public void setFailCount(Integer failCount) { this.failCount = failCount; }
        public List<FileResponse> getSuccessFiles() { return successFiles; }
        public void setSuccessFiles(List<FileResponse> successFiles) { this.successFiles = successFiles; }
        public List<String> getErrors() { return errors; }
        public void setErrors(List<String> errors) { this.errors = errors; }
        public LocalDateTime getSyncTime() { return syncTime; }
        public void setSyncTime(LocalDateTime syncTime) { this.syncTime = syncTime; }
        public Long getUserId() { return userId; }
        public void setUserId(Long userId) { this.userId = userId; }
        public String getSyncStatus() { return syncStatus; }
        public void setSyncStatus(String syncStatus) { this.syncStatus = syncStatus; }
    }

    /**
     * 文件下载响应DTO
     */
    public static class FileDownloadResponse {
        private String fileName;
        private String fileType;
        private Long fileSize;
        private String downloadUrl;
        private LocalDateTime expireTime;
        private LocalDateTime exportTime;
        private Long userId;
        private String exportFilePath;
        private Integer totalFiles;
        private String exportStatus;
        private String message;
        
        // Getters and Setters
        public String getFileName() { return fileName; }
        public void setFileName(String fileName) { this.fileName = fileName; }
        public String getFileType() { return fileType; }
        public void setFileType(String fileType) { this.fileType = fileType; }
        public Long getFileSize() { return fileSize; }
        public void setFileSize(Long fileSize) { this.fileSize = fileSize; }
        public String getDownloadUrl() { return downloadUrl; }
        public void setDownloadUrl(String downloadUrl) { this.downloadUrl = downloadUrl; }
        public LocalDateTime getExpireTime() { return expireTime; }
        public void setExpireTime(LocalDateTime expireTime) { this.expireTime = expireTime; }
        public LocalDateTime getExportTime() { return exportTime; }
        public void setExportTime(LocalDateTime exportTime) { this.exportTime = exportTime; }
        public Long getUserId() { return userId; }
        public void setUserId(Long userId) { this.userId = userId; }
        public String getExportFilePath() { return exportFilePath; }
        public void setExportFilePath(String exportFilePath) { this.exportFilePath = exportFilePath; }
        public Integer getTotalFiles() { return totalFiles; }
        public void setTotalFiles(Integer totalFiles) { this.totalFiles = totalFiles; }
        public String getExportStatus() { return exportStatus; }
        public void setExportStatus(String exportStatus) { this.exportStatus = exportStatus; }
        public String getMessage() { return message; }
        public void setMessage(String message) { this.message = message; }
    }

    /**
     * 文件预览响应DTO
     */
    public static class FilePreviewResponse {
        private String previewUrl;
        private String previewType;
        private Boolean canPreview;
        private String message;
        private String integrityStatus;
        private String errorMessage;
        private Boolean isValid;
        private Long expectedSize;
        private Long actualSize;
        private LocalDateTime lastChecked;
        
        // Getters and Setters
        public String getPreviewUrl() { return previewUrl; }
        public void setPreviewUrl(String previewUrl) { this.previewUrl = previewUrl; }
        public String getPreviewType() { return previewType; }
        public void setPreviewType(String previewType) { this.previewType = previewType; }
        public Boolean getCanPreview() { return canPreview; }
        public void setCanPreview(Boolean canPreview) { this.canPreview = canPreview; }
        public String getMessage() { return message; }
        public void setMessage(String message) { this.message = message; }
        public String getIntegrityStatus() { return integrityStatus; }
        public void setIntegrityStatus(String integrityStatus) { this.integrityStatus = integrityStatus; }
        public String getErrorMessage() { return errorMessage; }
        public void setErrorMessage(String errorMessage) { this.errorMessage = errorMessage; }
        public Boolean getIsValid() { return isValid; }
        public void setIsValid(Boolean isValid) { this.isValid = isValid; }
        public Long getExpectedSize() { return expectedSize; }
        public void setExpectedSize(Long expectedSize) { this.expectedSize = expectedSize; }
        public Long getActualSize() { return actualSize; }
        public void setActualSize(Long actualSize) { this.actualSize = actualSize; }
        public LocalDateTime getLastChecked() { return lastChecked; }
        public void setLastChecked(LocalDateTime lastChecked) { this.lastChecked = lastChecked; }
    }

    /**
     * 文件信息响应DTO
     */
    public static class FileInfoResponse {
        private Long fileId;
        private String fileName;
        private String fileType;
        private Long fileSize;
        private String description;
        private String category;
        private Boolean isPublic;
        private String tags;
        private LocalDateTime uploadTime;
        private LocalDateTime updateTime;
        private String uploaderName;
        
        // Getters and Setters
        public Long getFileId() { return fileId; }
        public void setFileId(Long fileId) { this.fileId = fileId; }
        public String getFileName() { return fileName; }
        public void setFileName(String fileName) { this.fileName = fileName; }
        public String getFileType() { return fileType; }
        public void setFileType(String fileType) { this.fileType = fileType; }
        public Long getFileSize() { return fileSize; }
        public void setFileSize(Long fileSize) { this.fileSize = fileSize; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
        public String getCategory() { return category; }
        public void setCategory(String category) { this.category = category; }
        public Boolean getIsPublic() { return isPublic; }
        public void setIsPublic(Boolean isPublic) { this.isPublic = isPublic; }
        public String getTags() { return tags; }
        public void setTags(String tags) { this.tags = tags; }
        public LocalDateTime getUploadTime() { return uploadTime; }
        public void setUploadTime(LocalDateTime uploadTime) { this.uploadTime = uploadTime; }
        public LocalDateTime getUpdateTime() { return updateTime; }
        public void setUpdateTime(LocalDateTime updateTime) { this.updateTime = updateTime; }
        public String getUploaderName() { return uploaderName; }
        public void setUploaderName(String uploaderName) { this.uploaderName = uploaderName; }
    }

    /**
     * 文件列表请求DTO
     */
    public static class FileListRequest {
        private String keyword;
        private String fileType;
        private String category;
        private Boolean isPublic;
        private String sortBy;
        private String sortOrder;
        private Integer pageNum = 1;
        private Integer pageSize = 20;
        
        // Getters and Setters
        public String getKeyword() { return keyword; }
        public void setKeyword(String keyword) { this.keyword = keyword; }
        public String getFileType() { return fileType; }
        public void setFileType(String fileType) { this.fileType = fileType; }
        public String getCategory() { return category; }
        public void setCategory(String category) { this.category = category; }
        public Boolean getIsPublic() { return isPublic; }
        public void setIsPublic(Boolean isPublic) { this.isPublic = isPublic; }
        public String getSortBy() { return sortBy; }
        public void setSortBy(String sortBy) { this.sortBy = sortBy; }
        public String getSortOrder() { return sortOrder; }
        public void setSortOrder(String sortOrder) { this.sortOrder = sortOrder; }
        public Integer getPageNum() { return pageNum; }
        public void setPageNum(Integer pageNum) { this.pageNum = pageNum; }
        public Integer getPageSize() { return pageSize; }
        public void setPageSize(Integer pageSize) { this.pageSize = pageSize; }
    }

    /**
     * 文件列表响应DTO
     */
    public static class FileListResponse {
        private Long fileId;
        private String fileName;
        private String fileType;
        private Long fileSize;
        private String category;
        private Boolean isPublic;
        private LocalDateTime uploadTime;
        private String uploaderName;
        
        // Getters and Setters
        public Long getFileId() { return fileId; }
        public void setFileId(Long fileId) { this.fileId = fileId; }
        public String getFileName() { return fileName; }
        public void setFileName(String fileName) { this.fileName = fileName; }
        public String getFileType() { return fileType; }
        public void setFileType(String fileType) { this.fileType = fileType; }
        public Long getFileSize() { return fileSize; }
        public void setFileSize(Long fileSize) { this.fileSize = fileSize; }
        public String getCategory() { return category; }
        public void setCategory(String category) { this.category = category; }
        public Boolean getIsPublic() { return isPublic; }
        public void setIsPublic(Boolean isPublic) { this.isPublic = isPublic; }
        public LocalDateTime getUploadTime() { return uploadTime; }
        public void setUploadTime(LocalDateTime uploadTime) { this.uploadTime = uploadTime; }
        public String getUploaderName() { return uploaderName; }
        public void setUploaderName(String uploaderName) { this.uploaderName = uploaderName; }
    }

    /**
     * 文件夹树形结构响应DTO
     */
    public static class FolderTreeResponse {
        private Long folderId;
        private String folderName;
        private Long parentId;
        private List<FolderTreeResponse> children;
        private Integer fileCount;
        
        // Getters and Setters
        public Long getFolderId() { return folderId; }
        public void setFolderId(Long folderId) { this.folderId = folderId; }
        public String getFolderName() { return folderName; }
        public void setFolderName(String folderName) { this.folderName = folderName; }
        public Long getParentId() { return parentId; }
        public void setParentId(Long parentId) { this.parentId = parentId; }
        public List<FolderTreeResponse> getChildren() { return children; }
        public void setChildren(List<FolderTreeResponse> children) { this.children = children; }
        public Integer getFileCount() { return fileCount; }
        public void setFileCount(Integer fileCount) { this.fileCount = fileCount; }
    }

    /**
     * 文件夹内容响应DTO
     */
    public static class FolderContentResponse {
        private List<FileListResponse> files;
        private List<FolderTreeResponse> folders;
        private Integer totalFiles;
        private Integer totalFolders;
        
        // Getters and Setters
        public List<FileListResponse> getFiles() { return files; }
        public void setFiles(List<FileListResponse> files) { this.files = files; }
        public List<FolderTreeResponse> getFolders() { return folders; }
        public void setFolders(List<FolderTreeResponse> folders) { this.folders = folders; }
        public Integer getTotalFiles() { return totalFiles; }
        public void setTotalFiles(Integer totalFiles) { this.totalFiles = totalFiles; }
        public Integer getTotalFolders() { return totalFolders; }
        public void setTotalFolders(Integer totalFolders) { this.totalFolders = totalFolders; }
    }

    /**
     * 共享文件响应DTO
     */
    public static class SharedFileResponse {
        private Long shareId;
        private Long fileId;
        private String fileName;
        private String shareCode;
        private String shareUrl;
        private LocalDateTime expireTime;
        private Boolean hasPassword;
        private Integer downloadCount;
        private String fileType;
        private Long fileSize;
        private String description;
        private LocalDateTime uploadTime;
        private Boolean allowDownload;
        private String sharerName;
        
        // Getters and Setters
        public Long getShareId() { return shareId; }
        public void setShareId(Long shareId) { this.shareId = shareId; }
        public Long getFileId() { return fileId; }
        public void setFileId(Long fileId) { this.fileId = fileId; }
        public String getFileName() { return fileName; }
        public void setFileName(String fileName) { this.fileName = fileName; }
        public String getShareCode() { return shareCode; }
        public void setShareCode(String shareCode) { this.shareCode = shareCode; }
        public String getShareUrl() { return shareUrl; }
        public void setShareUrl(String shareUrl) { this.shareUrl = shareUrl; }
        public LocalDateTime getExpireTime() { return expireTime; }
        public void setExpireTime(LocalDateTime expireTime) { this.expireTime = expireTime; }
        public Boolean getHasPassword() { return hasPassword; }
        public void setHasPassword(Boolean hasPassword) { this.hasPassword = hasPassword; }
        public Integer getDownloadCount() { return downloadCount; }
        public void setDownloadCount(Integer downloadCount) { this.downloadCount = downloadCount; }
        public String getFileType() { return fileType; }
        public void setFileType(String fileType) { this.fileType = fileType; }
        public Long getFileSize() { return fileSize; }
        public void setFileSize(Long fileSize) { this.fileSize = fileSize; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
        public LocalDateTime getUploadTime() { return uploadTime; }
        public void setUploadTime(LocalDateTime uploadTime) { this.uploadTime = uploadTime; }
        public Boolean getAllowDownload() { return allowDownload; }
        public void setAllowDownload(Boolean allowDownload) { this.allowDownload = allowDownload; }
        public String getSharerName() { return sharerName; }
        public void setSharerName(String sharerName) { this.sharerName = sharerName; }
    }

    /**
     * 存储统计响应DTO
     */
    public static class StorageStatisticsResponse {
        private Long totalStorage;
        private Long usedStorage;
        private Long availableStorage;
        private Integer totalFiles;
        private Long totalSize;
        private Map<String, Integer> fileTypeCount;
        private Map<String, Long> categorySize;
        private Map<String, Long> typeCount;
        private Map<String, Long> typeSize;
        
        // Getters and Setters
        public Long getTotalStorage() { return totalStorage; }
        public void setTotalStorage(Long totalStorage) { this.totalStorage = totalStorage; }
        public Long getUsedStorage() { return usedStorage; }
        public void setUsedStorage(Long usedStorage) { this.usedStorage = usedStorage; }
        public Long getAvailableStorage() { return availableStorage; }
        public void setAvailableStorage(Long availableStorage) { this.availableStorage = availableStorage; }
        public Integer getTotalFiles() { return totalFiles; }
        public void setTotalFiles(Integer totalFiles) { this.totalFiles = totalFiles; }
        public Long getTotalSize() { return totalSize; }
        public void setTotalSize(Long totalSize) { this.totalSize = totalSize; }
        public Map<String, Integer> getFileTypeCount() { return fileTypeCount; }
        public void setFileTypeCount(Map<String, Integer> fileTypeCount) { this.fileTypeCount = fileTypeCount; }
        public Map<String, Long> getCategorySize() { return categorySize; }
        public void setCategorySize(Map<String, Long> categorySize) { this.categorySize = categorySize; }
        public Map<String, Long> getTypeCount() { return typeCount; }
        public void setTypeCount(Map<String, Long> typeCount) { this.typeCount = typeCount; }
        public Map<String, Long> getTypeSize() { return typeSize; }
        public void setTypeSize(Map<String, Long> typeSize) { this.typeSize = typeSize; }
    }

    /**
     * 文件版本响应DTO
     */
    public static class FileVersionResponse {
        private Long versionId;
        private String versionNumber;
        private Long fileSize;
        private LocalDateTime createTime;
        private String description;
        private Boolean isCurrent;
        private String fileName;
        private LocalDateTime uploadTime;
        private Long uploaderId;
        private String versionComment;
        
        // Getters and Setters
        public Long getVersionId() { return versionId; }
        public void setVersionId(Long versionId) { this.versionId = versionId; }
        public String getVersionNumber() { return versionNumber; }
        public void setVersionNumber(String versionNumber) { this.versionNumber = versionNumber; }
        public Long getFileSize() { return fileSize; }
        public void setFileSize(Long fileSize) { this.fileSize = fileSize; }
        public LocalDateTime getCreateTime() { return createTime; }
        public void setCreateTime(LocalDateTime createTime) { this.createTime = createTime; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
        public Boolean getIsCurrent() { return isCurrent; }
        public void setIsCurrent(Boolean isCurrent) { this.isCurrent = isCurrent; }
        public String getFileName() { return fileName; }
        public void setFileName(String fileName) { this.fileName = fileName; }
        public LocalDateTime getUploadTime() { return uploadTime; }
        public void setUploadTime(LocalDateTime uploadTime) { this.uploadTime = uploadTime; }
        public Long getUploaderId() { return uploaderId; }
        public void setUploaderId(Long uploaderId) { this.uploaderId = uploaderId; }
        public String getVersionComment() { return versionComment; }
        public void setVersionComment(String versionComment) { this.versionComment = versionComment; }
    }

    /**
     * 回收站文件响应DTO
     */
    public static class RecycleBinFileResponse {
        private Long fileId;
        private String fileName;
        private String fileType;
        private Long fileSize;
        private LocalDateTime deleteTime;
        private LocalDateTime expireTime;
        private String deletedBy;
        private String originalPath;
        
        // Getters and Setters
        public Long getFileId() { return fileId; }
        public void setFileId(Long fileId) { this.fileId = fileId; }
        public String getFileName() { return fileName; }
        public void setFileName(String fileName) { this.fileName = fileName; }
        public String getFileType() { return fileType; }
        public void setFileType(String fileType) { this.fileType = fileType; }
        public Long getFileSize() { return fileSize; }
        public void setFileSize(Long fileSize) { this.fileSize = fileSize; }
        public LocalDateTime getDeleteTime() { return deleteTime; }
        public void setDeleteTime(LocalDateTime deleteTime) { this.deleteTime = deleteTime; }
        public LocalDateTime getExpireTime() { return expireTime; }
        public void setExpireTime(LocalDateTime expireTime) { this.expireTime = expireTime; }
        public String getDeletedBy() { return deletedBy; }
        public void setDeletedBy(String deletedBy) { this.deletedBy = deletedBy; }
        public String getOriginalPath() { return originalPath; }
        public void setOriginalPath(String originalPath) { this.originalPath = originalPath; }
    }

    /**
     * 文件分享请求DTO
     */
    public static class FileShareRequest {
        private Long fileId;
        private String password;
        private LocalDateTime expireTime;
        private Boolean allowDownload;
        private Boolean allowPreview;
        
        // Getters and Setters
        public Long getFileId() { return fileId; }
        public void setFileId(Long fileId) { this.fileId = fileId; }
        public String getPassword() { return password; }
        public void setPassword(String password) { this.password = password; }
        public String getAccessPassword() { return password; }
        public LocalDateTime getExpireTime() { return expireTime; }
        public void setExpireTime(LocalDateTime expireTime) { this.expireTime = expireTime; }
        public Boolean getAllowDownload() { return allowDownload; }
        public void setAllowDownload(Boolean allowDownload) { this.allowDownload = allowDownload; }
        public Boolean getAllowPreview() { return allowPreview; }
        public void setAllowPreview(Boolean allowPreview) { this.allowPreview = allowPreview; }
    }

    /**
     * 文件分享响应DTO
     */
    public static class FileShareResponse {
        private String shareCode;
        private String shareUrl;
        private LocalDateTime expireTime;
        private Boolean hasPassword;
        private Boolean allowDownload;
        private LocalDateTime createTime;
        private String accessPassword;
        
        // Getters and Setters
        public String getShareCode() { return shareCode; }
        public void setShareCode(String shareCode) { this.shareCode = shareCode; }
        public String getShareUrl() { return shareUrl; }
        public void setShareUrl(String shareUrl) { this.shareUrl = shareUrl; }
        public LocalDateTime getExpireTime() { return expireTime; }
        public void setExpireTime(LocalDateTime expireTime) { this.expireTime = expireTime; }
        public Boolean getHasPassword() { return hasPassword; }
        public void setHasPassword(Boolean hasPassword) { this.hasPassword = hasPassword; }
        public Boolean getAllowDownload() { return allowDownload; }
        public void setAllowDownload(Boolean allowDownload) { this.allowDownload = allowDownload; }
        public LocalDateTime getCreateTime() { return createTime; }
        public void setCreateTime(LocalDateTime createTime) { this.createTime = createTime; }
        public String getAccessPassword() { return accessPassword; }
        public void setAccessPassword(String accessPassword) { this.accessPassword = accessPassword; }
    }

    /**
     * 文件搜索请求DTO
     */
    public static class FileSearchRequest {
        private String keyword;
        private String fileType;
        private String category;
        private LocalDateTime startDate;
        private LocalDateTime endDate;
        private LocalDateTime startTime;
        private LocalDateTime endTime;
        private Long minSize;
        private Long maxSize;
        private String sortBy;
        private String sortOrder;
        private Integer pageNum;
        private Integer pageSize;
        
        // Getters and Setters
        public String getKeyword() { return keyword; }
        public void setKeyword(String keyword) { this.keyword = keyword; }
        public String getFileType() { return fileType; }
        public void setFileType(String fileType) { this.fileType = fileType; }
        public String getCategory() { return category; }
        public void setCategory(String category) { this.category = category; }
        public LocalDateTime getStartDate() { return startDate; }
        public void setStartDate(LocalDateTime startDate) { this.startDate = startDate; }
        public LocalDateTime getEndDate() { return endDate; }
        public void setEndDate(LocalDateTime endDate) { this.endDate = endDate; }
        public LocalDateTime getStartTime() { return startTime; }
        public void setStartTime(LocalDateTime startTime) { this.startTime = startTime; }
        public LocalDateTime getEndTime() { return endTime; }
        public void setEndTime(LocalDateTime endTime) { this.endTime = endTime; }
        public Long getMinSize() { return minSize; }
        public void setMinSize(Long minSize) { this.minSize = minSize; }
        public Long getMaxSize() { return maxSize; }
        public void setMaxSize(Long maxSize) { this.maxSize = maxSize; }
        public String getSortBy() { return sortBy; }
        public void setSortBy(String sortBy) { this.sortBy = sortBy; }
        public String getSortOrder() { return sortOrder; }
        public void setSortOrder(String sortOrder) { this.sortOrder = sortOrder; }
        public Integer getPageNum() { return pageNum; }
        public void setPageNum(Integer pageNum) { this.pageNum = pageNum; }
        public Integer getPageSize() { return pageSize; }
        public void setPageSize(Integer pageSize) { this.pageSize = pageSize; }
    }

    /**
     * 文件访问记录响应DTO
     */
    public static class FileAccessRecordResponse {
        private Long recordId;
        private String userName;
        private String operation;
        private LocalDateTime accessTime;
        private String ipAddress;
        private String userAgent;
        private Long accessId;
        private Long fileId;
        private Long userId;
        private String accessType;
        
        // Getters and Setters
        public Long getRecordId() { return recordId; }
        public void setRecordId(Long recordId) { this.recordId = recordId; }
        public String getUserName() { return userName; }
        public void setUserName(String userName) { this.userName = userName; }
        public String getOperation() { return operation; }
        public void setOperation(String operation) { this.operation = operation; }
        public LocalDateTime getAccessTime() { return accessTime; }
        public void setAccessTime(LocalDateTime accessTime) { this.accessTime = accessTime; }
        public String getIpAddress() { return ipAddress; }
        public void setIpAddress(String ipAddress) { this.ipAddress = ipAddress; }
        public String getUserAgent() { return userAgent; }
        public void setUserAgent(String userAgent) { this.userAgent = userAgent; }
        public Long getAccessId() { return accessId; }
        public void setAccessId(Long accessId) { this.accessId = accessId; }
        public Long getFileId() { return fileId; }
        public void setFileId(Long fileId) { this.fileId = fileId; }
        public Long getUserId() { return userId; }
        public void setUserId(Long userId) { this.userId = userId; }
        public String getAccessType() { return accessType; }
        public void setAccessType(String accessType) { this.accessType = accessType; }
    }

    /**
     * 文件权限请求DTO
     */
    public static class FilePermissionRequest {
        private Long fileId;
        private Long userId;
        private String permission;
        private LocalDateTime expireTime;
        private Boolean canRead;
        private Boolean canWrite;
        private Boolean canDelete;
        private Boolean canDownload;
        
        // Getters and Setters
        public Long getFileId() { return fileId; }
        public void setFileId(Long fileId) { this.fileId = fileId; }
        public Long getUserId() { return userId; }
        public void setUserId(Long userId) { this.userId = userId; }
        public String getPermission() { return permission; }
        public void setPermission(String permission) { this.permission = permission; }
        public LocalDateTime getExpireTime() { return expireTime; }
        public void setExpireTime(LocalDateTime expireTime) { this.expireTime = expireTime; }
        public Boolean getCanRead() { return canRead; }
        public void setCanRead(Boolean canRead) { this.canRead = canRead; }
        public Boolean getCanWrite() { return canWrite; }
        public void setCanWrite(Boolean canWrite) { this.canWrite = canWrite; }
        public Boolean getCanDelete() { return canDelete; }
        public void setCanDelete(Boolean canDelete) { this.canDelete = canDelete; }
        public Boolean getCanDownload() { return canDownload; }
        public void setCanDownload(Boolean canDownload) { this.canDownload = canDownload; }
    }

    /**
     * 文件权限响应DTO
     */
    public static class FilePermissionResponse {
        private Long permissionId;
        private String userName;
        private String permission;
        private LocalDateTime grantTime;
        private LocalDateTime expireTime;
        private Boolean isActive;
        private Long fileId;
        private Long ownerId;
        private Boolean canRead;
        private Boolean canWrite;
        private Boolean canDownload;
        private Boolean canShare;
        private Boolean canDelete;
        private Boolean isPublic;
        private String permissionLevel;
        
        // Getters and Setters
        public Long getPermissionId() { return permissionId; }
        public void setPermissionId(Long permissionId) { this.permissionId = permissionId; }
        public String getUserName() { return userName; }
        public void setUserName(String userName) { this.userName = userName; }
        public String getPermission() { return permission; }
        public void setPermission(String permission) { this.permission = permission; }
        public LocalDateTime getGrantTime() { return grantTime; }
        public void setGrantTime(LocalDateTime grantTime) { this.grantTime = grantTime; }
        public LocalDateTime getExpireTime() { return expireTime; }
        public void setExpireTime(LocalDateTime expireTime) { this.expireTime = expireTime; }
        public Boolean getIsActive() { return isActive; }
        public void setIsActive(Boolean isActive) { this.isActive = isActive; }
        public Long getFileId() { return fileId; }
        public void setFileId(Long fileId) { this.fileId = fileId; }
        public Long getOwnerId() { return ownerId; }
        public void setOwnerId(Long ownerId) { this.ownerId = ownerId; }
        public Boolean getCanRead() { return canRead; }
        public void setCanRead(Boolean canRead) { this.canRead = canRead; }
        public Boolean getCanWrite() { return canWrite; }
        public void setCanWrite(Boolean canWrite) { this.canWrite = canWrite; }
        public Boolean getCanDownload() { return canDownload; }
        public void setCanDownload(Boolean canDownload) { this.canDownload = canDownload; }
        public Boolean getCanShare() { return canShare; }
        public void setCanShare(Boolean canShare) { this.canShare = canShare; }
        public Boolean getCanDelete() { return canDelete; }
        public void setCanDelete(Boolean canDelete) { this.canDelete = canDelete; }
        public Boolean getIsPublic() { return isPublic; }
        public void setIsPublic(Boolean isPublic) { this.isPublic = isPublic; }
        public String getPermissionLevel() { return permissionLevel; }
        public void setPermissionLevel(String permissionLevel) { this.permissionLevel = permissionLevel; }
    }

    /**
     * 文件使用统计响应DTO
     */
    public static class FileUsageStatisticsResponse {
        private Long fileId;
        private String fileName;
        private Integer downloadCount;
        private Integer viewCount;
        private Integer shareCount;
        private LocalDateTime lastAccess;
        private Double popularityScore;
        private Long totalViews;
        private Long totalDownloads;
        private Long totalShares;
        private LocalDateTime lastAccessTime;
        private LocalDateTime createdTime;
        private Long fileSize;
        private Long storageUsed;
        private Map<String, Long> dailyViews;
        
        // Getters and Setters
        public Long getFileId() { return fileId; }
        public void setFileId(Long fileId) { this.fileId = fileId; }
        public String getFileName() { return fileName; }
        public void setFileName(String fileName) { this.fileName = fileName; }
        public Integer getDownloadCount() { return downloadCount; }
        public void setDownloadCount(Integer downloadCount) { this.downloadCount = downloadCount; }
        public Integer getViewCount() { return viewCount; }
        public void setViewCount(Integer viewCount) { this.viewCount = viewCount; }
        public Integer getShareCount() { return shareCount; }
        public void setShareCount(Integer shareCount) { this.shareCount = shareCount; }
        public LocalDateTime getLastAccess() { return lastAccess; }
        public void setLastAccess(LocalDateTime lastAccess) { this.lastAccess = lastAccess; }
        public Double getPopularityScore() { return popularityScore; }
        public void setPopularityScore(Double popularityScore) { this.popularityScore = popularityScore; }
        public Long getTotalViews() { return totalViews; }
        public void setTotalViews(Long totalViews) { this.totalViews = totalViews; }
        public Long getTotalDownloads() { return totalDownloads; }
        public void setTotalDownloads(Long totalDownloads) { this.totalDownloads = totalDownloads; }
        public Long getTotalShares() { return totalShares; }
        public void setTotalShares(Long totalShares) { this.totalShares = totalShares; }
        public LocalDateTime getLastAccessTime() { return lastAccessTime; }
        public void setLastAccessTime(LocalDateTime lastAccessTime) { this.lastAccessTime = lastAccessTime; }
        public LocalDateTime getCreatedTime() { return createdTime; }
        public void setCreatedTime(LocalDateTime createdTime) { this.createdTime = createdTime; }
        public Long getFileSize() { return fileSize; }
        public void setFileSize(Long fileSize) { this.fileSize = fileSize; }
        public Long getStorageUsed() { return storageUsed; }
        public void setStorageUsed(Long storageUsed) { this.storageUsed = storageUsed; }
        public Map<String, Long> getDailyViews() { return dailyViews; }
        public void setDailyViews(Map<String, Long> dailyViews) { this.dailyViews = dailyViews; }
    }

    /**
     * 文件压缩请求DTO
     */
    public static class FileCompressRequest {
        private List<Long> fileIds;
        private String compressType;
        private String password;
        private String archiveName;
        private String compressName;
        
        // Getters and Setters
        public List<Long> getFileIds() { return fileIds; }
        public void setFileIds(List<Long> fileIds) { this.fileIds = fileIds; }
        public String getCompressType() { return compressType; }
        public void setCompressType(String compressType) { this.compressType = compressType; }
        public String getPassword() { return password; }
        public void setPassword(String password) { this.password = password; }
        public String getArchiveName() { return archiveName; }
        public void setArchiveName(String archiveName) { this.archiveName = archiveName; }
        public String getCompressName() { return compressName; }
        public void setCompressName(String compressName) { this.compressName = compressName; }
    }

    /**
     * 文件压缩响应DTO
     */
    public static class FileCompressResponse {
        private String taskId;
        private String status;
        private String downloadUrl;
        private Long compressedSize;
        private Integer progress;
        private Long compressId;
        private String compressName;
        private Long originalSize;
        private Double compressionRatio;
        private Integer fileCount;
        private LocalDateTime createTime;
        
        // Getters and Setters
        public String getTaskId() { return taskId; }
        public void setTaskId(String taskId) { this.taskId = taskId; }
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
        public String getDownloadUrl() { return downloadUrl; }
        public void setDownloadUrl(String downloadUrl) { this.downloadUrl = downloadUrl; }
        public Long getCompressedSize() { return compressedSize; }
        public void setCompressedSize(Long compressedSize) { this.compressedSize = compressedSize; }
        public Integer getProgress() { return progress; }
        public void setProgress(Integer progress) { this.progress = progress; }
        public Long getCompressId() { return compressId; }
        public void setCompressId(Long compressId) { this.compressId = compressId; }
        public String getCompressName() { return compressName; }
        public void setCompressName(String compressName) { this.compressName = compressName; }
        public Long getOriginalSize() { return originalSize; }
        public void setOriginalSize(Long originalSize) { this.originalSize = originalSize; }
        public Double getCompressionRatio() { return compressionRatio; }
        public void setCompressionRatio(Double compressionRatio) { this.compressionRatio = compressionRatio; }
        public Integer getFileCount() { return fileCount; }
        public void setFileCount(Integer fileCount) { this.fileCount = fileCount; }
        public LocalDateTime getCreateTime() { return createTime; }
        public void setCreateTime(LocalDateTime createTime) { this.createTime = createTime; }
    }

    /**
     * 文件解压响应DTO
     */
    public static class FileExtractResponse {
        private Long extractId;
        private Long originalFileId;
        private String extractPath;
        private Long totalSize;
        private String taskId;
        private String status;
        private List<String> extractedFiles;
        private Integer progress;
        private String message;
        private Long compressId;
        private String compressName;
        private Long originalSize;
        private Double compressionRatio;
        private Integer fileCount;
        private LocalDateTime createTime;
        private LocalDateTime extractTime;
        
        // Getters and Setters
        public Long getExtractId() { return extractId; }
        public void setExtractId(Long extractId) { this.extractId = extractId; }
        public Long getOriginalFileId() { return originalFileId; }
        public void setOriginalFileId(Long originalFileId) { this.originalFileId = originalFileId; }
        public String getExtractPath() { return extractPath; }
        public void setExtractPath(String extractPath) { this.extractPath = extractPath; }
        public Long getTotalSize() { return totalSize; }
        public void setTotalSize(Long totalSize) { this.totalSize = totalSize; }
        public String getTaskId() { return taskId; }
        public void setTaskId(String taskId) { this.taskId = taskId; }
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
        public List<String> getExtractedFiles() { return extractedFiles; }
        public void setExtractedFiles(List<String> extractedFiles) { this.extractedFiles = extractedFiles; }
        public Integer getProgress() { return progress; }
        public void setProgress(Integer progress) { this.progress = progress; }
        public Long getCompressId() { return compressId; }
        public void setCompressId(Long compressId) { this.compressId = compressId; }
        public String getCompressName() { return compressName; }
        public void setCompressName(String compressName) { this.compressName = compressName; }
        public Long getOriginalSize() { return originalSize; }
        public void setOriginalSize(Long originalSize) { this.originalSize = originalSize; }
        public Double getCompressionRatio() { return compressionRatio; }
        public void setCompressionRatio(Double compressionRatio) { this.compressionRatio = compressionRatio; }
        public Integer getFileCount() { return fileCount; }
        public void setFileCount(Integer fileCount) { this.fileCount = fileCount; }
        public LocalDateTime getCreateTime() { return createTime; }
        public void setCreateTime(LocalDateTime createTime) { this.createTime = createTime; }
        public String getMessage() { return message; }
        public void setMessage(String message) { this.message = message; }
        public LocalDateTime getExtractTime() { return extractTime; }
        public void setExtractTime(LocalDateTime extractTime) { this.extractTime = extractTime; }
    }

    /**
     * 存储使用响应DTO
     */
    public static class StorageUsageResponse {
        private Long userId;
        private String userName;
        private Long totalStorage;
        private Long usedStorage;
        private Long availableStorage;
        private Double usagePercentage;
        private Integer fileCount;
        private Long totalQuota;
        private Long usedSpace;
        private Long availableSpace;
        private Integer totalFiles;
        private Long recycleBinSize;
        private Long recycleBinFiles;
        
        // Getters and Setters
        public Long getUserId() { return userId; }
        public void setUserId(Long userId) { this.userId = userId; }
        public String getUserName() { return userName; }
        public void setUserName(String userName) { this.userName = userName; }
        public Long getTotalStorage() { return totalStorage; }
        public void setTotalStorage(Long totalStorage) { this.totalStorage = totalStorage; }
        public Long getUsedStorage() { return usedStorage; }
        public void setUsedStorage(Long usedStorage) { this.usedStorage = usedStorage; }
        public Long getAvailableStorage() { return availableStorage; }
        public void setAvailableStorage(Long availableStorage) { this.availableStorage = availableStorage; }
        public Double getUsagePercentage() { return usagePercentage; }
        public void setUsagePercentage(Double usagePercentage) { this.usagePercentage = usagePercentage; }
        public Integer getFileCount() { return fileCount; }
        public void setFileCount(Integer fileCount) { this.fileCount = fileCount; }
    }

    /**
     * 文件完整性响应DTO
     */
    public static class FileIntegrityResponse {
        private Long fileId;
        private String fileName;
        private Boolean isIntact;
        private String checksum;
        private String algorithm;
        private LocalDateTime checkTime;
        private String message;
        private String integrityStatus;
        private String errorMessage;
        private Boolean isValid;
        private Long expectedSize;
        private Long actualSize;
        private LocalDateTime lastChecked;
        
        // Getters and Setters
        public Long getFileId() { return fileId; }
        public void setFileId(Long fileId) { this.fileId = fileId; }
        public String getFileName() { return fileName; }
        public void setFileName(String fileName) { this.fileName = fileName; }
        public Boolean getIsIntact() { return isIntact; }
        public void setIsIntact(Boolean isIntact) { this.isIntact = isIntact; }
        public String getChecksum() { return checksum; }
        public void setChecksum(String checksum) { this.checksum = checksum; }
        public String getAlgorithm() { return algorithm; }
        public void setAlgorithm(String algorithm) { this.algorithm = algorithm; }
        public LocalDateTime getCheckTime() { return checkTime; }
        public void setCheckTime(LocalDateTime checkTime) { this.checkTime = checkTime; }
        public String getMessage() { return message; }
        public void setMessage(String message) { this.message = message; }
        public String getIntegrityStatus() { return integrityStatus; }
        public void setIntegrityStatus(String integrityStatus) { this.integrityStatus = integrityStatus; }
        public String getErrorMessage() { return errorMessage; }
        public void setErrorMessage(String errorMessage) { this.errorMessage = errorMessage; }
        public Boolean getIsValid() { return isValid; }
        public void setIsValid(Boolean isValid) { this.isValid = isValid; }
        public Long getExpectedSize() { return expectedSize; }
        public void setExpectedSize(Long expectedSize) { this.expectedSize = expectedSize; }
        public Long getActualSize() { return actualSize; }
        public void setActualSize(Long actualSize) { this.actualSize = actualSize; }
        public LocalDateTime getLastChecked() { return lastChecked; }
        public void setLastChecked(LocalDateTime lastChecked) { this.lastChecked = lastChecked; }
    }

    /**
     * 文件缩略图响应DTO
     */
    public static class FileThumbnailResponse {
        private String thumbnailUrl;
        private String thumbnailType;
        private Integer width;
        private Integer height;
        private Long thumbnailSize;
        private Boolean isGenerated;
        private Long fileId;
        private String originalFileName;
        private String requestedSize;
        private Boolean hasThumbnail;
        private String message;
        private String thumbnailPath;
        
        // Getters and Setters
        public String getThumbnailUrl() { return thumbnailUrl; }
        public void setThumbnailUrl(String thumbnailUrl) { this.thumbnailUrl = thumbnailUrl; }
        public String getThumbnailType() { return thumbnailType; }
        public void setThumbnailType(String thumbnailType) { this.thumbnailType = thumbnailType; }
        public Integer getWidth() { return width; }
        public void setWidth(Integer width) { this.width = width; }
        public Integer getHeight() { return height; }
        public void setHeight(Integer height) { this.height = height; }
        public Long getThumbnailSize() { return thumbnailSize; }
        public void setThumbnailSize(Long thumbnailSize) { this.thumbnailSize = thumbnailSize; }
        public Boolean getIsGenerated() { return isGenerated; }
        public void setIsGenerated(Boolean isGenerated) { this.isGenerated = isGenerated; }
        public Long getFileId() { return fileId; }
        public void setFileId(Long fileId) { this.fileId = fileId; }
        public String getOriginalFileName() { return originalFileName; }
        public void setOriginalFileName(String originalFileName) { this.originalFileName = originalFileName; }
        public String getRequestedSize() { return requestedSize; }
        public void setRequestedSize(String requestedSize) { this.requestedSize = requestedSize; }
        public Boolean getHasThumbnail() { return hasThumbnail; }
        public void setHasThumbnail(Boolean hasThumbnail) { this.hasThumbnail = hasThumbnail; }
        public String getMessage() { return message; }
        public void setMessage(String message) { this.message = message; }
        public String getThumbnailPath() { return thumbnailPath; }
        public void setThumbnailPath(String thumbnailPath) { this.thumbnailPath = thumbnailPath; }
    }

    /**
     * 文件元数据响应DTO
     */
    public static class FileMetadataResponse {
        private Map<String, Object> metadata;
        private String encoding;
        private String mimeType;
        private Integer width;
        private Integer height;
        private Long duration;
        private String author;
        private String title;
        private String subject;
        private Long fileId;
        private String fileName;
        private String fileType;
        private Long fileSize;
        private LocalDateTime uploadTime;
        private LocalDateTime updateTime;
        private String description;
        private String tags;
        private String filePath;
        private String thumbnailPath;
        private Boolean isDeleted;
        private Boolean canRead;
        private Boolean canWrite;
        private Boolean isHidden;
        private LocalDateTime lastModified;
        
        // Getters and Setters
        public Map<String, Object> getMetadata() { return metadata; }
        public void setMetadata(Map<String, Object> metadata) { this.metadata = metadata; }
        public String getEncoding() { return encoding; }
        public void setEncoding(String encoding) { this.encoding = encoding; }
        public String getMimeType() { return mimeType; }
        public void setMimeType(String mimeType) { this.mimeType = mimeType; }
        public Integer getWidth() { return width; }
        public void setWidth(Integer width) { this.width = width; }
        public Integer getHeight() { return height; }
        public void setHeight(Integer height) { this.height = height; }
        public Long getDuration() { return duration; }
        public void setDuration(Long duration) { this.duration = duration; }
        public String getAuthor() { return author; }
        public void setAuthor(String author) { this.author = author; }
        public String getTitle() { return title; }
        public void setTitle(String title) { this.title = title; }
        public String getSubject() { return subject; }
        public void setSubject(String subject) { this.subject = subject; }
        public Long getFileId() { return fileId; }
        public void setFileId(Long fileId) { this.fileId = fileId; }
        public String getFileName() { return fileName; }
        public void setFileName(String fileName) { this.fileName = fileName; }
        public String getFileType() { return fileType; }
        public void setFileType(String fileType) { this.fileType = fileType; }
        public Long getFileSize() { return fileSize; }
        public void setFileSize(Long fileSize) { this.fileSize = fileSize; }
        public LocalDateTime getUploadTime() { return uploadTime; }
        public void setUploadTime(LocalDateTime uploadTime) { this.uploadTime = uploadTime; }
        public LocalDateTime getUpdateTime() { return updateTime; }
        public void setUpdateTime(LocalDateTime updateTime) { this.updateTime = updateTime; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
        public String getTags() { return tags; }
        public void setTags(String tags) { this.tags = tags; }
        public String getFilePath() { return filePath; }
        public void setFilePath(String filePath) { this.filePath = filePath; }
        public String getThumbnailPath() { return thumbnailPath; }
        public void setThumbnailPath(String thumbnailPath) { this.thumbnailPath = thumbnailPath; }
        public Boolean getIsDeleted() { return isDeleted; }
        public void setIsDeleted(Boolean isDeleted) { this.isDeleted = isDeleted; }
        public Boolean getCanRead() { return canRead; }
        public void setCanRead(Boolean canRead) { this.canRead = canRead; }
        public Boolean getCanWrite() { return canWrite; }
        public void setCanWrite(Boolean canWrite) { this.canWrite = canWrite; }
        public Boolean getIsHidden() { return isHidden; }
        public void setIsHidden(Boolean isHidden) { this.isHidden = isHidden; }
        public LocalDateTime getLastModified() { return lastModified; }
        public void setLastModified(LocalDateTime lastModified) { this.lastModified = lastModified; }
    }

    /**
     * 文件元数据更新请求DTO
     */
    public static class FileMetadataUpdateRequest {
        private Long fileId;
        private String fileName;
        private String title;
        private String author;
        private String subject;
        private String description;
        private String tags;
        private Map<String, Object> customMetadata;
        
        // Getters and Setters
        public Long getFileId() { return fileId; }
        public void setFileId(Long fileId) { this.fileId = fileId; }
        public String getFileName() { return fileName; }
        public void setFileName(String fileName) { this.fileName = fileName; }
        public String getTags() { return tags; }
        public void setTags(String tags) { this.tags = tags; }
        public String getTitle() { return title; }
        public void setTitle(String title) { this.title = title; }
        public String getAuthor() { return author; }
        public void setAuthor(String author) { this.author = author; }
        public String getSubject() { return subject; }
        public void setSubject(String subject) { this.subject = subject; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
        public Map<String, Object> getCustomMetadata() { return customMetadata; }
        public void setCustomMetadata(Map<String, Object> customMetadata) { this.customMetadata = customMetadata; }
    }

    /**
     * 文件同步请求DTO
     */
    public static class FileSyncRequest {
        private List<Long> fileIds;
        private String targetLocation;
        private Boolean overwriteExisting;
        private Boolean preserveMetadata;
        private LocalDateTime lastSyncTime;
        
        // Getters and Setters
        public List<Long> getFileIds() { return fileIds; }
        public void setFileIds(List<Long> fileIds) { this.fileIds = fileIds; }
        public String getTargetLocation() { return targetLocation; }
        public void setTargetLocation(String targetLocation) { this.targetLocation = targetLocation; }
        public Boolean getOverwriteExisting() { return overwriteExisting; }
        public void setOverwriteExisting(Boolean overwriteExisting) { this.overwriteExisting = overwriteExisting; }
        public Boolean getPreserveMetadata() { return preserveMetadata; }
        public void setPreserveMetadata(Boolean preserveMetadata) { this.preserveMetadata = preserveMetadata; }
        public LocalDateTime getLastSyncTime() { return lastSyncTime; }
        public void setLastSyncTime(LocalDateTime lastSyncTime) { this.lastSyncTime = lastSyncTime; }
    }

    /**
     * 文件同步响应DTO
     */
    public static class FileSyncResponse {
        private String taskId;
        private String status;
        private Integer totalFiles;
        private Integer syncedFiles;
        private Integer failedFiles;
        private List<String> errors;
        private LocalDateTime syncTime;
        private Long userId;
        private String syncStatus;
        
        // Getters and Setters
        public String getTaskId() { return taskId; }
        public void setTaskId(String taskId) { this.taskId = taskId; }
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
        public Integer getTotalFiles() { return totalFiles; }
        public void setTotalFiles(Integer totalFiles) { this.totalFiles = totalFiles; }
        public Integer getSyncedFiles() { return syncedFiles; }
        public void setSyncedFiles(Integer syncedFiles) { this.syncedFiles = syncedFiles; }
        public Integer getFailedFiles() { return failedFiles; }
        public void setFailedFiles(Integer failedFiles) { this.failedFiles = failedFiles; }
        public List<String> getErrors() { return errors; }
        public void setErrors(List<String> errors) { this.errors = errors; }
        public LocalDateTime getSyncTime() { return syncTime; }
        public void setSyncTime(LocalDateTime syncTime) { this.syncTime = syncTime; }
        public Long getUserId() { return userId; }
        public void setUserId(Long userId) { this.userId = userId; }
        public String getSyncStatus() { return syncStatus; }
        public void setSyncStatus(String syncStatus) { this.syncStatus = syncStatus; }
    }

    /**
     * 文件导出请求DTO
     */
    public static class FileExportRequest {
        private List<Long> fileIds;
        private String exportFormat;
        private String exportType;
        private Map<String, Object> options;
        private Boolean includeDeleted;
        private String fileType;
        private LocalDateTime startDate;
        private LocalDateTime endDate;
        
        // Getters and Setters
        public List<Long> getFileIds() { return fileIds; }
        public void setFileIds(List<Long> fileIds) { this.fileIds = fileIds; }
        public String getExportFormat() { return exportFormat; }
        public void setExportFormat(String exportFormat) { this.exportFormat = exportFormat; }
        public String getExportType() { return exportType; }
        public void setExportType(String exportType) { this.exportType = exportType; }
        public Map<String, Object> getOptions() { return options; }
        public void setOptions(Map<String, Object> options) { this.options = options; }
        public Boolean getIncludeDeleted() { return includeDeleted; }
        public void setIncludeDeleted(Boolean includeDeleted) { this.includeDeleted = includeDeleted; }
        public String getFileType() { return fileType; }
        public void setFileType(String fileType) { this.fileType = fileType; }
        public LocalDateTime getStartDate() { return startDate; }
        public void setStartDate(LocalDateTime startDate) { this.startDate = startDate; }
        public LocalDateTime getEndDate() { return endDate; }
        public void setEndDate(LocalDateTime endDate) { this.endDate = endDate; }
    }

    /**
     * 文件导出响应DTO
     */
    public static class FileExportResponse {
        private String taskId;
        private String status;
        private String downloadUrl;
        private String exportFormat;
        private Long exportSize;
        private LocalDateTime expireTime;
        private LocalDateTime exportTime;
        private Long userId;
        private String exportFilePath;
        private Integer totalFiles;
        private String exportStatus;
        private String message;
        
        // Getters and Setters
        public String getTaskId() { return taskId; }
        public void setTaskId(String taskId) { this.taskId = taskId; }
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
        public String getDownloadUrl() { return downloadUrl; }
        public void setDownloadUrl(String downloadUrl) { this.downloadUrl = downloadUrl; }
        public String getExportFormat() { return exportFormat; }
        public void setExportFormat(String exportFormat) { this.exportFormat = exportFormat; }
        public Long getExportSize() { return exportSize; }
        public void setExportSize(Long exportSize) { this.exportSize = exportSize; }
        public LocalDateTime getExpireTime() { return expireTime; }
        public void setExpireTime(LocalDateTime expireTime) { this.expireTime = expireTime; }
        public LocalDateTime getExportTime() { return exportTime; }
        public void setExportTime(LocalDateTime exportTime) { this.exportTime = exportTime; }
        public Long getUserId() { return userId; }
        public void setUserId(Long userId) { this.userId = userId; }
        public String getExportFilePath() { return exportFilePath; }
        public void setExportFilePath(String exportFilePath) { this.exportFilePath = exportFilePath; }
        public Integer getTotalFiles() { return totalFiles; }
        public void setTotalFiles(Integer totalFiles) { this.totalFiles = totalFiles; }
        public String getExportStatus() { return exportStatus; }
        public void setExportStatus(String exportStatus) { this.exportStatus = exportStatus; }
        public String getMessage() { return message; }
        public void setMessage(String message) { this.message = message; }
    }

    /**
     * 同步文件信息DTO
     */
    public static class SyncFileInfo {
        private Long fileId;
        private String fileName;
        private String status;
        private String message;
        private LocalDateTime syncTime;
        private String fileType;
        private Long fileSize;
        private LocalDateTime lastModified;
        private String filePath;
        private String checksum;
        
        // Getters and Setters
        public Long getFileId() { return fileId; }
        public void setFileId(Long fileId) { this.fileId = fileId; }
        public String getFileName() { return fileName; }
        public void setFileName(String fileName) { this.fileName = fileName; }
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
        public String getMessage() { return message; }
        public void setMessage(String message) { this.message = message; }
        public LocalDateTime getSyncTime() { return syncTime; }
        public void setSyncTime(LocalDateTime syncTime) { this.syncTime = syncTime; }
        public String getFileType() { return fileType; }
        public void setFileType(String fileType) { this.fileType = fileType; }
        public Long getFileSize() { return fileSize; }
        public void setFileSize(Long fileSize) { this.fileSize = fileSize; }
        public LocalDateTime getLastModified() { return lastModified; }
        public void setLastModified(LocalDateTime lastModified) { this.lastModified = lastModified; }
        public String getFilePath() { return filePath; }
        public void setFilePath(String filePath) { this.filePath = filePath; }
        public String getChecksum() { return checksum; }
        public void setChecksum(String checksum) { this.checksum = checksum; }
    }

    /**
     * 批量上传请求DTO (修正名称)
     */
    public static class BatchUploadRequest {
        private List<FileUploadRequest> files;
        private String targetFolder;
        private Boolean overwriteExisting;
        
        // Getters and Setters
        public List<FileUploadRequest> getFiles() { return files; }
        public void setFiles(List<FileUploadRequest> files) { this.files = files; }
        public String getTargetFolder() { return targetFolder; }
        public void setTargetFolder(String targetFolder) { this.targetFolder = targetFolder; }
        public Boolean getOverwriteExisting() { return overwriteExisting; }
        public void setOverwriteExisting(Boolean overwriteExisting) { this.overwriteExisting = overwriteExisting; }
    }
}