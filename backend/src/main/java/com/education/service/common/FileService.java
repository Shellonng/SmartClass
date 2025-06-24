package com.education.service.common;

import com.education.dto.FileDTO;
import com.education.dto.common.PageRequest;
import com.education.dto.common.PageResponse;
import org.springframework.web.multipart.MultipartFile;

import java.util.List;

/**
 * 公共文件服务接口
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
public interface FileService {

    /**
     * 上传单个文件
     * 
     * @param file 文件
     * @param uploadRequest 上传请求参数
     * @param userId 用户ID
     * @return 文件信息
     */
    FileDTO.FileUploadResponse uploadFile(MultipartFile file, FileDTO.FileUploadRequest uploadRequest, Long userId);

    /**
     * 批量上传文件
     * 
     * @param files 文件列表
     * @param uploadRequest 上传请求参数
     * @param userId 用户ID
     * @return 文件信息列表
     */
    List<FileDTO.FileUploadResponse> uploadFiles(List<MultipartFile> files, FileDTO.BatchUploadRequest uploadRequest, Long userId);

    /**
     * 下载文件
     * 
     * @param fileId 文件ID
     * @param userId 用户ID
     * @return 文件下载信息
     */
    FileDTO.FileDownloadResponse downloadFile(Long fileId, Long userId);

    /**
     * 预览文件
     * 
     * @param fileId 文件ID
     * @param userId 用户ID
     * @return 文件预览信息
     */
    FileDTO.FilePreviewResponse previewFile(Long fileId, Long userId);

    /**
     * 获取文件信息
     * 
     * @param fileId 文件ID
     * @param userId 用户ID
     * @return 文件信息
     */
    FileDTO.FileInfoResponse getFileInfo(Long fileId, Long userId);

    /**
     * 删除单个文件
     * 
     * @param fileId 文件ID
     * @param userId 用户ID
     * @return 操作结果
     */
    Boolean deleteFile(Long fileId, Long userId);

    /**
     * 批量删除文件
     * 
     * @param fileIds 文件ID列表
     * @param userId 用户ID
     * @return 操作结果
     */
    Boolean deleteFiles(List<Long> fileIds, Long userId);

    /**
     * 获取文件列表
     * 
     * @param listRequest 列表请求参数
     * @param userId 用户ID
     * @return 文件列表
     */
    PageResponse<FileDTO.FileListResponse> getFileList(FileDTO.FileListRequest listRequest, Long userId);

    /**
     * 重命名文件
     * 
     * @param fileId 文件ID
     * @param newName 新文件名
     * @param userId 用户ID
     * @return 操作结果
     */
    Boolean renameFile(Long fileId, String newName, Long userId);

    /**
     * 移动文件
     * 
     * @param fileId 文件ID
     * @param targetPath 目标路径
     * @param userId 用户ID
     * @return 操作结果
     */
    Boolean moveFile(Long fileId, String targetPath, Long userId);

    /**
     * 复制文件
     * 
     * @param fileId 文件ID
     * @param targetPath 目标路径
     * @param userId 用户ID
     * @return 新文件ID
     */
    Long copyFile(Long fileId, String targetPath, Long userId);

    /**
     * 生成文件分享链接
     * 
     * @param shareRequest 分享请求
     * @param userId 用户ID
     * @return 分享链接信息
     */
    FileDTO.FileShareResponse generateShareLink(FileDTO.FileShareRequest shareRequest, Long userId);

    /**
     * 取消文件分享
     * 
     * @param shareId 分享ID
     * @param userId 用户ID
     * @return 操作结果
     */
    Boolean cancelShare(Long shareId, Long userId);

    /**
     * 通过分享链接访问文件
     * 
     * @param shareToken 分享令牌
     * @param accessPassword 访问密码（可选）
     * @return 文件信息
     */
    FileDTO.SharedFileResponse accessSharedFile(String shareToken, String accessPassword);

    /**
     * 获取存储统计信息
     * 
     * @param userId 用户ID
     * @return 存储统计
     */
    FileDTO.StorageStatisticsResponse getStorageStatistics(Long userId);

    /**
     * 搜索文件
     * 
     * @param searchRequest 搜索请求
     * @param userId 用户ID
     * @return 搜索结果
     */
    PageResponse<FileDTO.FileListResponse> searchFiles(FileDTO.FileSearchRequest searchRequest, Long userId);

    /**
     * 获取文件版本历史
     * 
     * @param fileId 文件ID
     * @param userId 用户ID
     * @return 版本历史列表
     */
    List<FileDTO.FileVersionResponse> getFileVersions(Long fileId, Long userId);

    /**
     * 上传文件新版本
     * 
     * @param fileId 原文件ID
     * @param file 新版本文件
     * @param userId 用户ID
     * @return 新版本信息
     */
    FileDTO.FileVersionResponse uploadNewVersion(Long fileId, MultipartFile file, Long userId);

    /**
     * 恢复文件版本
     * 
     * @param fileId 文件ID
     * @param versionId 版本ID
     * @param userId 用户ID
     * @return 操作结果
     */
    Boolean restoreVersion(Long fileId, Long versionId, Long userId);

    /**
     * 设置文件标签
     * 
     * @param fileId 文件ID
     * @param tags 标签列表
     * @param userId 用户ID
     * @return 操作结果
     */
    Boolean setFileTags(Long fileId, List<String> tags, Long userId);

    /**
     * 获取文件标签
     * 
     * @param fileId 文件ID
     * @param userId 用户ID
     * @return 标签列表
     */
    List<String> getFileTags(Long fileId, Long userId);

    /**
     * 按标签筛选文件
     * 
     * @param tags 标签列表
     * @param userId 用户ID
     * @param pageRequest 分页请求
     * @return 文件列表
     */
    PageResponse<FileDTO.FileListResponse> getFilesByTags(List<String> tags, Long userId, PageRequest pageRequest);

    /**
     * 获取文件访问记录
     * 
     * @param fileId 文件ID
     * @param userId 用户ID
     * @param pageRequest 分页请求
     * @return 访问记录列表
     */
    PageResponse<FileDTO.FileAccessRecordResponse> getFileAccessRecords(Long fileId, Long userId, PageRequest pageRequest);

    /**
     * 设置文件权限
     * 
     * @param fileId 文件ID
     * @param permissionRequest 权限设置请求
     * @param userId 用户ID
     * @return 操作结果
     */
    Boolean setFilePermissions(Long fileId, FileDTO.FilePermissionRequest permissionRequest, Long userId);

    /**
     * 获取文件权限
     * 
     * @param fileId 文件ID
     * @param userId 用户ID
     * @return 文件权限信息
     */
    FileDTO.FilePermissionResponse getFilePermissions(Long fileId, Long userId);

    /**
     * 获取文件使用统计
     * 
     * @param fileId 文件ID
     * @param userId 用户ID
     * @return 使用统计
     */
    FileDTO.FileUsageStatisticsResponse getFileUsageStatistics(Long fileId, Long userId);

    /**
     * 收藏文件
     * 
     * @param fileId 文件ID
     * @param userId 用户ID
     * @return 操作结果
     */
    Boolean favoriteFile(Long fileId, Long userId);

    /**
     * 取消收藏文件
     * 
     * @param fileId 文件ID
     * @param userId 用户ID
     * @return 操作结果
     */
    Boolean unfavoriteFile(Long fileId, Long userId);

    /**
     * 获取收藏文件列表
     * 
     * @param userId 用户ID
     * @param pageRequest 分页请求
     * @return 收藏文件列表
     */
    PageResponse<FileDTO.FileListResponse> getFavoriteFiles(Long userId, PageRequest pageRequest);

    /**
     * 获取最近访问文件
     * 
     * @param userId 用户ID
     * @param limit 限制数量
     * @return 最近访问文件列表
     */
    List<FileDTO.FileListResponse> getRecentFiles(Long userId, Integer limit);

    /**
     * 获取推荐文件
     * 
     * @param userId 用户ID
     * @param pageRequest 分页请求
     * @return 推荐文件列表
     */
    PageResponse<FileDTO.FileListResponse> getRecommendedFiles(Long userId, PageRequest pageRequest);

    /**
     * 压缩文件
     * 
     * @param fileIds 文件ID列表
     * @param compressRequest 压缩请求
     * @param userId 用户ID
     * @return 压缩文件信息
     */
    FileDTO.FileCompressResponse compressFiles(List<Long> fileIds, FileDTO.FileCompressRequest compressRequest, Long userId);

    /**
     * 解压文件
     * 
     * @param fileId 压缩文件ID
     * @param extractPath 解压路径
     * @param userId 用户ID
     * @return 解压结果
     */
    FileDTO.FileExtractResponse extractFile(Long fileId, String extractPath, Long userId);

    /**
     * 获取存储空间使用情况
     * 
     * @param userId 用户ID
     * @return 存储空间使用情况
     */
    FileDTO.StorageUsageResponse getStorageUsage(Long userId);

    /**
     * 清理回收站
     * 
     * @param userId 用户ID
     * @return 操作结果
     */
    Boolean cleanRecycleBin(Long userId);

    /**
     * 从回收站恢复文件
     * 
     * @param fileId 文件ID
     * @param userId 用户ID
     * @return 操作结果
     */
    Boolean restoreFromRecycleBin(Long fileId, Long userId);

    /**
     * 获取回收站文件列表
     * 
     * @param userId 用户ID
     * @param pageRequest 分页请求
     * @return 回收站文件列表
     */
    PageResponse<FileDTO.RecycleBinFileResponse> getRecycleBinFiles(Long userId, PageRequest pageRequest);

    /**
     * 永久删除文件
     * 
     * @param fileId 文件ID
     * @param userId 用户ID
     * @return 操作结果
     */
    Boolean permanentlyDeleteFile(Long fileId, Long userId);

    /**
     * 检查文件完整性
     * 
     * @param fileId 文件ID
     * @param userId 用户ID
     * @return 完整性检查结果
     */
    FileDTO.FileIntegrityResponse checkFileIntegrity(Long fileId, Long userId);

    /**
     * 获取文件缩略图
     * 
     * @param fileId 文件ID
     * @param size 缩略图尺寸
     * @param userId 用户ID
     * @return 缩略图信息
     */
    FileDTO.FileThumbnailResponse getFileThumbnail(Long fileId, String size, Long userId);

    /**
     * 生成文件缩略图
     * 
     * @param fileId 文件ID
     * @param userId 用户ID
     * @return 操作结果
     */
    Boolean generateThumbnail(Long fileId, Long userId);

    /**
     * 获取文件元数据
     * 
     * @param fileId 文件ID
     * @param userId 用户ID
     * @return 文件元数据
     */
    FileDTO.FileMetadataResponse getFileMetadata(Long fileId, Long userId);

    /**
     * 更新文件元数据
     * 
     * @param fileId 文件ID
     * @param metadataRequest 元数据更新请求
     * @param userId 用户ID
     * @return 操作结果
     */
    Boolean updateFileMetadata(Long fileId, FileDTO.FileMetadataUpdateRequest metadataRequest, Long userId);

    /**
     * 同步文件
     * 
     * @param syncRequest 同步请求
     * @param userId 用户ID
     * @return 同步结果
     */
    FileDTO.FileSyncResponse syncFiles(FileDTO.FileSyncRequest syncRequest, Long userId);

    /**
     * 导出文件列表
     * 
     * @param exportRequest 导出请求
     * @param userId 用户ID
     * @return 导出文件信息
     */
    FileDTO.FileExportResponse exportFileList(FileDTO.FileExportRequest exportRequest, Long userId);
}