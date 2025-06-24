package com.education.service.teacher;

import com.education.dto.ResourceDTO;
import com.education.dto.common.PageRequest;
import com.education.dto.common.PageResponse;

import java.util.List;

/**
 * 教师端资源服务接口
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
public interface ResourceService {

    /**
     * 上传资源
     * 
     * @param uploadRequest 上传请求
     * @param teacherId 教师ID
     * @return 资源信息
     */
    ResourceDTO.ResourceResponse uploadResource(ResourceDTO.ResourceUploadRequest uploadRequest, Long teacherId);

    /**
     * 获取资源列表
     * 
     * @param teacherId 教师ID
     * @param pageRequest 分页请求
     * @return 资源列表
     */
    PageResponse<ResourceDTO.ResourceResponse> getResourceList(Long teacherId, PageRequest pageRequest);

    /**
     * 获取资源详情
     * 
     * @param resourceId 资源ID
     * @param teacherId 教师ID
     * @return 资源详情
     */
    ResourceDTO.ResourceDetailResponse getResourceDetail(Long resourceId, Long teacherId);

    /**
     * 更新资源信息
     * 
     * @param resourceId 资源ID
     * @param updateRequest 更新请求
     * @param teacherId 教师ID
     * @return 更新后的资源信息
     */
    ResourceDTO.ResourceResponse updateResource(Long resourceId, ResourceDTO.ResourceUpdateRequest updateRequest, Long teacherId);

    /**
     * 删除资源
     * 
     * @param resourceId 资源ID
     * @param teacherId 教师ID
     * @return 操作结果
     */
    Boolean deleteResource(Long resourceId, Long teacherId);

    /**
     * 批量删除资源
     * 
     * @param resourceIds 资源ID列表
     * @param teacherId 教师ID
     * @return 操作结果
     */
    Boolean batchDeleteResources(List<Long> resourceIds, Long teacherId);

    /**
     * 分享资源
     * 
     * @param resourceId 资源ID
     * @param shareRequest 分享请求
     * @param teacherId 教师ID
     * @return 分享信息
     */
    ResourceDTO.ResourceShareResponse shareResource(Long resourceId, ResourceDTO.ResourceShareRequest shareRequest, Long teacherId);

    /**
     * 取消分享资源
     * 
     * @param resourceId 资源ID
     * @param teacherId 教师ID
     * @return 操作结果
     */
    Boolean unshareResource(Long resourceId, Long teacherId);

    /**
     * 获取资源分享记录
     * 
     * @param resourceId 资源ID
     * @param teacherId 教师ID
     * @param pageRequest 分页请求
     * @return 分享记录
     */
    PageResponse<ResourceDTO.ResourceShareRecordResponse> getResourceShareRecords(Long resourceId, Long teacherId, PageRequest pageRequest);

    /**
     * 下载资源
     * 
     * @param resourceId 资源ID
     * @param teacherId 教师ID
     * @return 下载链接或文件流
     */
    Object downloadResource(Long resourceId, Long teacherId);

    /**
     * 获取资源统计
     * 
     * @param teacherId 教师ID
     * @return 资源统计
     */
    ResourceDTO.ResourceStatisticsResponse getResourceStatistics(Long teacherId);

    /**
     * 批量上传资源
     * 
     * @param uploadRequests 批量上传请求
     * @param teacherId 教师ID
     * @return 上传结果
     */
    ResourceDTO.BatchUploadResponse batchUploadResources(List<ResourceDTO.ResourceUploadRequest> uploadRequests, Long teacherId);

    /**
     * 创建文件夹
     * 
     * @param folderRequest 文件夹创建请求
     * @param teacherId 教师ID
     * @return 文件夹信息
     */
    ResourceDTO.FolderResponse createFolder(ResourceDTO.FolderCreateRequest folderRequest, Long teacherId);

    /**
     * 移动资源
     * 
     * @param resourceId 资源ID
     * @param targetFolderId 目标文件夹ID
     * @param teacherId 教师ID
     * @return 操作结果
     */
    Boolean moveResource(Long resourceId, Long targetFolderId, Long teacherId);

    /**
     * 复制资源
     * 
     * @param resourceId 资源ID
     * @param targetFolderId 目标文件夹ID
     * @param teacherId 教师ID
     * @return 新资源信息
     */
    ResourceDTO.ResourceResponse copyResource(Long resourceId, Long targetFolderId, Long teacherId);

    /**
     * 搜索资源
     * 
     * @param keyword 关键词
     * @param teacherId 教师ID
     * @param pageRequest 分页请求
     * @return 搜索结果
     */
    PageResponse<ResourceDTO.ResourceResponse> searchResources(String keyword, Long teacherId, PageRequest pageRequest);

    /**
     * 获取资源访问记录
     * 
     * @param resourceId 资源ID
     * @param teacherId 教师ID
     * @param pageRequest 分页请求
     * @return 访问记录
     */
    PageResponse<ResourceDTO.ResourceAccessRecordResponse> getResourceAccessRecords(Long resourceId, Long teacherId, PageRequest pageRequest);

    /**
     * 设置资源权限
     * 
     * @param resourceId 资源ID
     * @param permissions 权限设置
     * @param teacherId 教师ID
     * @return 操作结果
     */
    Boolean setResourcePermissions(Long resourceId, ResourceDTO.ResourcePermissionRequest permissions, Long teacherId);

    /**
     * 获取资源权限
     * 
     * @param resourceId 资源ID
     * @param teacherId 教师ID
     * @return 权限信息
     */
    ResourceDTO.ResourcePermissionResponse getResourcePermissions(Long resourceId, Long teacherId);

    /**
     * 预览资源
     * 
     * @param resourceId 资源ID
     * @param teacherId 教师ID
     * @return 预览信息
     */
    Object previewResource(Long resourceId, Long teacherId);

    /**
     * 获取资源版本历史
     * 
     * @param resourceId 资源ID
     * @param teacherId 教师ID
     * @return 版本历史
     */
    List<ResourceDTO.ResourceVersionResponse> getResourceVersions(Long resourceId, Long teacherId);

    /**
     * 上传资源新版本
     * 
     * @param resourceId 资源ID
     * @param uploadRequest 上传请求
     * @param teacherId 教师ID
     * @return 新版本信息
     */
    ResourceDTO.ResourceVersionResponse uploadResourceVersion(Long resourceId, ResourceDTO.ResourceUploadRequest uploadRequest, Long teacherId);

    /**
     * 恢复资源版本
     * 
     * @param resourceId 资源ID
     * @param versionId 版本ID
     * @param teacherId 教师ID
     * @return 操作结果
     */
    Boolean restoreResourceVersion(Long resourceId, Long versionId, Long teacherId);

    /**
     * 设置资源标签
     * 
     * @param resourceId 资源ID
     * @param tags 标签列表
     * @param teacherId 教师ID
     * @return 操作结果
     */
    Boolean setResourceTags(Long resourceId, List<String> tags, Long teacherId);

    /**
     * 获取资源标签
     * 
     * @param resourceId 资源ID
     * @param teacherId 教师ID
     * @return 标签列表
     */
    List<String> getResourceTags(Long resourceId, Long teacherId);

    /**
     * 按标签筛选资源
     * 
     * @param tags 标签列表
     * @param teacherId 教师ID
     * @param pageRequest 分页请求
     * @return 资源列表
     */
    PageResponse<ResourceDTO.ResourceResponse> getResourcesByTags(List<String> tags, Long teacherId, PageRequest pageRequest);

    /**
     * 获取资源使用统计
     * 
     * @param resourceId 资源ID
     * @param teacherId 教师ID
     * @param timeRange 时间范围
     * @return 使用统计
     */
    Object getResourceUsageStatistics(Long resourceId, Long teacherId, String timeRange);

    /**
     * 收藏资源
     * 
     * @param resourceId 资源ID
     * @param teacherId 教师ID
     * @return 操作结果
     */
    Boolean favoriteResource(Long resourceId, Long teacherId);

    /**
     * 取消收藏资源
     * 
     * @param resourceId 资源ID
     * @param teacherId 教师ID
     * @return 操作结果
     */
    Boolean unfavoriteResource(Long resourceId, Long teacherId);

    /**
     * 获取收藏的资源列表
     * 
     * @param teacherId 教师ID
     * @param pageRequest 分页请求
     * @return 收藏资源列表
     */
    PageResponse<ResourceDTO.ResourceResponse> getFavoriteResources(Long teacherId, PageRequest pageRequest);

    /**
     * 获取最近访问的资源
     * 
     * @param teacherId 教师ID
     * @param limit 限制数量
     * @return 最近访问资源列表
     */
    List<ResourceDTO.ResourceResponse> getRecentResources(Long teacherId, Integer limit);

    /**
     * 获取推荐资源
     * 
     * @param teacherId 教师ID
     * @param limit 限制数量
     * @return 推荐资源列表
     */
    List<ResourceDTO.ResourceResponse> getRecommendedResources(Long teacherId, Integer limit);

    /**
     * 压缩资源
     * 
     * @param resourceIds 资源ID列表
     * @param teacherId 教师ID
     * @return 压缩文件信息
     */
    ResourceDTO.ResourceResponse compressResources(List<Long> resourceIds, Long teacherId);

    /**
     * 解压资源
     * 
     * @param resourceId 压缩文件资源ID
     * @param targetFolderId 目标文件夹ID
     * @param teacherId 教师ID
     * @return 解压结果
     */
    List<ResourceDTO.ResourceResponse> extractResource(Long resourceId, Long targetFolderId, Long teacherId);

    /**
     * 获取存储空间使用情况
     * 
     * @param teacherId 教师ID
     * @return 存储空间信息
     */
    ResourceDTO.StorageUsageResponse getStorageUsage(Long teacherId);

    /**
     * 清理回收站
     * 
     * @param teacherId 教师ID
     * @return 操作结果
     */
    Boolean cleanRecycleBin(Long teacherId);

    /**
     * 从回收站恢复资源
     * 
     * @param resourceId 资源ID
     * @param teacherId 教师ID
     * @return 操作结果
     */
    Boolean restoreFromRecycleBin(Long resourceId, Long teacherId);

    /**
     * 获取回收站资源列表
     * 
     * @param teacherId 教师ID
     * @param pageRequest 分页请求
     * @return 回收站资源列表
     */
    PageResponse<ResourceDTO.ResourceResponse> getRecycleBinResources(Long teacherId, PageRequest pageRequest);

    /**
     * 永久删除资源
     * 
     * @param resourceId 资源ID
     * @param teacherId 教师ID
     * @return 操作结果
     */
    Boolean permanentDeleteResource(Long resourceId, Long teacherId);
}