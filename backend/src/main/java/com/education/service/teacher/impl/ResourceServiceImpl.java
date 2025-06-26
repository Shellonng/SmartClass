package com.education.service.teacher.impl;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.education.dto.*;
import com.education.dto.common.PageRequest;
import com.education.dto.common.PageResponse;
import com.education.entity.Resource;
import com.education.entity.Teacher;
import com.education.entity.User;
import com.education.mapper.ResourceMapper;
import com.education.mapper.TeacherMapper;
import com.education.mapper.UserMapper;
import com.education.service.teacher.ResourceService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.BeanUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.util.StringUtils;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import java.util.List;
import java.util.UUID;
import java.util.stream.Collectors;

/**
 * 教师端资源服务实现类
 */
@Slf4j
@Service
public class ResourceServiceImpl implements ResourceService {
    
    @Autowired
    private ResourceMapper resourceMapper;
    
    @Autowired
    private TeacherMapper teacherMapper;
    
    @Autowired
    private UserMapper userMapper;
    
    // 资源存储路径
    private static final String UPLOAD_PATH = "uploads/resources/";
    
    /**
     * 转换资源实体为响应对象
     */
    private ResourceResponse convertToResourceResponse(Resource resource) {
        ResourceResponse response = new ResourceResponse();
        response.setResourceId(resource.getId());
        response.setResourceName(resource.getResourceName());
        response.setResourceType(resource.getResourceType());
        response.setDescription(resource.getDescription());
        response.setFileUrl(resource.getFileUrl());
        response.setFileName(resource.getResourceName());
        response.setFileSize(resource.getFileSize());
        response.setTags(resource.getTags());
        response.setVisibility(resource.getIsPublic() ? "public" : "private");
        response.setDownloadCount(resource.getDownloadCount() != null ? resource.getDownloadCount() : 0);
        response.setCreateTime(resource.getCreateTime());
        response.setUpdateTime(resource.getUpdateTime());
        
        // 获取上传者信息
        Teacher teacher = teacherMapper.selectById(resource.getUploaderId());
        if (teacher != null) {
            User user = userMapper.selectById(teacher.getUserId());
            if (user != null) {
                response.setUploaderName(user.getRealName());
            }
        }
        
        return response;
    }

    @Override
    @Transactional
    public ResourceResponse uploadResource(ResourceUploadRequest uploadRequest, Long teacherId) {
        log.info("上传资源，教师ID: {}, 资源名称: {}", teacherId, uploadRequest.getResourceName());
        
        // 验证教师是否存在
        Teacher teacher = teacherMapper.selectById(teacherId);
        if (teacher == null) {
            throw new RuntimeException("教师不存在");
        }
        
        try {
            // 处理文件上传
            MultipartFile file = uploadRequest.getFile();
            if (file == null || file.isEmpty()) {
                throw new RuntimeException("文件不能为空");
            }
            
            // 生成唯一文件名
            String originalFilename = file.getOriginalFilename();
            String fileExtension = originalFilename.substring(originalFilename.lastIndexOf("."));
            String fileName = UUID.randomUUID().toString() + fileExtension;
            
            // 创建上传目录
            Path uploadDir = Paths.get(UPLOAD_PATH);
            if (!Files.exists(uploadDir)) {
                Files.createDirectories(uploadDir);
            }
            
            // 保存文件
            Path filePath = uploadDir.resolve(fileName);
            Files.copy(file.getInputStream(), filePath);
            
            // 创建资源实体
            Resource resource = new Resource();
            resource.setResourceName(uploadRequest.getResourceName());
            resource.setResourceType(uploadRequest.getResourceType());
            resource.setDescription(uploadRequest.getDescription());
            resource.setFilePath(filePath.toString());
            resource.setFileSize(file.getSize());
            resource.setMimeType(file.getContentType());
            resource.setUploaderId(teacherId);
            resource.setCourseId(uploadRequest.getCourseId());
            resource.setIsPublic("public".equals(uploadRequest.getVisibility()));
            resource.setCreateTime(LocalDateTime.now());
            resource.setUpdateTime(LocalDateTime.now());
            resource.setIsDeleted(false);
            
            // 保存资源记录
            resourceMapper.insert(resource);
            
            return convertToResourceResponse(resource);
            
        } catch (IOException e) {
            log.error("文件上传失败", e);
            throw new RuntimeException("文件上传失败: " + e.getMessage());
        }
    }

    @Override
    public PageResponse<ResourceResponse> getResourceList(Long teacherId, PageRequest pageRequest) {
        log.info("获取资源列表，教师ID: {}, 页码: {}, 页大小: {}", teacherId, pageRequest.getPageNum(), pageRequest.getPageSize());
        
        // 构建分页对象
        Page<Resource> page = new Page<>(pageRequest.getPageNum(), pageRequest.getPageSize());
        
        // 构建查询条件
        QueryWrapper<Resource> wrapper = new QueryWrapper<>();
        wrapper.eq("uploader_id", teacherId)
               .eq("is_deleted", false)
               .orderByDesc("create_time");
        
        // 执行分页查询
        IPage<Resource> resourcePage = resourceMapper.selectPage(page, wrapper);
        
        // 转换为响应对象
        List<ResourceResponse> resourceResponses = resourcePage.getRecords().stream()
                .map(this::convertToResourceResponse)
                .collect(Collectors.toList());
        
        return new PageResponse<>(
                resourcePage.getCurrent(),
                resourcePage.getSize(),
                resourcePage.getTotal(),
                resourceResponses
        );
    }

    @Override
    public ResourceDetailResponse getResourceDetail(Long resourceId, Long teacherId) {
        log.info("获取资源详情，资源ID: {}, 教师ID: {}", resourceId, teacherId);
        
        // 查询资源
        Resource resource = resourceMapper.selectById(resourceId);
        if (resource == null || resource.getIsDeleted()) {
            throw new RuntimeException("资源不存在");
        }
        
        // 验证权限（只有上传者或公开资源可以查看）
        if (!resource.getUploaderId().equals(teacherId) && !resource.getIsPublic()) {
            throw new RuntimeException("无权限访问该资源");
        }
        
        // 转换为详情响应对象
        ResourceDetailResponse response = new ResourceDetailResponse();
        response.setResourceId(resource.getId());
        response.setTitle(resource.getResourceName());
        response.setDescription(resource.getDescription());
        response.setFileType(resource.getFileType());
        response.setFileSize(resource.getFileSize());
        response.setFilePath(resource.getFilePath());
        response.setDownloadCount(resource.getDownloadCount() != null ? resource.getDownloadCount() : 0);
        response.setCreatedTime(resource.getCreateTime());
        
        // 获取上传者信息
        Teacher teacher = teacherMapper.selectById(resource.getUploaderId());
        if (teacher != null) {
            User user = userMapper.selectById(teacher.getUserId());
            if (user != null) {
                // 设置上传者信息到resource中
                ResourceResponse resourceResponse = convertToResourceResponse(resource);
                response.setResource(resourceResponse);
            }
        }
        
        // TODO: 添加下载次数、使用统计等信息
        
        return response;
    }

    @Override
    @Transactional
    public ResourceResponse updateResource(Long resourceId, ResourceUpdateRequest updateRequest, Long teacherId) {
        log.info("更新资源，资源ID: {}, 教师ID: {}", resourceId, teacherId);
        
        // 查询资源
        Resource resource = resourceMapper.selectById(resourceId);
        if (resource == null || resource.getIsDeleted()) {
            throw new RuntimeException("资源不存在");
        }
        
        // 验证权限
        if (!resource.getUploaderId().equals(teacherId)) {
            throw new RuntimeException("无权限修改该资源");
        }
        
        // 更新资源信息
        if (StringUtils.hasText(updateRequest.getResourceName())) {
            resource.setResourceName(updateRequest.getResourceName());
        }
        if (StringUtils.hasText(updateRequest.getDescription())) {
            resource.setDescription(updateRequest.getDescription());
        }
        if (updateRequest.getVisibility() != null) {
            resource.setIsPublic("public".equals(updateRequest.getVisibility()));
        }
        
        resource.setUpdateTime(LocalDateTime.now());
        
        // 保存更新
        resourceMapper.updateById(resource);
        
        return convertToResourceResponse(resource);
    }

    @Override
    @Transactional
    public Boolean deleteResource(Long resourceId, Long teacherId) {
        log.info("删除资源，资源ID: {}, 教师ID: {}", resourceId, teacherId);
        
        // 查询资源
        Resource resource = resourceMapper.selectById(resourceId);
        if (resource == null || resource.getIsDeleted()) {
            throw new RuntimeException("资源不存在");
        }
        
        // 验证权限
        if (!resource.getUploaderId().equals(teacherId)) {
            throw new RuntimeException("无权限删除该资源");
        }
        
        // 软删除资源
        resource.setIsDeleted(true);
        resource.setUpdateTime(LocalDateTime.now());
        resourceMapper.updateById(resource);
        
        // TODO: 可以考虑物理删除文件
        
        return true;
    }

    @Override
    @Transactional
    public Boolean batchDeleteResources(List<Long> resourceIds, Long teacherId) {
        log.info("批量删除资源，资源ID列表: {}, 教师ID: {}", resourceIds, teacherId);
        
        if (resourceIds == null || resourceIds.isEmpty()) {
            throw new RuntimeException("资源ID列表不能为空");
        }
        
        for (Long resourceId : resourceIds) {
            // 查询资源
            Resource resource = resourceMapper.selectById(resourceId);
            if (resource == null || resource.getIsDeleted()) {
                continue; // 跳过不存在的资源
            }
            
            // 验证权限
            if (!resource.getUploaderId().equals(teacherId)) {
                throw new RuntimeException("无权限删除资源ID: " + resourceId);
            }
            
            // 软删除资源
            resource.setIsDeleted(true);
            resource.setUpdateTime(LocalDateTime.now());
            resourceMapper.updateById(resource);
        }
        
        return true;
    }

    @Override
    @Transactional
    public ResourceDTO.ResourceShareResponse shareResource(Long resourceId, ResourceShareRequest shareRequest, Long teacherId) {
        log.info("分享资源，资源ID: {}, 教师ID: {}", resourceId, teacherId);
        
        // 查询资源
        Resource resource = resourceMapper.selectById(resourceId);
        if (resource == null || resource.getIsDeleted()) {
            throw new RuntimeException("资源不存在");
        }
        
        // 验证权限
        if (!resource.getUploaderId().equals(teacherId)) {
            throw new RuntimeException("无权限分享该资源");
        }
        
        // 设置为公开
        resource.setIsPublic(true);
        resource.setUpdateTime(LocalDateTime.now());
        resourceMapper.updateById(resource);
        
        // 创建分享响应
        ResourceDTO.ResourceShareResponse response = new ResourceDTO.ResourceShareResponse();
        response.setShareUrl("/api/resources/public/" + resourceId);
        
        return response;
    }

    @Override
    public Boolean unshareResource(Long resourceId, Long teacherId) {
        // TODO: 实现取消分享资源逻辑
        throw new UnsupportedOperationException("Method not implemented yet");
    }

    @Override
    public PageResponse<ResourceShareRecordResponse> getResourceShareRecords(Long resourceId, Long teacherId, PageRequest pageRequest) {
        log.info("获取资源分享记录，资源ID: {}, 教师ID: {}", resourceId, teacherId);
        
        // 查询资源
        Resource resource = resourceMapper.selectById(resourceId);
        if (resource == null || resource.getIsDeleted()) {
            throw new RuntimeException("资源不存在");
        }
        
        // 验证权限
        if (!resource.getUploaderId().equals(teacherId)) {
            throw new RuntimeException("无权限查看该资源的分享记录");
        }
        
        // 构建分页对象
        Page<Resource> page = new Page<>(pageRequest.getPageNum(), pageRequest.getPageSize());
        
        // 构建查询条件 - 查询该教师的公开资源
        QueryWrapper<Resource> wrapper = new QueryWrapper<>();
        wrapper.eq("id", resourceId)
               .eq("is_public", true)
               .eq("is_deleted", false)
               .orderByDesc("update_time");
        
        // 执行分页查询
        IPage<Resource> resourcePage = resourceMapper.selectPage(page, wrapper);
        
        // 转换为分享记录响应
        List<ResourceShareRecordResponse> records = resourcePage.getRecords().stream().map(r -> {
            ResourceShareRecordResponse record = new ResourceShareRecordResponse();
            record.setResourceId(r.getId());
            record.setResourceName(r.getResourceName());
            record.setShareTime(r.getUpdateTime());
            return record;
        }).collect(Collectors.toList());
        
        return new PageResponse<>(
                resourcePage.getCurrent(),
                resourcePage.getSize(),
                resourcePage.getTotal(),
                records
        );
    }

    @Override
    public Object downloadResource(Long resourceId, Long teacherId) {
        log.info("下载资源，资源ID: {}, 教师ID: {}", resourceId, teacherId);
        
        // 查询资源
        Resource resource = resourceMapper.selectById(resourceId);
        if (resource == null || resource.getIsDeleted()) {
            throw new RuntimeException("资源不存在");
        }
        
        // 验证权限（只有上传者或公开资源可以下载）
        if (!resource.getUploaderId().equals(teacherId) && !resource.getIsPublic()) {
            throw new RuntimeException("无权限下载该资源");
        }
        
        try {
            // 读取文件内容
            Path filePath = Paths.get(resource.getFilePath());
            if (!Files.exists(filePath)) {
                throw new RuntimeException("文件不存在");
            }
            
            // TODO: 增加下载次数统计
            
            return Files.readAllBytes(filePath);
            
        } catch (IOException e) {
            log.error("文件下载失败", e);
            throw new RuntimeException("文件下载失败: " + e.getMessage());
        }
    }

    @Override
    public ResourceStatisticsResponse getResourceStatistics(Long teacherId) {
        log.info("获取资源统计，教师ID: {}", teacherId);
        
        // 验证教师是否存在
        Teacher teacher = teacherMapper.selectById(teacherId);
        if (teacher == null) {
            throw new RuntimeException("教师不存在");
        }
        
        // 构建查询条件
        QueryWrapper<Resource> wrapper = new QueryWrapper<>();
        wrapper.eq("uploader_id", teacherId)
               .eq("is_deleted", false);
        
        List<Resource> resources = resourceMapper.selectList(wrapper);
        
        // 统计数据
        ResourceStatisticsResponse statistics = new ResourceStatisticsResponse();
        statistics.setTotalResources((long) resources.size());
        statistics.setTotalDownloads(0L); // TODO: 实际下载次数统计
        statistics.setTotalViews(0L); // TODO: 实际查看次数统计
        statistics.setTotalShares((long) resources.stream().filter(Resource::getIsPublic).count());
        statistics.setAverageRating(0.0); // TODO: 实际评分统计
        
        // 按类型统计
        Map<String, Long> resourceTypeCount = new HashMap<>();
        resourceTypeCount.put("document", resources.stream().filter(r -> "document".equals(r.getResourceType())).count());
        resourceTypeCount.put("video", resources.stream().filter(r -> "video".equals(r.getResourceType())).count());
        resourceTypeCount.put("image", resources.stream().filter(r -> "image".equals(r.getResourceType())).count());
        statistics.setResourceTypeCount(resourceTypeCount);
        
        // TODO: 添加月度使用统计
        statistics.setMonthlyUsage(new HashMap<>());
        
        return statistics;
    }

    @Override
    @Transactional
    public BatchUploadResponse batchUploadResources(List<ResourceUploadRequest> uploadRequests, Long teacherId) {
        log.info("批量上传资源，教师ID: {}, 资源数量: {}", teacherId, uploadRequests.size());
        
        if (uploadRequests == null || uploadRequests.isEmpty()) {
            throw new RuntimeException("上传请求列表不能为空");
        }
        
        List<ResourceResponse> successList = new ArrayList<>();
        List<String> failureList = new ArrayList<>();
        
        for (ResourceUploadRequest request : uploadRequests) {
            try {
                ResourceResponse response = uploadResource(request, teacherId);
                successList.add(response);
            } catch (Exception e) {
                failureList.add("文件 " + request.getResourceName() + " 上传失败: " + e.getMessage());
                log.error("批量上传中单个文件失败", e);
            }
        }
        
        BatchUploadResponse batchResponse = new BatchUploadResponse();
        batchResponse.setTotalCount(uploadRequests.size());
        batchResponse.setSuccessCount(successList.size());
        batchResponse.setFailureCount(failureList.size());
        batchResponse.setSuccessfulUploads(successList);
        batchResponse.setFailedUploads(failureList);
        
        return batchResponse;
    }

    @Override
    @Transactional
    public FolderResponse createFolder(FolderCreateRequest folderRequest, Long teacherId) {
        log.info("创建文件夹，教师ID: {}, 文件夹名称: {}", teacherId, folderRequest.getFolderName());
        
        // 验证教师是否存在
        Teacher teacher = teacherMapper.selectById(teacherId);
        if (teacher == null) {
            throw new RuntimeException("教师不存在");
        }
        
        // 如果有父文件夹，验证父文件夹是否存在且属于当前教师
        if (folderRequest.getParentFolderId() != null) {
            Resource parentFolder = resourceMapper.selectById(folderRequest.getParentFolderId());
            if (parentFolder == null || parentFolder.getIsDeleted() || !parentFolder.getUploaderId().equals(teacherId)) {
                throw new RuntimeException("父文件夹不存在或无权限访问");
            }
        }
        
        // 创建文件夹资源
        Resource folder = new Resource();
        folder.setResourceName(folderRequest.getFolderName());
        folder.setResourceType("folder");
        folder.setDescription(folderRequest.getDescription());
        folder.setUploaderId(teacherId);
        folder.setParentId(folderRequest.getParentFolderId());
        folder.setIsPublic("public".equals(folderRequest.getVisibility()));
        folder.setCreateTime(LocalDateTime.now());
        folder.setUpdateTime(LocalDateTime.now());
        folder.setIsDeleted(false);
        
        resourceMapper.insert(folder);
        
        // 转换为响应对象
        FolderResponse response = new FolderResponse();
        response.setFolderId(folder.getId());
        response.setFolderName(folder.getResourceName());
        response.setDescription(folder.getDescription());
        response.setParentFolderId(folder.getParentId());
        response.setCreateTime(folder.getCreateTime());
        response.setUpdateTime(folder.getUpdateTime());
        
        return response;
    }

    @Override
    @Transactional
    public Boolean moveResource(Long resourceId, Long targetFolderId, Long teacherId) {
        log.info("移动资源，资源ID: {}, 目标文件夹ID: {}, 教师ID: {}", resourceId, targetFolderId, teacherId);
        
        // 查询要移动的资源
        Resource resource = resourceMapper.selectById(resourceId);
        if (resource == null || resource.getIsDeleted()) {
            throw new RuntimeException("资源不存在");
        }
        
        // 验证权限
        if (!resource.getUploaderId().equals(teacherId)) {
            throw new RuntimeException("无权限移动该资源");
        }
        
        // 如果目标文件夹不为空，验证目标文件夹
        if (targetFolderId != null) {
            Resource targetFolder = resourceMapper.selectById(targetFolderId);
            if (targetFolder == null || targetFolder.getIsDeleted()) {
                throw new RuntimeException("目标文件夹不存在");
            }
            
            if (!"folder".equals(targetFolder.getResourceType())) {
                throw new RuntimeException("目标不是文件夹");
            }
            
            if (!targetFolder.getUploaderId().equals(teacherId)) {
                throw new RuntimeException("无权限访问目标文件夹");
            }
        }
        
        // 移动资源
        resource.setParentId(targetFolderId);
        resource.setUpdateTime(LocalDateTime.now());
        resourceMapper.updateById(resource);
        
        // TODO: 实现资源的文件夹关联逻辑
        
        return true;
    }

    @Override
    public ResourceResponse copyResource(Long resourceId, Long targetFolderId, Long teacherId) {
        // TODO: 实现复制资源逻辑
        throw new UnsupportedOperationException("Method not implemented yet");
    }

    @Override
    public PageResponse<ResourceResponse> searchResources(String keyword, Long teacherId, PageRequest pageRequest) {
        // TODO: 实现搜索资源逻辑
        throw new UnsupportedOperationException("Method not implemented yet");
    }

    @Override
    public PageResponse<ResourceDTO.ResourceAccessRecordResponse> getResourceAccessRecords(Long resourceId, Long teacherId, PageRequest pageRequest) {
        // TODO: 实现获取资源访问记录逻辑
        throw new UnsupportedOperationException("Method not implemented yet");
    }

    @Override
    public Boolean setResourcePermissions(Long resourceId, ResourceDTO.ResourcePermissionRequest permissions, Long teacherId) {
        // TODO: 实现设置资源权限逻辑
        throw new UnsupportedOperationException("Method not implemented yet");
    }

    @Override
    public ResourceDTO.ResourcePermissionResponse getResourcePermissions(Long resourceId, Long teacherId) {
        // TODO: 实现获取资源权限逻辑
        throw new UnsupportedOperationException("Method not implemented yet");
    }

    @Override
    public Object previewResource(Long resourceId, Long teacherId) {
        // TODO: 实现预览资源逻辑
        throw new UnsupportedOperationException("Method not implemented yet");
    }

    @Override
    public List<ResourceDTO.ResourceVersionResponse> getResourceVersions(Long resourceId, Long teacherId) {
        // TODO: 实现获取资源版本历史逻辑
        throw new UnsupportedOperationException("Method not implemented yet");
    }

    @Override
    public ResourceDTO.ResourceVersionResponse uploadResourceVersion(Long resourceId, ResourceDTO.ResourceUploadRequest uploadRequest, Long teacherId) {
        // TODO: 实现上传资源新版本逻辑
        throw new UnsupportedOperationException("Method not implemented yet");
    }

    @Override
    public Boolean restoreResourceVersion(Long resourceId, Long versionId, Long teacherId) {
        // TODO: 实现恢复资源版本逻辑
        throw new UnsupportedOperationException("Method not implemented yet");
    }

    @Override
    public Boolean setResourceTags(Long resourceId, List<String> tags, Long teacherId) {
        // TODO: 实现设置资源标签逻辑
        throw new UnsupportedOperationException("Method not implemented yet");
    }

    @Override
    public List<String> getResourceTags(Long resourceId, Long teacherId) {
        // TODO: 实现获取资源标签逻辑
        throw new UnsupportedOperationException("Method not implemented yet");
    }

    @Override
    public PageResponse<ResourceDTO.ResourceResponse> getResourcesByTags(List<String> tags, Long teacherId, PageRequest pageRequest) {
        // TODO: 实现按标签筛选资源逻辑
        throw new UnsupportedOperationException("Method not implemented yet");
    }

    @Override
    public Object getResourceUsageStatistics(Long resourceId, Long teacherId, String timeRange) {
        // TODO: 实现获取资源使用统计逻辑
        throw new UnsupportedOperationException("Method not implemented yet");
    }

    @Override
    public Boolean favoriteResource(Long resourceId, Long teacherId) {
        // TODO: 实现收藏资源逻辑
        throw new UnsupportedOperationException("Method not implemented yet");
    }

    @Override
    public Boolean unfavoriteResource(Long resourceId, Long teacherId) {
        // TODO: 实现取消收藏资源逻辑
        throw new UnsupportedOperationException("Method not implemented yet");
    }

    @Override
    public PageResponse<ResourceDTO.ResourceResponse> getFavoriteResources(Long teacherId, PageRequest pageRequest) {
        // TODO: 实现获取收藏资源列表逻辑
        throw new UnsupportedOperationException("Method not implemented yet");
    }

    @Override
    public List<ResourceDTO.ResourceResponse> getRecentResources(Long teacherId, Integer limit) {
        // TODO: 实现获取最近访问资源逻辑
        throw new UnsupportedOperationException("Method not implemented yet");
    }

    @Override
    public List<ResourceDTO.ResourceResponse> getRecommendedResources(Long teacherId, Integer limit) {
        // TODO: 实现获取推荐资源逻辑
        throw new UnsupportedOperationException("Method not implemented yet");
    }

    @Override
    public ResourceDTO.ResourceResponse compressResources(List<Long> resourceIds, Long teacherId) {
        // TODO: 实现压缩资源逻辑
        throw new UnsupportedOperationException("Method not implemented yet");
    }

    @Override
    public List<ResourceDTO.ResourceResponse> extractResource(Long resourceId, Long targetFolderId, Long teacherId) {
        // TODO: 实现解压资源逻辑
        throw new UnsupportedOperationException("Method not implemented yet");
    }

    @Override
    public ResourceDTO.StorageUsageResponse getStorageUsage(Long teacherId) {
        // TODO: 实现获取存储空间使用情况逻辑
        throw new UnsupportedOperationException("Method not implemented yet");
    }

    @Override
    public Boolean cleanRecycleBin(Long teacherId) {
        // TODO: 实现清理回收站逻辑
        throw new UnsupportedOperationException("Method not implemented yet");
    }

    @Override
    public Boolean restoreFromRecycleBin(Long resourceId, Long teacherId) {
        // TODO: 实现从回收站恢复资源逻辑
        throw new UnsupportedOperationException("Method not implemented yet");
    }

    @Override
    public PageResponse<ResourceDTO.ResourceResponse> getRecycleBinResources(Long teacherId, PageRequest pageRequest) {
        // TODO: 实现获取回收站资源列表逻辑
        throw new UnsupportedOperationException("Method not implemented yet");
    }

    @Override
    public Boolean permanentDeleteResource(Long resourceId, Long teacherId) {
        // TODO: 实现永久删除资源逻辑
        throw new UnsupportedOperationException("Method not implemented yet");
    }
}