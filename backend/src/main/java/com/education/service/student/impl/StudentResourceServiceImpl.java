package com.education.service.student.impl;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.education.exception.BusinessException;
import com.education.dto.ResourceDTO;
import com.education.dto.common.PageRequest;
import com.education.dto.common.PageResponse;
import com.education.exception.ResultCode;
import com.education.entity.*;
import com.education.mapper.*;
import com.education.service.student.StudentResourceService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.List;
import java.util.stream.Collectors;

/**
 * 学生端资源服务实现类
 */
@Slf4j
@Service
public class StudentResourceServiceImpl implements StudentResourceService {

    @Autowired
    private ResourceMapper resourceMapper;
    
    @Autowired
    private StudentMapper studentMapper;
    
    @Autowired
    private CourseMapper courseMapper;
    
    @Autowired
    private ClassMapper classMapper;
    
    @Override
    public PageResponse<ResourceDTO.ResourceListResponse> getAccessibleResources(Long studentId, PageRequest pageRequest) {
        log.info("获取学生可访问的资源列表，学生ID: {}", studentId);
        
        Student student = studentMapper.selectById(studentId);
        if (student == null) {
            throw new BusinessException(ResultCode.USER_NOT_FOUND, "学生不存在");
        }
        
        // 构建分页查询
        Page<Resource> page = new Page<>(pageRequest.getPageNum(), pageRequest.getPageSize());
        
        // 查询学生可访问的资源（通过班级关联的课程）
        QueryWrapper<Resource> queryWrapper = new QueryWrapper<>();
        queryWrapper.exists("SELECT 1 FROM course c WHERE c.id = resource.course_id AND c.class_id = {0}", student.getClassId())
                   .eq("status", 1)
                   .orderByDesc("create_time");
        
        Page<Resource> resourcePage = resourceMapper.selectPage(page, queryWrapper);
        
        ResourceDTO.ResourceListResponse listResponse = convertToResourceListResponse(resourcePage.getRecords(), resourcePage);
        return PageResponse.of(List.of(listResponse));
    }

    @Override
    public PageResponse<ResourceDTO.ResourceListResponse> getCourseResources(Long courseId, Long studentId, PageRequest pageRequest) {
        log.info("获取课程资源列表，课程ID: {}, 学生ID: {}", courseId, studentId);
        
        // 验证权限
        if (!hasCourseAccess(courseId, studentId)) {
            throw new BusinessException(ResultCode.ACCESS_DENIED, "无权限访问该课程资源");
        }
        
        // 构建分页查询
        Page<Resource> page = new Page<>(pageRequest.getPageNum(), pageRequest.getPageSize());
        
        QueryWrapper<Resource> queryWrapper = new QueryWrapper<>();
        queryWrapper.eq("course_id", courseId)
                   .eq("status", 1)
                   .orderByDesc("create_time");
        
        Page<Resource> resourcePage = resourceMapper.selectPage(page, queryWrapper);
        
        ResourceDTO.ResourceListResponse listResponse = convertToResourceListResponse(resourcePage.getRecords(), resourcePage);
        return PageResponse.of(List.of(listResponse));
    }

    @Override
    public ResourceDTO.ResourceDetailResponse getResourceDetail(Long resourceId, Long studentId) {
        log.info("获取资源详情，资源ID: {}, 学生ID: {}", resourceId, studentId);
        
        Resource resource = resourceMapper.selectById(resourceId);
        if (resource == null) {
            throw new BusinessException(ResultCode.NOT_FOUND, "资源不存在");
        }
        
        // 验证权限
        if (!hasResourceAccess(resourceId, studentId)) {
            throw new BusinessException(ResultCode.ACCESS_DENIED, "无权限访问该资源");
        }
        
        ResourceDTO.ResourceDetailResponse response = new ResourceDTO.ResourceDetailResponse();
        response.setResourceId(resource.getId());
        response.setCourseId(resource.getCourseId());
        response.setTitle(resource.getResourceName());
        response.setDescription(resource.getDescription());
        response.setFileType(resource.getFileType());
        response.setFileSize(resource.getFileSize());
        response.setFilePath(resource.getFilePath());
        response.setDownloadCount(resource.getDownloadCount());
        response.setCreatedTime(resource.getCreateTime());
        
        return response;
    }

    @Override
    public ResourceDTO.ResourceDownloadResponse downloadResource(Long resourceId, Long studentId) {
        log.info("下载资源，资源ID: {}, 学生ID: {}", resourceId, studentId);
        
        Resource resource = resourceMapper.selectById(resourceId);
        if (resource == null) {
            throw new BusinessException(ResultCode.NOT_FOUND, "资源不存在");
        }
        
        // 验证权限
        if (!hasResourceAccess(resourceId, studentId)) {
            throw new BusinessException(ResultCode.ACCESS_DENIED, "无权限下载该资源");
        }
        
        // 增加下载次数
        resource.setDownloadCount(resource.getDownloadCount() + 1);
        resourceMapper.updateById(resource);
        
        ResourceDTO.ResourceDownloadResponse response = new ResourceDTO.ResourceDownloadResponse();
        response.setResourceId(resourceId);
        response.setFileName(resource.getResourceName());
        response.setFileType(resource.getFileType());
        response.setFileSize(resource.getFileSize());
        response.setDownloadUrl(resource.getFilePath());
        response.setDownloadTime(LocalDateTime.now());
        
        return response;
    }

    @Override
    public ResourceDTO.ResourcePreviewResponse previewResource(Long resourceId, Long studentId) {
        log.info("预览资源，资源ID: {}, 学生ID: {}", resourceId, studentId);
        
        Resource resource = resourceMapper.selectById(resourceId);
        if (resource == null) {
            throw new BusinessException(ResultCode.NOT_FOUND, "资源不存在");
        }
        
        // 验证权限
        if (!hasResourceAccess(resourceId, studentId)) {
            throw new BusinessException(ResultCode.ACCESS_DENIED, "无权限预览该资源");
        }
        
        ResourceDTO.ResourcePreviewResponse response = new ResourceDTO.ResourcePreviewResponse();
        response.setPreviewUrl(resource.getFilePath());
        response.setPreviewType(resource.getFileType());
        
        return response;
    }

    @Override
    public PageResponse<ResourceDTO.ResourceListResponse> searchResources(ResourceDTO.ResourceSearchRequest searchRequest, Long studentId) {
        log.info("搜索资源，学生ID: {}, 搜索条件: {}", studentId, searchRequest);
        
        Student student = studentMapper.selectById(studentId);
        if (student == null) {
            throw new BusinessException(ResultCode.USER_NOT_FOUND, "学生不存在");
        }
        
        // 构建分页查询
        Page<Resource> page = new Page<>(searchRequest.getPage(), searchRequest.getSize());
        
        // 搜索学生可访问的资源
        QueryWrapper<Resource> queryWrapper = new QueryWrapper<>();
        queryWrapper.exists("SELECT 1 FROM course c WHERE c.id = resource.course_id AND c.class_id = {0}", student.getClassId())
                   .eq("status", 1);
        
        if (searchRequest.getKeyword() != null && !searchRequest.getKeyword().trim().isEmpty()) {
            queryWrapper.and(wrapper -> wrapper.like("title", searchRequest.getKeyword())
                                              .or()
                                              .like("description", searchRequest.getKeyword()));
        }
        
        queryWrapper.orderByDesc("create_time");
        
        Page<Resource> resourcePage = resourceMapper.selectPage(page, queryWrapper);
        
        ResourceDTO.ResourceListResponse listResponse = convertToResourceListResponse(resourcePage.getRecords(), resourcePage);
        return PageResponse.of(List.of(listResponse));
    }

    @Override
    @Transactional
    public Boolean favoriteResource(Long resourceId, Long studentId) {
        log.info("收藏资源，资源ID: {}, 学生ID: {}", resourceId, studentId);
        
        // 验证权限
        if (!hasResourceAccess(resourceId, studentId)) {
            throw new BusinessException(ResultCode.ACCESS_DENIED, "无权限访问该资源");
        }
        
        // 这里可以实现收藏逻辑，比如在用户偏好表中添加记录
        // 暂时返回true表示操作成功
        return true;
    }

    @Override
    @Transactional
    public Boolean unfavoriteResource(Long resourceId, Long studentId) {
        log.info("取消收藏资源，资源ID: {}, 学生ID: {}", resourceId, studentId);
        
        // 验证资源是否存在
        Resource resource = resourceMapper.selectById(resourceId);
        if (resource == null) {
            throw new BusinessException(ResultCode.NOT_FOUND, "资源不存在");
        }
        
        // 这里可以实现取消收藏逻辑
        // 暂时返回true表示操作成功
        return true;
    }

    @Override
    public PageResponse<ResourceDTO.ResourceListResponse> getFavoriteResources(Long studentId, PageRequest pageRequest) {
        log.info("获取收藏的资源列表，学生ID: {}", studentId);
        
        Student student = studentMapper.selectById(studentId);
        if (student == null) {
            throw new BusinessException(ResultCode.USER_NOT_FOUND, "学生不存在");
        }
        
        // 这里可以实现获取收藏资源的逻辑
        // 暂时返回空列表
        ResourceDTO.ResourceListResponse emptyResponse = new ResourceDTO.ResourceListResponse();
        emptyResponse.setResources(List.of());
        emptyResponse.setTotalCount(0);
        emptyResponse.setCurrentPage(pageRequest.getPageNum());
        emptyResponse.setPageSize(pageRequest.getPageSize());
        emptyResponse.setTotalPages(0);
        return PageResponse.of(List.of(emptyResponse));
    }

    @Override
    public List<ResourceDTO.ResourceListResponse> getRecentResources(Long studentId, Integer limit) {
        log.info("获取最近访问的资源，学生ID: {}, 限制数量: {}", studentId, limit);
        
        Student student = studentMapper.selectById(studentId);
        if (student == null) {
            throw new BusinessException(ResultCode.USER_NOT_FOUND, "学生不存在");
        }
        
        // 查询最近访问的资源（按访问时间倒序）
        QueryWrapper<Resource> queryWrapper = new QueryWrapper<>();
        queryWrapper.exists("SELECT 1 FROM course c WHERE c.id = resource.course_id AND c.class_id = {0}", student.getClassId())
                   .eq("status", 1)
                   .orderByDesc("update_time")
                   .last("LIMIT " + limit);
        
        List<Resource> resources = resourceMapper.selectList(queryWrapper);
        
        // 为每个资源创建ResourceListResponse
        return resources.stream()
                .map(resource -> {
                    ResourceDTO.ResourceListResponse response = new ResourceDTO.ResourceListResponse();
                    response.setResources(List.of(convertToResourceResponse(resource)));
                    response.setTotalCount(1);
                    response.setCurrentPage(1);
                    response.setPageSize(1);
                    response.setTotalPages(1);
                    return response;
                })
                .collect(Collectors.toList());
    }

    @Override
    public PageResponse<ResourceDTO.ResourceListResponse> getRecommendedResources(Long studentId, PageRequest pageRequest) {
        log.info("获取推荐资源，学生ID: {}", studentId);
        
        Student student = studentMapper.selectById(studentId);
        if (student == null) {
            throw new BusinessException(ResultCode.USER_NOT_FOUND, "学生不存在");
        }
        
        // 构建分页查询
        Page<Resource> page = new Page<>(pageRequest.getPageNum(), pageRequest.getPageSize());
        
        // 推荐逻辑：按下载次数和创建时间排序
        QueryWrapper<Resource> queryWrapper = new QueryWrapper<>();
        queryWrapper.exists("SELECT 1 FROM course c WHERE c.id = resource.course_id AND c.class_id = {0}", student.getClassId())
                   .eq("status", 1)
                   .orderByDesc("download_count")
                   .orderByDesc("create_time");
        
        Page<Resource> resourcePage = resourceMapper.selectPage(page, queryWrapper);
        
        ResourceDTO.ResourceListResponse listResponse = convertToResourceListResponse(resourcePage.getRecords(), resourcePage);
        return PageResponse.of(List.of(listResponse));
    }

    @Override
    public PageResponse<ResourceDTO.ResourceListResponse> getPopularResources(Long studentId, PageRequest pageRequest) {
        log.info("获取热门资源，学生ID: {}", studentId);
        
        Student student = studentMapper.selectById(studentId);
        if (student == null) {
            throw new BusinessException(ResultCode.USER_NOT_FOUND, "学生不存在");
        }
        
        // 构建分页查询
        Page<Resource> page = new Page<>(pageRequest.getPageNum(), pageRequest.getPageSize());
        
        // 热门资源：按下载次数排序
        QueryWrapper<Resource> queryWrapper = new QueryWrapper<>();
        queryWrapper.exists("SELECT 1 FROM course c WHERE c.id = resource.course_id AND c.class_id = {0}", student.getClassId())
                   .eq("status", 1)
                   .gt("download_count", 0)
                   .orderByDesc("download_count");
        
        Page<Resource> resourcePage = resourceMapper.selectPage(page, queryWrapper);
        
        ResourceDTO.ResourceListResponse listResponse = convertToResourceListResponse(resourcePage.getRecords(), resourcePage);
        return PageResponse.of(List.of(listResponse));
    }

    @Override
    public PageResponse<ResourceDTO.ResourceListResponse> getResourcesByType(String resourceType, Long studentId, PageRequest pageRequest) {
        log.info("按类型获取资源，资源类型: {}, 学生ID: {}", resourceType, studentId);
        
        Student student = studentMapper.selectById(studentId);
        if (student == null) {
            throw new BusinessException(ResultCode.USER_NOT_FOUND, "学生不存在");
        }
        
        // 构建分页查询
        Page<Resource> page = new Page<>(pageRequest.getPageNum(), pageRequest.getPageSize());
        
        QueryWrapper<Resource> queryWrapper = new QueryWrapper<>();
        queryWrapper.exists("SELECT 1 FROM course c WHERE c.id = resource.course_id AND c.class_id = {0}", student.getClassId())
                   .eq("status", 1)
                   .eq("file_type", resourceType)
                   .orderByDesc("create_time");
        
        Page<Resource> resourcePage = resourceMapper.selectPage(page, queryWrapper);
        
        ResourceDTO.ResourceListResponse listResponse = convertToResourceListResponse(resourcePage.getRecords(), resourcePage);
        return PageResponse.of(List.of(listResponse));
    }

    @Override
    public PageResponse<ResourceDTO.ResourceListResponse> getResourcesByTags(List<String> tags, Long studentId, PageRequest pageRequest) {
        log.info("按标签获取资源，标签: {}, 学生ID: {}", tags, studentId);
        
        Student student = studentMapper.selectById(studentId);
        if (student == null) {
            throw new BusinessException(ResultCode.USER_NOT_FOUND, "学生不存在");
        }
        
        // 构建分页查询
        Page<Resource> page = new Page<>(pageRequest.getPageNum(), pageRequest.getPageSize());
        
        QueryWrapper<Resource> queryWrapper = new QueryWrapper<>();
        queryWrapper.exists("SELECT 1 FROM course c WHERE c.id = resource.course_id AND c.class_id = {0}", student.getClassId())
                   .eq("status", 1);
        
        // 如果有标签条件，添加标签筛选（假设tags字段存储为逗号分隔的字符串）
        if (tags != null && !tags.isEmpty()) {
            for (String tag : tags) {
                queryWrapper.like("tags", tag);
            }
        }
        
        queryWrapper.orderByDesc("create_time");
        
        Page<Resource> resourcePage = resourceMapper.selectPage(page, queryWrapper);
        
        ResourceDTO.ResourceListResponse listResponse = convertToResourceListResponse(resourcePage.getRecords(), resourcePage);
        return PageResponse.of(List.of(listResponse));
    }

    @Override
    public PageResponse<ResourceDTO.ResourceAccessRecordResponse> getAccessRecords(Long studentId, PageRequest pageRequest) {
        log.info("获取资源访问记录，学生ID: {}", studentId);
        
        Student student = studentMapper.selectById(studentId);
        if (student == null) {
            throw new BusinessException(ResultCode.USER_NOT_FOUND, "学生不存在");
        }
        
        // 暂时返回空列表，实际需要访问记录表
        return PageResponse.of((long)pageRequest.getPageNum(), (long)pageRequest.getPageSize(), 0L, List.of());
    }





    @Override
    public Boolean recordAccess(ResourceDTO.ResourceAccessRequest accessRequest, Long studentId) {
        // TODO: 实现记录资源访问逻辑
        throw new UnsupportedOperationException("Method not implemented yet");
    }

    @Override
    public Boolean evaluateResource(ResourceDTO.ResourceEvaluationRequest evaluationRequest, Long studentId) {
        // TODO: 实现评价资源逻辑
        throw new UnsupportedOperationException("Method not implemented yet");
    }

    @Override
    public PageResponse<ResourceDTO.ResourceEvaluationResponse> getResourceEvaluations(Long resourceId, Long studentId, PageRequest pageRequest) {
        // TODO: 实现获取资源评价逻辑
        throw new UnsupportedOperationException("Method not implemented yet");
    }

    @Override
    public ResourceDTO.ResourceEvaluationResponse getMyResourceEvaluation(Long resourceId, Long studentId) {
        // TODO: 实现获取我的资源评价逻辑
        throw new UnsupportedOperationException("Method not implemented yet");
    }

    @Override
    public ResourceDTO.ResourceShareResponse shareResource(ResourceDTO.ResourceShareRequest shareRequest, Long studentId) {
        // TODO: 实现分享资源逻辑
        throw new UnsupportedOperationException("Method not implemented yet");
    }

    @Override
    public PageResponse<ResourceDTO.ResourceShareRecordResponse> getShareRecords(Long studentId, PageRequest pageRequest) {
        // TODO: 实现获取分享记录逻辑
        throw new UnsupportedOperationException("Method not implemented yet");
    }

    @Override
    public Boolean cancelShare(Long shareId, Long studentId) {
        // TODO: 实现取消资源分享逻辑
        throw new UnsupportedOperationException("Method not implemented yet");
    }

    @Override
    public ResourceDTO.SharedResourceResponse accessSharedResource(String shareToken, Long studentId) {
        // TODO: 实现通过分享链接访问资源逻辑
        throw new UnsupportedOperationException("Method not implemented yet");
    }

    @Override
    public PageResponse<ResourceDTO.ResourceNoteResponse> getResourceNotes(Long resourceId, Long studentId, PageRequest pageRequest) {
        // TODO: 实现获取资源笔记逻辑
        throw new UnsupportedOperationException("Method not implemented yet");
    }

    @Override
    public Long createResourceNote(ResourceDTO.ResourceNoteCreateRequest noteRequest, Long studentId) {
        // TODO: 实现创建资源笔记逻辑
        throw new UnsupportedOperationException("Method not implemented yet");
    }

    @Override
    public Boolean updateResourceNote(Long noteId, ResourceDTO.ResourceNoteUpdateRequest noteRequest, Long studentId) {
        // TODO: 实现更新资源笔记逻辑
        throw new UnsupportedOperationException("Method not implemented yet");
    }

    @Override
    public Boolean deleteResourceNote(Long noteId, Long studentId) {
        // TODO: 实现删除资源笔记逻辑
        throw new UnsupportedOperationException("Method not implemented yet");
    }

    @Override
    public PageResponse<ResourceDTO.ResourceDiscussionResponse> getResourceDiscussions(Long resourceId, Long studentId, PageRequest pageRequest) {
        // TODO: 实现获取资源讨论逻辑
        throw new UnsupportedOperationException("Method not implemented yet");
    }

    @Override
    public Long createResourceDiscussion(ResourceDTO.ResourceDiscussionCreateRequest discussionRequest, Long studentId) {
        // TODO: 实现创建资源讨论逻辑
        throw new UnsupportedOperationException("Method not implemented yet");
    }

    @Override
    public Long replyResourceDiscussion(Long discussionId, ResourceDTO.ResourceDiscussionReplyRequest replyRequest, Long studentId) {
        // TODO: 实现回复资源讨论逻辑
        throw new UnsupportedOperationException("Method not implemented yet");
    }

    @Override
    public List<ResourceDTO.ResourceVersionResponse> getResourceVersions(Long resourceId, Long studentId) {
        // TODO: 实现获取资源版本历史逻辑
        throw new UnsupportedOperationException("Method not implemented yet");
    }

    @Override
    public ResourceDTO.ResourceDetailResponse getResourceVersion(Long resourceId, String version, Long studentId) {
        // TODO: 实现获取指定版本资源逻辑
        throw new UnsupportedOperationException("Method not implemented yet");
    }

    @Override
    public ResourceDTO.ResourceStatisticsResponse getResourceStatistics(Long studentId) {
        // TODO: 实现获取资源统计信息逻辑
        throw new UnsupportedOperationException("Method not implemented yet");
    }

    @Override
    public List<ResourceDTO.LearningResourceRecommendationResponse> getLearningResourceRecommendations(Long studentId, Long courseId) {
        // TODO: 实现获取学习资源推荐逻辑
        throw new UnsupportedOperationException("Method not implemented yet");
    }

    @Override
    public ResourceDTO.ResourceLearningProgressResponse getResourceLearningProgress(Long resourceId, Long studentId) {
        // TODO: 实现获取资源学习进度逻辑
        throw new UnsupportedOperationException("Method not implemented yet");
    }

    @Override
    public Boolean updateResourceLearningProgress(ResourceDTO.ResourceProgressUpdateRequest progressRequest, Long studentId) {
        // TODO: 实现更新资源学习进度逻辑
        throw new UnsupportedOperationException("Method not implemented yet");
    }

    @Override
    public List<ResourceDTO.ResourceTagResponse> getAvailableTags(Long studentId) {
        // TODO: 实现获取资源标签逻辑
        throw new UnsupportedOperationException("Method not implemented yet");
    }

    @Override
    public List<ResourceDTO.ResourceTypeResponse> getAvailableResourceTypes(Long studentId) {
        // TODO: 实现获取资源类型逻辑
        throw new UnsupportedOperationException("Method not implemented yet");
    }

    @Override
    public ResourceDTO.ResourceUsageReportResponse getResourceUsageReport(Long studentId, String timeRange) {
        // TODO: 实现获取资源使用报告逻辑
        throw new UnsupportedOperationException("Method not implemented yet");
    }

    @Override
    public ResourceDTO.ExportResponse exportResourceData(Long studentId, ResourceDTO.ResourceDataExportRequest exportRequest) {
        // TODO: 实现导出资源数据逻辑
        throw new UnsupportedOperationException("Method not implemented yet");
    }

    @Override
    public PageResponse<ResourceDTO.OfflineResourceResponse> getOfflineResources(Long studentId, PageRequest pageRequest) {
        // TODO: 实现获取离线资源逻辑
        throw new UnsupportedOperationException("Method not implemented yet");
    }

    @Override
    public ResourceDTO.OfflinePackageResponse downloadOfflinePackage(ResourceDTO.OfflinePackageRequest packageRequest, Long studentId) {
        // TODO: 实现下载离线资源包逻辑
        throw new UnsupportedOperationException("Method not implemented yet");
    }

    @Override
    public ResourceDTO.ResourceSyncResponse syncOfflineResources(ResourceDTO.ResourceSyncRequest syncRequest, Long studentId) {
        // TODO: 实现同步离线资源逻辑
        throw new UnsupportedOperationException("Method not implemented yet");
    }
    
    // 辅助方法：验证学生是否有权限访问课程
    private boolean hasCourseAccess(Long courseId, Long studentId) {
        Student student = studentMapper.selectById(studentId);
        if (student == null) {
            return false;
        }
        
        Course course = courseMapper.selectById(courseId);
        if (course == null) {
            return false;
        }
        
        // 通过班级关联验证权限
        return course.getClassId().equals(student.getClassId());
    }
    
    // 辅助方法：验证学生是否有权限访问资源
    private boolean hasResourceAccess(Long resourceId, Long studentId) {
        Student student = studentMapper.selectById(studentId);
        if (student == null) {
            return false;
        }
        
        Resource resource = resourceMapper.selectById(resourceId);
        if (resource == null) {
            return false;
        }
        
        // 通过课程和班级关联验证权限
        if (resource.getCourseId() != null) {
            QueryWrapper<Course> courseQuery = new QueryWrapper<>();
            courseQuery.eq("id", resource.getCourseId())
                      .eq("class_id", student.getClassId());
            
            return courseMapper.selectCount(courseQuery) > 0;
        }
        
        // 如果资源没有关联课程，默认允许访问
        return true;
    }
    
    // 辅助方法：转换资源为响应对象
    private ResourceDTO.ResourceResponse convertToResourceResponse(Resource resource) {
        ResourceDTO.ResourceResponse response = new ResourceDTO.ResourceResponse();
        response.setResourceId(resource.getId());
        response.setResourceName(resource.getResourceName());
        response.setResourceType(resource.getResourceType());
        response.setDescription(resource.getDescription());
        response.setFileName(resource.getResourceName());
        response.setFileSize(resource.getFileSize());
        response.setDownloadCount(resource.getDownloadCount());
        response.setCreateTime(resource.getCreateTime());
        return response;
    }
    
    // 辅助方法：转换资源列表为ResourceListResponse
    private ResourceDTO.ResourceListResponse convertToResourceListResponse(List<Resource> resources, Page<Resource> page) {
        ResourceDTO.ResourceListResponse response = new ResourceDTO.ResourceListResponse();
        
        List<ResourceDTO.ResourceResponse> resourceResponses = resources.stream()
                .map(this::convertToResourceResponse)
                .collect(Collectors.toList());
        
        response.setResources(resourceResponses);
        response.setTotalCount((int) page.getTotal());
        response.setCurrentPage((int) page.getCurrent());
        response.setPageSize((int) page.getSize());
        response.setTotalPages((int) page.getPages());
        
        return response;
    }
}