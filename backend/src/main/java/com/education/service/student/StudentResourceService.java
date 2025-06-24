package com.education.service.student;

import com.education.dto.ResourceDTO;
import com.education.dto.common.PageRequest;
import com.education.dto.common.PageResponse;

import java.util.List;

/**
 * 学生端资源服务接口
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
public interface StudentResourceService {

    /**
     * 获取学生可访问的资源列表
     * 
     * @param studentId 学生ID
     * @param pageRequest 分页请求
     * @return 资源列表
     */
    PageResponse<ResourceDTO.ResourceListResponse> getAccessibleResources(Long studentId, PageRequest pageRequest);

    /**
     * 获取课程资源列表
     * 
     * @param courseId 课程ID
     * @param studentId 学生ID
     * @param pageRequest 分页请求
     * @return 课程资源列表
     */
    PageResponse<ResourceDTO.ResourceListResponse> getCourseResources(Long courseId, Long studentId, PageRequest pageRequest);

    /**
     * 获取资源详情
     * 
     * @param resourceId 资源ID
     * @param studentId 学生ID
     * @return 资源详情
     */
    ResourceDTO.ResourceDetailResponse getResourceDetail(Long resourceId, Long studentId);

    /**
     * 下载资源
     * 
     * @param resourceId 资源ID
     * @param studentId 学生ID
     * @return 下载信息
     */
    ResourceDTO.ResourceDownloadResponse downloadResource(Long resourceId, Long studentId);

    /**
     * 预览资源
     * 
     * @param resourceId 资源ID
     * @param studentId 学生ID
     * @return 预览信息
     */
    ResourceDTO.ResourcePreviewResponse previewResource(Long resourceId, Long studentId);

    /**
     * 搜索资源
     * 
     * @param searchRequest 搜索请求
     * @param studentId 学生ID
     * @return 搜索结果
     */
    PageResponse<ResourceDTO.ResourceListResponse> searchResources(ResourceDTO.ResourceSearchRequest searchRequest, Long studentId);

    /**
     * 收藏资源
     * 
     * @param resourceId 资源ID
     * @param studentId 学生ID
     * @return 操作结果
     */
    Boolean favoriteResource(Long resourceId, Long studentId);

    /**
     * 取消收藏资源
     * 
     * @param resourceId 资源ID
     * @param studentId 学生ID
     * @return 操作结果
     */
    Boolean unfavoriteResource(Long resourceId, Long studentId);

    /**
     * 获取收藏的资源列表
     * 
     * @param studentId 学生ID
     * @param pageRequest 分页请求
     * @return 收藏资源列表
     */
    PageResponse<ResourceDTO.ResourceListResponse> getFavoriteResources(Long studentId, PageRequest pageRequest);

    /**
     * 获取最近访问的资源
     * 
     * @param studentId 学生ID
     * @param limit 限制数量
     * @return 最近访问资源列表
     */
    List<ResourceDTO.ResourceListResponse> getRecentResources(Long studentId, Integer limit);

    /**
     * 获取推荐资源
     * 
     * @param studentId 学生ID
     * @param pageRequest 分页请求
     * @return 推荐资源列表
     */
    PageResponse<ResourceDTO.ResourceListResponse> getRecommendedResources(Long studentId, PageRequest pageRequest);

    /**
     * 获取热门资源
     * 
     * @param studentId 学生ID
     * @param pageRequest 分页请求
     * @return 热门资源列表
     */
    PageResponse<ResourceDTO.ResourceListResponse> getPopularResources(Long studentId, PageRequest pageRequest);

    /**
     * 按类型获取资源
     * 
     * @param resourceType 资源类型
     * @param studentId 学生ID
     * @param pageRequest 分页请求
     * @return 指定类型资源列表
     */
    PageResponse<ResourceDTO.ResourceListResponse> getResourcesByType(String resourceType, Long studentId, PageRequest pageRequest);

    /**
     * 按标签获取资源
     * 
     * @param tags 标签列表
     * @param studentId 学生ID
     * @param pageRequest 分页请求
     * @return 指定标签资源列表
     */
    PageResponse<ResourceDTO.ResourceListResponse> getResourcesByTags(List<String> tags, Long studentId, PageRequest pageRequest);

    /**
     * 获取资源访问记录
     * 
     * @param studentId 学生ID
     * @param pageRequest 分页请求
     * @return 访问记录列表
     */
    PageResponse<ResourceDTO.ResourceAccessRecordResponse> getAccessRecords(Long studentId, PageRequest pageRequest);

    /**
     * 记录资源访问
     * 
     * @param accessRequest 访问记录请求
     * @param studentId 学生ID
     * @return 操作结果
     */
    Boolean recordAccess(ResourceDTO.ResourceAccessRequest accessRequest, Long studentId);

    /**
     * 评价资源
     * 
     * @param evaluationRequest 评价请求
     * @param studentId 学生ID
     * @return 操作结果
     */
    Boolean evaluateResource(ResourceDTO.ResourceEvaluationRequest evaluationRequest, Long studentId);

    /**
     * 获取资源评价
     * 
     * @param resourceId 资源ID
     * @param studentId 学生ID
     * @param pageRequest 分页请求
     * @return 评价列表
     */
    PageResponse<ResourceDTO.ResourceEvaluationResponse> getResourceEvaluations(Long resourceId, Long studentId, PageRequest pageRequest);

    /**
     * 获取我的资源评价
     * 
     * @param resourceId 资源ID
     * @param studentId 学生ID
     * @return 我的评价
     */
    ResourceDTO.ResourceEvaluationResponse getMyResourceEvaluation(Long resourceId, Long studentId);

    /**
     * 分享资源
     * 
     * @param shareRequest 分享请求
     * @param studentId 学生ID
     * @return 分享链接
     */
    ResourceDTO.ResourceShareResponse shareResource(ResourceDTO.ResourceShareRequest shareRequest, Long studentId);

    /**
     * 获取分享记录
     * 
     * @param studentId 学生ID
     * @param pageRequest 分页请求
     * @return 分享记录列表
     */
    PageResponse<ResourceDTO.ResourceShareRecordResponse> getShareRecords(Long studentId, PageRequest pageRequest);

    /**
     * 取消资源分享
     * 
     * @param shareId 分享ID
     * @param studentId 学生ID
     * @return 操作结果
     */
    Boolean cancelShare(Long shareId, Long studentId);

    /**
     * 通过分享链接访问资源
     * 
     * @param shareToken 分享令牌
     * @param studentId 学生ID
     * @return 资源信息
     */
    ResourceDTO.SharedResourceResponse accessSharedResource(String shareToken, Long studentId);

    /**
     * 获取资源笔记
     * 
     * @param resourceId 资源ID
     * @param studentId 学生ID
     * @param pageRequest 分页请求
     * @return 笔记列表
     */
    PageResponse<ResourceDTO.ResourceNoteResponse> getResourceNotes(Long resourceId, Long studentId, PageRequest pageRequest);

    /**
     * 创建资源笔记
     * 
     * @param noteRequest 笔记创建请求
     * @param studentId 学生ID
     * @return 笔记ID
     */
    Long createResourceNote(ResourceDTO.ResourceNoteCreateRequest noteRequest, Long studentId);

    /**
     * 更新资源笔记
     * 
     * @param noteId 笔记ID
     * @param noteRequest 笔记更新请求
     * @param studentId 学生ID
     * @return 操作结果
     */
    Boolean updateResourceNote(Long noteId, ResourceDTO.ResourceNoteUpdateRequest noteRequest, Long studentId);

    /**
     * 删除资源笔记
     * 
     * @param noteId 笔记ID
     * @param studentId 学生ID
     * @return 操作结果
     */
    Boolean deleteResourceNote(Long noteId, Long studentId);

    /**
     * 获取资源讨论
     * 
     * @param resourceId 资源ID
     * @param studentId 学生ID
     * @param pageRequest 分页请求
     * @return 讨论列表
     */
    PageResponse<ResourceDTO.ResourceDiscussionResponse> getResourceDiscussions(Long resourceId, Long studentId, PageRequest pageRequest);

    /**
     * 创建资源讨论
     * 
     * @param discussionRequest 讨论创建请求
     * @param studentId 学生ID
     * @return 讨论ID
     */
    Long createResourceDiscussion(ResourceDTO.ResourceDiscussionCreateRequest discussionRequest, Long studentId);

    /**
     * 回复资源讨论
     * 
     * @param discussionId 讨论ID
     * @param replyRequest 回复请求
     * @param studentId 学生ID
     * @return 回复ID
     */
    Long replyResourceDiscussion(Long discussionId, ResourceDTO.ResourceDiscussionReplyRequest replyRequest, Long studentId);

    /**
     * 获取资源版本历史
     * 
     * @param resourceId 资源ID
     * @param studentId 学生ID
     * @return 版本历史列表
     */
    List<ResourceDTO.ResourceVersionResponse> getResourceVersions(Long resourceId, Long studentId);

    /**
     * 获取指定版本资源
     * 
     * @param resourceId 资源ID
     * @param version 版本号
     * @param studentId 学生ID
     * @return 指定版本资源信息
     */
    ResourceDTO.ResourceDetailResponse getResourceVersion(Long resourceId, String version, Long studentId);

    /**
     * 获取资源统计信息
     * 
     * @param studentId 学生ID
     * @return 资源统计
     */
    ResourceDTO.ResourceStatisticsResponse getResourceStatistics(Long studentId);

    /**
     * 获取学习资源推荐
     * 
     * @param studentId 学生ID
     * @param courseId 课程ID（可选）
     * @return 学习资源推荐
     */
    List<ResourceDTO.LearningResourceRecommendationResponse> getLearningResourceRecommendations(Long studentId, Long courseId);

    /**
     * 获取资源学习进度
     * 
     * @param resourceId 资源ID
     * @param studentId 学生ID
     * @return 学习进度
     */
    ResourceDTO.ResourceLearningProgressResponse getResourceLearningProgress(Long resourceId, Long studentId);

    /**
     * 更新资源学习进度
     * 
     * @param progressRequest 进度更新请求
     * @param studentId 学生ID
     * @return 操作结果
     */
    Boolean updateResourceLearningProgress(ResourceDTO.ResourceProgressUpdateRequest progressRequest, Long studentId);

    /**
     * 获取资源标签
     * 
     * @param studentId 学生ID
     * @return 可用标签列表
     */
    List<ResourceDTO.ResourceTagResponse> getAvailableTags(Long studentId);

    /**
     * 获取资源类型
     * 
     * @param studentId 学生ID
     * @return 可用资源类型列表
     */
    List<ResourceDTO.ResourceTypeResponse> getAvailableResourceTypes(Long studentId);

    /**
     * 获取资源使用报告
     * 
     * @param studentId 学生ID
     * @param timeRange 时间范围
     * @return 使用报告
     */
    ResourceDTO.ResourceUsageReportResponse getResourceUsageReport(Long studentId, String timeRange);

    /**
     * 导出资源数据
     * 
     * @param studentId 学生ID
     * @param exportRequest 导出请求
     * @return 导出文件信息
     */
    ResourceDTO.ExportResponse exportResourceData(Long studentId, ResourceDTO.ResourceDataExportRequest exportRequest);

    /**
     * 获取离线资源
     * 
     * @param studentId 学生ID
     * @param pageRequest 分页请求
     * @return 离线资源列表
     */
    PageResponse<ResourceDTO.OfflineResourceResponse> getOfflineResources(Long studentId, PageRequest pageRequest);

    /**
     * 下载离线资源包
     * 
     * @param packageRequest 离线包请求
     * @param studentId 学生ID
     * @return 下载信息
     */
    ResourceDTO.OfflinePackageResponse downloadOfflinePackage(ResourceDTO.OfflinePackageRequest packageRequest, Long studentId);

    /**
     * 同步离线资源
     * 
     * @param syncRequest 同步请求
     * @param studentId 学生ID
     * @return 同步结果
     */
    ResourceDTO.ResourceSyncResponse syncOfflineResources(ResourceDTO.ResourceSyncRequest syncRequest, Long studentId);
}