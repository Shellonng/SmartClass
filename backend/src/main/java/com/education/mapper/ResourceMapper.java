package com.education.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.education.entity.Resource;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

import java.util.List;
import java.util.Map;

/**
 * 资源数据访问层
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
@Mapper
public interface ResourceMapper extends BaseMapper<Resource> {
    
    /**
     * 根据课程ID查询资源列表
     */
    List<Resource> selectByCourseId(@Param("courseId") Long courseId);
    
    /**
     * 根据创建者ID查询资源列表
     */
    List<Resource> selectByCreatedBy(@Param("createdBy") Long createdBy);
    
    /**
     * 根据资源类型查询资源列表
     */
    List<Resource> selectByResourceType(@Param("resourceType") String resourceType);
    
    /**
     * 搜索资源
     */
    List<Resource> searchResources(@Param("keyword") String keyword, @Param("resourceType") String resourceType, @Param("courseId") Long courseId);
    
    /**
     * 查询热门资源
     */
    List<Resource> selectPopularResources(@Param("limit") Integer limit);
    
    /**
     * 查询最新资源
     */
    List<Resource> selectLatestResources(@Param("limit") Integer limit);
    
    /**
     * 查询推荐资源
     */
    List<Resource> selectRecommendedResources(@Param("userId") Long userId, @Param("limit") Integer limit);
    
    /**
     * 查询用户收藏的资源
     */
    List<Resource> selectFavoriteResources(@Param("userId") Long userId);
    
    /**
     * 查询用户最近访问的资源
     */
    List<Resource> selectRecentlyAccessedResources(@Param("userId") Long userId, @Param("limit") Integer limit);
    
    /**
     * 根据标签查询资源
     */
    List<Resource> selectByTags(@Param("tags") List<String> tags);
    
    /**
     * 查询资源统计信息
     */
    Map<String, Object> selectResourceStats(@Param("resourceId") Long resourceId);
    
    /**
     * 查询资源使用统计
     */
    List<Map<String, Object>> selectResourceUsageStats(@Param("createdBy") Long createdBy);
    
    /**
     * 更新资源下载次数
     */
    int updateDownloadCount(@Param("resourceId") Long resourceId);
    
    /**
     * 更新资源访问次数
     */
    int updateViewCount(@Param("resourceId") Long resourceId);
    
    /**
     * 查询存储空间使用情况
     */
    Map<String, Object> selectStorageUsage(@Param("userId") Long userId);
    
    /**
     * 查询回收站资源
     */
    List<Resource> selectDeletedResources(@Param("userId") Long userId);
    
    /**
     * 物理删除资源
     */
    int physicalDelete(@Param("resourceId") Long resourceId);
    
    /**
     * 从回收站恢复资源
     */
    int restoreFromTrash(@Param("resourceId") Long resourceId);
    
    /**
     * 批量删除资源
     */
    int batchDelete(@Param("resourceIds") List<Long> resourceIds);
    
    /**
     * 批量恢复资源
     */
    int batchRestore(@Param("resourceIds") List<Long> resourceIds);
}