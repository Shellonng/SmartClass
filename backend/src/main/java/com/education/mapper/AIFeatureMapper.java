package com.education.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.education.entity.AIFeature;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

import java.util.List;
import java.util.Map;

/**
 * AI功能数据访问层
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
@Mapper
public interface AIFeatureMapper extends BaseMapper<AIFeature> {
    
    /**
     * 根据用户ID查询AI功能记录
     */
    List<AIFeature> selectByUserId(@Param("userId") Long userId);
    
    /**
     * 根据功能类型查询AI功能记录
     */
    List<AIFeature> selectByFeatureType(@Param("featureType") String featureType);
    
    /**
     * 根据用户类型查询AI功能记录
     */
    List<AIFeature> selectByUserType(@Param("userType") String userType);
    
    /**
     * 根据会话ID查询AI功能记录
     */
    List<AIFeature> selectBySessionId(@Param("sessionId") String sessionId);
    
    /**
     * 根据课程ID查询AI功能记录
     */
    List<AIFeature> selectByCourseId(@Param("courseId") Long courseId);
    
    /**
     * 根据任务ID查询AI功能记录
     */
    List<AIFeature> selectByTaskId(@Param("taskId") Long taskId);
    
    /**
     * 查询用户收藏的AI功能记录
     */
    List<AIFeature> selectFavoritesByUserId(@Param("userId") Long userId);
    
    /**
     * 查询AI功能使用统计
     */
    List<Map<String, Object>> selectUsageStats(@Param("userId") Long userId, @Param("featureType") String featureType);
    
    /**
     * 查询AI功能成功率统计
     */
    Map<String, Object> selectSuccessRateStats(@Param("featureType") String featureType);
    
    /**
     * 查询AI功能平均处理时间
     */
    Map<String, Object> selectAverageProcessingTime(@Param("featureType") String featureType);
    
    /**
     * 查询用户AI功能使用频率
     */
    List<Map<String, Object>> selectUserUsageFrequency(@Param("userId") Long userId);
    
    /**
     * 查询热门AI功能
     */
    List<Map<String, Object>> selectPopularFeatures(@Param("limit") Integer limit);
    
    /**
     * 查询AI功能错误统计
     */
    List<Map<String, Object>> selectErrorStats(@Param("featureType") String featureType);
    
    /**
     * 查询用户反馈统计
     */
    Map<String, Object> selectFeedbackStats(@Param("featureType") String featureType);
    
    /**
     * 查询模型版本使用统计
     */
    List<Map<String, Object>> selectModelVersionStats();
    
    /**
     * 批量插入AI功能记录
     */
    int batchInsert(@Param("features") List<AIFeature> features);
    
    /**
     * 批量更新AI功能记录状态
     */
    int batchUpdateStatus(@Param("ids") List<Long> ids, @Param("status") String status);
    
    /**
     * 清理过期的AI功能记录
     */
    int cleanExpiredRecords(@Param("days") Integer days);
    
    /**
     * 导出AI功能使用数据
     */
    List<Map<String, Object>> exportUsageData(@Param("userId") Long userId, @Param("startTime") String startTime, @Param("endTime") String endTime);
}