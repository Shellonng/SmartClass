package com.education.dto;

import lombok.Data;

import java.util.List;
import java.util.Map;

@Data
public class LearningStatisticsDTO {
    // 日期范围内每天的学习时长数据
    private List<Map<String, Object>> dailyDurations;
    
    // 章节学习时长分布
    private List<Map<String, Object>> sectionDistribution;
    
    // 资源类型学习时长分布
    private List<Map<String, Object>> resourceTypeDistribution;
    
    // 学习统计摘要
    private Integer totalLearningDays;  // 总学习天数
    private Integer totalDuration;      // 总学习时长（秒）
    private Integer avgDailyDuration;   // 平均每日学习时长（秒）
    private Integer completedSections;  // 已完成章节数
    private Integer totalSections;      // 总章节数
    private Integer viewedResources;    // 已查看资源数
} 