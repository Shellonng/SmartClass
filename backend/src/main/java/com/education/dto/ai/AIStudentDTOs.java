package com.education.dto.ai;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

/**
 * 学生端AI学习功能DTO类集合
 */
public class AIStudentDTOs {

    @Data
    @Schema(description = "AI个性化推荐响应")
    public static class AIPersonalRecommendationResponse {
        @Schema(description = "推荐类型")
        private String recommendationType;
        
        @Schema(description = "推荐内容列表")
        private List<RecommendationItem> recommendations;
        
        @Schema(description = "推荐理由")
        private String reason;
        
        @Schema(description = "推荐生成时间")
        private LocalDateTime generatedTime;
        
        @Data
        @Schema(description = "推荐项目")
        public static class RecommendationItem {
            @Schema(description = "内容ID")
            private Long contentId;
            
            @Schema(description = "内容标题")
            private String title;
            
            @Schema(description = "内容类型")
            private String contentType;
            
            @Schema(description = "推荐分数")
            private Double score;
            
            @Schema(description = "预计学习时长(分钟)")
            private Integer estimatedMinutes;
        }
    }

    @Data
    @Schema(description = "AI问题请求")
    public static class AIQuestionRequest {
        @Schema(description = "问题内容", required = true)
        @NotBlank(message = "问题内容不能为空")
        private String question;
        
        @Schema(description = "课程ID")
        private Long courseId;
        
        @Schema(description = "章节ID")
        private Long chapterId;
        
        @Schema(description = "问题类型")
        private String questionType;
    }

    @Data
    @Schema(description = "AI问题回答响应")
    public static class AIQuestionAnswerResponse {
        @Schema(description = "回答内容")
        private String answer;
        
        @Schema(description = "相关资源")
        private List<RelatedResource> relatedResources;
        
        @Schema(description = "置信度")
        private Double confidence;
        
        @Schema(description = "回答类型")
        private String answerType;
        
        @Data
        @Schema(description = "相关资源")
        public static class RelatedResource {
            @Schema(description = "资源ID")
            private Long resourceId;
            
            @Schema(description = "资源标题")
            private String title;
            
            @Schema(description = "资源类型")
            private String type;
            
            @Schema(description = "相关度分数")
            private Double relevanceScore;
        }
    }

    @Data
    @Schema(description = "AI学生能力分析响应")
    public static class AIStudentAbilityResponse {
        @Schema(description = "综合能力分数")
        private Double overallScore;
        
        @Schema(description = "各科目能力分析")
        private Map<String, SubjectAbility> subjectAbilities;
        
        @Schema(description = "学习建议")
        private List<String> suggestions;
        
        @Schema(description = "分析生成时间")
        private LocalDateTime analysisTime;
        
        @Data
        @Schema(description = "科目能力")
        public static class SubjectAbility {
            @Schema(description = "科目名称")
            private String subjectName;
            
            @Schema(description = "能力分数")
            private Double score;
            
            @Schema(description = "掌握的知识点")
            private List<String> masteredTopics;
            
            @Schema(description = "薄弱的知识点")
            private List<String> weakTopics;
            
            @Schema(description = "学习进度")
            private Double progress;
        }
    }

    @Data
    @Schema(description = "AI学习计划请求")
    public static class AIStudyPlanRequest {
        @Schema(description = "目标课程ID", required = true)
        @NotNull(message = "课程ID不能为空")
        private Long courseId;
        
        @Schema(description = "计划周期(天)")
        private Integer planDays;
        
        @Schema(description = "每日学习时长(分钟)")
        private Integer dailyMinutes;
        
        @Schema(description = "学习目标")
        private String goal;
        
        @Schema(description = "当前水平")
        private String currentLevel;
    }

    @Data
    @Schema(description = "AI学习计划响应")
    public static class AIStudyPlanResponse {
        @Schema(description = "计划ID")
        private String planId;
        
        @Schema(description = "计划标题")
        private String title;
        
        @Schema(description = "总计划周期")
        private Integer totalDays;
        
        @Schema(description = "日计划列表")
        private List<DailyPlan> dailyPlans;
        
        @Schema(description = "计划创建时间")
        private LocalDateTime createdTime;
        
        @Data
        @Schema(description = "日计划")
        public static class DailyPlan {
            @Schema(description = "日期")
            private String date;
            
            @Schema(description = "学习任务")
            private List<StudyTask> tasks;
            
            @Schema(description = "预计总时长(分钟)")
            private Integer totalMinutes;
            
            @Data
            @Schema(description = "学习任务")
            public static class StudyTask {
                @Schema(description = "任务标题")
                private String title;
                
                @Schema(description = "任务类型")
                private String type;
                
                @Schema(description = "内容ID")
                private Long contentId;
                
                @Schema(description = "预计时长(分钟)")
                private Integer minutes;
                
                @Schema(description = "难度等级")
                private String difficulty;
            }
        }
    }

    @Data
    @Schema(description = "AI学习进度分析响应")
    public static class AIProgressAnalysisResponse {
        @Schema(description = "总体进度百分比")
        private Double overallProgress;
        
        @Schema(description = "各课程进度")
        private Map<String, CourseProgress> courseProgressMap;
        
        @Schema(description = "学习时长统计")
        private StudyTimeStats studyTimeStats;
        
        @Schema(description = "成就列表")
        private List<Achievement> achievements;
        
        @Data
        @Schema(description = "课程进度")
        public static class CourseProgress {
            @Schema(description = "课程名称")
            private String courseName;
            
            @Schema(description = "完成百分比")
            private Double completionRate;
            
            @Schema(description = "已完成章节数")
            private Integer completedChapters;
            
            @Schema(description = "总章节数")
            private Integer totalChapters;
            
            @Schema(description = "最近学习时间")
            private LocalDateTime lastStudyTime;
        }
        
        @Data
        @Schema(description = "学习时长统计")
        public static class StudyTimeStats {
            @Schema(description = "本周学习时长(分钟)")
            private Integer weeklyMinutes;
            
            @Schema(description = "本月学习时长(分钟)")
            private Integer monthlyMinutes;
            
            @Schema(description = "累计学习时长(分钟)")
            private Integer totalMinutes;
            
            @Schema(description = "平均每日学习时长(分钟)")
            private Integer dailyAverage;
        }
        
        @Data
        @Schema(description = "成就")
        public static class Achievement {
            @Schema(description = "成就名称")
            private String name;
            
            @Schema(description = "成就描述")
            private String description;
            
            @Schema(description = "获得时间")
            private LocalDateTime achievedTime;
            
            @Schema(description = "成就类型")
            private String type;
        }
    }

    @Data
    @Schema(description = "AI复习推荐响应")
    public static class AIReviewRecommendationResponse {
        @Schema(description = "推荐复习内容")
        private List<ReviewItem> reviewItems;
        
        @Schema(description = "复习策略")
        private String strategy;
        
        @Schema(description = "预计复习时长(分钟)")
        private Integer estimatedMinutes;
        
        @Data
        @Schema(description = "复习项目")
        public static class ReviewItem {
            @Schema(description = "内容ID")
            private Long contentId;
            
            @Schema(description = "内容标题")
            private String title;
            
            @Schema(description = "内容类型")
            private String type;
            
            @Schema(description = "重要性级别")
            private String priority;
            
            @Schema(description = "预计复习时长(分钟)")
            private Integer minutes;
            
            @Schema(description = "上次学习时间")
            private LocalDateTime lastStudiedTime;
        }
    }

    @Data
    @Schema(description = "AI学习方法优化响应")
    public static class AILearningOptimizationResponse {
        @Schema(description = "学习方法建议")
        private List<String> methodSuggestions;
        
        @Schema(description = "学习时间建议")
        private TimeRecommendation timeRecommendation;
        
        @Schema(description = "学习环境建议")
        private List<String> environmentSuggestions;
        
        @Schema(description = "个性化建议")
        private String personalizedAdvice;
        
        @Data
        @Schema(description = "时间推荐")
        public static class TimeRecommendation {
            @Schema(description = "最佳学习时段")
            private List<String> optimalTimes;
            
            @Schema(description = "建议单次学习时长(分钟)")
            private Integer suggestedSessionLength;
            
            @Schema(description = "建议休息间隔(分钟)")
            private Integer suggestedBreakLength;
        }
    }

    @Data
    @Schema(description = "AI知识点掌握度响应")
    public static class AIKnowledgeMasteryResponse {
        @Schema(description = "知识点掌握情况")
        private Map<String, KnowledgePointMastery> knowledgePoints;
        
        @Schema(description = "总体掌握度")
        private Double overallMastery;
        
        @Schema(description = "推荐学习路径")
        private List<String> recommendedPath;
        
        @Data
        @Schema(description = "知识点掌握度")
        public static class KnowledgePointMastery {
            @Schema(description = "知识点名称")
            private String pointName;
            
            @Schema(description = "掌握度百分比")
            private Double masteryLevel;
            
            @Schema(description = "练习次数")
            private Integer practiceCount;
            
            @Schema(description = "正确率")
            private Double accuracy;
            
            @Schema(description = "最后练习时间")
            private LocalDateTime lastPracticeTime;
        }
    }

    @Data
    @Schema(description = "AI学习效率分析响应")
    public static class AIEfficiencyAnalysisResponse {
        @Schema(description = "学习效率分数")
        private Double efficiencyScore;
        
        @Schema(description = "效率趋势")
        private List<EfficiencyTrend> trends;
        
        @Schema(description = "效率提升建议")
        private List<String> improvements;
        
        @Data
        @Schema(description = "效率趋势")
        public static class EfficiencyTrend {
            @Schema(description = "日期")
            private String date;
            
            @Schema(description = "效率分数")
            private Double score;
            
            @Schema(description = "学习时长(分钟)")
            private Integer studyMinutes;
            
            @Schema(description = "完成任务数")
            private Integer tasksCompleted;
        }
    }

    @Data
    @Schema(description = "AI练习推荐请求")
    public static class AIPracticeRequest {
        @Schema(description = "科目ID")
        private Long subjectId;
        
        @Schema(description = "难度级别")
        private String difficultyLevel;
        
        @Schema(description = "练习类型")
        private String practiceType;
        
        @Schema(description = "练习时长(分钟)")
        private Integer duration;
    }

    @Data
    @Schema(description = "AI练习推荐响应")
    public static class AIPracticeRecommendationResponse {
        @Schema(description = "推荐练习列表")
        private List<PracticeItem> practices;
        
        @Schema(description = "总预计时长(分钟)")
        private Integer totalMinutes;
        
        @Data
        @Schema(description = "练习项目")
        public static class PracticeItem {
            @Schema(description = "练习ID")
            private Long practiceId;
            
            @Schema(description = "练习标题")
            private String title;
            
            @Schema(description = "练习类型")
            private String type;
            
            @Schema(description = "难度级别")
            private String difficulty;
            
            @Schema(description = "预计时长(分钟)")
            private Integer minutes;
            
            @Schema(description = "知识点")
            private List<String> knowledgePoints;
        }
    }

    @Data
    @Schema(description = "AI学习状态评估请求")
    public static class AIStateAssessmentRequest {
        @Schema(description = "评估类型")
        private String assessmentType;
        
        @Schema(description = "评估时间范围(天)")
        private Integer timeRangeDays;
    }

    @Data
    @Schema(description = "AI学习状态响应")
    public static class AILearningStateResponse {
        @Schema(description = "学习状态")
        private String learningState;
        
        @Schema(description = "专注度分数")
        private Double focusScore;
        
        @Schema(description = "学习动机分数")
        private Double motivationScore;
        
        @Schema(description = "疲劳度分数")
        private Double fatigueScore;
        
        @Schema(description = "状态建议")
        private List<String> stateRecommendations;
    }

    @Data
    @Schema(description = "AI学习报告响应")
    public static class AILearningReportResponse {
        @Schema(description = "报告标题")
        private String title;
        
        @Schema(description = "报告周期")
        private String period;
        
        @Schema(description = "学习总结")
        private LearningReport summary;
        
        @Schema(description = "详细分析")
        private Map<String, Object> detailedAnalysis;
        
        @Schema(description = "改进建议")
        private List<String> improvements;
        
        @Data
        @Schema(description = "学习报告")
        public static class LearningReport {
            @Schema(description = "总学习时长(分钟)")
            private Integer totalStudyMinutes;
            
            @Schema(description = "完成任务数")
            private Integer completedTasks;
            
            @Schema(description = "平均分数")
            private Double averageScore;
            
            @Schema(description = "进步幅度")
            private Double progressRate;
        }
    }

    @Data
    @Schema(description = "AI学习目标请求")
    public static class AILearningGoalRequest {
        @Schema(description = "目标类型")
        private String goalType;
        
        @Schema(description = "目标描述")
        private String description;
        
        @Schema(description = "目标期限")
        private LocalDateTime deadline;
        
        @Schema(description = "相关课程ID")
        private Long courseId;
    }

    @Data
    @Schema(description = "AI学习历史响应")
    public static class AILearningHistoryResponse {
        @Schema(description = "学习记录列表")
        private List<LearningRecord> records;
        
        @Schema(description = "总记录数")
        private Long totalCount;
        
        @Data
        @Schema(description = "学习记录")
        public static class LearningRecord {
            @Schema(description = "记录ID")
            private Long recordId;
            
            @Schema(description = "学习时间")
            private LocalDateTime studyTime;
            
            @Schema(description = "学习内容")
            private String content;
            
            @Schema(description = "学习时长(分钟)")
            private Integer duration;
            
            @Schema(description = "学习成果")
            private String outcome;
        }
    }

    @Data
    @Schema(description = "AI反馈请求")
    public static class AIFeedbackRequest {
        @Schema(description = "反馈类型")
        private String feedbackType;
        
        @Schema(description = "反馈内容")
        private String content;
        
        @Schema(description = "评分")
        private Integer rating;
        
        @Schema(description = "相关功能")
        private String relatedFeature;
    }
} 