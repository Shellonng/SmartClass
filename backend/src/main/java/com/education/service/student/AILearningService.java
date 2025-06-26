package com.education.service.student;

import com.education.dto.ai.AIStudentDTOs.*;
import java.util.List;

/**
 * 学生端AI学习服务接口
 */
public interface AILearningService {

    /**
     * 获取个性化学习推荐
     */
    AIPersonalRecommendationResponse getPersonalRecommendations(Long courseId, String learningGoal, Integer limit);

    /**
     * AI智能答疑
     */
    AIQuestionAnswerResponse answerQuestion(AIQuestionRequest request);

    /**
     * 获取学习能力分析
     */
    AIStudentAbilityResponse getAbilityAnalysis(Long courseId, String timeRange);

    /**
     * 生成个性化学习计划
     */
    AIStudyPlanResponse generateStudyPlan(AIStudyPlanRequest request);

    /**
     * 学习进度智能分析
     */
    AIProgressAnalysisResponse analyzeProgress(Long courseId, String period);

    /**
     * 智能复习推荐
     */
    AIReviewRecommendationResponse getReviewRecommendations(Long courseId, String reviewCycle);

    /**
     * 学习方法优化建议
     */
    AILearningOptimizationResponse getLearningOptimization(Long courseId);

    /**
     * 知识点掌握度分析
     */
    AIKnowledgeMasteryResponse analyzeKnowledgeMastery(Long courseId, Long chapterId);

    /**
     * 学习效率分析
     */
    AIEfficiencyAnalysisResponse analyzeEfficiency(String timeRange);

    /**
     * 智能练习推荐
     */
    AIPracticeRecommendationResponse getPracticeRecommendations(AIPracticeRequest request);

    /**
     * 学习状态评估
     */
    AILearningStateResponse assessLearningState(AIStateAssessmentRequest request);

    /**
     * 获取AI学习报告
     */
    AILearningReportResponse getLearningReport(String reportType, String timeRange);

    /**
     * 设置学习目标
     */
    void setLearningGoals(AILearningGoalRequest request);

    /**
     * 获取学习历史
     */
    List<AILearningHistoryResponse> getLearningHistory(Long courseId, Integer page, Integer size);

    /**
     * 反馈AI推荐质量
     */
    void submitFeedback(AIFeedbackRequest request);
} 