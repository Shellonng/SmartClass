package com.education.controller.student;

import com.education.dto.common.Result;
import com.education.dto.ai.AIStudentDTOs.*;
import com.education.service.student.AILearningService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.*;

import jakarta.validation.Valid;
import java.util.List;

/**
 * 学生端AI学习助手控制器
 */
@Tag(name = "学生端AI学习助手", description = "个性化学习推荐、智能答疑、学习分析等功能")
@RestController
@RequestMapping("/api/student/ai-learning")
@RequiredArgsConstructor
@Slf4j
public class AILearningController {

    private final AILearningService aiLearningService;

    /**
     * 获取个性化学习推荐
     */
    @Operation(summary = "获取个性化学习推荐")
    @GetMapping("/recommendations")
    public Result<AIPersonalRecommendationResponse> getPersonalRecommendations(
            @RequestParam(required = false) Long courseId,
            @RequestParam(required = false) String learningGoal,
            @RequestParam(defaultValue = "10") Integer limit) {
        log.info("获取个性化学习推荐，课程ID：{}，学习目标：{}", courseId, learningGoal);
        
        AIPersonalRecommendationResponse response = aiLearningService.getPersonalRecommendations(courseId, learningGoal, limit);
        return Result.success(response);
    }

    /**
     * AI智能答疑
     */
    @Operation(summary = "AI智能答疑")
    @PostMapping("/question-answer")
    public Result<AIQuestionAnswerResponse> askQuestion(@Valid @RequestBody AIQuestionRequest request) {
        log.info("AI智能答疑，问题类型：{}，问题长度：{}", request.getQuestionType(), request.getQuestion().length());
        
        AIQuestionAnswerResponse response = aiLearningService.answerQuestion(request);
        return Result.success(response);
    }

    /**
     * 获取学习能力分析
     */
    @Operation(summary = "获取学习能力分析")
    @GetMapping("/ability-analysis")
    public Result<AIStudentAbilityResponse> getAbilityAnalysis(
            @RequestParam(required = false) Long courseId,
            @RequestParam(required = false) String timeRange) {
        log.info("获取学习能力分析，课程ID：{}，时间范围：{}", courseId, timeRange);
        
        AIStudentAbilityResponse response = aiLearningService.getAbilityAnalysis(courseId, timeRange);
        return Result.success(response);
    }

    /**
     * 生成个性化学习计划
     */
    @Operation(summary = "生成个性化学习计划")
    @PostMapping("/study-plan")
    public Result<AIStudyPlanResponse> generateStudyPlan(@Valid @RequestBody AIStudyPlanRequest request) {
        log.info("生成个性化学习计划，目标：{}，时间：{}天", request.getGoal(), request.getPlanDays());
        
        AIStudyPlanResponse response = aiLearningService.generateStudyPlan(request);
        return Result.success("学习计划生成成功", response);
    }

    /**
     * 学习进度智能分析
     */
    @Operation(summary = "学习进度智能分析")
    @GetMapping("/progress-analysis")
    public Result<AIProgressAnalysisResponse> analyzeProgress(
            @RequestParam(required = false) Long courseId,
            @RequestParam(required = false) String period) {
        log.info("学习进度智能分析，课程ID：{}，周期：{}", courseId, period);
        
        AIProgressAnalysisResponse response = aiLearningService.analyzeProgress(courseId, period);
        return Result.success(response);
    }

    /**
     * 智能复习推荐
     */
    @Operation(summary = "智能复习推荐")
    @GetMapping("/review-recommendations")
    public Result<AIReviewRecommendationResponse> getReviewRecommendations(
            @RequestParam(required = false) Long courseId,
            @RequestParam(defaultValue = "week") String reviewCycle) {
        log.info("获取智能复习推荐，课程ID：{}，复习周期：{}", courseId, reviewCycle);
        
        AIReviewRecommendationResponse response = aiLearningService.getReviewRecommendations(courseId, reviewCycle);
        return Result.success(response);
    }

    /**
     * 学习方法优化建议
     */
    @Operation(summary = "学习方法优化建议")
    @GetMapping("/learning-optimization")
    public Result<AILearningOptimizationResponse> getLearningOptimization(
            @RequestParam(required = false) Long courseId) {
        log.info("获取学习方法优化建议，课程ID：{}", courseId);
        
        AILearningOptimizationResponse response = aiLearningService.getLearningOptimization(courseId);
        return Result.success(response);
    }

    /**
     * 知识点掌握度分析
     */
    @Operation(summary = "知识点掌握度分析")
    @GetMapping("/knowledge-mastery")
    public Result<AIKnowledgeMasteryResponse> analyzeKnowledgeMastery(
            @RequestParam Long courseId,
            @RequestParam(required = false) Long chapterId) {
        log.info("分析知识点掌握度，课程ID：{}，章节ID：{}", courseId, chapterId);
        
        AIKnowledgeMasteryResponse response = aiLearningService.analyzeKnowledgeMastery(courseId, chapterId);
        return Result.success(response);
    }

    /**
     * 学习效率分析
     */
    @Operation(summary = "学习效率分析")
    @GetMapping("/efficiency-analysis")
    public Result<AIEfficiencyAnalysisResponse> analyzeEfficiency(
            @RequestParam(required = false) String timeRange) {
        log.info("分析学习效率，时间范围：{}", timeRange);
        
        AIEfficiencyAnalysisResponse response = aiLearningService.analyzeEfficiency(timeRange);
        return Result.success(response);
    }

    /**
     * 智能练习推荐
     */
    @Operation(summary = "智能练习推荐")
    @PostMapping("/practice-recommendations")
    public Result<AIPracticeRecommendationResponse> getPracticeRecommendations(
            @Valid @RequestBody AIPracticeRequest request) {
        log.info("获取智能练习推荐，科目ID：{}，难度：{}", request.getSubjectId(), request.getDifficultyLevel());
        
        AIPracticeRecommendationResponse response = aiLearningService.getPracticeRecommendations(request);
        return Result.success(response);
    }

    /**
     * 学习状态评估
     */
    @Operation(summary = "学习状态评估")
    @PostMapping("/state-assessment")
    public Result<AILearningStateResponse> assessLearningState(@Valid @RequestBody AIStateAssessmentRequest request) {
        log.info("评估学习状态，评估类型：{}", request.getAssessmentType());
        
        AILearningStateResponse response = aiLearningService.assessLearningState(request);
        return Result.success(response);
    }

    /**
     * 获取AI学习报告
     */
    @Operation(summary = "获取AI学习报告")
    @GetMapping("/learning-report")
    public Result<AILearningReportResponse> getLearningReport(
            @RequestParam(required = false) String reportType,
            @RequestParam(required = false) String timeRange) {
        log.info("获取AI学习报告，报告类型：{}，时间范围：{}", reportType, timeRange);
        
        AILearningReportResponse response = aiLearningService.getLearningReport(reportType, timeRange);
        return Result.success(response);
    }

    /**
     * 设置学习目标
     */
    @Operation(summary = "设置学习目标")
    @PostMapping("/learning-goals")
    public Result<Void> setLearningGoals(@Valid @RequestBody AILearningGoalRequest request) {
        log.info("设置学习目标，目标类型：{}", request.getGoalType());
        
        aiLearningService.setLearningGoals(request);
        return Result.success("学习目标设置成功");
    }

    /**
     * 获取学习历史
     */
    @Operation(summary = "获取学习历史")
    @GetMapping("/learning-history")
    public Result<List<AILearningHistoryResponse>> getLearningHistory(
            @RequestParam(required = false) Long courseId,
            @RequestParam(defaultValue = "1") Integer page,
            @RequestParam(defaultValue = "20") Integer size) {
        log.info("获取学习历史，课程ID：{}，页码：{}", courseId, page);
        
        List<AILearningHistoryResponse> response = aiLearningService.getLearningHistory(courseId, page, size);
        return Result.success(response);
    }

    /**
     * 反馈AI推荐质量
     */
    @Operation(summary = "反馈AI推荐质量")
    @PostMapping("/feedback")
    public Result<Void> submitFeedback(@Valid @RequestBody AIFeedbackRequest request) {
        log.info("提交AI推荐反馈，反馈类型：{}，评分：{}", request.getFeedbackType(), request.getRating());
        
        aiLearningService.submitFeedback(request);
        return Result.success("反馈提交成功");
    }
}