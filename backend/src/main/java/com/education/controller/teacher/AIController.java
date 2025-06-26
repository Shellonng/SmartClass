package com.education.controller.teacher;

import com.education.dto.common.Result;
import com.education.dto.ai.AICommonDTOs.*;
import com.education.service.teacher.AIService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import jakarta.validation.Valid;
import java.util.List;

/**
 * 教师端AI工具控制器
 */
@Tag(name = "教师端AI工具", description = "智能批改、推荐、分析等AI功能")
@RestController
@RequestMapping("/api/teacher/ai")
@RequiredArgsConstructor
@Slf4j
public class AIController {

    private final AIService aiService;

    /**
     * 智能批改作业
     */
    @Operation(summary = "智能批改作业")
    @PostMapping("/grade")
    public Result<AIGradeResponse> intelligentGrading(@Valid @RequestBody AIGradeRequest request) {
        log.info("启动智能批改，任务ID：{}，提交ID：{}", request.getTaskId(), request.getSubmissionId());
        
        AIGradeResponse response = aiService.intelligentGrading(request);
        return Result.success(response);
    }

    /**
     * 批量智能批改
     */
    @Operation(summary = "批量智能批改")
    @PostMapping("/batch-grade")
    public Result<AIBatchGradeResponse> batchIntelligentGrading(@Valid @RequestBody AIBatchGradeRequest request) {
        log.info("启动批量智能批改，任务ID：{}，提交数量：{}", request.getTaskId(), request.getSubmissionIds().size());
        
        AIBatchGradeResponse response = aiService.batchIntelligentGrading(request);
        return Result.success(response);
    }

    /**
     * 生成学生学习推荐
     */
    @Operation(summary = "生成学生学习推荐")
    @PostMapping("/recommend")
    public Result<AIRecommendationResponse> generateRecommendations(@Valid @RequestBody AIRecommendationRequest request) {
        log.info("生成学习推荐，学生ID：{}，课程ID：{}", request.getStudentId(), request.getCourseId());
        
        AIRecommendationResponse response = aiService.generateRecommendations(request);
        return Result.success(response);
    }

    /**
     * 分析学生能力图谱
     */
    @Operation(summary = "分析学生能力图谱")
    @PostMapping("/ability-analysis")
    public Result<AIAbilityAnalysisResponse> analyzeStudentAbility(@Valid @RequestBody AIAbilityAnalysisRequest request) {
        log.info("分析学生能力图谱，学生ID：{}，分析维度：{}", request.getStudentId(), request.getAnalysisDimensions());
        
        AIAbilityAnalysisResponse response = aiService.analyzeStudentAbility(request);
        return Result.success(response);
    }

    /**
     * 自动生成知识图谱
     */
    @Operation(summary = "自动生成知识图谱")
    @PostMapping("/knowledge-graph")
    public Result<AIKnowledgeGraphResponse> generateKnowledgeGraph(@Valid @RequestBody AIKnowledgeGraphRequest request) {
        log.info("生成知识图谱，课程ID：{}，章节数：{}", request.getCourseId(), request.getChapterCount());
        
        AIKnowledgeGraphResponse response = aiService.generateKnowledgeGraph(request);
        return Result.success(response);
    }

    /**
     * 智能题目生成
     */
    @Operation(summary = "智能题目生成")
    @PostMapping("/generate-questions")
    public Result<AIQuestionGenerationResponse> generateQuestions(@Valid @RequestBody AIQuestionGenerationRequest request) {
        log.info("生成智能题目，知识点：{}，题目类型：{}，数量：{}", 
                request.getKnowledgePoints(), request.getQuestionType(), request.getQuestionCount());
        
        AIQuestionGenerationResponse response = aiService.generateQuestions(request);
        return Result.success(response);
    }

    /**
     * 学习路径优化
     */
    @Operation(summary = "学习路径优化")
    @PostMapping("/optimize-path")
    public Result<AILearningPathResponse> optimizeLearningPath(@Valid @RequestBody AILearningPathRequest request) {
        log.info("优化学习路径，学生ID：{}，目标技能：{}", request.getStudentId(), request.getTargetSkills());
        
        AILearningPathResponse response = aiService.optimizeLearningPath(request);
        return Result.success(response);
    }

    /**
     * 课堂表现分析
     */
    @Operation(summary = "课堂表现分析")
    @PostMapping("/classroom-analysis")
    public Result<AIClassroomAnalysisResponse> analyzeClassroomPerformance(@Valid @RequestBody AIClassroomAnalysisRequest request) {
        log.info("分析课堂表现，班级ID：{}，时间范围：{}", request.getClassId(), request.getTimeRange());
        
        AIClassroomAnalysisResponse response = aiService.analyzeClassroomPerformance(request);
        return Result.success(response);
    }

    /**
     * 智能教学建议
     */
    @Operation(summary = "智能教学建议")
    @PostMapping("/teaching-suggestions")
    public Result<AITeachingSuggestionResponse> generateTeachingSuggestions(@Valid @RequestBody AITeachingSuggestionRequest request) {
        log.info("生成教学建议，课程ID：{}，学生群体：{}", request.getCourseId(), request.getStudentGroup());
        
        AITeachingSuggestionResponse response = aiService.generateTeachingSuggestions(request);
        return Result.success(response);
    }

    /**
     * 上传文档进行AI分析
     */
    @Operation(summary = "上传文档进行AI分析")
    @PostMapping("/analyze-document")
    public Result<AIDocumentAnalysisResponse> analyzeDocument(
            @RequestParam("file") MultipartFile file,
            @RequestParam("analysisType") String analysisType,
            @RequestParam(value = "courseId", required = false) Long courseId) {
        log.info("上传文档进行AI分析，文件名：{}，分析类型：{}", file.getOriginalFilename(), analysisType);
        
        AIDocumentAnalysisResponse response = aiService.analyzeDocument(file, analysisType, courseId);
        return Result.success(response);
    }

    /**
     * 获取AI分析历史
     */
    @Operation(summary = "获取AI分析历史")
    @GetMapping("/analysis-history")
    public Result<List<AIAnalysisHistoryResponse>> getAnalysisHistory(
            @RequestParam(required = false) String type,
            @RequestParam(required = false) Long courseId,
            @RequestParam(defaultValue = "1") Integer page,
            @RequestParam(defaultValue = "10") Integer size) {
        log.info("获取AI分析历史，类型：{}，课程ID：{}", type, courseId);
        
        List<AIAnalysisHistoryResponse> response = (List<AIAnalysisHistoryResponse>) aiService.getAnalysisHistory(type, courseId, page, size);
        return Result.success(response);
    }

    /**
     * 配置AI模型参数
     */
    @Operation(summary = "配置AI模型参数")
    @PostMapping("/config")
    public Result<Void> configureAIModel(@Valid @RequestBody AIModelConfigRequest request) {
        log.info("配置AI模型参数，模型类型：{}，参数：{}", request.getModelType(), request.getParameters());
        
        aiService.configureAIModel(request);
        return Result.success("AI模型配置成功");
    }

    /**
     * 获取AI模型状态
     */
    @Operation(summary = "获取AI模型状态")
    @GetMapping("/model-status")
    public Result<AIModelStatusResponse> getAIModelStatus() {
        log.info("获取AI模型状态");
        
        AIModelStatusResponse response = aiService.getAIModelStatus();
        return Result.success(response);
    }

    /**
     * 训练个性化模型
     */
    @Operation(summary = "训练个性化模型")
    @PostMapping("/train-model")
    public Result<AIModelTrainingResponse> trainPersonalizedModel(@Valid @RequestBody AIModelTrainingRequest request) {
        log.info("训练个性化模型，数据集：{}，模型类型：{}", request.getDatasetId(), request.getModelType());
        
        AIModelTrainingResponse response = aiService.trainPersonalizedModel(request);
        return Result.success(response);
    }

    /**
     * 获取训练进度
     */
    @Operation(summary = "获取训练进度")
    @GetMapping("/training-progress/{trainingId}")
    public Result<AITrainingProgressResponse> getTrainingProgress(@PathVariable String trainingId) {
        log.info("获取训练进度，训练ID：{}", trainingId);
        
        AITrainingProgressResponse response = aiService.getTrainingProgress(trainingId);
        return Result.success(response);
    }
}