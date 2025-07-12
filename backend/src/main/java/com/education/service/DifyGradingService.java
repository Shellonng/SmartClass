package com.education.service;

import com.education.dto.GradingRequestDTO;
import com.education.dto.GradingResultDTO;
import org.springframework.web.multipart.MultipartFile;

import java.util.Map;

/**
 * Dify 智能批改服务接口
 */
public interface DifyGradingService {
    
    /**
     * 上传文件到 Dify 平台
     * @param file 要上传的文件
     * @param user 用户标识
     * @return 文件上传结果，包含文件ID
     */
    Map<String, Object> uploadFile(MultipartFile file, String user);
    
    /**
     * 执行智能批改工作流
     * @param request 批改请求
     * @return 批改结果
     */
    GradingResultDTO executeGradingWorkflow(GradingRequestDTO request);
    
    /**
     * 批改作业提交
     * @param submissionId 提交记录ID
     * @param referenceAnswer 参考答案
     * @param submittedFile 提交的文件
     * @return 批改结果
     */
    GradingResultDTO gradeAssignmentSubmission(Long submissionId, String referenceAnswer, MultipartFile submittedFile);
    
    /**
     * 批改考试提交
     * @param submissionId 提交记录ID
     * @param referenceAnswer 参考答案
     * @param submittedFile 提交的文件
     * @return 批改结果
     */
    GradingResultDTO gradeExamSubmission(Long submissionId, String referenceAnswer, MultipartFile submittedFile);
    
    /**
     * 批量批改作业
     * @param assignmentId 作业ID
     * @param referenceAnswer 参考答案
     * @return 批改结果汇总
     */
    Map<String, Object> batchGradeAssignments(Long assignmentId, String referenceAnswer);
    
    /**
     * 获取批改工作流状态
     * @param workflowRunId 工作流运行ID
     * @return 工作流状态信息
     */
    Map<String, Object> getWorkflowStatus(String workflowRunId);
} 