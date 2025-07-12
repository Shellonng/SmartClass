package com.education.service.impl;

import com.education.dto.GradingRequestDTO;
import com.education.dto.GradingResultDTO;
import com.education.entity.AssignmentSubmission;
import com.education.mapper.AssignmentSubmissionMapper;
import com.education.service.DifyGradingService;
import com.education.service.ExamService;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.time.LocalDateTime;
import java.util.*;

/**
 * Dify 智能批改服务实现类
 */
@Service
@Slf4j
public class DifyGradingServiceImpl implements DifyGradingService {
    
    @Value("${education.dify.api-url}")
    private String difyBaseUrl;
    
    @Value("${education.dify.api-keys.auto-grading}")
    private String apiKey;
    
    @Autowired
    private RestTemplate restTemplate;
    
    @Autowired
    private ObjectMapper objectMapper;
    
    @Autowired
    private AssignmentSubmissionMapper assignmentSubmissionMapper;
    
    @Autowired
    private ExamService examService;
    
    @Override
    @SuppressWarnings("unchecked")
    public Map<String, Object> uploadFile(MultipartFile file, String user) {
        try {
            String uploadUrl = difyBaseUrl + "/files/upload";
            
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.MULTIPART_FORM_DATA);
            headers.set("Authorization", "Bearer " + apiKey);
            
            MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
            body.add("file", new ByteArrayResource(file.getBytes()) {
                @Override
                public String getFilename() {
                    return file.getOriginalFilename();
                }
            });
            body.add("user", user);
            
            // 根据文件类型设置type参数
            String fileType = getFileType(file.getOriginalFilename());
            body.add("type", fileType);
            
            HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, headers);
            
            ResponseEntity<Map> response = restTemplate.exchange(uploadUrl, HttpMethod.POST, requestEntity, Map.class);
            
            if (response.getStatusCode() == HttpStatus.CREATED) {
                log.info("文件上传成功，文件ID：{}", response.getBody().get("id"));
                return response.getBody();
            } else {
                log.error("文件上传失败，状态码：{}", response.getStatusCode());
                return null;
            }
        } catch (Exception e) {
            log.error("文件上传发生错误", e);
            return null;
        }
    }
    
    @Override
    @SuppressWarnings("unchecked")
    public GradingResultDTO executeGradingWorkflow(GradingRequestDTO request) {
        try {
            String workflowUrl = difyBaseUrl + "/workflows/run";
            
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);
            headers.set("Authorization", "Bearer " + apiKey);
            
            // 构建请求体
            Map<String, Object> requestBody = new HashMap<>();
            requestBody.put("inputs", buildWorkflowInputs(request));
            requestBody.put("response_mode", request.getResponseMode());
            requestBody.put("user", request.getUser());
            
            HttpEntity<Map<String, Object>> requestEntity = new HttpEntity<>(requestBody, headers);
            
            ResponseEntity<Map> response = restTemplate.exchange(workflowUrl, HttpMethod.POST, requestEntity, Map.class);
            
            if (response.getStatusCode() == HttpStatus.OK) {
                return parseWorkflowResponse(response.getBody());
            } else {
                log.error("工作流执行失败，状态码：{}", response.getStatusCode());
                return GradingResultDTO.failed("工作流执行失败，状态码：" + response.getStatusCode());
            }
        } catch (Exception e) {
            log.error("工作流执行发生错误", e);
            return GradingResultDTO.failed("工作流执行发生错误：" + e.getMessage());
        }
    }
    
    @Override
    @SuppressWarnings("unchecked")
    public GradingResultDTO gradeAssignmentSubmission(Long submissionId, String referenceAnswer, MultipartFile submittedFile) {
        try {
            // 上传文件
            Map<String, Object> uploadResult = uploadFile(submittedFile, "teacher-" + System.currentTimeMillis());
            if (uploadResult == null) {
                return GradingResultDTO.failed("文件上传失败");
            }
            
            // 构建批改请求
            GradingRequestDTO request = new GradingRequestDTO();
            request.setUser("teacher-" + System.currentTimeMillis());
            request.setResponseMode("blocking");
            
            GradingRequestDTO.GradingInputs inputs = new GradingRequestDTO.GradingInputs();
            inputs.setReferenceAnswer(referenceAnswer);
            inputs.setFullScore(100); // 默认满分100
            inputs.setQuestionType("assignment");
            inputs.setGradingCriteria("请根据参考答案对学生提交的作业进行批改，给出合理的分数和详细的批改意见");
            
            // 设置提交的文件
            List<GradingRequestDTO.FileInput> fileInputs = new ArrayList<>();
            GradingRequestDTO.FileInput fileInput = new GradingRequestDTO.FileInput();
            fileInput.setType("document");
            fileInput.setTransferMethod("local_file");
            fileInput.setUploadFileId(uploadResult.get("id").toString());
            fileInputs.add(fileInput);
            inputs.setSubmittedFiles(fileInputs);
            
            request.setInputs(inputs);
            
            // 执行批改
            GradingResultDTO result = executeGradingWorkflow(request);
            
            // 更新提交记录
            if (result.getSuccess()) {
                AssignmentSubmission submission = assignmentSubmissionMapper.selectById(submissionId);
                if (submission != null) {
                    submission.setScore(result.getScore());
                    submission.setFeedback(result.getFeedback());
                    submission.setStatus(2); // 已批改
                    submission.setGradeTime(new Date());
                    assignmentSubmissionMapper.updateById(submission);
                }
            }
            
            return result;
        } catch (Exception e) {
            log.error("批改作业提交发生错误", e);
            return GradingResultDTO.failed("批改作业提交发生错误：" + e.getMessage());
        }
    }
    
    @Override
    @SuppressWarnings("unchecked")
    public GradingResultDTO gradeExamSubmission(Long submissionId, String referenceAnswer, MultipartFile submittedFile) {
        // 与作业批改类似，但设置题目类型为exam
        try {
            // 上传文件
            Map<String, Object> uploadResult = uploadFile(submittedFile, "teacher-" + System.currentTimeMillis());
            if (uploadResult == null) {
                return GradingResultDTO.failed("文件上传失败");
            }
            
            // 构建批改请求
            GradingRequestDTO request = new GradingRequestDTO();
            request.setUser("teacher-" + System.currentTimeMillis());
            request.setResponseMode("blocking");
            
            GradingRequestDTO.GradingInputs inputs = new GradingRequestDTO.GradingInputs();
            inputs.setReferenceAnswer(referenceAnswer);
            inputs.setFullScore(100); // 默认满分100
            inputs.setQuestionType("exam");
            inputs.setGradingCriteria("请根据参考答案对学生提交的考试答案进行批改，给出合理的分数和详细的批改意见");
            
            // 设置提交的文件
            List<GradingRequestDTO.FileInput> fileInputs = new ArrayList<>();
            GradingRequestDTO.FileInput fileInput = new GradingRequestDTO.FileInput();
            fileInput.setType("document");
            fileInput.setTransferMethod("local_file");
            fileInput.setUploadFileId(uploadResult.get("id").toString());
            fileInputs.add(fileInput);
            inputs.setSubmittedFiles(fileInputs);
            
            request.setInputs(inputs);
            
            // 执行批改
            GradingResultDTO result = executeGradingWorkflow(request);
            
            // 更新提交记录（如果有相应的考试提交记录表）
            if (result.getSuccess()) {
                // 这里可以添加考试提交记录的更新逻辑
                log.info("考试提交批改完成，提交ID：{}，得分：{}", submissionId, result.getScore());
            }
            
            return result;
        } catch (Exception e) {
            log.error("批改考试提交发生错误", e);
            return GradingResultDTO.failed("批改考试提交发生错误：" + e.getMessage());
        }
    }
    
    @Override
    @SuppressWarnings("unchecked")
    public Map<String, Object> batchGradeAssignments(Long assignmentId, String referenceAnswer) {
        try {
            // 获取所有未批改的提交记录
            List<AssignmentSubmission> submissions = assignmentSubmissionMapper.selectList(
                    new com.baomidou.mybatisplus.core.conditions.query.QueryWrapper<AssignmentSubmission>()
                            .eq("assignment_id", assignmentId)
                            .eq("status", 1) // 已提交未批改
            );
            
            Map<String, Object> batchResult = new HashMap<>();
            List<Map<String, Object>> results = new ArrayList<>();
            int successCount = 0;
            int failCount = 0;
            
            for (AssignmentSubmission submission : submissions) {
                try {
                    // 对于批量批改，这里假设提交的是文本内容而非文件
                    // 实际应用中可能需要根据具体情况处理
                    Map<String, Object> result = new HashMap<>();
                    result.put("submissionId", submission.getId());
                    result.put("studentId", submission.getStudentId());
                    
                    // 如果有文件路径，需要读取文件内容进行批改
                    if (submission.getFilePath() != null) {
                        // 这里需要实现文件读取和批改逻辑
                        result.put("status", "skipped");
                        result.put("message", "文件批改需要单独处理");
                    } else if (submission.getContent() != null) {
                        // 对文本内容进行批改
                        GradingResultDTO gradingResult = gradeTextContent(submission.getContent(), referenceAnswer);
                        if (gradingResult.getSuccess()) {
                            // 更新提交记录
                            submission.setScore(gradingResult.getScore());
                            submission.setFeedback(gradingResult.getFeedback());
                            submission.setStatus(2); // 已批改
                            submission.setGradeTime(new Date());
                            assignmentSubmissionMapper.updateById(submission);
                            
                            result.put("status", "success");
                            result.put("score", gradingResult.getScore());
                            result.put("feedback", gradingResult.getFeedback());
                            successCount++;
                        } else {
                            result.put("status", "failed");
                            result.put("error", gradingResult.getError());
                            failCount++;
                        }
                    } else {
                        result.put("status", "skipped");
                        result.put("message", "无提交内容");
                    }
                    
                    results.add(result);
                } catch (Exception e) {
                    log.error("批改提交记录失败，提交ID：{}", submission.getId(), e);
                    failCount++;
                }
            }
            
            batchResult.put("total", submissions.size());
            batchResult.put("success", successCount);
            batchResult.put("failed", failCount);
            batchResult.put("results", results);
            
            return batchResult;
        } catch (Exception e) {
            log.error("批量批改作业发生错误", e);
            Map<String, Object> errorResult = new HashMap<>();
            errorResult.put("error", "批量批改作业发生错误：" + e.getMessage());
            return errorResult;
        }
    }
    
    @Override
    @SuppressWarnings("unchecked")
    public Map<String, Object> getWorkflowStatus(String workflowRunId) {
        try {
            String statusUrl = difyBaseUrl + "/workflows/run/" + workflowRunId;
            
            HttpHeaders headers = new HttpHeaders();
            headers.set("Authorization", "Bearer " + apiKey);
            headers.setContentType(MediaType.APPLICATION_JSON);
            
            HttpEntity<String> requestEntity = new HttpEntity<>(headers);
            
            ResponseEntity<Map> response = restTemplate.exchange(statusUrl, HttpMethod.GET, requestEntity, Map.class);
            
            if (response.getStatusCode() == HttpStatus.OK) {
                return response.getBody();
            } else {
                log.error("获取工作流状态失败，状态码：{}", response.getStatusCode());
                return null;
            }
        } catch (Exception e) {
            log.error("获取工作流状态发生错误", e);
            return null;
        }
    }
    
    /**
     * 根据文件名获取文件类型
     */
    private String getFileType(String fileName) {
        if (fileName == null) return "TXT";
        
        String extension = fileName.substring(fileName.lastIndexOf('.') + 1).toUpperCase();
        switch (extension) {
            case "PDF":
                return "PDF";
            case "DOC":
            case "DOCX":
                return "DOCX";
            case "XLS":
            case "XLSX":
                return "XLSX";
            case "PPT":
            case "PPTX":
                return "PPTX";
            case "TXT":
                return "TXT";
            case "MD":
                return "MD";
            default:
                return "TXT";
        }
    }
    
    /**
     * 构建工作流输入参数
     */
    private Map<String, Object> buildWorkflowInputs(GradingRequestDTO request) {
        Map<String, Object> inputs = new HashMap<>();
        
        GradingRequestDTO.GradingInputs gradingInputs = request.getInputs();
        if (gradingInputs != null) {
            inputs.put("reference_answer", gradingInputs.getReferenceAnswer());
            inputs.put("grading_criteria", gradingInputs.getGradingCriteria());
            inputs.put("full_score", gradingInputs.getFullScore());
            inputs.put("question_type", gradingInputs.getQuestionType());
            inputs.put("course_info", gradingInputs.getCourseInfo());
            
            // 处理文件输入
            if (gradingInputs.getSubmittedFiles() != null && !gradingInputs.getSubmittedFiles().isEmpty()) {
                List<Map<String, Object>> fileList = new ArrayList<>();
                for (GradingRequestDTO.FileInput fileInput : gradingInputs.getSubmittedFiles()) {
                    Map<String, Object> fileMap = new HashMap<>();
                    fileMap.put("type", fileInput.getType());
                    fileMap.put("transfer_method", fileInput.getTransferMethod());
                    if ("remote_url".equals(fileInput.getTransferMethod())) {
                        fileMap.put("url", fileInput.getUrl());
                    } else {
                        fileMap.put("upload_file_id", fileInput.getUploadFileId());
                    }
                    fileList.add(fileMap);
                }
                inputs.put("submitted_files", fileList);
            }
        }
        
        return inputs;
    }
    
    /**
     * 解析工作流响应
     */
    @SuppressWarnings("unchecked")
    private GradingResultDTO parseWorkflowResponse(Map<String, Object> response) {
        try {
            GradingResultDTO result = new GradingResultDTO();
            
            result.setWorkflowRunId((String) response.get("workflow_run_id"));
            result.setTaskId((String) response.get("task_id"));
            
            Map<String, Object> data = (Map<String, Object>) response.get("data");
            if (data != null) {
                result.setStatus((String) data.get("status"));
                result.setError((String) data.get("error"));
                result.setElapsedTime((Double) data.get("elapsed_time"));
                result.setTotalTokens((Integer) data.get("total_tokens"));
                result.setTotalSteps((Integer) data.get("total_steps"));
                
                // 解析输出内容
                Map<String, Object> outputs = (Map<String, Object>) data.get("outputs");
                if (outputs != null) {
                    result.setOutputs(outputs);
                    
                    // 尝试从输出中提取分数和反馈
                    try {
                        if (outputs.containsKey("score")) {
                            result.setScore(Integer.parseInt(outputs.get("score").toString()));
                        }
                        if (outputs.containsKey("feedback")) {
                            result.setFeedback(outputs.get("feedback").toString());
                        }
                        if (outputs.containsKey("detailed_feedback")) {
                            result.setDetailedFeedback(outputs.get("detailed_feedback").toString());
                        }
                        if (outputs.containsKey("suggestions")) {
                            result.setSuggestions(outputs.get("suggestions").toString());
                        }
                    } catch (Exception e) {
                        log.warn("解析输出内容时发生错误", e);
                    }
                }
            }
            
            result.setSuccess("succeeded".equals(result.getStatus()));
            result.setCreatedAt(LocalDateTime.now());
            
            return result;
        } catch (Exception e) {
            log.error("解析工作流响应发生错误", e);
            return GradingResultDTO.failed("解析工作流响应发生错误：" + e.getMessage());
        }
    }
    
    /**
     * 批改文本内容
     */
    private GradingResultDTO gradeTextContent(String content, String referenceAnswer) {
        try {
            GradingRequestDTO request = new GradingRequestDTO();
            request.setUser("teacher-" + System.currentTimeMillis());
            request.setResponseMode("blocking");
            
            GradingRequestDTO.GradingInputs inputs = new GradingRequestDTO.GradingInputs();
            inputs.setReferenceAnswer(referenceAnswer);
            inputs.setFullScore(100);
            inputs.setQuestionType("text");
            inputs.setGradingCriteria("请根据参考答案对学生提交的文本答案进行批改，给出合理的分数和详细的批改意见");
            
            request.setInputs(inputs);
            
            return executeGradingWorkflow(request);
        } catch (Exception e) {
            log.error("批改文本内容发生错误", e);
            return GradingResultDTO.failed("批改文本内容发生错误：" + e.getMessage());
        }
    }
} 