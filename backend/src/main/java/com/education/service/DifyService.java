package com.education.service;

import com.education.config.DifyConfig;
import com.education.dto.DifyDTO;
import com.education.dto.KnowledgeGraphDTO;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.client.ResourceAccessException;
import org.springframework.http.client.ClientHttpRequestFactory;
import org.springframework.http.client.HttpComponentsClientHttpRequestFactory;

import java.util.*;
import java.util.concurrent.CompletableFuture;
import java.util.Collections;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Dify AI服务
 * @author Education Platform Team
 */
@Slf4j
@Service
public class DifyService {

    @Autowired
    private DifyConfig difyConfig;

    @Autowired
    private ObjectMapper objectMapper;
    
    @Autowired
    private RestTemplate restTemplate;  // 直接注入配置好的RestTemplate

    /**
     * 调用Dify工作流API
     */
    public DifyDTO.DifyResponse callWorkflowApi(String appType, Map<String, Object> inputs, String userId) {
        int maxRetries = 2; // 最大重试次数
        int currentRetry = 0;
        
        while (currentRetry <= maxRetries) {
            // 正确获取毫秒为单位的超时时间并转换为秒
            int timeoutMillis = difyConfig.getTimeout();
            int timeoutSeconds = timeoutMillis / 1000;
            if (timeoutSeconds <= 0) timeoutSeconds = 480; // 设置默认值为480秒（8分钟）
            
            if (currentRetry > 0) {
                log.info("正在进行第{}次重试，超时设置为{}秒", currentRetry, timeoutSeconds);
            }
            
            try {
                log.info("开始调用Dify API，应用类型: {}, 用户ID: {}, 重试次数: {}, 超时设置: {}秒", 
                        appType, userId, currentRetry, timeoutSeconds);
                
                String apiKey = difyConfig.getApiKey(appType);
                if (apiKey == null) {
                    log.error("未找到API密钥，应用类型: {}", appType);
                    throw new RuntimeException("未配置" + appType + "的API密钥");
                }
                log.info("成功获取API密钥");
    
                // 构建请求 - 按照Dify API文档格式
                Map<String, Object> requestBody = new HashMap<>();
                
                // 将inputs内容转换为query字符串
                StringBuilder queryBuilder = new StringBuilder();
                for (Map.Entry<String, Object> entry : inputs.entrySet()) {
                    queryBuilder.append(entry.getKey()).append(": ");
                    if (entry.getValue() instanceof String) {
                        queryBuilder.append(entry.getValue());
                    } else {
                        queryBuilder.append(objectMapper.writeValueAsString(entry.getValue()));
                    }
                    queryBuilder.append("\n");
                }
                
                // 添加特殊指令，让AI不生成thinking过程
                queryBuilder.append("\n重要指令：\n");
                queryBuilder.append("1. 请直接生成最终结果，按照指定格式返回JSON对象\n");
                queryBuilder.append("2. 严格禁止使用<think>标签或输出任何思考过程\n");
                queryBuilder.append("3. 禁止在JSON前后添加任何说明、注释或markdown标记\n");
                queryBuilder.append("4. 直接提供格式化的JSON响应，不要包含任何额外文本\n");
                queryBuilder.append("5. 确保JSON格式正确，可以被直接解析\n");
                
                // 为知识图谱特别添加JSON格式要求
                if ("knowledge-graph".equals(appType)) {
                    queryBuilder.append("\n\n### 知识图谱输出格式必须严格符合以下要求：\n");
                    queryBuilder.append("1. 节点必须使用name字段表示名称（严禁使用content字段）\n");
                    queryBuilder.append("2. 每个节点必须包含id、name、type、level和description字段\n");
                    queryBuilder.append("3. 每条边必须包含id、source、target、type和description字段\n");
                    queryBuilder.append("4. 直接输出JSON，不要使用markdown代码块，不要添加任何说明\n");
                    queryBuilder.append("5. 请严格按照以下JSON结构输出：\n");
                    queryBuilder.append("{\n");
                    queryBuilder.append("  \"title\": \"知识图谱标题\",\n");
                    queryBuilder.append("  \"description\": \"知识图谱描述\",\n");
                    queryBuilder.append("  \"nodes\": [\n");
                    queryBuilder.append("    {\n");
                    queryBuilder.append("      \"id\": \"node_1\",\n");
                    queryBuilder.append("      \"name\": \"节点名称\",\n");
                    queryBuilder.append("      \"type\": \"concept\",\n");
                    queryBuilder.append("      \"level\": 1,\n");
                    queryBuilder.append("      \"description\": \"节点详细描述\"\n");
                    queryBuilder.append("    }\n");
                    queryBuilder.append("  ],\n");
                    queryBuilder.append("  \"edges\": [\n");
                    queryBuilder.append("    {\n");
                    queryBuilder.append("      \"id\": \"edge_1\",\n");
                    queryBuilder.append("      \"source\": \"node_1\",\n");
                    queryBuilder.append("      \"target\": \"node_2\",\n");
                    queryBuilder.append("      \"type\": \"contains\",\n");
                    queryBuilder.append("      \"description\": \"关系描述\"\n");
                    queryBuilder.append("    }\n");
                    queryBuilder.append("  ]\n");
                    queryBuilder.append("}\n");
                }
                
                // 按照Dify API文档设置请求参数
                requestBody.put("query", queryBuilder.toString());  // 用户输入/提问内容
                requestBody.put("inputs", inputs);                  // 变量值
                requestBody.put("response_mode", "blocking");       // 使用阻塞模式，避免流式响应可能导致的格式问题
                requestBody.put("user", userId);                    // 用户标识
                
                // 添加禁用思考过程的参数
                Map<String, Object> parameters = new HashMap<>();
                parameters.put("disable_thinking", true);           // 禁用思考过程
                parameters.put("return_json", true);                // 强制返回JSON格式
                parameters.put("thinking_visible", false);          // 确保思考过程不可见
                parameters.put("response_format", "json");          // 指定响应格式为JSON
                requestBody.put("parameters", parameters);
                
                log.info("已构建请求体，使用blocking响应模式，已添加禁用思考过程的参数");
                
                String requestBodyJson = objectMapper.writeValueAsString(requestBody);
                log.debug("请求体JSON: {}", requestBodyJson);
                
                // 构建API URL
                String apiUrl = difyConfig.getApiUrl() + "/chat-messages";
                log.info("API URL: {}", apiUrl);
                
                log.info("开始创建HTTP客户端...");
                // 创建HTTP客户端，使用8分钟的超时设置
                java.net.http.HttpClient httpClient = java.net.http.HttpClient.newBuilder()
                    .version(java.net.http.HttpClient.Version.HTTP_2)
                    .connectTimeout(java.time.Duration.ofSeconds(480)) // 统一设置为480秒（8分钟）
                    .build();
                log.info("HTTP客户端创建成功，连接超时设置为480秒");
                    
                log.info("开始构建HTTP请求...");
                java.net.http.HttpRequest request = java.net.http.HttpRequest.newBuilder()
                    .uri(java.net.URI.create(apiUrl))
                    .timeout(java.time.Duration.ofSeconds(480)) // 统一设置为480秒（8分钟）
                    .header("Content-Type", "application/json")
                    .header("Authorization", "Bearer " + apiKey)
                    .header("Accept", "application/json")
                    .POST(java.net.http.HttpRequest.BodyPublishers.ofString(requestBodyJson))
                    .build();
                log.info("HTTP请求构建成功，请求超时设置为480秒");
                    
                log.info("准备发送请求到Dify API: {}", apiUrl);
                
                try {
                    log.info("开始发送HTTP请求...");
                    // 记录请求开始时间
                    long startTime = System.currentTimeMillis();
                    
                    // 发送请求并获取响应
                    java.net.http.HttpResponse<String> response = httpClient.send(request, 
                        java.net.http.HttpResponse.BodyHandlers.ofString());
                    
                    // 计算请求耗时
                    long endTime = System.currentTimeMillis();
                    long duration = endTime - startTime;
                    log.info("HTTP请求已完成，耗时: {}毫秒", duration);
                    
                    int statusCode = response.statusCode();
                    String responseBody = response.body();
                    
                    log.info("收到HTTP响应，状态码: {}", statusCode);
                    if (responseBody != null) {
                        log.info("响应体长度: {} 字节", responseBody.length());
                        if (responseBody.length() <= 1000) {
                            log.debug("完整响应体: {}", responseBody);
                        } else {
                            log.debug("响应体字符: {}", responseBody.substring(0, 1000) + "...");
                        }
                    } else {
                        log.warn("响应体为空");
                    }
                    
                    if (statusCode == 200) {
                        log.info("HTTP状态码200，开始处理流式响应...");
                        
                        // 处理流式响应，将所有的数据块合并
                        StringBuilder fullAnswer = new StringBuilder();
                        Map<String, Object> metadataMap = new HashMap<>();
                        String taskId = null;
                        String messageId = null;
                        String conversationId = null;
                        
                        // 解析流式响应
                        String[] chunks = responseBody.split("data: ");
                        for (String chunk : chunks) {
                            if (chunk.trim().isEmpty()) continue;
                            
                            try {
                                Map<String, Object> chunkData = objectMapper.readValue(chunk.trim(), Map.class);
                                String event = (String) chunkData.get("event");
                                
                                if ("message".equals(event)) {
                                    // 累积文本响应
                                    String answerPart = (String) chunkData.get("answer");
                                    if (answerPart != null) {
                                        fullAnswer.append(answerPart);
                                    }
                                    
                                    // 获取消息ID和会话ID
                                    if (messageId == null) {
                                        messageId = (String) chunkData.get("message_id");
                                    }
                                    
                                    if (conversationId == null) {
                                        conversationId = (String) chunkData.get("conversation_id");
                                    }
                                    
                                    // 获取任务ID
                                    if (taskId == null) {
                                        taskId = (String) chunkData.get("task_id");
                                    }
                                } else if ("message_end".equals(event)) {
                                    // 获取元数据
                                    metadataMap = (Map<String, Object>) chunkData.get("metadata");
                                    
                                    // 获取任务ID
                                    if (taskId == null) {
                                        taskId = (String) chunkData.get("task_id");
                                    }
                                }
                            } catch (Exception e) {
                                log.warn("解析流式数据块失败: {}", e.getMessage());
                            }
                        }
                        
                        // 创建响应对象
                        DifyDTO.DifyResponse difyResponse = new DifyDTO.DifyResponse();
                            difyResponse.setTaskId(taskId);
                        difyResponse.setStatus("completed");
                        
                        // 设置消息ID和会话ID
                        difyResponse.setMessageId(messageId);
                        difyResponse.setConversationId(conversationId);
                        
                        // 将累积的回答作为结果
                        String finalAnswer = fullAnswer.toString();
                        
                        // 先检查并处理可能的<think>标签
                        if (finalAnswer.contains("<think>") || finalAnswer.contains("```")) {
                            log.info("检测到答案可能包含<think>标签或代码块，尝试提取JSON...");
                            String extractedJson = extractJsonFromText(finalAnswer);
                            if (extractedJson != null && !extractedJson.isEmpty()) {
                                log.info("成功从答案中提取JSON，长度: {} 字符", extractedJson.length());
                                finalAnswer = extractedJson;
                            } else {
                                log.warn("无法从答案中提取有效JSON，将使用原始答案");
                            }
                        }
                        
                        // 尝试将处理后的answer解析为JSON对象
                        try {
                            log.debug("尝试将答案解析为JSON...");
                            Map<String, Object> data = objectMapper.readValue(finalAnswer, Map.class);
                            difyResponse.setData(data);
                            log.info("成功将答案解析为JSON对象");
                        } catch (Exception e) {
                            log.info("答案不是JSON格式，作为普通文本处理: {}", e.getMessage());
                            // 如果不是有效的JSON，则创建一个包含answer的数据对象
                            Map<String, Object> data = new HashMap<>();
                            data.put("content", finalAnswer);
                            difyResponse.setData(data);
                            log.info("已将答案作为content字段存储");
                        }
                        
                        // 设置元数据
                        if (metadataMap != null && !metadataMap.isEmpty()) {
                            difyResponse.setMetadata(metadataMap);
                            }
                            
                        log.info("成功解析Dify API响应");
                            
                            // 检查响应数据是否为空
                            if (difyResponse.getData() == null || difyResponse.getData().isEmpty()) {
                                log.warn("Dify API响应数据为空，将使用本地模板");
                                // 创建一个带有空数据的响应对象，避免空指针异常
                                difyResponse.setData(new HashMap<>());
                            }
                            
                            log.info("Dify API调用成功完成");
                            return difyResponse;
                    } else {
                        log.error("Dify API调用失败: HTTP状态码 {}, 响应: {}", statusCode, responseBody);
                        
                        // 对于某些状态码，我们可以重试
                        if ((statusCode == 429 || statusCode >= 500) && currentRetry < maxRetries) {
                            log.info("遇到可重试的状态码 {}，将进行重试", statusCode);
                            currentRetry++;
                            // 指数退避，等待一段时间后重试
                            int waitTime = (int) Math.pow(2, currentRetry) * 1000;
                            log.info("等待{}毫秒后重试", waitTime);
                            Thread.sleep(waitTime);
                            continue;
                        }
                        
                        // 创建一个失败的响应对象
                        DifyDTO.DifyResponse failureResponse = new DifyDTO.DifyResponse();
                        failureResponse.setStatus("failed");
                        failureResponse.setError("API调用失败: HTTP状态码 " + statusCode);
                        return failureResponse;
                    }
                } catch (java.net.ConnectException e) {
                    log.error("连接到Dify API服务器失败: {}", e.getMessage(), e);
                    
                    // 尝试诊断连接问题
                    try {
                        log.info("尝试诊断连接问题...");
                        java.net.InetAddress address = java.net.InetAddress.getByName(new java.net.URL(apiUrl).getHost());
                        boolean reachable = address.isReachable(5000);
                        log.info("主机可达性测试: {}", reachable ? "可达" : "不可达");
                    } catch (Exception diagEx) {
                        log.error("诊断连接问题时出错: {}", diagEx.getMessage());
                    }
                    
                    if (currentRetry < maxRetries) {
                        log.info("连接失败，将进行重试");
                        currentRetry++;
                        // 指数退避，等待一段时间后重试
                        int waitTime = (int) Math.pow(2, currentRetry) * 1000;
                        log.info("等待{}毫秒后重试", waitTime);
                        Thread.sleep(waitTime);
                        continue;
                    }
                    
                    DifyDTO.DifyResponse failureResponse = new DifyDTO.DifyResponse();
                    failureResponse.setStatus("failed");
                    failureResponse.setError("连接到AI服务失败: " + e.getMessage());
                    return failureResponse;
                } catch (java.net.http.HttpTimeoutException e) {
                    log.error("Dify API请求超时: {}", e.getMessage(), e);
                    
                    if (currentRetry < maxRetries) {
                        log.info("请求超时，将进行重试");
                        currentRetry++;
                        continue;
                    }
                    
                    DifyDTO.DifyResponse failureResponse = new DifyDTO.DifyResponse();
                    failureResponse.setStatus("failed");
                    failureResponse.setError("AI服务请求超时: " + e.getMessage());
                    return failureResponse;
                } catch (Exception e) {
                    log.error("Dify API调用异常，将使用本地模板: {}", e.getMessage(), e);
                    
                    if (currentRetry < maxRetries) {
                        log.info("请求异常，将进行重试");
                        currentRetry++;
                        // 指数退避，等待一段时间后重试
                        int waitTime = (int) Math.pow(2, currentRetry) * 1000;
                        log.info("等待{}毫秒后重试", waitTime);
                        Thread.sleep(waitTime);
                        continue;
                    }
                    
                    // 创建一个失败的响应对象，让后续逻辑使用本地模板
                    DifyDTO.DifyResponse failureResponse = new DifyDTO.DifyResponse();
                    failureResponse.setStatus("failed");
                    failureResponse.setError("AI服务调用失败: " + e.getMessage());
                    return failureResponse;
                }
            } catch (ResourceAccessException e) {
                log.error("Dify API连接超时: {}", e.getMessage(), e);
                
                if (currentRetry < maxRetries) {
                    log.info("资源访问异常，将进行重试");
                    currentRetry++;
                    try {
                        // 指数退避，等待一段时间后重试
                        int waitTime = (int) Math.pow(2, currentRetry) * 1000;
                        log.info("等待{}毫秒后重试", waitTime);
                        Thread.sleep(waitTime);
                    } catch (InterruptedException ie) {
                        Thread.currentThread().interrupt();
                    }
                    continue;
                }
                
                throw new RuntimeException("AI服务暂时不可用，请稍后重试");
            } catch (InterruptedException e) {
                log.error("线程被中断: {}", e.getMessage(), e);
                Thread.currentThread().interrupt();
                throw new RuntimeException("AI服务调用被中断");
            } catch (Exception e) {
                log.error("调用Dify API异常: {}", e.getMessage(), e);
                
                if (currentRetry < maxRetries) {
                    log.info("发生异常，将进行重试");
                    currentRetry++;
                    try {
                        // 指数退避，等待一段时间后重试
                        int waitTime = (int) Math.pow(2, currentRetry) * 1000;
                        log.info("等待{}毫秒后重试", waitTime);
                        Thread.sleep(waitTime);
                    } catch (InterruptedException ie) {
                        Thread.currentThread().interrupt();
                    }
                    continue;
                }
                
                throw new RuntimeException("AI服务调用失败: " + e.getMessage());
            }
        }
        
        // 如果所有重试都失败了
        log.error("所有重试都失败，无法调用Dify API");
        DifyDTO.DifyResponse failureResponse = new DifyDTO.DifyResponse();
        failureResponse.setStatus("failed");
        failureResponse.setError("经过多次尝试，AI服务仍然不可用，请稍后重试");
        return failureResponse;
    }

    /**
     * 查询任务状态
     */
    public DifyDTO.DifyResponse getTaskStatus(String taskId, String appType) {
        try {
            String apiKey = difyConfig.getApiKey(appType);
            if (apiKey == null) {
                throw new RuntimeException("未配置" + appType + "的API密钥");
            }
            
            // 根据Dify API文档，需要在请求体中包含user参数
            Map<String, Object> requestBody = new HashMap<>();
            requestBody.put("user", "system"); // 使用system作为用户ID，因为这是系统调用
            
            String requestBodyJson = objectMapper.writeValueAsString(requestBody);
            
            // 使用最原始的方式发送HTTP请求，避免RestTemplate的消息转换器问题
            java.net.http.HttpClient httpClient = java.net.http.HttpClient.newBuilder()
                .version(java.net.http.HttpClient.Version.HTTP_2)
                .connectTimeout(java.time.Duration.ofSeconds(30))
                .build();
                
            java.net.http.HttpRequest request = java.net.http.HttpRequest.newBuilder()
                .uri(java.net.URI.create(difyConfig.getApiUrl() + "/chat-messages/" + taskId + "/stop"))
                .timeout(java.time.Duration.ofSeconds(60))
                .header("Content-Type", "application/json")
                .header("Authorization", "Bearer " + apiKey)
                .header("Accept", "application/json")
                .POST(java.net.http.HttpRequest.BodyPublishers.ofString(requestBodyJson))
                .build();

            // 根据Dify API文档，使用正确的API端点
            log.info("查询Dify任务状态: {}", difyConfig.getApiUrl() + "/chat-messages/" + taskId + "/stop");

            try {
                // 发送请求并获取响应
                java.net.http.HttpResponse<String> response = httpClient.send(request, 
                    java.net.http.HttpResponse.BodyHandlers.ofString());
                
                int statusCode = response.statusCode();
                String responseBody = response.body();
                
                log.info("Dify API任务状态查询响应状态码: {}", statusCode);
                log.debug("Dify API原始响应: {}", responseBody);
                
                if (statusCode == 200) {
                    // 尝试解析JSON响应
                    Map<String, Object> responseMap = objectMapper.readValue(responseBody, Map.class);
                    
                    // 创建我们自己的响应对象
                    DifyDTO.DifyResponse difyResponse = new DifyDTO.DifyResponse();
                    
                    // 从响应中提取关键信息
                    String result = (String) responseMap.get("result");
                    if ("success".equals(result)) {
                        difyResponse.setStatus("completed");
                    } else {
                        difyResponse.setStatus("failed");
                    }
                    
                    log.info("Dify任务状态查询成功: {}", difyResponse.getStatus());
                    return difyResponse;
                } else {
                    log.error("Dify任务状态查询失败: HTTP状态码 {}, 响应: {}", statusCode, responseBody);
                    
                    // 创建一个失败的响应对象
                    DifyDTO.DifyResponse failureResponse = new DifyDTO.DifyResponse();
                    failureResponse.setStatus("failed");
                    failureResponse.setError("任务状态查询失败: HTTP状态码 " + statusCode);
                    return failureResponse;
                }
            } catch (Exception e) {
                log.error("查询任务状态失败: {}", e.getMessage());
                // 创建一个失败的响应对象
                DifyDTO.DifyResponse failureResponse = new DifyDTO.DifyResponse();
                failureResponse.setStatus("failed");
                failureResponse.setError("查询任务状态失败: " + e.getMessage());
                return failureResponse;
            }
        } catch (Exception e) {
            log.error("查询Dify任务状态异常: {}", e.getMessage(), e);
            throw new RuntimeException("查询AI任务状态失败: " + e.getMessage());
        }
    }

    /**
     * 异步调用Dify工作流API
     */
    public CompletableFuture<DifyDTO.DifyResponse> callWorkflowApiAsync(String appType, Map<String, Object> inputs, String userId) {
        return CompletableFuture.supplyAsync(() -> {
            try {
                // 直接调用已修改的同步方法，不需要额外修改inputs
                return callWorkflowApi(appType, inputs, userId);
            } catch (Exception e) {
                log.error("异步调用Dify API失败: {}", e.getMessage(), e);
                throw new RuntimeException(e);
            }
        });
    }

    /**
     * 组卷功能
     */
    public DifyDTO.PaperGenerationResponse generatePaper(DifyDTO.PaperGenerationRequest request, String userId) {
        try {
            log.info("开始生成试卷，请求: {}", request);

            // 构建输入参数
            Map<String, Object> inputs = new HashMap<>();
            inputs.put("course_id", request.getCourseId());
            
            // 将知识点作为数组传递，而不是逗号分隔的字符串
            inputs.put("knowledge_points", request.getKnowledgePoints());
            
            inputs.put("difficulty", request.getDifficulty());
            inputs.put("question_count", request.getQuestionCount());
            inputs.put("question_types", objectMapper.writeValueAsString(request.getQuestionTypes()));
            inputs.put("duration", request.getDuration());
            inputs.put("total_score", request.getTotalScore());
            
            // 构建更详细的提示，要求生成有意义的选项内容
            StringBuilder promptBuilder = new StringBuilder();
            if (request.getAdditionalRequirements() != null && !request.getAdditionalRequirements().isEmpty()) {
                promptBuilder.append(request.getAdditionalRequirements()).append("\n\n");
            }
            
            // 添加结构化的提示
            promptBuilder.append("### 试卷生成要求\n");
            promptBuilder.append("- 难度级别: ").append(request.getDifficulty()).append("\n");
            promptBuilder.append("- 题目数量: ").append(request.getQuestionCount()).append("\n");
            promptBuilder.append("- 题型分布: ").append(objectMapper.writeValueAsString(request.getQuestionTypes())).append("\n");
            promptBuilder.append("- 总分: ").append(request.getTotalScore()).append("\n");
            promptBuilder.append("- 考试时长: ").append(request.getDuration()).append("分钟\n\n");

            promptBuilder.append("### 内容要求\n");
            promptBuilder.append("1. 生成的题目必须与指定知识点直接相关\n");
            promptBuilder.append("2. 选项内容必须具体、有意义，与题目主题紧密相关\n");
            promptBuilder.append("3. 严禁使用'Option A'、'Option B'这样的通用占位符\n");
            promptBuilder.append("4. 选项应包含实际的概念、定义、公式或例子\n");
            promptBuilder.append("5. 单选题和多选题的选项之间必须有明确的区分度\n");
            promptBuilder.append("6. 判断题的陈述必须清晰明确，能够明确判断真假\n\n");

            // 添加知识点特定的上下文信息
            if (request.getKnowledgePoints() != null && !request.getKnowledgePoints().isEmpty()) {
                promptBuilder.append("### 知识点上下文\n");
                
                for (String knowledgePoint : request.getKnowledgePoints()) {
                    if (knowledgePoint.equalsIgnoreCase("section-14") || knowledgePoint.contains("计算机组成")) {
                        promptBuilder.append("【").append(knowledgePoint).append("】计算机组成原理核心内容：\n");
                        promptBuilder.append("- CPU结构与功能：包括运算器、控制器、寄存器等组成部分\n");
                        promptBuilder.append("- 存储系统：主存储器、Cache、虚拟存储等\n");
                        promptBuilder.append("- 指令系统：指令格式、寻址方式、指令执行过程\n");
                        promptBuilder.append("- 总线结构：数据总线、地址总线、控制总线\n");
                        promptBuilder.append("- I/O系统：I/O接口、I/O方式（程序查询、中断、DMA）\n\n");
                    } 
                    else if (knowledgePoint.equalsIgnoreCase("section-15") || knowledgePoint.contains("操作系统")) {
                        promptBuilder.append("【").append(knowledgePoint).append("】操作系统核心内容：\n");
                        promptBuilder.append("- 进程管理：进程概念、进程状态转换、进程调度算法\n");
                        promptBuilder.append("- 内存管理：内存分配策略、分页、分段、虚拟内存\n");
                        promptBuilder.append("- 文件系统：文件组织、目录结构、文件操作\n");
                        promptBuilder.append("- 设备管理：I/O软件层次、设备分配、缓冲区管理\n");
                        promptBuilder.append("- 死锁处理：死锁的条件、预防、避免、检测和恢复\n\n");
                    }
                    else if (knowledgePoint.equalsIgnoreCase("section-18") || knowledgePoint.contains("数据结构")) {
                        promptBuilder.append("【").append(knowledgePoint).append("】数据结构核心内容：\n");
                        promptBuilder.append("- 线性结构：数组、链表、栈、队列的特点与操作\n");
                        promptBuilder.append("- 树结构：二叉树、平衡树、B树、红黑树等\n");
                        promptBuilder.append("- 图结构：图的表示、图的遍历、最短路径算法\n");
                        promptBuilder.append("- 查找算法：顺序查找、二分查找、哈希查找\n");
                        promptBuilder.append("- 排序算法：冒泡排序、快速排序、归并排序等\n\n");
                    }
                    else if (knowledgePoint.equalsIgnoreCase("section-20") || knowledgePoint.contains("数据库")) {
                        promptBuilder.append("【").append(knowledgePoint).append("】数据库系统核心内容：\n");
                        promptBuilder.append("- 关系数据库：关系模型、关系代数、规范化理论\n");
                        promptBuilder.append("- SQL语言：DDL、DML、DCL、查询语句\n");
                        promptBuilder.append("- 数据库设计：E-R模型、逻辑设计、物理设计\n");
                        promptBuilder.append("- 事务处理：ACID特性、并发控制、恢复技术\n");
                        promptBuilder.append("- 索引与优化：B+树索引、哈希索引、查询优化\n\n");
                    }
                    else {
                        promptBuilder.append("【").append(knowledgePoint).append("】请基于课程相关内容生成题目\n\n");
                    }
                }
            }

            // 添加输出格式要求
            promptBuilder.append("### 输出格式要求\n");
            promptBuilder.append("请直接生成JSON格式的试卷，包含以下结构：\n");
            promptBuilder.append("```json\n");
            promptBuilder.append("{\n");
            promptBuilder.append("  \"exam_paper\": {\n");
            promptBuilder.append("    \"title\": \"试卷标题\",\n");
            promptBuilder.append("    \"course_id\": 课程ID,\n");
            promptBuilder.append("    \"total_score\": 总分,\n");
            promptBuilder.append("    \"duration\": 考试时长,\n");
            promptBuilder.append("    \"question_count\": 题目数量,\n");
            promptBuilder.append("    \"knowledge_points\": [\"知识点1\", \"知识点2\"],\n");
            promptBuilder.append("    \"questions\": [\n");
            promptBuilder.append("      {\n");
            promptBuilder.append("        \"type\": \"SINGLE_CHOICE\",\n");
            promptBuilder.append("        \"score\": 分值,\n");
            promptBuilder.append("        \"content\": \"题目内容\",\n");
            promptBuilder.append("        \"options\": [\"具体选项A\", \"具体选项B\", \"具体选项C\", \"具体选项D\"],\n");
            promptBuilder.append("        \"answer\": \"正确答案\"\n");
            promptBuilder.append("      },\n");
            promptBuilder.append("      // 更多题目...\n");
            promptBuilder.append("    ]\n");
            promptBuilder.append("  }\n");
            promptBuilder.append("}\n");
            promptBuilder.append("```\n\n");

            promptBuilder.append("请直接生成最终结果，不要包含思考过程，不要使用任何标签，仅提供格式化的JSON响应。\n");

            inputs.put("additional_requirements", promptBuilder.toString());
            
            try {
                // 调用Dify API
                DifyDTO.DifyResponse difyResponse = callWorkflowApi("paper-generation", inputs, userId);

                // 解析响应
                if ("completed".equals(difyResponse.getStatus())) {
                    return parsePaperGenerationResponse(difyResponse);
                } else if ("failed".equals(difyResponse.getStatus())) {
                    // 如果API调用失败，记录错误并使用本地模板
                    log.error("AI服务调用失败，将使用本地模板生成试卷: {}", difyResponse.getError());
                    
                    // 构建失败响应，包含错误信息
                    DifyDTO.PaperGenerationResponse failedResponse = DifyDTO.PaperGenerationResponse.builder()
                            .status("failed")
                            .errorMessage(difyResponse.getError() != null ? 
                                difyResponse.getError() : "AI服务调用失败，将使用本地模板")
                            .build();
                    
                    return failedResponse;
                } else {
                    return DifyDTO.PaperGenerationResponse.builder()
                            .status(difyResponse.getStatus())
                            .taskId(difyResponse.getTaskId())
                            .build();
                }
            } catch (Exception e) {
                log.error("AI服务调用失败，将使用本地模板生成试卷: {}", e.getMessage());
                
                // 构建失败响应，包含错误信息
                DifyDTO.PaperGenerationResponse failedResponse = DifyDTO.PaperGenerationResponse.builder()
                        .status("failed")
                        .errorMessage("AI服务调用失败，将使用本地模板")
                        .build();
                
                return failedResponse;
            }
        } catch (Exception e) {
            log.error("生成试卷过程中发生异常: {}", e.getMessage(), e);
            return DifyDTO.PaperGenerationResponse.builder()
                    .status("failed")
                    .errorMessage("生成试卷失败: " + e.getMessage())
                    .build();
        }
    }

    /**
     * 使用本地模板生成试卷
     */
    public DifyDTO.PaperGenerationResponse generateLocalPaperTemplate(DifyDTO.PaperGenerationRequest request) {
        try {
            log.info("使用本地模板生成试卷，课程ID: {}", request.getCourseId());
            
            // 根据课程ID选择不同的模板
            String title = "";
            List<DifyDTO.GeneratedQuestion> questions = new ArrayList<>();
            
            switch(request.getCourseId().intValue()) {
                case 19: // Java程序设计
                    title = "Java程序设计期末考试";
                    questions = generateJavaQuestions(request.getDifficulty(), request.getQuestionTypes());
                    break;
                case 20: // 数据结构与算法
                    title = "数据结构与算法期末考试";
                    questions = generateDataStructureQuestions(request.getDifficulty(), request.getQuestionTypes());
                    break;
                case 21: // Python程序设计
                    title = "Python程序设计期末考试";
                    questions = generatePythonQuestions(request.getDifficulty(), request.getQuestionTypes());
                    break;
                default:
                    title = "课程期末考试";
                    questions = generateDefaultQuestions(request.getDifficulty(), request.getQuestionTypes());
            }
            
            // 构建响应
            return DifyDTO.PaperGenerationResponse.builder()
                    .title(title)
                    .questions(questions)
                    .status("completed")
                    .build();
            
        } catch (Exception e) {
            log.error("本地模板生成失败: {}", e.getMessage(), e);
            return DifyDTO.PaperGenerationResponse.builder()
                    .status("failed")
                    .errorMessage("本地模板生成失败: " + e.getMessage())
                    .build();
        }
    }
    
    /**
     * 生成Java题目
     */
    private List<DifyDTO.GeneratedQuestion> generateJavaQuestions(String difficulty, Map<String, Integer> questionTypes) {
        List<DifyDTO.GeneratedQuestion> questions = new ArrayList<>();
        
        // 单选题
        if (questionTypes.containsKey("SINGLE_CHOICE") && questionTypes.get("SINGLE_CHOICE") > 0) {
            int count = questionTypes.get("SINGLE_CHOICE");
            for (int i = 0; i < count; i++) {
                DifyDTO.GeneratedQuestion question = DifyDTO.GeneratedQuestion.builder()
                        .questionText(i == 0 ? "Java中，以下哪个关键字用于继承？" : 
                                     i == 1 ? "以下哪个不是Java的基本数据类型？" :
                                     "Java中，以下哪个修饰符表示类只能在同一个包中访问？")
                        .questionType("SINGLE_CHOICE")
                        .options(i == 0 ? Arrays.asList("extends", "implements", "inherits", "extends from") :
                                i == 1 ? Arrays.asList("int", "boolean", "String", "char") :
                                Arrays.asList("public", "protected", "private", "default"))
                        .correctAnswer(i == 0 ? "extends" : i == 1 ? "String" : "default")
                        .score(2)
                        .knowledgePoint(i == 0 ? "Java基础语法" : i == 1 ? "Java数据类型" : "Java访问修饰符")
                        .difficulty(difficulty)
                        .explanation(i == 0 ? "在Java中，extends关键字用于类的继承，表示一个类继承另一个类的特性。" :
                                   i == 1 ? "String是引用类型，不是基本数据类型。Java的基本数据类型有byte、short、int、long、float、double、char和boolean。" :
                                   "default（默认）修饰符表示在同一个包内可见，不使用任何修饰符时即为default访问级别。")
                        .build();
                questions.add(question);
            }
        }
        
        // 多选题
        if (questionTypes.containsKey("MULTIPLE_CHOICE") && questionTypes.get("MULTIPLE_CHOICE") > 0) {
            int count = questionTypes.get("MULTIPLE_CHOICE");
            for (int i = 0; i < count; i++) {
                DifyDTO.GeneratedQuestion question = DifyDTO.GeneratedQuestion.builder()
                        .questionText(i == 0 ? "以下哪些是Java中的集合框架接口？" :
                                     "以下哪些是Java中的线程安全的集合类？")
                        .questionType("MULTIPLE_CHOICE")
                        .options(i == 0 ? Arrays.asList("List", "Map", "Queue", "Array") :
                                Arrays.asList("ArrayList", "Vector", "CopyOnWriteArrayList", "HashMap"))
                        .correctAnswer(i == 0 ? "List,Map,Queue" : "Vector,CopyOnWriteArrayList")
                        .score(4)
                        .knowledgePoint(i == 0 ? "Java集合框架" : "Java并发编程")
                        .difficulty(difficulty)
                        .explanation(i == 0 ? "List、Map和Queue都是Java集合框架中的接口，而Array是Java的数组类型，不是集合框架接口。" :
                                   "Vector和CopyOnWriteArrayList是线程安全的集合类，而ArrayList和HashMap不是线程安全的。")
                        .build();
                questions.add(question);
            }
        }
        
        // 判断题
        if (questionTypes.containsKey("TRUE_FALSE") && questionTypes.get("TRUE_FALSE") > 0) {
            int count = questionTypes.get("TRUE_FALSE");
            for (int i = 0; i < count; i++) {
                DifyDTO.GeneratedQuestion question = DifyDTO.GeneratedQuestion.builder()
                        .questionText(i == 0 ? "Java中的接口可以包含默认方法实现。" :
                                     "Java中的字符串是不可变的。")
                        .questionType("TRUE_FALSE")
                        .correctAnswer(i == 0 ? "true" : "true")
                        .score(2)
                        .knowledgePoint(i == 0 ? "Java接口" : "Java字符串")
                        .difficulty(difficulty)
                        .explanation(i == 0 ? "Java 8及以后版本中，接口可以包含默认方法实现，使用default关键字。" :
                                   "在Java中，String类的对象是不可变的，这意味着一旦创建，其值就不能被修改。")
                        .build();
                questions.add(question);
            }
        }
        
        // 填空题
        if (questionTypes.containsKey("FILL_BLANK") && questionTypes.get("FILL_BLANK") > 0) {
            int count = questionTypes.get("FILL_BLANK");
            for (int i = 0; i < count; i++) {
                DifyDTO.GeneratedQuestion question = DifyDTO.GeneratedQuestion.builder()
                        .questionText("Java中，用于处理异常的关键字有try、catch、finally、throw和_____。")
                        .questionType("FILL_BLANK")
                        .correctAnswer("throws")
                        .score(3)
                        .knowledgePoint("Java异常处理")
                        .difficulty(difficulty)
                        .explanation("throws关键字用于在方法签名中声明该方法可能抛出的异常类型。")
                        .build();
                questions.add(question);
            }
        }
        
        // 简答题
        if (questionTypes.containsKey("ESSAY") && questionTypes.get("ESSAY") > 0) {
            int count = questionTypes.get("ESSAY");
            for (int i = 0; i < count; i++) {
                DifyDTO.GeneratedQuestion question = DifyDTO.GeneratedQuestion.builder()
                        .questionText("请简述Java中的多线程实现方式及其区别。")
                        .questionType("ESSAY")
                        .correctAnswer("在Java中实现多线程有两种主要方式：\n1. 继承Thread类并重写run()方法\n2. 实现Runnable接口并实现run()方法\n\n区别：\n- 继承Thread类的方式不支持多重继承，而实现Runnable接口的方式可以继承其他类\n- 实现Runnable接口的方式更适合多个线程共享同一个目标对象的情况\n- 实现Runnable接口的方式可以更好地体现面向对象的设计思想，将线程的控制和业务逻辑分离")
                        .score(10)
                        .knowledgePoint("Java多线程")
                        .difficulty(difficulty)
                        .explanation("这个问题考察学生对Java多线程基础概念的理解，包括实现方式和各自的优缺点。")
                        .build();
                questions.add(question);
            }
        }
        
        return questions;
    }
    
    /**
     * 生成数据结构题目
     */
    private List<DifyDTO.GeneratedQuestion> generateDataStructureQuestions(String difficulty, Map<String, Integer> questionTypes) {
        List<DifyDTO.GeneratedQuestion> questions = new ArrayList<>();
        
        // 单选题
        if (questionTypes.containsKey("SINGLE_CHOICE") && questionTypes.get("SINGLE_CHOICE") > 0) {
            int count = questionTypes.get("SINGLE_CHOICE");
            for (int i = 0; i < count; i++) {
                DifyDTO.GeneratedQuestion question = DifyDTO.GeneratedQuestion.builder()
                        .questionText(i == 0 ? "以下哪种数据结构是线性的？" : 
                                     i == 1 ? "快速排序的平均时间复杂度是？" :
                                     "二叉树的前序遍历顺序是？")
                        .questionType("SINGLE_CHOICE")
                        .options(i == 0 ? Arrays.asList("树", "图", "栈", "二叉树") :
                                i == 1 ? Arrays.asList("O(n)", "O(n log n)", "O(n²)", "O(1)") :
                                Arrays.asList("根-左-右", "左-根-右", "左-右-根", "根-右-左"))
                        .correctAnswer(i == 0 ? "栈" : i == 1 ? "O(n log n)" : "根-左-右")
                        .score(2)
                        .knowledgePoint(i == 0 ? "数据结构基础" : i == 1 ? "排序算法" : "树的遍历")
                        .difficulty(difficulty)
                        .explanation(i == 0 ? "栈是一种线性数据结构，而树、图和二叉树都是非线性数据结构。" :
                                   i == 1 ? "快速排序的平均时间复杂度是O(n log n)。" :
                                   "二叉树的前序遍历顺序是先访问根节点，然后递归地前序遍历左子树，最后递归地前序遍历右子树。")
                        .build();
                questions.add(question);
            }
        }
        
        // 多选题
        if (questionTypes.containsKey("MULTIPLE_CHOICE") && questionTypes.get("MULTIPLE_CHOICE") > 0) {
            int count = questionTypes.get("MULTIPLE_CHOICE");
            for (int i = 0; i < count; i++) {
                DifyDTO.GeneratedQuestion question = DifyDTO.GeneratedQuestion.builder()
                        .questionText(i == 0 ? "以下哪些排序算法的平均时间复杂度是O(n log n)？" :
                                     "以下哪些数据结构可以用于实现优先队列？")
                        .questionType("MULTIPLE_CHOICE")
                        .options(i == 0 ? Arrays.asList("快速排序", "冒泡排序", "归并排序", "插入排序") :
                                Arrays.asList("数组", "链表", "堆", "二叉搜索树"))
                        .correctAnswer(i == 0 ? "快速排序,归并排序" : "堆,二叉搜索树")
                        .score(4)
                        .knowledgePoint(i == 0 ? "排序算法" : "优先队列")
                        .difficulty(difficulty)
                        .explanation(i == 0 ? "快速排序和归并排序的平均时间复杂度是O(n log n)，冒泡排序和插入排序的平均时间复杂度是O(n²)。" :
                                   "堆和二叉搜索树可以有效地实现优先队列，支持快速的插入和删除最大/最小元素操作。")
                        .build();
                questions.add(question);
            }
        }
        
        // 判断题
        if (questionTypes.containsKey("TRUE_FALSE") && questionTypes.get("TRUE_FALSE") > 0) {
            int count = questionTypes.get("TRUE_FALSE");
            for (int i = 0; i < count; i++) {
                DifyDTO.GeneratedQuestion question = DifyDTO.GeneratedQuestion.builder()
                        .questionText(i == 0 ? "在最坏情况下，快速排序的时间复杂度是O(n²)。" :
                                     "哈希表的查找操作的平均时间复杂度是O(1)。")
                        .questionType("TRUE_FALSE")
                        .correctAnswer(i == 0 ? "true" : "true")
                        .score(2)
                        .knowledgePoint(i == 0 ? "排序算法" : "哈希表")
                        .difficulty(difficulty)
                        .explanation(i == 0 ? "快速排序在最坏情况下（如已排序数组）的时间复杂度是O(n²)。" :
                                   "哈希表在理想情况下（没有冲突）的查找操作时间复杂度是O(1)，即常数时间。")
                        .build();
                questions.add(question);
            }
        }
        
        return questions;
    }
    
    /**
     * 生成Python题目
     */
    private List<DifyDTO.GeneratedQuestion> generatePythonQuestions(String difficulty, Map<String, Integer> questionTypes) {
        List<DifyDTO.GeneratedQuestion> questions = new ArrayList<>();
        
        // 单选题
        if (questionTypes.containsKey("SINGLE_CHOICE") && questionTypes.get("SINGLE_CHOICE") > 0) {
            int count = questionTypes.get("SINGLE_CHOICE");
            for (int i = 0; i < count; i++) {
                DifyDTO.GeneratedQuestion question = DifyDTO.GeneratedQuestion.builder()
                        .questionText(i == 0 ? "Python中，以下哪种数据类型是不可变的？" : 
                                     i == 1 ? "Python中，以下哪个不是列表的方法？" :
                                     "Python中，用于定义函数的关键字是？")
                        .questionType("SINGLE_CHOICE")
                        .options(i == 0 ? Arrays.asList("列表(list)", "字典(dict)", "集合(set)", "元组(tuple)") :
                                i == 1 ? Arrays.asList("append()", "extend()", "keys()", "pop()") :
                                Arrays.asList("func", "define", "def", "function"))
                        .correctAnswer(i == 0 ? "元组(tuple)" : i == 1 ? "keys()" : "def")
                        .score(2)
                        .knowledgePoint(i == 0 ? "Python数据类型" : i == 1 ? "Python列表操作" : "Python函数定义")
                        .difficulty(difficulty)
                        .explanation(i == 0 ? "在Python中，元组(tuple)是不可变的数据类型，而列表(list)、字典(dict)和集合(set)都是可变的。" :
                                   i == 1 ? "keys()是字典的方法，不是列表的方法。append()、extend()和pop()都是列表的方法。" :
                                   "Python中使用def关键字来定义函数。")
                        .build();
                questions.add(question);
            }
        }
        
        // 多选题
        if (questionTypes.containsKey("MULTIPLE_CHOICE") && questionTypes.get("MULTIPLE_CHOICE") > 0) {
            int count = questionTypes.get("MULTIPLE_CHOICE");
            for (int i = 0; i < count; i++) {
                DifyDTO.GeneratedQuestion question = DifyDTO.GeneratedQuestion.builder()
                        .questionText(i == 0 ? "以下哪些是Python的内置函数？" :
                                     "以下哪些是Python中的可迭代对象？")
                        .questionType("MULTIPLE_CHOICE")
                        .options(i == 0 ? Arrays.asList("map()", "reduce()", "filter()", "foreach()") :
                                Arrays.asList("列表", "元组", "字典", "整数"))
                        .correctAnswer(i == 0 ? "map(),filter()" : "列表,元组,字典")
                        .score(4)
                        .knowledgePoint(i == 0 ? "Python内置函数" : "Python迭代器")
                        .difficulty(difficulty)
                        .explanation(i == 0 ? "map()和filter()是Python的内置函数，而reduce()在Python 3中被移到functools模块中，foreach()不是Python的内置函数。" :
                                   "列表、元组和字典都是可迭代对象，而整数不是可迭代对象。")
                        .build();
                questions.add(question);
            }
        }
        
        return questions;
    }
    
    /**
     * 生成默认题目
     */
    private List<DifyDTO.GeneratedQuestion> generateDefaultQuestions(String difficulty, Map<String, Integer> questionTypes) {
        List<DifyDTO.GeneratedQuestion> questions = new ArrayList<>();
        
        // 单选题
        if (questionTypes.containsKey("SINGLE_CHOICE") && questionTypes.get("SINGLE_CHOICE") > 0) {
            int count = questionTypes.get("SINGLE_CHOICE");
            for (int i = 0; i < count; i++) {
                DifyDTO.GeneratedQuestion question = DifyDTO.GeneratedQuestion.builder()
                        .questionText("单选题示例 " + (i+1))
                        .questionType("SINGLE_CHOICE")
                        .options(Arrays.asList("选项A", "选项B", "选项C", "选项D"))
                        .correctAnswer("选项A")
                        .score(2)
                        .knowledgePoint("基础知识点")
                        .difficulty(difficulty)
                        .explanation("这是一个单选题示例。")
                        .build();
                questions.add(question);
            }
        }
        
        // 多选题
        if (questionTypes.containsKey("MULTIPLE_CHOICE") && questionTypes.get("MULTIPLE_CHOICE") > 0) {
            int count = questionTypes.get("MULTIPLE_CHOICE");
            for (int i = 0; i < count; i++) {
                DifyDTO.GeneratedQuestion question = DifyDTO.GeneratedQuestion.builder()
                        .questionText("多选题示例 " + (i+1))
                        .questionType("MULTIPLE_CHOICE")
                        .options(Arrays.asList("选项A", "选项B", "选项C", "选项D"))
                        .correctAnswer("选项A,选项C")
                        .score(4)
                        .knowledgePoint("基础知识点")
                        .difficulty(difficulty)
                        .explanation("这是一个多选题示例。")
                        .build();
                questions.add(question);
            }
        }
        
        // 判断题
        if (questionTypes.containsKey("TRUE_FALSE") && questionTypes.get("TRUE_FALSE") > 0) {
            int count = questionTypes.get("TRUE_FALSE");
            for (int i = 0; i < count; i++) {
                DifyDTO.GeneratedQuestion question = DifyDTO.GeneratedQuestion.builder()
                        .questionText("判断题示例 " + (i+1))
                        .questionType("TRUE_FALSE")
                        .correctAnswer(i % 2 == 0 ? "true" : "false")
                        .score(2)
                        .knowledgePoint("基础知识点")
                        .difficulty(difficulty)
                        .explanation("这是一个判断题示例。")
                        .build();
                questions.add(question);
            }
        }
        
        return questions;
    }

    /**
     * 智能批改单份作业
     */
    public DifyDTO.AutoGradingResponse gradeAssignment(DifyDTO.AutoGradingRequest request, String userId) {
        return autoGrading(request, userId);
    }

    /**
     * 批量智能批改
     */
    public List<DifyDTO.AutoGradingResponse> batchGradeAssignments(List<DifyDTO.AutoGradingRequest> requests, String userId) {
        List<DifyDTO.AutoGradingResponse> responses = new ArrayList<>();
        
        for (DifyDTO.AutoGradingRequest request : requests) {
            try {
                DifyDTO.AutoGradingResponse response = autoGrading(request, userId);
                responses.add(response);
                
                // 避免并发过高，添加短暂延迟
                Thread.sleep(100);
                
            } catch (Exception e) {
                log.error("批改作业{}失败: {}", request.getSubmissionId(), e.getMessage());
                
                // 添加失败的响应
                DifyDTO.AutoGradingResponse failedResponse = DifyDTO.AutoGradingResponse.builder()
                        .status("failed")
                        .errorMessage(e.getMessage())
                        .build();
                responses.add(failedResponse);
            }
        }
        
        return responses;
    }

    /**
     * 自动批改功能
     */
    public DifyDTO.AutoGradingResponse autoGrading(DifyDTO.AutoGradingRequest request, String userId) {
        try {
            log.info("开始自动批改，请求: {}", request);

            // 构建输入参数
            Map<String, Object> inputs = new HashMap<>();
            inputs.put("assignment_id", request.getAssignmentId());
            inputs.put("student_id", request.getStudentId());
            inputs.put("answers", objectMapper.writeValueAsString(request.getAnswers()));
            inputs.put("grading_type", request.getGradingType());
            inputs.put("grading_criteria", request.getGradingCriteria());

            // 调用Dify API
            DifyDTO.DifyResponse difyResponse = callWorkflowApi("auto-grading", inputs, userId);

            // 解析响应
            if ("completed".equals(difyResponse.getStatus())) {
                return parseAutoGradingResponse(difyResponse);
            } else {
                return DifyDTO.AutoGradingResponse.builder()
                        .status(difyResponse.getStatus())
                        .taskId(difyResponse.getTaskId())
                        .build();
            }

        } catch (Exception e) {
            log.error("自动批改失败: {}", e.getMessage(), e);
            return DifyDTO.AutoGradingResponse.builder()
                    .status("failed")
                    .build();
        }
    }

    /**
     * 解析组卷响应
     */
    private DifyDTO.PaperGenerationResponse parsePaperGenerationResponse(DifyDTO.DifyResponse difyResponse) {
        try {
            Map<String, Object> data = difyResponse.getData();
            if (data == null || data.isEmpty()) {
                log.warn("组卷响应数据为空");
                return DifyDTO.PaperGenerationResponse.builder()
                        .status("failed")
                        .errorMessage("AI服务返回空数据")
                        .build();
            }

            log.info("开始处理AI响应数据");
            String content = "";
            
            // 如果有content字段，使用它
            if (data.containsKey("content")) {
                content = (String) data.get("content");
            } else {
                // 否则将整个data转为JSON字符串
                try {
                    content = objectMapper.writeValueAsString(data);
                } catch (Exception e) {
                    content = "数据转换错误: " + e.getMessage();
                    log.error("转换AI响应数据失败: {}", e.getMessage());
                }
            }
            
            // 尝试从内容中提取JSON部分
            String jsonContent = extractJsonFromContent(content);
            if (jsonContent != null && !jsonContent.isEmpty()) {
                try {
                    // 解析JSON内容
                    Map<String, Object> paperData = objectMapper.readValue(jsonContent, Map.class);
                    
                    // 检查是否有exam_paper字段（新格式）
                    Map<String, Object> examPaperData = paperData;
                    if (paperData.containsKey("exam_paper")) {
                        examPaperData = (Map<String, Object>) paperData.get("exam_paper");
                        log.info("检测到exam_paper格式，使用其中的数据");
                    }
                    
                    // 获取题目列表
                    List<Map<String, Object>> questionsData = null;
                    if (examPaperData.containsKey("questions")) {
                        questionsData = (List<Map<String, Object>>) examPaperData.get("questions");
                    }
                    
                    if (questionsData != null && !questionsData.isEmpty()) {
                        // 解析试卷标题
                        String title = examPaperData.containsKey("title") ? 
                                (String) examPaperData.get("title") : "AI生成试卷";
                        
                        // 解析题目列表
                        List<DifyDTO.GeneratedQuestion> questions = new ArrayList<>();
                        for (Map<String, Object> questionData : questionsData) {
                            DifyDTO.GeneratedQuestion question = parseQuestion(questionData);
                            if (question != null) {
                                questions.add(question);
                            }
                        }
                        
                        log.info("成功解析AI生成的试卷，包含{}道题目", questions.size());
                        
                        return DifyDTO.PaperGenerationResponse.builder()
                                .title(title)
                                .questions(questions)
                                .status("completed")
                                .taskId(difyResponse.getTaskId())
                                .build();
                    }
                } catch (Exception e) {
                    log.error("解析JSON内容失败: {}", e.getMessage(), e);
                    // 如果解析失败，继续使用原始方式处理
                }
            }
            
            // 如果无法解析JSON或没有题目，则使用原始方式处理
            log.warn("无法从AI响应中提取有效的试卷数据，使用原始内容");
            
            // 创建一个题目对象，直接使用AI的输出作为题目内容
            DifyDTO.GeneratedQuestion question = DifyDTO.GeneratedQuestion.builder()
                    .questionText(content)
                    .questionType("AI_OUTPUT")
                    .build();
            
            List<DifyDTO.GeneratedQuestion> questions = new ArrayList<>();
            questions.add(question);

            return DifyDTO.PaperGenerationResponse.builder()
                    .title("AI生成试卷")
                    .questions(questions)
                    .status("completed")
                    .taskId(difyResponse.getTaskId())
                    .build();
        } catch (Exception e) {
            log.error("解析组卷响应失败: {}", e.getMessage(), e);
            return DifyDTO.PaperGenerationResponse.builder()
                    .status("failed")
                    .errorMessage("解析试卷数据失败: " + e.getMessage())
                    .build();
        }
    }

    /**
     * 从内容中提取JSON部分
     */
    private String extractJsonFromContent(String content) {
        try {
            // 移除<think>标签及其内容
            content = content.replaceAll("(?s)<think>.*?</think>", "");
            
            // 查找JSON部分（通常在```json和```之间）
            java.util.regex.Pattern pattern = java.util.regex.Pattern.compile("```json\\s*(.*?)\\s*```", java.util.regex.Pattern.DOTALL);
            java.util.regex.Matcher matcher = pattern.matcher(content);
            
            if (matcher.find()) {
                return matcher.group(1).trim();
            }
            
            // 如果没有找到```json标记，尝试查找可能的JSON对象
            if (content.trim().startsWith("{") && content.trim().endsWith("}")) {
                return content.trim();
            }
            
            return null;
        } catch (Exception e) {
            log.error("提取JSON内容失败: {}", e.getMessage(), e);
            return null;
        }
    }

    /**
     * 解析题目数据
     */
    private DifyDTO.GeneratedQuestion parseQuestion(Map<String, Object> questionData) {
        try {
            // 获取题目类型，支持type或questionType字段
            String questionType = questionData.containsKey("type") ? 
                    (String) questionData.get("type") : 
                    (String) questionData.get("questionType");
            
            // 获取题目内容，支持content或questionText字段
            String content = questionData.containsKey("content") ? 
                    (String) questionData.get("content") : 
                    (String) questionData.get("questionText");
            
            // 获取分数，支持score字段
            Object scoreObj = questionData.get("score");
            Integer score = scoreObj instanceof Integer ? (Integer) scoreObj : 
                            scoreObj instanceof Number ? ((Number) scoreObj).intValue() : null;
            
            // 解析选项，支持options字段
            List<String> options = null;
            if (questionData.containsKey("options")) {
                Object optionsObj = questionData.get("options");
                if (optionsObj instanceof List) {
                    options = (List<String>) optionsObj;
                }
            }
            
            // 解析答案，支持answer、correct_answer、correctAnswer和correct_answers字段
            String correctAnswer = null;
            if (questionData.containsKey("answer")) {
                Object answerObj = questionData.get("answer");
                correctAnswer = parseAnswerObject(answerObj);
            } else if (questionData.containsKey("correct_answer")) {
                Object answerObj = questionData.get("correct_answer");
                correctAnswer = parseAnswerObject(answerObj);
            } else if (questionData.containsKey("correctAnswer")) {
                Object answerObj = questionData.get("correctAnswer");
                correctAnswer = parseAnswerObject(answerObj);
            } else if (questionData.containsKey("correct_answers")) {
                Object answerObj = questionData.get("correct_answers");
                correctAnswer = parseAnswerObject(answerObj);
            }
            
            // 获取知识点，支持knowledgePoint或knowledge_point字段
            String knowledgePoint = null;
            if (questionData.containsKey("knowledgePoint")) {
                knowledgePoint = (String) questionData.get("knowledgePoint");
            } else if (questionData.containsKey("knowledge_point")) {
                knowledgePoint = (String) questionData.get("knowledge_point");
            }
            
            // 获取难度，支持difficulty字段
            String difficulty = null;
            if (questionData.containsKey("difficulty")) {
                difficulty = (String) questionData.get("difficulty");
            }
            
            // 获取解析，支持explanation字段
            String explanation = null;
            if (questionData.containsKey("explanation")) {
                explanation = (String) questionData.get("explanation");
            }
            
            return DifyDTO.GeneratedQuestion.builder()
                    .questionText(content)
                    .questionType(questionType)
                    .options(options)
                    .correctAnswer(correctAnswer)
                    .score(score)
                    .knowledgePoint(knowledgePoint)
                    .difficulty(difficulty)
                    .explanation(explanation)
                    .build();
        } catch (Exception e) {
            log.error("解析题目数据失败: {}", e.getMessage(), e);
            return null;
        }
    }

    /**
     * 解析答案对象，支持不同格式的答案
     */
    private String parseAnswerObject(Object answerObj) {
        if (answerObj == null) {
            return null;
        }
        
        if (answerObj instanceof String) {
            return (String) answerObj;
        } else if (answerObj instanceof List) {
            // 如果答案是列表（如多选题），将其转换为逗号分隔的字符串
            return String.join(",", (List<String>) answerObj);
        } else if (answerObj instanceof Boolean) {
            // 如果答案是布尔值（如判断题），转换为字符串
            return ((Boolean) answerObj) ? "True" : "False";
        } else {
            // 其他类型，尝试转换为字符串
            return String.valueOf(answerObj);
        }
    }
    
    /**
     * 根据题目类型获取默认分数
     */
    private int getDefaultScoreByQuestionType(String questionType) {
        switch (questionType) {
            case "SINGLE_CHOICE": return 2; // 单选题默认2分
            case "MULTIPLE_CHOICE": return 4; // 多选题默认4分
            case "TRUE_FALSE": return 2; // 判断题默认2分
            case "FILL_BLANK": return 3; // 填空题默认3分
            case "SHORT_ANSWER": return 5; // 简答题默认5分
            case "ESSAY": return 10; // 论述题默认10分
            default: return 3; // 其他类型默认3分
        }
    }
    
    /**
     * 标准化题目类型
     */
    private String normalizeQuestionType(String type) {
        if (type == null) return "SHORT_ANSWER";
        
        String upperType = type.toUpperCase();
        
        // 匹配单选题
        if (upperType.contains("SINGLE") || upperType.equals("CHOICE") || 
            upperType.contains("选择") && !upperType.contains("多选")) {
            return "SINGLE_CHOICE";
        }
        
        // 匹配多选题
        if (upperType.contains("MULTIPLE") || upperType.contains("MULTI") || upperType.contains("多选")) {
            return "MULTIPLE_CHOICE";
        }
        
        // 匹配判断题
        if (upperType.contains("TRUE") || upperType.contains("FALSE") || 
            upperType.equals("TF") || upperType.contains("判断")) {
            return "TRUE_FALSE";
        }
        
        // 匹配填空题
        if (upperType.contains("FILL") || upperType.contains("BLANK") || upperType.contains("填空")) {
            return "FILL_BLANK";
        }
        
        // 匹配简答题
        if (upperType.contains("SHORT") || upperType.contains("简答")) {
            return "SHORT_ANSWER";
        }
        
        // 匹配论述题
        if (upperType.contains("ESSAY") || upperType.contains("LONG") || 
            upperType.contains("论述") || upperType.contains("论文")) {
            return "ESSAY";
        }
        
        // 匹配编程题
        if (upperType.contains("CODE") || upperType.contains("PROGRAM") || 
            upperType.contains("编程") || upperType.contains("代码")) {
            return "CODING";
        }
        
        // 匹配计算题
        if (upperType.contains("CALCULATION") || upperType.contains("COMPUTE") || 
            upperType.contains("计算")) {
            return "CALCULATION";
        }
        
        // 匹配其他通知类型
        if (upperType.contains("NOTICE") || upperType.contains("INFO") || 
            upperType.contains("通知") || upperType.contains("提示")) {
            return "NOTICE";
        }
        
        // 默认返回简答题
        return "SHORT_ANSWER";
    }
    
    /**
     * 根据题目内容和属性推断题目类型
     */
    private String inferQuestionType(Map<String, Object> questionData, String questionText) {
        // 检查是否有选项，如果有很可能是单选或多选
        if (questionData.containsKey("options") || questionData.containsKey("choices")) {
            // 检查是否有明确表示多选的信息
            Object answerObj = null;
            for (String key : Arrays.asList("correct_answer", "correctAnswer", "answer", "key")) {
                if (questionData.containsKey(key)) {
                    answerObj = questionData.get(key);
                    break;
                }
            }
            
            if (answerObj instanceof List || 
                (answerObj instanceof String && ((String)answerObj).contains(","))) {
                return "MULTIPLE_CHOICE"; // 多个答案，应该是多选题
            }
            return "SINGLE_CHOICE"; // 默认为单选题
        }
        
        // 检查题目文本特征
        if (questionText != null) {
            String text = questionText.toLowerCase();
            
            // 判断题通常包含"判断"或是非常简短的陈述句
            if (text.contains("判断") || text.contains("正确") || text.contains("错误") ||
                text.contains("是非") || text.contains("true") || text.contains("false")) {
                return "TRUE_FALSE";
            }
            
            // 填空题通常包含空格或下划线
            if (text.contains("_____") || text.contains("____") || text.contains("___") || 
                text.contains("填空") || text.contains("补充")) {
                return "FILL_BLANK";
            }
            
            // 编程题通常包含编写代码、算法等关键词
            if (text.contains("编写") || text.contains("代码") || text.contains("算法") || 
                text.contains("编程") || text.contains("实现函数") || text.contains("函数实现")) {
                return "CODING";
            }
            
            // 计算题通常包含计算、求值等关键词
            if (text.contains("计算") || text.contains("求值") || text.contains("求解") || 
                text.contains("求出") || text.contains("求得")) {
                return "CALCULATION";
            }
            
            // 如果题目文本较长，可能是论述题
            if (text.length() > 100 && 
                (text.contains("论述") || text.contains("讨论") || text.contains("分析") || 
                 text.contains("评价") || text.contains("比较"))) {
                return "ESSAY";
            }
        }
        
        // 默认返回简答题
        return "SHORT_ANSWER";
    }

    /**
     * 解析自动批改响应
     */
    private DifyDTO.AutoGradingResponse parseAutoGradingResponse(DifyDTO.DifyResponse difyResponse) {
        try {
            Map<String, Object> data = difyResponse.getData();
            if (data == null) {
                throw new RuntimeException("响应数据为空");
            }

            List<Map<String, Object>> resultsData = (List<Map<String, Object>>) data.get("results");
            List<DifyDTO.GradingResult> results = new ArrayList<>();

            if (resultsData != null) {
                for (Map<String, Object> resultData : resultsData) {
                    DifyDTO.GradingResult result = DifyDTO.GradingResult.builder()
                            .questionId(((Number) resultData.get("question_id")).longValue())
                            .isCorrect((Boolean) resultData.get("is_correct"))
                            .score((Integer) resultData.get("score"))
                            .totalScore((Integer) resultData.get("total_score"))
                            .comment((String) resultData.get("comment"))
                            .errorType((String) resultData.get("error_type"))
                            .suggestion((String) resultData.get("suggestion"))
                            .build();
                    results.add(result);
                }
            }

            return DifyDTO.AutoGradingResponse.builder()
                    .results(results)
                    .totalScore((Integer) data.get("total_score"))
                    .earnedScore((Integer) data.get("earned_score"))
                    .percentage((Double) data.get("percentage"))
                    .overallComment((String) data.get("overall_comment"))
                    .status("completed")
                    .taskId(difyResponse.getTaskId())
                    .build();

        } catch (Exception e) {
            log.error("解析自动批改响应失败: {}", e.getMessage(), e);
            throw new RuntimeException("解析批改结果失败");
        }
    }

    /**
     * 生成知识图谱
     */
    public KnowledgeGraphDTO.GenerationResponse generateKnowledgeGraph(
            KnowledgeGraphDTO.GenerationRequest request, String userId) {
        try {
            log.info("开始生成知识图谱，课程ID: {}, 章节数: {}", request.getCourseId(), 
                    request.getChapterIds() != null ? request.getChapterIds().size() : 0);

            // 构建智能体输入参数
            Map<String, Object> inputs = new HashMap<>();
            
            // 基础生成参数
            inputs.put("course_id", request.getCourseId());
            inputs.put("chapter_ids", request.getChapterIds());
            inputs.put("graph_type", request.getGraphType() != null ? request.getGraphType() : "comprehensive");
            inputs.put("depth_level", request.getDepth() != null ? request.getDepth() : 3);
            inputs.put("include_prerequisites", request.getIncludePrerequisites() != null ? request.getIncludePrerequisites() : true);
            inputs.put("include_applications", request.getIncludeApplications() != null ? request.getIncludeApplications() : true);
            
            // 核心：结构化的课程内容（由KnowledgeGraphService构建）
            inputs.put("course_content", request.getAdditionalRequirements());
            
            // 生成任务配置
            inputs.put("user_id", userId);
            inputs.put("task_type", "knowledge_graph_generation");
            inputs.put("response_format", "json");
            
            // 质量控制参数
            inputs.put("min_nodes", 5);  // 最少节点数
            inputs.put("max_nodes", getMaxNodesByDepth(request.getDepth())); // 根据深度限制最大节点数
            inputs.put("require_validation", true); // 要求输出验证
            
            log.info("调用Dify智能体生成知识图谱，参数配置完成");

            // 调用Dify智能体API
            DifyDTO.DifyResponse difyResponse = callWorkflowApi("knowledge-graph", inputs, userId);

            // 解析和处理响应
            if ("completed".equals(difyResponse.getStatus())) {
                log.info("Dify智能体生成完成，开始解析结果");
                return parseKnowledgeGraphResponse(difyResponse);
            } else if ("failed".equals(difyResponse.getStatus())) {
                log.error("Dify智能体生成失败");
                return KnowledgeGraphDTO.GenerationResponse.builder()
                        .status("failed")
                        .errorMessage("智能体生成失败，请检查输入数据或稍后重试")
                        .build();
            } else {
                log.info("Dify智能体处理中，任务ID: {}", difyResponse.getTaskId());
                return KnowledgeGraphDTO.GenerationResponse.builder()
                        .status(difyResponse.getStatus())
                        .taskId(difyResponse.getTaskId())
                        .build();
            }

        } catch (Exception e) {
            log.error("生成知识图谱过程中发生异常: {}", e.getMessage(), e);
            return KnowledgeGraphDTO.GenerationResponse.builder()
                    .status("failed")
                    .errorMessage("生成过程中发生错误: " + e.getMessage())
                    .build();
        }
    }
    
    /**
     * 根据深度级别确定最大节点数
     */
    private int getMaxNodesByDepth(Integer depth) {
        if (depth == null) depth = 3;
        switch (depth) {
            case 1: return 10;
            case 2: return 20;
            case 3: return 35;
            case 4: return 50;
            case 5: return 80;
            default: return 35;
        }
    }

    /**
     * 解析知识图谱生成响应 - 处理Dify智能体返回的结构化数据
     */
    private KnowledgeGraphDTO.GenerationResponse parseKnowledgeGraphResponse(DifyDTO.DifyResponse difyResponse) {
        try {
            Map<String, Object> data = difyResponse.getData();
            if (data == null) {
                log.error("Dify响应数据为空");
                throw new RuntimeException("智能体返回的响应数据为空");
            }
            
            log.info("开始解析Dify智能体返回的知识图谱数据");
            
            // 首先检查是否有content或text字段，这些可能包含<think>标签
            Map<String, Object> jsonData = data;
            String textContent = null;
            
            // 检查可能包含文本内容的字段
            if (data.containsKey("text")) {
                textContent = (String) data.get("text");
            } else if (data.containsKey("content")) {
                textContent = (String) data.get("content");
            } else if (data.containsKey("answer")) {
                textContent = (String) data.get("answer");
            }
            
            // 如果找到了文本内容，尝试处理<think>标签并提取JSON
            if (textContent != null) {
                log.info("Dify返回的响应包含文本内容，尝试提取JSON部分");
                log.debug("原始响应文本长度: {} 字符", textContent.length());
                
                // 尝试从文本中提取JSON部分
                String jsonStr = extractJsonFromText(textContent);
                if (jsonStr != null && !jsonStr.isEmpty()) {
                    try {
                        jsonData = objectMapper.readValue(jsonStr, Map.class);
                        log.info("从文本中成功提取并解析JSON数据");
                    } catch (Exception e) {
                        log.error("从文本中提取的JSON解析失败: {}", e.getMessage());
                        // 如果解析失败，仍然使用原始数据
                        jsonData = data;
                    }
                } else {
                    log.warn("无法从文本内容中提取有效的JSON");
                }
            } else {
                // 如果没有找到文本内容字段，尝试直接将整个data转换为JSON字符串，然后提取
                try {
                    String dataStr = objectMapper.writeValueAsString(data);
                    if (dataStr.contains("<think>") || dataStr.contains("```")) {
                        log.info("检测到数据可能包含<think>标签或代码块，尝试提取JSON...");
                        String jsonStr = extractJsonFromText(dataStr);
                        if (jsonStr != null && !jsonStr.isEmpty()) {
                            jsonData = objectMapper.readValue(jsonStr, Map.class);
                            log.info("成功从数据中提取并解析JSON");
                        }
                    }
                } catch (Exception e) {
                    log.warn("处理数据时出错: {}", e.getMessage());
                }
            }
            
            // 解析图谱数据 - 现在支持ECharts格式
            KnowledgeGraphDTO.GraphData graphData = parseGraphData(jsonData);
            
            // 数据质量验证
            if (graphData.getNodes() == null || graphData.getNodes().isEmpty()) {
                throw new RuntimeException("生成的知识图谱没有包含任何节点");
            }
            
            // 统计信息
            int nodeCount = graphData.getNodes().size();
            int edgeCount = graphData.getEdges() != null ? graphData.getEdges().size() : 0;
            log.info("知识图谱解析成功: {} 个节点, {} 条边", nodeCount, edgeCount);
            
            // 添加统计元数据
            if (graphData.getMetadata() == null) {
                graphData.setMetadata(new HashMap<>());
            }
            graphData.getMetadata().put("nodeCount", nodeCount);
            graphData.getMetadata().put("edgeCount", edgeCount);
            graphData.getMetadata().put("generatedAt", java.time.LocalDateTime.now().toString());
            graphData.getMetadata().put("source", "dify_agent");
            graphData.getMetadata().put("format", "echarts");
            
            return KnowledgeGraphDTO.GenerationResponse.builder()
                    .status("completed")
                    .taskId(difyResponse.getTaskId())
                    .graphData(graphData)
                    .suggestions((String) jsonData.get("suggestions"))
                    .build();
                    
        } catch (Exception e) {
            log.error("解析知识图谱响应失败: {}", e.getMessage(), e);
            return KnowledgeGraphDTO.GenerationResponse.builder()
                    .status("failed")
                    .errorMessage("解析智能体响应失败: " + e.getMessage())
                    .build();
        }
    }

    /**
     * 解析图谱数据，支持原有格式和ECharts格式
     */
    @SuppressWarnings("unchecked")
    private KnowledgeGraphDTO.GraphData parseGraphData(Map<String, Object> data) {
        try {
            List<KnowledgeGraphDTO.GraphNode> nodes = new ArrayList<>();
            List<KnowledgeGraphDTO.GraphEdge> edges = new ArrayList<>();
            String title = "";
            String description = "";
            
            // 检查是否是ECharts格式
            if (data.containsKey("series") && data.containsKey("title")) {
                log.info("检测到ECharts格式的知识图谱数据");
                
                // 从ECharts格式中提取标题
                if (data.get("title") instanceof Map) {
                    Map<String, Object> titleObj = (Map<String, Object>) data.get("title");
                    title = (String) titleObj.getOrDefault("text", "知识图谱");
                }
                
                // 获取series数据
                List<Map<String, Object>> seriesList = (List<Map<String, Object>>) data.get("series");
                if (seriesList != null && !seriesList.isEmpty()) {
                    Map<String, Object> series = seriesList.get(0);
                    
                    // 从series中提取description (如果有)
                    description = (String) series.getOrDefault("name", "");
                    
                    // 处理节点数据
                    List<Map<String, Object>> nodesList = (List<Map<String, Object>>) series.get("data");
                    if (nodesList != null) {
                        for (Map<String, Object> nodeData : nodesList) {
                            String nodeId = (String) nodeData.get("id");
                            String nodeName = (String) nodeData.get("name");
                            Integer category = nodeData.get("category") instanceof Integer ? 
                                (Integer) nodeData.get("category") : 0;
                            
                            // 根据category获取节点类型
                            String nodeType;
                            switch (category) {
                                case 0: nodeType = "topic"; break;
                                case 1: nodeType = "chapter"; break;
                                case 2: nodeType = "concept"; break;
                                case 3: nodeType = "skill"; break;
                                default: nodeType = "concept";
                            }
                            
                            // 获取节点级别，从symbolSize逆推或从value获取
                            Integer symbolSize = nodeData.get("symbolSize") instanceof Integer ? 
                                (Integer) nodeData.get("symbolSize") : 50;
                            Integer level = nodeData.get("value") instanceof Integer ? 
                                (Integer) nodeData.get("value") : 
                                Math.max(1, (symbolSize - 30) / 10);
                            
                            // 获取描述，从tooltip或其他属性
                            String nodeDesc = null;
                            if (nodeData.containsKey("tooltip")) {
                                Object tooltip = nodeData.get("tooltip");
                                if (tooltip instanceof Map) {
                                    nodeDesc = (String) ((Map<String, Object>) tooltip).get("formatter");
                                } else if (tooltip instanceof String) {
                                    nodeDesc = (String) tooltip;
                                }
                            }
                            
                            if (nodeDesc == null) {
                                nodeDesc = nodeName + "的详细描述";
                            }
                            
                            KnowledgeGraphDTO.GraphNode node = KnowledgeGraphDTO.GraphNode.builder()
                                    .id(nodeId)
                                    .name(nodeName)
                                    .type(nodeType)
                                    .level(level)
                                    .description(nodeDesc)
                                    .style(parseNodeStyle(
                                        Map.of(
                                            "color", "#" + Integer.toHexString((category * 20 + 100) * 65536 + 255),
                                            "size", symbolSize
                                        )
                                    ))
                                    .build();
                            nodes.add(node);
                        }
                    }
                    
                    // 处理关系数据
                    List<Map<String, Object>> linksList = (List<Map<String, Object>>) series.get("links");
                    if (linksList != null) {
                        int edgeIndex = 1;
                        for (Map<String, Object> linkData : linksList) {
                            String source = (String) linkData.get("source");
                            String target = (String) linkData.get("target");
                            Integer value = linkData.get("value") instanceof Integer ? 
                                (Integer) linkData.get("value") : 1;
                                
                            // 根据value推断关系类型
                            String edgeType;
                            switch (value) {
                                case 2: edgeType = "prerequisite"; break;
                                case 3: edgeType = "application"; break;
                                default: edgeType = "contains"; 
                            }
                            
                            // 获取关系描述
                            String edgeDesc = null;
                            if (linkData.containsKey("tooltip")) {
                                Object tooltip = linkData.get("tooltip");
                                if (tooltip instanceof Map) {
                                    edgeDesc = (String) ((Map<String, Object>) tooltip).get("formatter");
                                } else if (tooltip instanceof String) {
                                    edgeDesc = (String) tooltip;
                                }
                            }
                            
                            if (edgeDesc == null) {
                                edgeDesc = edgeType + "关系";
                            }
                            
                            KnowledgeGraphDTO.GraphEdge edge = KnowledgeGraphDTO.GraphEdge.builder()
                                    .id("edge_" + edgeIndex++)
                                    .source(source)
                                    .target(target)
                                    .type(edgeType)
                                    .description(edgeDesc)
                                    .weight(value.doubleValue())
                                    .style(parseEdgeStyle(
                                        Map.of(
                                            "lineType", value == 2 ? "dashed" : (value == 3 ? "dotted" : "solid")
                                        )
                                    ))
                                    .build();
                            edges.add(edge);
                        }
                    }
                }
            } else if (data.containsKey("nodes") && data.containsKey("edges")) {
                // 原有格式的处理逻辑
                log.info("检测到原有格式的知识图谱数据");
                
                title = (String) data.get("title");
                description = (String) data.get("description");
                
                // 解析节点数据
                List<Map<String, Object>> nodesList = (List<Map<String, Object>>) data.get("nodes");
                if (nodesList != null) {
                    for (Map<String, Object> nodeData : nodesList) {
                        KnowledgeGraphDTO.GraphNode node = KnowledgeGraphDTO.GraphNode.builder()
                                .id((String) nodeData.get("id"))
                                .name((String) nodeData.get("name"))
                                .type((String) nodeData.get("type"))
                                .level((Integer) nodeData.get("level"))
                                .description((String) nodeData.get("description"))
                                .chapterId(nodeData.get("chapter_id") != null ? Long.valueOf(nodeData.get("chapter_id").toString()) : null)
                                .sectionId(nodeData.get("section_id") != null ? Long.valueOf(nodeData.get("section_id").toString()) : null)
                                .style(parseNodeStyle((Map<String, Object>) nodeData.get("style")))
                                .position(parseNodePosition((Map<String, Object>) nodeData.get("position")))
                                .properties((Map<String, Object>) nodeData.get("properties"))
                                .build();
                        nodes.add(node);
                    }
                }
                
                // 解析边数据
                List<Map<String, Object>> edgesList = (List<Map<String, Object>>) data.get("edges");
                if (edgesList != null) {
                    for (Map<String, Object> edgeData : edgesList) {
                        KnowledgeGraphDTO.GraphEdge edge = KnowledgeGraphDTO.GraphEdge.builder()
                                .id((String) edgeData.get("id"))
                                .source((String) edgeData.get("source"))
                                .target((String) edgeData.get("target"))
                                .type((String) edgeData.get("type"))
                                .description((String) edgeData.get("description"))
                                .weight(edgeData.get("weight") != null ? Double.valueOf(edgeData.get("weight").toString()) : null)
                                .style(parseEdgeStyle((Map<String, Object>) edgeData.get("style")))
                                .properties((Map<String, Object>) edgeData.get("properties"))
                                .build();
                        edges.add(edge);
                    }
                }
            } else {
                log.error("未知的知识图谱数据格式");
                throw new RuntimeException("未知的知识图谱数据格式，缺少必要的nodes和edges或ECharts格式");
            }
            
            // 保存原始数据为元数据
            Map<String, Object> metadata = new HashMap<>();
            metadata.put("rawData", data);
            
            return KnowledgeGraphDTO.GraphData.builder()
                    .title(title)
                    .description(description)
                    .nodes(nodes)
                    .edges(edges)
                    .metadata(metadata)
                    .build();
                    
        } catch (Exception e) {
            log.error("解析图谱数据失败: {}", e.getMessage(), e);
            throw new RuntimeException("解析图谱数据失败", e);
        }
    }

    /**
     * 解析节点样式
     */
    private KnowledgeGraphDTO.NodeStyle parseNodeStyle(Map<String, Object> styleData) {
        if (styleData == null) {
            return KnowledgeGraphDTO.NodeStyle.builder().build();
        }
        
        return KnowledgeGraphDTO.NodeStyle.builder()
                .color((String) styleData.get("color"))
                .size(styleData.get("size") != null ? (Integer) styleData.get("size") : null)
                .shape((String) styleData.get("shape"))
                .fontSize(styleData.get("fontSize") != null ? (Integer) styleData.get("fontSize") : null)
                .highlighted(styleData.get("highlighted") != null ? (Boolean) styleData.get("highlighted") : false)
                .build();
    }

    /**
     * 解析边样式
     */
    private KnowledgeGraphDTO.EdgeStyle parseEdgeStyle(Map<String, Object> styleData) {
        if (styleData == null) {
            return KnowledgeGraphDTO.EdgeStyle.builder().build();
        }
        
        return KnowledgeGraphDTO.EdgeStyle.builder()
                .color((String) styleData.get("color"))
                .width(styleData.get("width") != null ? (Integer) styleData.get("width") : null)
                .lineType((String) styleData.get("lineType"))
                .showArrow(styleData.get("showArrow") != null ? (Boolean) styleData.get("showArrow") : true)
                .build();
    }

    /**
     * 解析节点位置
     */
    private KnowledgeGraphDTO.NodePosition parseNodePosition(Map<String, Object> positionData) {
        if (positionData == null) {
            return KnowledgeGraphDTO.NodePosition.builder().build();
        }
        
        return KnowledgeGraphDTO.NodePosition.builder()
                .x(positionData.get("x") != null ? Double.valueOf(positionData.get("x").toString()) : null)
                .y(positionData.get("y") != null ? Double.valueOf(positionData.get("y").toString()) : null)
                .fixed(positionData.get("fixed") != null ? (Boolean) positionData.get("fixed") : false)
                .build();
    }

    /**
     * 测试Dify API连接
     * 这是一个简单的测试方法，用于检查API连接是否正常
     * @return 连接测试结果
     */
    public String testDifyApiConnection() {
        try {
            log.info("开始测试Dify API连接...");
            String apiUrl = difyConfig.getApiUrl();
            log.info("API基础URL: {}", apiUrl);
            
            // 构建一个简单的HTTP请求，只检查连接性
            java.net.http.HttpClient httpClient = java.net.http.HttpClient.newBuilder()
                .version(java.net.http.HttpClient.Version.HTTP_2)
                .connectTimeout(java.time.Duration.ofSeconds(10))
                .build();
                
            // 首先尝试连接API基础URL
            java.net.http.HttpRequest baseRequest = java.net.http.HttpRequest.newBuilder()
                .uri(java.net.URI.create(apiUrl))
                .timeout(java.time.Duration.ofSeconds(10))
                .GET()
                .build();
                
            log.info("发送测试请求到基础URL: {}", apiUrl);
            
            try {
                java.net.http.HttpResponse<String> baseResponse = httpClient.send(
                    baseRequest, java.net.http.HttpResponse.BodyHandlers.ofString());
                
                int baseStatusCode = baseResponse.statusCode();
                log.info("基础URL连接测试结果 - 状态码: {}", baseStatusCode);
                
                // 再尝试连接chat-messages端点
                String chatEndpoint = apiUrl + "/chat-messages";
                java.net.http.HttpRequest chatRequest = java.net.http.HttpRequest.newBuilder()
                    .uri(java.net.URI.create(chatEndpoint))
                    .timeout(java.time.Duration.ofSeconds(10))
                    .header("Content-Type", "application/json")
                    .method("OPTIONS", java.net.http.HttpRequest.BodyPublishers.noBody())
                    .build();
                    
                log.info("发送测试请求到聊天端点: {}", chatEndpoint);
                
                try {
                    java.net.http.HttpResponse<String> chatResponse = httpClient.send(
                        chatRequest, java.net.http.HttpResponse.BodyHandlers.ofString());
                    
                    int chatStatusCode = chatResponse.statusCode();
                    log.info("聊天端点连接测试结果 - 状态码: {}", chatStatusCode);
                    
                    return String.format("连接测试完成。基础URL状态码: %d, 聊天端点状态码: %d", 
                        baseStatusCode, chatStatusCode);
                } catch (java.net.ConnectException e) {
                    log.error("无法连接到聊天端点: {}", e.getMessage());
                    return "无法连接到聊天端点: " + e.getMessage();
                } catch (Exception e) {
                    log.error("测试聊天端点时发生错误: {}", e.getMessage());
                    return "测试聊天端点时发生错误: " + e.getMessage();
                }
                
            } catch (java.net.ConnectException e) {
                log.error("无法连接到API基础URL: {}", e.getMessage());
                return "无法连接到API基础URL: " + e.getMessage();
            } catch (Exception e) {
                log.error("测试API基础URL时发生错误: {}", e.getMessage());
                return "测试API基础URL时发生错误: " + e.getMessage();
            }
            
        } catch (Exception e) {
            log.error("测试Dify API连接时发生异常: {}", e.getMessage());
            return "测试Dify API连接时发生异常: " + e.getMessage();
        }
    }
    
    /**
     * 测试Dify API认证
     * 这个方法测试API密钥是否有效
     * @param appType 应用类型
     * @return 认证测试结果
     */
    public String testDifyApiAuthentication(String appType) {
        try {
            log.info("开始测试Dify API认证，应用类型: {}", appType);
            
            String apiKey = difyConfig.getApiKey(appType);
            if (apiKey == null) {
                return "未配置" + appType + "的API密钥";
            }
            
            String apiUrl = difyConfig.getApiUrl() + "/info";
            log.info("API信息URL: {}", apiUrl);
            
            // 构建一个简单的HTTP请求，测试认证
            java.net.http.HttpClient httpClient = java.net.http.HttpClient.newBuilder()
                .version(java.net.http.HttpClient.Version.HTTP_2)
                .connectTimeout(java.time.Duration.ofSeconds(10))
                .build();
                
            java.net.http.HttpRequest request = java.net.http.HttpRequest.newBuilder()
                .uri(java.net.URI.create(apiUrl))
                .timeout(java.time.Duration.ofSeconds(10))
                .header("Authorization", "Bearer " + apiKey)
                .GET()
                .build();
                
            log.info("发送认证测试请求...");
            
            try {
                java.net.http.HttpResponse<String> response = httpClient.send(
                    request, java.net.http.HttpResponse.BodyHandlers.ofString());
                
                int statusCode = response.statusCode();
                String responseBody = response.body();
                
                log.info("认证测试结果 - 状态码: {}", statusCode);
                if (responseBody != null && responseBody.length() <= 1000) {
                    log.debug("认证测试响应: {}", responseBody);
                }
                
                if (statusCode == 200) {
                    return "认证成功，API密钥有效";
                } else if (statusCode == 401) {
                    return "认证失败，API密钥无效";
                } else {
                    return "认证测试返回非预期状态码: " + statusCode;
                }
                
            } catch (java.net.ConnectException e) {
                log.error("无法连接到API: {}", e.getMessage());
                return "无法连接到API: " + e.getMessage();
            } catch (Exception e) {
                log.error("测试认证时发生错误: {}", e.getMessage());
                return "测试认证时发生错误: " + e.getMessage();
            }
            
        } catch (Exception e) {
            log.error("测试Dify API认证时发生异常: {}", e.getMessage());
            return "测试Dify API认证时发生异常: " + e.getMessage();
        }
    }

    /**
     * 执行详细的Dify API连接诊断
     * 测试网络连接、DNS解析和API端点可用性
     * @return 诊断结果，包含多个测试的详细信息
     */
    public Map<String, Object> diagnoseDifyApiConnection() {
        Map<String, Object> diagnosticResults = new HashMap<>();
        Map<String, Object> tests = new HashMap<>();
        diagnosticResults.put("tests", tests);
        
        try {
            log.info("开始执行详细的Dify API连接诊断...");
            String apiUrl = difyConfig.getApiUrl();
            
            if (apiUrl == null || apiUrl.isEmpty()) {
                diagnosticResults.put("status", "failed");
                diagnosticResults.put("message", "API URL未配置");
                return diagnosticResults;
            }
            
            diagnosticResults.put("api_url", apiUrl);
            log.info("API URL: {}", apiUrl);
            
            // 解析URL，获取主机名
            java.net.URL url = new java.net.URL(apiUrl);
            String host = url.getHost();
            int port = url.getPort() == -1 ? url.getDefaultPort() : url.getPort();
            String protocol = url.getProtocol();
            
            diagnosticResults.put("host", host);
            diagnosticResults.put("port", port);
            diagnosticResults.put("protocol", protocol);
            
            // 测试1: DNS解析
            Map<String, Object> dnsTest = new HashMap<>();
            tests.put("dns_resolution", dnsTest);
            
            try {
                log.info("测试DNS解析: {}", host);
                long dnsStart = System.currentTimeMillis();
                java.net.InetAddress address = java.net.InetAddress.getByName(host);
                long dnsEnd = System.currentTimeMillis();
                
                String ipAddress = address.getHostAddress();
                dnsTest.put("status", "success");
                dnsTest.put("ip_address", ipAddress);
                dnsTest.put("time_ms", dnsEnd - dnsStart);
                log.info("DNS解析成功: {} -> {}, 耗时: {}毫秒", host, ipAddress, dnsEnd - dnsStart);
            } catch (Exception e) {
                dnsTest.put("status", "failed");
                dnsTest.put("error", e.getMessage());
                log.error("DNS解析失败: {}", e.getMessage());
            }
            
            // 测试2: ICMP Ping测试
            Map<String, Object> pingTest = new HashMap<>();
            tests.put("ping", pingTest);
            
            try {
                log.info("执行Ping测试: {}", host);
                long pingStart = System.currentTimeMillis();
                boolean reachable = java.net.InetAddress.getByName(host).isReachable(5000);
                long pingEnd = System.currentTimeMillis();
                
                pingTest.put("status", reachable ? "success" : "failed");
                pingTest.put("reachable", reachable);
                pingTest.put("time_ms", pingEnd - pingStart);
                log.info("Ping测试结果: {}, 耗时: {}毫秒", reachable ? "可达" : "不可达", pingEnd - pingStart);
            } catch (Exception e) {
                pingTest.put("status", "failed");
                pingTest.put("error", e.getMessage());
                log.error("Ping测试失败: {}", e.getMessage());
            }
            
            // 测试3: TCP连接测试
            Map<String, Object> tcpTest = new HashMap<>();
            tests.put("tcp_connection", tcpTest);
            
            try {
                log.info("执行TCP连接测试: {}:{}", host, port);
                long tcpStart = System.currentTimeMillis();
                
                java.net.Socket socket = new java.net.Socket();
                socket.connect(new java.net.InetSocketAddress(host, port), 5000);
                boolean connected = socket.isConnected();
                socket.close();
                
                long tcpEnd = System.currentTimeMillis();
                
                tcpTest.put("status", connected ? "success" : "failed");
                tcpTest.put("connected", connected);
                tcpTest.put("time_ms", tcpEnd - tcpStart);
                log.info("TCP连接测试结果: {}, 耗时: {}毫秒", connected ? "成功" : "失败", tcpEnd - tcpStart);
            } catch (Exception e) {
                tcpTest.put("status", "failed");
                tcpTest.put("error", e.getMessage());
                log.error("TCP连接测试失败: {}", e.getMessage());
            }
            
            // 测试4: HTTP HEAD请求测试
            Map<String, Object> httpHeadTest = new HashMap<>();
            tests.put("http_head", httpHeadTest);
            
            try {
                log.info("执行HTTP HEAD请求测试: {}", apiUrl);
                long headStart = System.currentTimeMillis();
                
                java.net.http.HttpClient httpClient = java.net.http.HttpClient.newBuilder()
                    .version(java.net.http.HttpClient.Version.HTTP_2)
                    .connectTimeout(java.time.Duration.ofSeconds(5))
                    .build();
                    
                java.net.http.HttpRequest headRequest = java.net.http.HttpRequest.newBuilder()
                    .uri(java.net.URI.create(apiUrl))
                    .timeout(java.time.Duration.ofSeconds(5))
                    .method("HEAD", java.net.http.HttpRequest.BodyPublishers.noBody())
                    .build();
                    
                java.net.http.HttpResponse<Void> headResponse = httpClient.send(
                    headRequest, java.net.http.HttpResponse.BodyHandlers.discarding());
                
                long headEnd = System.currentTimeMillis();
                
                int statusCode = headResponse.statusCode();
                httpHeadTest.put("status", statusCode < 400 ? "success" : "failed");
                httpHeadTest.put("status_code", statusCode);
                httpHeadTest.put("time_ms", headEnd - headStart);
                
                // 收集响应头
                Map<String, List<String>> headers = headResponse.headers().map();
                Map<String, String> simplifiedHeaders = new HashMap<>();
                for (Map.Entry<String, List<String>> entry : headers.entrySet()) {
                    simplifiedHeaders.put(entry.getKey(), String.join(", ", entry.getValue()));
                }
                httpHeadTest.put("headers", simplifiedHeaders);
                
                log.info("HTTP HEAD请求测试结果: 状态码 {}, 耗时: {}毫秒", statusCode, headEnd - headStart);
            } catch (Exception e) {
                httpHeadTest.put("status", "failed");
                httpHeadTest.put("error", e.getMessage());
                log.error("HTTP HEAD请求测试失败: {}", e.getMessage());
            }
            
            // 测试5: HTTP OPTIONS请求测试
            Map<String, Object> httpOptionsTest = new HashMap<>();
            tests.put("http_options", httpOptionsTest);
            
            try {
                log.info("执行HTTP OPTIONS请求测试: {}", apiUrl);
                long optionsStart = System.currentTimeMillis();
                
                java.net.http.HttpClient httpClient = java.net.http.HttpClient.newBuilder()
                    .version(java.net.http.HttpClient.Version.HTTP_2)
                    .connectTimeout(java.time.Duration.ofSeconds(5))
                    .build();
                    
                java.net.http.HttpRequest optionsRequest = java.net.http.HttpRequest.newBuilder()
                    .uri(java.net.URI.create(apiUrl))
                    .timeout(java.time.Duration.ofSeconds(5))
                    .method("OPTIONS", java.net.http.HttpRequest.BodyPublishers.noBody())
                    .build();
                    
                java.net.http.HttpResponse<String> optionsResponse = httpClient.send(
                    optionsRequest, java.net.http.HttpResponse.BodyHandlers.ofString());
                
                long optionsEnd = System.currentTimeMillis();
                
                int statusCode = optionsResponse.statusCode();
                httpOptionsTest.put("status", statusCode < 400 ? "success" : "failed");
                httpOptionsTest.put("status_code", statusCode);
                httpOptionsTest.put("time_ms", optionsEnd - optionsStart);
                
                // 收集响应头
                Map<String, List<String>> headers = optionsResponse.headers().map();
                Map<String, String> simplifiedHeaders = new HashMap<>();
                for (Map.Entry<String, List<String>> entry : headers.entrySet()) {
                    simplifiedHeaders.put(entry.getKey(), String.join(", ", entry.getValue()));
                }
                httpOptionsTest.put("headers", simplifiedHeaders);
                
                log.info("HTTP OPTIONS请求测试结果: 状态码 {}, 耗时: {}毫秒", statusCode, optionsEnd - optionsStart);
            } catch (Exception e) {
                httpOptionsTest.put("status", "failed");
                httpOptionsTest.put("error", e.getMessage());
                log.error("HTTP OPTIONS请求测试失败: {}", e.getMessage());
            }
            
            // 测试6: 尝试连接聊天端点
            Map<String, Object> chatEndpointTest = new HashMap<>();
            tests.put("chat_endpoint", chatEndpointTest);
            
            try {
                String chatEndpoint = apiUrl + "/chat-messages";
                log.info("测试聊天端点: {}", chatEndpoint);
                long chatStart = System.currentTimeMillis();
                
                java.net.http.HttpClient httpClient = java.net.http.HttpClient.newBuilder()
                    .version(java.net.http.HttpClient.Version.HTTP_2)
                    .connectTimeout(java.time.Duration.ofSeconds(5))
                    .build();
                    
                java.net.http.HttpRequest chatRequest = java.net.http.HttpRequest.newBuilder()
                    .uri(java.net.URI.create(chatEndpoint))
                    .timeout(java.time.Duration.ofSeconds(5))
                    .method("OPTIONS", java.net.http.HttpRequest.BodyPublishers.noBody())
                    .build();
                    
                java.net.http.HttpResponse<String> chatResponse = httpClient.send(
                    chatRequest, java.net.http.HttpResponse.BodyHandlers.ofString());
                
                long chatEnd = System.currentTimeMillis();
                
                int statusCode = chatResponse.statusCode();
                chatEndpointTest.put("status", statusCode < 400 ? "success" : "failed");
                chatEndpointTest.put("status_code", statusCode);
                chatEndpointTest.put("time_ms", chatEnd - chatStart);
                
                log.info("聊天端点测试结果: 状态码 {}, 耗时: {}毫秒", statusCode, chatEnd - chatStart);
            } catch (Exception e) {
                chatEndpointTest.put("status", "failed");
                chatEndpointTest.put("error", e.getMessage());
                log.error("聊天端点测试失败: {}", e.getMessage());
            }
            
            // 汇总诊断结果
            int successCount = 0;
            int totalTests = tests.size();
            for (Object testObj : tests.values()) {
                if (testObj instanceof Map) {
                    Map<String, Object> test = (Map<String, Object>) testObj;
                    if ("success".equals(test.get("status"))) {
                        successCount++;
                    }
                }
            }
            
            if (successCount == totalTests) {
                diagnosticResults.put("status", "success");
                diagnosticResults.put("message", "所有测试通过");
            } else if (successCount > 0) {
                diagnosticResults.put("status", "partial");
                diagnosticResults.put("message", String.format("部分测试通过 (%d/%d)", successCount, totalTests));
            } else {
                diagnosticResults.put("status", "failed");
                diagnosticResults.put("message", "所有测试失败");
            }
            
            log.info("诊断完成: {}/{} 测试通过", successCount, totalTests);
            return diagnosticResults;
            
        } catch (Exception e) {
            log.error("执行诊断时发生异常: {}", e.getMessage(), e);
            diagnosticResults.put("status", "error");
            diagnosticResults.put("message", "诊断过程中发生异常: " + e.getMessage());
            return diagnosticResults;
        }
    }

    /**
     * 从文本中提取JSON部分
     */
    private String extractJsonFromText(String text) {
        try {
            if (text == null || text.isEmpty()) {
                log.warn("输入文本为空，无法提取JSON");
                return null;
            }
            
            log.debug("开始从文本中提取JSON，文本长度: {}", text.length());
            
            // 移除<think>标签及其内容 - 使用更强大的正则表达式
            if (text.contains("<think") || text.contains("<think>")) {
                log.info("检测到<think>标签，正在移除思考过程...");
                
                // 移除完整的<think>...</think>标签及其内容 - 处理可能的空格和换行
                text = text.replaceAll("(?is)<\\s*think\\s*>.*?<\\s*/\\s*think\\s*>", "").trim();
                
                // 移除可能没有闭合的<think>标签及其后内容
                int thinkIndex = text.indexOf("<think");
                if (thinkIndex >= 0) {
                    text = text.substring(0, thinkIndex).trim();
                }
                
                log.debug("移除思考过程后的文本长度: {}", text.length());
            }
            
            // 移除可能的markdown代码块标记和其他非JSON文本
            // 匹配JSON代码块 - 支持多种格式的代码块标记
            java.util.regex.Pattern jsonBlockPattern = java.util.regex.Pattern.compile(
                "```(?:json)?\\s*(\\{[\\s\\S]*?\\})\\s*```", 
                java.util.regex.Pattern.DOTALL
            );
            java.util.regex.Matcher jsonBlockMatcher = jsonBlockPattern.matcher(text);
            
            if (jsonBlockMatcher.find()) {
                // 从代码块中提取JSON
                String jsonCandidate = jsonBlockMatcher.group(1).trim();
                log.debug("从代码块中提取的JSON: {}", 
                    jsonCandidate.length() > 100 ? 
                    jsonCandidate.substring(0, 100) + "..." : 
                    jsonCandidate);
                return jsonCandidate;
            }
            
            // 如果没有找到代码块，尝试直接匹配JSON对象
            // 首先查找第一个{和最后一个}
            int firstBrace = text.indexOf('{');
            int lastBrace = text.lastIndexOf('}');
            
            if (firstBrace >= 0 && lastBrace > firstBrace) {
                String jsonCandidate = text.substring(firstBrace, lastBrace + 1).trim();
                log.debug("使用大括号位置提取的JSON: {}", 
                    jsonCandidate.length() > 100 ? 
                    jsonCandidate.substring(0, 100) + "..." : 
                    jsonCandidate);
                
                // 验证JSON格式
                try {
                    objectMapper.readTree(jsonCandidate);
                    log.info("成功验证提取的JSON格式");
                    return jsonCandidate;
                } catch (Exception e) {
                    log.warn("提取的JSON格式验证失败: {}", e.getMessage());
                    // 继续尝试其他方法
                }
            }
            
            // 尝试使用正则表达式匹配完整的JSON对象 - 更强大的正则表达式
            java.util.regex.Pattern jsonPattern = java.util.regex.Pattern.compile(
                "(\\{[\\s\\S]*?\\})[\\s\\S]*$", 
                java.util.regex.Pattern.DOTALL
            );
            java.util.regex.Matcher jsonMatcher = jsonPattern.matcher(text);
            
            if (jsonMatcher.find()) {
                String jsonCandidate = jsonMatcher.group(1).trim();
                log.debug("使用正则表达式匹配的JSON: {}", 
                    jsonCandidate.length() > 100 ? 
                    jsonCandidate.substring(0, 100) + "..." : 
                    jsonCandidate);
                
                // 验证JSON格式
                try {
                    objectMapper.readTree(jsonCandidate);
                    log.info("成功验证正则匹配的JSON格式");
                    return jsonCandidate;
                } catch (Exception e) {
                    log.warn("正则匹配的JSON格式验证失败: {}", e.getMessage());
                }
            }
            
            // 如果以上方法都失败，但文本本身看起来像JSON对象
            if (text.trim().startsWith("{") && text.trim().endsWith("}")) {
                String jsonCandidate = text.trim();
                log.debug("直接使用文本作为JSON: {}", 
                    jsonCandidate.length() > 100 ? 
                    jsonCandidate.substring(0, 100) + "..." : 
                    jsonCandidate);
                
                // 验证JSON格式
                try {
                    objectMapper.readTree(jsonCandidate);
                    log.info("成功验证完整文本作为JSON");
                    return jsonCandidate;
                } catch (Exception e) {
                    log.warn("完整文本作为JSON验证失败: {}", e.getMessage());
                }
            }
            
            log.warn("未能从文本中提取有效JSON");
            return null;
        } catch (Exception e) {
            log.error("提取JSON过程中发生异常: {}", e.getMessage(), e);
            return null;
        }
    }
} 
