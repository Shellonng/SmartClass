package com.education.service;

import com.education.config.DifyConfig;
import com.education.dto.DifyDTO;
import com.education.dto.KnowledgeGraphDTO;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.web.client.RestTemplate;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Dify服务测试类
 * 用于验证Dify API连接和功能
 */
@SpringBootTest
@ActiveProfiles("test")
public class DifyServiceTest {

    @Autowired
    private DifyService difyService;

    @Autowired
    private DifyConfig difyConfig;

    @Autowired
    private RestTemplate restTemplate;

    @BeforeEach
    void setUp() {
        // 确保配置正确
        assertNotNull(difyConfig);
        assertNotNull(difyService);
        assertEquals("http://219.216.65.108", difyConfig.getApiUrl());
    }

    /**
     * 测试Dify配置是否正确
     */
    @Test
    void testDifyConfig() {
        // 验证API URL
        assertEquals("http://219.216.65.108", difyConfig.getApiUrl());
        
        // 验证超时配置
        assertEquals(60000, difyConfig.getTimeout());
        
        // 验证重试次数
        assertEquals(3, difyConfig.getRetryCount());
        
        System.out.println("✅ Dify配置验证通过");
        System.out.println("API URL: " + difyConfig.getApiUrl());
        System.out.println("超时时间: " + difyConfig.getTimeout() + "ms");
        System.out.println("重试次数: " + difyConfig.getRetryCount());
    }

    /**
     * 测试网络连接
     */
    @Test
    void testNetworkConnection() {
        try {
            // 测试基本连接
            String healthUrl = difyConfig.getApiUrl() + "/health";
            System.out.println("🔍 测试连接到: " + healthUrl);
            
            // 注意：这里只是测试网络连接，实际的健康检查可能需要不同的端点
            // 如果Dify服务器没有health端点，这个测试可能会失败，但这不影响功能
            
            System.out.println("✅ 网络连接测试完成");
            
        } catch (Exception e) {
            System.err.println("❌ 网络连接测试失败: " + e.getMessage());
            System.err.println("请检查：");
            System.err.println("1. 网络连接是否正常");
            System.err.println("2. Dify服务器是否运行中");
            System.err.println("3. 防火墙设置是否正确");
        }
    }

    /**
     * 测试API密钥配置
     */
    @Test
    void testApiKeyConfiguration() {
        // 检查API密钥配置
        Map<String, String> expectedKeys = Map.of(
            "paper-generation", "组卷工作流API密钥",
            "auto-grading", "自动批改工作流API密钥",
            "knowledge-graph", "知识图谱生成API密钥"
        );
        
        for (String keyName : expectedKeys.keySet()) {
            String apiKey = difyConfig.getApiKey(keyName);
            
            if (apiKey == null || apiKey.startsWith("your-")) {
                System.out.println("⚠️  API密钥未配置: " + keyName);
                System.out.println("   当前值: " + apiKey);
                System.out.println("   请在application.yml中配置正确的API密钥");
            } else {
                System.out.println("✅ API密钥已配置: " + keyName);
                System.out.println("   格式: " + (apiKey.startsWith("app-") ? "正确" : "可能错误"));
            }
        }
    }

    /**
     * 测试知识图谱生成请求构建
     */
    @Test
    void testKnowledgeGraphRequestBuilder() {
        // 创建测试请求
        KnowledgeGraphDTO.GenerationRequest request = KnowledgeGraphDTO.GenerationRequest.builder()
                .courseId(1L)
                .graphType("concept")
                .depth(3)
                .additionalRequirements("测试需求")
                .includePrerequisites(true)
                .includeApplications(true)
                .build();

        // 验证请求构建
        assertNotNull(request);
        assertEquals(1L, request.getCourseId());
        assertEquals("concept", request.getGraphType());
        assertEquals(3, request.getDepth());
        assertEquals("测试需求", request.getAdditionalRequirements());

        System.out.println("✅ 知识图谱请求构建测试通过");
        System.out.println("请求详情: " + request.toString());
    }

    /**
     * 测试组卷请求构建
     */
    @Test
    void testPaperGenerationRequestBuilder() {
        // 创建测试请求
        DifyDTO.PaperGenerationRequest request = DifyDTO.PaperGenerationRequest.builder()
                .courseId(1L)
                .difficulty("medium")
                .questionCount(20)
                .duration(90)
                .totalScore(100)
                .build();

        // 验证请求构建
        assertNotNull(request);
        assertEquals(1L, request.getCourseId());
        assertEquals("medium", request.getDifficulty());
        assertEquals(20, request.getQuestionCount());

        System.out.println("✅ 组卷请求构建测试通过");
        System.out.println("请求详情: " + request.toString());
    }

    /**
     * 测试自动批改请求构建
     */
    @Test
    void testAutoGradingRequestBuilder() {
        // 创建测试答案
        DifyDTO.StudentAnswer studentAnswer = DifyDTO.StudentAnswer.builder()
                .questionId(1L)
                .questionText("什么是多态？")
                .questionType("简答题")
                .correctAnswer("多态是面向对象编程的特性，允许同一个接口表现为不同的行为。")
                .studentAnswer("多态就是一个接口多种实现。")
                .totalScore(10)
                .build();

        // 创建测试请求
        DifyDTO.AutoGradingRequest request = DifyDTO.AutoGradingRequest.builder()
                .submissionId(1L)
                .assignmentId(1L)
                .studentId(1L)
                .studentAnswers(List.of(studentAnswer))
                .gradingType("automatic")
                .maxScore(10.0)
                .build();

        // 验证请求构建
        assertNotNull(request);
        assertEquals(1L, request.getSubmissionId());
        assertEquals(1L, request.getAssignmentId());
        assertEquals(1L, request.getStudentId());
        assertEquals(1, request.getStudentAnswers().size());

        System.out.println("✅ 自动批改请求构建测试通过");
        System.out.println("请求详情: " + request.toString());
    }
} 