package com.education.controller;

import com.education.service.DifyService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.Map;

/**
 * Dify API控制器
 * 提供测试Dify API连接的接口
 */
@RestController
@RequestMapping("/api/dify")
@RequiredArgsConstructor
@Slf4j
public class DifyController {

    private final DifyService difyService;

    /**
     * 测试Dify API连接
     * @return 连接测试结果
     */
    @GetMapping("/test-connection")
    public Map<String, String> testConnection() {
        log.info("收到测试Dify API连接请求");
        String result = difyService.testDifyApiConnection();
        log.info("连接测试结果: {}", result);
        
        Map<String, String> response = new HashMap<>();
        response.put("status", "success");
        response.put("message", result);
        return response;
    }

    /**
     * 测试Dify API认证
     * @param appType 应用类型
     * @return 认证测试结果
     */
    @GetMapping("/test-authentication")
    public Map<String, String> testAuthentication(@RequestParam String appType) {
        log.info("收到测试Dify API认证请求，应用类型: {}", appType);
        String result = difyService.testDifyApiAuthentication(appType);
        log.info("认证测试结果: {}", result);
        
        Map<String, String> response = new HashMap<>();
        response.put("status", "success");
        response.put("message", result);
        return response;
    }
    
    /**
     * 执行详细的Dify API连接诊断
     * @return 诊断结果，包含多个测试的详细信息
     */
    @GetMapping("/diagnose")
    public Map<String, Object> diagnoseDifyApiConnection() {
        log.info("收到执行详细Dify API连接诊断请求");
        Map<String, Object> result = difyService.diagnoseDifyApiConnection();
        log.info("诊断完成，状态: {}", result.get("status"));
        return result;
    }
} 