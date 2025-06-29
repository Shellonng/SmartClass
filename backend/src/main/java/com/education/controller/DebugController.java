package com.education.controller;

import com.education.dto.common.Result;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.*;
import jakarta.servlet.http.HttpServletRequest;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.Map;
import java.time.LocalDateTime;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;

/**
 * 调试控制器 - 用于诊断CORS和路径问题
 */
@Tag(name = "调试接口", description = "用于测试和调试的接口")
@RestController
@RequestMapping("/debug")
@Slf4j
public class DebugController {

    @Operation(summary = "应用状态检查", description = "检查应用是否正常运行")
    @GetMapping("/status")
    public Result<Map<String, Object>> getStatus() {
        log.info("Debug status check called");
        
        Map<String, Object> status = new HashMap<>();
        status.put("status", "running");
        status.put("timestamp", LocalDateTime.now());
        status.put("message", "应用运行正常");
        status.put("version", "1.0.0");
        
        return Result.success(status);
    }

    @Operation(summary = "健康检查", description = "简单的健康检查接口")
    @GetMapping("/health")
    public Result<String> health() {
        log.info("Debug health check called");
        return Result.success("OK");
    }

    @Operation(summary = "测试接口", description = "用于测试的简单接口")
    @GetMapping("/test")
    public Result<String> test() {
        log.info("Debug test called");
        return Result.success("Test successful!");
    }

    @Operation(summary = "JWT认证测试", description = "需要认证的测试接口")
    @GetMapping("/auth-test")
    public Result<Map<String, Object>> authTest(HttpServletRequest request) {
        log.info("Debug auth test called");
        
        Map<String, Object> result = new HashMap<>();
        
        // 获取认证信息
        Authentication auth = SecurityContextHolder.getContext().getAuthentication();
        if (auth != null) {
            result.put("authenticated", auth.isAuthenticated());
            result.put("username", auth.getName());
            result.put("authorities", auth.getAuthorities());
        } else {
            result.put("authenticated", false);
            result.put("message", "No authentication found");
        }
        
        // 获取Authorization头
        String authHeader = request.getHeader("Authorization");
        result.put("authorizationHeader", authHeader);
        
        return Result.success(result);
    }

    @GetMapping("/info")
    public Map<String, Object> getDebugInfo(HttpServletRequest request) {
        Map<String, Object> info = new HashMap<>();
        
        // 请求信息
        info.put("method", request.getMethod());
        info.put("requestURI", request.getRequestURI());
        info.put("servletPath", request.getServletPath());
        info.put("contextPath", request.getContextPath());
        info.put("pathInfo", request.getPathInfo());
        info.put("queryString", request.getQueryString());
        info.put("requestURL", request.getRequestURL().toString());
        
        // 请求头
        Map<String, String> headers = new HashMap<>();
        Enumeration<String> headerNames = request.getHeaderNames();
        while (headerNames.hasMoreElements()) {
            String headerName = headerNames.nextElement();
            headers.put(headerName, request.getHeader(headerName));
        }
        info.put("headers", headers);
        
        return info;
    }
    
    @PostMapping("/post-test")
    public Map<String, Object> testPost(@RequestBody Map<String, Object> body, HttpServletRequest request) {
        Map<String, Object> result = new HashMap<>();
        result.put("message", "调试接口收到POST请求");
        result.put("method", request.getMethod());
        result.put("path", request.getRequestURI());
        result.put("body", body);
        result.put("contentType", request.getContentType());
        
        return result;
    }
    
    @RequestMapping(value = "/cors-test", method = {RequestMethod.GET, RequestMethod.POST, RequestMethod.OPTIONS})
    public Map<String, Object> corsTest(HttpServletRequest request) {
        Map<String, Object> result = new HashMap<>();
        result.put("message", "CORS测试成功");
        result.put("method", request.getMethod());
        result.put("path", request.getRequestURI());
        result.put("origin", request.getHeader("Origin"));
        
        return result;
    }
} 