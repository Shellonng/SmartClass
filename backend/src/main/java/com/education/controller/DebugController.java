package com.education.controller;

import org.springframework.web.bind.annotation.*;
import jakarta.servlet.http.HttpServletRequest;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.Map;

/**
 * 调试控制器 - 用于诊断CORS和路径问题
 */
@RestController
@RequestMapping("/debug")
public class DebugController {

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
    
    @PostMapping("/auth-test")
    public Map<String, Object> testAuth(@RequestBody Map<String, Object> body, HttpServletRequest request) {
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