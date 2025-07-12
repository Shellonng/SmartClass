package com.education.config;

import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Configuration;

import java.util.Map;

/**
 * Dify AI平台配置
 * @author Education Platform Team
 */
@Data
@Configuration
@ConfigurationProperties(prefix = "education.dify")
public class DifyConfig {
    
    /**
     * Dify API服务器地址
     */
    private String apiUrl = "http://localhost:3000";
    
    /**
     * API密钥配置
     */
    private Map<String, String> apiKeys;
    
    /**
     * 请求超时时间(毫秒)
     */
    private Integer timeout = 120000;
    
    /**
     * 重试次数
     */
    private Integer retryCount = 3;
    
    /**
     * 模型URL
     */
    private String modelUrl;
    
    /**
     * Ollama配置
     */
    private OllamaConfig ollama = new OllamaConfig();
    
    @Data
    public static class OllamaConfig {
        /**
         * 默认使用的模型
         */
        private String model = "llama2";
    }
    
    /**
     * 获取指定应用的API密钥
     */
    public String getApiKey(String appType) {
        return apiKeys != null ? apiKeys.get(appType) : null;
    }
} 