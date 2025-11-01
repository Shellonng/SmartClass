package com.education.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.cors.CorsConfiguration;
import org.springframework.web.cors.UrlBasedCorsConfigurationSource;
import org.springframework.web.filter.CorsFilter;

import java.util.Arrays;

/**
 * CORS跨域配置
 */
@Configuration
public class CorsConfig {

    /**
     * 允许跨域请求的配置
     */
    @Bean
    public CorsFilter corsFilter() {
        UrlBasedCorsConfigurationSource source = new UrlBasedCorsConfigurationSource();
        CorsConfiguration config = new CorsConfiguration();
        
        // 允许特定来源
        config.addAllowedOrigin("http://localhost:5173"); // 前端开发服务器
        config.addAllowedOrigin("http://127.0.0.1:5173");
        config.addAllowedOrigin("http://localhost:8080"); // 后端服务器
        
        // 允许所有头信息
        config.addAllowedHeader("*");
        
        // 允许所有方法
        config.addAllowedMethod("*");
        
        // 允许携带凭证信息（cookies等）
        config.setAllowCredentials(true);
        
        // 暴露响应头
        config.setExposedHeaders(Arrays.asList(
            "Authorization", "Content-Disposition", "Content-Type", 
            "Content-Length", "Cache-Control"
        ));
        
        // 预检请求的有效期，单位为秒
        config.setMaxAge(3600L);
        
        // 对所有接口应用CORS配置
        source.registerCorsConfiguration("/**", config);
        
        return new CorsFilter(source);
    }
}