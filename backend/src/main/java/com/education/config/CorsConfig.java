package com.education.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.cors.CorsConfiguration;
import org.springframework.web.cors.UrlBasedCorsConfigurationSource;
import org.springframework.web.filter.CorsFilter;

/**
 * 跨域配置类
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
@Configuration
public class CorsConfig {

    /**
     * 跨域过滤器配置
     */
    @Bean
    public CorsFilter corsFilter() {
        System.out.println("🔧 初始化 CORS 过滤器");
        
        CorsConfiguration config = new CorsConfiguration();
        
        // 允许所有域名进行跨域调用
        config.addAllowedOriginPattern("*");
        // 允许跨域发送cookie
        config.setAllowCredentials(true);
        // 放行全部原始头信息
        config.addAllowedHeader("*");
        // 允许所有请求方法跨域调用
        config.addAllowedMethod("*");
        // 预检请求的有效期，单位为秒
        config.setMaxAge(3600L);
        
        System.out.println("✅ CORS 配置: 允许所有域名、方法和头部");
        
        UrlBasedCorsConfigurationSource source = new UrlBasedCorsConfigurationSource();
        source.registerCorsConfiguration("/**", config);
        
        CorsFilter filter = new CorsFilter(source);
        System.out.println("✅ CORS 过滤器创建完成");
        
        return filter;
    }
}