package com.education.config;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.config.annotation.ResourceHandlerRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Web配置类
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
@Configuration
public class WebConfig implements WebMvcConfigurer {
    
    private static final Logger logger = LoggerFactory.getLogger(WebConfig.class);
    
    @Value("${education.file.upload.path:./uploads}")
    private String uploadPath;
    
    @Value("${education.file.access.url.prefix:/files}")
    private String accessUrlPrefix;
    
    @Value("${file.upload.path:D:/my_git_code/SmartClass/resource/file}")
    private String fileUploadPath;
    
    @Override
    public void addResourceHandlers(ResourceHandlerRegistry registry) {
        // 配置默认的上传目录映射
        String filePath = "file:" + uploadPath + "/";
        logger.info("配置静态资源映射: {} -> {}", accessUrlPrefix + "/**", filePath);
        registry.addResourceHandler(accessUrlPrefix + "/**")
                .addResourceLocations(filePath);
        
        // 配置实际文件目录映射，用于访问课程封面等静态资源
        String resourceFilePath = "file:" + fileUploadPath + "/";
        logger.info("配置实际文件目录映射: /resource/file/** -> {}", resourceFilePath);
        registry.addResourceHandler("/resource/file/**")
                .addResourceLocations(resourceFilePath);
        
        logger.info("静态资源映射配置完成");
    }
} 