package com.education.config;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.config.annotation.CorsRegistry;
import org.springframework.web.servlet.config.annotation.ResourceHandlerRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;
import org.springframework.web.servlet.config.annotation.ViewControllerRegistry;
import org.springframework.http.CacheControl;
import java.util.concurrent.TimeUnit;

/**
 * Web MVC 配置
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
@Configuration
public class WebMvcConfig implements WebMvcConfigurer {

    @Value("${file.upload.path:/uploads}")
    private String uploadPath;
    
    @Value("${file.access.url.prefix:/files}")
    private String accessUrlPrefix;
    
    @Value("${video.upload.path}")
    private String videoUploadPath;

    /**
     * 添加资源处理器
     */
    @Override
    public void addResourceHandlers(ResourceHandlerRegistry registry) {
        // 配置静态资源处理
        registry.addResourceHandler("/static/**")
                .addResourceLocations("classpath:/static/");
        
        // 配置上传文件的访问
        registry.addResourceHandler("/files/**")
                .addResourceLocations("file:./uploads/");
                
        // 配置视频资源访问
        registry.addResourceHandler("/resource/video/**")
                .addResourceLocations("file:" + videoUploadPath + "/")
                .setCacheControl(CacheControl.maxAge(1, TimeUnit.HOURS))
                .resourceChain(true); // 启用资源链，提供更好的资源处理
        
        // 配置Swagger UI资源
        registry.addResourceHandler("/swagger-ui/**")
                .addResourceLocations("classpath:/META-INF/resources/webjars/springfox-swagger-ui/");
    }

    /**
     * 添加视图控制器
     */
    @Override
    public void addViewControllers(ViewControllerRegistry registry) {
        // 首页重定向
        registry.addViewController("/").setViewName("redirect:/index.html");
    }

    /**
     * 跨域配置
     */
    @Override
    public void addCorsMappings(CorsRegistry registry) {
        registry.addMapping("/**")
                .allowedOriginPatterns("*")
                .allowedMethods("GET", "POST", "PUT", "DELETE", "OPTIONS")
                .allowedHeaders("*")
                .allowCredentials(true)
                .maxAge(3600);
    }
}