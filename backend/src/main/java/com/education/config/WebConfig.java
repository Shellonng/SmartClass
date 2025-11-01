package com.education.config;

import org.apache.hc.client5.http.config.RequestConfig;
import org.apache.hc.client5.http.impl.classic.CloseableHttpClient;
import org.apache.hc.client5.http.impl.classic.HttpClientBuilder;
import org.apache.hc.client5.http.impl.io.PoolingHttpClientConnectionManager;
import org.apache.hc.core5.util.Timeout;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.client.ClientHttpRequestFactory;
import org.springframework.http.client.HttpComponentsClientHttpRequestFactory;
import org.springframework.http.converter.HttpMessageConverter;
import org.springframework.http.converter.StringHttpMessageConverter;
import org.springframework.http.converter.json.MappingJackson2HttpMessageConverter;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.servlet.config.annotation.CorsRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.springframework.http.MediaType;
import com.fasterxml.jackson.databind.ObjectMapper;

/**
 * Web配置类
 * @author Education Platform Team
 */
@Configuration
public class WebConfig implements WebMvcConfigurer {

    /**
     * RestTemplate配置
     */
    @Bean
    public RestTemplate restTemplate() {
        // 使用SimpleClientHttpRequestFactory，避免HttpComponentsClientHttpRequestFactory的警告
        org.springframework.http.client.SimpleClientHttpRequestFactory factory = 
            new org.springframework.http.client.SimpleClientHttpRequestFactory();
        
        // 设置超时时间，统一为8分钟（480秒）
        factory.setConnectTimeout(480000); // 连接超时480秒（8分钟）
        factory.setReadTimeout(480000);    // 读取超时480秒（8分钟）
        
        RestTemplate restTemplate = new RestTemplate(factory);
        
        // 配置消息转换器，确保能够处理各种响应
        configureMessageConverters(restTemplate);
        
        return restTemplate;
    }
    
    /**
     * 配置消息转换器，增强RestTemplate处理不同内容类型的能力
     */
    private void configureMessageConverters(RestTemplate restTemplate) {
        // 创建一个可以处理text/html的StringHttpMessageConverter
        StringHttpMessageConverter stringConverter = new StringHttpMessageConverter(StandardCharsets.UTF_8);
        stringConverter.setSupportedMediaTypes(Arrays.asList(
                MediaType.TEXT_HTML,
                MediaType.TEXT_PLAIN,
                MediaType.TEXT_XML,
                MediaType.APPLICATION_JSON,
                MediaType.APPLICATION_OCTET_STREAM,
                new MediaType("application", "*+json"),
                new MediaType("text", "*")
        ));
        
        // 创建一个增强的JSON转换器，可以处理更多内容类型
        MappingJackson2HttpMessageConverter jsonConverter = new MappingJackson2HttpMessageConverter();
        jsonConverter.setObjectMapper(new ObjectMapper());
        jsonConverter.setSupportedMediaTypes(Arrays.asList(
                MediaType.APPLICATION_JSON,
                MediaType.APPLICATION_OCTET_STREAM,
                MediaType.TEXT_HTML,
                MediaType.TEXT_PLAIN,
                new MediaType("application", "*+json"),
                new MediaType("text", "*")
        ));
        
        // 清除所有现有转换器并添加我们自定义的转换器
        List<HttpMessageConverter<?>> messageConverters = new ArrayList<>();
        messageConverters.add(jsonConverter);
        messageConverters.add(stringConverter);
        
        restTemplate.setMessageConverters(messageConverters);
    }

    /**
     * HTTP请求工厂配置 - 使用Apache HttpClient 5
     * 注意：此方法不再用于RestTemplate，仅用于其他可能需要的HTTP客户端
     */
    @Bean
    public ClientHttpRequestFactory clientHttpRequestFactory() {
        // 创建连接池管理器
        PoolingHttpClientConnectionManager connectionManager = new PoolingHttpClientConnectionManager();
        connectionManager.setMaxTotal(100); // 最大连接数
        connectionManager.setDefaultMaxPerRoute(20); // 每个路由的最大连接数
        
        // 配置请求超时，统一为8分钟（480秒）
        RequestConfig requestConfig = RequestConfig.custom()
                .setConnectionRequestTimeout(Timeout.ofSeconds(480)) // 连接请求超时480秒（8分钟）
                .setResponseTimeout(Timeout.ofSeconds(480)) // 响应超时480秒（8分钟）
                .build();
        
        // 创建HttpClient
        CloseableHttpClient httpClient = HttpClientBuilder.create()
                .setConnectionManager(connectionManager)
                .setDefaultRequestConfig(requestConfig)
                .build();
        
        // 创建请求工厂
        HttpComponentsClientHttpRequestFactory factory = new HttpComponentsClientHttpRequestFactory(httpClient);
        return factory;
    }

    @Override
    public void addCorsMappings(CorsRegistry registry) {
        registry.addMapping("/**")
                .allowedOriginPatterns("*")
                .allowedMethods("GET", "POST", "PUT", "DELETE", "OPTIONS")
                .allowedHeaders("*")
                .allowCredentials(true)
                .exposedHeaders("Set-Cookie")
                .maxAge(3600);
        
        System.out.println("✅ CORS配置已加载 - 允许跨域请求携带凭证");
    }
} 