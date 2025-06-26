package com.education.aspect;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import org.aspectj.lang.JoinPoint;
import org.aspectj.lang.ProceedingJoinPoint;
import org.aspectj.lang.annotation.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;
import org.springframework.web.context.request.RequestContextHolder;
import org.springframework.web.context.request.ServletRequestAttributes;

import jakarta.servlet.http.HttpServletRequest;
import java.util.Arrays;

/**
 * 日志切面
 * 记录Controller层方法的调用日志
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
@Aspect
@Component
public class LoggingAspect {

    private static final Logger logger = LoggerFactory.getLogger(LoggingAspect.class);
    private final ObjectMapper objectMapper;
    
    public LoggingAspect() {
        this.objectMapper = new ObjectMapper();
        this.objectMapper.registerModule(new JavaTimeModule());
    }

    /**
     * 定义切点：Controller层的所有方法
     */
    @Pointcut("execution(* com.education.controller..*.*(..))")
    public void controllerMethods() {}

    /**
     * 前置通知：记录请求信息
     */
    @Before("controllerMethods()")
    public void logBefore(JoinPoint joinPoint) {
        ServletRequestAttributes attributes = (ServletRequestAttributes) RequestContextHolder.getRequestAttributes();
        if (attributes != null) {
            HttpServletRequest request = attributes.getRequest();
            
            logger.info("=== 请求开始 ===");
            logger.info("请求URL: {}", request.getRequestURL().toString());
            logger.info("请求方法: {}", request.getMethod());
            logger.info("请求IP: {}", getClientIpAddress(request));
            logger.info("调用方法: {}.{}", joinPoint.getSignature().getDeclaringTypeName(), joinPoint.getSignature().getName());
            logger.info("请求参数: {}", Arrays.toString(joinPoint.getArgs()));
        }
    }

    /**
     * 环绕通知：记录方法执行时间和返回结果
     */
    @Around("controllerMethods()")
    public Object logAround(ProceedingJoinPoint proceedingJoinPoint) throws Throwable {
        long startTime = System.currentTimeMillis();
        
        try {
            Object result = proceedingJoinPoint.proceed();
            long endTime = System.currentTimeMillis();
            
            logger.info("方法执行时间: {}ms", endTime - startTime);
            logger.info("返回结果: {}", objectMapper.writeValueAsString(result));
            logger.info("=== 请求结束 ===");
            
            return result;
        } catch (Exception e) {
            long endTime = System.currentTimeMillis();
            logger.error("方法执行时间: {}ms", endTime - startTime);
            logger.error("方法执行异常: {}", e.getMessage(), e);
            logger.info("=== 请求异常结束 ===");
            throw e;
        }
    }

    /**
     * 异常通知：记录异常信息
     */
    @AfterThrowing(pointcut = "controllerMethods()", throwing = "exception")
    public void logAfterThrowing(JoinPoint joinPoint, Throwable exception) {
        logger.error("方法 {}.{} 抛出异常: {}", 
                joinPoint.getSignature().getDeclaringTypeName(),
                joinPoint.getSignature().getName(),
                exception.getMessage(), exception);
    }

    /**
     * 获取客户端真实IP地址
     */
    private String getClientIpAddress(HttpServletRequest request) {
        String xForwardedFor = request.getHeader("X-Forwarded-For");
        if (xForwardedFor != null && !xForwardedFor.isEmpty() && !"unknown".equalsIgnoreCase(xForwardedFor)) {
            return xForwardedFor.split(",")[0];
        }
        
        String xRealIp = request.getHeader("X-Real-IP");
        if (xRealIp != null && !xRealIp.isEmpty() && !"unknown".equalsIgnoreCase(xRealIp)) {
            return xRealIp;
        }
        
        return request.getRemoteAddr();
    }
}