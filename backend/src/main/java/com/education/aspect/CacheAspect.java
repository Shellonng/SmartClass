package com.education.aspect;

import lombok.extern.slf4j.Slf4j;
import org.aspectj.lang.ProceedingJoinPoint;
import org.aspectj.lang.annotation.Around;
import org.aspectj.lang.annotation.Aspect;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.stereotype.Component;

import java.lang.annotation.*;
import java.util.concurrent.TimeUnit;

/**
 * 缓存切面
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
@Aspect
@Component
@Slf4j
public class CacheAspect {

    @Autowired
    private RedisTemplate<String, Object> redisTemplate;

    /**
     * 缓存处理
     */
    @Around("@annotation(cacheable)")
    public Object handleCache(ProceedingJoinPoint joinPoint, Cacheable cacheable) throws Throwable {
        String key = generateCacheKey(joinPoint, cacheable);
        
        // 尝试从缓存获取数据
        Object cachedResult = redisTemplate.opsForValue().get(key);
        if (cachedResult != null) {
            log.debug("Cache hit for key: {}", key);
            return cachedResult;
        }
        
        // 缓存未命中，执行方法
        log.debug("Cache miss for key: {}", key);
        Object result = joinPoint.proceed();
        
        // 将结果存入缓存
        if (result != null) {
            redisTemplate.opsForValue().set(key, result, cacheable.expire(), TimeUnit.SECONDS);
            log.debug("Cached result for key: {}", key);
        }
        
        return result;
    }

    /**
     * 缓存清除处理
     */
    @Around("@annotation(cacheEvict)")
    public Object handleCacheEvict(ProceedingJoinPoint joinPoint, CacheEvict cacheEvict) throws Throwable {
        Object result = joinPoint.proceed();
        
        // 清除缓存
        String pattern = generateCachePattern(joinPoint, cacheEvict);
        if (cacheEvict.allEntries()) {
            // 清除所有匹配的缓存
            redisTemplate.delete(redisTemplate.keys(pattern + "*"));
            log.debug("Evicted all cache entries matching pattern: {}", pattern);
        } else {
            // 清除指定缓存
            String key = generateCacheKey(joinPoint, cacheEvict);
            redisTemplate.delete(key);
            log.debug("Evicted cache entry for key: {}", key);
        }
        
        return result;
    }

    /**
     * 生成缓存键
     */
    private String generateCacheKey(ProceedingJoinPoint joinPoint, Cacheable cacheable) {
        String className = joinPoint.getTarget().getClass().getSimpleName();
        String methodName = joinPoint.getSignature().getName();
        Object[] args = joinPoint.getArgs();
        
        StringBuilder keyBuilder = new StringBuilder();
        keyBuilder.append(cacheable.value()).append(":")
                  .append(className).append(":")
                  .append(methodName);
        
        if (args != null && args.length > 0) {
            keyBuilder.append(":");
            for (Object arg : args) {
                keyBuilder.append(arg != null ? arg.toString() : "null").append("_");
            }
            // 移除最后一个下划线
            keyBuilder.setLength(keyBuilder.length() - 1);
        }
        
        return keyBuilder.toString();
    }

    /**
     * 生成缓存键（用于缓存清除）
     */
    private String generateCacheKey(ProceedingJoinPoint joinPoint, CacheEvict cacheEvict) {
        String className = joinPoint.getTarget().getClass().getSimpleName();
        String methodName = joinPoint.getSignature().getName();
        Object[] args = joinPoint.getArgs();
        
        StringBuilder keyBuilder = new StringBuilder();
        keyBuilder.append(cacheEvict.value()).append(":")
                  .append(className).append(":")
                  .append(methodName);
        
        if (args != null && args.length > 0) {
            keyBuilder.append(":");
            for (Object arg : args) {
                keyBuilder.append(arg != null ? arg.toString() : "null").append("_");
            }
            // 移除最后一个下划线
            keyBuilder.setLength(keyBuilder.length() - 1);
        }
        
        return keyBuilder.toString();
    }

    /**
     * 生成缓存模式（用于批量清除）
     */
    private String generateCachePattern(ProceedingJoinPoint joinPoint, CacheEvict cacheEvict) {
        String className = joinPoint.getTarget().getClass().getSimpleName();
        return cacheEvict.value() + ":" + className;
    }
}

/**
 * 缓存注解
 */
@Target({ElementType.METHOD})
@Retention(RetentionPolicy.RUNTIME)
@Documented
@interface Cacheable {
    String value() default "default";
    int expire() default 3600; // 默认1小时过期
}

/**
 * 缓存清除注解
 */
@Target({ElementType.METHOD})
@Retention(RetentionPolicy.RUNTIME)
@Documented
@interface CacheEvict {
    String value() default "default";
    boolean allEntries() default false;
}