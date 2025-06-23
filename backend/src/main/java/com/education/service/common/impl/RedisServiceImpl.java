package com.education.service.common.impl;

import com.education.service.common.RedisService;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.stereotype.Service;

import java.time.Duration;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.function.Supplier;
import java.util.stream.Collectors;

/**
 * Redis服务实现类
 */
@Service
public class RedisServiceImpl implements RedisService {
    
    @Autowired
    private RedisTemplate<String, Object> redisTemplate;
    
    @Autowired
    private ObjectMapper objectMapper;
    
    // ========== String操作 ==========
    
    @Override
    public void set(String key, Object value) {
        redisTemplate.opsForValue().set(key, value);
    }
    
    @Override
    public void set(String key, Object value, Duration timeout) {
        redisTemplate.opsForValue().set(key, value, timeout);
    }
    
    @Override
    public <T> T get(String key, Class<T> clazz) {
        Object value = redisTemplate.opsForValue().get(key);
        if (value == null) {
            return null;
        }
        return convertValue(value, clazz);
    }
    
    @Override
    public Boolean delete(String key) {
        return redisTemplate.delete(key);
    }
    
    @Override
    public Long delete(String... keys) {
        return redisTemplate.delete(Arrays.asList(keys));
    }
    
    @Override
    public Boolean hasKey(String key) {
        return redisTemplate.hasKey(key);
    }
    
    @Override
    public Boolean expire(String key, Duration timeout) {
        return redisTemplate.expire(key, timeout);
    }
    
    @Override
    public Duration getExpire(String key) {
        Long expire = redisTemplate.getExpire(key, TimeUnit.SECONDS);
        return expire != null && expire > 0 ? Duration.ofSeconds(expire) : null;
    }
    
    // ========== Hash操作 ==========
    
    @Override
    public void hSet(String key, String hashKey, Object value) {
        redisTemplate.opsForHash().put(key, hashKey, value);
    }
    
    @Override
    public <T> T hGet(String key, String hashKey, Class<T> clazz) {
        Object value = redisTemplate.opsForHash().get(key, hashKey);
        if (value == null) {
            return null;
        }
        return convertValue(value, clazz);
    }
    
    @Override
    public Map<String, Object> hGetAll(String key) {
        Map<Object, Object> entries = redisTemplate.opsForHash().entries(key);
        Map<String, Object> result = new HashMap<>();
        for (Map.Entry<Object, Object> entry : entries.entrySet()) {
            result.put(entry.getKey().toString(), entry.getValue());
        }
        return result;
    }
    
    @Override
    public Long hDelete(String key, String... hashKeys) {
        return redisTemplate.opsForHash().delete(key, (Object[]) hashKeys);
    }
    
    @Override
    public Boolean hHasKey(String key, String hashKey) {
        return redisTemplate.opsForHash().hasKey(key, hashKey);
    }
    
    // ========== List操作 ==========
    
    @Override
    public Long lLeftPush(String key, Object value) {
        return redisTemplate.opsForList().leftPush(key, value);
    }
    
    @Override
    public Long lRightPush(String key, Object value) {
        return redisTemplate.opsForList().rightPush(key, value);
    }
    
    @Override
    public <T> T lLeftPop(String key, Class<T> clazz) {
        Object value = redisTemplate.opsForList().leftPop(key);
        if (value == null) {
            return null;
        }
        return convertValue(value, clazz);
    }
    
    @Override
    public <T> T lRightPop(String key, Class<T> clazz) {
        Object value = redisTemplate.opsForList().rightPop(key);
        if (value == null) {
            return null;
        }
        return convertValue(value, clazz);
    }
    
    @Override
    public Long lSize(String key) {
        return redisTemplate.opsForList().size(key);
    }
    
    @Override
    public <T> List<T> lRange(String key, long start, long end, Class<T> clazz) {
        List<Object> values = redisTemplate.opsForList().range(key, start, end);
        if (values == null || values.isEmpty()) {
            return new ArrayList<>();
        }
        return values.stream()
                .map(value -> convertValue(value, clazz))
                .collect(Collectors.toList());
    }
    
    // ========== Set操作 ==========
    
    @Override
    public Long sAdd(String key, Object... values) {
        return redisTemplate.opsForSet().add(key, values);
    }
    
    @Override
    public Long sRemove(String key, Object... values) {
        return redisTemplate.opsForSet().remove(key, values);
    }
    
    @Override
    public Boolean sIsMember(String key, Object value) {
        return redisTemplate.opsForSet().isMember(key, value);
    }
    
    @Override
    public Long sSize(String key) {
        return redisTemplate.opsForSet().size(key);
    }
    
    @Override
    public <T> Set<T> sMembers(String key, Class<T> clazz) {
        Set<Object> values = redisTemplate.opsForSet().members(key);
        if (values == null || values.isEmpty()) {
            return new HashSet<>();
        }
        return values.stream()
                .map(value -> convertValue(value, clazz))
                .collect(Collectors.toSet());
    }
    
    // ========== ZSet操作 ==========
    
    @Override
    public Boolean zAdd(String key, Object value, double score) {
        return redisTemplate.opsForZSet().add(key, value, score);
    }
    
    @Override
    public Long zRemove(String key, Object... values) {
        return redisTemplate.opsForZSet().remove(key, values);
    }
    
    @Override
    public Long zSize(String key) {
        return redisTemplate.opsForZSet().size(key);
    }
    
    @Override
    public <T> Set<T> zRange(String key, long start, long end, Class<T> clazz) {
        Set<Object> values = redisTemplate.opsForZSet().range(key, start, end);
        if (values == null || values.isEmpty()) {
            return new LinkedHashSet<>();
        }
        return values.stream()
                .map(value -> convertValue(value, clazz))
                .collect(Collectors.toCollection(LinkedHashSet::new));
    }
    
    @Override
    public <T> Set<T> zRangeByScore(String key, double min, double max, Class<T> clazz) {
        Set<Object> values = redisTemplate.opsForZSet().rangeByScore(key, min, max);
        if (values == null || values.isEmpty()) {
            return new LinkedHashSet<>();
        }
        return values.stream()
                .map(value -> convertValue(value, clazz))
                .collect(Collectors.toCollection(LinkedHashSet::new));
    }
    
    // ========== 缓存操作 ==========
    
    @Override
    public <T> T getOrSet(String key, Class<T> clazz, Supplier<T> supplier, Duration timeout) {
        T value = get(key, clazz);
        if (value != null) {
            return value;
        }
        
        value = supplier.get();
        if (value != null) {
            set(key, value, timeout);
        }
        return value;
    }
    
    @Override
    public void flushAll() {
        Set<String> keys = redisTemplate.keys("*");
        if (keys != null && !keys.isEmpty()) {
            redisTemplate.delete(keys);
        }
    }
    
    @Override
    public Map<String, String> getInfo() {
        try {
            Properties info = redisTemplate.getConnectionFactory().getConnection().info();
            Map<String, String> result = new HashMap<>();
            for (String key : info.stringPropertyNames()) {
                result.put(key, info.getProperty(key));
            }
            return result;
        } catch (Exception e) {
            // 如果获取Redis信息失败，返回基本信息
            Map<String, String> result = new HashMap<>();
            result.put("status", "connected");
            result.put("version", "unknown");
            return result;
        }
    }
    
    // ========== 辅助方法 ==========
    
    /**
     * 类型转换辅助方法
     */
    @SuppressWarnings("unchecked")
    private <T> T convertValue(Object value, Class<T> clazz) {
        if (value == null) {
            return null;
        }
        
        if (clazz.isInstance(value)) {
            return (T) value;
        }
        
        if (clazz == String.class) {
            return (T) value.toString();
        }
        
        try {
            if (value instanceof String) {
                return objectMapper.readValue((String) value, clazz);
            } else {
                String json = objectMapper.writeValueAsString(value);
                return objectMapper.readValue(json, clazz);
            }
        } catch (JsonProcessingException e) {
            throw new RuntimeException("Failed to convert value to " + clazz.getSimpleName(), e);
        }
    }
}