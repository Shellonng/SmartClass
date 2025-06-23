package com.education.service.common;

import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Redis服务接口
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
public interface RedisService {
    
    // ========== String操作 ==========
    
    /**
     * 设置字符串值
     */
    void set(String key, Object value);
    
    /**
     * 设置字符串值并指定过期时间
     */
    void set(String key, Object value, Duration timeout);
    
    /**
     * 获取字符串值
     */
    <T> T get(String key, Class<T> clazz);
    
    /**
     * 删除key
     */
    Boolean delete(String key);
    
    /**
     * 批量删除key
     */
    Long delete(String... keys);
    
    /**
     * 判断key是否存在
     */
    Boolean hasKey(String key);
    
    /**
     * 设置过期时间
     */
    Boolean expire(String key, Duration timeout);
    
    /**
     * 获取过期时间
     */
    Duration getExpire(String key);
    
    // ========== Hash操作 ==========
    
    /**
     * 设置Hash值
     */
    void hSet(String key, String hashKey, Object value);
    
    /**
     * 获取Hash值
     */
    <T> T hGet(String key, String hashKey, Class<T> clazz);
    
    /**
     * 获取所有Hash值
     */
    Map<String, Object> hGetAll(String key);
    
    /**
     * 删除Hash字段
     */
    Long hDelete(String key, String... hashKeys);
    
    /**
     * 判断Hash字段是否存在
     */
    Boolean hHasKey(String key, String hashKey);
    
    // ========== List操作 ==========
    
    /**
     * 从左侧推入列表
     */
    Long lLeftPush(String key, Object value);
    
    /**
     * 从右侧推入列表
     */
    Long lRightPush(String key, Object value);
    
    /**
     * 从左侧弹出列表
     */
    <T> T lLeftPop(String key, Class<T> clazz);
    
    /**
     * 从右侧弹出列表
     */
    <T> T lRightPop(String key, Class<T> clazz);
    
    /**
     * 获取列表长度
     */
    Long lSize(String key);
    
    /**
     * 获取列表范围内的元素
     */
    <T> List<T> lRange(String key, long start, long end, Class<T> clazz);
    
    // ========== Set操作 ==========
    
    /**
     * 添加到集合
     */
    Long sAdd(String key, Object... values);
    
    /**
     * 从集合中移除
     */
    Long sRemove(String key, Object... values);
    
    /**
     * 判断是否是集合成员
     */
    Boolean sIsMember(String key, Object value);
    
    /**
     * 获取集合大小
     */
    Long sSize(String key);
    
    /**
     * 获取集合所有成员
     */
    <T> Set<T> sMembers(String key, Class<T> clazz);
    
    // ========== ZSet操作 ==========
    
    /**
     * 添加到有序集合
     */
    Boolean zAdd(String key, Object value, double score);
    
    /**
     * 从有序集合中移除
     */
    Long zRemove(String key, Object... values);
    
    /**
     * 获取有序集合大小
     */
    Long zSize(String key);
    
    /**
     * 获取有序集合范围内的元素
     */
    <T> Set<T> zRange(String key, long start, long end, Class<T> clazz);
    
    /**
     * 获取有序集合范围内的元素（按分数）
     */
    <T> Set<T> zRangeByScore(String key, double min, double max, Class<T> clazz);
    
    // ========== 缓存操作 ==========
    
    /**
     * 获取缓存，如果不存在则执行回调函数并缓存结果
     */
    <T> T getOrSet(String key, Class<T> clazz, java.util.function.Supplier<T> supplier, Duration timeout);
    
    /**
     * 清空所有缓存
     */
    void flushAll();
    
    /**
     * 获取Redis信息
     */
    Map<String, String> getInfo();
}