package com.education.dto.common;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.Max;
import java.util.Map;
import java.util.HashMap;

/**
 * 分页请求基类
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
@Data
@Schema(description = "分页请求参数")
public class PageRequest {
    
    @Schema(description = "页码，从1开始", example = "1")
    @Min(value = 1, message = "页码不能小于1")
    private Integer pageNum = 1;
    
    @Schema(description = "每页大小", example = "10")
    @Min(value = 1, message = "每页大小不能小于1")
    @Max(value = 100, message = "每页大小不能超过100")
    private Integer pageSize = 10;
    
    @Schema(description = "排序字段", example = "createTime")
    private String orderBy;
    
    @Schema(description = "排序方向，asc/desc", example = "desc")
    private String sortDirection = "desc";
    
    @Schema(description = "搜索关键词")
    private String keyword;

    /**
     * 获取偏移量
     */
    public Long getOffset() {
        return (long) (pageNum - 1) * pageSize;
    }
    
    /**
     * 获取过滤条件
     */
    public Map<String, Object> getFilters() {
        return new HashMap<>();
    }

    /**
     * 无参构造函数
     */
    public PageRequest() {
    }

    /**
     * 带参构造函数
     */
    public PageRequest(Integer page, Integer size) {
        this.pageNum = page;
        this.pageSize = size;
    }

    /**
     * 获取当前页码 - 兼容方法
     */
    public Integer getCurrent() {
        return this.pageNum;
    }

    /**
     * 设置当前页码 - 兼容方法
     */
    public void setCurrent(Integer current) {
        this.pageNum = current;
    }

    /**
     * 获取页码 - 兼容方法
     */
    public Integer getPage() {
        return this.pageNum;
    }

    /**
     * 设置页码 - 兼容方法
     */
    public void setPage(Integer page) {
        this.pageNum = page;
    }

    /**
     * 获取每页大小 - 兼容方法
     */
    public Integer getSize() {
        return this.pageSize;
    }

    /**
     * 设置每页大小 - 兼容方法
     */
    public void setSize(Integer size) {
        this.pageSize = size;
    }

    /**
     * 获取限制数量
     */
    public Integer getLimit() {
        return pageSize;
    }
}