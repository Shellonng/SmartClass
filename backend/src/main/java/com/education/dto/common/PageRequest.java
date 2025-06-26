package com.education.dto.common;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotNull;
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
    
    @Schema(description = "页码", example = "1")
    @NotNull(message = "页码不能为空")
    @Min(value = 1, message = "页码必须大于0")
    private Integer pageNum = 1;
    
    @Schema(description = "每页大小", example = "10")
    @NotNull(message = "每页大小不能为空")
    @Min(value = 1, message = "每页大小必须大于0")
    private Integer pageSize = 10;
    
    @Schema(description = "排序字段", example = "createTime")
    private String orderBy;
    
    @Schema(description = "排序方向", example = "desc")
    private String orderDirection = "desc";
    
    @Schema(description = "搜索关键词", example = "Java")
    private String keyword;

    /**
     * 获取偏移量
     */
    public Integer getOffset() {
        return (pageNum - 1) * pageSize;
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
     * 设置页码
     */
    public void setPage(Integer page) {
        this.pageNum = page;
    }

    /**
     * 设置每页大小
     */
    public void setSize(Integer size) {
        this.pageSize = size;
    }

    /**
     * 获取页码（兼容方法）
     */
    public Integer getPage() {
        return this.pageNum;
    }

    /**
     * 获取每页大小（兼容方法）
     */
    public Integer getSize() {
        return this.pageSize;
    }
}