package com.education.dto.common;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.fasterxml.jackson.annotation.JsonInclude;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.io.Serializable;
import java.util.Collections;
import java.util.List;

/**
 * 分页响应结果类
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @param <T> 数据类型
 */
@Data
@NoArgsConstructor
@JsonInclude(JsonInclude.Include.NON_NULL)
@Schema(description = "分页响应结果")
public class PageResponse<T> implements Serializable {

    private static final long serialVersionUID = 1L;

    @Schema(description = "当前页码", example = "1")
    private Long current;

    @Schema(description = "每页大小", example = "20")
    private Long size;

    @Schema(description = "总记录数", example = "100")
    private Long total;

    @Schema(description = "总页数", example = "5")
    private Long pages;

    @Schema(description = "数据列表")
    private List<T> records;

    @Schema(description = "是否有上一页")
    private Boolean hasPrevious;

    @Schema(description = "是否有下一页")
    private Boolean hasNext;

    @Schema(description = "是否为第一页")
    private Boolean isFirst;

    @Schema(description = "是否为最后一页")
    private Boolean isLast;

    public PageResponse(Long current, Long size, Long total, List<T> records) {
        this.current = current;
        this.size = size;
        this.total = total;
        this.records = records != null ? records : Collections.emptyList();
        this.pages = calculatePages(total, size);
        this.hasPrevious = current > 1;
        this.hasNext = current < this.pages;
        this.isFirst = current == 1;
        this.isLast = current.equals(this.pages) || this.pages == 0;
    }

    /**
     * 从MyBatis-Plus的IPage创建分页响应
     * 
     * @param page MyBatis-Plus分页对象
     * @param <T> 数据类型
     * @return 分页响应对象
     */
    public static <T> PageResponse<T> of(IPage<T> page) {
        return new PageResponse<>(
            page.getCurrent(),
            page.getSize(),
            page.getTotal(),
            page.getRecords()
        );
    }

    /**
     * 创建空的分页响应
     * 
     * @param current 当前页码
     * @param size 每页大小
     * @param <T> 数据类型
     * @return 空的分页响应对象
     */
    public static <T> PageResponse<T> empty(Long current, Long size) {
        return new PageResponse<>(current, size, 0L, Collections.emptyList());
    }

    // Compatibility setters for legacy code
    public void setList(List<T> list) {
        this.records = list;
    }
    
    public void setPageNum(Integer pageNum) {
        this.current = pageNum != null ? pageNum.longValue() : 1L;
    }
    
    public void setPageSize(Integer pageSize) {
        this.size = pageSize != null ? pageSize.longValue() : 20L;
    }
    
    /**
     * 创建单页响应
     * 
     * @param records 数据列表
     * @param <T> 数据类型
     * @return 单页响应对象
     */
    public static <T> PageResponse<T> of(List<T> records) {
        long total = records != null ? records.size() : 0;
        return new PageResponse<>(1L, total, total, records);
    }

    /**
     * 创建自定义分页响应
     * 
     * @param current 当前页码
     * @param size 每页大小
     * @param total 总记录数
     * @param records 数据列表
     * @param <T> 数据类型
     * @return 分页响应对象
     */
    public static <T> PageResponse<T> of(Long current, Long size, Long total, List<T> records) {
        return new PageResponse<>(current, size, total, records);
    }

    /**
     * 转换数据类型
     * 
     * @param mapper 转换函数
     * @param <R> 目标数据类型
     * @return 转换后的分页响应对象
     */
    public <R> PageResponse<R> map(java.util.function.Function<T, R> mapper) {
        List<R> mappedRecords = this.records.stream()
            .map(mapper)
            .collect(java.util.stream.Collectors.toList());
        return new PageResponse<>(this.current, this.size, this.total, mappedRecords);
    }

    /**
     * 计算总页数
     * 
     * @param total 总记录数
     * @param size 每页大小
     * @return 总页数
     */
    private Long calculatePages(Long total, Long size) {
        if (total == null || total <= 0 || size == null || size <= 0) {
            return 0L;
        }
        return (total + size - 1) / size;
    }

    /**
     * 获取开始记录索引（从1开始）
     * 
     * @return 开始记录索引
     */
    public Long getStartIndex() {
        if (current == null || size == null || current <= 0 || size <= 0) {
            return 0L;
        }
        return (current - 1) * size + 1;
    }

    /**
     * 获取结束记录索引
     * 
     * @return 结束记录索引
     */
    public Long getEndIndex() {
        if (current == null || size == null || total == null || current <= 0 || size <= 0) {
            return 0L;
        }
        long endIndex = current * size;
        return Math.min(endIndex, total);
    }

    /**
     * 判断是否为空页面
     * 
     * @return 是否为空
     */
    public Boolean isEmpty() {
        return records == null || records.isEmpty();
    }

    /**
     * 获取当前页记录数
     * 
     * @return 当前页记录数
     */
    public Integer getCurrentPageSize() {
        return records != null ? records.size() : 0;
    }

    /**
     * 获取分页信息摘要
     * 
     * @return 分页信息摘要
     */
    @Schema(description = "分页信息摘要")
    public String getSummary() {
        if (total == null || total == 0) {
            return "暂无数据";
        }
        return String.format("第 %d-%d 条，共 %d 条记录，第 %d/%d 页", 
            getStartIndex(), getEndIndex(), total, current, pages);
    }

    /**
     * 构建分页导航信息
     * 
     * @param displayPages 显示的页码数量
     * @return 分页导航信息
     */
    public PageNavigation buildNavigation(int displayPages) {
        return new PageNavigation(current, pages, displayPages);
    }

    /**
     * 分页导航信息
     */
    @Data
    @Schema(description = "分页导航信息")
    public static class PageNavigation {
        @Schema(description = "当前页码")
        private Long current;
        
        @Schema(description = "总页数")
        private Long total;
        
        @Schema(description = "开始页码")
        private Long start;
        
        @Schema(description = "结束页码")
        private Long end;
        
        @Schema(description = "页码列表")
        private List<Long> pages;

        public PageNavigation(Long current, Long total, int displayPages) {
            this.current = current;
            this.total = total;
            
            if (total <= displayPages) {
                this.start = 1L;
                this.end = total;
            } else {
                long half = displayPages / 2;
                this.start = Math.max(1, current - half);
                this.end = Math.min(total, this.start + displayPages - 1);
                
                if (this.end - this.start + 1 < displayPages) {
                    this.start = Math.max(1, this.end - displayPages + 1);
                }
            }
            
            this.pages = java.util.stream.LongStream.rangeClosed(start, end)
                .boxed()
                .collect(java.util.stream.Collectors.toList());
        }
    }
}