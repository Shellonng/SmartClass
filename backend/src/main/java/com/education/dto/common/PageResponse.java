package com.education.dto.common;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.fasterxml.jackson.annotation.JsonInclude;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.AllArgsConstructor;
import lombok.Builder;
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
@Builder
@NoArgsConstructor
@AllArgsConstructor
@JsonInclude(JsonInclude.Include.NON_NULL)
@Schema(description = "分页响应结果")
public class PageResponse<T> implements Serializable {

    private static final long serialVersionUID = 1L;

    @Schema(description = "数据列表")
    private List<T> records;

    @Schema(description = "总记录数", example = "100")
    private Long total;

    @Schema(description = "当前页码", example = "1")
    private Integer current;

    @Schema(description = "每页大小", example = "10")
    private Integer pageSize;

    @Schema(description = "总页数", example = "10")
    private Long pages;

    @Schema(description = "是否有下一页")
    private Boolean hasNext;

    @Schema(description = "是否有上一页")
    private Boolean hasPrevious;

    @Schema(description = "分页导航信息")
    private PageNavigation navigation;

    /**
     * 构造函数 - 兼容旧版本
     */
    public PageResponse(Integer current, Integer pageSize, Long total, List<T> records) {
        this.current = current;
        this.pageSize = pageSize;
        this.total = total;
        this.records = records;
        this.pages = (total + pageSize - 1) / pageSize;
        this.hasNext = current < pages;
        this.hasPrevious = current > 1;
        this.navigation = new PageNavigation();
        this.navigation.setFirstPage(1);
        this.navigation.setLastPage(Math.toIntExact(pages));
        this.navigation.setNextPage(hasNext ? current + 1 : null);
        this.navigation.setPreviousPage(hasPrevious ? current - 1 : null);
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
            page.getCurrent().intValue(),
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
    public static <T> PageResponse<T> empty(Integer current, Integer size) {
        return new PageResponse<>(current, size, 0L, Collections.emptyList());
    }

    /**
     * 获取列表 - 兼容方法
     */
    public List<T> getList() {
        return this.records;
    }

    /**
     * 设置列表 - 兼容方法
     */
    public void setList(List<T> list) {
        this.records = list;
    }

    /**
     * 获取页码 - 兼容方法
     */
    public Integer getPageNum() {
        return this.current;
    }

    /**
     * 设置页码 - 兼容方法
     */
    public void setPageNum(Integer pageNum) {
        this.current = pageNum;
    }

    /**
     * 计算并设置分页信息
     */
    private void calculatePageInfo() {
        if (total != null && pageSize != null && pageSize > 0) {
            this.pages = (total + pageSize - 1) / pageSize;
            this.hasNext = current != null && current < pages;
            this.hasPrevious = current != null && current > 1;
            
            if (this.navigation == null) {
                this.navigation = new PageNavigation();
            }
            this.navigation.setFirstPage(1);
            this.navigation.setLastPage(Math.toIntExact(pages));
            this.navigation.setNextPage(hasNext ? current + 1 : null);
            this.navigation.setPreviousPage(hasPrevious ? current - 1 : null);
        }
    }

    /**
     * 设置总数时自动计算分页信息
     */
    public void setTotal(Long total) {
        this.total = total;
        calculatePageInfo();
    }

    /**
     * 设置当前页时自动计算分页信息
     */
    public void setCurrent(Integer current) {
        this.current = current;
        calculatePageInfo();
    }

    /**
     * 设置页面大小时自动计算分页信息
     */
    public void setPageSize(Integer pageSize) {
        this.pageSize = pageSize;
        calculatePageInfo();
    }

    /**
     * 静态builder方法 - 兼容链式调用
     */
    public static <T> PageResponseBuilder<T> builder() {
        return new PageResponseBuilder<T>();
    }

    /**
     * 创建单页响应
     * 
     * @param records 数据列表
     * @param <T> 数据类型
     * @return 单页响应对象
     */
    public static <T> PageResponse<T> of(List<T> records) {
        int total = records != null ? records.size() : 0;
        return new PageResponse<>(1, total, (long) total, records);
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
    public static <T> PageResponse<T> of(Integer current, Integer size, Long total, List<T> records) {
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
        return new PageResponse<>(this.current, this.pageSize, this.total, mappedRecords);
    }

    /**
     * 获取开始记录索引（从1开始）
     * 
     * @return 开始记录索引
     */
    public Long getStartIndex() {
        if (current == null || pageSize == null || current <= 0 || pageSize <= 0) {
            return 0L;
        }
        return (current - 1) * pageSize + 1;
    }

    /**
     * 获取结束记录索引
     * 
     * @return 结束记录索引
     */
    public Long getEndIndex() {
        if (current == null || pageSize == null || total == null || current <= 0 || pageSize <= 0) {
            return 0L;
        }
        long endIndex = current * pageSize;
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
        return new PageNavigation(current, Math.toIntExact(pages), displayPages);
    }

    /**
     * 分页导航信息
     */
    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class PageNavigation {
        private Integer firstPage;
        private Integer lastPage;
        private Integer nextPage;
        private Integer previousPage;

        public PageNavigation(Integer current, Integer total, int displayPages) {
            if (total <= displayPages) {
                this.firstPage = 1;
                this.lastPage = total;
            } else {
                int half = displayPages / 2;
                this.firstPage = Math.max(1, current - half);
                this.lastPage = Math.min(total, this.firstPage + displayPages - 1);
                
                if (this.lastPage - this.firstPage + 1 < displayPages) {
                    this.firstPage = Math.max(1, this.lastPage - displayPages + 1);
                }
            }
            
            this.nextPage = current < total ? current + 1 : null;
            this.previousPage = current > 1 ? current - 1 : null;
        }
    }
}