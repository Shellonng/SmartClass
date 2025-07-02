package com.education.dto.common;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.fasterxml.jackson.annotation.JsonInclude;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.io.Serializable;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * 分页响应结果类
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @param <T> 数据类型
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
@JsonInclude(JsonInclude.Include.NON_NULL)
@Schema(description = "分页响应结果")
public class PageResponse<T> implements Serializable {

    private static final long serialVersionUID = 1L;

    @Schema(description = "数据列表")
    private List<T> content;

    @Schema(description = "总记录数", example = "100")
    private Long totalElements;

    @Schema(description = "当前页码", example = "1")
    private Integer number;

    @Schema(description = "每页大小", example = "10")
    private Integer size;

    @Schema(description = "总页数", example = "10")
    private Integer totalPages;

    @Schema(description = "是否为第一页")
    private Boolean first;

    @Schema(description = "是否为最后一页")
    private Boolean last;

    @Schema(description = "是否为空")
    private Boolean empty;

    /**
     * 兼容性构造函数，用于支持直接构造PageResponse(current, size, total, records)
     * 
     * @param current 当前页码
     * @param size 每页大小
     * @param total 总记录数
     * @param content 数据列表
     */
    public PageResponse(int current, int size, long total, List<T> content) {
        int totalPages = size > 0 ? (int) Math.ceil((double) total / size) : 0;
        
        this.content = content;
        this.totalElements = total;
        this.number = current - 1; // 转换为0-based索引
        this.size = size;
        this.totalPages = totalPages;
        this.first = (current == 1);
        this.last = (current == totalPages);
        this.empty = (content == null || content.isEmpty());
    }

    // 兼容方法 - 为了支持不同的命名约定
    
    // Records 命名约定兼容
    public List<T> getRecords() {
        return this.content;
    }
    
    public void setRecords(List<T> records) {
        this.content = records;
    }
    
    // List 命名约定兼容
    public List<T> getList() {
        return this.content;
    }

    public void setList(List<T> list) {
        this.content = list;
    }

    // Current 命名约定兼容
    public int getCurrent() {
        return this.number != null ? this.number + 1 : 1; // 转换回1-based索引
    }
    
    public void setCurrent(int current) {
        this.number = current - 1; // 存储为0-based索引
        this.first = (current == 1);
    }

    // PageNum 命名约定兼容
    public int getPageNum() {
        return getCurrent();
    }
    
    public void setPageNum(int pageNum) {
        setCurrent(pageNum);
    }

    public void setPageNum(Integer pageNum) {
        if (pageNum != null) {
            setCurrent(pageNum);
        }
    }
    
    // PageSize 命名约定兼容
    public int getPageSize() {
        return this.size != null ? this.size : 0;
    }
    
    public void setPageSize(int pageSize) {
        this.size = pageSize;
    }
    
    // Total 命名约定兼容
    public long getTotal() {
        return this.totalElements != null ? this.totalElements : 0;
        }
    
    public void setTotal(long total) {
        this.totalElements = total;
    }

    // Pages 命名约定兼容
    public long getPages() {
        return this.totalPages != null ? this.totalPages : 0;
    }

    public void setPages(long pages) {
        this.totalPages = (int) pages;
        this.last = (this.number != null && this.totalPages != null && (this.number + 1) == this.totalPages);
    }

    // 自定义 Builder 类
    public static class PageResponseBuilder<T> {
        private List<T> content;
        private Long totalElements;
        private Integer number;
        private Integer size;
        private Integer totalPages;
        private Boolean first;
        private Boolean last;
        private Boolean empty;
        
        public PageResponseBuilder() {
            // 空构造函数
        }
        
        // 原始字段设置方法
        public PageResponseBuilder<T> content(List<T> content) {
            this.content = content;
            return this;
        }
        
        public PageResponseBuilder<T> totalElements(Long totalElements) {
            this.totalElements = totalElements;
            return this;
        }
        
        public PageResponseBuilder<T> number(Integer number) {
            this.number = number;
            return this;
        }
        
        public PageResponseBuilder<T> size(Integer size) {
            this.size = size;
            return this;
        }
        
        public PageResponseBuilder<T> totalPages(Integer totalPages) {
            this.totalPages = totalPages;
            return this;
        }
        
        public PageResponseBuilder<T> first(Boolean first) {
            this.first = first;
            return this;
        }
        
        public PageResponseBuilder<T> last(Boolean last) {
            this.last = last;
            return this;
        }
        
        public PageResponseBuilder<T> empty(Boolean empty) {
            this.empty = empty;
            return this;
        }
        
        // 兼容方法 - 为了支持不同的命名约定
        
        // 支持 records() 方法
        public PageResponseBuilder<T> records(List<T> records) {
            this.content = records;
            return this;
        }
        
        // 支持 list() 方法
        public PageResponseBuilder<T> list(List<T> list) {
            this.content = list;
            return this;
        }

        // 支持 total() 方法
        public PageResponseBuilder<T> total(Long total) {
            this.totalElements = total;
            return this;
        }

        // 支持 current() 方法
        public PageResponseBuilder<T> current(Integer current) {
            this.number = current - 1; // 存储为0-based索引
            this.first = (current == 1);
            return this;
        }

        // 支持 pageNum() 方法
        public PageResponseBuilder<T> pageNum(Integer pageNum) {
            return current(pageNum);
        }
        
        // 支持 pageSize() 方法
        public PageResponseBuilder<T> pageSize(Integer pageSize) {
            this.size = pageSize;
            return this;
        }
        
        // 支持 pages() 方法
        public PageResponseBuilder<T> pages(Integer pages) {
            this.totalPages = pages;
            return this;
        }

        // build 方法
        public PageResponse<T> build() {
            PageResponse<T> response = new PageResponse<>();
            response.content = this.content;
            response.totalElements = this.totalElements;
            response.number = this.number;
            response.size = this.size;
            response.totalPages = this.totalPages;
            response.first = this.first;
            response.last = this.last;
            response.empty = this.empty;
            return response;
        }
    }
    
    // 提供静态builder方法以支持泛型
    public static <T> PageResponseBuilder<T> builder() {
        return new PageResponseBuilder<T>();
    }

    /**
     * 从MyBatis-Plus的IPage创建分页响应
     * 
     * @param page MyBatis-Plus分页对象
     * @param <T> 数据类型
     * @return 分页响应对象
     */
    public static <T> PageResponse<T> of(IPage<T> page) {
        return PageResponse.<T>builder()
                .content(page.getRecords())
                .totalElements(page.getTotal())
                .number((int) page.getCurrent() - 1) // 转换为0-based索引
                .size((int) page.getSize())
                .totalPages((int) page.getPages())
                .first(page.getCurrent() == 1)
                .last(page.getCurrent() == page.getPages())
                .empty(page.getRecords() == null || page.getRecords().isEmpty())
                .build();
    }

    /**
     * 创建自定义分页响应
     * 
     * @param pageNum 当前页码 (1-based)
     * @param pageSize 每页大小
     * @param total 总记录数
     * @param content 数据列表
     * @param <T> 数据类型
     * @return 分页响应对象
     */
    public static <T> PageResponse<T> of(int pageNum, int pageSize, long total, List<T> content) {
        int totalPages = pageSize > 0 ? (int) Math.ceil((double) total / pageSize) : 0;
        return PageResponse.<T>builder()
                .content(content)
                .totalElements(total)
                .number(pageNum - 1) // 转换为0-based索引
                .size(pageSize)
                .totalPages(totalPages)
                .first(pageNum == 1)
                .last(pageNum == totalPages)
                .empty(content == null || content.isEmpty())
                .build();
    }

    /**
     * 创建单页响应
     * 
     * @param content 数据列表
     * @param <T> 数据类型
     * @return 单页响应对象
     */
    public static <T> PageResponse<T> of(List<T> content) {
        int total = content != null ? content.size() : 0;
        return of(1, total, total, content);
    }

    /**
     * 创建空的分页响应
     * 
     * @param <T> 数据类型
     * @return 空的分页响应对象
     */
    public static <T> PageResponse<T> empty() {
        return of(1, 10, 0, Collections.emptyList());
                }

    /**
     * 创建空的分页响应（指定页码和大小）
     * 
     * @param pageNum 页码
     * @param pageSize 每页大小
     * @param <T> 数据类型
     * @return 空的分页响应对象
     */
    public static <T> PageResponse<T> empty(int pageNum, int pageSize) {
        return of(pageNum, pageSize, 0, Collections.emptyList());
    }
}