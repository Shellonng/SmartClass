package com.education.dto.course;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.util.List;

/**
 * 课程搜索结果DTO
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
@Data
@Schema(description = "课程搜索结果")
public class CourseSearchDTO {
    
    @Schema(description = "搜索关键词", example = "数学")
    private String keyword;
    
    @Schema(description = "搜索结果总数", example = "25")
    private Long total;
    
    @Schema(description = "当前页码", example = "1")
    private Integer page;
    
    @Schema(description = "每页大小", example = "12")
    private Integer size;
    
    @Schema(description = "总页数", example = "3")
    private Integer totalPages;
    
    @Schema(description = "搜索耗时（毫秒）", example = "45")
    private Long searchTime;
    
    @Schema(description = "课程列表")
    private List<CourseDetailDTO> courses;
    
    @Schema(description = "搜索建议")
    private List<String> suggestions;
    
    @Schema(description = "相关分类")
    private List<String> relatedCategories;
    
    /**
     * 计算总页数
     */
    public Integer getTotalPages() {
        if (total == null || size == null || size == 0) {
            return 0;
        }
        return (int) Math.ceil((double) total / size);
    }
}