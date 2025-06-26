package com.education.dto.course;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.util.List;

/**
 * 课程列表DTO
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
@Data
@Schema(description = "课程列表响应")
public class CourseListDTO {
    
    @Schema(description = "总记录数", example = "50")
    private Long total;
    
    @Schema(description = "当前页码", example = "1")
    private Integer page;
    
    @Schema(description = "每页大小", example = "12")
    private Integer size;
    
    @Schema(description = "总页数", example = "5")
    private Integer totalPages;
    
    @Schema(description = "课程列表")
    private List<CourseDetailDTO> courses;
    
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