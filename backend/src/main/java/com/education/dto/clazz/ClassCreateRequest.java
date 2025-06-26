package com.education.dto.clazz;

import lombok.Data;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;

/**
 * 班级创建请求DTO
 */
@Data
public class ClassCreateRequest {
    
    /**
     * 班级名称
     */
    @NotBlank(message = "班级名称不能为空")
    @Size(max = 50, message = "班级名称长度不能超过50个字符")
    private String name;
    
    /**
     * 年级
     */
    @NotBlank(message = "年级不能为空")
    private String grade;
    
    /**
     * 专业
     */
    @NotBlank(message = "专业不能为空")
    private String major;
    
    /**
     * 班级描述
     */
    @Size(max = 500, message = "班级描述长度不能超过500个字符")
    private String description;
    
    /**
     * 班级容量
     */
    private Integer capacity;
    
    /**
     * 学期
     */
    private String semester;
} 