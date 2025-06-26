package com.education.dto.clazz;

import lombok.Data;
import jakarta.validation.constraints.Size;

/**
 * 班级更新请求DTO
 */
@Data
public class ClassUpdateRequest {
    
    @Size(max = 50, message = "班级名称长度不能超过50个字符")
    private String name;
    
    private String grade;
    private String major;
    
    @Size(max = 500, message = "班级描述长度不能超过500个字符")
    private String description;
    
    private Integer capacity;
    private String semester;
    private String status;
} 