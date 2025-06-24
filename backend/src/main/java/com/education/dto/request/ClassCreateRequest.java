package com.education.dto.request;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;
import java.time.LocalDateTime;

/**
 * 班级创建请求DTO
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
@Data
@Schema(description = "班级创建请求参数")
public class ClassCreateRequest {
    
    @Schema(description = "班级名称", example = "计算机科学与技术2024-1班")
    @NotBlank(message = "班级名称不能为空")
    @Size(max = 100, message = "班级名称长度不能超过100个字符")
    private String className;
    
    @Schema(description = "班级描述", example = "计算机科学与技术专业2024级1班")
    @Size(max = 500, message = "班级描述长度不能超过500个字符")
    private String description;
    
    @Schema(description = "年级", example = "2024")
    @NotNull(message = "年级不能为空")
    private Integer grade;
    
    @Schema(description = "专业", example = "计算机科学与技术")
    @NotBlank(message = "专业不能为空")
    @Size(max = 50, message = "专业名称长度不能超过50个字符")
    private String major;
    
    @Schema(description = "学院", example = "计算机学院")
    @NotBlank(message = "学院不能为空")
    @Size(max = 50, message = "学院名称长度不能超过50个字符")
    private String college;
    
    @Schema(description = "班主任ID", example = "1")
    @NotNull(message = "班主任ID不能为空")
    private Long teacherId;
    
    @Schema(description = "开班时间", example = "2024-09-01T08:00:00")
    private LocalDateTime startTime;
    
    @Schema(description = "结班时间", example = "2028-06-30T18:00:00")
    private LocalDateTime endTime;
    
    @Schema(description = "最大学生数", example = "50")
    private Integer maxStudents;
    
    @Schema(description = "班级状态", example = "ACTIVE")
    private String status = "ACTIVE";
}