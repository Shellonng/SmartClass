package com.education.dto.response;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.time.LocalDateTime;

/**
 * 班级响应DTO
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
@Data
@Schema(description = "班级响应数据")
public class ClassResponse {
    
    @Schema(description = "班级ID", example = "1")
    private Long classId;
    
    @Schema(description = "班级名称", example = "计算机科学与技术2024-1班")
    private String className;
    
    @Schema(description = "班级描述", example = "计算机科学与技术专业2024级1班")
    private String description;
    
    @Schema(description = "年级", example = "2024")
    private Integer grade;
    
    @Schema(description = "专业", example = "计算机科学与技术")
    private String major;
    
    @Schema(description = "学院", example = "计算机学院")
    private String college;
    
    @Schema(description = "班主任ID", example = "1")
    private Long teacherId;
    
    @Schema(description = "班主任姓名", example = "张老师")
    private String teacherName;
    
    @Schema(description = "开班时间", example = "2024-09-01T08:00:00")
    private LocalDateTime startTime;
    
    @Schema(description = "结班时间", example = "2028-06-30T18:00:00")
    private LocalDateTime endTime;
    
    @Schema(description = "最大学生数", example = "50")
    private Integer maxStudents;
    
    @Schema(description = "当前学生数", example = "45")
    private Integer currentStudents;
    
    @Schema(description = "班级状态", example = "ACTIVE")
    private String status;
    
    @Schema(description = "创建时间", example = "2024-01-01T12:00:00")
    private LocalDateTime createTime;
    
    @Schema(description = "更新时间", example = "2024-01-01T12:00:00")
    private LocalDateTime updateTime;
    
    @Schema(description = "创建者ID", example = "1")
    private Long createdBy;
    
    @Schema(description = "创建者姓名", example = "管理员")
    private String createdByName;
}