package com.education.entity;

import com.baomidou.mybatisplus.annotation.*;
import com.fasterxml.jackson.annotation.JsonFormat;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.experimental.Accessors;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;
import java.io.Serializable;
import java.time.LocalDateTime;

/**
 * 课程班级实体类
 */
@Data
@EqualsAndHashCode(callSuper = false)
@Accessors(chain = true)
@TableName("course_class")
@Schema(description = "课程班级信息")
public class CourseClass implements Serializable {

    private static final long serialVersionUID = 1L;

    @Schema(description = "班级ID")
    @TableId(value = "id", type = IdType.AUTO)
    private Long id;

    @Schema(description = "班级名称", example = "2025春季A班")
    @TableField("name")
    @NotBlank(message = "班级名称不能为空")
    @Size(max = 100, message = "班级名称不能超过100个字符")
    private String name;

    @Schema(description = "班级说明")
    @TableField("description")
    private String description;

    @Schema(description = "所属课程ID")
    @TableField("course_id")
    private Long courseId;

    @Schema(description = "创建教师ID")
    @TableField("teacher_id")
    private Long teacherId;

    @Schema(description = "是否为默认班级")
    @TableField("is_default")
    private Boolean isDefault;

    @Schema(description = "创建时间")
    @TableField(value = "create_time", fill = FieldFill.INSERT)
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime createTime;
    
    // 非数据库字段
    @TableField(exist = false)
    @Schema(description = "学生人数")
    private Integer studentCount;
    
    @TableField(exist = false)
    @Schema(description = "所属课程")
    private Course course;
} 