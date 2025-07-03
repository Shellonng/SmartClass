package com.education.entity;

import com.baomidou.mybatisplus.annotation.*;
import com.fasterxml.jackson.annotation.JsonFormat;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.experimental.Accessors;

import java.io.Serializable;
import java.time.LocalDateTime;

/**
 * 课程学生关联实体类
 */
@Data
@EqualsAndHashCode(callSuper = false)
@Accessors(chain = true)
@TableName("course_student")
@Schema(description = "课程学生关联信息")
public class CourseStudent implements Serializable {

    private static final long serialVersionUID = 1L;

    @Schema(description = "ID")
    @TableId(value = "id", type = IdType.AUTO)
    private Long id;

    @Schema(description = "课程ID")
    @TableField("course_id")
    private Long courseId;

    @Schema(description = "学生ID")
    @TableField("student_id")
    private Long studentId;

    @Schema(description = "选课时间")
    @TableField(value = "enroll_time", fill = FieldFill.INSERT)
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime enrollTime;
    
    // 非数据库字段
    @TableField(exist = false)
    @Schema(description = "课程信息")
    private Course course;
    
    @TableField(exist = false)
    @Schema(description = "学生信息")
    private Student student;
} 