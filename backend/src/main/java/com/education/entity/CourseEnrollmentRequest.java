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
 * 课程选课申请实体类
 */
@Data
@EqualsAndHashCode(callSuper = false)
@Accessors(chain = true)
@TableName("course_enrollment_request")
@Schema(description = "课程选课申请信息")
public class CourseEnrollmentRequest implements Serializable {

    private static final long serialVersionUID = 1L;

    @Schema(description = "申请ID")
    @TableId(value = "id", type = IdType.AUTO)
    private Long id;

    @Schema(description = "学生ID")
    @TableField("student_id")
    private Long studentId;

    @Schema(description = "课程ID")
    @TableField("course_id")
    private Long courseId;

    @Schema(description = "申请状态：0=待审核 1=已通过 2=已拒绝")
    @TableField("status")
    private Integer status;

    @Schema(description = "申请理由")
    @TableField("reason")
    private String reason;

    @Schema(description = "审核意见")
    @TableField("review_comment")
    private String reviewComment;

    @Schema(description = "提交时间")
    @TableField("submit_time")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime submitTime;

    @Schema(description = "审核时间")
    @TableField("review_time")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime reviewTime;
    
    // 非数据库字段
    @TableField(exist = false)
    @Schema(description = "学生信息")
    private Student student;
    
    @TableField(exist = false)
    @Schema(description = "课程信息")
    private Course course;
} 