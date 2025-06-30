package com.education.entity;

import com.baomidou.mybatisplus.annotation.*;
import com.fasterxml.jackson.annotation.JsonFormat;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.experimental.Accessors;

import java.io.Serializable;
import java.math.BigDecimal;
import java.time.LocalDateTime;

/**
 * 课程实体类
 * 
 * @author Education Platform Team
 * @version 1.0.0
 */
@Data
@EqualsAndHashCode(callSuper = false)
@Accessors(chain = true)
@TableName("course")
@Schema(description = "课程信息")
public class Course implements Serializable {

    private static final long serialVersionUID = 1L;

    @Schema(description = "课程ID")
    @TableId(value = "id", type = IdType.AUTO)
    private Long id;

    @Schema(description = "课程名称", example = "Java编程基础")
    @TableField("title")
    private String title;
    
    // 前端兼容字段
    @Schema(description = "课程名称(前端兼容字段)", hidden = true)
    @TableField(exist = false)
    private String courseName;

    @Schema(description = "课程简介")
    @TableField("description")
    private String description;

    @Schema(description = "课程封面图片URL")
    @TableField("cover_image")
    private String coverImage;

    @Schema(description = "课程学分", example = "3.0")
    @TableField("credit")
    private BigDecimal credit;

    @Schema(description = "课程类型", example = "必修课")
    @TableField("course_type")
    private String courseType;
    
    // 前端兼容字段
    @Schema(description = "课程类型(前端兼容字段)", hidden = true)
    @TableField(exist = false)
    private String category;

    @Schema(description = "课程开始时间")
    @TableField("start_time")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime startTime;

    @Schema(description = "课程结束时间")
    @TableField("end_time")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime endTime;

    @Schema(description = "教师ID")
    @TableField("teacher_id")
    private Long teacherId;

    @Schema(description = "课程状态", example = "未开始")
    @TableField("status")
    private String status;

    @Schema(description = "学期", example = "2024-2025-1")
    @TableField("term")
    private String term;
    
    // 前端兼容字段
    @Schema(description = "学期(前端兼容字段)", hidden = true)
    @TableField(exist = false)
    private String semester;
    
    @Schema(description = "选课学生数量", example = "120")
    @TableField("student_count")
    private Integer studentCount;
    
    @Schema(description = "平均成绩", example = "85.5")
    @TableField("average_score")
    private BigDecimal averageScore;

    @Schema(description = "创建时间")
    @TableField(value = "create_time", fill = FieldFill.INSERT)
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime createTime;

    @Schema(description = "更新时间")
    @TableField(value = "update_time", fill = FieldFill.INSERT_UPDATE)
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime updateTime;
    
    /**
     * 课程类型枚举
     */
    public enum CourseType {
        REQUIRED("必修课", "必修课"),
        ELECTIVE("选修课", "选修课");

        private final String code;
        private final String description;

        CourseType(String code, String description) {
            this.code = code;
            this.description = description;
        }

        public String getCode() {
            return code;
        }

        public String getDescription() {
            return description;
        }
    }
    
    /**
     * 课程状态枚举
     */
    public enum Status {
        NOT_STARTED("未开始", "未开始"),
        IN_PROGRESS("进行中", "进行中"),
        FINISHED("已结束", "已结束");

        private final String code;
        private final String description;

        Status(String code, String description) {
            this.code = code;
            this.description = description;
        }

        public String getCode() {
            return code;
        }

        public String getDescription() {
            return description;
        }
    }
    
    /**
     * 计算课程进度
     * 
     * @return 课程进度百分比
     */
    @Schema(description = "课程进度", accessMode = Schema.AccessMode.READ_ONLY)
    public Integer getProgress() {
        LocalDateTime now = LocalDateTime.now();
        
        // 课程未开始
        if (startTime != null && now.isBefore(startTime)) {
            return 0;
        }
        
        // 课程已结束
        if (endTime != null && now.isAfter(endTime)) {
            return 100;
        }
        
        // 计算进度
        if (startTime != null && endTime != null) {
            long totalDuration = endTime.toEpochSecond(java.time.ZoneOffset.UTC) - startTime.toEpochSecond(java.time.ZoneOffset.UTC);
            if (totalDuration > 0) {
                long passedDuration = now.toEpochSecond(java.time.ZoneOffset.UTC) - startTime.toEpochSecond(java.time.ZoneOffset.UTC);
                return (int) (passedDuration * 100 / totalDuration);
            }
        }
        
        return 0;
    }

    @Override
    public String toString() {
        return "Course{" +
                "id=" + id +
                ", title='" + title + '\'' +
                ", description='" + (description != null ? description.substring(0, Math.min(description.length(), 20)) + "..." : null) + '\'' +
                ", coverImage='" + coverImage + '\'' +
                ", credit=" + credit +
                ", courseType='" + courseType + '\'' +
                ", startTime=" + startTime +
                ", endTime=" + endTime +
                ", teacherId=" + teacherId +
                ", status='" + status + '\'' +
                ", term='" + term + '\'' +
                ", studentCount=" + studentCount +
                ", averageScore=" + averageScore +
                ", createTime=" + createTime +
                ", updateTime=" + updateTime +
                '}';
    }
} 