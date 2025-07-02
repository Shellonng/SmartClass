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
 * 课程资源实体类
 * 
 * @author Education Platform Team
 * @version 1.0.0
 */
@Data
@EqualsAndHashCode(callSuper = false)
@Accessors(chain = true)
@TableName("course_resource")
@Schema(description = "课程资源信息")
public class CourseResource implements Serializable {

    private static final long serialVersionUID = 1L;

    @Schema(description = "资源ID")
    @TableId(value = "id", type = IdType.AUTO)
    private Long id;

    @Schema(description = "所属课程ID")
    @TableField("course_id")
    private Long courseId;

    @Schema(description = "资源名称", example = "第一章PPT")
    @TableField("name")
    private String name;

    @Schema(description = "文件类型", example = "pdf")
    @TableField("file_type")
    private String fileType;

    @Schema(description = "文件大小(字节)")
    @TableField("file_size")
    private Long fileSize;

    @Schema(description = "文件URL")
    @TableField("file_url")
    private String fileUrl;

    @Schema(description = "资源描述")
    @TableField("description")
    private String description;

    @Schema(description = "下载次数")
    @TableField("download_count")
    private Integer downloadCount;

    @Schema(description = "上传用户ID")
    @TableField("upload_user_id")
    private Long uploadUserId;

    @Schema(description = "创建时间")
    @TableField(value = "create_time", fill = FieldFill.INSERT)
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime createTime;

    @Schema(description = "更新时间")
    @TableField(value = "update_time", fill = FieldFill.INSERT_UPDATE)
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime updateTime;

    @Schema(description = "上传用户名")
    @TableField(exist = false)
    private String uploadUserName;

    @Schema(description = "所属课程名称")
    @TableField(exist = false)
    private String courseName;

    @Schema(description = "格式化的文件大小")
    @TableField(exist = false)
    private String formattedSize;

    /**
     * 获取格式化的文件大小
     */
    public String getFormattedSize() {
        if (fileSize == null) {
            return "0 B";
        }
        
        final String[] units = new String[] { "B", "KB", "MB", "GB", "TB" };
        int digitGroups = (int) (Math.log10(fileSize) / Math.log10(1024));
        
        // 保留两位小数
        return String.format("%.2f %s", fileSize / Math.pow(1024, digitGroups), units[digitGroups]);
    }
} 