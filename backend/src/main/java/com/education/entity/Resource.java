package com.education.entity;

import com.baomidou.mybatisplus.annotation.*;
import com.fasterxml.jackson.annotation.JsonFormat;
import com.fasterxml.jackson.annotation.JsonIgnore;
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
 * 资源实体类
 * 
 * @author Education Platform Team
 * @version 1.0.0
 */
@Data
@EqualsAndHashCode(callSuper = false)
@Accessors(chain = true)
@TableName("resource")
@Schema(description = "资源信息")
public class Resource implements Serializable {

    private static final long serialVersionUID = 1L;

    @Schema(description = "资源ID")
    @TableId(value = "id", type = IdType.ASSIGN_ID)
    private Long id;

    @Schema(description = "资源名称", example = "Java编程基础教程.pdf")
    @TableField("resource_name")
    @NotBlank(message = "资源名称不能为空")
    @Size(max = 200, message = "资源名称长度不能超过200个字符")
    private String resourceName;

    @Schema(description = "资源类型", example = "DOCUMENT")
    @TableField("resource_type")
    @NotBlank(message = "资源类型不能为空")
    private String resourceType;

    @Schema(description = "文件类型", example = "pdf")
    @TableField("file_type")
    @Size(max = 50, message = "文件类型长度不能超过50个字符")
    private String fileType;

    @Schema(description = "文件大小（字节）", example = "1024000")
    @TableField("file_size")
    private Long fileSize;

    @Schema(description = "文件路径")
    @TableField("file_path")
    @NotBlank(message = "文件路径不能为空")
    @Size(max = 500, message = "文件路径长度不能超过500个字符")
    private String filePath;

    @Schema(description = "文件URL")
    @TableField("file_url")
    @Size(max = 500, message = "文件URL长度不能超过500个字符")
    private String fileUrl;

    @Schema(description = "缩略图URL")
    @TableField("thumbnail_url")
    @Size(max = 500, message = "缩略图URL长度不能超过500个字符")
    private String thumbnailUrl;

    @Schema(description = "缩略图路径")
    @TableField("thumbnail_path")
    @Size(max = 500, message = "缩略图路径长度不能超过500个字符")
    private String thumbnailPath;

    @Schema(description = "课程ID")
    @TableField("course_id")
    private Long courseId;

    @Schema(description = "任务ID")
    @TableField("task_id")
    private Long taskId;

    @Schema(description = "上传者ID")
    @TableField("uploader_id")
    @NotNull(message = "上传者ID不能为空")
    private Long uploaderId;

    @Schema(description = "上传者类型", example = "TEACHER")
    @TableField("uploader_type")
    private String uploaderType;

    @Schema(description = "资源描述")
    @TableField("description")
    @Size(max = 2000, message = "资源描述长度不能超过2000个字符")
    private String description;

    @Schema(description = "资源分类", example = "COURSE_MATERIAL")
    @TableField("category")
    @Size(max = 100, message = "资源分类长度不能超过100个字符")
    private String category;

    @Schema(description = "标签")
    @TableField("tags")
    @Size(max = 500, message = "标签长度不能超过500个字符")
    private String tags;

    @Schema(description = "访问权限", example = "PUBLIC")
    @TableField("access_level")
    private String accessLevel;

    @Schema(description = "是否公开", example = "true")
    @TableField("is_public")
    private Boolean isPublic;

    @Schema(description = "下载次数", example = "150")
    @TableField("download_count")
    private Integer downloadCount;

    @Schema(description = "查看次数", example = "300")
    @TableField("view_count")
    private Integer viewCount;

    @Schema(description = "资源状态", example = "ACTIVE")
    @TableField("status")
    private String status;

    @Schema(description = "版本号", example = "1.0")
    @TableField("version")
    @Size(max = 20, message = "版本号长度不能超过20个字符")
    private String version;

    @Schema(description = "是否为最新版本", example = "true")
    @TableField("is_latest")
    private Boolean isLatest;

    @Schema(description = "父资源ID")
    @TableField("parent_id")
    private Long parentId;

    @Schema(description = "排序顺序", example = "1")
    @TableField("sort_order")
    private Integer sortOrder;

    @Schema(description = "有效期开始时间")
    @TableField("valid_from")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime validFrom;

    @Schema(description = "有效期结束时间")
    @TableField("valid_until")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime validUntil;

    @Schema(description = "MD5哈希值")
    @TableField("md5_hash")
    @Size(max = 32, message = "MD5哈希值长度不能超过32个字符")
    private String md5Hash;

    @Schema(description = "SHA256哈希值")
    @TableField("sha256_hash")
    @Size(max = 64, message = "SHA256哈希值长度不能超过64个字符")
    private String sha256Hash;

    @Schema(description = "MIME类型", example = "application/pdf")
    @TableField("mime_type")
    @Size(max = 100, message = "MIME类型长度不能超过100个字符")
    private String mimeType;

    @Schema(description = "编码格式", example = "UTF-8")
    @TableField("encoding")
    @Size(max = 50, message = "编码格式长度不能超过50个字符")
    private String encoding;

    @Schema(description = "元数据")
    @TableField("metadata")
    @Size(max = 2000, message = "元数据长度不能超过2000个字符")
    private String metadata;

    @Schema(description = "备注")
    @TableField("remarks")
    @Size(max = 1000, message = "备注长度不能超过1000个字符")
    private String remarks;

    @Schema(description = "创建时间")
    @TableField(value = "create_time", fill = FieldFill.INSERT)
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime createTime;

    @Schema(description = "更新时间")
    @TableField(value = "update_time", fill = FieldFill.INSERT_UPDATE)
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime updateTime;

    @Schema(description = "是否删除")
    @TableField("is_deleted")
    @TableLogic
    @JsonIgnore
    private Boolean isDeleted;

    @Schema(description = "扩展字段1")
    @TableField("ext_field1")
    private String extField1;

    @Schema(description = "扩展字段2")
    @TableField("ext_field2")
    private String extField2;

    @Schema(description = "扩展字段3")
    @TableField("ext_field3")
    private String extField3;

    // 关联信息（非数据库字段）
    @TableField(exist = false)
    @Schema(description = "课程信息")
    private Course course;

    @TableField(exist = false)
    @Schema(description = "任务信息")
    private Task task;

    @TableField(exist = false)
    @Schema(description = "上传者信息")
    private User uploader;

    /**
     * 资源类型枚举
     */
    public enum ResourceType {
        DOCUMENT("DOCUMENT", "文档"),
        VIDEO("VIDEO", "视频"),
        AUDIO("AUDIO", "音频"),
        IMAGE("IMAGE", "图片"),
        ARCHIVE("ARCHIVE", "压缩包"),
        CODE("CODE", "代码文件"),
        PRESENTATION("PRESENTATION", "演示文稿"),
        SPREADSHEET("SPREADSHEET", "电子表格"),
        LINK("LINK", "链接"),
        OTHER("OTHER", "其他");

        private final String code;
        private final String description;

        ResourceType(String code, String description) {
            this.code = code;
            this.description = description;
        }

        public String getCode() {
            return code;
        }

        public String getDescription() {
            return description;
        }

        public static ResourceType fromCode(String code) {
            for (ResourceType type : values()) {
                if (type.code.equals(code)) {
                    return type;
                }
            }
            throw new IllegalArgumentException("未知的资源类型: " + code);
        }
    }

    /**
     * 访问权限枚举
     */
    public enum AccessLevel {
        PUBLIC("PUBLIC", "公开"),
        COURSE_ONLY("COURSE_ONLY", "仅课程内"),
        TEACHER_ONLY("TEACHER_ONLY", "仅教师"),
        STUDENT_ONLY("STUDENT_ONLY", "仅学生"),
        PRIVATE("PRIVATE", "私有"),
        RESTRICTED("RESTRICTED", "受限");

        private final String code;
        private final String description;

        AccessLevel(String code, String description) {
            this.code = code;
            this.description = description;
        }

        public String getCode() {
            return code;
        }

        public String getDescription() {
            return description;
        }

        public static AccessLevel fromCode(String code) {
            for (AccessLevel level : values()) {
                if (level.code.equals(code)) {
                    return level;
                }
            }
            throw new IllegalArgumentException("未知的访问权限: " + code);
        }
    }

    /**
     * 资源状态枚举
     */
    public enum Status {
        UPLOADING("UPLOADING", "上传中"),
        ACTIVE("ACTIVE", "可用"),
        PROCESSING("PROCESSING", "处理中"),
        ARCHIVED("ARCHIVED", "已归档"),
        DELETED("DELETED", "已删除"),
        CORRUPTED("CORRUPTED", "已损坏"),
        QUARANTINED("QUARANTINED", "已隔离");

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

        public static Status fromCode(String code) {
            for (Status status : values()) {
                if (status.code.equals(code)) {
                    return status;
                }
            }
            throw new IllegalArgumentException("未知的资源状态: " + code);
        }
    }

    /**
     * 上传者类型枚举
     */
    public enum UploaderType {
        TEACHER("TEACHER", "教师"),
        STUDENT("STUDENT", "学生"),
        ADMIN("ADMIN", "管理员"),
        SYSTEM("SYSTEM", "系统");

        private final String code;
        private final String description;

        UploaderType(String code, String description) {
            this.code = code;
            this.description = description;
        }

        public String getCode() {
            return code;
        }

        public String getDescription() {
            return description;
        }

        public static UploaderType fromCode(String code) {
            for (UploaderType type : values()) {
                if (type.code.equals(code)) {
                    return type;
                }
            }
            throw new IllegalArgumentException("未知的上传者类型: " + code);
        }
    }

    /**
     * 判断是否为文档类型
     * 
     * @return 是否为文档类型
     */
    public boolean isDocument() {
        return ResourceType.DOCUMENT.getCode().equals(this.resourceType);
    }

    /**
     * 判断是否为视频类型
     * 
     * @return 是否为视频类型
     */
    public boolean isVideo() {
        return ResourceType.VIDEO.getCode().equals(this.resourceType);
    }

    /**
     * 判断是否为音频类型
     * 
     * @return 是否为音频类型
     */
    public boolean isAudio() {
        return ResourceType.AUDIO.getCode().equals(this.resourceType);
    }

    /**
     * 判断是否为图片类型
     * 
     * @return 是否为图片类型
     */
    public boolean isImage() {
        return ResourceType.IMAGE.getCode().equals(this.resourceType);
    }

    /**
     * 判断是否为可用状态
     * 
     * @return 是否为可用状态
     */
    public boolean isActive() {
        return Status.ACTIVE.getCode().equals(this.status);
    }

    /**
     * 判断是否正在上传
     * 
     * @return 是否正在上传
     */
    public boolean isUploading() {
        return Status.UPLOADING.getCode().equals(this.status);
    }

    /**
     * 判断是否正在处理
     * 
     * @return 是否正在处理
     */
    public boolean isProcessing() {
        return Status.PROCESSING.getCode().equals(this.status);
    }

    /**
     * 判断是否已归档
     * 
     * @return 是否已归档
     */
    public boolean isArchived() {
        return Status.ARCHIVED.getCode().equals(this.status);
    }

    /**
     * 判断是否公开
     * 
     * @return 是否公开
     */
    public boolean isPublicResource() {
        return isPublic != null && isPublic && AccessLevel.PUBLIC.getCode().equals(this.accessLevel);
    }

    /**
     * 判断是否为最新版本
     * 
     * @return 是否为最新版本
     */
    public boolean isLatestVersion() {
        return isLatest != null && isLatest;
    }

    /**
     * 判断是否在有效期内
     * 
     * @return 是否在有效期内
     */
    public boolean isValid() {
        LocalDateTime now = LocalDateTime.now();
        
        boolean afterValidFrom = validFrom == null || !now.isBefore(validFrom);
        boolean beforeValidUntil = validUntil == null || !now.isAfter(validUntil);
        
        return afterValidFrom && beforeValidUntil;
    }

    /**
     * 判断是否即将过期（距离过期时间不足一周）
     * 
     * @return 是否即将过期
     */
    public boolean isExpiringSoon() {
        if (validUntil == null) {
            return false;
        }
        return LocalDateTime.now().plusWeeks(1).isAfter(validUntil);
    }

    /**
     * 获取状态描述
     * 
     * @return 状态描述
     */
    public String getStatusDescription() {
        try {
            return Status.fromCode(this.status).getDescription();
        } catch (IllegalArgumentException e) {
            return this.status;
        }
    }

    /**
     * 获取资源类型描述
     * 
     * @return 资源类型描述
     */
    public String getResourceTypeDescription() {
        try {
            return ResourceType.fromCode(this.resourceType).getDescription();
        } catch (IllegalArgumentException e) {
            return this.resourceType;
        }
    }

    /**
     * 获取访问权限描述
     * 
     * @return 访问权限描述
     */
    public String getAccessLevelDescription() {
        try {
            return AccessLevel.fromCode(this.accessLevel).getDescription();
        } catch (IllegalArgumentException e) {
            return this.accessLevel;
        }
    }

    /**
     * 获取上传者类型描述
     * 
     * @return 上传者类型描述
     */
    public String getUploaderTypeDescription() {
        try {
            return UploaderType.fromCode(this.uploaderType).getDescription();
        } catch (IllegalArgumentException e) {
            return this.uploaderType;
        }
    }

    /**
     * 获取格式化的文件大小
     * 
     * @return 格式化的文件大小
     */
    public String getFormattedFileSize() {
        if (fileSize == null || fileSize <= 0) {
            return "0 B";
        }
        
        String[] units = {"B", "KB", "MB", "GB", "TB"};
        int unitIndex = 0;
        double size = fileSize.doubleValue();
        
        while (size >= 1024 && unitIndex < units.length - 1) {
            size /= 1024;
            unitIndex++;
        }
        
        return String.format("%.2f %s", size, units[unitIndex]);
    }

    /**
     * 获取标签列表
     * 
     * @return 标签列表
     */
    public String[] getTagsList() {
        if (tags == null || tags.trim().isEmpty()) {
            return new String[0];
        }
        return tags.split(",");
    }

    /**
     * 设置标签列表
     * 
     * @param tagsList 标签列表
     */
    public void setTagsList(String[] tagsList) {
        if (tagsList == null || tagsList.length == 0) {
            this.tags = null;
        } else {
            this.tags = String.join(",", tagsList);
        }
    }

    /**
     * 增加下载次数
     */
    public void incrementDownloadCount() {
        this.downloadCount = (downloadCount != null ? downloadCount : 0) + 1;
    }

    /**
     * 增加查看次数
     */
    public void incrementViewCount() {
        this.viewCount = (viewCount != null ? viewCount : 0) + 1;
    }

    /**
     * 判断是否可以下载
     * 
     * @return 是否可以下载
     */
    public boolean canDownload() {
        return isActive() && isValid() && !isDeleted;
    }

    /**
     * 判断是否可以预览
     * 
     * @return 是否可以预览
     */
    public boolean canPreview() {
        if (!canDownload()) {
            return false;
        }
        
        // 根据文件类型判断是否支持预览
        if (fileType == null) {
            return false;
        }
        
        String type = fileType.toLowerCase();
        return type.equals("pdf") || type.equals("txt") || type.equals("md") ||
               type.equals("jpg") || type.equals("jpeg") || type.equals("png") || type.equals("gif") ||
               type.equals("mp4") || type.equals("webm") || type.equals("ogg") ||
               type.equals("mp3") || type.equals("wav");
    }

    /**
     * 获取文件扩展名
     * 
     * @return 文件扩展名
     */
    public String getFileExtension() {
        if (resourceName == null) {
            return null;
        }
        
        int lastDotIndex = resourceName.lastIndexOf('.');
        if (lastDotIndex > 0 && lastDotIndex < resourceName.length() - 1) {
            return resourceName.substring(lastDotIndex + 1).toLowerCase();
        }
        
        return null;
    }

    /**
     * 判断是否为媒体文件
     * 
     * @return 是否为媒体文件
     */
    public boolean isMediaFile() {
        return isVideo() || isAudio() || isImage();
    }

    /**
     * 获取资源摘要信息
     * 
     * @return 资源摘要信息
     */
    public String getResourceSummary() {
        StringBuilder summary = new StringBuilder();
        
        summary.append(resourceName);
        
        if (fileSize != null) {
            summary.append(" (").append(getFormattedFileSize()).append(")");
        }
        
        if (downloadCount != null && downloadCount > 0) {
            summary.append(" - 下载: ").append(downloadCount).append("次");
        }
        
        return summary.toString();
    }

    /**
     * 检查文件完整性
     * 
     * @param actualMd5 实际MD5值
     * @return 是否完整
     */
    public boolean checkIntegrity(String actualMd5) {
        return md5Hash != null && md5Hash.equals(actualMd5);
    }
}