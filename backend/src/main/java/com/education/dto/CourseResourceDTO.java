package com.education.dto;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;
import lombok.experimental.Accessors;

/**
 * 课程资源DTO
 */
@Data
@Accessors(chain = true)
@Schema(description = "课程资源数据传输对象")
public class CourseResourceDTO {

    @Schema(description = "资源请求类")
    @Data
    @Accessors(chain = true)
    public static class ResourceRequest {
        @Schema(description = "课程ID")
        private Long courseId;
        
        @Schema(description = "资源名称")
        private String name;
        
        @Schema(description = "资源描述")
        private String description;
    }
    
    @Schema(description = "资源响应类")
    @Data
    @Accessors(chain = true)
    public static class ResourceResponse {
        @Schema(description = "资源ID")
        private Long id;
        
        @Schema(description = "课程ID")
        private Long courseId;
        
        @Schema(description = "资源名称")
        private String name;
        
        @Schema(description = "文件类型")
        private String fileType;
        
        @Schema(description = "文件大小(字节)")
        private Long fileSize;
        
        @Schema(description = "格式化的文件大小")
        private String formattedSize;
        
        @Schema(description = "文件URL")
        private String fileUrl;
        
        @Schema(description = "资源描述")
        private String description;
        
        @Schema(description = "下载次数")
        private Integer downloadCount;
        
        @Schema(description = "上传用户ID")
        private Long uploadUserId;
        
        @Schema(description = "上传用户名")
        private String uploadUserName;
        
        @Schema(description = "创建时间")
        private String createTime;
    }
} 