package com.education.dto;

import lombok.Data;

import java.time.LocalDateTime;

@Data
public class SectionCommentDTO {
    private Long id;
    private Long sectionId;
    private Long userId;
    private String content;
    private Long parentId;
    private LocalDateTime createTime;
    private LocalDateTime updateTime;
    
    // 额外的展示字段
    private String userName;
    private String userAvatar;
    private String userRole;
    private Integer replyCount;
    
    @Data
    public static class CreateRequest {
        private Long sectionId;
        private String content;
        private Long parentId;
    }
    
    @Data
    public static class UpdateRequest {
        private Long id;
        private String content;
    }
} 