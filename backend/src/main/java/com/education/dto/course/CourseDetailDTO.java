package com.education.dto.course;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.time.LocalDateTime;
import java.util.List;

/**
 * 课程详情DTO
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
@Data
@Schema(description = "课程详情信息")
public class CourseDetailDTO {
    
    @Schema(description = "课程ID", example = "1")
    private Long id;
    
    @Schema(description = "课程标题", example = "高等数学A")
    private String title;
    
    @Schema(description = "课程描述", example = "系统讲解高等数学的基本概念、理论和方法")
    private String description;
    
    @Schema(description = "详细描述", example = "本课程系统地介绍微积分的基本概念...")
    private String longDescription;
    
    @Schema(description = "讲师姓名", example = "张教授")
    private String instructor;
    
    @Schema(description = "所属大学", example = "清华大学")
    private String university;
    
    @Schema(description = "课程分类", example = "数学")
    private String category;
    
    @Schema(description = "难度等级", example = "beginner", allowableValues = {"beginner", "intermediate", "advanced"})
    private String level;
    
    @Schema(description = "学习人数", example = "15420")
    private Integer students;
    
    @Schema(description = "课程评分", example = "4.8")
    private Double rating;
    
    @Schema(description = "评价数量", example = "1250")
    private Integer reviewCount;
    
    @Schema(description = "课程时长", example = "16周")
    private String duration;
    
    @Schema(description = "学习强度", example = "每周4-6小时")
    private String effort;
    
    @Schema(description = "课程语言", example = "中文")
    private String language;
    
    @Schema(description = "课程封面图片", example = "/api/placeholder/300/200")
    private String image;
    
    @Schema(description = "当前价格", example = "0")
    private Integer price;
    
    @Schema(description = "原价", example = "299")
    private Integer originalPrice;
    
    @Schema(description = "开课时间")
    private String startDate;
    
    @Schema(description = "结课时间")
    private String endDate;
    
    @Schema(description = "是否已报名", example = "false")
    private Boolean enrolled;
    
    @Schema(description = "是否提供证书", example = "true")
    private Boolean certificate;
    
    @Schema(description = "课程标签")
    private List<String> tags;
    
    @Schema(description = "先修要求")
    private List<String> prerequisites;
    
    @Schema(description = "学习收获")
    private List<String> skills;
    
    @Schema(description = "讲师信息")
    private InstructorInfo instructorInfo;
    
    @Schema(description = "课程状态", example = "published", allowableValues = {"draft", "published", "archived"})
    private String status;
    
    @Schema(description = "创建时间")
    private LocalDateTime createTime;
    
    @Schema(description = "更新时间")
    private LocalDateTime updateTime;
    
    /**
     * 讲师信息内部类
     */
    @Data
    @Schema(description = "讲师信息")
    public static class InstructorInfo {
        
        @Schema(description = "讲师姓名", example = "张教授")
        private String name;
        
        @Schema(description = "讲师职称", example = "数学系教授")
        private String title;
        
        @Schema(description = "所属大学", example = "清华大学")
        private String university;
        
        @Schema(description = "讲师简介", example = "清华大学数学系教授，博士生导师...")
        private String bio;
        
        @Schema(description = "讲师头像", example = "/api/placeholder/100/100")
        private String avatar;
        
        @Schema(description = "开设课程数", example = "8")
        private Integer courses;
        
        @Schema(description = "教授学生数", example = "45000")
        private Integer students;
    }
}