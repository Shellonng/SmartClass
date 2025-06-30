package com.education.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.time.LocalDateTime;

@Data
@TableName("question_bank")
@Schema(description = "题库")
public class QuestionBank {
    @TableId(type = IdType.AUTO)
    @Schema(description = "题库ID")
    private Long id;
    
    @TableField("teacher_id")
    @Schema(description = "教师ID")
    private Long teacherId;
    
    @Schema(description = "题库标题")
    private String title;
    
    @Schema(description = "题库描述")
    private String description;
    
    @TableField("create_time")
    @Schema(description = "创建时间")
    private LocalDateTime createTime;
} 