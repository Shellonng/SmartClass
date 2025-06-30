package com.education.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

import java.time.LocalDateTime;

@Data
@TableName("section_comment")
public class SectionComment {
    @TableId(type = IdType.AUTO)
    private Long id;
    private Long sectionId;
    private Long userId;
    private String content;
    private Long parentId;
    private LocalDateTime createTime;
    private LocalDateTime updateTime;
} 