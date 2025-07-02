package com.education.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

@Data
@TableName("question_bank_item")
@Schema(description = "题库题目关联")
public class QuestionBankItem {
    @TableId(type = IdType.AUTO)
    @Schema(description = "ID")
    private Long id;
    
    @TableField("bank_id")
    @Schema(description = "题库ID")
    private Long bankId;
    
    @TableField("question_id")
    @Schema(description = "题目ID")
    private Long questionId;
} 