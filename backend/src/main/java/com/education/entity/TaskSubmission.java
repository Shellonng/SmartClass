package com.education.entity;

import com.baomidou.mybatisplus.annotation.*;
import com.fasterxml.jackson.annotation.JsonFormat;
import com.fasterxml.jackson.annotation.JsonIgnore;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.experimental.Accessors;

import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;
import java.io.Serializable;
import java.math.BigDecimal;
import java.time.LocalDateTime;

/**
 * 任务提交实体类
 * 
 * @author Education Platform Team
 * @version 1.0.0
 */
@Data
@EqualsAndHashCode(callSuper = false)
@Accessors(chain = true)
@TableName("task_submission")
@Schema(description = "任务提交信息")
public class TaskSubmission implements Serializable {

    private static final long serialVersionUID = 1L;

    @Schema(description = "提交ID")
    @TableId(value = "id", type = IdType.ASSIGN_ID)
    private Long id;

    @Schema(description = "任务ID")
    @TableField("task_id")
    @NotNull(message = "任务ID不能为空")
    private Long taskId;

    @Schema(description = "学生ID")
    @TableField("student_id")
    @NotNull(message = "学生ID不能为空")
    private Long studentId;

    @Schema(description = "提交内容")
    @TableField("content")
    @Size(max = 10000, message = "提交内容长度不能超过10000个字符")
    private String content;

    @Schema(description = "提交文件")
    @TableField("files")
    @Size(max = 2000, message = "提交文件URL长度不能超过2000个字符")
    private String files;

    @Schema(description = "提交链接")
    @TableField("links")
    @Size(max = 1000, message = "提交链接长度不能超过1000个字符")
    private String links;

    @Schema(description = "提交时间")
    @TableField("submit_time")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime submitTime;

    @Schema(description = "是否迟交", example = "false")
    @TableField("is_late")
    private Boolean isLate;

    @Schema(description = "迟交天数", example = "0")
    @TableField("late_days")
    private Integer lateDays;

    @Schema(description = "提交状态", example = "SUBMITTED")
    @TableField("status")
    private String status;

    @Schema(description = "分数", example = "85.5")
    @TableField("score")
    private BigDecimal score;

    @Schema(description = "原始分数（扣分前）", example = "90.0")
    @TableField("original_score")
    private BigDecimal originalScore;

    @Schema(description = "扣分", example = "4.5")
    @TableField("deduction")
    private BigDecimal deduction;

    @Schema(description = "评分教师ID")
    @TableField("grader_id")
    private Long graderId;

    @Schema(description = "评分时间")
    @TableField("grade_time")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime gradeTime;

    @Schema(description = "教师评语")
    @TableField("feedback")
    @Size(max = 2000, message = "教师评语长度不能超过2000个字符")
    private String feedback;

    @Schema(description = "评分详情")
    @TableField("grading_details")
    @Size(max = 3000, message = "评分详情长度不能超过3000个字符")
    private String gradingDetails;

    @Schema(description = "提交次数", example = "1")
    @TableField("attempt_number")
    private Integer attemptNumber;

    @Schema(description = "是否最终提交", example = "true")
    @TableField("is_final")
    private Boolean isFinal;

    @Schema(description = "提交方式", example = "ONLINE")
    @TableField("submission_method")
    private String submissionMethod;

    @Schema(description = "IP地址", example = "192.168.1.100")
    @TableField("ip_address")
    @Size(max = 50, message = "IP地址长度不能超过50个字符")
    private String ipAddress;

    @Schema(description = "用户代理")
    @TableField("user_agent")
    @Size(max = 500, message = "用户代理长度不能超过500个字符")
    private String userAgent;

    @Schema(description = "文件大小（字节）", example = "1024000")
    @TableField("file_size")
    private Long fileSize;

    @Schema(description = "文件类型")
    @TableField("file_type")
    @Size(max = 100, message = "文件类型长度不能超过100个字符")
    private String fileType;

    @Schema(description = "查重结果")
    @TableField("plagiarism_result")
    @Size(max = 1000, message = "查重结果长度不能超过1000个字符")
    private String plagiarismResult;

    @Schema(description = "相似度百分比", example = "15.5")
    @TableField("similarity_percentage")
    private BigDecimal similarityPercentage;

    @Schema(description = "是否通过查重", example = "true")
    @TableField("plagiarism_passed")
    private Boolean plagiarismPassed;

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
    @Schema(description = "任务信息")
    private Task task;

    @TableField(exist = false)
    @Schema(description = "学生信息")
    private Student student;

    @TableField(exist = false)
    @Schema(description = "评分教师信息")
    private Teacher grader;

    /**
     * 提交状态枚举
     */
    public enum Status {
        DRAFT("DRAFT", "草稿"),
        SUBMITTED("SUBMITTED", "已提交"),
        GRADING("GRADING", "评分中"),
        GRADED("GRADED", "已评分"),
        RETURNED("RETURNED", "已退回"),
        RESUBMITTED("RESUBMITTED", "重新提交"),
        CANCELLED("CANCELLED", "已取消");

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
            throw new IllegalArgumentException("未知的提交状态: " + code);
        }
    }

    /**
     * 提交方式枚举
     */
    public enum SubmissionMethod {
        ONLINE("ONLINE", "在线提交"),
        FILE_UPLOAD("FILE_UPLOAD", "文件上传"),
        TEXT_INPUT("TEXT_INPUT", "文本输入"),
        LINK_SUBMISSION("LINK_SUBMISSION", "链接提交"),
        OFFLINE("OFFLINE", "线下提交"),
        EMAIL("EMAIL", "邮件提交");

        private final String code;
        private final String description;

        SubmissionMethod(String code, String description) {
            this.code = code;
            this.description = description;
        }

        public String getCode() {
            return code;
        }

        public String getDescription() {
            return description;
        }

        public static SubmissionMethod fromCode(String code) {
            for (SubmissionMethod method : values()) {
                if (method.code.equals(code)) {
                    return method;
                }
            }
            throw new IllegalArgumentException("未知的提交方式: " + code);
        }
    }

    /**
     * 判断是否为草稿状态
     * 
     * @return 是否为草稿状态
     */
    public boolean isDraft() {
        return Status.DRAFT.getCode().equals(this.status);
    }

    /**
     * 判断是否已提交
     * 
     * @return 是否已提交
     */
    public boolean isSubmitted() {
        return Status.SUBMITTED.getCode().equals(this.status) ||
               Status.GRADING.getCode().equals(this.status) ||
               Status.GRADED.getCode().equals(this.status) ||
               Status.RESUBMITTED.getCode().equals(this.status);
    }

    /**
     * 判断是否正在评分
     * 
     * @return 是否正在评分
     */
    public boolean isGrading() {
        return Status.GRADING.getCode().equals(this.status);
    }

    /**
     * 判断是否已评分
     * 
     * @return 是否已评分
     */
    public boolean isGraded() {
        return Status.GRADED.getCode().equals(this.status);
    }

    /**
     * 判断是否已退回
     * 
     * @return 是否已退回
     */
    public boolean isReturned() {
        return Status.RETURNED.getCode().equals(this.status);
    }

    /**
     * 判断是否重新提交
     * 
     * @return 是否重新提交
     */
    public boolean isResubmitted() {
        return Status.RESUBMITTED.getCode().equals(this.status);
    }

    /**
     * 判断是否已取消
     * 
     * @return 是否已取消
     */
    public boolean isCancelled() {
        return Status.CANCELLED.getCode().equals(this.status);
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
     * 获取提交方式描述
     * 
     * @return 提交方式描述
     */
    public String getSubmissionMethodDescription() {
        try {
            return SubmissionMethod.fromCode(this.submissionMethod).getDescription();
        } catch (IllegalArgumentException e) {
            return this.submissionMethod;
        }
    }

    /**
     * 判断是否迟交
     * 
     * @return 是否迟交
     */
    public boolean isLateSubmission() {
        return isLate != null && isLate;
    }

    /**
     * 判断是否有分数
     * 
     * @return 是否有分数
     */
    public boolean hasScore() {
        return score != null;
    }

    /**
     * 判断是否通过
     * 
     * @param passingScore 及格分数
     * @return 是否通过
     */
    public boolean isPassed(BigDecimal passingScore) {
        if (score == null || passingScore == null) {
            return false;
        }
        return score.compareTo(passingScore) >= 0;
    }

    /**
     * 获取分数等级
     * 
     * @return 分数等级
     */
    public String getScoreGrade() {
        if (score == null) {
            return "未评分";
        }
        
        BigDecimal scoreValue = score;
        if (scoreValue.compareTo(new BigDecimal("90")) >= 0) {
            return "优秀";
        } else if (scoreValue.compareTo(new BigDecimal("80")) >= 0) {
            return "良好";
        } else if (scoreValue.compareTo(new BigDecimal("70")) >= 0) {
            return "中等";
        } else if (scoreValue.compareTo(new BigDecimal("60")) >= 0) {
            return "及格";
        } else {
            return "不及格";
        }
    }

    /**
     * 获取文件列表
     * 
     * @return 文件列表
     */
    public String[] getFilesList() {
        if (files == null || files.trim().isEmpty()) {
            return new String[0];
        }
        return files.split(",");
    }

    /**
     * 设置文件列表
     * 
     * @param filesList 文件列表
     */
    public void setFilesList(String[] filesList) {
        if (filesList == null || filesList.length == 0) {
            this.files = null;
        } else {
            this.files = String.join(",", filesList);
        }
    }

    /**
     * 获取链接列表
     * 
     * @return 链接列表
     */
    public String[] getLinksList() {
        if (links == null || links.trim().isEmpty()) {
            return new String[0];
        }
        return links.split(",");
    }

    /**
     * 设置链接列表
     * 
     * @param linksList 链接列表
     */
    public void setLinksList(String[] linksList) {
        if (linksList == null || linksList.length == 0) {
            this.links = null;
        } else {
            this.links = String.join(",", linksList);
        }
    }

    /**
     * 计算迟交天数
     * 
     * @param dueTime 截止时间
     */
    public void calculateLateDays(LocalDateTime dueTime) {
        if (submitTime == null || dueTime == null) {
            this.isLate = false;
            this.lateDays = 0;
            return;
        }
        
        if (submitTime.isAfter(dueTime)) {
            this.isLate = true;
            this.lateDays = (int) java.time.Duration.between(dueTime, submitTime).toDays();
            if (this.lateDays == 0) {
                this.lateDays = 1; // 至少算一天
            }
        } else {
            this.isLate = false;
            this.lateDays = 0;
        }
    }

    /**
     * 设置评分信息
     * 
     * @param score 分数
     * @param feedback 评语
     * @param graderId 评分教师ID
     */
    public void setGradeInfo(BigDecimal score, String feedback, Long graderId) {
        this.score = score;
        this.feedback = feedback;
        this.graderId = graderId;
        this.gradeTime = LocalDateTime.now();
        this.status = Status.GRADED.getCode();
        this.updateTime = LocalDateTime.now();
    }

    /**
     * 设置评分教师ID（兼容性方法）
     * 
     * @param gradedBy 评分教师ID
     */
    public void setGradedBy(Long gradedBy) {
        this.graderId = gradedBy;
    }

    /**
     * 获取评分教师ID（兼容性方法）
     * 
     * @return 评分教师ID
     */
    public Long getGradedBy() {
        return this.graderId;
    }

    /**
     * 设置扣分信息
     * 
     * @param originalScore 原始分数
     * @param deduction 扣分
     */
    public void setDeductionInfo(BigDecimal originalScore, BigDecimal deduction) {
        this.originalScore = originalScore;
        this.deduction = deduction;
        if (originalScore != null && deduction != null) {
            this.score = originalScore.subtract(deduction);
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
     * 判断是否通过查重
     * 
     * @return 是否通过查重
     */
    public boolean isPlagiarismCheckPassed() {
        return plagiarismPassed != null && plagiarismPassed;
    }

    /**
     * 获取相似度等级
     * 
     * @return 相似度等级
     */
    public String getSimilarityLevel() {
        if (similarityPercentage == null) {
            return "未检测";
        }
        
        BigDecimal percentage = similarityPercentage;
        if (percentage.compareTo(new BigDecimal("30")) >= 0) {
            return "高";
        } else if (percentage.compareTo(new BigDecimal("15")) >= 0) {
            return "中";
        } else {
            return "低";
        }
    }

    /**
     * 判断是否为最终提交
     * 
     * @return 是否为最终提交
     */
    public boolean isFinalSubmission() {
        return isFinal != null && isFinal;
    }

    /**
     * 获取提交摘要信息
     * 
     * @return 提交摘要信息
     */
    public String getSubmissionSummary() {
        StringBuilder summary = new StringBuilder();
        
        summary.append("第").append(attemptNumber != null ? attemptNumber : 1).append("次提交");
        
        if (isLateSubmission()) {
            summary.append("（迟交").append(lateDays).append("天）");
        }
        
        if (hasScore()) {
            summary.append("，得分：").append(score);
        }
        
        return summary.toString();
    }

    /**
     * 判断是否可以重新提交
     * 
     * @param maxAttempts 最大提交次数
     * @return 是否可以重新提交
     */
    public boolean canResubmit(Integer maxAttempts) {
        if (isCancelled()) {
            return false;
        }
        
        if (maxAttempts == null) {
            return true; // 无限制
        }
        
        int currentAttempt = attemptNumber != null ? attemptNumber : 1;
        return currentAttempt < maxAttempts;
    }
}