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
 * 成绩实体类
 * 
 * @author Education Platform Team
 * @version 1.0.0
 */
@Data
@EqualsAndHashCode(callSuper = false)
@Accessors(chain = true)
@TableName("grade")
@Schema(description = "成绩信息")
public class Grade implements Serializable {

    private static final long serialVersionUID = 1L;

    @Schema(description = "成绩ID")
    @TableId(value = "id", type = IdType.ASSIGN_ID)
    private Long id;

    @Schema(description = "学生ID")
    @TableField("student_id")
    @NotNull(message = "学生ID不能为空")
    private Long studentId;

    @Schema(description = "课程ID")
    @TableField("course_id")
    @NotNull(message = "课程ID不能为空")
    private Long courseId;

    @Schema(description = "任务ID")
    @TableField("task_id")
    private Long taskId;

    @Schema(description = "成绩类型", example = "ASSIGNMENT")
    @TableField("grade_type")
    @NotNull(message = "成绩类型不能为空")
    private String gradeType;

    @Schema(description = "分数", example = "85.5")
    @TableField("score")
    private BigDecimal score;

    @Schema(description = "最大分数", example = "100.0")
    @TableField("max_score")
    private BigDecimal maxScore;

    @Schema(description = "百分比分数", example = "85.5")
    @TableField("percentage")
    private BigDecimal percentage;

    @Schema(description = "等级", example = "B+")
    @TableField("letter_grade")
    @Size(max = 10, message = "等级长度不能超过10个字符")
    private String letterGrade;

    @Schema(description = "绩点", example = "3.5")
    @TableField("gpa_points")
    private BigDecimal gpaPoints;

    @Schema(description = "权重", example = "0.3")
    @TableField("weight")
    private BigDecimal weight;

    @Schema(description = "加权分数", example = "25.65")
    @TableField("weighted_score")
    private BigDecimal weightedScore;

    @Schema(description = "是否通过", example = "true")
    @TableField("is_passed")
    private Boolean isPassed;

    @Schema(description = "评分教师ID")
    @TableField("grader_id")
    private Long graderId;

    @Schema(description = "评分时间")
    @TableField("grade_time")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime gradeTime;

    @Schema(description = "评语")
    @TableField("comments")
    @Size(max = 2000, message = "评语长度不能超过2000个字符")
    private String comments;

    @Schema(description = "评分详情")
    @TableField("grading_details")
    @Size(max = 3000, message = "评分详情长度不能超过3000个字符")
    private String gradingDetails;

    @Schema(description = "学期", example = "2024-1")
    @TableField("semester")
    @Size(max = 20, message = "学期长度不能超过20个字符")
    private String semester;

    @Schema(description = "学年", example = "2023-2024")
    @TableField("academic_year")
    @Size(max = 20, message = "学年长度不能超过20个字符")
    private String academicYear;

    @Schema(description = "成绩状态", example = "FINAL")
    @TableField("status")
    private String status;

    @Schema(description = "是否公布", example = "true")
    @TableField("is_published")
    private Boolean isPublished;

    @Schema(description = "公布时间")
    @TableField("publish_time")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime publishTime;

    @Schema(description = "提交时间")
    @TableField("submission_time")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime submissionTime;

    @Schema(description = "是否迟交", example = "false")
    @TableField("is_late")
    private Boolean isLate;

    @Schema(description = "迟交天数", example = "0")
    @TableField("late_days")
    private Integer lateDays;

    @Schema(description = "原始分数（扣分前）", example = "90.0")
    @TableField("original_score")
    private BigDecimal originalScore;

    @Schema(description = "扣分", example = "4.5")
    @TableField("deduction")
    private BigDecimal deduction;

    @Schema(description = "扣分原因")
    @TableField("deduction_reason")
    @Size(max = 500, message = "扣分原因长度不能超过500个字符")
    private String deductionReason;

    @Schema(description = "排名", example = "5")
    @TableField("rank")
    private Integer rank;

    @Schema(description = "总人数", example = "30")
    @TableField("total_students")
    private Integer totalStudents;

    @Schema(description = "百分位", example = "83.33")
    @TableField("percentile")
    private BigDecimal percentile;

    @Schema(description = "是否最佳成绩", example = "false")
    @TableField("is_best")
    private Boolean isBest;

    @Schema(description = "改进建议")
    @TableField("improvement_suggestions")
    @Size(max = 1000, message = "改进建议长度不能超过1000个字符")
    private String improvementSuggestions;

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
    @Schema(description = "学生信息")
    private Student student;

    @TableField(exist = false)
    @Schema(description = "课程信息")
    private Course course;

    @TableField(exist = false)
    @Schema(description = "任务信息")
    private Task task;

    @TableField(exist = false)
    @Schema(description = "评分教师信息")
    private Teacher grader;

    /**
     * 成绩类型枚举
     */
    public enum GradeType {
        ASSIGNMENT("ASSIGNMENT", "作业"),
        QUIZ("QUIZ", "测验"),
        EXAM("EXAM", "考试"),
        PROJECT("PROJECT", "项目"),
        PARTICIPATION("PARTICIPATION", "课堂参与"),
        ATTENDANCE("ATTENDANCE", "出勤"),
        FINAL_EXAM("FINAL_EXAM", "期末考试"),
        MIDTERM_EXAM("MIDTERM_EXAM", "期中考试"),
        LAB("LAB", "实验"),
        PRESENTATION("PRESENTATION", "演示"),
        DISCUSSION("DISCUSSION", "讨论"),
        OVERALL("OVERALL", "总评");

        private final String code;
        private final String description;

        GradeType(String code, String description) {
            this.code = code;
            this.description = description;
        }

        public String getCode() {
            return code;
        }

        public String getDescription() {
            return description;
        }

        public static GradeType fromCode(String code) {
            for (GradeType type : values()) {
                if (type.code.equals(code)) {
                    return type;
                }
            }
            throw new IllegalArgumentException("未知的成绩类型: " + code);
        }
    }

    /**
     * 成绩状态枚举
     */
    public enum Status {
        DRAFT("DRAFT", "草稿"),
        PENDING("PENDING", "待审核"),
        APPROVED("APPROVED", "已审核"),
        FINAL("FINAL", "最终成绩"),
        DISPUTED("DISPUTED", "有争议"),
        REVISED("REVISED", "已修订"),
        LOCKED("LOCKED", "已锁定");

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
            throw new IllegalArgumentException("未知的成绩状态: " + code);
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
     * 判断是否为最终成绩
     * 
     * @return 是否为最终成绩
     */
    public boolean isFinal() {
        return Status.FINAL.getCode().equals(this.status);
    }

    /**
     * 判断是否已锁定
     * 
     * @return 是否已锁定
     */
    public boolean isLocked() {
        return Status.LOCKED.getCode().equals(this.status);
    }

    /**
     * 判断是否有争议
     * 
     * @return 是否有争议
     */
    public boolean isDisputed() {
        return Status.DISPUTED.getCode().equals(this.status);
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
     * 获取成绩类型描述
     * 
     * @return 成绩类型描述
     */
    public String getGradeTypeDescription() {
        try {
            return GradeType.fromCode(this.gradeType).getDescription();
        } catch (IllegalArgumentException e) {
            return this.gradeType;
        }
    }

    /**
     * 判断是否通过
     * 
     * @return 是否通过
     */
    public boolean isPassedGrade() {
        return isPassed != null && isPassed;
    }

    /**
     * 判断是否已公布
     * 
     * @return 是否已公布
     */
    public boolean isPublishedGrade() {
        return isPublished != null && isPublished;
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
     * 判断是否最佳成绩
     * 
     * @return 是否最佳成绩
     */
    public boolean isBestGrade() {
        return isBest != null && isBest;
    }

    /**
     * 计算百分比分数
     */
    public void calculatePercentage() {
        if (score != null && maxScore != null && maxScore.compareTo(BigDecimal.ZERO) > 0) {
            this.percentage = score.divide(maxScore, 2, BigDecimal.ROUND_HALF_UP)
                                 .multiply(new BigDecimal("100"));
        }
    }

    /**
     * 计算加权分数
     */
    public void calculateWeightedScore() {
        if (score != null && weight != null) {
            this.weightedScore = score.multiply(weight);
        }
    }

    /**
     * 根据分数计算等级
     * 
     * @return 等级
     */
    public String calculateLetterGrade() {
        if (percentage == null) {
            calculatePercentage();
        }
        
        if (percentage == null) {
            return "N/A";
        }
        
        BigDecimal p = percentage;
        if (p.compareTo(new BigDecimal("97")) >= 0) {
            return "A+";
        } else if (p.compareTo(new BigDecimal("93")) >= 0) {
            return "A";
        } else if (p.compareTo(new BigDecimal("90")) >= 0) {
            return "A-";
        } else if (p.compareTo(new BigDecimal("87")) >= 0) {
            return "B+";
        } else if (p.compareTo(new BigDecimal("83")) >= 0) {
            return "B";
        } else if (p.compareTo(new BigDecimal("80")) >= 0) {
            return "B-";
        } else if (p.compareTo(new BigDecimal("77")) >= 0) {
            return "C+";
        } else if (p.compareTo(new BigDecimal("73")) >= 0) {
            return "C";
        } else if (p.compareTo(new BigDecimal("70")) >= 0) {
            return "C-";
        } else if (p.compareTo(new BigDecimal("67")) >= 0) {
            return "D+";
        } else if (p.compareTo(new BigDecimal("65")) >= 0) {
            return "D";
        } else if (p.compareTo(new BigDecimal("60")) >= 0) {
            return "D-";
        } else {
            return "F";
        }
    }

    /**
     * 根据等级计算绩点
     * 
     * @return 绩点
     */
    public BigDecimal calculateGpaPoints() {
        String grade = letterGrade != null ? letterGrade : calculateLetterGrade();
        
        switch (grade) {
            case "A+":
            case "A":
                return new BigDecimal("4.0");
            case "A-":
                return new BigDecimal("3.7");
            case "B+":
                return new BigDecimal("3.3");
            case "B":
                return new BigDecimal("3.0");
            case "B-":
                return new BigDecimal("2.7");
            case "C+":
                return new BigDecimal("2.3");
            case "C":
                return new BigDecimal("2.0");
            case "C-":
                return new BigDecimal("1.7");
            case "D+":
                return new BigDecimal("1.3");
            case "D":
                return new BigDecimal("1.0");
            case "D-":
                return new BigDecimal("0.7");
            case "F":
            default:
                return new BigDecimal("0.0");
        }
    }

    /**
     * 判断是否及格
     * 
     * @param passingScore 及格分数
     * @return 是否及格
     */
    public boolean isPassing(BigDecimal passingScore) {
        if (score == null || passingScore == null) {
            return false;
        }
        return score.compareTo(passingScore) >= 0;
    }

    /**
     * 获取成绩等级描述
     * 
     * @return 成绩等级描述
     */
    public String getGradeDescription() {
        if (percentage == null) {
            calculatePercentage();
        }
        
        if (percentage == null) {
            return "未评分";
        }
        
        BigDecimal p = percentage;
        if (p.compareTo(new BigDecimal("90")) >= 0) {
            return "优秀";
        } else if (p.compareTo(new BigDecimal("80")) >= 0) {
            return "良好";
        } else if (p.compareTo(new BigDecimal("70")) >= 0) {
            return "中等";
        } else if (p.compareTo(new BigDecimal("60")) >= 0) {
            return "及格";
        } else {
            return "不及格";
        }
    }

    /**
     * 计算百分位
     * 
     * @param rank 排名
     * @param totalStudents 总人数
     */
    public void calculatePercentile(int rank, int totalStudents) {
        if (totalStudents > 0) {
            this.rank = rank;
            this.totalStudents = totalStudents;
            this.percentile = new BigDecimal(totalStudents - rank + 1)
                                .divide(new BigDecimal(totalStudents), 4, BigDecimal.ROUND_HALF_UP)
                                .multiply(new BigDecimal("100"));
        }
    }

    /**
     * 设置扣分信息
     * 
     * @param originalScore 原始分数
     * @param deduction 扣分
     * @param reason 扣分原因
     */
    public void setDeductionInfo(BigDecimal originalScore, BigDecimal deduction, String reason) {
        this.originalScore = originalScore;
        this.deduction = deduction;
        this.deductionReason = reason;
        if (originalScore != null && deduction != null) {
            this.score = originalScore.subtract(deduction).max(BigDecimal.ZERO);
        }
    }

    /**
     * 更新成绩信息
     * 
     * @param score 分数
     * @param maxScore 最大分数
     * @param graderId 评分教师ID
     * @param comments 评语
     */
    public void updateGrade(BigDecimal score, BigDecimal maxScore, Long graderId, String comments) {
        this.score = score;
        this.maxScore = maxScore;
        this.graderId = graderId;
        this.comments = comments;
        this.gradeTime = LocalDateTime.now();
        
        // 自动计算相关字段
        calculatePercentage();
        this.letterGrade = calculateLetterGrade();
        this.gpaPoints = calculateGpaPoints();
        calculateWeightedScore();
        
        // 判断是否通过（默认60分及格）
        this.isPassed = isPassing(new BigDecimal("60"));
    }

    /**
     * 获取成绩摘要
     * 
     * @return 成绩摘要
     */
    public String getGradeSummary() {
        StringBuilder summary = new StringBuilder();
        
        if (score != null) {
            summary.append("分数: ").append(score);
            if (maxScore != null) {
                summary.append("/").append(maxScore);
            }
        }
        
        if (percentage != null) {
            summary.append(" (").append(percentage).append("%)");
        }
        
        if (letterGrade != null) {
            summary.append(" - ").append(letterGrade);
        }
        
        if (rank != null && totalStudents != null) {
            summary.append(" - 排名: ").append(rank).append("/").append(totalStudents);
        }
        
        return summary.toString();
    }

    /**
     * 判断是否可以修改
     * 
     * @return 是否可以修改
     */
    public boolean canModify() {
        return !isLocked() && !isFinal();
    }

    /**
     * 判断是否为优秀成绩
     * 
     * @return 是否为优秀成绩
     */
    public boolean isExcellent() {
        if (percentage == null) {
            calculatePercentage();
        }
        return percentage != null && percentage.compareTo(new BigDecimal("90")) >= 0;
    }

    /**
     * 判断是否需要改进
     * 
     * @return 是否需要改进
     */
    public boolean needsImprovement() {
        if (percentage == null) {
            calculatePercentage();
        }
        return percentage != null && percentage.compareTo(new BigDecimal("70")) < 0;
    }
}