package com.education.dto;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;
import lombok.Builder;

import jakarta.validation.constraints.Email;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

/**
 * 邮件相关DTO类
 */
public class EmailDTO {

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    @Schema(description = "带附件邮件请求")
    public static class EmailWithAttachmentRequest {
        @NotBlank(message = "收件人不能为空")
        @Email(message = "收件人邮箱格式不正确")
        @Schema(description = "收件人邮箱")
        private String to;

        @Schema(description = "抄送人邮箱列表")
        private List<String> cc;

        @Schema(description = "密送人邮箱列表")
        private List<String> bcc;

        @NotBlank(message = "邮件主题不能为空")
        @Schema(description = "邮件主题")
        private String subject;

        @NotBlank(message = "邮件内容不能为空")
        @Schema(description = "邮件内容")
        private String content;

        @Schema(description = "附件文件路径列表")
        private List<String> attachmentPaths;

        @Schema(description = "是否为HTML格式")
        private Boolean isHtml = false;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    @Schema(description = "批量邮件请求")
    public static class BatchEmailRequest {
        @NotNull(message = "收件人列表不能为空")
        @Size(min = 1, message = "至少需要一个收件人")
        @Schema(description = "收件人邮箱列表")
        private List<String> recipients;

        @NotBlank(message = "邮件主题不能为空")
        @Schema(description = "邮件主题")
        private String subject;

        @NotBlank(message = "邮件内容不能为空")
        @Schema(description = "邮件内容")
        private String content;

        @Schema(description = "是否为HTML格式")
        private Boolean isHtml = false;

        @Schema(description = "发送优先级")
        private Integer priority = 1;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    @Schema(description = "批量邮件响应")
    public static class BatchEmailResponse {
        @Schema(description = "总发送数量")
        private Integer totalCount;

        @Schema(description = "成功发送数量")
        private Integer successCount;

        @Schema(description = "失败发送数量")
        private Integer failedCount;

        @Schema(description = "失败的邮箱列表")
        private List<String> failedEmails;

        @Schema(description = "批次ID")
        private String batchId;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    @Schema(description = "模板邮件请求")
    public static class TemplateEmailRequest {
        @NotNull(message = "模板ID不能为空")
        @Schema(description = "邮件模板ID")
        private Long templateId;

        @NotBlank(message = "收件人不能为空")
        @Email(message = "收件人邮箱格式不正确")
        @Schema(description = "收件人邮箱")
        private String to;

        @Schema(description = "模板变量")
        private Map<String, Object> variables;

        @Schema(description = "抄送人邮箱列表")
        private List<String> cc;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    @Schema(description = "通知邮件请求")
    public static class NotificationEmailRequest {
        @NotNull(message = "用户ID不能为空")
        @Schema(description = "用户ID")
        private Long userId;

        @NotBlank(message = "通知类型不能为空")
        @Schema(description = "通知类型")
        private String notificationType;

        @NotBlank(message = "通知标题不能为空")
        @Schema(description = "通知标题")
        private String title;

        @NotBlank(message = "通知内容不能为空")
        @Schema(description = "通知内容")
        private String content;

        @Schema(description = "相关数据")
        private Map<String, Object> data;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    @Schema(description = "课程提醒请求")
    public static class CourseReminderRequest {
        @NotNull(message = "课程ID不能为空")
        @Schema(description = "课程ID")
        private Long courseId;

        @NotNull(message = "学生ID列表不能为空")
        @Schema(description = "学生ID列表")
        private List<Long> studentIds;

        @NotBlank(message = "提醒类型不能为空")
        @Schema(description = "提醒类型")
        private String reminderType;

        @Schema(description = "课程开始时间")
        private LocalDateTime courseStartTime;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    @Schema(description = "任务截止提醒请求")
    public static class TaskDeadlineReminderRequest {
        @NotNull(message = "任务ID不能为空")
        @Schema(description = "任务ID")
        private Long taskId;

        @NotNull(message = "学生ID列表不能为空")
        @Schema(description = "学生ID列表")
        private List<Long> studentIds;

        @Schema(description = "截止时间")
        private LocalDateTime deadline;

        @Schema(description = "提前提醒时间（小时）")
        private Integer reminderHours = 24;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    @Schema(description = "成绩通知请求")
    public static class GradeNotificationRequest {
        @NotNull(message = "学生ID不能为空")
        @Schema(description = "学生ID")
        private Long studentId;

        @NotNull(message = "任务ID不能为空")
        @Schema(description = "任务ID")
        private Long taskId;

        @Schema(description = "成绩")
        private Double score;

        @Schema(description = "评语")
        private String comment;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    @Schema(description = "系统维护通知请求")
    public static class MaintenanceNotificationRequest {
        @NotNull(message = "用户ID列表不能为空")
        @Schema(description = "用户ID列表")
        private List<Long> userIds;

        @NotBlank(message = "维护标题不能为空")
        @Schema(description = "维护标题")
        private String title;

        @NotBlank(message = "维护内容不能为空")
        @Schema(description = "维护内容")
        private String content;

        @Schema(description = "维护开始时间")
        private LocalDateTime maintenanceStartTime;

        @Schema(description = "维护结束时间")
        private LocalDateTime maintenanceEndTime;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    @Schema(description = "邀请邮件请求")
    public static class InvitationEmailRequest {
        @NotBlank(message = "收件人邮箱不能为空")
        @Email(message = "收件人邮箱格式不正确")
        @Schema(description = "收件人邮箱")
        private String inviteeEmail;

        @NotBlank(message = "邀请人姓名不能为空")
        @Schema(description = "邀请人姓名")
        private String inviterName;

        @NotBlank(message = "邀请类型不能为空")
        @Schema(description = "邀请类型")
        private String invitationType;

        @Schema(description = "邀请链接")
        private String invitationLink;

        @Schema(description = "过期时间")
        private LocalDateTime expirationTime;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    @Schema(description = "邮件历史请求")
    public static class EmailHistoryRequest {
        @Schema(description = "用户ID")
        private Long userId;

        @Schema(description = "邮件类型")
        private String emailType;

        @Schema(description = "开始时间")
        private LocalDateTime startTime;

        @Schema(description = "结束时间")
        private LocalDateTime endTime;

        @Schema(description = "发送状态")
        private String status;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    @Schema(description = "邮件历史响应")
    public static class EmailHistoryResponse {
        @Schema(description = "邮件ID")
        private Long emailId;

        @Schema(description = "收件人")
        private String recipient;

        @Schema(description = "邮件主题")
        private String subject;

        @Schema(description = "发送时间")
        private LocalDateTime sentTime;

        @Schema(description = "发送状态")
        private String status;

        @Schema(description = "邮件类型")
        private String emailType;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    @Schema(description = "邮件统计请求")
    public static class EmailStatisticsRequest {
        @Schema(description = "统计开始时间")
        private LocalDateTime startTime;

        @Schema(description = "统计结束时间")
        private LocalDateTime endTime;

        @Schema(description = "统计维度")
        private String dimension;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    @Schema(description = "邮件统计响应")
    public static class EmailStatisticsResponse {
        @Schema(description = "总发送数量")
        private Long totalSent;

        @Schema(description = "成功发送数量")
        private Long successCount;

        @Schema(description = "失败发送数量")
        private Long failedCount;

        @Schema(description = "成功率")
        private Double successRate;

        @Schema(description = "统计详情")
        private Map<String, Object> details;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    @Schema(description = "邮件模板创建请求")
    public static class EmailTemplateCreateRequest {
        @NotBlank(message = "模板名称不能为空")
        @Schema(description = "模板名称")
        private String templateName;

        @NotBlank(message = "模板主题不能为空")
        @Schema(description = "模板主题")
        private String subject;

        @NotBlank(message = "模板内容不能为空")
        @Schema(description = "模板内容")
        private String content;

        @Schema(description = "模板类型")
        private String templateType;

        @Schema(description = "模板描述")
        private String description;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    @Schema(description = "邮件模板更新请求")
    public static class EmailTemplateUpdateRequest {
        @Schema(description = "模板名称")
        private String templateName;

        @Schema(description = "模板主题")
        private String subject;

        @Schema(description = "模板内容")
        private String content;

        @Schema(description = "模板描述")
        private String description;

        @Schema(description = "是否启用")
        private Boolean enabled;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    @Schema(description = "邮件模板响应")
    public static class EmailTemplateResponse {
        @Schema(description = "模板ID")
        private Long templateId;

        @Schema(description = "模板名称")
        private String templateName;

        @Schema(description = "模板主题")
        private String subject;

        @Schema(description = "模板类型")
        private String templateType;

        @Schema(description = "创建时间")
        private LocalDateTime createTime;

        @Schema(description = "是否启用")
        private Boolean enabled;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    @Schema(description = "邮件模板详情响应")
    public static class EmailTemplateDetailResponse {
        @Schema(description = "模板ID")
        private Long templateId;

        @Schema(description = "模板名称")
        private String templateName;

        @Schema(description = "模板主题")
        private String subject;

        @Schema(description = "模板内容")
        private String content;

        @Schema(description = "模板类型")
        private String templateType;

        @Schema(description = "模板描述")
        private String description;

        @Schema(description = "创建时间")
        private LocalDateTime createTime;

        @Schema(description = "更新时间")
        private LocalDateTime updateTime;

        @Schema(description = "是否启用")
        private Boolean enabled;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    @Schema(description = "邮件预览响应")
    public static class EmailPreviewResponse {
        @Schema(description = "预览主题")
        private String subject;

        @Schema(description = "预览内容")
        private String content;

        @Schema(description = "变量列表")
        private List<String> variables;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    @Schema(description = "邮件模板测试请求")
    public static class EmailTemplateTestRequest {
        @NotNull(message = "模板ID不能为空")
        @Schema(description = "模板ID")
        private Long templateId;

        @NotBlank(message = "测试邮箱不能为空")
        @Email(message = "测试邮箱格式不正确")
        @Schema(description = "测试邮箱")
        private String testEmail;

        @Schema(description = "测试变量")
        private Map<String, Object> testVariables;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    @Schema(description = "邮件配置响应")
    public static class EmailConfigResponse {
        @Schema(description = "SMTP服务器")
        private String smtpHost;

        @Schema(description = "SMTP端口")
        private Integer smtpPort;

        @Schema(description = "发件人邮箱")
        private String fromEmail;

        @Schema(description = "发件人名称")
        private String fromName;

        @Schema(description = "是否启用SSL")
        private Boolean sslEnabled;

        @Schema(description = "每日发送限制")
        private Integer dailyLimit;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    @Schema(description = "邮件配置更新请求")
    public static class EmailConfigUpdateRequest {
        @Schema(description = "SMTP服务器")
        private String smtpHost;

        @Schema(description = "SMTP端口")
        private Integer smtpPort;

        @Schema(description = "SMTP用户名")
        private String smtpUsername;

        @Schema(description = "SMTP密码")
        private String smtpPassword;

        @Schema(description = "发件人邮箱")
        private String fromEmail;

        @Schema(description = "发件人名称")
        private String fromName;

        @Schema(description = "是否启用SSL")
        private Boolean sslEnabled;

        @Schema(description = "每日发送限制")
        private Integer dailyLimit;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    @Schema(description = "邮件黑名单响应")
    public static class EmailBlacklistResponse {
        @Schema(description = "黑名单ID")
        private Long blacklistId;

        @Schema(description = "邮箱地址")
        private String email;

        @Schema(description = "添加原因")
        private String reason;

        @Schema(description = "添加时间")
        private LocalDateTime createTime;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    @Schema(description = "邮件订阅请求")
    public static class EmailSubscriptionRequest {
        @NotNull(message = "用户ID不能为空")
        @Schema(description = "用户ID")
        private Long userId;

        @NotBlank(message = "订阅类型不能为空")
        @Schema(description = "订阅类型")
        private String subscriptionType;

        @Schema(description = "订阅频率")
        private String frequency;

        @Schema(description = "是否启用")
        private Boolean enabled = true;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    @Schema(description = "邮件订阅响应")
    public static class EmailSubscriptionResponse {
        @Schema(description = "订阅ID")
        private Long subscriptionId;

        @Schema(description = "用户ID")
        private Long userId;

        @Schema(description = "订阅类型")
        private String subscriptionType;

        @Schema(description = "订阅频率")
        private String frequency;

        @Schema(description = "是否启用")
        private Boolean enabled;

        @Schema(description = "创建时间")
        private LocalDateTime createTime;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    @Schema(description = "邮件订阅更新请求")
    public static class EmailSubscriptionUpdateRequest {
        @Schema(description = "订阅频率")
        private String frequency;

        @Schema(description = "是否启用")
        private Boolean enabled;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    @Schema(description = "订阅邮件请求")
    public static class SubscriptionEmailRequest {
        @NotBlank(message = "订阅类型不能为空")
        @Schema(description = "订阅类型")
        private String subscriptionType;

        @NotBlank(message = "邮件内容不能为空")
        @Schema(description = "邮件内容")
        private String content;

        @Schema(description = "邮件主题")
        private String subject;

        @Schema(description = "目标用户ID列表")
        private List<Long> targetUserIds;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    @Schema(description = "邮件队列状态响应")
    public static class EmailQueueStatusResponse {
        @Schema(description = "队列中邮件数量")
        private Integer queueSize;

        @Schema(description = "处理中邮件数量")
        private Integer processingCount;

        @Schema(description = "失败邮件数量")
        private Integer failedCount;

        @Schema(description = "队列状态")
        private String queueStatus;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    @Schema(description = "批量重试响应")
    public static class BatchRetryResponse {
        @Schema(description = "重试邮件数量")
        private Integer retryCount;

        @Schema(description = "成功重试数量")
        private Integer successCount;

        @Schema(description = "失败重试数量")
        private Integer failedCount;

        @Schema(description = "批次ID")
        private String batchId;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    @Schema(description = "邮件状态响应")
    public static class EmailStatusResponse {
        @Schema(description = "邮件ID")
        private Long emailId;

        @Schema(description = "发送状态")
        private String status;

        @Schema(description = "发送时间")
        private LocalDateTime sentTime;

        @Schema(description = "错误信息")
        private String errorMessage;

        @Schema(description = "重试次数")
        private Integer retryCount;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    @Schema(description = "邮件报告请求")
    public static class EmailReportRequest {
        @Schema(description = "报告类型")
        private String reportType;

        @Schema(description = "开始时间")
        private LocalDateTime startTime;

        @Schema(description = "结束时间")
        private LocalDateTime endTime;

        @Schema(description = "报告格式")
        private String format;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    @Schema(description = "邮件报告响应")
    public static class EmailReportResponse {
        @Schema(description = "报告ID")
        private String reportId;

        @Schema(description = "报告文件路径")
        private String reportFilePath;

        @Schema(description = "生成时间")
        private LocalDateTime generateTime;

        @Schema(description = "报告大小")
        private Long fileSize;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    @Schema(description = "邮件导出请求")
    public static class EmailExportRequest {
        @Schema(description = "导出类型")
        private String exportType;

        @Schema(description = "开始时间")
        private LocalDateTime startTime;

        @Schema(description = "结束时间")
        private LocalDateTime endTime;

        @Schema(description = "导出格式")
        private String format;

        @Schema(description = "过滤条件")
        private Map<String, Object> filters;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    @Schema(description = "邮件导出响应")
    public static class EmailExportResponse {
        @Schema(description = "导出任务ID")
        private String taskId;

        @Schema(description = "导出文件路径")
        private String filePath;

        @Schema(description = "导出状态")
        private String status;

        @Schema(description = "导出记录数")
        private Integer recordCount;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    @Schema(description = "邮件清理请求")
    public static class EmailCleanupRequest {
        @Schema(description = "清理类型")
        private String cleanupType;

        @Schema(description = "保留天数")
        private Integer retentionDays;

        @Schema(description = "是否删除附件")
        private Boolean deleteAttachments = false;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    @Schema(description = "邮件清理响应")
    public static class EmailCleanupResponse {
        @Schema(description = "清理任务ID")
        private String taskId;

        @Schema(description = "清理记录数")
        private Integer cleanedCount;

        @Schema(description = "释放空间大小")
        private Long freedSpace;

        @Schema(description = "清理状态")
        private String status;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    @Schema(description = "邮件服务健康响应")
    public static class EmailServiceHealthResponse {
        @Schema(description = "服务状态")
        private String status;

        @Schema(description = "SMTP连接状态")
        private String smtpStatus;

        @Schema(description = "队列状态")
        private String queueStatus;

        @Schema(description = "最后检查时间")
        private LocalDateTime lastCheckTime;

        @Schema(description = "错误信息")
        private String errorMessage;
    }
}