package com.education.service.common;

import com.education.dto.EmailDTO;
import com.education.dto.common.PageRequest;
import com.education.dto.common.PageResponse;

import java.util.List;
import java.util.Map;

/**
 * 公共邮件服务接口
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
public interface EmailService {

    /**
     * 发送简单文本邮件
     * 
     * @param to 收件人邮箱
     * @param subject 邮件主题
     * @param content 邮件内容
     * @return 发送结果
     */
    Boolean sendSimpleEmail(String to, String subject, String content);

    /**
     * 发送HTML邮件
     * 
     * @param to 收件人邮箱
     * @param subject 邮件主题
     * @param htmlContent HTML内容
     * @return 发送结果
     */
    Boolean sendHtmlEmail(String to, String subject, String htmlContent);

    /**
     * 发送带附件的邮件
     * 
     * @param emailRequest 邮件发送请求
     * @return 发送结果
     */
    Boolean sendEmailWithAttachment(EmailDTO.EmailWithAttachmentRequest emailRequest);

    /**
     * 批量发送邮件
     * 
     * @param batchRequest 批量发送请求
     * @return 发送结果
     */
    EmailDTO.BatchEmailResponse sendBatchEmails(EmailDTO.BatchEmailRequest batchRequest);

    /**
     * 发送模板邮件
     * 
     * @param templateRequest 模板邮件请求
     * @return 发送结果
     */
    Boolean sendTemplateEmail(EmailDTO.TemplateEmailRequest templateRequest);

    /**
     * 发送验证码邮件
     * 
     * @param email 邮箱地址
     * @param verificationCode 验证码
     * @param purpose 验证目的
     * @return 发送结果
     */
    Boolean sendVerificationEmail(String email, String verificationCode, String purpose);

    /**
     * 发送密码重置邮件
     * 
     * @param email 邮箱地址
     * @param resetToken 重置令牌
     * @param userName 用户名
     * @return 发送结果
     */
    Boolean sendPasswordResetEmail(String email, String resetToken, String userName);

    /**
     * 发送欢迎邮件
     * 
     * @param email 邮箱地址
     * @param userName 用户名
     * @param userType 用户类型
     * @return 发送结果
     */
    Boolean sendWelcomeEmail(String email, String userName, String userType);

    /**
     * 发送通知邮件
     * 
     * @param notificationRequest 通知邮件请求
     * @return 发送结果
     */
    Boolean sendNotificationEmail(EmailDTO.NotificationEmailRequest notificationRequest);

    /**
     * 发送课程提醒邮件
     * 
     * @param reminderRequest 课程提醒请求
     * @return 发送结果
     */
    Boolean sendCourseReminderEmail(EmailDTO.CourseReminderRequest reminderRequest);

    /**
     * 发送任务截止提醒邮件
     * 
     * @param taskReminderRequest 任务提醒请求
     * @return 发送结果
     */
    Boolean sendTaskDeadlineEmail(EmailDTO.TaskDeadlineReminderRequest taskReminderRequest);

    /**
     * 发送成绩通知邮件
     * 
     * @param gradeNotificationRequest 成绩通知请求
     * @return 发送结果
     */
    Boolean sendGradeNotificationEmail(EmailDTO.GradeNotificationRequest gradeNotificationRequest);

    /**
     * 发送系统维护通知邮件
     * 
     * @param maintenanceRequest 维护通知请求
     * @return 发送结果
     */
    Boolean sendMaintenanceNotificationEmail(EmailDTO.MaintenanceNotificationRequest maintenanceRequest);

    /**
     * 发送邀请邮件
     * 
     * @param invitationRequest 邀请邮件请求
     * @return 发送结果
     */
    Boolean sendInvitationEmail(EmailDTO.InvitationEmailRequest invitationRequest);

    /**
     * 验证邮箱地址格式
     * 
     * @param email 邮箱地址
     * @return 是否有效
     */
    Boolean validateEmailFormat(String email);

    /**
     * 检查邮箱是否存在
     * 
     * @param email 邮箱地址
     * @return 是否存在
     */
    Boolean checkEmailExists(String email);

    /**
     * 获取邮件发送历史
     * 
     * @param historyRequest 历史查询请求
     * @return 发送历史列表
     */
    PageResponse<EmailDTO.EmailHistoryResponse> getEmailHistory(EmailDTO.EmailHistoryRequest historyRequest);

    /**
     * 获取邮件发送统计
     * 
     * @param statisticsRequest 统计请求
     * @return 发送统计
     */
    EmailDTO.EmailStatisticsResponse getEmailStatistics(EmailDTO.EmailStatisticsRequest statisticsRequest);

    /**
     * 创建邮件模板
     * 
     * @param templateRequest 模板创建请求
     * @return 模板ID
     */
    Long createEmailTemplate(EmailDTO.EmailTemplateCreateRequest templateRequest);

    /**
     * 更新邮件模板
     * 
     * @param templateId 模板ID
     * @param templateRequest 模板更新请求
     * @return 操作结果
     */
    Boolean updateEmailTemplate(Long templateId, EmailDTO.EmailTemplateUpdateRequest templateRequest);

    /**
     * 删除邮件模板
     * 
     * @param templateId 模板ID
     * @return 操作结果
     */
    Boolean deleteEmailTemplate(Long templateId);

    /**
     * 获取邮件模板列表
     * 
     * @param pageRequest 分页请求
     * @return 模板列表
     */
    PageResponse<EmailDTO.EmailTemplateResponse> getEmailTemplates(PageRequest pageRequest);

    /**
     * 获取邮件模板详情
     * 
     * @param templateId 模板ID
     * @return 模板详情
     */
    EmailDTO.EmailTemplateDetailResponse getEmailTemplateDetail(Long templateId);

    /**
     * 预览邮件模板
     * 
     * @param templateId 模板ID
     * @param variables 模板变量
     * @return 预览内容
     */
    EmailDTO.EmailPreviewResponse previewEmailTemplate(Long templateId, Map<String, Object> variables);

    /**
     * 测试邮件模板
     * 
     * @param testRequest 测试请求
     * @return 测试结果
     */
    Boolean testEmailTemplate(EmailDTO.EmailTemplateTestRequest testRequest);

    /**
     * 获取邮件配置
     * 
     * @return 邮件配置
     */
    EmailDTO.EmailConfigResponse getEmailConfig();

    /**
     * 更新邮件配置
     * 
     * @param configRequest 配置更新请求
     * @return 操作结果
     */
    Boolean updateEmailConfig(EmailDTO.EmailConfigUpdateRequest configRequest);

    /**
     * 测试邮件配置
     * 
     * @param testEmail 测试邮箱
     * @return 测试结果
     */
    Boolean testEmailConfig(String testEmail);

    /**
     * 添加邮件黑名单
     * 
     * @param email 邮箱地址
     * @param reason 加入原因
     * @return 操作结果
     */
    Boolean addEmailBlacklist(String email, String reason);

    /**
     * 移除邮件黑名单
     * 
     * @param email 邮箱地址
     * @return 操作结果
     */
    Boolean removeEmailBlacklist(String email);

    /**
     * 检查邮箱是否在黑名单中
     * 
     * @param email 邮箱地址
     * @return 是否在黑名单中
     */
    Boolean isEmailBlacklisted(String email);

    /**
     * 获取邮件黑名单
     * 
     * @param pageRequest 分页请求
     * @return 黑名单列表
     */
    PageResponse<EmailDTO.EmailBlacklistResponse> getEmailBlacklist(PageRequest pageRequest);

    /**
     * 创建邮件订阅
     * 
     * @param subscriptionRequest 订阅请求
     * @return 订阅ID
     */
    Long createEmailSubscription(EmailDTO.EmailSubscriptionRequest subscriptionRequest);

    /**
     * 取消邮件订阅
     * 
     * @param subscriptionId 订阅ID
     * @return 操作结果
     */
    Boolean cancelEmailSubscription(Long subscriptionId);

    /**
     * 通过令牌取消订阅
     * 
     * @param unsubscribeToken 取消订阅令牌
     * @return 操作结果
     */
    Boolean unsubscribeByToken(String unsubscribeToken);

    /**
     * 获取用户邮件订阅列表
     * 
     * @param userId 用户ID
     * @return 订阅列表
     */
    List<EmailDTO.EmailSubscriptionResponse> getUserEmailSubscriptions(Long userId);

    /**
     * 更新邮件订阅设置
     * 
     * @param subscriptionId 订阅ID
     * @param updateRequest 更新请求
     * @return 操作结果
     */
    Boolean updateEmailSubscription(Long subscriptionId, EmailDTO.EmailSubscriptionUpdateRequest updateRequest);

    /**
     * 发送订阅邮件
     * 
     * @param subscriptionEmailRequest 订阅邮件请求
     * @return 发送结果
     */
    EmailDTO.BatchEmailResponse sendSubscriptionEmails(EmailDTO.SubscriptionEmailRequest subscriptionEmailRequest);

    /**
     * 获取邮件发送队列状态
     * 
     * @return 队列状态
     */
    EmailDTO.EmailQueueStatusResponse getEmailQueueStatus();

    /**
     * 重试失败的邮件
     * 
     * @param emailId 邮件ID
     * @return 操作结果
     */
    Boolean retryFailedEmail(Long emailId);

    /**
     * 批量重试失败的邮件
     * 
     * @param emailIds 邮件ID列表
     * @return 操作结果
     */
    EmailDTO.BatchRetryResponse retryFailedEmails(List<Long> emailIds);

    /**
     * 获取邮件发送状态
     * 
     * @param emailId 邮件ID
     * @return 发送状态
     */
    EmailDTO.EmailStatusResponse getEmailStatus(Long emailId);

    /**
     * 取消待发送的邮件
     * 
     * @param emailId 邮件ID
     * @return 操作结果
     */
    Boolean cancelPendingEmail(Long emailId);

    /**
     * 获取邮件发送报告
     * 
     * @param reportRequest 报告请求
     * @return 发送报告
     */
    EmailDTO.EmailReportResponse getEmailReport(EmailDTO.EmailReportRequest reportRequest);

    /**
     * 导出邮件发送数据
     * 
     * @param exportRequest 导出请求
     * @return 导出文件信息
     */
    EmailDTO.EmailExportResponse exportEmailData(EmailDTO.EmailExportRequest exportRequest);

    /**
     * 清理邮件发送历史
     * 
     * @param cleanupRequest 清理请求
     * @return 清理结果
     */
    EmailDTO.EmailCleanupResponse cleanupEmailHistory(EmailDTO.EmailCleanupRequest cleanupRequest);

    /**
     * 获取邮件服务健康状态
     * 
     * @return 健康状态
     */
    EmailDTO.EmailServiceHealthResponse getEmailServiceHealth();
}