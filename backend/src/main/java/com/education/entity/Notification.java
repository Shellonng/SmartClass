package com.education.entity;

import com.baomidou.mybatisplus.annotation.*;
import com.fasterxml.jackson.annotation.JsonFormat;
import com.fasterxml.jackson.annotation.JsonIgnore;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.experimental.Accessors;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;
import java.io.Serializable;
import java.time.LocalDateTime;

/**
 * 通知实体类
 * 
 * @author Education Platform Team
 * @version 1.0.0
 */
@Data
@EqualsAndHashCode(callSuper = false)
@Accessors(chain = true)
@TableName("notification")
@Schema(description = "通知信息")
public class Notification implements Serializable {

    private static final long serialVersionUID = 1L;

    @Schema(description = "通知ID")
    @TableId(value = "id", type = IdType.ASSIGN_ID)
    private Long id;

    @Schema(description = "通知标题", example = "新作业发布")
    @TableField("title")
    @NotBlank(message = "通知标题不能为空")
    @Size(max = 200, message = "通知标题长度不能超过200个字符")
    private String title;

    @Schema(description = "通知内容")
    @TableField("content")
    @NotBlank(message = "通知内容不能为空")
    @Size(max = 5000, message = "通知内容长度不能超过5000个字符")
    private String content;

    @Schema(description = "通知类型", example = "TASK")
    @TableField("notification_type")
    @NotBlank(message = "通知类型不能为空")
    private String notificationType;

    @Schema(description = "通知级别", example = "NORMAL")
    @TableField("level")
    private String level;

    @Schema(description = "发送者ID")
    @TableField("sender_id")
    @NotNull(message = "发送者ID不能为空")
    private Long senderId;

    @Schema(description = "发送者类型", example = "TEACHER")
    @TableField("sender_type")
    private String senderType;

    @Schema(description = "接收者ID")
    @TableField("receiver_id")
    private Long receiverId;

    @Schema(description = "接收者类型", example = "STUDENT")
    @TableField("receiver_type")
    private String receiverType;

    @Schema(description = "目标类型", example = "COURSE")
    @TableField("target_type")
    private String targetType;

    @Schema(description = "目标ID")
    @TableField("target_id")
    private Long targetId;

    @Schema(description = "课程ID")
    @TableField("course_id")
    private Long courseId;

    @Schema(description = "班级ID")
    @TableField("class_id")
    private Long classId;

    @Schema(description = "是否已读", example = "false")
    @TableField("is_read")
    private Boolean isRead;

    @Schema(description = "阅读时间")
    @TableField("read_time")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime readTime;

    @Schema(description = "发送时间")
    @TableField("send_time")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime sendTime;

    @Schema(description = "计划发送时间")
    @TableField("scheduled_time")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime scheduledTime;

    @Schema(description = "过期时间")
    @TableField("expire_time")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime expireTime;

    @Schema(description = "通知状态", example = "SENT")
    @TableField("status")
    private String status;

    @Schema(description = "优先级", example = "NORMAL")
    @TableField("priority")
    private String priority;

    @Schema(description = "是否需要确认", example = "false")
    @TableField("require_confirmation")
    private Boolean requireConfirmation;

    @Schema(description = "确认时间")
    @TableField("confirmation_time")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime confirmationTime;

    @Schema(description = "是否已确认", example = "false")
    @TableField("is_confirmed")
    private Boolean isConfirmed;

    @Schema(description = "推送渠道")
    @TableField("push_channels")
    @Size(max = 200, message = "推送渠道长度不能超过200个字符")
    private String pushChannels;

    @Schema(description = "附件URL")
    @TableField("attachment_url")
    @Size(max = 500, message = "附件URL长度不能超过500个字符")
    private String attachmentUrl;

    @Schema(description = "跳转链接")
    @TableField("action_url")
    @Size(max = 500, message = "跳转链接长度不能超过500个字符")
    private String actionUrl;

    @Schema(description = "操作按钮")
    @TableField("action_buttons")
    @Size(max = 1000, message = "操作按钮长度不能超过1000个字符")
    private String actionButtons;

    @Schema(description = "标签")
    @TableField("tags")
    @Size(max = 500, message = "标签长度不能超过500个字符")
    private String tags;

    @Schema(description = "分组")
    @TableField("group_name")
    @Size(max = 100, message = "分组长度不能超过100个字符")
    private String groupName;

    @Schema(description = "重试次数", example = "0")
    @TableField("retry_count")
    private Integer retryCount;

    @Schema(description = "最大重试次数", example = "3")
    @TableField("max_retry")
    private Integer maxRetry;

    @Schema(description = "下次重试时间")
    @TableField("next_retry_time")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime nextRetryTime;

    @Schema(description = "错误信息")
    @TableField("error_message")
    @Size(max = 1000, message = "错误信息长度不能超过1000个字符")
    private String errorMessage;

    @Schema(description = "扩展数据")
    @TableField("extra_data")
    @Size(max = 2000, message = "扩展数据长度不能超过2000个字符")
    private String extraData;

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
    @Schema(description = "发送者信息")
    private User sender;

    @TableField(exist = false)
    @Schema(description = "接收者信息")
    private User receiver;

    @TableField(exist = false)
    @Schema(description = "课程信息")
    private Course course;

    @TableField(exist = false)
    @Schema(description = "班级信息")
    private Class classInfo;

    /**
     * 通知类型枚举
     */
    public enum NotificationType {
        SYSTEM("SYSTEM", "系统通知"),
        TASK("TASK", "任务通知"),
        COURSE("COURSE", "课程通知"),
        GRADE("GRADE", "成绩通知"),
        ANNOUNCEMENT("ANNOUNCEMENT", "公告通知"),
        REMINDER("REMINDER", "提醒通知"),
        MESSAGE("MESSAGE", "消息通知"),
        WARNING("WARNING", "警告通知"),
        MAINTENANCE("MAINTENANCE", "维护通知"),
        UPDATE("UPDATE", "更新通知");

        private final String code;
        private final String description;

        NotificationType(String code, String description) {
            this.code = code;
            this.description = description;
        }

        public String getCode() {
            return code;
        }

        public String getDescription() {
            return description;
        }

        public static NotificationType fromCode(String code) {
            for (NotificationType type : values()) {
                if (type.code.equals(code)) {
                    return type;
                }
            }
            throw new IllegalArgumentException("未知的通知类型: " + code);
        }
    }

    /**
     * 通知级别枚举
     */
    public enum Level {
        LOW("LOW", "低"),
        NORMAL("NORMAL", "普通"),
        HIGH("HIGH", "高"),
        URGENT("URGENT", "紧急"),
        CRITICAL("CRITICAL", "严重");

        private final String code;
        private final String description;

        Level(String code, String description) {
            this.code = code;
            this.description = description;
        }

        public String getCode() {
            return code;
        }

        public String getDescription() {
            return description;
        }

        public static Level fromCode(String code) {
            for (Level level : values()) {
                if (level.code.equals(code)) {
                    return level;
                }
            }
            throw new IllegalArgumentException("未知的通知级别: " + code);
        }
    }

    /**
     * 通知状态枚举
     */
    public enum Status {
        DRAFT("DRAFT", "草稿"),
        SCHEDULED("SCHEDULED", "已安排"),
        SENDING("SENDING", "发送中"),
        SENT("SENT", "已发送"),
        DELIVERED("DELIVERED", "已送达"),
        READ("READ", "已阅读"),
        FAILED("FAILED", "发送失败"),
        CANCELLED("CANCELLED", "已取消"),
        EXPIRED("EXPIRED", "已过期");

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
            throw new IllegalArgumentException("未知的通知状态: " + code);
        }
    }

    /**
     * 优先级枚举
     */
    public enum Priority {
        LOWEST("LOWEST", "最低", 1),
        LOW("LOW", "低", 2),
        NORMAL("NORMAL", "普通", 3),
        HIGH("HIGH", "高", 4),
        HIGHEST("HIGHEST", "最高", 5);

        private final String code;
        private final String description;
        private final int value;

        Priority(String code, String description, int value) {
            this.code = code;
            this.description = description;
            this.value = value;
        }

        public String getCode() {
            return code;
        }

        public String getDescription() {
            return description;
        }

        public int getValue() {
            return value;
        }

        public static Priority fromCode(String code) {
            for (Priority priority : values()) {
                if (priority.code.equals(code)) {
                    return priority;
                }
            }
            throw new IllegalArgumentException("未知的优先级: " + code);
        }
    }

    /**
     * 用户类型枚举
     */
    public enum UserType {
        TEACHER("TEACHER", "教师"),
        STUDENT("STUDENT", "学生"),
        ADMIN("ADMIN", "管理员"),
        SYSTEM("SYSTEM", "系统");

        private final String code;
        private final String description;

        UserType(String code, String description) {
            this.code = code;
            this.description = description;
        }

        public String getCode() {
            return code;
        }

        public String getDescription() {
            return description;
        }

        public static UserType fromCode(String code) {
            for (UserType type : values()) {
                if (type.code.equals(code)) {
                    return type;
                }
            }
            throw new IllegalArgumentException("未知的用户类型: " + code);
        }
    }

    /**
     * 判断是否已读
     * 
     * @return 是否已读
     */
    public boolean isReadNotification() {
        return isRead != null && isRead;
    }

    /**
     * 判断是否已确认
     * 
     * @return 是否已确认
     */
    public boolean isConfirmedNotification() {
        return isConfirmed != null && isConfirmed;
    }

    /**
     * 判断是否需要确认
     * 
     * @return 是否需要确认
     */
    public boolean requiresConfirmation() {
        return requireConfirmation != null && requireConfirmation;
    }

    /**
     * 判断是否已过期
     * 
     * @return 是否已过期
     */
    public boolean isExpired() {
        return expireTime != null && LocalDateTime.now().isAfter(expireTime);
    }

    /**
     * 判断是否即将过期（距离过期时间不足一小时）
     * 
     * @return 是否即将过期
     */
    public boolean isExpiringSoon() {
        if (expireTime == null) {
            return false;
        }
        return LocalDateTime.now().plusHours(1).isAfter(expireTime);
    }

    /**
     * 判断是否为高优先级
     * 
     * @return 是否为高优先级
     */
    public boolean isHighPriority() {
        try {
            Priority p = Priority.fromCode(this.priority);
            return p.getValue() >= Priority.HIGH.getValue();
        } catch (IllegalArgumentException e) {
            return false;
        }
    }

    /**
     * 判断是否为紧急通知
     * 
     * @return 是否为紧急通知
     */
    public boolean isUrgent() {
        return Level.URGENT.getCode().equals(this.level) || Level.CRITICAL.getCode().equals(this.level);
    }

    /**
     * 判断是否已发送
     * 
     * @return 是否已发送
     */
    public boolean isSent() {
        return Status.SENT.getCode().equals(this.status) || 
               Status.DELIVERED.getCode().equals(this.status) || 
               Status.READ.getCode().equals(this.status);
    }

    /**
     * 判断是否发送失败
     * 
     * @return 是否发送失败
     */
    public boolean isFailed() {
        return Status.FAILED.getCode().equals(this.status);
    }

    /**
     * 判断是否可以重试
     * 
     * @return 是否可以重试
     */
    public boolean canRetry() {
        return isFailed() && 
               (retryCount == null || retryCount < (maxRetry != null ? maxRetry : 3)) &&
               !isExpired();
    }

    /**
     * 判断是否为系统通知
     * 
     * @return 是否为系统通知
     */
    public boolean isSystemNotification() {
        return NotificationType.SYSTEM.getCode().equals(this.notificationType);
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
     * 获取通知类型描述
     * 
     * @return 通知类型描述
     */
    public String getNotificationTypeDescription() {
        try {
            return NotificationType.fromCode(this.notificationType).getDescription();
        } catch (IllegalArgumentException e) {
            return this.notificationType;
        }
    }

    /**
     * 获取级别描述
     * 
     * @return 级别描述
     */
    public String getLevelDescription() {
        try {
            return Level.fromCode(this.level).getDescription();
        } catch (IllegalArgumentException e) {
            return this.level;
        }
    }

    /**
     * 获取优先级描述
     * 
     * @return 优先级描述
     */
    public String getPriorityDescription() {
        try {
            return Priority.fromCode(this.priority).getDescription();
        } catch (IllegalArgumentException e) {
            return this.priority;
        }
    }

    /**
     * 获取发送者类型描述
     * 
     * @return 发送者类型描述
     */
    public String getSenderTypeDescription() {
        try {
            return UserType.fromCode(this.senderType).getDescription();
        } catch (IllegalArgumentException e) {
            return this.senderType;
        }
    }

    /**
     * 获取接收者类型描述
     * 
     * @return 接收者类型描述
     */
    public String getReceiverTypeDescription() {
        try {
            return UserType.fromCode(this.receiverType).getDescription();
        } catch (IllegalArgumentException e) {
            return this.receiverType;
        }
    }

    /**
     * 获取推送渠道列表
     * 
     * @return 推送渠道列表
     */
    public String[] getPushChannelsList() {
        if (pushChannels == null || pushChannels.trim().isEmpty()) {
            return new String[0];
        }
        return pushChannels.split(",");
    }

    /**
     * 设置推送渠道列表
     * 
     * @param channelsList 推送渠道列表
     */
    public void setPushChannelsList(String[] channelsList) {
        if (channelsList == null || channelsList.length == 0) {
            this.pushChannels = null;
        } else {
            this.pushChannels = String.join(",", channelsList);
        }
    }

    /**
     * 获取标签列表
     * 
     * @return 标签列表
     */
    public String[] getTagsList() {
        if (tags == null || tags.trim().isEmpty()) {
            return new String[0];
        }
        return tags.split(",");
    }

    /**
     * 设置标签列表
     * 
     * @param tagsList 标签列表
     */
    public void setTagsList(String[] tagsList) {
        if (tagsList == null || tagsList.length == 0) {
            this.tags = null;
        } else {
            this.tags = String.join(",", tagsList);
        }
    }

    /**
     * 标记为已读
     */
    public void markAsRead() {
        this.isRead = true;
        this.readTime = LocalDateTime.now();
    }

    /**
     * 标记为已确认
     */
    public void markAsConfirmed() {
        this.isConfirmed = true;
        this.confirmationTime = LocalDateTime.now();
    }

    /**
     * 增加重试次数
     */
    public void incrementRetryCount() {
        this.retryCount = (retryCount != null ? retryCount : 0) + 1;
    }

    /**
     * 设置下次重试时间
     * 
     * @param minutes 延迟分钟数
     */
    public void setNextRetryTime(int minutes) {
        this.nextRetryTime = LocalDateTime.now().plusMinutes(minutes);
    }

    /**
     * 获取通知摘要
     * 
     * @return 通知摘要
     */
    public String getNotificationSummary() {
        StringBuilder summary = new StringBuilder();
        
        summary.append("[").append(getNotificationTypeDescription()).append("] ");
        summary.append(title);
        
        if (isUrgent()) {
            summary.append(" [紧急]");
        }
        
        if (requiresConfirmation()) {
            summary.append(" [需确认]");
        }
        
        return summary.toString();
    }

    /**
     * 判断是否可以编辑
     * 
     * @return 是否可以编辑
     */
    public boolean canEdit() {
        return Status.DRAFT.getCode().equals(this.status) || 
               Status.SCHEDULED.getCode().equals(this.status);
    }

    /**
     * 判断是否可以取消
     * 
     * @return 是否可以取消
     */
    public boolean canCancel() {
        return Status.DRAFT.getCode().equals(this.status) || 
               Status.SCHEDULED.getCode().equals(this.status) ||
               Status.SENDING.getCode().equals(this.status);
    }

    /**
     * 获取剩余有效时间（小时）
     * 
     * @return 剩余有效时间
     */
    public long getRemainingHours() {
        if (expireTime == null) {
            return Long.MAX_VALUE;
        }
        
        LocalDateTime now = LocalDateTime.now();
        if (now.isAfter(expireTime)) {
            return 0;
        }
        
        return java.time.Duration.between(now, expireTime).toHours();
    }
}