package com.education.service.common;

import com.education.dto.UserDTO;
import com.education.dto.common.PageRequest;
import com.education.dto.common.PageResponse;
import org.springframework.web.multipart.MultipartFile;

import java.util.List;

/**
 * 公共用户服务接口
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
public interface UserService {

    /**
     * 获取当前用户信息
     * 
     * @param userId 用户ID
     * @return 用户信息
     */
    UserDTO.UserInfoResponse getCurrentUserInfo(Long userId);

    /**
     * 更新用户基本信息
     * 
     * @param updateRequest 更新请求
     * @param userId 用户ID
     * @return 操作结果
     */
    Boolean updateUserInfo(UserDTO.UserInfoUpdateRequest updateRequest, Long userId);

    /**
     * 上传用户头像
     * 
     * @param avatar 头像文件
     * @param userId 用户ID
     * @return 头像URL
     */
    String uploadAvatar(MultipartFile avatar, Long userId);

    /**
     * 修改密码
     * 
     * @param passwordRequest 密码修改请求
     * @param userId 用户ID
     * @return 操作结果
     */
    Boolean changePassword(UserDTO.PasswordChangeRequest passwordRequest, Long userId);

    /**
     * 绑定邮箱
     * 
     * @param bindRequest 邮箱绑定请求
     * @param userId 用户ID
     * @return 操作结果
     */
    Boolean bindEmail(UserDTO.EmailBindRequest bindRequest, Long userId);

    /**
     * 绑定手机号
     * 
     * @param bindRequest 手机号绑定请求
     * @param userId 用户ID
     * @return 操作结果
     */
    Boolean bindPhone(UserDTO.PhoneBindRequest bindRequest, Long userId);

    /**
     * 获取用户设置
     * 
     * @param userId 用户ID
     * @return 用户设置
     */
    UserDTO.UserSettingsResponse getUserSettings(Long userId);

    /**
     * 更新用户设置
     * 
     * @param settingsRequest 设置更新请求
     * @param userId 用户ID
     * @return 操作结果
     */
    Boolean updateUserSettings(UserDTO.UserSettingsUpdateRequest settingsRequest, Long userId);

    /**
     * 获取通知设置
     * 
     * @param userId 用户ID
     * @return 通知设置
     */
    UserDTO.NotificationSettingsResponse getNotificationSettings(Long userId);

    /**
     * 更新通知设置
     * 
     * @param notificationRequest 通知设置更新请求
     * @param userId 用户ID
     * @return 操作结果
     */
    Boolean updateNotificationSettings(UserDTO.NotificationSettingsUpdateRequest notificationRequest, Long userId);

    /**
     * 获取用户活动日志
     * 
     * @param userId 用户ID
     * @param pageRequest 分页请求
     * @return 活动日志列表
     */
    PageResponse<UserDTO.ActivityLogResponse> getActivityLogs(Long userId, PageRequest pageRequest);

    /**
     * 获取登录历史
     * 
     * @param userId 用户ID
     * @param pageRequest 分页请求
     * @return 登录历史列表
     */
    PageResponse<UserDTO.LoginHistoryResponse> getLoginHistory(Long userId, PageRequest pageRequest);

    /**
     * 注销账户
     * 
     * @param deactivateRequest 注销请求
     * @param userId 用户ID
     * @return 操作结果
     */
    Boolean deactivateAccount(UserDTO.AccountDeactivateRequest deactivateRequest, Long userId);

    /**
     * 导出用户数据
     * 
     * @param exportRequest 导出请求
     * @param userId 用户ID
     * @return 导出文件信息
     */
    UserDTO.DataExportResponse exportUserData(UserDTO.DataExportRequest exportRequest, Long userId);

    /**
     * 验证用户身份
     * 
     * @param verificationRequest 身份验证请求
     * @param userId 用户ID
     * @return 验证结果
     */
    Boolean verifyIdentity(UserDTO.IdentityVerificationRequest verificationRequest, Long userId);

    /**
     * 获取用户统计信息
     * 
     * @param userId 用户ID
     * @return 统计信息
     */
    UserDTO.UserStatisticsResponse getUserStatistics(Long userId);

    /**
     * 搜索用户
     * 
     * @param searchRequest 搜索请求
     * @param currentUserId 当前用户ID
     * @return 搜索结果
     */
    PageResponse<UserDTO.UserSearchResponse> searchUsers(UserDTO.UserSearchRequest searchRequest, Long currentUserId);

    /**
     * 获取用户详细信息（管理员权限）
     * 
     * @param targetUserId 目标用户ID
     * @param currentUserId 当前用户ID
     * @return 用户详细信息
     */
    UserDTO.UserDetailResponse getUserDetail(Long targetUserId, Long currentUserId);

    /**
     * 获取用户角色信息
     * 
     * @param userId 用户ID
     * @return 角色信息
     */
    UserDTO.UserRoleResponse getUserRoles(Long userId);

    /**
     * 获取用户权限信息
     * 
     * @param userId 用户ID
     * @return 权限信息
     */
    UserDTO.UserPermissionResponse getUserPermissions(Long userId);

    /**
     * 检查用户权限
     * 
     * @param userId 用户ID
     * @param permission 权限标识
     * @return 是否有权限
     */
    Boolean hasPermission(Long userId, String permission);

    /**
     * 获取用户偏好设置
     * 
     * @param userId 用户ID
     * @return 偏好设置
     */
    UserDTO.UserPreferencesResponse getUserPreferences(Long userId);

    /**
     * 更新用户偏好设置
     * 
     * @param preferencesRequest 偏好设置更新请求
     * @param userId 用户ID
     * @return 操作结果
     */
    Boolean updateUserPreferences(UserDTO.UserPreferencesUpdateRequest preferencesRequest, Long userId);

    /**
     * 获取用户安全设置
     * 
     * @param userId 用户ID
     * @return 安全设置
     */
    UserDTO.SecuritySettingsResponse getSecuritySettings(Long userId);

    /**
     * 更新用户安全设置
     * 
     * @param securityRequest 安全设置更新请求
     * @param userId 用户ID
     * @return 操作结果
     */
    Boolean updateSecuritySettings(UserDTO.SecuritySettingsUpdateRequest securityRequest, Long userId);

    /**
     * 启用两步验证
     * 
     * @param twoFactorRequest 两步验证启用请求
     * @param userId 用户ID
     * @return 操作结果
     */
    UserDTO.TwoFactorAuthResponse enableTwoFactorAuth(UserDTO.TwoFactorAuthRequest twoFactorRequest, Long userId);

    /**
     * 禁用两步验证
     * 
     * @param disableRequest 两步验证禁用请求
     * @param userId 用户ID
     * @return 操作结果
     */
    Boolean disableTwoFactorAuth(UserDTO.TwoFactorAuthDisableRequest disableRequest, Long userId);

    /**
     * 获取用户通知列表
     * 
     * @param userId 用户ID
     * @param pageRequest 分页请求
     * @return 通知列表
     */
    PageResponse<UserDTO.NotificationResponse> getUserNotifications(Long userId, PageRequest pageRequest);

    /**
     * 标记通知为已读
     * 
     * @param notificationId 通知ID
     * @param userId 用户ID
     * @return 操作结果
     */
    Boolean markNotificationAsRead(Long notificationId, Long userId);

    /**
     * 批量标记通知为已读
     * 
     * @param notificationIds 通知ID列表
     * @param userId 用户ID
     * @return 操作结果
     */
    Boolean markNotificationsAsRead(List<Long> notificationIds, Long userId);

    /**
     * 删除通知
     * 
     * @param notificationId 通知ID
     * @param userId 用户ID
     * @return 操作结果
     */
    Boolean deleteNotification(Long notificationId, Long userId);

    /**
     * 获取未读通知数量
     * 
     * @param userId 用户ID
     * @return 未读通知数量
     */
    Integer getUnreadNotificationCount(Long userId);

    /**
     * 获取用户关注列表
     * 
     * @param userId 用户ID
     * @param pageRequest 分页请求
     * @return 关注列表
     */
    PageResponse<UserDTO.UserFollowResponse> getUserFollowing(Long userId, PageRequest pageRequest);

    /**
     * 获取用户粉丝列表
     * 
     * @param userId 用户ID
     * @param pageRequest 分页请求
     * @return 粉丝列表
     */
    PageResponse<UserDTO.UserFollowResponse> getUserFollowers(Long userId, PageRequest pageRequest);

    /**
     * 关注用户
     * 
     * @param targetUserId 目标用户ID
     * @param userId 当前用户ID
     * @return 操作结果
     */
    Boolean followUser(Long targetUserId, Long userId);

    /**
     * 取消关注用户
     * 
     * @param targetUserId 目标用户ID
     * @param userId 当前用户ID
     * @return 操作结果
     */
    Boolean unfollowUser(Long targetUserId, Long userId);

    /**
     * 检查是否关注某用户
     * 
     * @param targetUserId 目标用户ID
     * @param userId 当前用户ID
     * @return 是否关注
     */
    Boolean isFollowing(Long targetUserId, Long userId);

    /**
     * 获取用户标签
     * 
     * @param userId 用户ID
     * @return 用户标签列表
     */
    List<String> getUserTags(Long userId);

    /**
     * 设置用户标签
     * 
     * @param userId 用户ID
     * @param tags 标签列表
     * @param operatorId 操作者ID
     * @return 操作结果
     */
    Boolean setUserTags(Long userId, List<String> tags, Long operatorId);

    /**
     * 获取用户学习报告
     * 
     * @param userId 用户ID
     * @param reportType 报告类型
     * @param timeRange 时间范围
     * @return 学习报告
     */
    UserDTO.LearningReportResponse getLearningReport(Long userId, String reportType, String timeRange);

    /**
     * 获取用户成就列表
     * 
     * @param userId 用户ID
     * @param pageRequest 分页请求
     * @return 成就列表
     */
    PageResponse<UserDTO.AchievementResponse> getUserAchievements(Long userId, PageRequest pageRequest);

    /**
     * 获取用户积分信息
     * 
     * @param userId 用户ID
     * @return 积分信息
     */
    UserDTO.UserPointsResponse getUserPoints(Long userId);

    /**
     * 获取积分历史记录
     * 
     * @param userId 用户ID
     * @param pageRequest 分页请求
     * @return 积分历史记录
     */
    PageResponse<UserDTO.PointsHistoryResponse> getPointsHistory(Long userId, PageRequest pageRequest);

    /**
     * 获取用户等级信息
     * 
     * @param userId 用户ID
     * @return 等级信息
     */
    UserDTO.UserLevelResponse getUserLevel(Long userId);

    /**
     * 获取用户徽章列表
     * 
     * @param userId 用户ID
     * @return 徽章列表
     */
    List<UserDTO.BadgeResponse> getUserBadges(Long userId);

    /**
     * 获取用户学习时长统计
     * 
     * @param userId 用户ID
     * @param timeRange 时间范围
     * @return 学习时长统计
     */
    UserDTO.StudyTimeStatisticsResponse getStudyTimeStatistics(Long userId, String timeRange);

    /**
     * 获取用户在线状态
     * 
     * @param userId 用户ID
     * @return 在线状态
     */
    UserDTO.OnlineStatusResponse getOnlineStatus(Long userId);

    /**
     * 更新用户在线状态
     * 
     * @param statusRequest 状态更新请求
     * @param userId 用户ID
     * @return 操作结果
     */
    Boolean updateOnlineStatus(UserDTO.OnlineStatusUpdateRequest statusRequest, Long userId);

    /**
     * 获取用户个人主页信息
     * 
     * @param targetUserId 目标用户ID
     * @param currentUserId 当前用户ID
     * @return 个人主页信息
     */
    UserDTO.UserProfileResponse getUserProfile(Long targetUserId, Long currentUserId);

    /**
     * 更新用户个人主页
     * 
     * @param profileRequest 主页更新请求
     * @param userId 用户ID
     * @return 操作结果
     */
    Boolean updateUserProfile(UserDTO.UserProfileUpdateRequest profileRequest, Long userId);

    /**
     * 重置用户设置为默认值
     * 
     * @param userId 用户ID
     * @return 操作结果
     */
    Boolean resetUserSettings(Long userId);

    /**
     * 获取用户反馈列表
     * 
     * @param userId 用户ID
     * @param pageRequest 分页请求
     * @return 反馈列表
     */
    PageResponse<UserDTO.UserFeedbackResponse> getUserFeedback(Long userId, PageRequest pageRequest);

    /**
     * 提交用户反馈
     * 
     * @param feedbackRequest 反馈请求
     * @param userId 用户ID
     * @return 反馈ID
     */
    Long submitFeedback(UserDTO.FeedbackSubmitRequest feedbackRequest, Long userId);
}