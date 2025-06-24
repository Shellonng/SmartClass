package com.education.service.auth;

import com.education.dto.AuthDTO;

/**
 * 认证服务接口
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
public interface AuthService {

    /**
     * 用户登录
     * 
     * @param loginRequest 登录请求
     * @return 登录响应
     */
    AuthDTO.LoginResponse login(AuthDTO.LoginRequest loginRequest);

    /**
     * 用户登出
     * 
     * @param token JWT令牌
     * @return 操作结果
     */
    Boolean logout(String token);

    /**
     * 刷新Token
     * 
     * @param refreshTokenRequest 刷新Token请求
     * @return 新的Token信息
     */
    AuthDTO.LoginResponse refreshToken(AuthDTO.RefreshTokenRequest refreshTokenRequest);

    /**
     * 修改密码
     * 
     * @param changePasswordRequest 修改密码请求
     * @param userId 用户ID
     * @return 操作结果
     */
    Boolean changePassword(AuthDTO.ChangePasswordRequest changePasswordRequest, Long userId);

    /**
     * 发送重置密码邮件
     * 
     * @param email 邮箱地址
     * @return 操作结果
     */
    Boolean sendResetPasswordEmail(String email);

    /**
     * 重置密码
     * 
     * @param resetPasswordRequest 重置密码请求
     * @return 操作结果
     */
    Boolean resetPassword(AuthDTO.ResetPasswordRequest resetPasswordRequest);

    /**
     * 获取验证码
     * 
     * @param email 邮箱地址
     * @param type 验证码类型
     * @return 操作结果
     */
    Boolean sendVerificationCode(String email, String type);

    /**
     * 验证验证码
     * 
     * @param email 邮箱地址
     * @param code 验证码
     * @param type 验证码类型
     * @return 验证结果
     */
    Boolean verifyCode(String email, String code, String type);

    /**
     * 获取当前用户信息
     * 
     * @param userId 用户ID
     * @return 用户信息
     */
    Object getCurrentUserInfo(Long userId);

    /**
     * 验证Token有效性
     * 
     * @param token JWT令牌
     * @return 验证结果
     */
    Boolean validateToken(String token);

    /**
     * 检查用户是否存在
     * 
     * @param username 用户名
     * @return 检查结果
     */
    Boolean checkUserExists(String username);

    /**
     * 检查邮箱是否存在
     * 
     * @param email 邮箱地址
     * @return 检查结果
     */
    Boolean checkEmailExists(String email);

    /**
     * 记录登录日志
     * 
     * @param userId 用户ID
     * @param ip IP地址
     * @param userAgent 用户代理
     * @param success 登录是否成功
     */
    void recordLoginLog(Long userId, String ip, String userAgent, Boolean success);

    /**
     * 获取用户权限
     * 
     * @param userId 用户ID
     * @return 权限列表
     */
    Object getUserPermissions(Long userId);

    /**
     * 检查用户权限
     * 
     * @param userId 用户ID
     * @param permission 权限标识
     * @return 权限检查结果
     */
    Boolean hasPermission(Long userId, String permission);

    /**
     * 锁定用户账户
     * 
     * @param userId 用户ID
     * @param reason 锁定原因
     * @return 操作结果
     */
    Boolean lockUser(Long userId, String reason);

    /**
     * 解锁用户账户
     * 
     * @param userId 用户ID
     * @return 操作结果
     */
    Boolean unlockUser(Long userId);

    /**
     * 检查账户是否被锁定
     * 
     * @param userId 用户ID
     * @return 锁定状态
     */
    Boolean isUserLocked(Long userId);

    /**
     * 更新最后登录时间
     * 
     * @param userId 用户ID
     */
    void updateLastLoginTime(Long userId);

    /**
     * 获取在线用户数量
     * 
     * @return 在线用户数量
     */
    Long getOnlineUserCount();

    /**
     * 强制用户下线
     * 
     * @param userId 用户ID
     * @return 操作结果
     */
    Boolean forceLogout(Long userId);
}