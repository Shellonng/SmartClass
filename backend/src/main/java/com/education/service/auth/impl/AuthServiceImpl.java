package com.education.service.auth.impl;

import com.education.dto.AuthDTO;
import com.education.entity.User;
import com.education.entity.Student;
import com.education.entity.Teacher;
import com.education.exception.BusinessException;
import com.education.exception.ResultCode;
import com.education.mapper.UserMapper;
import com.education.mapper.StudentMapper;
import com.education.mapper.TeacherMapper;
import com.education.service.auth.AuthService;
import com.education.service.common.EmailService;
import com.education.utils.JwtUtils;
import com.education.utils.PasswordUtils;
import com.education.utils.RedisUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.stereotype.Service;
import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.TimeUnit;
import java.util.HashMap;
import java.util.Map;
import java.util.Date;

/**
 * 认证服务实现类
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
@Service
public class AuthServiceImpl implements AuthService {

    private static final Logger log = LoggerFactory.getLogger(AuthServiceImpl.class);

    @Autowired
    private UserMapper userMapper;
    
    @Autowired
    private StudentMapper studentMapper;
    
    @Autowired
    private TeacherMapper teacherMapper;
    
    @Autowired
    private JwtUtils jwtUtils;
    
    @Autowired
    private PasswordUtils passwordUtils;
    
    @Autowired
    private RedisUtils redisUtils;
    
    @Autowired
    private EmailService emailService;

    @Override
    public AuthDTO.LoginResponse login(AuthDTO.LoginRequest loginRequest) {
        // 1. 验证用户名和密码
        QueryWrapper<User> userQuery = new QueryWrapper<>();
        userQuery.eq("username", loginRequest.getUsername())
                .or().eq("email", loginRequest.getUsername());
        User user = userMapper.selectOne(userQuery);
        
        if (user == null) {
            throw new BusinessException(ResultCode.USER_PASSWORD_ERROR);
        }
        
        // 验证密码 - 临时支持明文密码比较
        boolean passwordMatches = false;
        try {
            // 先尝试BCrypt验证
            passwordMatches = passwordUtils.matches(loginRequest.getPassword(), user.getPassword());
        } catch (Exception e) {
            // BCrypt验证失败，尝试明文比较
            passwordMatches = loginRequest.getPassword().equals(user.getPassword());
        }
        
        // 如果BCrypt验证失败，再尝试明文比较
        if (!passwordMatches) {
            passwordMatches = loginRequest.getPassword().equals(user.getPassword());
        }
        
        if (!passwordMatches) {
            throw new BusinessException(ResultCode.USER_PASSWORD_ERROR);
        }
        
        // 2. 检查账户状态
        if (user.getIsDeleted() != null && user.getIsDeleted()) {
            throw new BusinessException(ResultCode.USER_DISABLED);
        }
        
        // 3. 生成JWT Token
        List<SimpleGrantedAuthority> authorities = new ArrayList<>();
        if ("STUDENT".equals(user.getRole())) {
            authorities.add(new SimpleGrantedAuthority("ROLE_STUDENT"));
        } else if ("TEACHER".equals(user.getRole())) {
            authorities.add(new SimpleGrantedAuthority("ROLE_TEACHER"));
        } else {
            authorities.add(new SimpleGrantedAuthority("ROLE_USER"));
        }
        
        UserDetails userDetails = org.springframework.security.core.userdetails.User.builder()
                .username(user.getUsername())
                .password(user.getPassword())
                .authorities(authorities)
                .build();
        
        String token = jwtUtils.generateToken(userDetails);
        String refreshToken = jwtUtils.generateRefreshToken(userDetails);
        
        // 4. 更新最后登录时间
        updateLastLoginTime(user.getId());
        
        // 5. 返回登录响应 - 匹配前端期望的数据结构
        AuthDTO.LoginResponse response = new AuthDTO.LoginResponse();
        response.setToken(token);
        response.setRefreshToken(refreshToken);
        
        // 创建用户信息对象，匹配前端期望结构
        Map<String, Object> userInfo = new HashMap<>();
        userInfo.put("id", user.getId());
        userInfo.put("username", user.getUsername());
        userInfo.put("realName", user.getRealName());
        userInfo.put("email", user.getEmail());
        userInfo.put("role", user.getRole().toLowerCase()); // 转换为小写，匹配前端期望
        userInfo.put("avatar", user.getAvatar());
        
        // 由于AuthDTO.LoginResponse不能直接存储userInfo，我们需要修改响应结构
        // 暂时先设置现有字段
        response.setUserType(user.getRole().toLowerCase());
        response.setUserId(user.getId());
        response.setUsername(user.getUsername());
        response.setRealName(user.getRealName());
        response.setEmail(user.getEmail());
        response.setExpiresIn(3600L); // 1小时过期
        
        return response;
    }

    @Override
    public Boolean logout(String token) {
        try {
            // 1. 验证Token有效性和获取用户信息
            String username = jwtUtils.getUsernameFromToken(token);
            if (username == null || jwtUtils.getExpirationDateFromToken(token).before(new Date())) {
                return false;
            }
            
            // 2. 将Token加入黑名单
            Long expiration = jwtUtils.getExpirationDateFromToken(token).getTime();
            long ttl = expiration - System.currentTimeMillis();
            if (ttl > 0) {
                redisUtils.set("blacklist:" + token, true, ttl, TimeUnit.MILLISECONDS);
            }
            
            // 3. 清除Redis中的用户会话信息
            redisUtils.delete("user_session:" + username);
            
            // 4. 记录登出日志（这里简化处理）
            return true;
        } catch (Exception e) {
            return false;
        }
    }

    @Override
    public AuthDTO.LoginResponse refreshToken(AuthDTO.RefreshTokenRequest refreshTokenRequest) {
        String refreshToken = refreshTokenRequest.getRefreshToken();
        
        // 1. 验证Refresh Token有效性
        try {
            String username = jwtUtils.getUsernameFromToken(refreshToken);
            if (username == null || jwtUtils.getExpirationDateFromToken(refreshToken).before(new Date())) {
                throw new BusinessException(ResultCode.TOKEN_INVALID);
            }
        } catch (Exception e) {
            throw new BusinessException(ResultCode.TOKEN_INVALID);
        }
        
        // 2. 检查用户状态
        String username = jwtUtils.getUsernameFromToken(refreshToken);
        QueryWrapper<User> userQuery = new QueryWrapper<>();
        userQuery.eq("username", username);
        User user = userMapper.selectOne(userQuery);
        
        if (user == null || (user.getIsDeleted() != null && user.getIsDeleted())) {
            throw new BusinessException(ResultCode.USER_NOT_FOUND);
        }
        
        // 3. 生成新的Access Token
        List<SimpleGrantedAuthority> authorities = new ArrayList<>();
        if ("STUDENT".equals(user.getRole())) {
            authorities.add(new SimpleGrantedAuthority("ROLE_STUDENT"));
        } else if ("TEACHER".equals(user.getRole())) {
            authorities.add(new SimpleGrantedAuthority("ROLE_TEACHER"));
        } else {
            authorities.add(new SimpleGrantedAuthority("ROLE_USER"));
        }
        
        UserDetails userDetails = org.springframework.security.core.userdetails.User.builder()
                .username(user.getUsername())
                .password(user.getPassword())
                .authorities(authorities)
                .build();
        
        String newToken = jwtUtils.generateToken(userDetails);
        String newRefreshToken = jwtUtils.generateRefreshToken(userDetails);
        
        // 4. 返回新的Token信息
        AuthDTO.LoginResponse response = new AuthDTO.LoginResponse();
        response.setToken(newToken);
        response.setRefreshToken(newRefreshToken);
        response.setUserType(user.getRole());
        response.setUserId(user.getId());
        response.setUsername(user.getUsername());
        response.setRealName(user.getRealName());
        response.setEmail(user.getEmail());
        response.setExpiresIn(3600L);
        
        return response;
    }

    @Override
    public Boolean changePassword(AuthDTO.ChangePasswordRequest changePasswordRequest, Long userId) {
        // 1. 验证原密码正确性
        User user = userMapper.selectById(userId);
        if (user == null) {
            throw new BusinessException(ResultCode.USER_NOT_FOUND);
        }
        
        if (!passwordUtils.matches(changePasswordRequest.getOldPassword(), user.getPassword())) {
            throw new BusinessException(ResultCode.USER_PASSWORD_ERROR);
        }
        
        // 2. 验证新密码强度（检查新密码和确认密码是否一致）
        if (!changePasswordRequest.getNewPassword().equals(changePasswordRequest.getConfirmPassword())) {
            throw new BusinessException(ResultCode.PARAM_ERROR);
        }
        
        // 3. 加密新密码
        String encodedPassword = passwordUtils.encode(changePasswordRequest.getNewPassword());
        
        // 4. 更新数据库中的密码
        User updateUser = new User();
        updateUser.setId(userId);
        updateUser.setPassword(encodedPassword);
        updateUser.setUpdateTime(LocalDateTime.now());
        userMapper.updateById(updateUser);
        
        // 5. 使所有现有Token失效（将用户加入强制登出列表）
        redisUtils.set("force_logout:" + userId, true, 24, TimeUnit.HOURS);
        
        // 6. 发送密码修改通知邮件
        try {
            emailService.sendSimpleEmail(user.getEmail(), "密码修改通知", 
                "您的账户密码已成功修改，如非本人操作请及时联系管理员。");
        } catch (Exception e) {
            // 邮件发送失败不影响密码修改
        }
        
        return true;
    }

    @Override
    public Boolean sendResetPasswordEmail(String email) {
        // 1. 检查邮箱是否存在
        QueryWrapper<User> userQuery = new QueryWrapper<>();
        userQuery.eq("email", email);
        User user = userMapper.selectOne(userQuery);
        
        if (user == null) {
            throw new BusinessException(ResultCode.USER_NOT_FOUND);
        }
        
        // 2. 生成重置密码Token（6位随机数字）
        Random random = new Random();
        String resetToken = String.format("%06d", random.nextInt(1000000));
        
        // 3. 将Token存储到Redis（设置过期时间15分钟）
        redisUtils.set("reset_password:" + email, resetToken, 15, TimeUnit.MINUTES);
        
        // 4. 发送重置密码邮件
        try {
            String subject = "密码重置验证码";
            String content = String.format("您的密码重置验证码是：%s，有效期15分钟，请勿泄露给他人。", resetToken);
            return emailService.sendSimpleEmail(email, subject, content);
        } catch (Exception e) {
            throw new BusinessException(ResultCode.EMAIL_SEND_FAILED);
        }
    }

    @Override
    public Boolean resetPassword(AuthDTO.ResetPasswordRequest resetPasswordRequest) {
        String email = resetPasswordRequest.getEmail();
        String verificationCode = resetPasswordRequest.getVerificationCode();
        String newPassword = resetPasswordRequest.getNewPassword();
        
        // 1. 验证重置Token有效性
        String storedCode = (String) redisUtils.get("reset_password:" + email);
        if (storedCode == null || !storedCode.equals(verificationCode)) {
            throw new BusinessException(ResultCode.VERIFICATION_CODE_ERROR);
        }
        
        // 2. 查找用户
        QueryWrapper<User> userQuery = new QueryWrapper<>();
        userQuery.eq("email", email);
        User user = userMapper.selectOne(userQuery);
        
        if (user == null) {
            throw new BusinessException(ResultCode.USER_NOT_FOUND);
        }
        
        // 3. 加密新密码
        String encodedPassword = passwordUtils.encode(newPassword);
        
        // 4. 更新数据库中的密码
        User updateUser = new User();
        updateUser.setId(user.getId());
        updateUser.setPassword(encodedPassword);
        updateUser.setUpdateTime(LocalDateTime.now());
        userMapper.updateById(updateUser);
        
        // 5. 删除重置Token
        redisUtils.delete("reset_password:" + email);
        
        // 6. 发送密码重置成功通知
        try {
            emailService.sendSimpleEmail(email, "密码重置成功", 
                "您的账户密码已成功重置，请使用新密码登录。");
        } catch (Exception e) {
            // 邮件发送失败不影响密码重置
        }
        
        return true;
    }

    @Override
    public Boolean sendVerificationCode(String email, String type) {
        // 1. 生成6位数字验证码
        Random random = new Random();
        String verificationCode = String.format("%06d", random.nextInt(1000000));
        
        // 2. 将验证码存储到Redis（设置过期时间5分钟）
        String redisKey = "verification_code:" + type + ":" + email;
        redisUtils.set(redisKey, verificationCode, 5, TimeUnit.MINUTES);
        
        // 3. 发送验证码邮件
        try {
            String subject = "验证码";
            String content = String.format("您的验证码是：%s，有效期5分钟，请勿泄露给他人。", verificationCode);
            boolean result = emailService.sendSimpleEmail(email, subject, content);
            
            // 4. 记录发送日志（这里简化处理）
            if (result) {
                redisUtils.set("verification_log:" + email + ":" + System.currentTimeMillis(), 
                    "发送验证码成功", 1, TimeUnit.HOURS);
            }
            
            return result;
        } catch (Exception e) {
            throw new BusinessException(ResultCode.EMAIL_SEND_FAILED);
        }
    }

    @Override
    public Boolean verifyCode(String email, String code, String type) {
        // 1. 从Redis获取验证码
        String redisKey = "verification_code:" + type + ":" + email;
        String storedCode = (String) redisUtils.get(redisKey);
        
        // 2. 比较验证码是否正确
        if (storedCode == null || !storedCode.equals(code)) {
            // 记录验证失败日志
            redisUtils.set("verification_log:" + email + ":" + System.currentTimeMillis(), 
                "验证码验证失败", 1, TimeUnit.HOURS);
            return false;
        }
        
        // 3. 验证成功后删除验证码
        redisUtils.delete(redisKey);
        
        // 4. 记录验证日志
        redisUtils.set("verification_log:" + email + ":" + System.currentTimeMillis(), 
            "验证码验证成功", 1, TimeUnit.HOURS);
        
        return true;
    }

    @Override
    public Object getCurrentUserInfo(Long userId) {
        // 1. 根据用户ID查询用户基本信息
        User user = userMapper.selectById(userId);
        if (user == null) {
            throw new BusinessException(ResultCode.USER_NOT_FOUND);
        }
        
        // 2. 根据用户类型查询详细信息（学生/教师）
        Map<String, Object> userInfo = new HashMap<>();
        userInfo.put("userId", user.getId());
        userInfo.put("username", user.getUsername());
        userInfo.put("email", user.getEmail());
        userInfo.put("phone", user.getPhone());
        userInfo.put("realName", user.getRealName());
        userInfo.put("avatar", user.getAvatar());
        userInfo.put("role", user.getRole());
        userInfo.put("status", user.getStatus());
        userInfo.put("lastLoginTime", user.getLastLoginTime());
        
        if ("STUDENT".equals(user.getRole())) {
            QueryWrapper<Student> studentQuery = new QueryWrapper<>();
            studentQuery.eq("user_id", userId);
            Student student = studentMapper.selectOne(studentQuery);
            if (student != null) {
                userInfo.put("studentNo", student.getStudentNo());
                userInfo.put("classId", student.getClassId());
                userInfo.put("major", student.getMajor());
                userInfo.put("grade", student.getGrade());
            }
        } else if ("TEACHER".equals(user.getRole())) {
            QueryWrapper<Teacher> teacherQuery = new QueryWrapper<>();
            teacherQuery.eq("user_id", userId);
            Teacher teacher = teacherMapper.selectOne(teacherQuery);
            if (teacher != null) {
                userInfo.put("teacherNo", teacher.getTeacherNo());
                userInfo.put("department", teacher.getDepartment());
                userInfo.put("title", teacher.getTitle());
            }
        }
        
        // 3. 脱敏处理敏感信息（密码已在User实体中用@JsonIgnore处理）
        // 4. 返回用户信息
        return userInfo;
    }

    @Override
    public Boolean validateToken(String token) {
        try {
            // 1. 检查Token格式和验证Token签名、检查Token是否过期
            String username = jwtUtils.getUsernameFromToken(token);
            if (username == null || jwtUtils.getExpirationDateFromToken(token).before(new Date())) {
                return false;
            }
            
            // 2. 检查Token是否在黑名单中
            Boolean isBlacklisted = (Boolean) redisUtils.get("blacklist:" + token);
            if (isBlacklisted != null && isBlacklisted) {
                return false;
            }
            
            // 3. 检查用户是否被强制登出
            QueryWrapper<User> userQuery = new QueryWrapper<>();
            userQuery.eq("username", username);
            User user = userMapper.selectOne(userQuery);
            
            if (user != null) {
                Boolean forceLogout = (Boolean) redisUtils.get("force_logout:" + user.getId());
                if (forceLogout != null && forceLogout) {
                    return false;
                }
            }
            
            return true;
        } catch (Exception e) {
            return false;
        }
    }

    @Override
    public Boolean checkUserExists(String username) {
        // 1. 查询用户表
        QueryWrapper<User> userQuery = new QueryWrapper<>();
        userQuery.eq("username", username);
        User user = userMapper.selectOne(userQuery);
        
        // 2. 返回是否存在
        return user != null;
    }

    @Override
    public Boolean checkEmailExists(String email) {
        // 1. 查询用户表中的邮箱字段
        QueryWrapper<User> userQuery = new QueryWrapper<>();
        userQuery.eq("email", email);
        User user = userMapper.selectOne(userQuery);
        
        // 2. 返回是否存在
        return user != null;
    }

    @Override
    public void recordLoginLog(Long userId, String ip, String userAgent, Boolean success) {
        try {
            // 1. 创建登录日志记录
            Map<String, Object> loginLog = new HashMap<>();
            loginLog.put("userId", userId);
            loginLog.put("ip", ip);
            loginLog.put("userAgent", userAgent);
            loginLog.put("success", success);
            loginLog.put("loginTime", new Date());
            
            // 2. 保存到Redis（简化处理，实际项目中可能需要专门的日志表）
            String logKey = "login_log:" + userId + ":" + System.currentTimeMillis();
            redisUtils.set(logKey, loginLog, 30, TimeUnit.DAYS);
            
            // 3. 记录日志
            log.info("用户登录日志记录 - 用户ID: {}, IP: {}, 成功: {}", userId, ip, success);
        } catch (Exception e) {
            log.error("记录登录日志失败", e);
        }
    }

    @Override
    public Object getUserPermissions(Long userId) {
        // 1. 根据用户类型获取基础权限
        User user = userMapper.selectById(userId);
        if (user == null) {
            throw new BusinessException(ResultCode.USER_NOT_FOUND);
        }
        
        Map<String, Object> permissions = new HashMap<>();
        permissions.put("userId", userId);
        permissions.put("role", user.getRole());
        
        // 2. 根据用户角色设置权限
        if ("ADMIN".equals(user.getRole())) {
            permissions.put("permissions", java.util.Arrays.asList(
                "user:read", "user:write", "user:delete",
                "course:read", "course:write", "course:delete",
                "system:read", "system:write", "system:delete"
            ));
        } else if ("TEACHER".equals(user.getRole())) {
            permissions.put("permissions", java.util.Arrays.asList(
                "course:read", "course:write",
                "student:read", "grade:write"
            ));
        } else if ("STUDENT".equals(user.getRole())) {
            permissions.put("permissions", java.util.Arrays.asList(
                "course:read", "profile:read", "profile:write"
            ));
        } else {
            permissions.put("permissions", java.util.Arrays.asList());
        }
        
        // 3. 返回权限信息
        return permissions;
    }

    @Override
    public Boolean hasPermission(Long userId, String permission) {
        try {
            // 1. 获取用户权限列表
            Object userPermissionsObj = getUserPermissions(userId);
            if (!(userPermissionsObj instanceof Map)) {
                return false;
            }
            
            @SuppressWarnings("unchecked")
            Map<String, Object> userPermissions = (Map<String, Object>) userPermissionsObj;
            Object permissionsObj = userPermissions.get("permissions");
            
            if (!(permissionsObj instanceof java.util.List)) {
                return false;
            }
            
            @SuppressWarnings("unchecked")
            java.util.List<String> permissions = (java.util.List<String>) permissionsObj;
            
            // 2. 检查是否包含指定权限
            return permissions.contains(permission);
        } catch (Exception e) {
            log.error("检查用户权限失败", e);
            return false;
        }
    }

    @Override
    public Boolean lockUser(Long userId, String reason) {
        try {
            // 1. 更新用户状态为锁定
            User user = userMapper.selectById(userId);
            if (user == null) {
                throw new BusinessException(ResultCode.USER_NOT_FOUND);
            }
            
            user.setStatus("LOCKED");
            int result = userMapper.updateById(user);
            
            if (result > 0) {
                // 2. 记录锁定原因和时间
                Map<String, Object> lockInfo = new HashMap<>();
                lockInfo.put("userId", userId);
                lockInfo.put("reason", reason);
                lockInfo.put("lockTime", new Date());
                redisUtils.set("user_lock:" + userId, lockInfo, 365, TimeUnit.DAYS);
                
                // 3. 使所有现有Token失效
                redisUtils.set("force_logout:" + userId, true, 24, TimeUnit.HOURS);
                
                // 4. 发送账户锁定通知
                try {
                    emailService.sendSimpleEmail(
                        user.getEmail(),
                        "账户锁定通知",
                        "您的账户已被锁定。锁定原因：" + reason + "。如有疑问，请联系管理员。"
                    );
                } catch (Exception e) {
                    log.error("发送账户锁定通知邮件失败", e);
                }
                
                log.info("用户账户已锁定 - 用户ID: {}, 原因: {}", userId, reason);
                return true;
            }
            return false;
        } catch (Exception e) {
            log.error("锁定用户账户失败", e);
            return false;
        }
    }

    @Override
    public Boolean unlockUser(Long userId) {
        try {
            // 1. 更新用户状态为正常
            User user = userMapper.selectById(userId);
            if (user == null) {
                throw new BusinessException(ResultCode.USER_NOT_FOUND);
            }
            
            user.setStatus("ACTIVE");
            int result = userMapper.updateById(user);
            
            if (result > 0) {
                // 2. 清除锁定记录
                redisUtils.delete("user_lock:" + userId);
                redisUtils.delete("force_logout:" + userId);
                
                // 3. 发送账户解锁通知
                try {
                    emailService.sendSimpleEmail(
                        user.getEmail(),
                        "账户解锁通知",
                        "您的账户已被解锁，现在可以正常使用。如有疑问，请联系管理员。"
                    );
                } catch (Exception e) {
                    log.error("发送账户解锁通知邮件失败", e);
                }
                
                log.info("用户账户已解锁 - 用户ID: {}", userId);
                return true;
            }
            return false;
        } catch (Exception e) {
            log.error("解锁用户账户失败", e);
            return false;
        }
    }

    @Override
    public Boolean isUserLocked(Long userId) {
        try {
            // 1. 查询用户状态
            User user = userMapper.selectById(userId);
            if (user == null) {
                return false;
            }
            
            // 2. 检查用户状态是否为锁定
            if ("LOCKED".equals(user.getStatus())) {
                return true;
            }
            
            // 3. 检查Redis中的锁定记录
            Object lockInfo = redisUtils.get("user_lock:" + userId);
            return lockInfo != null;
        } catch (Exception e) {
            log.error("检查用户锁定状态失败", e);
            return false;
        }
    }

    @Override
    public void updateLastLoginTime(Long userId) {
        // 更新用户表中的最后登录时间
        User user = new User();
        user.setId(userId);
        user.setLastLoginTime(LocalDateTime.now());
        user.setUpdateTime(LocalDateTime.now());
        userMapper.updateById(user);
    }

    @Override
    public Long getOnlineUserCount() {
        // TODO: 实现获取在线用户数量逻辑
        // 1. 从Redis获取在线用户信息
        // 2. 统计在线用户数量
        // 3. 返回统计结果
        throw new RuntimeException("方法未实现");
    }

    @Override
    public Boolean forceLogout(Long userId) {
        // TODO: 实现强制用户下线逻辑
        // 1. 将用户所有Token加入黑名单
        // 2. 清除Redis中的用户会话信息
        // 3. 记录强制下线日志
        // 4. 发送下线通知
        throw new RuntimeException("方法未实现");
    }
}