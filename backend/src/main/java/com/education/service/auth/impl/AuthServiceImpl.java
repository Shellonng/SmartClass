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
    public AuthDTO.LoginResponse register(AuthDTO.RegisterRequest registerRequest) {
        // 1. 验证密码一致性
        if (!registerRequest.getPassword().equals(registerRequest.getConfirmPassword())) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "两次输入的密码不一致");
        }
        
        // 2. 检查用户名是否已存在
        QueryWrapper<User> usernameQuery = new QueryWrapper<>();
        usernameQuery.eq("username", registerRequest.getUsername());
        User existingUser = userMapper.selectOne(usernameQuery);
        if (existingUser != null) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "用户名已存在");
        }
        
        // 3. 检查邮箱是否已存在
        QueryWrapper<User> emailQuery = new QueryWrapper<>();
        emailQuery.eq("email", registerRequest.getEmail());
        User existingEmail = userMapper.selectOne(emailQuery);
        if (existingEmail != null) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "邮箱已被注册");
        }
        
        // 4. 创建新用户
        User newUser = new User();
        newUser.setUsername(registerRequest.getUsername());
        newUser.setPassword(passwordUtils.encode(registerRequest.getPassword())); // 加密密码
        newUser.setEmail(registerRequest.getEmail());
        newUser.setRealName(registerRequest.getRealName());
        newUser.setRole(registerRequest.getRole().toUpperCase()); // STUDENT 或 TEACHER
        newUser.setCreateTime(LocalDateTime.now());
        newUser.setUpdateTime(LocalDateTime.now());
        newUser.setIsDeleted(false);
        
        // 5. 保存用户到数据库
        int result = userMapper.insert(newUser);
        if (result <= 0) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "用户注册失败");
        }
        
        // 6. 根据用户类型创建对应的学生或教师记录
        if ("STUDENT".equals(newUser.getRole())) {
            Student student = new Student();
            student.setUserId(newUser.getId());
            student.setStudentNo(generateStudentNumber());
            student.setCreateTime(LocalDateTime.now());
            student.setUpdateTime(LocalDateTime.now());
            studentMapper.insert(student);
        } else if ("TEACHER".equals(newUser.getRole())) {
            Teacher teacher = new Teacher();
            teacher.setUserId(newUser.getId());
            teacher.setTeacherNo(generateTeacherNumber());
            teacher.setCreateTime(LocalDateTime.now());
            teacher.setUpdateTime(LocalDateTime.now());
            teacherMapper.insert(teacher);
        }
        
        // 7. 自动登录，生成Token
        List<SimpleGrantedAuthority> authorities = new ArrayList<>();
        if ("STUDENT".equals(newUser.getRole())) {
            authorities.add(new SimpleGrantedAuthority("ROLE_STUDENT"));
        } else if ("TEACHER".equals(newUser.getRole())) {
            authorities.add(new SimpleGrantedAuthority("ROLE_TEACHER"));
        }
        
        UserDetails userDetails = org.springframework.security.core.userdetails.User.builder()
                .username(newUser.getUsername())
                .password(newUser.getPassword())
                .authorities(authorities)
                .build();
        
        String token = jwtUtils.generateToken(userDetails);
        String refreshToken = jwtUtils.generateRefreshToken(userDetails);
        
        // 8. 返回注册响应
        AuthDTO.LoginResponse response = new AuthDTO.LoginResponse();
        response.setToken(token);
        response.setRefreshToken(refreshToken);
        response.setUserType(newUser.getRole().toLowerCase());
        response.setUserId(newUser.getId());
        response.setUsername(newUser.getUsername());
        response.setRealName(newUser.getRealName());
        response.setEmail(newUser.getEmail());
        response.setExpiresIn(3600L);
        
        log.info("用户注册成功: username={}, email={}, role={}", 
                newUser.getUsername(), newUser.getEmail(), newUser.getRole());
        
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
    public Object generateCaptcha() {
        // 简单的验证码实现，返回一个4位数字验证码
        String captcha = String.format("%04d", new Random().nextInt(10000));
        
        // 这里可以将验证码存储到Redis中，设置过期时间
        // redisUtils.set("captcha:" + sessionId, captcha, 300, TimeUnit.SECONDS);
        
        // 返回验证码信息
        Map<String, Object> result = new HashMap<>();
        result.put("captcha", captcha);
        result.put("timestamp", System.currentTimeMillis());
        
        return result;
    }

    /**
     * 生成学生学号
     */
    private String generateStudentNumber() {
        // 生成格式: S + 年份 + 6位随机数
        String year = String.valueOf(LocalDateTime.now().getYear());
        String randomNum = String.format("%06d", new Random().nextInt(1000000));
        return "S" + year + randomNum;
    }
    
    /**
     * 生成教师工号
     */
    private String generateTeacherNumber() {
        // 生成格式: T + 年份 + 6位随机数
        String year = String.valueOf(LocalDateTime.now().getYear());
        String randomNum = String.format("%06d", new Random().nextInt(1000000));
        return "T" + year + randomNum;
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
        log.info("获取用户信息: userId={}", userId);
        try {
            // 1. 查询用户基本信息
        User user = userMapper.selectById(userId);
        if (user == null) {
            throw new BusinessException(ResultCode.USER_NOT_FOUND);
        }
        
            // 2. 构建用户信息响应
        Map<String, Object> userInfo = new HashMap<>();
            userInfo.put("id", user.getId());
        userInfo.put("username", user.getUsername());
            userInfo.put("realName", user.getRealName());
        userInfo.put("email", user.getEmail());
        userInfo.put("phone", user.getPhone());
        userInfo.put("avatar", user.getAvatar());
            userInfo.put("role", user.getRole().toLowerCase());
        userInfo.put("status", user.getStatus());
        userInfo.put("lastLoginTime", user.getLastLoginTime());
        
            // 3. 根据角色获取额外信息
            if (user.isStudent()) {
                // 获取学生信息
            QueryWrapper<Student> studentQuery = new QueryWrapper<>();
            studentQuery.eq("user_id", userId);
            Student student = studentMapper.selectOne(studentQuery);
            if (student != null) {
                userInfo.put("studentNo", student.getStudentNo());
                userInfo.put("major", student.getMajor());
                userInfo.put("grade", student.getGrade());
                    userInfo.put("classId", student.getClassId());
            }
            } else if (user.isTeacher()) {
                // 获取教师信息
            QueryWrapper<Teacher> teacherQuery = new QueryWrapper<>();
            teacherQuery.eq("user_id", userId);
            Teacher teacher = teacherMapper.selectOne(teacherQuery);
            if (teacher != null) {
                userInfo.put("teacherNo", teacher.getTeacherNo());
                userInfo.put("department", teacher.getDepartment());
                userInfo.put("title", teacher.getTitle());
                    userInfo.put("specialty", teacher.getSpecialization());
            }
        }
        
        return userInfo;
        } catch (BusinessException e) {
            throw e;
        } catch (Exception e) {
            log.error("获取用户信息失败: userId={}", userId, e);
            throw new BusinessException(ResultCode.INTERNAL_SERVER_ERROR, "获取用户信息失败");
        }
    }

    @Override
    public Boolean validateToken(String token) {
        try {
            // 1. 检查token是否为空
            if (token == null || token.isEmpty()) {
                return false;
            }
            
            // 2. 检查token是否在黑名单中
            if (redisUtils.hasKey("blacklist:" + token)) {
                return false;
            }
            
            // 3. 从token中获取用户名
            String username = jwtUtils.getUsernameFromToken(token);
            if (username == null) {
                return false;
            }
            
            // 4. 检查token是否过期
            if (jwtUtils.getExpirationDateFromToken(token).before(new Date())) {
                return false;
            }
            
            // 5. 验证用户是否存在
            QueryWrapper<User> userQuery = new QueryWrapper<>();
            userQuery.eq("username", username);
            User user = userMapper.selectOne(userQuery);
            if (user == null) {
                return false;
            }
            
            // 6. 检查用户状态
            if (user.getIsDeleted() != null && user.getIsDeleted()) {
                    return false;
            }
            
            return true;
        } catch (Exception e) {
            log.error("Token验证失败", e);
            return false;
        }
    }

    @Override
    public Boolean checkUserExists(String username) {
        QueryWrapper<User> userQuery = new QueryWrapper<>();
        userQuery.eq("username", username);
        return userMapper.selectCount(userQuery) > 0;
    }

    @Override
    public Boolean checkEmailExists(String email) {
        QueryWrapper<User> emailQuery = new QueryWrapper<>();
        emailQuery.eq("email", email);
        return userMapper.selectCount(emailQuery) > 0;
    }

    @Override
    public void recordLoginLog(Long userId, String ip, String userAgent, Boolean success) {
        // 这里应该记录登录日志到数据库
        // 简单实现：记录到Redis中
        try {
            Map<String, Object> logInfo = new HashMap<>();
            logInfo.put("userId", userId);
            logInfo.put("ip", ip);
            logInfo.put("userAgent", userAgent);
            logInfo.put("success", success);
            logInfo.put("time", LocalDateTime.now().toString());
            
            String logKey = "login_log:" + userId + ":" + System.currentTimeMillis();
            redisUtils.set(logKey, logInfo, 7, TimeUnit.DAYS);
            
            // 更新最后一次登录信息
            if (success) {
                User user = new User();
                user.setId(userId);
                user.setLastLoginTime(LocalDateTime.now());
                user.setLastLoginIp(ip);
                userMapper.updateById(user);
            }
        } catch (Exception e) {
            log.error("记录登录日志失败: userId={}", userId, e);
        }
    }

    @Override
    public Object getUserPermissions(Long userId) {
        try {
            // 1. 查询用户信息
        User user = userMapper.selectById(userId);
        if (user == null) {
            throw new BusinessException(ResultCode.USER_NOT_FOUND);
        }
        
            // 2. 构建权限信息
            Map<String, Object> permissionInfo = new HashMap<>();
            List<String> permissions = new ArrayList<>();
            List<String> roles = new ArrayList<>();
            
            // 3. 添加基本角色
            roles.add(user.getRole());
        
            // 4. 根据角色添加权限
            if ("STUDENT".equals(user.getRole())) {
                permissions.add("course:view");
                permissions.add("task:view");
                permissions.add("task:submit");
                permissions.add("resource:view");
                permissions.add("grade:view");
        } else if ("TEACHER".equals(user.getRole())) {
                permissions.add("course:*");
                permissions.add("task:*");
                permissions.add("resource:*");
                permissions.add("grade:*");
                permissions.add("class:*");
                permissions.add("student:view");
            } else if ("ADMIN".equals(user.getRole())) {
                permissions.add("*:*");  // 管理员拥有所有权限
            }
            
            permissionInfo.put("roles", roles);
            permissionInfo.put("permissions", permissions);
            
            return permissionInfo;
        } catch (BusinessException e) {
            throw e;
        } catch (Exception e) {
            log.error("获取用户权限失败: userId={}", userId, e);
            throw new BusinessException(ResultCode.INTERNAL_SERVER_ERROR, "获取用户权限失败");
        }
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
            
            if (!(permissionsObj instanceof List)) {
                return false;
            }
            
            @SuppressWarnings("unchecked")
            List<String> permissions = (List<String>) permissionsObj;
            
            // 2. 检查是否包含通配符权限
            if (permissions.contains("*:*")) {
                return true;
            }
            
            // 3. 检查是否包含模块通配符权限
            String module = permission.split(":")[0];
            if (permissions.contains(module + ":*")) {
                return true;
            }
            
            // 4. 检查是否包含指定权限
            return permissions.contains(permission);
        } catch (Exception e) {
            log.error("检查用户权限失败", e);
            return false;
        }
    }

    @Override
    public Boolean lockUser(Long userId, String reason) {
        try {
            // 1. 查询用户信息
            User user = userMapper.selectById(userId);
            if (user == null) {
                throw new BusinessException(ResultCode.USER_NOT_FOUND);
            }
            
            // 2. 更新用户状态为锁定
            user.setStatus(User.Status.LOCKED.getCode());
            user.setUpdateTime(LocalDateTime.now());
            
            // 3. 记录锁定原因
            if (reason != null && !reason.isEmpty()) {
                // 可以将锁定原因存储在扩展字段中
                user.setExtField1(reason);
            }
            
            // 4. 更新用户信息
            int result = userMapper.updateById(user);
            
            // 5. 记录操作日志
            if (result > 0) {
                // 记录锁定信息到Redis
                Map<String, Object> lockInfo = new HashMap<>();
                lockInfo.put("userId", userId);
                lockInfo.put("reason", reason);
                lockInfo.put("lockTime", LocalDateTime.now().toString());
                
                redisUtils.set("user_lock:" + userId, lockInfo);
                
                return true;
            }
            
            return false;
        } catch (BusinessException e) {
            throw e;
        } catch (Exception e) {
            log.error("锁定用户失败: userId={}", userId, e);
            throw new BusinessException(ResultCode.INTERNAL_SERVER_ERROR, "锁定用户失败");
        }
    }

    @Override
    public Boolean unlockUser(Long userId) {
        try {
            // 1. 查询用户信息
            User user = userMapper.selectById(userId);
            if (user == null) {
                throw new BusinessException(ResultCode.USER_NOT_FOUND);
            }
            
            // 2. 更新用户状态为正常
            user.setStatus(User.Status.ACTIVE.getCode());
            user.setUpdateTime(LocalDateTime.now());
            
            // 3. 清除锁定原因
            user.setExtField1(null);
            
            // 4. 更新用户信息
            int result = userMapper.updateById(user);
            
            // 5. 记录操作日志
            if (result > 0) {
                // 删除Redis中的锁定信息
                redisUtils.delete("user_lock:" + userId);
                
                return true;
            }
            
            return false;
        } catch (BusinessException e) {
            throw e;
        } catch (Exception e) {
            log.error("解锁用户失败: userId={}", userId, e);
            throw new BusinessException(ResultCode.INTERNAL_SERVER_ERROR, "解锁用户失败");
        }
    }

    @Override
    public Boolean isUserLocked(Long userId) {
        try {
            // 1. 查询用户信息
            User user = userMapper.selectById(userId);
            if (user == null) {
                throw new BusinessException(ResultCode.USER_NOT_FOUND);
            }
            
            // 2. 检查用户状态
            return User.Status.LOCKED.getCode().equals(user.getStatus());
        } catch (BusinessException e) {
            throw e;
        } catch (Exception e) {
            log.error("检查用户锁定状态失败: userId={}", userId, e);
            throw new BusinessException(ResultCode.INTERNAL_SERVER_ERROR, "检查用户锁定状态失败");
        }
    }

    @Override
    public void updateLastLoginTime(Long userId) {
        try {
        User user = new User();
        user.setId(userId);
        user.setLastLoginTime(LocalDateTime.now());
        userMapper.updateById(user);
        } catch (Exception e) {
            log.error("更新最后登录时间失败: userId={}", userId, e);
        }
    }

    @Override
    public Long getOnlineUserCount() {
        try {
            // 简单实现：通过Redis中的活跃用户会话数量来统计
            // 实际应用中，可能需要更复杂的逻辑
            return Long.valueOf(redisUtils.keys("user_session:*").size());
        } catch (Exception e) {
            log.error("获取在线用户数量失败", e);
            return 0L;
        }
    }

    @Override
    public Boolean forceLogout(Long userId) {
        try {
            // 1. 查询用户信息
            User user = userMapper.selectById(userId);
            if (user == null) {
                throw new BusinessException(ResultCode.USER_NOT_FOUND);
            }
            
            // 2. 清除用户会话
            redisUtils.delete("user_session:" + user.getUsername());
            
            // 3. 记录操作日志
            Map<String, Object> logoutInfo = new HashMap<>();
            logoutInfo.put("userId", userId);
            logoutInfo.put("time", LocalDateTime.now().toString());
            logoutInfo.put("type", "force");
            
            redisUtils.set("logout_log:" + userId + ":" + System.currentTimeMillis(), logoutInfo, 7, TimeUnit.DAYS);
            
            return true;
        } catch (BusinessException e) {
            throw e;
        } catch (Exception e) {
            log.error("强制用户下线失败: userId={}", userId, e);
            throw new BusinessException(ResultCode.INTERNAL_SERVER_ERROR, "强制用户下线失败");
        }
    }
}