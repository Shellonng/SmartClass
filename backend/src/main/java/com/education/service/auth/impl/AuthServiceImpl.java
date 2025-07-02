package com.education.service.auth.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.education.dto.AuthDTO;
import com.education.entity.User;
import com.education.entity.Teacher;
import com.education.entity.Student;
import com.education.exception.BusinessException;
import com.education.mapper.UserMapper;
import com.education.mapper.TeacherMapper;
import com.education.mapper.StudentMapper;
import com.education.service.auth.AuthService;
import com.education.utils.PasswordUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;

/**
 * 认证服务实现类
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
@Service
public class AuthServiceImpl implements AuthService {

    @Autowired
    private UserMapper userMapper;
    
    @Autowired
    private TeacherMapper teacherMapper;
    
    @Autowired
    private StudentMapper studentMapper;
    
    @Autowired
    private PasswordUtils passwordUtils;

    @Override
    public AuthDTO.SimpleLoginResponse simpleLogin(AuthDTO.LoginRequest request) {
        // 参数校验
        if (request.getUsername() == null || request.getUsername().trim().isEmpty()) {
            throw new BusinessException("用户名不能为空");
        }
        if (request.getPassword() == null || request.getPassword().trim().isEmpty()) {
            throw new BusinessException("密码不能为空");
        }
        
        // 查询用户
        User user = userMapper.selectByUsernameWithPassword(request.getUsername());
        if (user == null) {
            throw new BusinessException("用户不存在");
        }
        
        // 验证密码
        if (!passwordUtils.matches(request.getPassword(), user.getPassword())) {
            throw new BusinessException("密码错误");
        }
        
        // 验证用户状态
        if (!user.isActive()) {
            throw new BusinessException("账户状态异常: " + user.getStatusDescription());
        }
        
        // 验证用户角色（如果请求中指定了角色）
        if (request.getRole() != null && !request.getRole().isEmpty()) {
            if (!user.getRole().equalsIgnoreCase(request.getRole())) {
                throw new BusinessException("用户角色不匹配");
            }
        }
        
        // 构建返回对象
        AuthDTO.SimpleLoginResponse response = new AuthDTO.SimpleLoginResponse();
        response.setUserId(user.getId());
        response.setUsername(user.getUsername());
        response.setRealName(user.getRealName());
        response.setEmail(user.getEmail());
        response.setUserType(user.getRole());
        
        return response;
    }

    @Override
    @Transactional
    public void simpleRegister(AuthDTO.RegisterRequest request) {
        // 参数校验
        if (request.getUsername() == null || request.getUsername().trim().isEmpty()) {
            throw new BusinessException("用户名不能为空");
        }
        if (request.getPassword() == null || request.getPassword().trim().isEmpty()) {
            throw new BusinessException("密码不能为空");
        }
        if (request.getConfirmPassword() == null || !request.getConfirmPassword().equals(request.getPassword())) {
            throw new BusinessException("两次输入的密码不一致");
        }
        if (request.getEmail() == null || request.getEmail().trim().isEmpty()) {
            throw new BusinessException("邮箱不能为空");
        }
        if (request.getRole() == null || request.getRole().trim().isEmpty()) {
            throw new BusinessException("用户角色不能为空");
        }
        
        // 检查用户名是否已存在
        if (userMapper.checkUsernameExists(request.getUsername(), null) > 0) {
            throw new BusinessException("用户名已存在");
        }
        
        // 检查邮箱是否已存在
        if (userMapper.checkEmailExists(request.getEmail(), null) > 0) {
            throw new BusinessException("邮箱已被注册");
        }
        
        // 创建用户对象
        User user = new User();
        user.setUsername(request.getUsername());
        user.setPassword(passwordUtils.encode(request.getPassword()));
        user.setEmail(request.getEmail());
        user.setRealName(request.getRealName());
        user.setRole(request.getRole().toUpperCase());
        user.setStatus(User.Status.ACTIVE.getCode());
        user.setCreateTime(LocalDateTime.now());
        user.setUpdateTime(LocalDateTime.now());
        
        // 保存用户
        userMapper.insert(user);
        
        // 根据角色创建对应的角色记录
        String role = request.getRole().toUpperCase();
        if ("TEACHER".equals(role)) {
            // 创建教师记录
            Teacher teacher = new Teacher();
            teacher.setUserId(user.getId());
            teacher.setDepartment(request.getDepartment());
            teacher.setTitle(request.getTitle());
            teacher.setCreateTime(LocalDateTime.now());
            teacher.setUpdateTime(LocalDateTime.now());
            teacherMapper.insert(teacher);
            System.out.println("✅ 创建教师记录成功，ID: " + teacher.getId() + ", 用户ID: " + user.getId());
        } else if ("STUDENT".equals(role)) {
            // 创建学生记录
            Student student = new Student();
            student.setUserId(user.getId());
            student.setEnrollmentStatus("ENROLLED");
            student.setCreateTime(LocalDateTime.now());
            student.setUpdateTime(LocalDateTime.now());
            studentMapper.insert(student);
            System.out.println("✅ 创建学生记录成功，ID: " + student.getId() + ", 用户ID: " + user.getId());
        }
    }

    @Override
    @Transactional
    public void changePassword(Long userId, AuthDTO.ChangePasswordRequest request) {
        // 参数校验
        if (userId == null) {
            throw new BusinessException("用户ID不能为空");
        }
        if (request.getOldPassword() == null || request.getOldPassword().trim().isEmpty()) {
            throw new BusinessException("原密码不能为空");
        }
        if (request.getNewPassword() == null || request.getNewPassword().trim().isEmpty()) {
            throw new BusinessException("新密码不能为空");
        }
        if (request.getConfirmPassword() == null || !request.getConfirmPassword().equals(request.getNewPassword())) {
            throw new BusinessException("两次输入的密码不一致");
        }
        
        // 查询用户
        User user = userMapper.selectById(userId);
        if (user == null) {
            throw new BusinessException("用户不存在");
        }
        
        // 验证原密码
        if (!passwordUtils.matches(request.getOldPassword(), user.getPassword())) {
            throw new BusinessException("原密码错误");
        }
        
        // 更新密码
        user.setPassword(passwordUtils.encode(request.getNewPassword()));
        user.setUpdateTime(LocalDateTime.now());
        userMapper.updateById(user);
    }
} 