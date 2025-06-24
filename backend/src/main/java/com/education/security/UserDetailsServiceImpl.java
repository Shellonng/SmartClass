package com.education.security;

import com.education.entity.Student;
import com.education.entity.Teacher;
import com.education.mapper.StudentMapper;
import com.education.mapper.TeacherMapper;
import com.education.mapper.UserMapper;
import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.security.core.userdetails.User;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;

/**
 * 用户详情服务实现
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
@Service
public class UserDetailsServiceImpl implements UserDetailsService {

    @Autowired
    private StudentMapper studentMapper;

    @Autowired
    private TeacherMapper teacherMapper;

    @Autowired
    private UserMapper userMapper;

    @Override
    public UserDetails loadUserByUsername(String username) throws UsernameNotFoundException {
        // 直接从用户表查找
        QueryWrapper<com.education.entity.User> userQuery = new QueryWrapper<>();
        userQuery.eq("username", username).or().eq("email", username);
        com.education.entity.User user = userMapper.selectOne(userQuery);
        
        if (user != null) {
            List<SimpleGrantedAuthority> authorities = new ArrayList<>();
            
            // 根据用户角色添加权限
            if ("STUDENT".equals(user.getRole())) {
                authorities.add(new SimpleGrantedAuthority("ROLE_STUDENT"));
            } else if ("TEACHER".equals(user.getRole())) {
                authorities.add(new SimpleGrantedAuthority("ROLE_TEACHER"));
            } else {
                authorities.add(new SimpleGrantedAuthority("ROLE_USER"));
            }
            
            return User.builder()
                    .username(user.getUsername())
                    .password(user.getPassword())
                    .authorities(authorities)
                    .accountExpired(false)
                    .accountLocked(false)
                    .credentialsExpired(false)
                    .disabled(user.getIsDeleted() != null && user.getIsDeleted())
                    .build();
        }
        
        throw new UsernameNotFoundException("用户不存在: " + username);
    }
}