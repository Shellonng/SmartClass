package com.education.security;

import org.springframework.stereotype.Component;
import org.springframework.web.context.request.RequestContextHolder;
import org.springframework.web.context.request.ServletRequestAttributes;

import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpSession;

/**
 * 安全工具类，用于获取当前登录用户信息
 */
@Component
public class SecurityUtil {

    /**
     * 获取当前登录用户ID
     *
     * @return 用户ID
     */
    public Long getCurrentUserId() {
        HttpServletRequest request = ((ServletRequestAttributes) RequestContextHolder.getRequestAttributes()).getRequest();
        HttpSession session = request.getSession(false);
        if (session == null) {
            return null;
        }
        
        return (Long) session.getAttribute("userId");
    }

    /**
     * 获取当前登录用户名
     *
     * @return 用户名
     */
    public String getCurrentUsername() {
        HttpServletRequest request = ((ServletRequestAttributes) RequestContextHolder.getRequestAttributes()).getRequest();
        HttpSession session = request.getSession(false);
        if (session == null) {
            return null;
        }
        
        return (String) session.getAttribute("username");
    }

    /**
     * 获取当前登录用户角色
     *
     * @return 用户角色
     */
    public String getCurrentUserRole() {
        HttpServletRequest request = ((ServletRequestAttributes) RequestContextHolder.getRequestAttributes()).getRequest();
        HttpSession session = request.getSession(false);
        if (session == null) {
            return null;
        }
        
        return (String) session.getAttribute("role");
    }

    /**
     * 判断当前用户是否具有指定角色
     *
     * @param role 角色名
     * @return 是否具有该角色
     */
    public boolean hasRole(String role) {
        String currentRole = getCurrentUserRole();
        return currentRole != null && currentRole.equals(role);
    }

    /**
     * 判断当前用户是否已登录
     *
     * @return 是否已登录
     */
    public boolean isAuthenticated() {
        return getCurrentUserId() != null;
    }
} 