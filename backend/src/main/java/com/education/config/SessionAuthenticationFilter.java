package com.education.config;

import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import jakarta.servlet.http.HttpSession;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.web.authentication.WebAuthenticationDetailsSource;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

import com.education.mapper.UserMapper;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

/**
 * 自定义Session认证过滤器
 */
@Component
public class SessionAuthenticationFilter extends OncePerRequestFilter {

    private static final Logger logger = LoggerFactory.getLogger(SessionAuthenticationFilter.class);
    
    @Autowired
    private UserMapper userMapper;

    // 不需要认证的路径
    private static final List<String> AUTH_WHITELIST = Arrays.asList(
        "/auth/login", "/auth/register", "/auth/logout",
        "/api/auth/login", "/api/auth/register", "/api/auth/logout",
        "/swagger-ui", "/v3/api-docs", "/debug", "/files", "/api/common/files",
        "/api/courses/public", "/api/courses/categories"
    );

    @Override
    protected void doFilterInternal(HttpServletRequest request, HttpServletResponse response, 
            FilterChain filterChain) throws ServletException, IOException {
        
        String requestPath = request.getServletPath();
        logger.debug("处理请求: {} {}", request.getMethod(), requestPath);
        logger.debug("请求Cookie: {}", Arrays.toString(request.getCookies()));
        
        // 检查是否是白名单路径
        boolean isWhitelisted = AUTH_WHITELIST.stream()
                .anyMatch(path -> requestPath.startsWith(path));
        
        if (isWhitelisted) {
            logger.debug("白名单路径: {}, 跳过认证", requestPath);
            filterChain.doFilter(request, response);
            return;
        }
        
        // 首先尝试从请求头中获取Bearer Token
        String authHeader = request.getHeader("Authorization");
        logger.info("请求头中的Authorization: {}", authHeader);
        
        if (authHeader != null && authHeader.startsWith("Bearer ")) {
            String token = authHeader.substring(7);
            logger.info("从请求头中找到Bearer Token: {}", token);
            
            // 处理简化的token格式: token-{userId} 或 token-{userId}-{username}
            if (token.startsWith("token-")) {
                try {
                    logger.info("解析token: {}", token);
                    
                    // 检查是否是新格式: token-{userId}-{username}
                    if (token.indexOf("-", 6) > -1) {
                        // 新格式: token-{userId}-{username}
                        String[] parts = token.substring(6).split("-", 2);
                        String userId = parts[0];
                        String username = parts[1];
                        
                        logger.info("使用新格式token，用户ID: {}, 用户名: {}", userId, username);
                    
                    // 从数据库中查询用户信息
                    com.education.entity.User user = userMapper.selectById(Long.parseLong(userId));
                    if (user != null) {
                            logger.info("找到用户: {}, ID: {}, 角色: {}", user.getUsername(), user.getId(), user.getRole());
                            
                            // 验证用户名是否匹配
                            if (!user.getUsername().equals(username)) {
                                logger.warn("Token中的用户名({})与数据库中的用户名({})不匹配", username, user.getUsername());
                            }
                        
                        // 创建认证对象
                        UsernamePasswordAuthenticationToken authentication = 
                            new UsernamePasswordAuthenticationToken(user.getUsername(), null, 
                                Arrays.asList(() -> "ROLE_" + user.getRole().toUpperCase()));
                        authentication.setDetails(new WebAuthenticationDetailsSource().buildDetails(request));
                        
                        // 设置认证信息到SecurityContext
                        SecurityContextHolder.getContext().setAuthentication(authentication);
                        
                        // 同时将用户信息存入Session，以便SecurityUtil可以获取
                        HttpSession session = request.getSession(true);
                        session.setAttribute("userId", user.getId());
                        session.setAttribute("username", user.getUsername());
                        session.setAttribute("role", user.getRole());
                        
                            logger.info("用户 {} (ID: {}) 已通过Token认证，并设置到Session", user.getUsername(), user.getId());
                        } else {
                            logger.error("未找到用户ID: {}", userId);
                        }
                    } else {
                        // 旧格式: token-{userId}
                        String userId = token.substring(6); // 提取userId部分
                        logger.info("使用旧格式token，用户ID: {}", userId);
                        
                        // 从数据库中查询用户信息
                        com.education.entity.User user = userMapper.selectById(Long.parseLong(userId));
                        if (user != null) {
                            logger.info("找到用户: {}, ID: {}, 角色: {}", user.getUsername(), user.getId(), user.getRole());
                            
                            // 创建认证对象
                            UsernamePasswordAuthenticationToken authentication = 
                                new UsernamePasswordAuthenticationToken(user.getUsername(), null, 
                                    Arrays.asList(() -> "ROLE_" + user.getRole().toUpperCase()));
                            authentication.setDetails(new WebAuthenticationDetailsSource().buildDetails(request));
                            
                            // 设置认证信息到SecurityContext
                            SecurityContextHolder.getContext().setAuthentication(authentication);
                            
                            // 同时将用户信息存入Session，以便SecurityUtil可以获取
                            HttpSession session = request.getSession(true);
                            session.setAttribute("userId", user.getId());
                            session.setAttribute("username", user.getUsername());
                            session.setAttribute("role", user.getRole());
                            
                            logger.info("用户 {} (ID: {}) 已通过Token认证，并设置到Session", user.getUsername(), user.getId());
                    } else {
                        logger.error("未找到用户ID: {}", userId);
                        }
                    }
                } catch (Exception e) {
                    logger.error("处理Token时发生错误: {}", e.getMessage(), e);
                }
            } else {
                // 处理标准JWT Token（此处仅为示例，实际应该进行JWT验证）
                logger.info("标准Token认证暂未实现");
            }
        } else {
            // 如果没有Bearer Token，则尝试Session认证
            HttpSession session = request.getSession(false);
            if (session != null) {
                Long userId = (Long) session.getAttribute("userId");
                String username = (String) session.getAttribute("username");
                String role = (String) session.getAttribute("role");
                
                logger.info("Session ID: {}", session.getId());
                logger.info("Session 属性: userId={}, username={}, role={}", userId, username, role);
                
                if (userId != null && username != null && role != null) {
                    logger.info("从Session中找到用户信息: username={}, userId={}, role={}", username, userId, role);
                    
                    // 创建认证对象
                    UsernamePasswordAuthenticationToken authentication = 
                        new UsernamePasswordAuthenticationToken(username, null, 
                            Arrays.asList(() -> "ROLE_" + role.toUpperCase()));
                    authentication.setDetails(new WebAuthenticationDetailsSource().buildDetails(request));
                    
                    // 设置认证信息到SecurityContext
                    SecurityContextHolder.getContext().setAuthentication(authentication);
                    logger.info("用户 {} (ID: {}) 已通过Session认证", username, userId);
                } else {
                    logger.warn("Session中没有完整的用户信息，路径: {}", requestPath);
                }
            } else {
                logger.warn("没有找到Session，路径: {}", requestPath);
            }
        }
        
        // 记录最终的认证状态
        org.springframework.security.core.Authentication finalAuth = 
                SecurityContextHolder.getContext().getAuthentication();
        if (finalAuth != null) {
            logger.info("请求 {} 的最终认证状态: 用户={}, 已认证={}", 
                    requestPath, finalAuth.getName(), finalAuth.isAuthenticated());
        } else {
            logger.warn("请求 {} 没有认证信息", requestPath);
        }
        
        filterChain.doFilter(request, response);
    }
} 