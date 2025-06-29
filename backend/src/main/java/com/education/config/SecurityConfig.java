package com.education.config;

import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import jakarta.servlet.http.HttpSession;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.config.annotation.web.configurers.AbstractHttpConfigurer;
import org.springframework.security.config.http.SessionCreationPolicy;
import org.springframework.security.core.userdetails.User;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.security.web.SecurityFilterChain;
import org.springframework.security.web.authentication.UsernamePasswordAuthenticationFilter;
import org.springframework.web.cors.CorsConfiguration;
import org.springframework.web.cors.CorsConfigurationSource;
import org.springframework.web.cors.UrlBasedCorsConfigurationSource;
import org.springframework.web.filter.OncePerRequestFilter;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.web.authentication.WebAuthenticationDetailsSource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.education.mapper.UserMapper;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

@Configuration
@EnableWebSecurity
public class SecurityConfig {

    private static final Logger logger = LoggerFactory.getLogger(SecurityConfig.class);
    
    @Autowired
    private UserMapper userMapper;
    
    @Autowired
    private SessionAuthenticationFilter sessionAuthenticationFilter;

    @Bean
    public SecurityFilterChain filterChain(HttpSecurity http) throws Exception {
        logger.info("配置Spring Security过滤链 - 基于Session认证");
        
        http
            .csrf(AbstractHttpConfigurer::disable)
            .cors(cors -> cors.configurationSource(corsConfigurationSource()))
            .sessionManagement(session -> {
                session.sessionCreationPolicy(SessionCreationPolicy.IF_REQUIRED);
                session.maximumSessions(5); // 允许同一用户有多个活动会话
                session.sessionFixation().migrateSession(); // 登录后迁移会话
            })
            .authorizeHttpRequests(auth -> auth
                .requestMatchers("/auth/**", "/api/auth/**").permitAll()
                .requestMatchers("/api/common/files/**").permitAll()
                .requestMatchers("/files/**").permitAll()
                .requestMatchers("/resource/video/**").permitAll()
                .requestMatchers("/swagger-ui/**", "/v3/api-docs/**").permitAll()
                .requestMatchers("/debug/**").permitAll()
                .anyRequest().authenticated()
            )
            .addFilterBefore(sessionAuthenticationFilter, UsernamePasswordAuthenticationFilter.class);
        
        logger.info("Spring Security配置完成 - 使用Session认证");
        return http.build();
    }

    @Bean
    public CorsConfigurationSource corsConfigurationSource() {
        logger.info("配置CORS");
        
        CorsConfiguration configuration = new CorsConfiguration();
        configuration.setAllowedOriginPatterns(Arrays.asList("*"));
        configuration.setAllowedMethods(Arrays.asList("GET", "POST", "PUT", "DELETE", "OPTIONS"));
        configuration.setAllowedHeaders(Arrays.asList("*"));
        configuration.setAllowCredentials(true);
        configuration.setMaxAge(3600L);
        
        configuration.setExposedHeaders(Arrays.asList("Set-Cookie"));

        UrlBasedCorsConfigurationSource source = new UrlBasedCorsConfigurationSource();
        source.registerCorsConfiguration("/**", configuration);
        
        logger.info("CORS配置完成");
        return source;
    }

    @Bean
    public PasswordEncoder passwordEncoder() {
        return new BCryptPasswordEncoder();
    }
    
    @Bean
    public UserDetailsService userDetailsService() {
        logger.info("配置UserDetailsService - 基于Session认证");
        
        return new UserDetailsService() {
            @Override
            public UserDetails loadUserByUsername(String username) throws UsernameNotFoundException {
                logger.info("尝试加载用户: {}", username);
                
                try {
                    com.education.entity.User user = userMapper.selectByUsernameWithPassword(username);
                    if (user != null) {
                        logger.info("找到用户: {}, 角色: {}", user.getUsername(), user.getRole());
                        return User.withUsername(user.getUsername())
                                .password(user.getPassword() != null ? user.getPassword() : passwordEncoder().encode("defaultpwd"))
                                .roles(user.getRole())
                                .build();
                    }
                    
                    if ("admin".equals(username)) {
                        logger.info("创建默认管理员用户: admin");
                        return User.withUsername("admin")
                                .password(passwordEncoder().encode("admin123"))
                                .roles("ADMIN")
                                .build();
                    }
                    
                    logger.error("用户不存在: {}", username);
                    throw new UsernameNotFoundException("用户不存在: " + username);
                } catch (Exception e) {
                    logger.error("加载用户时发生错误: {}", e.getMessage(), e);
                    throw new UsernameNotFoundException("加载用户时发生错误", e);
                }
            }
        };
    }

    /**
     * 自定义Session认证过滤器
     */
    @Component
    public static class SessionAuthenticationFilter extends OncePerRequestFilter {

        private static final Logger logger = LoggerFactory.getLogger(SessionAuthenticationFilter.class);

        // 不需要认证的路径
        private static final List<String> AUTH_WHITELIST = Arrays.asList(
            "/auth/login", "/auth/register", "/auth/logout",
            "/api/auth/login", "/api/auth/register", "/api/auth/logout",
            "/swagger-ui", "/v3/api-docs", "/debug", "/files", "/api/common/files",
            "/resource/video"
        );

        @Override
        protected void doFilterInternal(HttpServletRequest request, HttpServletResponse response, 
                FilterChain filterChain) throws ServletException, IOException {
            
            String requestPath = request.getServletPath();
            
            // 检查是否是白名单路径
            boolean isWhitelisted = AUTH_WHITELIST.stream()
                    .anyMatch(path -> requestPath.startsWith(path));
            
            if (!isWhitelisted) {
                // 检查Session中是否有用户信息
                HttpSession session = request.getSession(false);
                if (session != null) {
                    Long userId = (Long) session.getAttribute("userId");
                    String username = (String) session.getAttribute("username");
                    String role = (String) session.getAttribute("role");
                    
                    if (userId != null && username != null && role != null) {
                        logger.debug("从Session中找到用户信息: username={}, role={}", username, role);
                        
                        // 创建认证对象
                        UsernamePasswordAuthenticationToken authentication = 
                            new UsernamePasswordAuthenticationToken(username, null, 
                                Arrays.asList(() -> "ROLE_" + role.toUpperCase()));
                        authentication.setDetails(new WebAuthenticationDetailsSource().buildDetails(request));
                        
                        // 设置认证信息到SecurityContext
                        SecurityContextHolder.getContext().setAuthentication(authentication);
                        logger.debug("用户 {} 已通过Session认证", username);
                    } else {
                        logger.debug("Session中没有完整的用户信息，路径: {}", requestPath);
                    }
                } else {
                    logger.debug("没有找到Session，路径: {}", requestPath);
                }
            }
            
            filterChain.doFilter(request, response);
        }
    }
} 