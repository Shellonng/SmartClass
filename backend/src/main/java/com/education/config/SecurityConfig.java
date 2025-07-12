package com.education.config;

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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.education.mapper.UserMapper;

import org.springframework.beans.factory.annotation.Autowired;

import java.util.Arrays;

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
                .requestMatchers("/swagger-ui/**", "/v3/api-docs/**").permitAll()
                .requestMatchers("/debug/**").permitAll()
                .requestMatchers("/api/courses/public", "/api/courses/public/**").permitAll()
                .requestMatchers("/api/courses/categories").permitAll()
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
} 