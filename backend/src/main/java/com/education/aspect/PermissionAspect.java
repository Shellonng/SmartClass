package com.education.aspect;

import com.education.exception.BusinessException;
import com.education.exception.ResultCode;
import com.education.utils.JwtUtils;
import lombok.extern.slf4j.Slf4j;
import org.aspectj.lang.ProceedingJoinPoint;
import org.aspectj.lang.annotation.Around;
import org.aspectj.lang.annotation.Aspect;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;
import org.springframework.web.context.request.RequestContextHolder;
import org.springframework.web.context.request.ServletRequestAttributes;

import jakarta.servlet.http.HttpServletRequest;
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;
import java.util.Arrays;
import java.util.List;

/**
 * 权限切面
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
@Aspect
@Component
@Slf4j
public class PermissionAspect {

    @Autowired
    private JwtUtils jwtUtils;

    /**
     * 权限检查处理
     */
    @Around("@annotation(requirePermission)")
    public Object handlePermissionCheck(ProceedingJoinPoint joinPoint, RequirePermission requirePermission) throws Throwable {
        // 获取当前请求
        ServletRequestAttributes attributes = (ServletRequestAttributes) RequestContextHolder.getRequestAttributes();
        if (attributes == null) {
            throw new BusinessException(ResultCode.UNAUTHORIZED.getCode(), "无法获取请求上下文");
        }
        
        HttpServletRequest request = attributes.getRequest();
        
        // 获取token
        String token = getTokenFromRequest(request);
        if (token == null) {
            throw new BusinessException(ResultCode.UNAUTHORIZED.getCode(), "未提供访问令牌");
        }
        
        // 验证token
        if (isTokenExpired(token)) {
            throw new BusinessException(ResultCode.UNAUTHORIZED.getCode(), "访问令牌无效或已过期");
        }
        
        // 获取用户信息
        String userId = getUserIdFromToken(token);
        String userRole = getUserRoleFromToken(token);
        List<String> userPermissions = getUserPermissionsFromToken(token);
        
        // 检查角色权限
        if (requirePermission.roles().length > 0) {
            List<String> requiredRoles = Arrays.asList(requirePermission.roles());
            if (!requiredRoles.contains(userRole)) {
                log.warn("用户 {} 角色 {} 尝试访问需要角色 {} 的资源", userId, userRole, requiredRoles);
                throw new BusinessException(ResultCode.FORBIDDEN.getCode(), "权限不足，需要角色: " + String.join(", ", requiredRoles));
            }
        }
        
        // 检查具体权限
        if (requirePermission.permissions().length > 0) {
            List<String> requiredPermissions = Arrays.asList(requirePermission.permissions());
            boolean hasPermission = requiredPermissions.stream()
                    .anyMatch(userPermissions::contains);
            
            if (!hasPermission) {
                log.warn("用户 {} 尝试访问需要权限 {} 的资源，当前权限: {}", userId, requiredPermissions, userPermissions);
                throw new BusinessException(ResultCode.FORBIDDEN.getCode(), "权限不足，需要权限: " + String.join(", ", requiredPermissions));
            }
        }
        
        // 检查资源所有权（如果需要）
        if (requirePermission.checkOwnership()) {
            boolean isOwner = checkResourceOwnership(joinPoint, userId, requirePermission);
            if (!isOwner && !isAdmin(userRole)) {
                log.warn("用户 {} 尝试访问不属于自己的资源", userId);
                throw new BusinessException(ResultCode.FORBIDDEN.getCode(), "只能访问自己的资源");
            }
        }
        
        log.debug("用户 {} 权限检查通过，角色: {}, 权限: {}", userId, userRole, userPermissions);
        
        // 权限检查通过，执行方法
        return joinPoint.proceed();
    }

    /**
     * 从请求中获取token
     */
    private String getTokenFromRequest(HttpServletRequest request) {
        String bearerToken = request.getHeader("Authorization");
        if (bearerToken != null && bearerToken.startsWith("Bearer ")) {
            return bearerToken.substring(7);
        }
        return null;
    }

    /**
     * 检查token是否过期
     */
    private boolean isTokenExpired(String token) {
        try {
            return jwtUtils.getExpirationDateFromToken(token).before(new java.util.Date());
        } catch (Exception e) {
            return true;
        }
    }

    /**
     * 从token中获取用户ID
     */
    private String getUserIdFromToken(String token) {
        try {
            return jwtUtils.getClaimFromToken(token, claims -> claims.get("userId", String.class));
        } catch (Exception e) {
            return null;
        }
    }

    /**
     * 从token中获取用户角色
     */
    private String getUserRoleFromToken(String token) {
        try {
            return jwtUtils.getClaimFromToken(token, claims -> claims.get("role", String.class));
        } catch (Exception e) {
            return null;
        }
    }

    /**
     * 从token中获取用户权限列表
     */
    @SuppressWarnings("unchecked")
    private List<String> getUserPermissionsFromToken(String token) {
        try {
            return jwtUtils.getClaimFromToken(token, claims -> {
                Object permissions = claims.get("permissions");
                if (permissions instanceof List) {
                    return (List<String>) permissions;
                }
                return Arrays.asList();
            });
        } catch (Exception e) {
            return Arrays.asList();
        }
    }

    /**
     * 检查资源所有权
     */
    private boolean checkResourceOwnership(ProceedingJoinPoint joinPoint, String userId, RequirePermission requirePermission) {
        // 获取方法参数
        Object[] args = joinPoint.getArgs();
        
        // 根据资源类型检查所有权
        String resourceType = requirePermission.resourceType();
        String resourceIdParam = requirePermission.resourceIdParam();
        
        // 这里可以根据具体业务逻辑实现资源所有权检查
        // 例如：检查课程是否属于该教师，任务是否属于该学生等
        
        // 简化实现：如果参数中包含当前用户ID，则认为有权限
        for (Object arg : args) {
            if (arg != null && arg.toString().equals(userId)) {
                return true;
            }
        }
        
        return false;
    }

    /**
     * 检查是否为管理员
     */
    private boolean isAdmin(String userRole) {
        return "ADMIN".equals(userRole) || "SUPER_ADMIN".equals(userRole);
    }
}

/**
 * 权限要求注解
 */
@Target(ElementType.METHOD)
@Retention(RetentionPolicy.RUNTIME)
@interface RequirePermission {
    /**
     * 需要的角色
     */
    String[] roles() default {};
    
    /**
     * 需要的权限
     */
    String[] permissions() default {};
    
    /**
     * 是否检查资源所有权
     */
    boolean checkOwnership() default false;
    
    /**
     * 资源类型
     */
    String resourceType() default "";
    
    /**
     * 资源ID参数名
     */
    String resourceIdParam() default "id";
}