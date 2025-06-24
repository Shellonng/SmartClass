package com.education.exception;

import com.education.dto.common.Result;
import com.education.exception.ResultCode;
import lombok.extern.slf4j.Slf4j;
import org.springframework.dao.DataIntegrityViolationException;
import org.springframework.dao.DuplicateKeyException;
import org.springframework.http.HttpStatus;
import org.springframework.http.converter.HttpMessageNotReadableException;
import org.springframework.security.access.AccessDeniedException;
import org.springframework.security.authentication.BadCredentialsException;
import org.springframework.security.authentication.DisabledException;
import org.springframework.security.authentication.LockedException;
import org.springframework.security.core.AuthenticationException;
import org.springframework.validation.BindException;
import org.springframework.validation.FieldError;
import org.springframework.web.HttpMediaTypeNotSupportedException;
import org.springframework.web.HttpRequestMethodNotSupportedException;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.MissingServletRequestParameterException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.ResponseStatus;
import org.springframework.web.bind.annotation.RestControllerAdvice;
import org.springframework.web.method.annotation.MethodArgumentTypeMismatchException;
import org.springframework.web.multipart.MaxUploadSizeExceededException;
import org.springframework.web.servlet.NoHandlerFoundException;

import jakarta.servlet.http.HttpServletRequest;
import jakarta.validation.ConstraintViolation;
import jakarta.validation.ConstraintViolationException;
// 使用完全限定名避免与Spring Security的AccessDeniedException冲突
import java.sql.SQLException;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * 全局异常处理器
 * 
 * @author Education Platform Team
 * @version 1.0.0
 */
@Slf4j
@RestControllerAdvice
public class GlobalExceptionHandler {

    /**
     * 处理业务异常
     */
    @ExceptionHandler(BusinessException.class)
    @ResponseStatus(HttpStatus.OK)
    public Result<Void> handleBusinessException(BusinessException e, HttpServletRequest request) {
        log.warn("业务异常: {} - {}", e.getMessage(), request.getRequestURI());
        return Result.error(e.getCode(), e.getMessage());
    }

    /**
     * 处理参数校验异常 - @Valid
     */
    @ExceptionHandler(MethodArgumentNotValidException.class)
    @ResponseStatus(HttpStatus.BAD_REQUEST)
    public Result<Void> handleMethodArgumentNotValidException(MethodArgumentNotValidException e) {
        log.warn("参数校验异常: {}", e.getMessage());
        
        String errorMessage = e.getBindingResult().getFieldErrors().stream()
            .map(FieldError::getDefaultMessage)
            .collect(Collectors.joining("; "));
        
        return Result.error(ResultCode.PARAM_ERROR.getCode(), errorMessage);
    }

    /**
     * 处理参数绑定异常 - @ModelAttribute
     */
    @ExceptionHandler(BindException.class)
    @ResponseStatus(HttpStatus.BAD_REQUEST)
    public Result<Void> handleBindException(BindException e) {
        log.warn("参数绑定异常: {}", e.getMessage());
        
        String errorMessage = e.getFieldErrors().stream()
            .map(FieldError::getDefaultMessage)
            .collect(Collectors.joining("; "));
        
        return Result.error(ResultCode.PARAM_ERROR.getCode(), errorMessage);
    }

    /**
     * 处理约束校验异常 - @Validated
     */
    @ExceptionHandler(ConstraintViolationException.class)
    @ResponseStatus(HttpStatus.BAD_REQUEST)
    public Result<Void> handleConstraintViolationException(ConstraintViolationException e) {
        log.warn("约束校验异常: {}", e.getMessage());
        
        Set<ConstraintViolation<?>> violations = e.getConstraintViolations();
        String errorMessage = violations.stream()
            .map(ConstraintViolation::getMessage)
            .collect(Collectors.joining("; "));
        
        return Result.error(ResultCode.PARAM_ERROR.getCode(), errorMessage);
    }

    /**
     * 处理缺少请求参数异常
     */
    @ExceptionHandler(MissingServletRequestParameterException.class)
    @ResponseStatus(HttpStatus.BAD_REQUEST)
    public Result<Void> handleMissingServletRequestParameterException(MissingServletRequestParameterException e) {
        log.warn("缺少请求参数: {}", e.getMessage());
        return Result.error(ResultCode.PARAM_ERROR.getCode(), "缺少必需的请求参数: " + e.getParameterName());
    }

    /**
     * 处理参数类型不匹配异常
     */
    @ExceptionHandler(MethodArgumentTypeMismatchException.class)
    @ResponseStatus(HttpStatus.BAD_REQUEST)
    public Result<Void> handleMethodArgumentTypeMismatchException(MethodArgumentTypeMismatchException e) {
        log.warn("参数类型不匹配: {}", e.getMessage());
        return Result.error(ResultCode.PARAM_ERROR.getCode(), 
            String.format("参数 '%s' 类型不正确，期望类型: %s", e.getName(), e.getRequiredType().getSimpleName()));
    }

    /**
     * 处理HTTP消息不可读异常
     */
    @ExceptionHandler(HttpMessageNotReadableException.class)
    @ResponseStatus(HttpStatus.BAD_REQUEST)
    public Result<Void> handleHttpMessageNotReadableException(HttpMessageNotReadableException e) {
        log.warn("HTTP消息不可读: {}", e.getMessage());
        return Result.error(ResultCode.PARAM_ERROR.getCode(), "请求体格式错误或无法解析");
    }

    /**
     * 处理不支持的HTTP方法异常
     */
    @ExceptionHandler(HttpRequestMethodNotSupportedException.class)
    @ResponseStatus(HttpStatus.METHOD_NOT_ALLOWED)
    public Result<Void> handleHttpRequestMethodNotSupportedException(HttpRequestMethodNotSupportedException e) {
        log.warn("不支持的HTTP方法: {}", e.getMessage());
        return Result.error(ResultCode.METHOD_NOT_ALLOWED.getCode(), 
            String.format("不支持的请求方法: %s，支持的方法: %s", e.getMethod(), String.join(", ", e.getSupportedMethods())));
    }

    /**
     * 处理不支持的媒体类型异常
     */
    @ExceptionHandler(HttpMediaTypeNotSupportedException.class)
    @ResponseStatus(HttpStatus.UNSUPPORTED_MEDIA_TYPE)
    public Result<Void> handleHttpMediaTypeNotSupportedException(HttpMediaTypeNotSupportedException e) {
        log.warn("不支持的媒体类型: {}", e.getMessage());
        return Result.error(ResultCode.PARAM_ERROR.getCode(), 
            String.format("不支持的媒体类型: %s", e.getContentType()));
    }

    /**
     * 处理文件上传大小超限异常
     */
    @ExceptionHandler(MaxUploadSizeExceededException.class)
    @ResponseStatus(HttpStatus.BAD_REQUEST)
    public Result<Void> handleMaxUploadSizeExceededException(MaxUploadSizeExceededException e) {
        log.warn("文件上传大小超限: {}", e.getMessage());
        return Result.error(ResultCode.FILE_SIZE_ERROR.getCode(), "上传文件大小超出限制");
    }

    /**
     * 处理404异常
     */
    @ExceptionHandler(NoHandlerFoundException.class)
    @ResponseStatus(HttpStatus.NOT_FOUND)
    public Result<Void> handleNoHandlerFoundException(NoHandlerFoundException e) {
        log.warn("404异常: {}", e.getMessage());
        return Result.error(ResultCode.NOT_FOUND.getCode(), "请求的资源不存在");
    }

    /**
     * 处理认证异常
     */
    @ExceptionHandler(AuthenticationException.class)
    @ResponseStatus(HttpStatus.UNAUTHORIZED)
    public Result<Void> handleAuthenticationException(AuthenticationException e) {
        log.warn("认证异常: {}", e.getMessage());
        
        if (e instanceof BadCredentialsException) {
            return Result.error(ResultCode.USER_PASSWORD_ERROR.getCode(), "用户名或密码错误");
        } else if (e instanceof DisabledException) {
            return Result.error(ResultCode.USER_DISABLED.getCode(), "账户已被禁用");
        } else if (e instanceof LockedException) {
            return Result.error(ResultCode.USER_DISABLED.getCode(), "账户已被锁定");
        }
        
        return Result.error(ResultCode.UNAUTHORIZED.getCode(), "认证失败");
    }

    /**
     * 处理权限不足异常
     */
    @ExceptionHandler(AccessDeniedException.class)
    @ResponseStatus(HttpStatus.FORBIDDEN)
    public Result<Void> handleAccessDeniedException(AccessDeniedException e) {
        log.warn("权限不足: {}", e.getMessage());
        return Result.error(ResultCode.FORBIDDEN.getCode(), "权限不足，拒绝访问");
    }

    /**
     * 处理文件访问拒绝异常
     */
    @ExceptionHandler(java.nio.file.AccessDeniedException.class)
    @ResponseStatus(HttpStatus.FORBIDDEN)
    public Result<Void> handleFileAccessDeniedException(java.nio.file.AccessDeniedException e) {
        log.warn("文件访问拒绝: {}", e.getMessage());
        return Result.error(ResultCode.FORBIDDEN.getCode(), "文件访问被拒绝");
    }

    /**
     * 处理数据库重复键异常
     */
    @ExceptionHandler(DuplicateKeyException.class)
    @ResponseStatus(HttpStatus.CONFLICT)
    public Result<Void> handleDuplicateKeyException(DuplicateKeyException e) {
        log.warn("数据库重复键异常: {}", e.getMessage());
        return Result.error(ResultCode.DATA_ALREADY_EXISTS.getCode(), "数据已存在，请检查唯一性约束");
    }

    /**
     * 处理数据完整性违反异常
     */
    @ExceptionHandler(DataIntegrityViolationException.class)
    @ResponseStatus(HttpStatus.CONFLICT)
    public Result<Void> handleDataIntegrityViolationException(DataIntegrityViolationException e) {
        log.warn("数据完整性违反: {}", e.getMessage());
        return Result.error(ResultCode.DATA_SAVE_ERROR.getCode(), "数据完整性约束违反");
    }

    /**
     * 处理SQL异常
     */
    @ExceptionHandler(SQLException.class)
    @ResponseStatus(HttpStatus.INTERNAL_SERVER_ERROR)
    public Result<Void> handleSQLException(SQLException e) {
        log.error("SQL异常: {}", e.getMessage(), e);
        return Result.error(ResultCode.DATA_SAVE_ERROR.getCode(), "数据库操作异常");
    }

    /**
     * 处理空指针异常
     */
    @ExceptionHandler(NullPointerException.class)
    @ResponseStatus(HttpStatus.INTERNAL_SERVER_ERROR)
    public Result<Void> handleNullPointerException(NullPointerException e, HttpServletRequest request) {
        log.error("空指针异常: {} - {}", e.getMessage(), request.getRequestURI(), e);
        return Result.error(ResultCode.INTERNAL_SERVER_ERROR.getCode(), "系统内部错误");
    }

    /**
     * 处理非法参数异常
     */
    @ExceptionHandler(IllegalArgumentException.class)
    @ResponseStatus(HttpStatus.BAD_REQUEST)
    public Result<Void> handleIllegalArgumentException(IllegalArgumentException e) {
        log.warn("非法参数异常: {}", e.getMessage());
        return Result.error(ResultCode.PARAM_ERROR.getCode(), e.getMessage());
    }

    /**
     * 处理非法状态异常
     */
    @ExceptionHandler(IllegalStateException.class)
    @ResponseStatus(HttpStatus.CONFLICT)
    public Result<Void> handleIllegalStateException(IllegalStateException e) {
        log.warn("非法状态异常: {}", e.getMessage());
        return Result.error(ResultCode.BUSINESS_ERROR.getCode(), e.getMessage());
    }

    /**
     * 处理运行时异常
     */
    @ExceptionHandler(RuntimeException.class)
    @ResponseStatus(HttpStatus.INTERNAL_SERVER_ERROR)
    public Result<Void> handleRuntimeException(RuntimeException e, HttpServletRequest request) {
        log.error("运行时异常: {} - {}", e.getMessage(), request.getRequestURI(), e);
        return Result.error(ResultCode.INTERNAL_SERVER_ERROR.getCode(), "系统运行时异常");
    }

    /**
     * 处理其他未知异常
     */
    @ExceptionHandler(Exception.class)
    @ResponseStatus(HttpStatus.INTERNAL_SERVER_ERROR)
    public Result<Void> handleException(Exception e, HttpServletRequest request) {
        log.error("未知异常: {} - {}", e.getMessage(), request.getRequestURI(), e);
        return Result.error(ResultCode.INTERNAL_SERVER_ERROR.getCode(), "系统异常，请联系管理员");
    }
}