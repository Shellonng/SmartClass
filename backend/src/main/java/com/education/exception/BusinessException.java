package com.education.exception;

import com.education.exception.ResultCode;
import lombok.Getter;

/**
 * 业务异常类
 * 
 * @author Education Platform Team
 * @version 1.0.0
 */
@Getter
public class BusinessException extends RuntimeException {

    private static final long serialVersionUID = 1L;

    /**
     * 错误码
     */
    private final Integer code;

    /**
     * 错误消息
     */
    private final String message;

    /**
     * 详细错误信息
     */
    private final String detail;

    /**
     * 构造函数
     * 
     * @param code 错误码
     * @param message 错误消息
     */
    public BusinessException(Integer code, String message) {
        super(message);
        this.code = code;
        this.message = message;
        this.detail = null;
    }

    /**
     * 构造函数
     * 
     * @param code 错误码
     * @param message 错误消息
     * @param detail 详细错误信息
     */
    public BusinessException(Integer code, String message, String detail) {
        super(message);
        this.code = code;
        this.message = message;
        this.detail = detail;
    }

    /**
     * 构造函数
     * 
     * @param code 错误码
     * @param message 错误消息
     * @param cause 原因
     */
    public BusinessException(Integer code, String message, Throwable cause) {
        super(message, cause);
        this.code = code;
        this.message = message;
        this.detail = cause != null ? cause.getMessage() : null;
    }

    /**
     * 构造函数
     * 
     * @param resultCode 结果码枚举
     */
    public BusinessException(ResultCode resultCode) {
        super(resultCode.getMessage());
        this.code = resultCode.getCode();
        this.message = resultCode.getMessage();
        this.detail = null;
    }

    /**
     * 构造函数
     * 
     * @param resultCode 结果码枚举
     * @param detail 详细错误信息
     */
    public BusinessException(ResultCode resultCode, String detail) {
        super(resultCode.getMessage());
        this.code = resultCode.getCode();
        this.message = resultCode.getMessage();
        this.detail = detail;
    }

    /**
     * 构造函数
     * 
     * @param resultCode 结果码枚举
     * @param cause 原因
     */
    public BusinessException(ResultCode resultCode, Throwable cause) {
        super(resultCode.getMessage(), cause);
        this.code = resultCode.getCode();
        this.message = resultCode.getMessage();
        this.detail = cause != null ? cause.getMessage() : null;
    }

    /**
     * 构造函数
     * 
     * @param message 错误消息
     */
    public BusinessException(String message) {
        super(message);
        this.code = ResultCode.BUSINESS_ERROR.getCode();
        this.message = message;
        this.detail = null;
    }

    /**
     * 构造函数
     * 
     * @param message 错误消息
     * @param cause 原因
     */
    public BusinessException(String message, Throwable cause) {
        super(message, cause);
        this.code = ResultCode.BUSINESS_ERROR.getCode();
        this.message = message;
        this.detail = cause != null ? cause.getMessage() : null;
    }

    // 静态工厂方法

    /**
     * 创建业务异常
     * 
     * @param message 错误消息
     * @return 业务异常
     */
    public static BusinessException of(String message) {
        return new BusinessException(message);
    }

    /**
     * 创建业务异常
     * 
     * @param code 错误码
     * @param message 错误消息
     * @return 业务异常
     */
    public static BusinessException of(Integer code, String message) {
        return new BusinessException(code, message);
    }

    /**
     * 创建业务异常
     * 
     * @param resultCode 结果码枚举
     * @return 业务异常
     */
    public static BusinessException of(ResultCode resultCode) {
        return new BusinessException(resultCode);
    }

    /**
     * 创建业务异常
     * 
     * @param resultCode 结果码枚举
     * @param detail 详细错误信息
     * @return 业务异常
     */
    public static BusinessException of(ResultCode resultCode, String detail) {
        return new BusinessException(resultCode, detail);
    }

    // 常用业务异常静态方法

    /**
     * 参数错误异常
     * 
     * @param message 错误消息
     * @return 业务异常
     */
    public static BusinessException paramError(String message) {
        return new BusinessException(ResultCode.PARAM_ERROR, message);
    }

    /**
     * 数据不存在异常
     * 
     * @param message 错误消息
     * @return 业务异常
     */
    public static BusinessException dataNotFound(String message) {
        return new BusinessException(ResultCode.DATA_NOT_FOUND, message);
    }

    /**
     * 数据已存在异常
     * 
     * @param message 错误消息
     * @return 业务异常
     */
    public static BusinessException dataExists(String message) {
        return new BusinessException(ResultCode.DATA_ALREADY_EXISTS, message);
    }

    /**
     * 权限不足异常
     * 
     * @return 业务异常
     */
    public static BusinessException forbidden() {
        return new BusinessException(ResultCode.FORBIDDEN);
    }

    /**
     * 权限不足异常
     * 
     * @param message 错误消息
     * @return 业务异常
     */
    public static BusinessException forbidden(String message) {
        return new BusinessException(ResultCode.FORBIDDEN, message);
    }

    /**
     * 未授权异常
     * 
     * @return 业务异常
     */
    public static BusinessException unauthorized() {
        return new BusinessException(ResultCode.UNAUTHORIZED);
    }

    /**
     * 未授权异常
     * 
     * @param message 错误消息
     * @return 业务异常
     */
    public static BusinessException unauthorized(String message) {
        return new BusinessException(ResultCode.UNAUTHORIZED, message);
    }

    /**
     * 用户不存在异常
     * 
     * @return 业务异常
     */
    public static BusinessException userNotFound() {
        return new BusinessException(ResultCode.USER_NOT_FOUND);
    }

    /**
     * 用户已存在异常
     * 
     * @return 业务异常
     */
    public static BusinessException userExists() {
        return new BusinessException(ResultCode.USER_ALREADY_EXISTS);
    }

    /**
     * 登录失败异常
     * 
     * @return 业务异常
     */
    public static BusinessException loginFailed() {
        return new BusinessException(ResultCode.USER_PASSWORD_ERROR);
    }

    /**
     * 账户被禁用异常
     * 
     * @return 业务异常
     */
    public static BusinessException accountDisabled() {
        return new BusinessException(ResultCode.USER_DISABLED);
    }

    /**
     * 账户被锁定异常
     * 
     * @return 业务异常
     */
    public static BusinessException accountLocked() {
        return new BusinessException(ResultCode.USER_DISABLED);
    }

    /**
     * 密码错误异常
     * 
     * @return 业务异常
     */
    public static BusinessException passwordError() {
        return new BusinessException(ResultCode.USER_PASSWORD_ERROR);
    }

    /**
     * 班级不存在异常
     * 
     * @return 业务异常
     */
    public static BusinessException classNotFound() {
        return new BusinessException(ResultCode.DATA_NOT_FOUND);
    }

    /**
     * 课程不存在异常
     * 
     * @return 业务异常
     */
    public static BusinessException courseNotFound() {
        return new BusinessException(ResultCode.DATA_NOT_FOUND);
    }

    /**
     * 任务不存在异常
     * 
     * @return 业务异常
     */
    public static BusinessException taskNotFound() {
        return new BusinessException(ResultCode.DATA_NOT_FOUND);
    }

    /**
     * 文件不存在异常
     * 
     * @return 业务异常
     */
    public static BusinessException fileNotFound() {
        return new BusinessException(ResultCode.FILE_NOT_FOUND);
    }

    /**
     * 文件上传失败异常
     * 
     * @return 业务异常
     */
    public static BusinessException fileUploadFailed() {
        return new BusinessException(ResultCode.FILE_UPLOAD_ERROR);
    }

    /**
     * 文件大小超限异常
     * 
     * @return 业务异常
     */
    public static BusinessException fileSizeExceeded() {
        return new BusinessException(ResultCode.FILE_SIZE_ERROR);
    }

    /**
     * AI服务异常
     * 
     * @return 业务异常
     */
    public static BusinessException aiServiceError() {
        return new BusinessException(ResultCode.AI_SERVICE_ERROR);
    }

    /**
     * AI服务异常
     * 
     * @param message 错误消息
     * @return 业务异常
     */
    public static BusinessException aiServiceError(String message) {
        return new BusinessException(ResultCode.AI_SERVICE_ERROR, message);
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("BusinessException{")
          .append("code=").append(code)
          .append(", message='").append(message).append('\'')
          .append(", detail='").append(detail).append('\'')
          .append('}');
        return sb.toString();
    }
}