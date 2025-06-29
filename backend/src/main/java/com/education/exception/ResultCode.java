package com.education.exception;

/**
 * 统一结果码枚举
 * 定义系统中所有可能的返回状态码和对应的消息
 */
public enum ResultCode {
    
    // 通用结果码 (1000-1999)
    SUCCESS(200, "操作成功"),
    FAIL(500, "操作失败"),
    PARAM_ERROR(400, "参数错误"),
    UNAUTHORIZED(401, "未登录或登录已过期"),
    FORBIDDEN(403, "无权限"),
    ACCESS_DENIED(403, "访问被拒绝"),
    NOT_FOUND(404, "资源不存在"),
    METHOD_NOT_ALLOWED(405, "请求方法不支持"),
    INTERNAL_SERVER_ERROR(500, "系统错误"),
    
    // 用户相关 (2000-2999)
    USER_NOT_FOUND(2001, "用户不存在"),
    USER_ALREADY_EXISTS(2002, "用户已存在"),
    USER_PASSWORD_ERROR(2003, "密码错误"),
    USER_DISABLED(2004, "用户已被禁用"),
    USER_LOGIN_EXPIRED(2005, "登录已过期"),
    USER_PERMISSION_DENIED(2006, "用户权限不足"),
    TOKEN_INVALID(2007, "Token无效"),
    EMAIL_SEND_FAILED(2008, "邮件发送失败"),
    VERIFICATION_CODE_ERROR(2009, "验证码错误"),
    
    // 数据相关 (3000-3999)
    DATA_NOT_FOUND(3001, "数据不存在"),
    DATA_ALREADY_EXISTS(3002, "数据已存在"),
    DATA_SAVE_ERROR(3003, "数据保存失败"),
    DATA_UPDATE_ERROR(3004, "数据更新失败"),
    DATA_DELETE_ERROR(3005, "数据删除失败"),
    
    // 业务相关 (4000-4999)
    BUSINESS_ERROR(501, "业务错误"),
    VALIDATION_ERROR(4002, "数据验证失败"),
    
    // 文件相关 (5000-5999)
    FILE_UPLOAD_ERROR(10001, "文件上传失败"),
    FILE_UPLOAD_FAILED(5002, "文件上传失败"),
    FILE_DOWNLOAD_ERROR(10002, "文件下载失败"),
    FILE_NOT_FOUND(5004, "文件不存在"),
    FILE_TYPE_ERROR(10003, "文件类型不支持"),
    FILE_SIZE_ERROR(10004, "文件大小超出限制"),
    FILE_SIZE_EXCEEDED(5007, "文件大小超出限制"),
    FILE_TYPE_NOT_ALLOWED(5008, "文件类型不允许"),
    NO_PERMISSION(5009, "没有权限"),
    SHARE_NOT_FOUND(5010, "分享不存在"),
    FILE_OPERATION_FAILED(5011, "文件操作失败"),
    FILE_DELETE_ERROR(10005, "文件删除失败"),
    
    // AI服务相关 (6000-6999)
    AI_SERVICE_ERROR(6001, "AI服务调用失败"),
    AI_SERVICE_TIMEOUT(6002, "AI服务超时"),
    AI_SERVICE_UNAVAILABLE(6003, "AI服务不可用"),
    
    // 任务相关 (7000-7999)
    TASK_NOT_FOUND(7001, "任务不存在"),
    TASK_ALREADY_SUBMITTED(7002, "任务已提交"),
    SUBMISSION_NOT_FOUND(7003, "提交记录不存在"),
    SUBMISSION_ALREADY_GRADED(7004, "提交记录已评分"),
    TASK_OVERDUE(7005, "任务已过期"),
    
    // 资源相关 (8000-8999)
    RESOURCE_NOT_FOUND(8001, "资源不存在");
    
    private final int code;
    private final String message;
    
    ResultCode(int code, String message) {
        this.code = code;
        this.message = message;
    }
    
    public int getCode() {
        return code;
    }
    
    public String getMessage() {
        return message;
    }
    
    /**
     * 根据状态码获取对应的枚举
     */
    public static ResultCode getByCode(int code) {
        for (ResultCode resultCode : values()) {
            if (resultCode.getCode() == code) {
                return resultCode;
            }
        }
        return null;
    }
    
    /**
     * 判断是否为成功状态
     */
    public boolean isSuccess() {
        return this == SUCCESS;
    }
    
    @Override
    public String toString() {
        return String.format("ResultCode{code=%d, message='%s'}", code, message);
    }
}