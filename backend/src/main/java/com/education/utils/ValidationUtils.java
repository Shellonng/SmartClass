package com.education.utils;

import java.util.regex.Pattern;

/**
 * 验证工具类
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
public class ValidationUtils {

    /**
     * 邮箱正则表达式
     */
    private static final String EMAIL_REGEX = "^[a-zA-Z0-9_+&*-]+(?:\\.[a-zA-Z0-9_+&*-]+)*@(?:[a-zA-Z0-9-]+\\.)+[a-zA-Z]{2,7}$";

    /**
     * 手机号正则表达式（中国大陆）
     */
    private static final String PHONE_REGEX = "^1[3-9]\\d{9}$";

    /**
     * 身份证号正则表达式
     */
    private static final String ID_CARD_REGEX = "^[1-9]\\d{5}(19|20)\\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\\d|3[01])\\d{3}[0-9Xx]$";

    /**
     * 用户名正则表达式（字母开头，允许5-20字节，允许字母数字下划线）
     */
    private static final String USERNAME_REGEX = "^[a-zA-Z][a-zA-Z0-9_]{4,19}$";

    /**
     * 密码正则表达式（至少8位，包含大小写字母、数字和特殊字符）
     */
    private static final String PASSWORD_REGEX = "^(?=.*[a-z])(?=.*[A-Z])(?=.*\\d)(?=.*[@$!%*?&])[A-Za-z\\d@$!%*?&]{8,}$";

    /**
     * 验证邮箱
     * 
     * @param email 邮箱
     * @return 是否有效
     */
    public static boolean isValidEmail(String email) {
        return email != null && Pattern.matches(EMAIL_REGEX, email);
    }

    /**
     * 验证手机号
     * 
     * @param phone 手机号
     * @return 是否有效
     */
    public static boolean isValidPhone(String phone) {
        return phone != null && Pattern.matches(PHONE_REGEX, phone);
    }

    /**
     * 验证身份证号
     * 
     * @param idCard 身份证号
     * @return 是否有效
     */
    public static boolean isValidIdCard(String idCard) {
        if (idCard == null || !Pattern.matches(ID_CARD_REGEX, idCard)) {
            return false;
        }
        
        // 进一步验证身份证校验码
        return validateIdCardChecksum(idCard);
    }

    /**
     * 验证用户名
     * 
     * @param username 用户名
     * @return 是否有效
     */
    public static boolean isValidUsername(String username) {
        return username != null && Pattern.matches(USERNAME_REGEX, username);
    }

    /**
     * 验证密码
     * 
     * @param password 密码
     * @return 是否有效
     */
    public static boolean isValidPassword(String password) {
        return password != null && Pattern.matches(PASSWORD_REGEX, password);
    }

    /**
     * 验证身份证校验码
     * 
     * @param idCard 身份证号
     * @return 是否有效
     */
    private static boolean validateIdCardChecksum(String idCard) {
        char[] chars = idCard.toCharArray();
        int[] weights = {7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2};
        char[] checkCodes = {'1', '0', 'X', '9', '8', '7', '6', '5', '4', '3', '2'};
        
        int sum = 0;
        for (int i = 0; i < 17; i++) {
            sum += (chars[i] - '0') * weights[i];
        }
        
        int mod = sum % 11;
        char checkCode = checkCodes[mod];
        
        return Character.toUpperCase(chars[17]) == checkCode;
    }

    /**
     * 验证学号（纯数字，长度为8-12位）
     * 
     * @param studentId 学号
     * @return 是否有效
     */
    public static boolean isValidStudentId(String studentId) {
        return studentId != null && studentId.matches("^\\d{8,12}$");
    }

    /**
     * 验证教师工号（纯数字，长度为5-10位）
     * 
     * @param teacherId 教师工号
     * @return 是否有效
     */
    public static boolean isValidTeacherId(String teacherId) {
        return teacherId != null && teacherId.matches("^\\d{5,10}$");
    }

    /**
     * 验证班级代码（字母和数字的组合，长度为4-10位）
     * 
     * @param classCode 班级代码
     * @return 是否有效
     */
    public static boolean isValidClassCode(String classCode) {
        return classCode != null && classCode.matches("^[a-zA-Z0-9]{4,10}$");
    }

    /**
     * 验证课程代码（字母和数字的组合，长度为4-10位）
     * 
     * @param courseCode 课程代码
     * @return 是否有效
     */
    public static boolean isValidCourseCode(String courseCode) {
        return courseCode != null && courseCode.matches("^[a-zA-Z0-9]{4,10}$");
    }
}