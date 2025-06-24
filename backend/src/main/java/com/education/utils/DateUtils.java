package com.education.utils;

import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.time.*;
import java.time.format.DateTimeFormatter;
import java.time.temporal.ChronoUnit;
import java.time.temporal.TemporalAdjusters;
import java.util.Date;

/**
 * 日期工具类
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
public class DateUtils {

    /**
     * 默认日期格式
     */
    public static final String DEFAULT_DATE_FORMAT = "yyyy-MM-dd";

    /**
     * 默认日期时间格式
     */
    public static final String DEFAULT_DATETIME_FORMAT = "yyyy-MM-dd HH:mm:ss";

    /**
     * 格式化日期为字符串
     * 
     * @param date 日期
     * @param pattern 格式
     * @return 格式化后的日期字符串
     */
    public static String format(Date date, String pattern) {
        if (date == null) {
            return null;
        }
        SimpleDateFormat formatter = new SimpleDateFormat(pattern);
        return formatter.format(date);
    }

    /**
     * 格式化日期为默认格式字符串
     * 
     * @param date 日期
     * @return 格式化后的日期字符串
     */
    public static String formatDate(Date date) {
        return format(date, DEFAULT_DATE_FORMAT);
    }

    /**
     * 格式化日期时间为默认格式字符串
     * 
     * @param date 日期
     * @return 格式化后的日期时间字符串
     */
    public static String formatDateTime(Date date) {
        return format(date, DEFAULT_DATETIME_FORMAT);
    }

    /**
     * 解析字符串为日期
     * 
     * @param dateStr 日期字符串
     * @param pattern 格式
     * @return 解析后的日期
     * @throws ParseException 解析异常
     */
    public static Date parse(String dateStr, String pattern) throws ParseException {
        if (dateStr == null || dateStr.isEmpty()) {
            return null;
        }
        SimpleDateFormat formatter = new SimpleDateFormat(pattern);
        return formatter.parse(dateStr);
    }

    /**
     * 解析字符串为默认格式日期
     * 
     * @param dateStr 日期字符串
     * @return 解析后的日期
     * @throws ParseException 解析异常
     */
    public static Date parseDate(String dateStr) throws ParseException {
        return parse(dateStr, DEFAULT_DATE_FORMAT);
    }

    /**
     * 解析字符串为默认格式日期时间
     * 
     * @param dateStr 日期时间字符串
     * @return 解析后的日期时间
     * @throws ParseException 解析异常
     */
    public static Date parseDateTime(String dateStr) throws ParseException {
        return parse(dateStr, DEFAULT_DATETIME_FORMAT);
    }

    /**
     * 获取当前日期
     * 
     * @return 当前日期
     */
    public static Date getCurrentDate() {
        return new Date();
    }

    /**
     * 获取当前日期字符串
     * 
     * @return 当前日期字符串
     */
    public static String getCurrentDateStr() {
        return formatDate(getCurrentDate());
    }

    /**
     * 获取当前日期时间字符串
     * 
     * @return 当前日期时间字符串
     */
    public static String getCurrentDateTimeStr() {
        return formatDateTime(getCurrentDate());
    }

    /**
     * 获取当前学年
     * 
     * @return 当前学年（如：2023-2024）
     */
    public static String getCurrentAcademicYear() {
        LocalDate now = LocalDate.now();
        int year = now.getYear();
        int month = now.getMonthValue();
        
        // 如果当前月份小于9月，则学年为上一年到当前年
        if (month < 9) {
            return (year - 1) + "-" + year;
        } else {
            // 否则学年为当前年到下一年
            return year + "-" + (year + 1);
        }
    }

    /**
     * 获取当前学期
     * 
     * @return 当前学期（1：秋季学期，2：春季学期）
     */
    public static int getCurrentSemester() {
        LocalDate now = LocalDate.now();
        int month = now.getMonthValue();
        
        // 9月到次年2月为秋季学期，3月到8月为春季学期
        if (month >= 9 || month <= 2) {
            return 1; // 秋季学期
        } else {
            return 2; // 春季学期
        }
    }

    /**
     * 计算两个日期之间的天数差
     * 
     * @param date1 日期1
     * @param date2 日期2
     * @return 天数差
     */
    public static long daysBetween(Date date1, Date date2) {
        LocalDate localDate1 = date1.toInstant().atZone(ZoneId.systemDefault()).toLocalDate();
        LocalDate localDate2 = date2.toInstant().atZone(ZoneId.systemDefault()).toLocalDate();
        return ChronoUnit.DAYS.between(localDate1, localDate2);
    }

    /**
     * 获取指定日期的开始时间
     * 
     * @param date 日期
     * @return 开始时间
     */
    public static Date getStartOfDay(Date date) {
        LocalDateTime localDateTime = date.toInstant().atZone(ZoneId.systemDefault()).toLocalDate().atStartOfDay();
        return Date.from(localDateTime.atZone(ZoneId.systemDefault()).toInstant());
    }

    /**
     * 获取指定日期的结束时间
     * 
     * @param date 日期
     * @return 结束时间
     */
    public static Date getEndOfDay(Date date) {
        LocalDateTime localDateTime = date.toInstant().atZone(ZoneId.systemDefault()).toLocalDate().atTime(23, 59, 59, 999999999);
        return Date.from(localDateTime.atZone(ZoneId.systemDefault()).toInstant());
    }

    /**
     * 获取指定日期所在月的第一天
     * 
     * @param date 日期
     * @return 所在月的第一天
     */
    public static Date getFirstDayOfMonth(Date date) {
        LocalDateTime localDateTime = date.toInstant().atZone(ZoneId.systemDefault())
                .with(TemporalAdjusters.firstDayOfMonth())
                .toLocalDate().atStartOfDay();
        return Date.from(localDateTime.atZone(ZoneId.systemDefault()).toInstant());
    }

    /**
     * 获取指定日期所在月的最后一天
     * 
     * @param date 日期
     * @return 所在月的最后一天
     */
    public static Date getLastDayOfMonth(Date date) {
        LocalDateTime localDateTime = date.toInstant().atZone(ZoneId.systemDefault())
                .with(TemporalAdjusters.lastDayOfMonth())
                .toLocalDate().atTime(23, 59, 59, 999999999);
        return Date.from(localDateTime.atZone(ZoneId.systemDefault()).toInstant());
    }
}