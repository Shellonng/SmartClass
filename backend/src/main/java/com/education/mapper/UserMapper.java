package com.education.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.education.entity.User;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;
import org.apache.ibatis.annotations.Update;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

/**
 * 用户Mapper接口
 * 
 * @author Education Platform Team
 * @version 1.0.0
 */
@Mapper
public interface UserMapper extends BaseMapper<User> {

    /**
     * 根据用户名查询用户（包含密码）
     * 
     * @param username 用户名
     * @return 用户信息
     */
    @Select("SELECT * FROM user WHERE username = #{username} AND is_deleted = 0")
    User selectByUsernameWithPassword(@Param("username") String username);

    /**
     * 根据邮箱查询用户
     * 
     * @param email 邮箱
     * @return 用户信息
     */
    @Select("SELECT * FROM user WHERE email = #{email} AND is_deleted = 0")
    User selectByEmail(@Param("email") String email);

    /**
     * 根据手机号查询用户
     * 
     * @param phone 手机号
     * @return 用户信息
     */
    @Select("SELECT * FROM user WHERE phone = #{phone} AND is_deleted = 0")
    User selectByPhone(@Param("phone") String phone);

    /**
     * 更新最后登录信息
     * 
     * @param userId 用户ID
     * @param loginTime 登录时间
     * @param loginIp 登录IP
     * @return 更新结果
     */
    @Update("UPDATE user SET last_login_time = #{loginTime}, last_login_ip = #{loginIp}, update_time = NOW() WHERE id = #{userId}")
    int updateLastLogin(@Param("userId") Long userId, 
                       @Param("loginTime") LocalDateTime loginTime, 
                       @Param("loginIp") String loginIp);

    /**
     * 批量更新用户状态
     * 
     * @param userIds 用户ID列表
     * @param status 状态
     * @return 更新结果
     */
    int batchUpdateStatus(@Param("userIds") List<Long> userIds, @Param("status") String status);

    /**
     * 统计用户总数
     * 
     * @return 用户总数
     */
    @Select("SELECT COUNT(*) FROM user WHERE is_deleted = 0")
    Long countTotalUsers();

    /**
     * 统计活跃用户数（最近N天有登录）
     * 
     * @param days 天数
     * @return 活跃用户数
     */
    @Select("SELECT COUNT(*) FROM user WHERE is_deleted = 0 AND last_login_time >= DATE_SUB(NOW(), INTERVAL #{days} DAY)")
    Long countActiveUsers(@Param("days") int days);

    /**
     * 统计新用户数（最近N天注册）
     * 
     * @param days 天数
     * @return 新用户数
     */
    @Select("SELECT COUNT(*) FROM user WHERE is_deleted = 0 AND create_time >= DATE_SUB(NOW(), INTERVAL #{days} DAY)")
    Long countNewUsers(@Param("days") int days);

    /**
     * 根据角色统计用户数
     * 
     * @return 角色统计结果
     */
    @Select("SELECT role, COUNT(*) as count FROM user WHERE is_deleted = 0 GROUP BY role")
    List<Map<String, Object>> countUsersByRole();

    /**
     * 根据状态统计用户数
     * 
     * @return 状态统计结果
     */
    @Select("SELECT status, COUNT(*) as count FROM user WHERE is_deleted = 0 GROUP BY status")
    List<Map<String, Object>> countUsersByStatus();

    /**
     * 获取用户注册趋势（按天）
     * 
     * @param days 天数
     * @return 注册趋势
     */
    @Select("SELECT DATE(create_time) as date, COUNT(*) as count " +
            "FROM user " +
            "WHERE is_deleted = 0 AND create_time >= DATE_SUB(NOW(), INTERVAL #{days} DAY) " +
            "GROUP BY DATE(create_time) " +
            "ORDER BY date")
    List<Map<String, Object>> getRegistrationTrend(@Param("days") int days);

    /**
     * 获取用户登录趋势（按天）
     * 
     * @param days 天数
     * @return 登录趋势
     */
    @Select("SELECT DATE(last_login_time) as date, COUNT(*) as count, COUNT(DISTINCT id) as unique_users " +
            "FROM user " +
            "WHERE is_deleted = 0 AND last_login_time >= DATE_SUB(NOW(), INTERVAL #{days} DAY) " +
            "GROUP BY DATE(last_login_time) " +
            "ORDER BY date")
    List<Map<String, Object>> getLoginTrend(@Param("days") int days);

    /**
     * 查询长时间未登录的用户
     * 
     * @param days 天数
     * @return 用户列表
     */
    @Select("SELECT * FROM user " +
            "WHERE is_deleted = 0 " +
            "AND (last_login_time IS NULL OR last_login_time < DATE_SUB(NOW(), INTERVAL #{days} DAY)) " +
            "ORDER BY COALESCE(last_login_time, create_time) ASC")
    List<User> selectInactiveUsers(@Param("days") int days);

    /**
     * 查询最近活跃的用户
     * 
     * @param days 天数
     * @param limit 限制数量
     * @return 用户列表
     */
    @Select("SELECT * FROM user " +
            "WHERE is_deleted = 0 AND last_login_time >= DATE_SUB(NOW(), INTERVAL #{days} DAY) " +
            "ORDER BY last_login_time DESC " +
            "LIMIT #{limit}")
    List<User> selectRecentActiveUsers(@Param("days") int days, @Param("limit") int limit);

    /**
     * 搜索用户（模糊查询）
     * 
     * @param keyword 关键词
     * @return 用户列表
     */
    @Select("SELECT * FROM user " +
            "WHERE is_deleted = 0 " +
            "AND (username LIKE CONCAT('%', #{keyword}, '%') " +
            "OR real_name LIKE CONCAT('%', #{keyword}, '%') " +
            "OR email LIKE CONCAT('%', #{keyword}, '%') " +
            "OR phone LIKE CONCAT('%', #{keyword}, '%')) " +
            "ORDER BY create_time DESC")
    List<User> searchUsers(@Param("keyword") String keyword);

    /**
     * 检查用户名是否存在
     * 
     * @param username 用户名
     * @param excludeUserId 排除的用户ID（用于更新时检查）
     * @return 是否存在
     */
    @Select("SELECT COUNT(*) FROM user " +
            "WHERE username = #{username} AND is_deleted = 0 " +
            "AND (#{excludeUserId} IS NULL OR id != #{excludeUserId})")
    int checkUsernameExists(@Param("username") String username, @Param("excludeUserId") Long excludeUserId);

    /**
     * 检查邮箱是否存在
     * 
     * @param email 邮箱
     * @param excludeUserId 排除的用户ID
     * @return 是否存在
     */
    @Select("SELECT COUNT(*) FROM user " +
            "WHERE email = #{email} AND is_deleted = 0 " +
            "AND (#{excludeUserId} IS NULL OR id != #{excludeUserId})")
    int checkEmailExists(@Param("email") String email, @Param("excludeUserId") Long excludeUserId);

    /**
     * 检查手机号是否存在
     * 
     * @param phone 手机号
     * @param excludeUserId 排除的用户ID
     * @return 是否存在
     */
    @Select("SELECT COUNT(*) FROM user " +
            "WHERE phone = #{phone} AND is_deleted = 0 " +
            "AND (#{excludeUserId} IS NULL OR id != #{excludeUserId})")
    int checkPhoneExists(@Param("phone") String phone, @Param("excludeUserId") Long excludeUserId);

    /**
     * 获取用户详细统计信息
     * 
     * @return 统计信息
     */
    @Select("SELECT " +
            "COUNT(*) as total_users, " +
            "SUM(CASE WHEN last_login_time >= DATE_SUB(NOW(), INTERVAL 30 DAY) THEN 1 ELSE 0 END) as active_users, " +
            "SUM(CASE WHEN DATE(create_time) = CURDATE() THEN 1 ELSE 0 END) as new_users_today, " +
            "SUM(CASE WHEN create_time >= DATE_SUB(NOW(), INTERVAL 7 DAY) THEN 1 ELSE 0 END) as new_users_this_week, " +
            "SUM(CASE WHEN create_time >= DATE_SUB(NOW(), INTERVAL 30 DAY) THEN 1 ELSE 0 END) as new_users_this_month, " +
            "SUM(CASE WHEN role = 'TEACHER' THEN 1 ELSE 0 END) as teacher_count, " +
            "SUM(CASE WHEN role = 'STUDENT' THEN 1 ELSE 0 END) as student_count, " +
            "SUM(CASE WHEN role = 'ADMIN' THEN 1 ELSE 0 END) as admin_count, " +
            "SUM(CASE WHEN status = 'ENABLED' THEN 1 ELSE 0 END) as enabled_users, " +
            "SUM(CASE WHEN status = 'DISABLED' THEN 1 ELSE 0 END) as disabled_users, " +
            "SUM(CASE WHEN status = 'LOCKED' THEN 1 ELSE 0 END) as locked_users " +
            "FROM user WHERE is_deleted = 0")
    Map<String, Object> getUserStatistics();

    /**
     * 根据用户ID列表查询用户基本信息
     * 
     * @param userIds 用户ID列表
     * @return 用户列表
     */
    @Select("<script>" +
            "SELECT id, username, real_name, email, phone, role, status, create_time " +
            "FROM user " +
            "WHERE is_deleted = 0 " +
            "<if test='userIds != null and userIds.size() > 0'>" +
            "AND id IN " +
            "<foreach collection='userIds' item='id' open='(' separator=',' close=')'>" +
            "#{id}" +
            "</foreach>" +
            "</if>" +
            "ORDER BY create_time DESC" +
            "</script>")
    List<User> selectUsersByIds(@Param("userIds") List<Long> userIds);

    /**
     * 更新用户扩展字段
     * 
     * @param userId 用户ID
     * @param extField1 扩展字段1
     * @param extField2 扩展字段2
     * @param extField3 扩展字段3
     * @return 更新结果
     */
    @Update("UPDATE user SET ext_field1 = #{extField1}, ext_field2 = #{extField2}, ext_field3 = #{extField3}, update_time = NOW() " +
            "WHERE id = #{userId} AND is_deleted = 0")
    int updateExtFields(@Param("userId") Long userId, 
                       @Param("extField1") String extField1, 
                       @Param("extField2") String extField2, 
                       @Param("extField3") String extField3);

    /**
     * 软删除用户
     * 
     * @param userId 用户ID
     * @return 删除结果
     */
    @Update("UPDATE user SET is_deleted = 1, update_time = NOW() WHERE id = #{userId}")
    int softDeleteUser(@Param("userId") Long userId);

    /**
     * 批量软删除用户
     * 
     * @param userIds 用户ID列表
     * @return 删除结果
     */
    @Update("<script>" +
            "UPDATE user SET is_deleted = 1, update_time = NOW() " +
            "WHERE id IN " +
            "<foreach collection='userIds' item='id' open='(' separator=',' close=')'>" +
            "#{id}" +
            "</foreach>" +
            "</script>")
    int batchSoftDeleteUsers(@Param("userIds") List<Long> userIds);

    /**
     * 恢复已删除的用户
     * 
     * @param userId 用户ID
     * @return 恢复结果
     */
    @Update("UPDATE user SET is_deleted = 0, update_time = NOW() WHERE id = #{userId}")
    int restoreUser(@Param("userId") Long userId);

    /**
     * 清理长时间未激活的用户（物理删除）
     * 
     * @param days 天数
     * @return 清理数量
     */
    int cleanupInactiveUsers(@Param("days") int days);

    /**
     * 获取用户权限信息
     * 
     * @param userId 用户ID
     * @return 权限信息
     */
    @Select("SELECT u.*, r.permissions " +
            "FROM user u " +
            "LEFT JOIN role r ON u.role = r.code " +
            "WHERE u.id = #{userId} AND u.is_deleted = 0")
    Map<String, Object> getUserWithPermissions(@Param("userId") Long userId);

    /**
     * 更新用户头像
     * 
     * @param userId 用户ID
     * @param avatarUrl 头像URL
     * @return 更新结果
     */
    @Update("UPDATE user SET avatar_url = #{avatarUrl}, update_time = NOW() WHERE id = #{userId} AND is_deleted = 0")
    int updateAvatar(@Param("userId") Long userId, @Param("avatarUrl") String avatarUrl);

    /**
     * 重置用户密码
     * 
     * @param userId 用户ID
     * @param newPassword 新密码（已加密）
     * @return 更新结果
     */
    @Update("UPDATE user SET password = #{newPassword}, update_time = NOW() WHERE id = #{userId} AND is_deleted = 0")
    int resetPassword(@Param("userId") Long userId, @Param("newPassword") String newPassword);
}