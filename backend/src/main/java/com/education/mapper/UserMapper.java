package com.education.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.education.entity.User;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;
import org.apache.ibatis.annotations.Update;

import java.time.LocalDateTime;

/**
 * 用户Mapper接口 - 简化版
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
    @Select("SELECT * FROM user WHERE username = #{username}")
    User selectByUsernameWithPassword(@Param("username") String username);

    /**
     * 根据邮箱查询用户
     * 
     * @param email 邮箱
     * @return 用户信息
     */
    @Select("SELECT * FROM user WHERE email = #{email}")
    User selectByEmail(@Param("email") String email);

    /**
     * 检查用户名是否存在
     * 
     * @param username 用户名
     * @param excludeUserId 排除的用户ID（用于更新时检查）
     * @return 是否存在
     */
    @Select("SELECT COUNT(*) FROM user WHERE username = #{username} AND (#{excludeUserId} IS NULL OR id != #{excludeUserId})")
    int checkUsernameExists(@Param("username") String username, @Param("excludeUserId") Long excludeUserId);

    /**
     * 检查邮箱是否存在
     * 
     * @param email 邮箱
     * @param excludeUserId 排除的用户ID
     * @return 是否存在
     */
    @Select("SELECT COUNT(*) FROM user WHERE email = #{email} AND (#{excludeUserId} IS NULL OR id != #{excludeUserId})")
    int checkEmailExists(@Param("email") String email, @Param("excludeUserId") Long excludeUserId);

    /**
     * 重置用户密码
     * 
     * @param userId 用户ID
     * @param newPassword 新密码（已加密）
     * @return 更新结果
     */
    @Update("UPDATE user SET password = #{newPassword}, update_time = NOW() WHERE id = #{userId}")
    int resetPassword(@Param("userId") Long userId, @Param("newPassword") String newPassword);

    /**
     * 根据用户ID查询用户
     * 
     * @param userId 用户ID
     * @return 用户信息
     */
    @Select("SELECT * FROM user WHERE id = #{userId}")
    User selectByUserId(@Param("userId") Long userId);
}