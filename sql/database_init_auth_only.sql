-- =============================================
-- SmartClass 教育平台 - 认证系统数据库初始化脚本
-- 版本: 1.0.0-simplified
-- 创建时间: 2024-06-28
-- 数据库: MySQL 8.0+
-- 说明: 只包含用户认证相关的核心表结构
-- =============================================

-- 创建数据库
CREATE DATABASE IF NOT EXISTS education_platform DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE education_platform;

-- =============================================
-- 1. 用户表 (user) - 用户认证核心表
-- =============================================
DROP TABLE IF EXISTS `user`;
CREATE TABLE `user` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '用户ID',
  `username` varchar(50) NOT NULL COMMENT '用户名',
  `password` varchar(255) NOT NULL COMMENT '密码(加密)',
  `email` varchar(100) DEFAULT NULL COMMENT '邮箱',
  `phone` varchar(20) DEFAULT NULL COMMENT '手机号',
  `real_name` varchar(50) DEFAULT NULL COMMENT '真实姓名',
  `avatar` varchar(255) DEFAULT NULL COMMENT '头像URL',
  `role` enum('TEACHER','STUDENT','ADMIN') NOT NULL COMMENT '用户角色',
  `status` tinyint DEFAULT '1' COMMENT '状态(0:禁用,1:启用)',
  `last_login_time` datetime DEFAULT NULL COMMENT '最后登录时间',
  `last_login_ip` varchar(50) DEFAULT NULL COMMENT '最后登录IP',
  `created_time` datetime DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `updated_time` datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_username` (`username`),
  UNIQUE KEY `uk_email` (`email`),
  KEY `idx_role` (`role`),
  KEY `idx_status` (`status`),
  KEY `idx_created_time` (`created_time`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='用户表';

-- =============================================
-- 2. 教师表 (teacher) - 教师基本信息
-- =============================================
DROP TABLE IF EXISTS `teacher`;
CREATE TABLE `teacher` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '教师ID',
  `user_id` bigint NOT NULL COMMENT '用户ID',
  `department` varchar(100) DEFAULT NULL COMMENT '所属院系',
  `title` varchar(50) DEFAULT NULL COMMENT '职称',
  `education` varchar(50) DEFAULT NULL COMMENT '学历',
  `specialty` varchar(200) DEFAULT NULL COMMENT '专业领域',
  `introduction` text COMMENT '个人简介',
  `office_location` varchar(100) DEFAULT NULL COMMENT '办公地点',
  `office_hours` varchar(200) DEFAULT NULL COMMENT '办公时间',
  `created_time` datetime DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `updated_time` datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_user_id` (`user_id`),
  KEY `idx_department` (`department`),
  CONSTRAINT `fk_teacher_user` FOREIGN KEY (`user_id`) REFERENCES `user` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='教师表';

-- =============================================
-- 3. 学生表 (student) - 学生基本信息
-- =============================================
DROP TABLE IF EXISTS `student`;
CREATE TABLE `student` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '学生ID',
  `user_id` bigint NOT NULL COMMENT '用户ID',
  `student_number` varchar(20) DEFAULT NULL COMMENT '学号',
  `major` varchar(100) DEFAULT NULL COMMENT '专业',
  `grade` varchar(10) DEFAULT NULL COMMENT '年级',
  `enrollment_year` int DEFAULT NULL COMMENT '入学年份',
  `graduation_year` int DEFAULT NULL COMMENT '毕业年份',
  `gpa` decimal(3,2) DEFAULT NULL COMMENT 'GPA',
  `status` enum('ACTIVE','SUSPENDED','GRADUATED','DROPPED') DEFAULT 'ACTIVE' COMMENT '学籍状态',
  `created_time` datetime DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `updated_time` datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_user_id` (`user_id`),
  UNIQUE KEY `uk_student_number` (`student_number`),
  KEY `idx_major` (`major`),
  KEY `idx_grade` (`grade`),
  KEY `idx_status` (`status`),
  CONSTRAINT `fk_student_user` FOREIGN KEY (`user_id`) REFERENCES `user` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='学生表';

-- =============================================
-- 4. 用户会话表 (user_session) - JWT Token管理
-- =============================================
DROP TABLE IF EXISTS `user_session`;
CREATE TABLE `user_session` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '会话ID',
  `user_id` bigint NOT NULL COMMENT '用户ID',
  `token` varchar(500) NOT NULL COMMENT 'JWT Token',
  `refresh_token` varchar(500) DEFAULT NULL COMMENT '刷新Token',
  `device_type` varchar(50) DEFAULT NULL COMMENT '设备类型',
  `device_info` varchar(200) DEFAULT NULL COMMENT '设备信息',
  `ip_address` varchar(50) DEFAULT NULL COMMENT 'IP地址',
  `login_time` datetime DEFAULT CURRENT_TIMESTAMP COMMENT '登录时间',
  `expire_time` datetime NOT NULL COMMENT '过期时间',
  `is_active` tinyint DEFAULT '1' COMMENT '是否活跃(0:否,1:是)',
  `created_time` datetime DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `updated_time` datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  KEY `idx_user_id` (`user_id`),
  KEY `idx_token` (`token`(100)),
  KEY `idx_expire_time` (`expire_time`),
  KEY `idx_is_active` (`is_active`),
  CONSTRAINT `fk_session_user` FOREIGN KEY (`user_id`) REFERENCES `user` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='用户会话表';

-- =============================================
-- 5. 密码重置表 (password_reset) - 密码重置验证码
-- =============================================
DROP TABLE IF EXISTS `password_reset`;
CREATE TABLE `password_reset` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '重置ID',
  `user_id` bigint NOT NULL COMMENT '用户ID',
  `email` varchar(100) NOT NULL COMMENT '邮箱',
  `verification_code` varchar(10) NOT NULL COMMENT '验证码',
  `expire_time` datetime NOT NULL COMMENT '过期时间',
  `is_used` tinyint DEFAULT '0' COMMENT '是否已使用(0:否,1:是)',
  `created_time` datetime DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  PRIMARY KEY (`id`),
  KEY `idx_user_id` (`user_id`),
  KEY `idx_email` (`email`),
  KEY `idx_verification_code` (`verification_code`),
  KEY `idx_expire_time` (`expire_time`),
  CONSTRAINT `fk_reset_user` FOREIGN KEY (`user_id`) REFERENCES `user` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='密码重置表';

-- =============================================
-- 6. 验证码表 (captcha) - 图形验证码
-- =============================================
DROP TABLE IF EXISTS `captcha`;
CREATE TABLE `captcha` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '验证码ID',
  `captcha_key` varchar(100) NOT NULL COMMENT '验证码key',
  `captcha_code` varchar(10) NOT NULL COMMENT '验证码',
  `captcha_image` text COMMENT '验证码图片(base64)',
  `expire_time` datetime NOT NULL COMMENT '过期时间',
  `is_used` tinyint DEFAULT '0' COMMENT '是否已使用(0:否,1:是)',
  `created_time` datetime DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  PRIMARY KEY (`id`),
  KEY `idx_captcha_key` (`captcha_key`),
  KEY `idx_expire_time` (`expire_time`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='验证码表';

-- =============================================
-- 初始化数据
-- =============================================

-- 插入管理员用户
INSERT INTO `user` (`username`, `password`, `email`, `real_name`, `role`, `status`) VALUES
('admin', '$2a$10$N.zmdr9k7uOCQb376NoUnuTJ8iKyVqNhKwvKS7VdwjnM6YYMNjvDe', 'admin@example.com', '系统管理员', 'ADMIN', 1);

-- 插入测试教师用户
INSERT INTO `user` (`username`, `password`, `email`, `real_name`, `role`, `status`) VALUES
('teacher1', '$2a$10$N.zmdr9k7uOCQb376NoUnuTJ8iKyVqNhKwvKS7VdwjnM6YYMNjvDe', 'teacher1@example.com', '张老师', 'TEACHER', 1),
('teacher2', '$2a$10$N.zmdr9k7uOCQb376NoUnuTJ8iKyVqNhKwvKS7VdwjnM6YYMNjvDe', 'teacher2@example.com', '李老师', 'TEACHER', 1);

-- 插入测试学生用户
INSERT INTO `user` (`username`, `password`, `email`, `real_name`, `role`, `status`) VALUES
('student1', '$2a$10$N.zmdr9k7uOCQb376NoUnuTJ8iKyVqNhKwvKS7VdwjnM6YYMNjvDe', 'student1@example.com', '王同学', 'STUDENT', 1),
('student2', '$2a$10$N.zmdr9k7uOCQb376NoUnuTJ8iKyVqNhKwvKS7VdwjnM6YYMNjvDe', 'student2@example.com', '刘同学', 'STUDENT', 1),
('student3', '$2a$10$N.zmdr9k7uOCQb376NoUnuTJ8iKyVqNhKwvKS7VdwjnM6YYMNjvDe', 'student3@example.com', '陈同学', 'STUDENT', 1);

-- 插入教师信息
INSERT INTO `teacher` (`user_id`, `department`, `title`, `education`, `specialty`) VALUES
((SELECT id FROM `user` WHERE username = 'teacher1'), '计算机学院', '副教授', '博士', '软件工程、人工智能'),
((SELECT id FROM `user` WHERE username = 'teacher2'), '数学学院', '讲师', '硕士', '高等数学、线性代数');

-- 插入学生信息
INSERT INTO `student` (`user_id`, `student_number`, `major`, `grade`, `enrollment_year`, `graduation_year`) VALUES
((SELECT id FROM `user` WHERE username = 'student1'), '2021001001', '计算机科学与技术', '2021级', 2021, 2025),
((SELECT id FROM `user` WHERE username = 'student2'), '2021001002', '计算机科学与技术', '2021级', 2021, 2025),
((SELECT id FROM `user` WHERE username = 'student3'), '2022001001', '软件工程', '2022级', 2022, 2026);

-- =============================================
-- 创建视图和索引优化
-- =============================================

-- 创建用户详情视图（包含角色信息）
CREATE OR REPLACE VIEW `v_user_details` AS
SELECT 
    u.id,
    u.username,
    u.email,
    u.phone,
    u.real_name,
    u.avatar,
    u.role,
    u.status,
    u.last_login_time,
    u.last_login_ip,
    u.created_time,
    u.updated_time,
    CASE 
        WHEN u.role = 'TEACHER' THEN t.department
        WHEN u.role = 'STUDENT' THEN s.major
        ELSE NULL
    END AS department_or_major,
    CASE 
        WHEN u.role = 'TEACHER' THEN t.title
        WHEN u.role = 'STUDENT' THEN s.grade
        ELSE NULL
    END AS title_or_grade
FROM `user` u
LEFT JOIN `teacher` t ON u.id = t.user_id AND u.role = 'TEACHER'
LEFT JOIN `student` s ON u.id = s.user_id AND u.role = 'STUDENT';

-- =============================================
-- 数据库初始化完成
-- =============================================

-- 显示初始化结果
SELECT '数据库初始化完成！' AS message;
SELECT COUNT(*) AS user_count FROM `user`;
SELECT COUNT(*) AS teacher_count FROM `teacher`;
SELECT COUNT(*) AS student_count FROM `student`;

-- 显示测试账号信息
SELECT 
    '测试账号信息（密码统一为：123456）' AS info,
    username,
    real_name,
    role,
    email
FROM `user` 
ORDER BY role, username; 