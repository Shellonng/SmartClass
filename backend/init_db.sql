-- 简化数据库初始化脚本
CREATE DATABASE IF NOT EXISTS education_platform DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE education_platform;

-- 用户表
CREATE TABLE IF NOT EXISTS `user` (
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
  `deleted` tinyint DEFAULT '0' COMMENT '是否删除(0:否,1:是)',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_username` (`username`),
  UNIQUE KEY `uk_email` (`email`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='用户表';

-- 插入测试用户 (密码为明文，实际使用时需要加密)
INSERT IGNORE INTO user (username, password, email, real_name, role, status) VALUES 
('root', '$2a$10$N.zmdr9k7uOCQb376NoUnuTJ8iAt6Z5EHsM8lE9lBOsl7.QdIuO3e', 'root@education.com', 'Root管理员', 'ADMIN', 1),
('teacher1', '$2a$10$N.zmdr9k7uOCQb376NoUnuTJ8iAt6Z5EHsM8lE9lBOsl7.QdIuO3e', 'teacher1@education.com', '张教授', 'TEACHER', 1),
('student1', '$2a$10$N.zmdr9k7uOCQb376NoUnuTJ8iAt6Z5EHsM8lE9lBOsl7.QdIuO3e', 'student1@education.com', '张三', 'STUDENT', 1);

-- 教师表
CREATE TABLE IF NOT EXISTS `teacher` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '教师ID',
  `user_id` bigint NOT NULL COMMENT '用户ID',
  `teacher_no` varchar(20) NOT NULL COMMENT '教师工号',
  `department` varchar(100) DEFAULT NULL COMMENT '所属院系',
  `created_time` datetime DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `updated_time` datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  `deleted` tinyint DEFAULT '0' COMMENT '是否删除(0:否,1:是)',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_user_id` (`user_id`),
  UNIQUE KEY `uk_teacher_no` (`teacher_no`),
  CONSTRAINT `fk_teacher_user` FOREIGN KEY (`user_id`) REFERENCES `user` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='教师表';

-- 学生表
CREATE TABLE IF NOT EXISTS `student` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '学生ID',
  `user_id` bigint NOT NULL COMMENT '用户ID',
  `student_no` varchar(20) NOT NULL COMMENT '学号',
  `class_id` bigint DEFAULT NULL COMMENT '班级ID',
  `major` varchar(100) DEFAULT NULL COMMENT '专业',
  `grade` varchar(10) DEFAULT NULL COMMENT '年级',
  `created_time` datetime DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `updated_time` datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  `deleted` tinyint DEFAULT '0' COMMENT '是否删除(0:否,1:是)',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_user_id` (`user_id`),
  UNIQUE KEY `uk_student_no` (`student_no`),
  CONSTRAINT `fk_student_user` FOREIGN KEY (`user_id`) REFERENCES `user` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='学生表';

-- 插入教师和学生记录
INSERT IGNORE INTO teacher (user_id, teacher_no, department) VALUES
((SELECT id FROM user WHERE username = 'teacher1'), 'T001', '计算机科学系');

INSERT IGNORE INTO student (user_id, student_no, major, grade) VALUES
((SELECT id FROM user WHERE username = 'student1'), 'S2024001', '计算机科学与技术', '大二');

SELECT 'Database initialization completed!' as message; 