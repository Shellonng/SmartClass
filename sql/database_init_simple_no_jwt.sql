-- 简化版数据库初始化脚本（无JWT）
-- 只包含基本的用户认证功能

-- 删除已有数据库并重新创建
DROP DATABASE IF EXISTS `education_platform`;
CREATE DATABASE `education_platform` DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE `education_platform`;

-- 用户表
CREATE TABLE `user` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键ID',
  `username` varchar(50) NOT NULL COMMENT '用户名',
  `password` varchar(255) NOT NULL COMMENT '密码（加密后）',
  `email` varchar(100) NOT NULL COMMENT '邮箱',
  `real_name` varchar(50) NOT NULL COMMENT '真实姓名',
  `avatar` varchar(255) DEFAULT NULL COMMENT '头像URL',
  `role` varchar(20) NOT NULL COMMENT '角色：STUDENT/TEACHER',
  `status` varchar(20) DEFAULT 'ACTIVE' COMMENT '状态：ACTIVE/LOCKED',
  `create_time` datetime DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_username` (`username`),
  UNIQUE KEY `uk_email` (`email`),
  KEY `idx_role` (`role`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='用户表';

-- 学生表
CREATE TABLE `student` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键ID',
  `user_id` bigint NOT NULL COMMENT '用户ID',
  `student_id` varchar(20) NOT NULL COMMENT '学号',
  `enrollment_status` varchar(20) DEFAULT 'ENROLLED' COMMENT '注册状态',
  `gpa` decimal(3,2) DEFAULT NULL COMMENT 'GPA',
  `gpa_level` varchar(10) DEFAULT NULL COMMENT 'GPA等级',
  `create_time` datetime DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_user_id` (`user_id`),
  UNIQUE KEY `uk_student_id` (`student_id`),
  CONSTRAINT `fk_student_user` FOREIGN KEY (`user_id`) REFERENCES `user` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='学生表';

-- 教师表
CREATE TABLE `teacher` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键ID',
  `user_id` bigint NOT NULL COMMENT '用户ID',
  `teacher_id` varchar(20) NOT NULL COMMENT '教师工号',
  `department` varchar(100) DEFAULT NULL COMMENT '所属部门',
  `title` varchar(50) DEFAULT NULL COMMENT '职称',
  `create_time` datetime DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_user_id` (`user_id`),
  UNIQUE KEY `uk_teacher_id` (`teacher_id`),
  CONSTRAINT `fk_teacher_user` FOREIGN KEY (`user_id`) REFERENCES `user` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='教师表';

-- 插入测试数据
-- 管理员用户（密码：123456）
INSERT INTO `user` (`username`, `password`, `email`, `real_name`, `role`) VALUES
('admin', '$2a$10$N.zmdr9k7uOCQb376NoUnuTJ8iKTnl5iKkpKvs5RFRE4B3TtQkfTu', 'admin@example.com', '管理员', 'TEACHER');

-- 教师用户（密码：123456）
INSERT INTO `user` (`username`, `password`, `email`, `real_name`, `role`) VALUES
('teacher1', '$2a$10$N.zmdr9k7uOCQb376NoUnuTJ8iKTnl5iKkpKvs5RFRE4B3TtQkfTu', 'teacher1@example.com', '张老师', 'TEACHER'),
('teacher2', '$2a$10$N.zmdr9k7uOCQb376NoUnuTJ8iKTnl5iKkpKvs5RFRE4B3TtQkfTu', 'teacher2@example.com', '李老师', 'TEACHER');

-- 学生用户（密码：123456）
INSERT INTO `user` (`username`, `password`, `email`, `real_name`, `role`) VALUES
('student1', '$2a$10$N.zmdr9k7uOCQb376NoUnuTJ8iKTnl5iKkpKvs5RFRE4B3TtQkfTu', 'student1@example.com', '王同学', 'STUDENT'),
('student2', '$2a$10$N.zmdr9k7uOCQb376NoUnuTJ8iKTnl5iKkpKvs5RFRE4B3TtQkfTu', 'student2@example.com', '刘同学', 'STUDENT'),
('student3', '$2a$10$N.zmdr9k7uOCQb376NoUnuTJ8iKTnl5iKkpKvs5RFRE4B3TtQkfTu', 'student3@example.com', '陈同学', 'STUDENT');

-- 插入教师详细信息
INSERT INTO `teacher` (`user_id`, `teacher_id`, `department`, `title`) VALUES
((SELECT id FROM `user` WHERE username = 'admin'), 'T20240001', '信息技术部', '高级工程师'),
((SELECT id FROM `user` WHERE username = 'teacher1'), 'T20240002', '计算机科学系', '副教授'),
((SELECT id FROM `user` WHERE username = 'teacher2'), 'T20240003', '软件工程系', '讲师');

-- 插入学生详细信息
INSERT INTO `student` (`user_id`, `student_id`, `enrollment_status`, `gpa`, `gpa_level`) VALUES
((SELECT id FROM `user` WHERE username = 'student1'), '20240001', 'ENROLLED', 3.8, 'A'),
((SELECT id FROM `user` WHERE username = 'student2'), '20240002', 'ENROLLED', 3.5, 'B'),
((SELECT id FROM `user` WHERE username = 'student3'), '20240003', 'ENROLLED', 3.2, 'B');

-- 显示初始化结果
SELECT '数据库初始化完成' as message;
SELECT COUNT(*) as user_count FROM `user`;
SELECT COUNT(*) as teacher_count FROM `teacher`;
SELECT COUNT(*) as student_count FROM `student`; 