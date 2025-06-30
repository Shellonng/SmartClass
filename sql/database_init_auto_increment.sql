-- =============================================
-- AI赋能教育管理与学习辅助平台 - 自增ID版数据库初始化脚本
-- 版本: 3.0
-- 创建时间: 2024
-- 数据库: MySQL 8.0+
-- 说明: 使用自增ID而不是雪花算法
-- =============================================

-- 创建数据库
CREATE DATABASE IF NOT EXISTS education_platform DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE education_platform;

-- =============================================
-- 1. 用户表 (user) - 从1开始自增
-- =============================================
DROP TABLE IF EXISTS `user`;
CREATE TABLE `user` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '用户ID',
  `username` varchar(50) NOT NULL COMMENT '用户名',
  `password` varchar(255) NOT NULL COMMENT '密码(加密)',
  `email` varchar(100) DEFAULT NULL COMMENT '邮箱',
  `real_name` varchar(50) DEFAULT NULL COMMENT '真实姓名',
  `avatar` varchar(255) DEFAULT NULL COMMENT '头像URL',
  `role` varchar(20) NOT NULL COMMENT '用户角色',
  `status` varchar(20) DEFAULT 'ACTIVE' COMMENT '状态',
  `create_time` datetime DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_username` (`username`),
  UNIQUE KEY `uk_email` (`email`)
) ENGINE=InnoDB AUTO_INCREMENT=1 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='用户表';

-- =============================================
-- 2. 教师表 (teacher) - 从2025000开始自增
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
  `contact_email` varchar(100) DEFAULT NULL COMMENT '联系邮箱',
  `contact_phone` varchar(20) DEFAULT NULL COMMENT '联系电话',
  `status` varchar(20) DEFAULT 'ACTIVE' COMMENT '状态',
  `hire_date` datetime DEFAULT NULL COMMENT '入职日期',
  `create_time` datetime DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_user_id` (`user_id`),
  KEY `idx_department` (`department`),
  CONSTRAINT `fk_teacher_user` FOREIGN KEY (`user_id`) REFERENCES `user` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=2025000 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='教师表';

-- =============================================
-- 3. 学生表 (student) - 从20250000开始自增
-- =============================================
DROP TABLE IF EXISTS `student`;
CREATE TABLE `student` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '学生ID',
  `user_id` bigint NOT NULL COMMENT '用户ID',
  `student_id` varchar(50) DEFAULT NULL COMMENT '学号',
  `enrollment_status` varchar(20) DEFAULT 'ENROLLED' COMMENT '学籍状态',
  `gpa` decimal(3,2) DEFAULT NULL COMMENT 'GPA',
  `gpa_level` varchar(5) DEFAULT NULL COMMENT 'GPA等级',
  `create_time` datetime DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_user_id` (`user_id`),
  UNIQUE KEY `uk_student_id` (`student_id`),
  CONSTRAINT `fk_student_user` FOREIGN KEY (`user_id`) REFERENCES `user` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=20250000 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='学生表';

-- =============================================
-- 4. 课程表 (course) - 从1开始自增
-- =============================================
DROP TABLE IF EXISTS `course`;
CREATE TABLE `course` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '课程ID',
  `title` varchar(100) NOT NULL COMMENT '课程名称',
  `description` text COMMENT '课程描述',
  `cover_image` varchar(500) DEFAULT NULL COMMENT '课程封面图片',
  `credit` decimal(3,1) DEFAULT '3.0' COMMENT '学分',
  `course_type` varchar(20) DEFAULT '必修课' COMMENT '课程类型',
  `start_time` datetime DEFAULT NULL COMMENT '开始时间',
  `end_time` datetime DEFAULT NULL COMMENT '结束时间',
  `teacher_id` bigint NOT NULL COMMENT '教师ID',
  `status` varchar(20) DEFAULT '未开始' COMMENT '课程状态',
  `term` varchar(20) DEFAULT NULL COMMENT '学期',
  `student_count` int DEFAULT '0' COMMENT '学生数量',
  `average_score` decimal(5,2) DEFAULT NULL COMMENT '平均分数',
  `create_time` datetime DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  KEY `idx_teacher_id` (`teacher_id`),
  KEY `idx_term` (`term`),
  KEY `idx_status` (`status`),
  CONSTRAINT `fk_course_teacher` FOREIGN KEY (`teacher_id`) REFERENCES `teacher` (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=1 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='课程表';

-- =============================================
-- 5. 初始化测试数据
-- =============================================

-- 创建管理员用户
INSERT INTO `user` (`username`, `password`, `email`, `real_name`, `role`, `status`) VALUES
('admin', '$2a$10$N.zmdr9k7uOCQb376NoUnuTJ8iAt6Z5EHsNSEYy4k4onaZDe17ZOS', 'admin@example.com', '系统管理员', 'ADMIN', 'ACTIVE');

-- 创建教师用户
INSERT INTO `user` (`username`, `password`, `email`, `real_name`, `role`, `status`) VALUES
('teacher1', '$2a$10$N.zmdr9k7uOCQb376NoUnuTJ8iAt6Z5EHsNSEYy4k4onaZDe17ZOS', 'teacher1@example.com', '张教授', 'TEACHER', 'ACTIVE'),
('teacher2', '$2a$10$N.zmdr9k7uOCQb376NoUnuTJ8iAt6Z5EHsNSEYy4k4onaZDe17ZOS', 'teacher2@example.com', '李教授', 'TEACHER', 'ACTIVE');

-- 创建学生用户
INSERT INTO `user` (`username`, `password`, `email`, `real_name`, `role`, `status`) VALUES
('student1', '$2a$10$N.zmdr9k7uOCQb376NoUnuTJ8iAt6Z5EHsNSEYy4k4onaZDe17ZOS', 'student1@example.com', '王同学', 'STUDENT', 'ACTIVE'),
('student2', '$2a$10$N.zmdr9k7uOCQb376NoUnuTJ8iAt6Z5EHsNSEYy4k4onaZDe17ZOS', 'student2@example.com', '赵同学', 'STUDENT', 'ACTIVE');

-- 创建教师信息
INSERT INTO `teacher` (`user_id`, `department`, `title`, `education`, `specialty`, `introduction`, `office_location`, `office_hours`, `contact_email`, `status`) VALUES
(2, '计算机科学与技术学院', '教授', '博士', '人工智能', '张教授是人工智能领域的专家', 'A栋201', '周一至周五 9:00-17:00', 'teacher1@example.com', 'ACTIVE'),
(3, '数学学院', '副教授', '博士', '应用数学', '李教授专注于应用数学研究', 'B栋305', '周一至周五 10:00-16:00', 'teacher2@example.com', 'ACTIVE');

-- 创建学生信息
INSERT INTO `student` (`user_id`, `student_id`, `enrollment_status`, `gpa`) VALUES
(4, '2025001', 'ENROLLED', 3.85),
(5, '2025002', 'ENROLLED', 3.75);

-- 创建课程
INSERT INTO `course` (`title`, `description`, `credit`, `course_type`, `start_time`, `end_time`, `teacher_id`, `status`, `term`, `student_count`) VALUES
('Java编程基础', 'Java编程语言入门课程', 3.0, '必修课', '2024-09-01 08:00:00', '2024-12-31 17:00:00', 2025000, '未开始', '2024-2025-1', 0),
('高等数学', '高等数学基础课程', 4.0, '必修课', '2024-09-01 08:00:00', '2024-12-31 17:00:00', 2025001, '未开始', '2024-2025-1', 0),
('人工智能导论', '人工智能基础理论与应用', 3.0, '选修课', '2024-09-01 08:00:00', '2024-12-31 17:00:00', 2025000, '未开始', '2024-2025-1', 0); 