-- =============================================
-- ID生成方式修改脚本
-- 将雪花算法生成的ID改为自增ID
-- 用户表从1开始自增
-- 教师表从2025000开始自增
-- 学生表从20250000开始自增
-- 课程表从1开始自增
-- =============================================

USE education_platform;

-- 备份现有表
CREATE TABLE IF NOT EXISTS `user_backup` LIKE `user`;
INSERT INTO `user_backup` SELECT * FROM `user`;

CREATE TABLE IF NOT EXISTS `teacher_backup` LIKE `teacher`;
INSERT INTO `teacher_backup` SELECT * FROM `teacher`;

CREATE TABLE IF NOT EXISTS `student_backup` LIKE `student`;
INSERT INTO `student_backup` SELECT * FROM `student`;

CREATE TABLE IF NOT EXISTS `course_backup` LIKE `course`;
INSERT INTO `course_backup` SELECT * FROM `course`;

-- 修改用户表
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

-- 修改教师表
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
  KEY `idx_department` (`department`)
) ENGINE=InnoDB AUTO_INCREMENT=2025000 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='教师表';

-- 修改学生表
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
  UNIQUE KEY `uk_student_id` (`student_id`)
) ENGINE=InnoDB AUTO_INCREMENT=20250000 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='学生表';

-- 修改课程表
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
  KEY `idx_status` (`status`)
) ENGINE=InnoDB AUTO_INCREMENT=1 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='课程表';

-- 恢复数据（需要手动执行，确保ID映射正确）
-- 注意：需要先创建用户，然后创建教师和学生，最后创建课程
-- INSERT INTO `user` (username, password, email, real_name, avatar, role, status, create_time, update_time)
-- SELECT username, password, email, real_name, avatar, role, status, create_time, update_time FROM `user_backup`;

-- INSERT INTO `teacher` (user_id, department, title, education, specialty, introduction, office_location, office_hours, contact_email, contact_phone, status, hire_date, create_time, update_time)
-- SELECT user_id, department, title, education, specialty, introduction, office_location, office_hours, contact_email, contact_phone, status, hire_date, create_time, update_time FROM `teacher_backup`;

-- INSERT INTO `student` (user_id, student_id, enrollment_status, gpa, gpa_level, create_time, update_time)
-- SELECT user_id, student_id, enrollment_status, gpa, gpa_level, create_time, update_time FROM `student_backup`;

-- INSERT INTO `course` (title, description, cover_image, credit, course_type, start_time, end_time, teacher_id, status, term, student_count, average_score, create_time, update_time)
-- SELECT title, description, cover_image, credit, course_type, start_time, end_time, teacher_id, status, term, student_count, average_score, create_time, update_time FROM `course_backup`; 