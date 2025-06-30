-- 创建课程表
CREATE TABLE `course` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '课程ID',
  `title` varchar(100) NOT NULL COMMENT '课程名称',
  `description` text COMMENT '课程简介',
  `cover_image` varchar(255) DEFAULT NULL COMMENT '课程封面图片URL',
  `credit` decimal(3,1) NOT NULL DEFAULT '0.0' COMMENT '课程学分',
  `course_type` ENUM('必修课', '选修课') NOT NULL DEFAULT '必修课' COMMENT '课程类型',
  `start_time` datetime NOT NULL COMMENT '课程开始时间',
  `end_time` datetime NOT NULL COMMENT '课程结束时间',
  `teacher_id` bigint NOT NULL COMMENT '教师ID',
  `status` ENUM('未开始', '进行中', '已结束') NOT NULL DEFAULT '未开始' COMMENT '课程状态',
  `term` varchar(20) NOT NULL COMMENT '学期，例如2024-2025-1',
  `student_count` int DEFAULT 0 COMMENT '选课学生数量',
  `average_score` decimal(5,2) DEFAULT NULL COMMENT '平均成绩',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  KEY `idx_teacher_id` (`teacher_id`),
  KEY `idx_term` (`term`),
  KEY `idx_status` (`status`),
  KEY `idx_course_type` (`course_type`),
  CONSTRAINT `fk_course_teacher` FOREIGN KEY (`teacher_id`) REFERENCES `teacher` (`id`) ON DELETE RESTRICT ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='课程表';

-- 添加测试数据
INSERT INTO `course` (`title`, `description`, `cover_image`, `credit`, `course_type`, `start_time`, `end_time`, `teacher_id`, `status`, `term`, `student_count`, `average_score`)
VALUES 
('Java编程基础', '本课程介绍Java编程语言的基础知识和应用', '/images/courses/java-basics.jpg', 4.0, '必修课', '2024-09-01 00:00:00', '2025-01-15 00:00:00', 1, '未开始', '2024-2025-1', 156, 85.2),
('数据结构与算法', '介绍常见数据结构和算法设计方法', '/images/courses/data-structures.jpg', 3.5, '必修课', '2024-09-01 00:00:00', '2025-01-15 00:00:00', 1, '未开始', '2024-2025-1', 128, 78.5),
('Web前端开发', '学习HTML、CSS和JavaScript等前端技术', '/images/courses/web-frontend.jpg', 3.0, '选修课', '2024-09-01 00:00:00', '2025-01-15 00:00:00', 2, '未开始', '2024-2025-1', 95, 82.1),
('数据库系统', '关系型数据库理论与实践', '/images/courses/database.jpg', 4.0, '必修课', '2025-02-20 00:00:00', '2025-06-30 00:00:00', 2, '未开始', '2024-2025-2', 0, NULL),
('软件工程', '软件开发流程与项目管理', '/images/courses/software-engineering.jpg', 3.0, '必修课', '2025-02-20 00:00:00', '2025-06-30 00:00:00', 1, '未开始', '2024-2025-2', 0, NULL); 