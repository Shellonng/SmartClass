-- =============================================
-- AI赋能教育管理与学习辅助平台 - 修复版数据库初始化脚本
-- 版本: 2.1
-- 创建时间: 2024
-- 数据库: MySQL 8.0+
-- 说明: 移除了ext_field1、ext_field2、ext_field3、deleted字段
-- =============================================

-- 创建数据库
CREATE DATABASE IF NOT EXISTS education_platform DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE education_platform;

-- =============================================
-- 1. 用户表 (user) - 移除扩展字段和deleted字段
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
-- 2. 教师表 (teacher) - 移除扩展字段和deleted字段
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
-- 3. 学生表 (student) - 移除扩展字段和deleted字段
-- =============================================
DROP TABLE IF EXISTS `student`;
CREATE TABLE `student` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '学生ID',
  `user_id` bigint NOT NULL COMMENT '用户ID',
  `class_id` bigint DEFAULT NULL COMMENT '班级ID',
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
  KEY `idx_class_id` (`class_id`),
  KEY `idx_major` (`major`),
  KEY `idx_grade` (`grade`),
  CONSTRAINT `fk_student_user` FOREIGN KEY (`user_id`) REFERENCES `user` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='学生表';

-- =============================================
-- 4. 班级表 (class) - 移除扩展字段和deleted字段
-- =============================================
DROP TABLE IF EXISTS `class`;
CREATE TABLE `class` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '班级ID',
  `class_name` varchar(100) NOT NULL COMMENT '班级名称',
  `class_code` varchar(20) NOT NULL COMMENT '班级代码',
  `teacher_id` bigint NOT NULL COMMENT '班主任ID',
  `major` varchar(100) DEFAULT NULL COMMENT '专业',
  `grade` varchar(10) DEFAULT NULL COMMENT '年级',
  `semester` varchar(20) DEFAULT NULL COMMENT '学期',
  `student_count` int DEFAULT '0' COMMENT '学生人数',
  `max_student_count` int DEFAULT '50' COMMENT '最大学生人数',
  `description` text COMMENT '班级描述',
  `status` tinyint DEFAULT '1' COMMENT '状态(0:禁用,1:启用)',
  `created_time` datetime DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `updated_time` datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_class_code` (`class_code`),
  KEY `idx_teacher_id` (`teacher_id`),
  KEY `idx_major_grade` (`major`,`grade`),
  KEY `idx_semester` (`semester`),
  CONSTRAINT `fk_class_teacher` FOREIGN KEY (`teacher_id`) REFERENCES `teacher` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='班级表';

-- 更新学生表外键约束
ALTER TABLE `student` ADD CONSTRAINT `fk_student_class` FOREIGN KEY (`class_id`) REFERENCES `class` (`id`);

-- =============================================
-- 5. 课程表 (course) - 移除扩展字段，保留is_deleted字段（因为course表中使用的是is_deleted）
-- =============================================
DROP TABLE IF EXISTS `course`;
CREATE TABLE `course` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '课程ID',
  `course_name` varchar(100) NOT NULL COMMENT '课程名称',
  `course_code` varchar(20) DEFAULT NULL COMMENT '课程代码',
  `teacher_id` bigint NOT NULL COMMENT '授课教师ID',
  `class_id` bigint DEFAULT NULL COMMENT '班级ID',
  `credits` decimal(3,1) DEFAULT NULL COMMENT '学分',
  `course_type` enum('REQUIRED','ELECTIVE','PUBLIC') DEFAULT 'REQUIRED' COMMENT '课程类型',
  `category` varchar(50) DEFAULT NULL COMMENT '课程分类',
  `difficulty` varchar(20) DEFAULT NULL COMMENT '难度等级',
  `cover_image` varchar(500) DEFAULT NULL COMMENT '课程封面图片',
  `semester` varchar(20) DEFAULT NULL COMMENT '学期',
  `start_date` date DEFAULT NULL COMMENT '开始日期',
  `end_date` date DEFAULT NULL COMMENT '结束日期',
  `start_time` datetime DEFAULT NULL COMMENT '开始时间',
  `end_time` datetime DEFAULT NULL COMMENT '结束时间',
  `schedule` text COMMENT '课程安排(JSON格式)',
  `description` text COMMENT '课程描述',
  `objectives` text COMMENT '课程目标',
  `requirements` text COMMENT '课程要求',
  `status` enum('DRAFT','PUBLISHED','ARCHIVED') DEFAULT 'DRAFT' COMMENT '课程状态',
  `is_public` tinyint DEFAULT '1' COMMENT '是否公开',
  `current_enrollment` int DEFAULT '0' COMMENT '当前选课人数',
  `max_enrollment` int DEFAULT '100' COMMENT '最大选课人数',
  `create_time` datetime DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_course_code` (`course_code`),
  KEY `idx_teacher_id` (`teacher_id`),
  KEY `idx_class_id` (`class_id`),
  KEY `idx_semester` (`semester`),
  KEY `idx_course_type` (`course_type`),
  KEY `idx_status` (`status`),
  CONSTRAINT `fk_course_teacher` FOREIGN KEY (`teacher_id`) REFERENCES `teacher` (`id`),
  CONSTRAINT `fk_course_class` FOREIGN KEY (`class_id`) REFERENCES `class` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='课程表';

-- =============================================
-- 6. 任务表 (task) - 移除扩展字段和deleted字段
-- =============================================
DROP TABLE IF EXISTS `task`;
CREATE TABLE `task` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '任务ID',
  `task_title` varchar(200) NOT NULL COMMENT '任务标题',
  `task_type` enum('HOMEWORK','EXAM','PROJECT','QUIZ') NOT NULL COMMENT '任务类型',
  `course_id` bigint NOT NULL COMMENT '课程ID',
  `teacher_id` bigint NOT NULL COMMENT '发布教师ID',
  `description` text COMMENT '任务描述',
  `requirements` text COMMENT '任务要求',
  `total_score` decimal(5,2) DEFAULT '100.00' COMMENT '总分',
  `start_time` datetime DEFAULT NULL COMMENT '开始时间',
  `end_time` datetime DEFAULT NULL COMMENT '截止时间',
  `submit_type` enum('TEXT','FILE','BOTH') DEFAULT 'BOTH' COMMENT '提交方式',
  `allow_late_submit` tinyint DEFAULT '0' COMMENT '是否允许迟交',
  `late_penalty` decimal(3,2) DEFAULT '0.00' COMMENT '迟交扣分比例',
  `max_attempts` int DEFAULT '1' COMMENT '最大提交次数',
  `auto_grade` tinyint DEFAULT '0' COMMENT '是否自动批改',
  `knowledge_points` text COMMENT '关联知识点(JSON格式)',
  `status` enum('DRAFT','PUBLISHED','CLOSED') DEFAULT 'DRAFT' COMMENT '状态',
  `created_time` datetime DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `updated_time` datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  KEY `idx_course_id` (`course_id`),
  KEY `idx_teacher_id` (`teacher_id`),
  KEY `idx_task_type` (`task_type`),
  KEY `idx_status` (`status`),
  KEY `idx_end_time` (`end_time`),
  CONSTRAINT `fk_task_course` FOREIGN KEY (`course_id`) REFERENCES `course` (`id`),
  CONSTRAINT `fk_task_teacher` FOREIGN KEY (`teacher_id`) REFERENCES `teacher` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='任务表';

-- =============================================
-- 7. 任务提交表 (task_submission) - 移除扩展字段和deleted字段
-- =============================================
DROP TABLE IF EXISTS `task_submission`;
CREATE TABLE `task_submission` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '提交ID',
  `task_id` bigint NOT NULL COMMENT '任务ID',
  `student_id` bigint NOT NULL COMMENT '学生ID',
  `content` text COMMENT '提交内容',
  `files` text COMMENT '提交文件(JSON格式)',
  `links` text COMMENT '提交链接(JSON格式)',
  `submit_time` datetime DEFAULT CURRENT_TIMESTAMP COMMENT '提交时间',
  `is_late` tinyint DEFAULT '0' COMMENT '是否迟交',
  `late_days` int DEFAULT '0' COMMENT '迟交天数',
  `status` enum('SUBMITTED','GRADED','RETURNED') DEFAULT 'SUBMITTED' COMMENT '状态',
  `score` decimal(5,2) DEFAULT NULL COMMENT '得分',
  `original_score` decimal(5,2) DEFAULT NULL COMMENT '原始分数',
  `deduction` decimal(5,2) DEFAULT '0.00' COMMENT '扣分',
  `grader_id` bigint DEFAULT NULL COMMENT '批改教师ID',
  `grade_time` datetime DEFAULT NULL COMMENT '批改时间',
  `feedback` text COMMENT '教师反馈',
  `grading_details` text COMMENT '批改详情(JSON格式)',
  `attempt_number` int DEFAULT '1' COMMENT '提交次数',
  `is_final` tinyint DEFAULT '1' COMMENT '是否最终提交',
  `submission_method` varchar(50) DEFAULT 'ONLINE' COMMENT '提交方式',
  `ip_address` varchar(50) DEFAULT NULL COMMENT '提交IP地址',
  `user_agent` varchar(500) DEFAULT NULL COMMENT '用户代理',
  `file_size` bigint DEFAULT NULL COMMENT '文件大小(字节)',
  `file_type` varchar(100) DEFAULT NULL COMMENT '文件类型',
  `plagiarism_result` text COMMENT '查重结果(JSON格式)',
  `similarity_percentage` decimal(5,2) DEFAULT NULL COMMENT '相似度百分比',
  `plagiarism_passed` tinyint DEFAULT '1' COMMENT '查重是否通过',
  `remarks` text COMMENT '备注',
  `create_time` datetime DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_task_student_attempt` (`task_id`,`student_id`,`attempt_number`),
  KEY `idx_student_id` (`student_id`),
  KEY `idx_status` (`status`),
  KEY `idx_submit_time` (`submit_time`),
  CONSTRAINT `fk_task_submission_task` FOREIGN KEY (`task_id`) REFERENCES `task` (`id`),
  CONSTRAINT `fk_task_submission_student` FOREIGN KEY (`student_id`) REFERENCES `student` (`id`),
  CONSTRAINT `fk_task_submission_grader` FOREIGN KEY (`grader_id`) REFERENCES `teacher` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='任务提交表';

-- =============================================
-- 8. 成绩表 (grade) - 移除扩展字段和deleted字段
-- =============================================
DROP TABLE IF EXISTS `grade`;
CREATE TABLE `grade` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '成绩ID',
  `student_id` bigint NOT NULL COMMENT '学生ID',
  `course_id` bigint NOT NULL COMMENT '课程ID',
  `task_id` bigint DEFAULT NULL COMMENT '任务ID(可为空，表示课程总成绩)',
  `grade_type` enum('TASK','MIDTERM','FINAL','TOTAL') NOT NULL COMMENT '成绩类型',
  `score` decimal(5,2) NOT NULL COMMENT '分数',
  `total_score` decimal(5,2) NOT NULL COMMENT '总分',
  `percentage` decimal(5,2) GENERATED ALWAYS AS ((`score` / `total_score`) * 100) STORED COMMENT '百分比',
  `grade_level` varchar(10) DEFAULT NULL COMMENT '等级(A+,A,B+,B,C+,C,D,F)',
  `rank_in_class` int DEFAULT NULL COMMENT '班级排名',
  `rank_in_course` int DEFAULT NULL COMMENT '课程排名',
  `weight` decimal(3,2) DEFAULT '1.00' COMMENT '权重',
  `is_final` tinyint DEFAULT '0' COMMENT '是否为最终成绩',
  `remarks` text COMMENT '备注',
  `graded_by` bigint DEFAULT NULL COMMENT '录入教师ID',
  `grade_time` datetime DEFAULT CURRENT_TIMESTAMP COMMENT '录入时间',
  `created_time` datetime DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `updated_time` datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_student_course_task_type` (`student_id`,`course_id`,`task_id`,`grade_type`),
  KEY `idx_course_id` (`course_id`),
  KEY `idx_task_id` (`task_id`),
  KEY `idx_grade_type` (`grade_type`),
  KEY `idx_grade_time` (`grade_time`),
  CONSTRAINT `fk_grade_student` FOREIGN KEY (`student_id`) REFERENCES `student` (`id`),
  CONSTRAINT `fk_grade_course` FOREIGN KEY (`course_id`) REFERENCES `course` (`id`),
  CONSTRAINT `fk_grade_task` FOREIGN KEY (`task_id`) REFERENCES `task` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='成绩表';

-- =============================================
-- 9. 资源表 (resource) - 移除扩展字段和deleted字段
-- =============================================
DROP TABLE IF EXISTS `resource`;
CREATE TABLE `resource` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '资源ID',
  `resource_name` varchar(200) NOT NULL COMMENT '资源名称',
  `resource_type` enum('DOCUMENT','VIDEO','AUDIO','IMAGE','OTHER') NOT NULL COMMENT '资源类型',
  `file_name` varchar(255) NOT NULL COMMENT '文件名',
  `file_path` varchar(500) NOT NULL COMMENT '文件路径',
  `file_size` bigint DEFAULT NULL COMMENT '文件大小(字节)',
  `file_extension` varchar(10) DEFAULT NULL COMMENT '文件扩展名',
  `mime_type` varchar(100) DEFAULT NULL COMMENT 'MIME类型',
  `course_id` bigint DEFAULT NULL COMMENT '关联课程ID',
  `task_id` bigint DEFAULT NULL COMMENT '关联任务ID',
  `uploaded_by` bigint NOT NULL COMMENT '上传者ID',
  `is_public` tinyint DEFAULT '1' COMMENT '是否公开',
  `description` text COMMENT '资源描述',
  `tags` varchar(500) DEFAULT NULL COMMENT '标签(逗号分隔)',
  `status` tinyint DEFAULT '1' COMMENT '状态(0:禁用,1:启用)',
  `created_time` datetime DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `updated_time` datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  KEY `idx_course_id` (`course_id`),
  KEY `idx_task_id` (`task_id`),
  KEY `idx_uploaded_by` (`uploaded_by`),
  KEY `idx_resource_type` (`resource_type`),
  KEY `idx_created_time` (`created_time`),
  CONSTRAINT `fk_resource_course` FOREIGN KEY (`course_id`) REFERENCES `course` (`id`),
  CONSTRAINT `fk_resource_task` FOREIGN KEY (`task_id`) REFERENCES `task` (`id`),
  CONSTRAINT `fk_resource_uploader` FOREIGN KEY (`uploaded_by`) REFERENCES `user` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='资源表';

-- =============================================
-- 10. 知识图谱表 (knowledge_graph) - 移除扩展字段和deleted字段
-- =============================================
DROP TABLE IF EXISTS `knowledge_graph`;
CREATE TABLE `knowledge_graph` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '节点ID',
  `node_name` varchar(100) NOT NULL COMMENT '节点名称',
  `node_type` enum('CONCEPT','SKILL','TOPIC','CHAPTER') NOT NULL COMMENT '节点类型',
  `course_id` bigint DEFAULT NULL COMMENT '关联课程ID',
  `parent_id` bigint DEFAULT NULL COMMENT '父节点ID',
  `level` int DEFAULT '1' COMMENT '层级',
  `description` text COMMENT '节点描述',
  `keywords` varchar(500) DEFAULT NULL COMMENT '关键词(逗号分隔)',
  `difficulty` enum('EASY','MEDIUM','HARD') DEFAULT 'MEDIUM' COMMENT '难度等级',
  `importance` decimal(3,2) DEFAULT '1.00' COMMENT '重要性权重',
  `learning_time` int DEFAULT NULL COMMENT '预计学习时间(分钟)',
  `prerequisites` text COMMENT '前置知识点(JSON格式)',
  `related_resources` text COMMENT '相关资源(JSON格式)',
  `ai_generated` tinyint DEFAULT '0' COMMENT '是否AI生成',
  `ai_confidence` decimal(3,2) DEFAULT NULL COMMENT 'AI置信度',
  `status` tinyint DEFAULT '1' COMMENT '状态(0:禁用,1:启用)',
  `created_time` datetime DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `updated_time` datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  KEY `idx_course_id` (`course_id`),
  KEY `idx_parent_id` (`parent_id`),
  KEY `idx_node_type` (`node_type`),
  KEY `idx_level` (`level`),
  CONSTRAINT `fk_knowledge_course` FOREIGN KEY (`course_id`) REFERENCES `course` (`id`),
  CONSTRAINT `fk_knowledge_parent` FOREIGN KEY (`parent_id`) REFERENCES `knowledge_graph` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='知识图谱表';

-- =============================================
-- 11. 题库表 (question_bank) - 移除扩展字段和deleted字段
-- =============================================
DROP TABLE IF EXISTS `question_bank`;
CREATE TABLE `question_bank` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '题目ID',
  `question_text` text NOT NULL COMMENT '题目内容',
  `question_type` enum('SINGLE_CHOICE','MULTIPLE_CHOICE','TRUE_FALSE','FILL_BLANK','ESSAY','CODING') NOT NULL COMMENT '题目类型',
  `course_id` bigint DEFAULT NULL COMMENT '关联课程ID',
  `knowledge_point_id` bigint DEFAULT NULL COMMENT '关联知识点ID',
  `difficulty` enum('EASY','MEDIUM','HARD') DEFAULT 'MEDIUM' COMMENT '难度等级',
  `options` text COMMENT '选项(JSON格式)',
  `correct_answer` text COMMENT '正确答案',
  `explanation` text COMMENT '答案解析',
  `score` decimal(5,2) DEFAULT '1.00' COMMENT '分值',
  `tags` varchar(500) DEFAULT NULL COMMENT '标签(逗号分隔)',
  `usage_count` int DEFAULT '0' COMMENT '使用次数',
  `correct_rate` decimal(5,2) DEFAULT NULL COMMENT '正确率',
  `created_by` bigint NOT NULL COMMENT '创建者ID',
  `ai_generated` tinyint DEFAULT '0' COMMENT '是否AI生成',
  `ai_quality_score` decimal(3,2) DEFAULT NULL COMMENT 'AI质量评分',
  `status` tinyint DEFAULT '1' COMMENT '状态(0:禁用,1:启用)',
  `created_time` datetime DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `updated_time` datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  KEY `idx_course_id` (`course_id`),
  KEY `idx_knowledge_point_id` (`knowledge_point_id`),
  KEY `idx_question_type` (`question_type`),
  KEY `idx_difficulty` (`difficulty`),
  KEY `idx_created_by` (`created_by`),
  CONSTRAINT `fk_question_course` FOREIGN KEY (`course_id`) REFERENCES `course` (`id`),
  CONSTRAINT `fk_question_knowledge` FOREIGN KEY (`knowledge_point_id`) REFERENCES `knowledge_graph` (`id`),
  CONSTRAINT `fk_question_creator` FOREIGN KEY (`created_by`) REFERENCES `user` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='题库表';

-- =============================================
-- 12. AI创新功能表 (ai_feature) - 移除扩展字段和deleted字段
-- =============================================
DROP TABLE IF EXISTS `ai_feature`;
CREATE TABLE `ai_feature` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT 'AI功能ID',
  `feature_name` varchar(100) NOT NULL COMMENT '功能名称',
  `feature_type` enum('KNOWLEDGE_EXTRACTION','AUTO_GRADING','CONTENT_RECOMMENDATION','ABILITY_ANALYSIS','PLAGIARISM_CHECK') NOT NULL COMMENT '功能类型',
  `description` text COMMENT '功能描述',
  `target_entity_type` varchar(50) COMMENT '目标实体类型(task/student/course等)',
  `target_entity_id` bigint COMMENT '目标实体ID',
  `input_data` text COMMENT '输入数据(JSON格式)',
  `output_data` text COMMENT '输出数据(JSON格式)',
  `confidence_score` decimal(5,4) DEFAULT NULL COMMENT '置信度分数',
  `processing_status` enum('PENDING','PROCESSING','COMPLETED','FAILED') DEFAULT 'PENDING' COMMENT '处理状态',
  `error_message` text COMMENT '错误信息',
  `processing_time` int DEFAULT NULL COMMENT '处理时间(毫秒)',
  `created_by` bigint NOT NULL COMMENT '创建者ID',
  `created_time` datetime DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `updated_time` datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  KEY `idx_feature_type` (`feature_type`),
  KEY `idx_target_entity` (`target_entity_type`,`target_entity_id`),
  KEY `idx_created_by` (`created_by`),
  KEY `idx_processing_status` (`processing_status`),
  KEY `idx_created_time` (`created_time`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='AI创新功能表';

-- =============================================
-- 13. 通知表 (notification) - 移除扩展字段和deleted字段
-- =============================================
DROP TABLE IF EXISTS `notification`;
CREATE TABLE `notification` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '通知ID',
  `title` varchar(200) NOT NULL COMMENT '通知标题',
  `content` text NOT NULL COMMENT '通知内容',
  `type` enum('SYSTEM','TASK','GRADE','COURSE','ANNOUNCEMENT') NOT NULL DEFAULT 'SYSTEM' COMMENT '通知类型',
  `priority` enum('LOW','NORMAL','HIGH','URGENT') DEFAULT 'NORMAL' COMMENT '优先级',
  `sender_id` bigint DEFAULT NULL COMMENT '发送者ID',
  `sender_type` enum('SYSTEM','TEACHER','STUDENT','ADMIN') DEFAULT 'SYSTEM' COMMENT '发送者类型',
  `recipient_id` bigint NOT NULL COMMENT '接收者ID',
  `recipient_type` enum('TEACHER','STUDENT','ALL') NOT NULL COMMENT '接收者类型',
  `related_entity_type` varchar(50) DEFAULT NULL COMMENT '关联实体类型',
  `related_entity_id` bigint DEFAULT NULL COMMENT '关联实体ID',
  `is_read` tinyint DEFAULT '0' COMMENT '是否已读(0:未读,1:已读)',
  `read_time` datetime DEFAULT NULL COMMENT '阅读时间',
  `is_sent` tinyint DEFAULT '0' COMMENT '是否已发送(0:未发送,1:已发送)',
  `send_time` datetime DEFAULT NULL COMMENT '发送时间',
  `scheduled_time` datetime DEFAULT NULL COMMENT '计划发送时间',
  `expire_time` datetime DEFAULT NULL COMMENT '过期时间',
  `action_url` varchar(500) DEFAULT NULL COMMENT '操作链接',
  `action_text` varchar(100) DEFAULT NULL COMMENT '操作按钮文本',
  `extra_data` text COMMENT '额外数据(JSON格式)',
  `create_time` datetime DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  KEY `idx_recipient` (`recipient_id`,`recipient_type`),
  KEY `idx_type` (`type`),
  KEY `idx_priority` (`priority`),
  KEY `idx_is_read` (`is_read`),
  KEY `idx_send_time` (`send_time`),
  KEY `idx_create_time` (`create_time`),
  KEY `idx_related_entity` (`related_entity_type`,`related_entity_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='通知表';

-- =============================================
-- 14. 学生课程关联表 (student_course)
-- =============================================
DROP TABLE IF EXISTS `student_course`;
CREATE TABLE `student_course` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '关联ID',
  `student_id` bigint NOT NULL COMMENT '学生ID',
  `course_id` bigint NOT NULL COMMENT '课程ID',
  `enrollment_time` datetime DEFAULT CURRENT_TIMESTAMP COMMENT '选课时间',
  `status` enum('ENROLLED','DROPPED','COMPLETED') DEFAULT 'ENROLLED' COMMENT '选课状态',
  `final_grade` decimal(5,2) DEFAULT NULL COMMENT '最终成绩',
  `grade_level` varchar(10) DEFAULT NULL COMMENT '等级',
  `credits_earned` decimal(3,1) DEFAULT NULL COMMENT '获得学分',
  `created_time` datetime DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `updated_time` datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_student_course` (`student_id`,`course_id`),
  KEY `idx_course_id` (`course_id`),
  KEY `idx_status` (`status`),
  CONSTRAINT `fk_sc_student` FOREIGN KEY (`student_id`) REFERENCES `student` (`id`),
  CONSTRAINT `fk_sc_course` FOREIGN KEY (`course_id`) REFERENCES `course` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='学生课程关联表';

-- =============================================
-- 复合索引优化
-- =============================================

-- 成绩查询优化
ALTER TABLE grade ADD INDEX idx_student_course_type (student_id, course_id, grade_type);

-- 任务查询优化
ALTER TABLE task ADD INDEX idx_course_status_time (course_id, status, end_time);

-- 提交记录查询优化
ALTER TABLE task_submission ADD INDEX idx_task_student_status (task_id, student_id, status);

-- 资源查询优化
ALTER TABLE resource ADD INDEX idx_course_type_public (course_id, resource_type, is_public);

-- =============================================
-- 全文索引
-- =============================================

-- 课程内容搜索
ALTER TABLE course ADD FULLTEXT INDEX ft_course_content (course_name, description, objectives);

-- 任务内容搜索
ALTER TABLE task ADD FULLTEXT INDEX ft_task_content (task_title, description, requirements);

-- 题库内容搜索
ALTER TABLE question_bank ADD FULLTEXT INDEX ft_question_content (question_text, explanation);

-- =============================================
-- 初始化数据 - 包含测试用户
-- =============================================

-- 插入测试用户 (密码: 123456)
INSERT INTO user (username, password, email, real_name, role, status) VALUES 
('root', '$2a$10$N.zmdr9k7uOCQb376NoUnuTJ8iAt6Z5EHsM8lE9P1.jO8m3.TjZ.y', 'root@education.com', 'Root管理员', 'ADMIN', 1),
('admin', '$2a$10$N.zmdr9k7uOCQb376NoUnuTJ8iAt6Z5EHsM8lE9P1.jO8m3.TjZ.y', 'admin@education.com', '系统管理员', 'ADMIN', 1),
('teacher1', '$2a$10$N.zmdr9k7uOCQb376NoUnuTJ8iAt6Z5EHsM8lE9P1.jO8m3.TjZ.y', 'teacher1@education.com', '张教授', 'TEACHER', 1),
('student1', '$2a$10$N.zmdr9k7uOCQb376NoUnuTJ8iAt6Z5EHsM8lE9P1.jO8m3.TjZ.y', 'student1@education.com', '张三', 'STUDENT', 1);

-- 插入教师信息
INSERT INTO teacher (user_id, department, title, education, specialty, introduction, office_location, office_hours) VALUES
((SELECT id FROM user WHERE username = 'teacher1'), '计算机科学系', '副教授', '博士', 'Java开发,数据库设计', '专注于软件工程和数据库设计', 'A101', '周一至周五 9:00-17:00');

-- 插入学生信息
INSERT INTO student (user_id, major, grade, enrollment_year, graduation_year, status) VALUES
((SELECT id FROM user WHERE username = 'student1'), '计算机科学与技术', '大二', 2023, 2027, 'ACTIVE');

-- 插入班级信息
INSERT INTO class (class_name, class_code, teacher_id, major, grade, semester, description) VALUES
('计科2班', 'CS2024-02', (SELECT t.id FROM teacher t JOIN user u ON t.user_id = u.id WHERE u.username = 'teacher1'), '计算机科学与技术', '大二', '2024-2025上', '计算机科学与技术专业二年级班级');

-- 更新学生班级信息
UPDATE student SET class_id = (SELECT id FROM class WHERE class_code = 'CS2024-02') WHERE user_id = (SELECT id FROM user WHERE username = 'student1');

-- 插入课程信息
INSERT INTO course (course_name, course_code, teacher_id, class_id, credits, course_type, category, difficulty, cover_image, semester, start_date, end_date, start_time, end_time, description, objectives, requirements, status, is_public, current_enrollment, max_enrollment) VALUES
('Java程序设计', 'CS101', (SELECT t.id FROM teacher t JOIN user u ON t.user_id = u.id WHERE u.username = 'teacher1'), (SELECT id FROM class WHERE class_code = 'CS2024-02'), 4.0, 'REQUIRED', '编程语言', 'EASY', 'java_cover.jpg', '2024-2025上', '2024-09-01', '2025-01-15', '2024-09-01 09:00:00', '2025-01-15 11:00:00', 'Java编程语言基础与面向对象编程', '掌握Java基础语法和面向对象编程思想', '需要有一定的编程基础', 'PUBLISHED', 1, 1, 100);

-- 插入学生选课记录
INSERT INTO student_course (student_id, course_id, enrollment_time, status) VALUES
((SELECT s.id FROM student s JOIN user u ON s.user_id = u.id WHERE u.username = 'student1'), (SELECT id FROM course WHERE course_code = 'CS101'), NOW(), 'ENROLLED');

-- =============================================
-- 数据库初始化完成
-- =============================================

SELECT 'Database initialization completed successfully!' as message;
SELECT 'Test users: root/123456, teacher1/123456, student1/123456' as credentials;
SELECT 'Database: education_platform' as database_info;
SELECT 'Character Set: utf8mb4' as charset_info;
SELECT 'Collation: utf8mb4_unicode_ci' as collation_info; 