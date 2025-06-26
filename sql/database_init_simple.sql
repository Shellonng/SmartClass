-- =============================================
-- AI赋能教育管理与学习辅助平台 - 简化数据库初始化脚本
-- 版本: 2.0
-- 创建时间: 2024
-- 数据库: MySQL 8.0+
-- 说明: 仅包含root用户的简化版本
-- =============================================

-- 创建数据库
CREATE DATABASE IF NOT EXISTS education_platform DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE education_platform;

-- =============================================
-- 1. 用户表 (user)
-- =============================================
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
  `deleted` tinyint DEFAULT '0' COMMENT '是否删除(0:否,1:是)',
  `ext_field1` varchar(255) DEFAULT NULL COMMENT '扩展字段1',
  `ext_field2` varchar(255) DEFAULT NULL COMMENT '扩展字段2',
  `ext_field3` text COMMENT '扩展字段3(JSON格式)',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_username` (`username`),
  UNIQUE KEY `uk_email` (`email`),
  KEY `idx_role` (`role`),
  KEY `idx_status` (`status`),
  KEY `idx_created_time` (`created_time`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='用户表';

-- =============================================
-- 2. 教师表 (teacher)
-- =============================================
CREATE TABLE `teacher` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '教师ID',
  `user_id` bigint NOT NULL COMMENT '用户ID',
  `teacher_no` varchar(20) NOT NULL COMMENT '教师工号',
  `department` varchar(100) DEFAULT NULL COMMENT '所属院系',
  `title` varchar(50) DEFAULT NULL COMMENT '职称',
  `education` varchar(50) DEFAULT NULL COMMENT '学历',
  `specialty` varchar(200) DEFAULT NULL COMMENT '专业领域',
  `introduction` text COMMENT '个人简介',
  `office_location` varchar(100) DEFAULT NULL COMMENT '办公地点',
  `office_hours` varchar(200) DEFAULT NULL COMMENT '办公时间',
  `created_time` datetime DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `updated_time` datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  `deleted` tinyint DEFAULT '0' COMMENT '是否删除(0:否,1:是)',
  `ext_field1` varchar(255) DEFAULT NULL COMMENT '扩展字段1',
  `ext_field2` varchar(255) DEFAULT NULL COMMENT '扩展字段2',
  `ext_field3` text COMMENT '扩展字段3(JSON格式)',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_user_id` (`user_id`),
  UNIQUE KEY `uk_teacher_no` (`teacher_no`),
  KEY `idx_department` (`department`),
  CONSTRAINT `fk_teacher_user` FOREIGN KEY (`user_id`) REFERENCES `user` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='教师表';

-- =============================================
-- 3. 学生表 (student)
-- =============================================
CREATE TABLE `student` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '学生ID',
  `user_id` bigint NOT NULL COMMENT '用户ID',
  `student_no` varchar(20) NOT NULL COMMENT '学号',
  `class_id` bigint DEFAULT NULL COMMENT '班级ID',
  `major` varchar(100) DEFAULT NULL COMMENT '专业',
  `grade` varchar(10) DEFAULT NULL COMMENT '年级',
  `enrollment_year` int DEFAULT NULL COMMENT '入学年份',
  `graduation_year` int DEFAULT NULL COMMENT '毕业年份',
  `gpa` decimal(3,2) DEFAULT NULL COMMENT 'GPA',
  `status` enum('ACTIVE','SUSPENDED','GRADUATED','DROPPED') DEFAULT 'ACTIVE' COMMENT '学籍状态',
  `created_time` datetime DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `updated_time` datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  `deleted` tinyint DEFAULT '0' COMMENT '是否删除(0:否,1:是)',
  `ext_field1` varchar(255) DEFAULT NULL COMMENT '扩展字段1',
  `ext_field2` varchar(255) DEFAULT NULL COMMENT '扩展字段2',
  `ext_field3` text COMMENT '扩展字段3(JSON格式)',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_user_id` (`user_id`),
  UNIQUE KEY `uk_student_no` (`student_no`),
  KEY `idx_class_id` (`class_id`),
  KEY `idx_major` (`major`),
  KEY `idx_grade` (`grade`),
  CONSTRAINT `fk_student_user` FOREIGN KEY (`user_id`) REFERENCES `user` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='学生表';

-- =============================================
-- 4. 班级表 (class)
-- =============================================
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
  `deleted` tinyint DEFAULT '0' COMMENT '是否删除(0:否,1:是)',
  `ext_field1` varchar(255) DEFAULT NULL COMMENT '扩展字段1',
  `ext_field2` varchar(255) DEFAULT NULL COMMENT '扩展字段2',
  `ext_field3` text COMMENT '扩展字段3(JSON格式)',
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
-- 5. 课程表 (course)
-- =============================================
CREATE TABLE `course` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '课程ID',
  `course_name` varchar(100) NOT NULL COMMENT '课程名称',
  `course_code` varchar(20) NOT NULL COMMENT '课程代码',
  `teacher_id` bigint NOT NULL COMMENT '授课教师ID',
  `class_id` bigint DEFAULT NULL COMMENT '班级ID',
  `credits` decimal(3,1) DEFAULT NULL COMMENT '学分',
  `course_type` enum('REQUIRED','ELECTIVE','PUBLIC') DEFAULT 'REQUIRED' COMMENT '课程类型',
  `semester` varchar(20) DEFAULT NULL COMMENT '学期',
  `start_date` date DEFAULT NULL COMMENT '开始日期',
  `end_date` date DEFAULT NULL COMMENT '结束日期',
  `schedule` text COMMENT '课程安排(JSON格式)',
  `description` text COMMENT '课程描述',
  `objectives` text COMMENT '课程目标',
  `requirements` text COMMENT '课程要求',
  `status` tinyint DEFAULT '1' COMMENT '状态(0:禁用,1:启用)',
  `created_time` datetime DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `updated_time` datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  `deleted` tinyint DEFAULT '0' COMMENT '是否删除(0:否,1:是)',
  `ext_field1` varchar(255) DEFAULT NULL COMMENT '扩展字段1',
  `ext_field2` varchar(255) DEFAULT NULL COMMENT '扩展字段2',
  `ext_field3` text COMMENT '扩展字段3(JSON格式)',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_course_code` (`course_code`),
  KEY `idx_teacher_id` (`teacher_id`),
  KEY `idx_class_id` (`class_id`),
  KEY `idx_semester` (`semester`),
  KEY `idx_course_type` (`course_type`),
  CONSTRAINT `fk_course_teacher` FOREIGN KEY (`teacher_id`) REFERENCES `teacher` (`id`),
  CONSTRAINT `fk_course_class` FOREIGN KEY (`class_id`) REFERENCES `class` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='课程表';

-- =============================================
-- 6. 任务表 (task)
-- =============================================
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
  `deleted` tinyint DEFAULT '0' COMMENT '是否删除(0:否,1:是)',
  `ext_field1` varchar(255) DEFAULT NULL COMMENT '扩展字段1',
  `ext_field2` varchar(255) DEFAULT NULL COMMENT '扩展字段2',
  `ext_field3` text COMMENT '扩展字段3(JSON格式)',
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
-- 7. 任务提交表 (task_submission)
-- =============================================
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
  `is_deleted` tinyint DEFAULT '0' COMMENT '是否删除(0:否,1:是)',
  `ext_field1` varchar(255) DEFAULT NULL COMMENT '扩展字段1',
  `ext_field2` varchar(255) DEFAULT NULL COMMENT '扩展字段2',
  `ext_field3` text COMMENT '扩展字段3(JSON格式)',
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
-- 8. 成绩表 (grade)
-- =============================================
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
  `deleted` tinyint DEFAULT '0' COMMENT '是否删除(0:否,1:是)',
  `ext_field1` varchar(255) DEFAULT NULL COMMENT '扩展字段1',
  `ext_field2` varchar(255) DEFAULT NULL COMMENT '扩展字段2',
  `ext_field3` text COMMENT '扩展字段3(JSON格式)',
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
-- 9. 资源表 (resource)
-- =============================================
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
  `deleted` tinyint DEFAULT '0' COMMENT '是否删除(0:否,1:是)',
  `ext_field1` varchar(255) DEFAULT NULL COMMENT '扩展字段1',
  `ext_field2` varchar(255) DEFAULT NULL COMMENT '扩展字段2',
  `ext_field3` text COMMENT '扩展字段3(JSON格式)',
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
-- 10. 知识图谱表 (knowledge_graph)
-- =============================================
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
  `deleted` tinyint DEFAULT '0' COMMENT '是否删除(0:否,1:是)',
  `ext_field1` varchar(255) DEFAULT NULL COMMENT '扩展字段1',
  `ext_field2` varchar(255) DEFAULT NULL COMMENT '扩展字段2',
  `ext_field3` text COMMENT '扩展字段3(JSON格式)',
  PRIMARY KEY (`id`),
  KEY `idx_course_id` (`course_id`),
  KEY `idx_parent_id` (`parent_id`),
  KEY `idx_node_type` (`node_type`),
  KEY `idx_level` (`level`),
  CONSTRAINT `fk_knowledge_course` FOREIGN KEY (`course_id`) REFERENCES `course` (`id`),
  CONSTRAINT `fk_knowledge_parent` FOREIGN KEY (`parent_id`) REFERENCES `knowledge_graph` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='知识图谱表';

-- =============================================
-- 11. 题库表 (question_bank)
-- =============================================
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
  `deleted` tinyint DEFAULT '0' COMMENT '是否删除(0:否,1:是)',
  `ext_field1` varchar(255) DEFAULT NULL COMMENT '扩展字段1',
  `ext_field2` varchar(255) DEFAULT NULL COMMENT '扩展字段2',
  `ext_field3` text COMMENT '扩展字段3(JSON格式)',
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
-- 12. AI创新功能表 (ai_feature)
-- =============================================
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
  `deleted` tinyint DEFAULT '0' COMMENT '是否删除(0:否,1:是)',
  `ext_field1` varchar(255) DEFAULT NULL COMMENT '扩展字段1',
  `ext_field2` varchar(255) DEFAULT NULL COMMENT '扩展字段2',
  `ext_field3` text COMMENT '扩展字段3(JSON格式)',
  PRIMARY KEY (`id`),
  KEY `idx_feature_type` (`feature_type`),
  KEY `idx_target_entity` (`target_entity_type`,`target_entity_id`),
  KEY `idx_created_by` (`created_by`),
  KEY `idx_processing_status` (`processing_status`),
  KEY `idx_created_time` (`created_time`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='AI创新功能表';

-- =============================================
-- 13. 通知表 (notification)
-- =============================================
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
  `is_deleted` tinyint DEFAULT '0' COMMENT '是否删除(0:否,1:是)',
  `ext_field1` varchar(255) DEFAULT NULL COMMENT '扩展字段1',
  `ext_field2` varchar(255) DEFAULT NULL COMMENT '扩展字段2',
  `ext_field3` text COMMENT '扩展字段3(JSON格式)',
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
-- 初始化数据 - 仅包含root用户
-- =============================================

-- 插入root管理员用户 (密码: root123)
INSERT INTO user (username, password, email, real_name, role, status) VALUES 
('root', 'root123', 'root@education.com', 'Root管理员', 'ADMIN', 1);

-- 插入更多测试用户 (所有密码均为明文)
INSERT INTO user (username, password, email, real_name, role, status) VALUES
('admin', 'admin123', 'admin@education.com', '系统管理员', 'ADMIN', 1),
('teacher1', 'teacher123', 'teacher1@education.com', '张教授', 'TEACHER', 1),
('teacher2', 'teacher123', 'teacher2@education.com', '李教授', 'TEACHER', 1),
('teacher3', 'teacher123', 'teacher3@education.com', '王老师', 'TEACHER', 1),
('teacher4', 'teacher123', 'teacher4@education.com', '陈教授', 'TEACHER', 1),
('teacher5', 'teacher123', 'teacher5@education.com', '刘老师', 'TEACHER', 1),
('student1', 'student123', 'student1@education.com', '张三', 'STUDENT', 1),
('student2', 'student123', 'student2@education.com', '李四', 'STUDENT', 1),
('student3', 'student123', 'student3@education.com', '王五', 'STUDENT', 1),
('student4', 'student123', 'student4@education.com', '赵六', 'STUDENT', 1),
('student5', 'student123', 'student5@education.com', '陈七', 'STUDENT', 1),
('student6', 'student123', 'student6@education.com', '刘八', 'STUDENT', 1),
('student7', 'student123', 'student7@education.com', '周九', 'STUDENT', 1),
('student8', 'student123', 'student8@education.com', '吴十', 'STUDENT', 1),
('student9', 'student123', 'student9@education.com', '郑十一', 'STUDENT', 1),
('student10', 'student123', 'student10@education.com', '孙十二', 'STUDENT', 1),
('student11', 'student123', 'student11@education.com', '钱十三', 'STUDENT', 1),
('student12', 'student123', 'student12@education.com', '孙十四', 'STUDENT', 1),
('student13', 'student123', 'student13@education.com', '李十五', 'STUDENT', 1),
('student14', 'student123', 'student14@education.com', '周十六', 'STUDENT', 1),
('student15', 'student123', 'student15@education.com', '吴十七', 'STUDENT', 1),
('student16', 'student123', 'student16@education.com', '郑十八', 'STUDENT', 1),
('student17', 'student123', 'student17@education.com', '王十九', 'STUDENT', 1),
('student18', 'student123', 'student18@education.com', '冯二十', 'STUDENT', 1),
('student19', 'student123', 'student19@education.com', '陈二一', 'STUDENT', 1),
('student20', 'student123', 'student20@education.com', '褚二二', 'STUDENT', 1);

-- 插入教师信息
INSERT INTO teacher (user_id, teacher_no, department, title, education, specialty, introduction, office_location, office_hours) VALUES
((SELECT id FROM user WHERE username = 'teacher1'), 'T001', '计算机科学系', '副教授', '博士', 'Java开发,数据库设计', '专注于软件工程和数据库设计', 'A101', '周一至周五 9:00-17:00'),
((SELECT id FROM user WHERE username = 'teacher2'), 'T002', '数学系', '教授', '博士', '算法分析,数学建模', '数学建模和算法分析专家', 'B201', '周一至周五 8:00-16:00'),
((SELECT id FROM user WHERE username = 'teacher3'), 'T003', '英语系', '讲师', '硕士', '英语教学,口语训练', '英语教学和跨文化交流', 'C301', '周一至周五 10:00-18:00'),
((SELECT id FROM user WHERE username = 'teacher4'), 'T004', '物理系', '教授', '博士', '理论物理,量子力学', '理论物理和量子力学研究专家', 'D401', '周一至周五 8:30-16:30'),
((SELECT id FROM user WHERE username = 'teacher5'), 'T005', '化学系', '副教授', '博士', '有机化学,材料科学', '有机化学合成和新材料开发', 'E501', '周一至周五 9:30-17:30');

-- 插入学生信息
INSERT INTO student (user_id, student_no, major, grade, enrollment_year, graduation_year, status) VALUES
((SELECT id FROM user WHERE username = 'student1'), 'S2023001', '计算机科学与技术', '大二', 2023, 2027, 'ACTIVE'),
((SELECT id FROM user WHERE username = 'student2'), 'S2023002', '计算机科学与技术', '大二', 2023, 2027, 'ACTIVE'),
((SELECT id FROM user WHERE username = 'student3'), 'S2023003', '软件工程', '大二', 2023, 2027, 'ACTIVE'),
((SELECT id FROM user WHERE username = 'student4'), 'S2023004', '软件工程', '大二', 2023, 2027, 'ACTIVE'),
((SELECT id FROM user WHERE username = 'student5'), 'S2023005', '数据科学与大数据技术', '大一', 2024, 2028, 'ACTIVE'),
((SELECT id FROM user WHERE username = 'student6'), 'S2023006', '数据科学与大数据技术', '大一', 2024, 2028, 'ACTIVE'),
((SELECT id FROM user WHERE username = 'student7'), 'S2023007', '信息安全', '大三', 2022, 2026, 'ACTIVE'),
((SELECT id FROM user WHERE username = 'student8'), 'S2023008', '信息安全', '大三', 2022, 2026, 'ACTIVE'),
((SELECT id FROM user WHERE username = 'student9'), 'S2023009', '人工智能', '大二', 2023, 2027, 'ACTIVE'),
((SELECT id FROM user WHERE username = 'student10'), 'S2023010', '人工智能', '大二', 2023, 2027, 'ACTIVE'),
((SELECT id FROM user WHERE username = 'student11'), 'S2023011', '计算机科学与技术', '大二', 2023, 2027, 'ACTIVE'),
((SELECT id FROM user WHERE username = 'student12'), 'S2023012', '计算机科学与技术', '大二', 2023, 2027, 'ACTIVE'),
((SELECT id FROM user WHERE username = 'student13'), 'S2023013', '软件工程', '大二', 2023, 2027, 'ACTIVE'),
((SELECT id FROM user WHERE username = 'student14'), 'S2023014', '软件工程', '大二', 2023, 2027, 'ACTIVE'),
((SELECT id FROM user WHERE username = 'student15'), 'S2024015', '数据科学与大数据技术', '大一', 2024, 2028, 'ACTIVE'),
((SELECT id FROM user WHERE username = 'student16'), 'S2024016', '数据科学与大数据技术', '大一', 2024, 2028, 'ACTIVE'),
((SELECT id FROM user WHERE username = 'student17'), 'S2022017', '信息安全', '大三', 2022, 2026, 'ACTIVE'),
((SELECT id FROM user WHERE username = 'student18'), 'S2022018', '信息安全', '大三', 2022, 2026, 'ACTIVE'),
((SELECT id FROM user WHERE username = 'student19'), 'S2023019', '人工智能', '大二', 2023, 2027, 'ACTIVE'),
((SELECT id FROM user WHERE username = 'student20'), 'S2023020', '人工智能', '大二', 2023, 2027, 'ACTIVE');

-- 插入班级信息
INSERT INTO class (class_name, class_code, teacher_id, major, grade, semester, description) VALUES
('计科2班', 'CS2024-02', (SELECT id FROM teacher WHERE teacher_no = 'T001'), '计算机科学与技术', '大二', '2024-2025上', '计算机科学与技术专业二年级班级'),
('软工1班', 'SE2024-01', (SELECT id FROM teacher WHERE teacher_no = 'T001'), '软件工程', '大二', '2024-2025上', '软件工程专业二年级班级'),
('数据1班', 'DS2024-01', (SELECT id FROM teacher WHERE teacher_no = 'T002'), '数据科学与大数据技术', '大一', '2024-2025上', '数据科学与大数据技术专业一年级班级'),
('信安1班', 'IS2024-01', (SELECT id FROM teacher WHERE teacher_no = 'T001'), '信息安全', '大三', '2024-2025上', '信息安全专业三年级班级'),
('AI1班', 'AI2024-01', (SELECT id FROM teacher WHERE teacher_no = 'T002'), '人工智能', '大二', '2024-2025上', '人工智能专业二年级班级');

-- 更新学生班级信息
UPDATE student SET class_id = (SELECT id FROM class WHERE class_code = 'CS2024-02') WHERE student_no IN ('S2023001', 'S2023002', 'S2023011', 'S2023012');
UPDATE student SET class_id = (SELECT id FROM class WHERE class_code = 'SE2024-01') WHERE student_no IN ('S2023003', 'S2023004', 'S2023013', 'S2023014');
UPDATE student SET class_id = (SELECT id FROM class WHERE class_code = 'DS2024-01') WHERE student_no IN ('S2023005', 'S2023006', 'S2024015', 'S2024016');
UPDATE student SET class_id = (SELECT id FROM class WHERE class_code = 'IS2024-01') WHERE student_no IN ('S2023007', 'S2023008', 'S2022017', 'S2022018');
UPDATE student SET class_id = (SELECT id FROM class WHERE class_code = 'AI2024-01') WHERE student_no IN ('S2023009', 'S2023010', 'S2023019', 'S2023020');

-- 插入课程信息
INSERT INTO course (course_name, course_code, teacher_id, class_id, credits, course_type, semester, start_date, end_date, description, objectives, requirements) VALUES
('Java程序设计', 'CS101', (SELECT id FROM teacher WHERE teacher_no = 'T001'), (SELECT id FROM class WHERE class_code = 'CS2024-02'), 4.0, 'REQUIRED', '2024-2025上', '2024-09-01', '2025-01-15', 'Java编程语言基础与面向对象编程', '掌握Java基础语法和面向对象编程思想', '需要有一定的编程基础'),
('数据结构与算法', 'CS102', (SELECT id FROM teacher WHERE teacher_no = 'T001'), (SELECT id FROM class WHERE class_code = 'CS2024-02'), 4.0, 'REQUIRED', '2024-2025上', '2024-09-01', '2025-01-15', '数据结构基础和常用算法分析', '掌握基本数据结构和算法设计', '需要掌握至少一门编程语言'),
('数据库系统原理', 'CS201', (SELECT id FROM teacher WHERE teacher_no = 'T001'), (SELECT id FROM class WHERE class_code = 'IS2024-01'), 3.0, 'REQUIRED', '2024-2025上', '2024-09-01', '2025-01-15', '关系数据库理论与SQL实践', '掌握数据库设计和SQL编程', '需要有编程基础'),
('高等数学', 'MATH101', (SELECT id FROM teacher WHERE teacher_no = 'T002'), (SELECT id FROM class WHERE class_code = 'DS2024-01'), 5.0, 'REQUIRED', '2024-2025上', '2024-09-01', '2025-01-15', '微积分和线性代数基础', '掌握高等数学基本概念和计算方法', '高中数学基础'),
('概率论与数理统计', 'MATH201', (SELECT id FROM teacher WHERE teacher_no = 'T002'), (SELECT id FROM class WHERE class_code = 'AI2024-01'), 3.0, 'REQUIRED', '2024-2025上', '2024-09-01', '2025-01-15', '概率论基础和统计学应用', '掌握概率论和统计学基本理论', '需要高等数学基础'),
('软件工程', 'SE301', (SELECT id FROM teacher WHERE teacher_no = 'T001'), (SELECT id FROM class WHERE class_code = 'SE2024-01'), 3.0, 'REQUIRED', '2024-2025上', '2024-09-01', '2025-01-15', '软件开发生命周期和项目管理', '掌握软件工程基本理论和实践方法', '需要编程基础和项目经验'),
('机器学习基础', 'AI201', (SELECT id FROM teacher WHERE teacher_no = 'T002'), (SELECT id FROM class WHERE class_code = 'AI2024-01'), 4.0, 'ELECTIVE', '2024-2025上', '2024-09-01', '2025-01-15', '机器学习算法和应用实践', '掌握常用机器学习算法', '需要数学基础和编程能力');

-- 插入学生选课记录
INSERT INTO student_course (student_id, course_id, enrollment_time, status) VALUES
-- 计科2班学生选课
((SELECT id FROM student WHERE student_no = 'S2023001'), (SELECT id FROM course WHERE course_code = 'CS101'), NOW(), 'ENROLLED'),
((SELECT id FROM student WHERE student_no = 'S2023001'), (SELECT id FROM course WHERE course_code = 'CS102'), NOW(), 'ENROLLED'),
((SELECT id FROM student WHERE student_no = 'S2023001'), (SELECT id FROM course WHERE course_code = 'MATH101'), NOW(), 'ENROLLED'),
((SELECT id FROM student WHERE student_no = 'S2023002'), (SELECT id FROM course WHERE course_code = 'CS101'), NOW(), 'ENROLLED'),
((SELECT id FROM student WHERE student_no = 'S2023002'), (SELECT id FROM course WHERE course_code = 'CS102'), NOW(), 'ENROLLED'),
((SELECT id FROM student WHERE student_no = 'S2023002'), (SELECT id FROM course WHERE course_code = 'MATH101'), NOW(), 'ENROLLED'),
-- 软工1班学生选课
((SELECT id FROM student WHERE student_no = 'S2023003'), (SELECT id FROM course WHERE course_code = 'CS101'), NOW(), 'ENROLLED'),
((SELECT id FROM student WHERE student_no = 'S2023003'), (SELECT id FROM course WHERE course_code = 'SE301'), NOW(), 'ENROLLED'),
((SELECT id FROM student WHERE student_no = 'S2023004'), (SELECT id FROM course WHERE course_code = 'CS101'), NOW(), 'ENROLLED'),
((SELECT id FROM student WHERE student_no = 'S2023004'), (SELECT id FROM course WHERE course_code = 'SE301'), NOW(), 'ENROLLED'),
-- 数据1班学生选课
((SELECT id FROM student WHERE student_no = 'S2023005'), (SELECT id FROM course WHERE course_code = 'MATH101'), NOW(), 'ENROLLED'),
((SELECT id FROM student WHERE student_no = 'S2023006'), (SELECT id FROM course WHERE course_code = 'MATH101'), NOW(), 'ENROLLED'),
-- 信安1班学生选课
((SELECT id FROM student WHERE student_no = 'S2023007'), (SELECT id FROM course WHERE course_code = 'CS201'), NOW(), 'ENROLLED'),
((SELECT id FROM student WHERE student_no = 'S2023007'), (SELECT id FROM course WHERE course_code = 'CS102'), NOW(), 'ENROLLED'),
((SELECT id FROM student WHERE student_no = 'S2023008'), (SELECT id FROM course WHERE course_code = 'CS201'), NOW(), 'ENROLLED'),
((SELECT id FROM student WHERE student_no = 'S2023008'), (SELECT id FROM course WHERE course_code = 'CS102'), NOW(), 'ENROLLED'),
-- AI1班学生选课
((SELECT id FROM student WHERE student_no = 'S2023009'), (SELECT id FROM course WHERE course_code = 'AI201'), NOW(), 'ENROLLED'),
((SELECT id FROM student WHERE student_no = 'S2023009'), (SELECT id FROM course WHERE course_code = 'MATH201'), NOW(), 'ENROLLED'),
((SELECT id FROM student WHERE student_no = 'S2023010'), (SELECT id FROM course WHERE course_code = 'AI201'), NOW(), 'ENROLLED'),
((SELECT id FROM student WHERE student_no = 'S2023010'), (SELECT id FROM course WHERE course_code = 'MATH201'), NOW(), 'ENROLLED');

-- 插入任务信息
INSERT INTO task (task_title, task_type, course_id, teacher_id, description, requirements, total_score, start_time, end_time, status) VALUES
('Java基础练习1', 'HOMEWORK', (SELECT id FROM course WHERE course_code = 'CS101'), (SELECT id FROM teacher WHERE teacher_no = 'T001'), '完成Java基本语法练习，包括变量、循环、条件语句', '提交完整的Java代码文件', 100.00, NOW(), DATE_ADD(NOW(), INTERVAL 7 DAY), 'PUBLISHED'),
('Java面向对象编程', 'PROJECT', (SELECT id FROM course WHERE course_code = 'CS101'), (SELECT id FROM teacher WHERE teacher_no = 'T001'), '设计一个简单的学生管理系统，体现面向对象思想', '包含类设计、继承、多态等概念', 150.00, NOW(), DATE_ADD(NOW(), INTERVAL 14 DAY), 'PUBLISHED'),
('数据结构实现', 'HOMEWORK', (SELECT id FROM course WHERE course_code = 'CS102'), (SELECT id FROM teacher WHERE teacher_no = 'T001'), '用Java实现栈、队列、链表等基本数据结构', '代码规范，注释清晰', 120.00, NOW(), DATE_ADD(NOW(), INTERVAL 10 DAY), 'PUBLISHED'),
('SQL查询练习', 'HOMEWORK', (SELECT id FROM course WHERE course_code = 'CS201'), (SELECT id FROM teacher WHERE teacher_no = 'T001'), '完成复杂SQL查询语句编写，包括多表连接和子查询', '提交SQL文件和执行结果', 80.00, NOW(), DATE_ADD(NOW(), INTERVAL 5 DAY), 'PUBLISHED'),
('微积分应用题', 'HOMEWORK', (SELECT id FROM course WHERE course_code = 'MATH101'), (SELECT id FROM teacher WHERE teacher_no = 'T002'), '解决实际问题中的微积分应用', '详细的解题过程和答案', 100.00, NOW(), DATE_ADD(NOW(), INTERVAL 3 DAY), 'PUBLISHED'),
('软件需求分析', 'PROJECT', (SELECT id FROM course WHERE course_code = 'SE301'), (SELECT id FROM teacher WHERE teacher_no = 'T001'), '为给定项目编写详细的需求分析文档', '包含功能需求和非功能需求', 200.00, NOW(), DATE_ADD(NOW(), INTERVAL 21 DAY), 'PUBLISHED'),
('机器学习算法实现', 'PROJECT', (SELECT id FROM course WHERE course_code = 'AI201'), (SELECT id FROM teacher WHERE teacher_no = 'T002'), '实现并比较不同的机器学习算法', '代码实现和实验报告', 180.00, NOW(), DATE_ADD(NOW(), INTERVAL 28 DAY), 'PUBLISHED');

-- 插入任务提交记录
INSERT INTO task_submission (task_id, student_id, content, files, submit_time, status, score, feedback, grader_id, grade_time) VALUES
-- Java基础练习1的提交
((SELECT id FROM task WHERE task_title = 'Java基础练习1'), (SELECT id FROM student WHERE student_no = 'S2023001'), '已完成所有Java基础语法练习', '[{"name":"java_basic.zip","path":"/uploads/submissions/s2023001_java_basic.zip","size":1024000}]', NOW(), 'GRADED', 95.00, '代码规范良好，逻辑清晰', (SELECT id FROM teacher WHERE teacher_no = 'T001'), NOW()),
((SELECT id FROM task WHERE task_title = 'Java基础练习1'), (SELECT id FROM student WHERE student_no = 'S2023002'), '完成了大部分练习题', '[{"name":"java_basic.zip","path":"/uploads/submissions/s2023002_java_basic.zip","size":896000}]', NOW(), 'GRADED', 88.00, '有几个小错误，整体不错', (SELECT id FROM teacher WHERE teacher_no = 'T001'), NOW()),
-- 数据结构实现的提交
((SELECT id FROM task WHERE task_title = '数据结构实现'), (SELECT id FROM student WHERE student_no = 'S2023001'), '实现了栈、队列、链表的基本操作', '[{"name":"data_structure.zip","path":"/uploads/submissions/s2023001_data_structure.zip","size":1536000}]', NOW(), 'GRADED', 110.00, '实现完整，代码质量高', (SELECT id FROM teacher WHERE teacher_no = 'T001'), NOW()),
((SELECT id FROM task WHERE task_title = '数据结构实现'), (SELECT id FROM student WHERE student_no = 'S2023002'), '实现了栈和队列，链表部分有问题', '[{"name":"data_structure.zip","path":"/uploads/submissions/s2023002_data_structure.zip","size":1280000}]', NOW(), 'GRADED', 85.00, '链表实现需要改进', (SELECT id FROM teacher WHERE teacher_no = 'T001'), NOW()),
-- SQL查询练习的提交
((SELECT id FROM task WHERE task_title = 'SQL查询练习'), (SELECT id FROM student WHERE student_no = 'S2023007'), '完成了所有SQL查询题目', '[{"name":"sql_practice.sql","path":"/uploads/submissions/s2023007_sql_practice.sql","size":51200}]', NOW(), 'GRADED', 75.00, 'SQL语法正确，但查询效率可以优化', (SELECT id FROM teacher WHERE teacher_no = 'T001'), NOW()),
((SELECT id FROM task WHERE task_title = 'SQL查询练习'), (SELECT id FROM student WHERE student_no = 'S2023008'), '完成了大部分查询，有两题未完成', '[{"name":"sql_practice.sql","path":"/uploads/submissions/s2023008_sql_practice.sql","size":40960}]', NOW(), 'GRADED', 65.00, '需要加强复杂查询的练习', (SELECT id FROM teacher WHERE teacher_no = 'T001'), NOW());

-- 插入成绩记录
INSERT INTO grade (student_id, course_id, task_id, grade_type, score, total_score, grade_level, graded_by, grade_time) VALUES
-- CS101课程成绩
((SELECT id FROM student WHERE student_no = 'S2023001'), (SELECT id FROM course WHERE course_code = 'CS101'), (SELECT id FROM task WHERE task_title = 'Java基础练习1'), 'TASK', 95.00, 100.00, 'A', (SELECT id FROM teacher WHERE teacher_no = 'T001'), NOW()),
((SELECT id FROM student WHERE student_no = 'S2023002'), (SELECT id FROM course WHERE course_code = 'CS101'), (SELECT id FROM task WHERE task_title = 'Java基础练习1'), 'TASK', 88.00, 100.00, 'B+', (SELECT id FROM teacher WHERE teacher_no = 'T001'), NOW()),
-- CS102课程成绩
((SELECT id FROM student WHERE student_no = 'S2023001'), (SELECT id FROM course WHERE course_code = 'CS102'), (SELECT id FROM task WHERE task_title = '数据结构实现'), 'TASK', 110.00, 120.00, 'A+', (SELECT id FROM teacher WHERE teacher_no = 'T001'), NOW()),
((SELECT id FROM student WHERE student_no = 'S2023002'), (SELECT id FROM course WHERE course_code = 'CS102'), (SELECT id FROM task WHERE task_title = '数据结构实现'), 'TASK', 85.00, 120.00, 'B', (SELECT id FROM teacher WHERE teacher_no = 'T001'), NOW()),
-- CS201课程成绩
((SELECT id FROM student WHERE student_no = 'S2023007'), (SELECT id FROM course WHERE course_code = 'CS201'), (SELECT id FROM task WHERE task_title = 'SQL查询练习'), 'TASK', 75.00, 80.00, 'A-', (SELECT id FROM teacher WHERE teacher_no = 'T001'), NOW()),
((SELECT id FROM student WHERE student_no = 'S2023008'), (SELECT id FROM course WHERE course_code = 'CS201'), (SELECT id FROM task WHERE task_title = 'SQL查询练习'), 'TASK', 65.00, 80.00, 'B-', (SELECT id FROM teacher WHERE teacher_no = 'T001'), NOW());

-- 插入通知信息
INSERT INTO notification (title, content, type, priority, sender_id, recipient_id, recipient_type, related_entity_type, related_entity_id) VALUES
('欢迎使用AI赋能教育管理平台', '欢迎大家使用AI赋能教育管理与学习辅助平台！请及时查看课程安排和作业通知。', 'SYSTEM', 'NORMAL', (SELECT id FROM user WHERE username = 'root'), (SELECT id FROM user WHERE username = 'student1'), 'STUDENT', NULL, NULL),
('Java程序设计课程开课通知', 'Java程序设计课程将于下周一开始，请同学们做好准备。', 'COURSE', 'HIGH', (SELECT id FROM user WHERE username = 'teacher1'), (SELECT id FROM user WHERE username = 'student1'), 'STUDENT', 'course', (SELECT id FROM course WHERE course_code = 'CS101')),
('数据结构作业提交提醒', '数据结构实现作业截止日期临近，请尚未提交的同学抓紧时间。', 'TASK', 'HIGH', (SELECT id FROM user WHERE username = 'teacher1'), (SELECT id FROM user WHERE username = 'student1'), 'STUDENT', 'task', (SELECT id FROM task WHERE task_title = '数据结构实现')),
('期中考试安排通知', '各科目期中考试时间已确定，请查看详细安排。', 'ANNOUNCEMENT', 'HIGH', (SELECT id FROM user WHERE username = 'admin'), (SELECT id FROM user WHERE username = 'student1'), 'STUDENT', NULL, NULL),
('系统维护通知', '系统将于本周六晚进行维护升级，届时可能无法正常访问。', 'SYSTEM', 'NORMAL', (SELECT id FROM user WHERE username = 'root'), (SELECT id FROM user WHERE username = 'student1'), 'STUDENT', NULL, NULL);

-- 插入资源信息
INSERT INTO resource (resource_name, resource_type, file_name, file_path, file_size, course_id, uploaded_by, description, tags) VALUES
('Java编程基础教程', 'DOCUMENT', 'java_basic_tutorial.pdf', '/resources/java_basic_tutorial.pdf', 2048000, (SELECT id FROM course WHERE course_code = 'CS101'), (SELECT id FROM user WHERE username = 'teacher1'), 'Java语言基础知识点总结', 'Java,编程,基础'),
('Java开发环境配置指南', 'DOCUMENT', 'java_env_setup.pdf', '/resources/java_env_setup.pdf', 1024000, (SELECT id FROM course WHERE course_code = 'CS101'), (SELECT id FROM user WHERE username = 'teacher1'), 'IntelliJ IDEA和JDK安装配置步骤', 'Java,环境配置,IDE'),
('数据结构课件第一章', 'DOCUMENT', 'ds_chapter1.pptx', '/resources/ds_chapter1.pptx', 3072000, (SELECT id FROM course WHERE course_code = 'CS102'), (SELECT id FROM user WHERE username = 'teacher1'), '线性表的基本概念和实现', '数据结构,线性表,课件'),
('算法复杂度分析', 'DOCUMENT', 'algorithm_complexity.pdf', '/resources/algorithm_complexity.pdf', 1536000, (SELECT id FROM course WHERE course_code = 'CS102'), (SELECT id FROM user WHERE username = 'teacher1'), '时间复杂度和空间复杂度的计算方法', '算法,复杂度,分析'),
('MySQL数据库教程', 'DOCUMENT', 'mysql_tutorial.pdf', '/resources/mysql_tutorial.pdf', 4096000, (SELECT id FROM course WHERE course_code = 'CS201'), (SELECT id FROM user WHERE username = 'teacher1'), 'MySQL数据库的安装、配置和基本操作', 'MySQL,数据库,教程'),
('高等数学公式手册', 'DOCUMENT', 'math_formulas.pdf', '/resources/math_formulas.pdf', 2560000, (SELECT id FROM course WHERE course_code = 'MATH101'), (SELECT id FROM user WHERE username = 'teacher2'), '常用数学公式和定理汇总', '数学,公式,手册'),
('软件工程案例分析', 'DOCUMENT', 'se_case_study.pdf', '/resources/se_case_study.pdf', 5120000, (SELECT id FROM course WHERE course_code = 'SE301'), (SELECT id FROM user WHERE username = 'teacher1'), '经典软件项目的开发过程分析', '软件工程,案例,分析'),
('机器学习算法代码示例', 'OTHER', 'ml_code_examples.zip', '/resources/ml_code_examples.zip', 8192000, (SELECT id FROM course WHERE course_code = 'AI201'), (SELECT id FROM user WHERE username = 'teacher2'), 'Python实现的常用机器学习算法', '机器学习,算法,代码');

-- 插入知识图谱数据
INSERT INTO knowledge_graph (node_name, node_type, course_id, description, difficulty, importance, learning_time) VALUES
('Java基础语法', 'CONCEPT', (SELECT id FROM course WHERE course_code = 'CS101'), 'Java语言的基本语法规则', 'EASY', 1.00, 120),
('面向对象编程', 'CONCEPT', (SELECT id FROM course WHERE course_code = 'CS101'), '面向对象编程的基本概念和原理', 'MEDIUM', 1.00, 180),
('线性表', 'CONCEPT', (SELECT id FROM course WHERE course_code = 'CS102'), '线性表的定义和基本操作', 'MEDIUM', 0.90, 90),
('栈和队列', 'CONCEPT', (SELECT id FROM course WHERE course_code = 'CS102'), '栈和队列的特点和应用', 'MEDIUM', 0.85, 120),
('关系数据库', 'CONCEPT', (SELECT id FROM course WHERE course_code = 'CS201'), '关系数据库的基本理论', 'MEDIUM', 0.95, 150),
('SQL语言', 'SKILL', (SELECT id FROM course WHERE course_code = 'CS201'), 'SQL查询语言的使用', 'MEDIUM', 0.90, 180),
('微积分基础', 'CONCEPT', (SELECT id FROM course WHERE course_code = 'MATH101'), '微积分的基本概念和计算', 'HARD', 1.00, 240),
('概率论基础', 'CONCEPT', (SELECT id FROM course WHERE course_code = 'MATH201'), '概率论的基本理论', 'HARD', 0.95, 200),
('软件生命周期', 'CONCEPT', (SELECT id FROM course WHERE course_code = 'SE301'), '软件开发的生命周期模型', 'MEDIUM', 0.90, 120),
('监督学习', 'CONCEPT', (SELECT id FROM course WHERE course_code = 'AI201'), '监督学习算法的原理和应用', 'HARD', 0.95, 300);

-- 插入题库数据
INSERT INTO question_bank (question_text, question_type, course_id, difficulty, options, correct_answer, explanation, score, created_by) VALUES
('Java中哪个关键字用于定义类？', 'SINGLE_CHOICE', (SELECT id FROM course WHERE course_code = 'CS101'), 'EASY', '["A. class", "B. Class", "C. define", "D. object"]', 'A', 'Java中使用class关键字定义类', 2.00, (SELECT id FROM user WHERE username = 'teacher1')),
('以下哪个不是面向对象编程的特征？', 'SINGLE_CHOICE', (SELECT id FROM course WHERE course_code = 'CS101'), 'MEDIUM', '["A. 封装", "B. 继承", "C. 多态", "D. 递归"]', 'D', '面向对象编程的三大特征是封装、继承和多态', 3.00, (SELECT id FROM user WHERE username = 'teacher1')),
('栈的特点是什么？', 'SINGLE_CHOICE', (SELECT id FROM course WHERE course_code = 'CS102'), 'EASY', '["A. 先进先出", "B. 后进先出", "C. 随机访问", "D. 顺序访问"]', 'B', '栈是后进先出(LIFO)的数据结构', 2.00, (SELECT id FROM user WHERE username = 'teacher1')),
('SQL中用于查询数据的关键字是？', 'SINGLE_CHOICE', (SELECT id FROM course WHERE course_code = 'CS201'), 'EASY', '["A. SELECT", "B. QUERY", "C. FIND", "D. GET"]', 'A', 'SELECT是SQL中用于查询数据的关键字', 2.00, (SELECT id FROM user WHERE username = 'teacher1')),
('请简述面向对象编程的三大特征。', 'ESSAY', (SELECT id FROM course WHERE course_code = 'CS101'), 'MEDIUM', NULL, '封装、继承、多态', '封装是将数据和方法包装在类中；继承是子类可以继承父类的属性和方法；多态是同一个接口可以有不同的实现', 10.00, (SELECT id FROM user WHERE username = 'teacher1'));

-- 插入AI功能记录
INSERT INTO ai_feature (feature_name, feature_type, description, target_entity_type, target_entity_id, input_data, output_data, confidence_score, processing_status, created_by) VALUES
('作业自动批改', 'AUTO_GRADING', '对Java编程作业进行自动批改', 'task', (SELECT id FROM task WHERE task_title = 'Java基础练习1'), '{"submission_id": 1, "code_content": "public class Hello {...}"}', '{"score": 95, "feedback": "代码规范良好", "errors": []}', 0.92, 'COMPLETED', (SELECT id FROM user WHERE username = 'teacher1')),
('知识点提取', 'KNOWLEDGE_EXTRACTION', '从课程内容中提取关键知识点', 'course', (SELECT id FROM course WHERE course_code = 'CS101'), '{"course_content": "Java编程基础课程"}', '{"knowledge_points": ["Java语法", "面向对象", "异常处理"]}', 0.88, 'COMPLETED', (SELECT id FROM user WHERE username = 'teacher1')),
('学习能力分析', 'ABILITY_ANALYSIS', '分析学生的学习能力和进度', 'student', (SELECT id FROM student WHERE student_no = 'S2023001'), '{"grades": [95, 88, 110], "submissions": 3}', '{"ability_level": "优秀", "weak_points": [], "suggestions": ["可以尝试更有挑战性的项目"]}', 0.85, 'COMPLETED', (SELECT id FROM user WHERE username = 'teacher1')),
('内容推荐', 'CONTENT_RECOMMENDATION', '为学生推荐合适的学习资源', 'student', (SELECT id FROM student WHERE student_no = 'S2023002'), '{"current_progress": "Java基础", "learning_style": "视觉型"}', '{"recommendations": [{"type": "video", "title": "Java面向对象编程视频教程"}]}', 0.78, 'COMPLETED', (SELECT id FROM user WHERE username = 'teacher1')),
('查重检测', 'PLAGIARISM_CHECK', '检测作业提交的原创性', 'submission', 1, '{"submission_content": "学生提交的代码内容"}', '{"similarity": 15, "sources": [], "is_original": true}', 0.95, 'COMPLETED', (SELECT id FROM user WHERE username = 'teacher1'));

-- =============================================
-- 数据库初始化完成
-- =============================================

SELECT 'Database initialization completed successfully!' as message;
SELECT 'Root user: root / root123' as root_info;
SELECT 'Database: education_platform' as database_info;
SELECT 'Character Set: utf8mb4' as charset_info;
SELECT 'Collation: utf8mb4_unicode_ci' as collation_info;