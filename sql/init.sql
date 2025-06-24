-- =============================================
-- AI赋能教育管理与学习辅助平台 - 数据库初始化脚本
-- 版本: 1.0
-- 创建时间: 2024
-- 数据库: MySQL 8.0+
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
  `download_count` int DEFAULT '0' COMMENT '下载次数',
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
-- 13. 学生课程关联表 (student_course)
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
-- 初始化数据
-- =============================================

-- 插入默认管理员用户 (密码: admin123)
INSERT INTO user (username, password, email, real_name, role, status) VALUES 
('admin', '$2a$10$N.zmdr9k7uOCQb376NoUnuTJ8iAt6Z5EHsM8lE9lBaLO.oj1UKJAy', 'admin@education.com', '系统管理员', 'ADMIN', 1);

-- 插入测试教师 (密码: teacher123)
INSERT INTO user (username, password, email, real_name, role, status) VALUES 
('teacher001', '$2a$10$N.zmdr9k7uOCQb376NoUnuTJ8iAt6Z5EHsM8lE9lBaLO.oj1UKJAy', 'teacher001@education.com', '张教授', 'TEACHER', 1),
('teacher002', '$2a$10$N.zmdr9k7uOCQb376NoUnuTJ8iAt6Z5EHsM8lE9lBaLO.oj1UKJAy', 'teacher002@education.com', '李老师', 'TEACHER', 1);

INSERT INTO teacher (user_id, teacher_no, department, title, education, specialty) VALUES 
(2, 'T001', '计算机科学与技术学院', '教授', '博士', '人工智能,机器学习'),
(3, 'T002', '计算机科学与技术学院', '副教授', '硕士', '软件工程,数据库');

-- 插入测试学生 (密码: student123)
INSERT INTO user (username, password, email, real_name, role, status) VALUES 
('student001', '$2a$10$N.zmdr9k7uOCQb376NoUnuTJ8iAt6Z5EHsM8lE9lBaLO.oj1UKJAy', 'student001@education.com', '王同学', 'STUDENT', 1),
('student002', '$2a$10$N.zmdr9k7uOCQb376NoUnuTJ8iAt6Z5EHsM8lE9lBaLO.oj1UKJAy', 'student002@education.com', '李同学', 'STUDENT', 1),
('student003', '$2a$10$N.zmdr9k7uOCQb376NoUnuTJ8iAt6Z5EHsM8lE9lBaLO.oj1UKJAy', 'student003@education.com', '张同学', 'STUDENT', 1);

-- 插入测试班级
INSERT INTO class (class_name, class_code, teacher_id, major, grade, semester, description) VALUES 
('计算机科学与技术2024-1班', 'CS2024-1', 1, '计算机科学与技术', '2024', '2024-1', '计算机科学与技术专业2024级1班'),
('软件工程2024-1班', 'SE2024-1', 2, '软件工程', '2024', '2024-1', '软件工程专业2024级1班');

INSERT INTO student (user_id, student_no, class_id, major, grade, enrollment_year) VALUES 
(4, 'S2024001', 1, '计算机科学与技术', '2024', 2024),
(5, 'S2024002', 1, '计算机科学与技术', '2024', 2024),
(6, 'S2024003', 2, '软件工程', '2024', 2024);

-- 插入测试课程
INSERT INTO course (course_name, course_code, teacher_id, class_id, credits, course_type, semester, description) VALUES 
('人工智能基础', 'AI101', 1, 1, 3.0, 'REQUIRED', '2024-1', '人工智能基础理论与应用'),
('数据结构与算法', 'CS201', 2, 1, 4.0, 'REQUIRED', '2024-1', '数据结构与算法设计'),
('软件工程导论', 'SE101', 2, 2, 3.0, 'REQUIRED', '2024-1', '软件工程基础理论与实践');

-- 插入学生选课记录
INSERT INTO student_course (student_id, course_id, status) VALUES 
(1, 1, 'ENROLLED'),
(1, 2, 'ENROLLED'),
(2, 1, 'ENROLLED'),
(2, 2, 'ENROLLED'),
(3, 3, 'ENROLLED');

-- 插入测试任务
INSERT INTO task (task_title, task_type, course_id, teacher_id, description, total_score, start_time, end_time, status) VALUES 
('第一章作业：AI概述', 'HOMEWORK', 1, 1, '请阅读教材第一章，完成课后习题1-5题', 100.00, '2024-03-01 08:00:00', '2024-03-08 23:59:59', 'PUBLISHED'),
('数据结构实验一：线性表', 'PROJECT', 2, 2, '实现线性表的基本操作，包括插入、删除、查找等功能', 100.00, '2024-03-01 08:00:00', '2024-03-15 23:59:59', 'PUBLISHED'),
('软件工程期中考试', 'EXAM', 3, 2, '软件工程基础理论考试', 100.00, '2024-04-15 14:00:00', '2024-04-15 16:00:00', 'DRAFT');

-- 更新班级学生人数
UPDATE class SET student_count = (
    SELECT COUNT(*) FROM student WHERE class_id = class.id AND deleted = 0
) WHERE id IN (1, 2);

-- 插入通知记录
INSERT INTO `notification` (`title`, `content`, `type`, `priority`, `sender_id`, `sender_type`, `recipient_id`, `recipient_type`, `related_entity_type`, `related_entity_id`, `is_read`, `send_time`) VALUES
('新任务发布', '您有一个新的Java基础练习任务需要完成，截止时间为2024-01-20', 'TASK', 'NORMAL', 1, 'TEACHER', 1, 'STUDENT', 'task', 1, 0, '2024-01-10 09:00:00'),
('成绩已发布', '您的Java基础练习任务成绩已发布，请查看', 'GRADE', 'NORMAL', 1, 'TEACHER', 1, 'STUDENT', 'task', 1, 1, '2024-01-16 10:00:00'),
('系统维护通知', '系统将于今晚22:00-24:00进行维护，请提前保存您的工作', 'SYSTEM', 'HIGH', NULL, 'SYSTEM', 1, 'ALL', NULL, NULL, 0, '2024-01-15 18:00:00');

-- =============================================
-- 数据库初始化完成
-- =============================================

SELECT 'Database initialization completed successfully!' as message;
SELECT 'Default admin user: admin / admin123' as admin_info;
SELECT 'Test teacher: teacher001 / teacher123' as teacher_info;
SELECT 'Test student: student001 / student123' as student_info;