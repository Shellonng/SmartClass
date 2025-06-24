# 数据库设计文档

## 一、数据库概述

### 1.1 数据库信息
- 数据库名称：education_platform
- 数据库版本：MySQL 8.0+
- 字符集：utf8mb4
- 排序规则：utf8mb4_unicode_ci

### 1.2 设计原则
- 遵循第三范式
- 合理使用外键约束
- 预留扩展字段
- 统一命名规范
- 软删除设计

## 二、核心表结构设计

### 2.1 用户表 (user)

用户基础信息表，存储所有用户的通用信息。

```sql
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
```

### 2.2 教师表 (teacher)

教师详细信息表，与用户表一对一关联。

```sql
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
```

### 2.3 学生表 (student)

学生详细信息表，与用户表一对一关联。

```sql
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
```

### 2.4 班级表 (class)

班级信息表，管理班级基本信息。

```sql
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
```

### 2.5 课程表 (course)

课程信息表，存储课程基本信息。

```sql
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
```

### 2.6 任务表 (task)

学习任务表，存储作业、考试等任务信息。

```sql
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
```

### 2.7 任务提交表 (submission)

学生任务提交记录表。

```sql
CREATE TABLE `submission` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '提交ID',
  `task_id` bigint NOT NULL COMMENT '任务ID',
  `student_id` bigint NOT NULL COMMENT '学生ID',
  `attempt_number` int DEFAULT '1' COMMENT '提交次数',
  `content` text COMMENT '提交内容',
  `file_urls` text COMMENT '提交文件URLs(JSON格式)',
  `submit_time` datetime DEFAULT CURRENT_TIMESTAMP COMMENT '提交时间',
  `is_late` tinyint DEFAULT '0' COMMENT '是否迟交',
  `status` enum('SUBMITTED','GRADED','RETURNED') DEFAULT 'SUBMITTED' COMMENT '状态',
  `score` decimal(5,2) DEFAULT NULL COMMENT '得分',
  `feedback` text COMMENT '教师反馈',
  `auto_grade_result` text COMMENT 'AI批改结果(JSON格式)',
  `grade_time` datetime DEFAULT NULL COMMENT '批改时间',
  `graded_by` bigint DEFAULT NULL COMMENT '批改教师ID',
  `created_time` datetime DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `updated_time` datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  `deleted` tinyint DEFAULT '0' COMMENT '是否删除(0:否,1:是)',
  `ext_field1` varchar(255) DEFAULT NULL COMMENT '扩展字段1',
  `ext_field2` varchar(255) DEFAULT NULL COMMENT '扩展字段2',
  `ext_field3` text COMMENT '扩展字段3(JSON格式)',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_task_student_attempt` (`task_id`,`student_id`,`attempt_number`),
  KEY `idx_student_id` (`student_id`),
  KEY `idx_status` (`status`),
  KEY `idx_submit_time` (`submit_time`),
  CONSTRAINT `fk_submission_task` FOREIGN KEY (`task_id`) REFERENCES `task` (`id`),
  CONSTRAINT `fk_submission_student` FOREIGN KEY (`student_id`) REFERENCES `student` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='任务提交表';
```

### 2.8 成绩表 (grade)

成绩记录表，存储各种成绩信息。

```sql
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
```

### 2.9 资源表 (resource)

教学资源表，存储课程相关的文件资源。

```sql
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
```

### 2.10 知识图谱表 (knowledge_graph)

AI功能扩展 - 知识图谱节点表。

```sql
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
```

### 2.11 题库表 (question_bank)

题库管理表，支持各种题型。

```sql
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
```

### 2.12 AI功能表 (ai_feature)

AI功能扩展表，记录AI相关的功能和数据。

```sql
CREATE TABLE `ai_feature` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT 'AI功能ID',
  `feature_type` enum('RECOMMENDATION','AUTO_GRADE','KNOWLEDGE_EXTRACT','ABILITY_ANALYSIS','QA_ASSISTANT') NOT NULL COMMENT '功能类型',
  `user_id` bigint NOT NULL COMMENT '用户ID',
  `target_type` enum('COURSE','TASK','STUDENT','RESOURCE') NOT NULL COMMENT '目标类型',
  `target_id` bigint NOT NULL COMMENT '目标ID',
  `input_data` text COMMENT '输入数据(JSON格式)',
  `output_data` text COMMENT '输出数据(JSON格式)',
  `confidence_score` decimal(3,2) DEFAULT NULL COMMENT '置信度',
  `processing_time` int DEFAULT NULL COMMENT '处理时间(毫秒)',
  `model_version` varchar(50) DEFAULT NULL COMMENT '模型版本',
  `status` enum('PENDING','PROCESSING','COMPLETED','FAILED') DEFAULT 'PENDING' COMMENT '处理状态',
  `error_message` text COMMENT '错误信息',
  `feedback_score` decimal(3,2) DEFAULT NULL COMMENT '用户反馈评分',
  `feedback_comment` text COMMENT '用户反馈意见',
  `created_time` datetime DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `updated_time` datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  `deleted` tinyint DEFAULT '0' COMMENT '是否删除(0:否,1:是)',
  `ext_field1` varchar(255) DEFAULT NULL COMMENT '扩展字段1',
  `ext_field2` varchar(255) DEFAULT NULL COMMENT '扩展字段2',
  `ext_field3` text COMMENT '扩展字段3(JSON格式)',
  PRIMARY KEY (`id`),
  KEY `idx_feature_type` (`feature_type`),
  KEY `idx_user_id` (`user_id`),
  KEY `idx_target` (`target_type`,`target_id`),
  KEY `idx_status` (`status`),
  KEY `idx_created_time` (`created_time`),
  CONSTRAINT `fk_ai_feature_user` FOREIGN KEY (`user_id`) REFERENCES `user` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='AI功能表';
```

## 三、关联关系表

### 3.1 学生课程关联表 (student_course)

```sql
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
```

## 四、索引优化建议

### 4.1 复合索引

```sql
-- 成绩查询优化
ALTER TABLE grade ADD INDEX idx_student_course_type (student_id, course_id, grade_type);

-- 任务查询优化
ALTER TABLE task ADD INDEX idx_course_status_time (course_id, status, end_time);

-- 提交记录查询优化
ALTER TABLE submission ADD INDEX idx_task_student_status (task_id, student_id, status);

-- 资源查询优化
ALTER TABLE resource ADD INDEX idx_course_type_public (course_id, resource_type, is_public);
```

### 4.2 全文索引

```sql
-- 课程内容搜索
ALTER TABLE course ADD FULLTEXT INDEX ft_course_content (course_name, description, objectives);

-- 任务内容搜索
ALTER TABLE task ADD FULLTEXT INDEX ft_task_content (task_title, description, requirements);

-- 题库内容搜索
ALTER TABLE question_bank ADD FULLTEXT INDEX ft_question_content (question_text, explanation);
```

## 五、数据初始化

### 5.1 管理员用户

```sql
-- 插入默认管理员用户
INSERT INTO user (username, password, email, real_name, role, status) VALUES 
('admin', '$2a$10$N.zmdr9k7uOCQb376NoUnuTJ8iAt6Z5EHsM8lE9lBaLO.oj1UKJAy', 'admin@education.com', '系统管理员', 'ADMIN', 1);
```

### 5.2 测试数据

```sql
-- 插入测试教师
INSERT INTO user (username, password, email, real_name, role, status) VALUES 
('teacher001', '$2a$10$N.zmdr9k7uOCQb376NoUnuTJ8iAt6Z5EHsM8lE9lBaLO.oj1UKJAy', 'teacher001@education.com', '张教授', 'TEACHER', 1);

INSERT INTO teacher (user_id, teacher_no, department, title, education) VALUES 
(2, 'T001', '计算机科学与技术学院', '教授', '博士');

-- 插入测试学生
INSERT INTO user (username, password, email, real_name, role, status) VALUES 
('student001', '$2a$10$N.zmdr9k7uOCQb376NoUnuTJ8iAt6Z5EHsM8lE9lBaLO.oj1UKJAy', 'student001@education.com', '李同学', 'STUDENT', 1);

INSERT INTO student (user_id, student_no, major, grade, enrollment_year) VALUES 
(3, 'S2024001', '计算机科学与技术', '2024', 2024);
```

## 六、Redis缓存键设计

### 6.1 缓存键命名规范

```
# 用户相关
user:session:{token}                    # 用户会话信息
user:info:{userId}                      # 用户基本信息
user:permissions:{userId}               # 用户权限信息

# 课程相关
course:list:{teacherId}                 # 教师课程列表
course:info:{courseId}                  # 课程详细信息
course:students:{courseId}              # 课程学生列表

# 班级相关
class:list:{teacherId}                  # 教师班级列表
class:info:{classId}                    # 班级详细信息
class:students:{classId}                # 班级学生列表

# 成绩相关
grade:statistics:{courseId}:{classId}   # 成绩统计信息
grade:trend:{studentId}:{courseId}      # 学生成绩趋势
grade:ranking:{courseId}                # 课程成绩排名

# 任务相关
task:list:{courseId}                    # 课程任务列表
task:info:{taskId}                      # 任务详细信息
task:submissions:{taskId}               # 任务提交统计

# 验证码相关
captcha:{sessionId}                     # 图形验证码
sms:code:{phone}                        # 短信验证码
email:code:{email}                      # 邮箱验证码

# AI功能相关
ai:recommendation:{userId}              # AI推荐结果
ai:analysis:{targetType}:{targetId}     # AI分析结果
ai:knowledge:graph:{courseId}           # 知识图谱缓存
```

### 6.2 缓存过期时间设置

```
用户会话: 2小时
用户信息: 30分钟
课程列表: 10分钟
成绩统计: 5分钟
验证码: 5分钟
AI分析结果: 1小时
```

这个数据库设计为AI赋能教育平台提供了完整的数据支撑，包含了所有核心功能的表结构，并为AI功能扩展预留了充足的字段和表结构。