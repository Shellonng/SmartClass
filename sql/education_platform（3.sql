/*
 Navicat Premium Data Transfer

 Source Server         : 111
 Source Server Type    : MySQL
 Source Server Version : 80013
 Source Host           : localhost:3306
 Source Schema         : education_platform

 Target Server Type    : MySQL
 Target Server Version : 80013
 File Encoding         : 65001

 Date: 12/07/2025 01:35:17
*/

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for assignment
-- ----------------------------
DROP TABLE IF EXISTS `assignment`;
CREATE TABLE `assignment`  (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `title` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '作业或考试标题',
  `course_id` bigint(20) NOT NULL,
  `user_id` bigint(20) NOT NULL COMMENT '发布作业的用户ID',
  `type` enum('homework','exam') CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT 'homework',
  `description` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL,
  `start_time` datetime NULL DEFAULT NULL,
  `end_time` datetime NULL DEFAULT NULL,
  `create_time` datetime NULL DEFAULT CURRENT_TIMESTAMP,
  `status` tinyint(4) NOT NULL DEFAULT 0 COMMENT '发布状态：0未发布，1已发布',
  `update_time` datetime NULL DEFAULT NULL,
  `mode` enum('question','file') CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL DEFAULT 'question' COMMENT '作业模式：question-答题型，file-上传型',
  `time_limit` int(11) NULL DEFAULT NULL COMMENT '时间限制（分钟）',
  `total_score` int(11) NULL DEFAULT 100 COMMENT '总分',
  `duration` int(11) NULL DEFAULT NULL COMMENT '考试时长（分钟）',
  `allowed_file_types` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL COMMENT '允许的文件类型（JSON格式）',
  `max_file_size` int(11) NULL DEFAULT 10 COMMENT '最大文件大小（MB）',
  `reference_answer` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL COMMENT '参考答案（用于智能批改）',
  PRIMARY KEY (`id`) USING BTREE,
  INDEX `course_id`(`course_id`) USING BTREE,
  INDEX `fk_assignment_user`(`user_id`) USING BTREE,
  CONSTRAINT `assignment_ibfk_1` FOREIGN KEY (`course_id`) REFERENCES `course` (`id`) ON DELETE RESTRICT ON UPDATE RESTRICT,
  CONSTRAINT `fk_assignment_user` FOREIGN KEY (`user_id`) REFERENCES `user` (`id`) ON DELETE CASCADE ON UPDATE RESTRICT
) ENGINE = InnoDB AUTO_INCREMENT = 30 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci COMMENT = '作业或考试表' ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of assignment
-- ----------------------------
INSERT INTO `assignment` VALUES (9, '测试1', 9, 6, 'homework', '测试1的说明', '2025-07-01 14:00:00', '2025-07-10 16:00:00', '2025-07-01 14:53:32', 1, '2025-07-03 01:46:44', 'question', NULL, 100, NULL, NULL, 10, NULL);
INSERT INTO `assignment` VALUES (15, '12', 9, 3, 'exam', '132', '2025-07-01 00:00:00', '2025-07-01 01:00:00', '2025-07-01 18:19:09', 1, '2025-07-01 18:19:09', 'question', NULL, 100, NULL, NULL, 10, NULL);
INSERT INTO `assignment` VALUES (17, '作业2', 9, 3, 'homework', '', '2025-07-01 00:00:00', '2025-07-01 01:00:00', '2025-07-01 18:41:05', 1, '2025-07-02 19:35:30', 'file', NULL, 100, NULL, NULL, 10, NULL);
INSERT INTO `assignment` VALUES (20, '测试2', 9, 6, 'exam', '', '2025-07-01 01:00:00', '2025-07-01 02:00:00', '2025-07-01 19:05:51', 1, '2025-07-01 19:05:59', 'question', NULL, 100, NULL, NULL, 10, NULL);
INSERT INTO `assignment` VALUES (24, '1', 9, 6, 'homework', '', '2025-07-02 20:07:35', '2025-07-12 21:00:00', '2025-07-02 20:07:41', 1, '2025-07-02 20:07:41', 'file', NULL, 100, NULL, NULL, 10, NULL);
INSERT INTO `assignment` VALUES (25, '2', 9, 6, 'homework', '', '2025-07-03 20:12:48', '2025-07-05 20:12:55', '2025-07-02 20:12:58', 1, '2025-07-02 20:12:58', 'question', NULL, 100, NULL, NULL, 10, NULL);
INSERT INTO `assignment` VALUES (26, '1', 9, 6, 'exam', '', '2025-07-11 18:49:17', '2025-07-12 18:49:19', '2025-07-11 18:49:23', 1, '2025-07-11 20:07:45', 'question', NULL, 100, NULL, NULL, 10, NULL);
INSERT INTO `assignment` VALUES (27, '12', 9, 6, 'exam', '', '2025-07-11 20:09:22', '2025-07-12 20:09:25', '2025-07-11 20:09:28', 0, '2025-07-11 20:09:28', 'question', NULL, 100, NULL, NULL, 10, NULL);
INSERT INTO `assignment` VALUES (28, '???????????', 9, 6, 'homework', '???????????????????????????', '2025-07-01 08:00:00', '2025-07-20 23:59:59', '2025-07-11 21:35:46', 1, NULL, 'question', NULL, 100, NULL, NULL, 10, NULL);
INSERT INTO `assignment` VALUES (29, '智能批改测试作业', 9, 6, 'homework', '这是一个用于测试智能批改功能的作业', '2025-07-04 21:47:01', '2025-07-18 21:47:01', '2025-07-11 21:47:01', 1, NULL, 'question', NULL, 100, NULL, NULL, 10, NULL);

-- ----------------------------
-- Table structure for assignment_config
-- ----------------------------
DROP TABLE IF EXISTS `assignment_config`;
CREATE TABLE `assignment_config`  (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `assignment_id` bigint(20) NOT NULL COMMENT '作业ID',
  `knowledge_points` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL COMMENT '知识点范围（JSON格式）',
  `difficulty` enum('EASY','MEDIUM','HARD') CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT 'MEDIUM' COMMENT '难度级别',
  `question_count` int(11) NULL DEFAULT 10 COMMENT '题目总数',
  `question_types` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL COMMENT '题型分布（JSON格式）',
  `additional_requirements` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL COMMENT '额外要求',
  `create_time` datetime NULL DEFAULT CURRENT_TIMESTAMP,
  `update_time` datetime NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE INDEX `uk_assignment_config`(`assignment_id`) USING BTREE,
  CONSTRAINT `fk_assignment_config_assignment` FOREIGN KEY (`assignment_id`) REFERENCES `assignment` (`id`) ON DELETE CASCADE ON UPDATE RESTRICT
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci COMMENT = '作业配置表' ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of assignment_config
-- ----------------------------

-- ----------------------------
-- Table structure for assignment_question
-- ----------------------------
DROP TABLE IF EXISTS `assignment_question`;
CREATE TABLE `assignment_question`  (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `assignment_id` bigint(20) NOT NULL,
  `question_id` bigint(20) NOT NULL,
  `score` int(11) NULL DEFAULT 5 COMMENT '该题满分',
  `sequence` int(11) NULL DEFAULT 1 COMMENT '题目顺序',
  PRIMARY KEY (`id`) USING BTREE,
  INDEX `assignment_id`(`assignment_id`) USING BTREE,
  INDEX `question_id`(`question_id`) USING BTREE,
  CONSTRAINT `assignment_question_ibfk_1` FOREIGN KEY (`assignment_id`) REFERENCES `assignment` (`id`) ON DELETE CASCADE ON UPDATE RESTRICT,
  CONSTRAINT `assignment_question_ibfk_2` FOREIGN KEY (`question_id`) REFERENCES `question` (`id`) ON DELETE CASCADE ON UPDATE RESTRICT
) ENGINE = InnoDB AUTO_INCREMENT = 42 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci COMMENT = '作业题目关联表' ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of assignment_question
-- ----------------------------
INSERT INTO `assignment_question` VALUES (18, 9, 13, 5, 1);
INSERT INTO `assignment_question` VALUES (19, 9, 17, 5, 2);
INSERT INTO `assignment_question` VALUES (20, 9, 12, 5, 3);
INSERT INTO `assignment_question` VALUES (21, 9, 12, 0, 4);
INSERT INTO `assignment_question` VALUES (28, 17, 13, 5, 1);
INSERT INTO `assignment_question` VALUES (29, 17, 12, 5, 2);
INSERT INTO `assignment_question` VALUES (31, 26, 13, 5, 1);
INSERT INTO `assignment_question` VALUES (32, 28, 18, 40, 1);
INSERT INTO `assignment_question` VALUES (33, 28, 19, 15, 2);
INSERT INTO `assignment_question` VALUES (34, 28, 20, 15, 3);
INSERT INTO `assignment_question` VALUES (35, 28, 21, 25, 4);
INSERT INTO `assignment_question` VALUES (39, 29, 22, 40, 1);
INSERT INTO `assignment_question` VALUES (40, 29, 23, 30, 2);
INSERT INTO `assignment_question` VALUES (41, 29, 24, 30, 3);

-- ----------------------------
-- Table structure for assignment_submission
-- ----------------------------
DROP TABLE IF EXISTS `assignment_submission`;
CREATE TABLE `assignment_submission`  (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '主键ID',
  `assignment_id` bigint(20) NOT NULL COMMENT '作业ID',
  `student_id` bigint(20) NOT NULL COMMENT '学生ID',
  `status` int(11) NOT NULL DEFAULT 0 COMMENT '状态：0-未提交，1-已提交未批改，2-已批改',
  `score` int(11) NULL DEFAULT NULL COMMENT '得分',
  `feedback` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL COMMENT '教师反馈',
  `submit_time` datetime NULL DEFAULT NULL COMMENT '提交时间',
  `grade_time` datetime NULL DEFAULT NULL COMMENT '批改时间',
  `graded_by` bigint(20) NULL DEFAULT NULL COMMENT '批改人ID',
  `content` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL COMMENT '提交内容',
  `create_time` datetime NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  `file_name` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '文件名称',
  `file_path` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '文件路径',
  PRIMARY KEY (`id`) USING BTREE,
  INDEX `idx_assignment_id`(`assignment_id`) USING BTREE,
  INDEX `idx_student_id`(`student_id`) USING BTREE,
  INDEX `idx_status`(`status`) USING BTREE,
  CONSTRAINT `fk_submission_assignment` FOREIGN KEY (`assignment_id`) REFERENCES `assignment` (`id`) ON DELETE CASCADE ON UPDATE RESTRICT,
  CONSTRAINT `fk_submission_student` FOREIGN KEY (`student_id`) REFERENCES `user` (`id`) ON DELETE CASCADE ON UPDATE RESTRICT
) ENGINE = InnoDB AUTO_INCREMENT = 43 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci COMMENT = '作业提交记录表' ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of assignment_submission
-- ----------------------------
INSERT INTO `assignment_submission` VALUES (12, 9, 12, 1, 10, NULL, '2025-07-03 03:46:22', NULL, NULL, NULL, '2025-07-03 02:53:30', '2025-07-03 03:46:22', NULL, NULL);
INSERT INTO `assignment_submission` VALUES (32, 24, 12, 0, NULL, NULL, NULL, NULL, NULL, NULL, '2025-07-03 14:32:32', '2025-07-03 14:32:32', NULL, NULL);
INSERT INTO `assignment_submission` VALUES (33, 25, 12, 0, NULL, NULL, NULL, NULL, NULL, NULL, '2025-07-03 15:18:00', '2025-07-03 15:17:59', NULL, NULL);
INSERT INTO `assignment_submission` VALUES (34, 20, 12, 0, NULL, NULL, NULL, NULL, NULL, NULL, '2025-07-03 15:19:09', '2025-07-03 15:19:09', NULL, NULL);
INSERT INTO `assignment_submission` VALUES (35, 15, 12, 0, NULL, NULL, NULL, NULL, NULL, NULL, '2025-07-03 15:27:45', '2025-07-03 15:27:45', NULL, NULL);
INSERT INTO `assignment_submission` VALUES (36, 17, 12, 0, NULL, NULL, NULL, NULL, NULL, NULL, '2025-07-03 15:27:50', '2025-07-03 15:27:49', NULL, NULL);
INSERT INTO `assignment_submission` VALUES (37, 28, 12, 1, NULL, NULL, '2025-07-11 21:35:46', NULL, NULL, NULL, '2025-07-11 21:35:46', '2025-07-11 21:35:46', NULL, NULL);
INSERT INTO `assignment_submission` VALUES (38, 28, 12, 1, NULL, NULL, '2025-07-11 21:35:46', NULL, NULL, NULL, '2025-07-11 21:35:46', '2025-07-11 21:35:46', NULL, NULL);
INSERT INTO `assignment_submission` VALUES (39, 29, 12, 1, NULL, NULL, '2025-07-11 21:47:01', NULL, NULL, NULL, '2025-07-11 21:47:01', '2025-07-11 21:47:01', NULL, NULL);
INSERT INTO `assignment_submission` VALUES (40, 29, 12, 1, NULL, NULL, '2025-07-10 21:47:01', NULL, NULL, NULL, '2025-07-10 21:47:01', '2025-07-11 21:47:01', NULL, NULL);
INSERT INTO `assignment_submission` VALUES (41, 29, 12, 1, NULL, NULL, '2025-07-09 21:47:01', NULL, NULL, NULL, '2025-07-09 21:47:01', '2025-07-11 21:47:01', NULL, NULL);
INSERT INTO `assignment_submission` VALUES (42, 29, 12, 2, 85, '整体表现良好，对计算机基础知识掌握较好，但在多选题上有小错误。', '2025-07-08 21:47:01', '2025-07-11 21:47:01', 6, NULL, '2025-07-08 21:47:01', '2025-07-11 21:47:01', NULL, NULL);

-- ----------------------------
-- Table structure for assignment_submission_answer
-- ----------------------------
DROP TABLE IF EXISTS `assignment_submission_answer`;
CREATE TABLE `assignment_submission_answer`  (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '主键',
  `submission_id` bigint(20) NOT NULL COMMENT '提交ID，关联assignment_submission',
  `question_id` bigint(20) NOT NULL COMMENT '题目ID',
  `student_answer` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL COMMENT '学生的答案',
  `is_correct` tinyint(1) NULL DEFAULT NULL COMMENT '是否正确：1-正确，0-错误，NULL-未批改',
  `score` int(11) NULL DEFAULT NULL COMMENT '得分（教师批改后记录）',
  `comment` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL COMMENT '教师对该题的点评（可选）',
  `create_time` datetime NULL DEFAULT CURRENT_TIMESTAMP,
  `update_time` datetime NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`) USING BTREE,
  INDEX `submission_id`(`submission_id`) USING BTREE,
  INDEX `question_id`(`question_id`) USING BTREE,
  CONSTRAINT `fk_submission_answer_question` FOREIGN KEY (`question_id`) REFERENCES `question` (`id`) ON DELETE CASCADE ON UPDATE RESTRICT,
  CONSTRAINT `fk_submission_answer_submission` FOREIGN KEY (`submission_id`) REFERENCES `assignment_submission` (`id`) ON DELETE CASCADE ON UPDATE RESTRICT
) ENGINE = InnoDB AUTO_INCREMENT = 3 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci COMMENT = '学生答题记录表' ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of assignment_submission_answer
-- ----------------------------
INSERT INTO `assignment_submission_answer` VALUES (1, 12, 13, 'B', 1, 5, NULL, '2025-07-03 03:27:11', '2025-07-03 03:46:22');
INSERT INTO `assignment_submission_answer` VALUES (2, 12, 17, 'T', 1, 5, NULL, '2025-07-03 03:46:22', '2025-07-03 03:46:22');
INSERT INTO `assignment_submission_answer` VALUES (3, 37, 18, '??????????????????????CPU??????????????????????????????', NULL, NULL, NULL, '2025-07-11 21:35:46', '2025-07-11 21:35:46');
INSERT INTO `assignment_submission_answer` VALUES (4, 37, 19, 'A', NULL, NULL, NULL, '2025-07-11 21:35:46', '2025-07-11 21:35:46');
INSERT INTO `assignment_submission_answer` VALUES (5, 37, 20, '13', NULL, NULL, NULL, '2025-07-11 21:35:46', '2025-07-11 21:35:46');
INSERT INTO `assignment_submission_answer` VALUES (6, 37, 21, 'A,B,C', NULL, NULL, NULL, '2025-07-11 21:35:46', '2025-07-11 21:35:46');
INSERT INTO `assignment_submission_answer` VALUES (7, 38, 18, '?????????CPU??????', NULL, NULL, NULL, '2025-07-11 21:35:46', '2025-07-11 21:35:46');
INSERT INTO `assignment_submission_answer` VALUES (8, 38, 19, 'B', NULL, NULL, NULL, '2025-07-11 21:35:46', '2025-07-11 21:35:46');
INSERT INTO `assignment_submission_answer` VALUES (9, 38, 20, '14', NULL, NULL, NULL, '2025-07-11 21:35:46', '2025-07-11 21:35:46');
INSERT INTO `assignment_submission_answer` VALUES (10, 38, 21, 'A,D', NULL, NULL, NULL, '2025-07-11 21:35:46', '2025-07-11 21:35:46');
INSERT INTO `assignment_submission_answer` VALUES (11, 39, 22, 'CPU是计算机的核心部件，负责执行指令和处理数据，控制计算机的运行。它执行算术运算和逻辑运算，并管理数据的流动。', NULL, NULL, NULL, '2025-07-11 21:47:01', '2025-07-11 21:47:01');
INSERT INTO `assignment_submission_answer` VALUES (12, 39, 23, 'A,B,D', NULL, NULL, NULL, '2025-07-11 21:47:01', '2025-07-11 21:47:01');
INSERT INTO `assignment_submission_answer` VALUES (13, 39, 24, '10', NULL, NULL, NULL, '2025-07-11 21:47:01', '2025-07-11 21:47:01');
INSERT INTO `assignment_submission_answer` VALUES (14, 38, 22, 'CPU负责执行计算机程序的指令，进行数据处理和运算。', NULL, NULL, NULL, '2025-07-11 21:47:01', '2025-07-11 21:47:01');
INSERT INTO `assignment_submission_answer` VALUES (15, 38, 23, 'A,C,D', NULL, NULL, NULL, '2025-07-11 21:47:01', '2025-07-11 21:47:01');
INSERT INTO `assignment_submission_answer` VALUES (16, 38, 24, '8', NULL, NULL, NULL, '2025-07-11 21:47:01', '2025-07-11 21:47:01');
INSERT INTO `assignment_submission_answer` VALUES (17, 37, 22, 'CPU是计算机的大脑，主要功能是运行程序、处理数据和协调各部件工作。', NULL, NULL, NULL, '2025-07-11 21:47:01', '2025-07-11 21:47:01');
INSERT INTO `assignment_submission_answer` VALUES (18, 37, 23, 'A,B', NULL, NULL, NULL, '2025-07-11 21:47:01', '2025-07-11 21:47:01');
INSERT INTO `assignment_submission_answer` VALUES (19, 37, 24, '10', NULL, NULL, NULL, '2025-07-11 21:47:01', '2025-07-11 21:47:01');
INSERT INTO `assignment_submission_answer` VALUES (20, 42, 22, 'CPU是计算机的核心，负责执行指令、处理数据和控制计算机运行。', 1, 40, '回答全面准确，包含了CPU的核心功能。', '2025-07-11 21:47:01', '2025-07-11 21:47:01');
INSERT INTO `assignment_submission_answer` VALUES (21, 42, 23, 'A,B,C', 0, 20, '部分正确，但C选项(打印机)是输出设备而非输入设备。', '2025-07-11 21:47:01', '2025-07-11 21:47:01');
INSERT INTO `assignment_submission_answer` VALUES (22, 42, 24, '10', 1, 25, '答案正确。', '2025-07-11 21:47:01', '2025-07-11 21:47:01');

-- ----------------------------
-- Table structure for chapter
-- ----------------------------
DROP TABLE IF EXISTS `chapter`;
CREATE TABLE `chapter`  (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '章节ID，主键',
  `course_id` bigint(20) NOT NULL COMMENT '所属课程ID，外键',
  `title` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '章节名称',
  `description` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL,
  `sort_order` int(11) NOT NULL,
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`) USING BTREE,
  INDEX `idx_course_id`(`course_id`) USING BTREE,
  CONSTRAINT `fk_chapter_course` FOREIGN KEY (`course_id`) REFERENCES `course` (`id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE = InnoDB AUTO_INCREMENT = 13 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci COMMENT = '课程章节表' ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of chapter
-- ----------------------------
INSERT INTO `chapter` VALUES (11, 9, '计算机系统的组成', '介绍计算机系统的基本概念和发展历史', 1, '2025-06-29 16:22:39', '2025-07-03 01:41:18');
INSERT INTO `chapter` VALUES (12, 9, '数字逻辑基础', '介绍数字电路和逻辑设计的基本原理', 2, '2025-06-29 21:07:41', '2025-07-03 01:42:55');

-- ----------------------------
-- Table structure for class_student
-- ----------------------------
DROP TABLE IF EXISTS `class_student`;
CREATE TABLE `class_student`  (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `class_id` bigint(20) NOT NULL,
  `student_id` bigint(20) NOT NULL,
  `join_time` datetime NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE INDEX `uk_class_student`(`class_id`, `student_id`) USING BTREE,
  INDEX `student_id`(`student_id`) USING BTREE,
  CONSTRAINT `class_student_ibfk_1` FOREIGN KEY (`class_id`) REFERENCES `course_class` (`id`) ON DELETE CASCADE ON UPDATE RESTRICT,
  CONSTRAINT `class_student_ibfk_2` FOREIGN KEY (`student_id`) REFERENCES `student` (`id`) ON DELETE CASCADE ON UPDATE RESTRICT
) ENGINE = InnoDB AUTO_INCREMENT = 2 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci COMMENT = '班级学生关联表' ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of class_student
-- ----------------------------
INSERT INTO `class_student` VALUES (1, 10, 20250005, NULL);

-- ----------------------------
-- Table structure for course
-- ----------------------------
DROP TABLE IF EXISTS `course`;
CREATE TABLE `course`  (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '课程ID',
  `title` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '课程名称',
  `description` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL COMMENT '课程描述',
  `cover_image` varchar(500) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '课程封面图片',
  `credit` decimal(3, 1) NULL DEFAULT 3.0 COMMENT '学分',
  `course_type` varchar(20) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT '必修课' COMMENT '课程类型',
  `start_time` datetime NULL DEFAULT NULL COMMENT '开始时间',
  `end_time` datetime NULL DEFAULT NULL COMMENT '结束时间',
  `teacher_id` bigint(20) NOT NULL COMMENT '教师ID',
  `status` varchar(20) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT '未开始' COMMENT '课程状态',
  `term` varchar(20) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '学期',
  `student_count` int(11) NULL DEFAULT 0 COMMENT '学生数量',
  `average_score` decimal(5, 2) NULL DEFAULT NULL COMMENT '平均分数',
  `create_time` datetime NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`) USING BTREE,
  INDEX `idx_teacher_id`(`teacher_id`) USING BTREE,
  INDEX `idx_term`(`term`) USING BTREE,
  INDEX `idx_status`(`status`) USING BTREE,
  CONSTRAINT `fk_course_teacher` FOREIGN KEY (`teacher_id`) REFERENCES `teacher` (`id`) ON DELETE RESTRICT ON UPDATE RESTRICT
) ENGINE = InnoDB AUTO_INCREMENT = 30 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_unicode_ci COMMENT = '课程表' ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of course
-- ----------------------------
INSERT INTO `course` VALUES (1, 'Java编程基础', 'Java编程语言入门课程', NULL, 3.0, '必修课', '2024-09-01 08:00:00', '2024-12-31 17:00:00', 2025000, '未开始', '2024-2025-1', 0, NULL, '2025-06-29 02:09:12', '2025-06-29 02:09:12');
INSERT INTO `course` VALUES (2, '高等数学', '高等数学基础课程', NULL, 4.0, '必修课', '2024-09-01 08:00:00', '2024-12-31 17:00:00', 2025001, '未开始', '2024-2025-1', 0, NULL, '2025-06-29 02:09:12', '2025-06-29 02:09:12');
INSERT INTO `course` VALUES (3, '人工智能导论', '人工智能基础理论与应用', NULL, 3.0, '选修课', '2024-09-01 08:00:00', '2024-12-31 17:00:00', 2025000, '未开始', '2024-2025-1', 0, NULL, '2025-06-29 02:09:12', '2025-06-29 02:09:12');
INSERT INTO `course` VALUES (5, '测试课程', '123', '', 3.0, '必修课', '2025-07-03 02:15:34', '2025-07-11 02:15:36', 2025002, '未开始', '2024-2025-1', 0, NULL, '2025-06-29 02:15:41', '2025-06-29 02:15:41');
INSERT INTO `course` VALUES (6, '123', '123', '/api/photo/202507/799f6631a8604aeaa5a59b4416cf43da.png', 3.0, '必修课', '2025-07-23 02:15:34', '2025-08-01 02:15:36', 2025002, '未开始', '2024-2025-1', 0, NULL, '2025-06-29 02:15:59', '2025-07-02 11:51:33');
INSERT INTO `course` VALUES (9, '计算机组成原理', '本课程介绍计算机系统的基本组成和工作原理，包括数字逻辑、CPU结构、存储系统、输入输出系统等内容。', 'https://img.freepik.com/free-vector/english-school-landing-page-template_23-2148475038.jpg', 4.0, '必修课', '2025-06-04 02:52:20', '2025-06-26 02:52:22', 2025002, '已结束', '2024-2025-1', 1, NULL, '2025-06-29 02:52:54', '2025-07-03 01:13:39');
INSERT INTO `course` VALUES (19, 'Java程序设计', 'Java编程基础课程，零基础向对象编程思想和Java核心技术，为开发企业级应用打下基础', 'https://img.freepik.com/free-vector/programming-concept-illustration_114360-1670.jpg', 3.0, '必修课', '2024-05-01 00:00:00', '2024-12-31 00:00:00', 2025002, '已结束', '2024-2025-1', 16421, NULL, '2025-07-02 02:13:16', '2025-07-02 02:13:16');
INSERT INTO `course` VALUES (20, '数据结构与算法', '深入学习常用数据结构和算法设计技巧，包括数组、链表、栈、队列、树、图以及常见算法', 'https://img.freepik.com/free-vector/programming-concept-illustration_114360-1213.jpg', 4.0, '必修课', '2024-05-01 00:00:00', '2024-12-31 00:00:00', 2025002, '已结束', '2024-2025-1', 9850, NULL, '2025-07-02 02:13:16', '2025-07-02 02:13:16');
INSERT INTO `course` VALUES (21, 'Python程序基础', '零基础入门Python编程，掌握Python基本语法、数据类型、控制结构、函数和模块开发', 'https://img.freepik.com/free-vector/programming-concept-illustration_114360-1351.jpg', 3.0, '必修课', '2024-05-01 00:00:00', '2024-12-31 00:00:00', 2025002, '已结束', '2024-2025-1', 23190, NULL, '2025-07-02 02:13:16', '2025-07-02 02:13:16');
INSERT INTO `course` VALUES (22, '微观经济学原理', '介绍微观经济学的基本理论，包括供需关系、市场结构、消费行为、生产理论等', 'https://img.freepik.com/free-vector/economy-concept-illustration_114360-7385.jpg', 3.0, '必修课', '2024-06-01 00:00:00', '2024-12-31 00:00:00', 2025002, '已结束', '2024-2025-1', 11201, NULL, '2025-07-02 02:13:16', '2025-07-02 02:13:16');
INSERT INTO `course` VALUES (23, '线性代数', '系统学习线性代数的基本概念和方法，包括向量运算、行列式、向量空间、特征值和特征向量', 'https://img.freepik.com/free-vector/mathematics-concept-illustration_114360-3972.jpg', 4.0, '必修课', '2024-06-01 00:00:00', '2024-12-31 00:00:00', 2025002, '已结束', '2024-2025-1', 13580, NULL, '2025-07-02 02:13:16', '2025-07-02 02:13:16');
INSERT INTO `course` VALUES (24, '大学物理（力学部分）', '系统讲授经典力学的基本概念、定律和方法，包括刚体力学、振动力学、流体力学等', 'https://img.freepik.com/free-vector/physics-concept-illustration_114360-3972.jpg', 3.0, '必修课', '2024-06-01 00:00:00', '2024-12-31 00:00:00', 2025002, '已结束', '2024-2025-1', 12680, NULL, '2025-07-02 02:13:16', '2025-07-02 02:13:16');
INSERT INTO `course` VALUES (25, '高等数学（上）', '本课程系统讲授高等数学的基本概念、理论和方法，包括极限、导数、微积分等', 'https://img.freepik.com/free-vector/hand-drawn-mathematics-background_23-2148157511.jpg', 4.0, '必修课', '2025-07-01 11:57:20', '2025-07-17 11:57:24', 2025002, '进行中', '2024-2025-1', 15420, NULL, '2025-07-02 02:13:16', '2025-07-02 11:57:30');
INSERT INTO `course` VALUES (26, '大学英语综合教程', '提升英语听说读写综合能力，涵盖语法、词汇、阅读理解和写作技巧，为四六级考试做准备', 'https://img.freepik.com/free-vector/english-school-landing-page-template_23-2148475038.jpg', 3.0, '必修课', '2024-06-01 00:00:00', '2024-12-31 00:00:00', 2025002, '已结束', '2024-2025-1', 18700, NULL, '2025-07-02 02:13:16', '2025-07-02 02:13:16');

-- ----------------------------
-- Table structure for course_class
-- ----------------------------
DROP TABLE IF EXISTS `course_class`;
CREATE TABLE `course_class`  (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '班级ID',
  `course_id` bigint(20) NULL DEFAULT NULL,
  `teacher_id` bigint(20) NOT NULL,
  `name` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `description` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL,
  `is_default` tinyint(1) NULL DEFAULT 0,
  `create_time` datetime NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`) USING BTREE,
  INDEX `course_class_ibfk_1`(`course_id`) USING BTREE,
  INDEX `course_class_ibfk_2`(`teacher_id`) USING BTREE,
  CONSTRAINT `course_class_ibfk_1` FOREIGN KEY (`course_id`) REFERENCES `course` (`id`) ON DELETE SET NULL ON UPDATE RESTRICT,
  CONSTRAINT `course_class_ibfk_2` FOREIGN KEY (`teacher_id`) REFERENCES `user` (`id`) ON DELETE CASCADE ON UPDATE RESTRICT
) ENGINE = InnoDB AUTO_INCREMENT = 17 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci COMMENT = '课程班级表' ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of course_class
-- ----------------------------
INSERT INTO `course_class` VALUES (10, 9, 6, '软件工程2306班', '123', 0, '2025-07-02 14:08:30');

-- ----------------------------
-- Table structure for course_enrollment_request
-- ----------------------------
DROP TABLE IF EXISTS `course_enrollment_request`;
CREATE TABLE `course_enrollment_request`  (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '申请ID',
  `student_id` bigint(20) NOT NULL COMMENT '学生ID',
  `course_id` bigint(20) NOT NULL COMMENT '申请加入的课程ID',
  `status` tinyint(4) NULL DEFAULT 0 COMMENT '申请状态：0=待审核 1=已通过 2=已拒绝',
  `reason` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL COMMENT '学生申请理由',
  `review_comment` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL COMMENT '教师审核意见',
  `submit_time` datetime NULL DEFAULT CURRENT_TIMESTAMP,
  `review_time` datetime NULL DEFAULT NULL COMMENT '审核时间',
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE INDEX `uk_student_course`(`student_id`, `course_id`) USING BTREE,
  INDEX `course_id`(`course_id`) USING BTREE,
  CONSTRAINT `course_enrollment_request_ibfk_1` FOREIGN KEY (`student_id`) REFERENCES `student` (`id`) ON DELETE CASCADE ON UPDATE RESTRICT,
  CONSTRAINT `course_enrollment_request_ibfk_2` FOREIGN KEY (`course_id`) REFERENCES `course` (`id`) ON DELETE CASCADE ON UPDATE RESTRICT
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci COMMENT = '学生选课申请表' ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of course_enrollment_request
-- ----------------------------

-- ----------------------------
-- Table structure for course_resource
-- ----------------------------
DROP TABLE IF EXISTS `course_resource`;
CREATE TABLE `course_resource`  (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '资源ID',
  `course_id` bigint(20) NOT NULL COMMENT '所属课程ID',
  `name` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '资源名称',
  `file_type` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '文件类型，如pdf、doc、ppt等',
  `file_size` bigint(20) NOT NULL COMMENT '文件大小(字节)',
  `file_url` varchar(500) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '文件URL',
  `description` varchar(500) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '资源描述',
  `download_count` int(11) NULL DEFAULT 0 COMMENT '下载次数',
  `upload_user_id` bigint(20) NOT NULL COMMENT '上传用户ID',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`) USING BTREE,
  INDEX `idx_course_id`(`course_id`) USING BTREE COMMENT '课程ID索引',
  INDEX `idx_upload_user_id`(`upload_user_id`) USING BTREE COMMENT '上传用户ID索引'
) ENGINE = InnoDB AUTO_INCREMENT = 6 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci COMMENT = '课程资源表' ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of course_resource
-- ----------------------------
INSERT INTO `course_resource` VALUES (1, 9, '1', 'pdf', 325858, '/files/resources/9/202506/7acbf820bec64795bedaca556c235c4a.pdf', '1', 4, 6, '2025-06-30 16:18:14', '2025-06-30 16:18:14');
INSERT INTO `course_resource` VALUES (5, 9, '测试', 'png', 48173, '/files/resources/9/202507/a54ec5a02dba4fc9882aaa1f23caf063.png', '1', 2, 6, '2025-07-02 20:39:01', '2025-07-02 20:39:01');

-- ----------------------------
-- Table structure for course_student
-- ----------------------------
DROP TABLE IF EXISTS `course_student`;
CREATE TABLE `course_student`  (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `course_id` bigint(20) NOT NULL,
  `student_id` bigint(20) NOT NULL,
  `enroll_time` datetime NULL DEFAULT CURRENT_TIMESTAMP,
  `collected` int(11) NULL DEFAULT 0 COMMENT '课程是否被该学生收藏，1为被收藏，0为未被收藏',
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE INDEX `uk_course_student`(`course_id`, `student_id`) USING BTREE,
  INDEX `student_id`(`student_id`) USING BTREE,
  CONSTRAINT `course_student_ibfk_1` FOREIGN KEY (`course_id`) REFERENCES `course` (`id`) ON DELETE CASCADE ON UPDATE RESTRICT,
  CONSTRAINT `course_student_ibfk_2` FOREIGN KEY (`student_id`) REFERENCES `student` (`id`) ON DELETE CASCADE ON UPDATE RESTRICT
) ENGINE = InnoDB AUTO_INCREMENT = 5 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci COMMENT = '学生选课表' ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of course_student
-- ----------------------------
INSERT INTO `course_student` VALUES (1, 9, 20250005, NULL, 0);
INSERT INTO `course_student` VALUES (2, 19, 20250005, NULL, 0);
INSERT INTO `course_student` VALUES (3, 22, 20250005, NULL, 0);

-- ----------------------------
-- Table structure for knowledge_graph
-- ----------------------------
DROP TABLE IF EXISTS `knowledge_graph`;
CREATE TABLE `knowledge_graph`  (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '知识图谱ID',
  `title` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '图谱标题',
  `description` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL COMMENT '图谱描述',
  `course_id` bigint(20) NOT NULL COMMENT '关联课程ID',
  `graph_type` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT 'COURSE' COMMENT '图谱类型：COURSE-课程图谱，CHAPTER-章节图谱',
  `graph_data` mediumtext CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL COMMENT '图谱数据（JSON格式）',
  `creator_id` bigint(20) NOT NULL COMMENT '创建者ID',
  `status` varchar(20) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT 'active' COMMENT '状态：active-活跃，archived-归档',
  `create_time` datetime NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`) USING BTREE,
  INDEX `idx_course_id`(`course_id`) USING BTREE,
  INDEX `idx_creator_id`(`creator_id`) USING BTREE,
  INDEX `idx_status`(`status`) USING BTREE,
  CONSTRAINT `fk_knowledge_graph_course` FOREIGN KEY (`course_id`) REFERENCES `course` (`id`) ON DELETE CASCADE ON UPDATE RESTRICT,
  CONSTRAINT `fk_knowledge_graph_creator` FOREIGN KEY (`creator_id`) REFERENCES `user` (`id`) ON DELETE CASCADE ON UPDATE RESTRICT
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci COMMENT = '知识图谱表' ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of knowledge_graph
-- ----------------------------
INSERT INTO `knowledge_graph` VALUES (1, '计算机组成原理知识图谱', '知识图谱', 9, 'comprehensive', '{\"id\":null,\"title\":\"计算机组成原理知识图谱\",\"description\":\"知识图谱\",\"nodes\":[{\"id\":\"node_1\",\"name\":\"计算机组成原理\",\"type\":\"topic\",\"level\":3,\"description\":\"本课程介绍计算机系统的基本组成和工作原理。\",\"chapterId\":null,\"sectionId\":null,\"style\":{\"color\":\"#6400ff\",\"size\":50,\"shape\":null,\"fontSize\":null,\"highlighted\":false},\"position\":null,\"properties\":null},{\"id\":\"node_2\",\"name\":\"计算机系统的组成\",\"type\":\"chapter\",\"level\":2,\"description\":\"介绍计算机系统的基本概念和发展历史。\",\"chapterId\":null,\"sectionId\":null,\"style\":{\"color\":\"#7800ff\",\"size\":40,\"shape\":null,\"fontSize\":null,\"highlighted\":false},\"position\":null,\"properties\":null},{\"id\":\"node_3\",\"name\":\"数字逻辑基础\",\"type\":\"chapter\",\"level\":2,\"description\":\"介绍数字电路和逻辑设计的基本原理。\",\"chapterId\":null,\"sectionId\":null,\"style\":{\"color\":\"#7800ff\",\"size\":40,\"shape\":null,\"fontSize\":null,\"highlighted\":false},\"position\":null,\"properties\":null},{\"id\":\"node_4\",\"name\":\"计算机系统的组成\",\"type\":\"concept\",\"level\":1,\"description\":\"介绍计算机系统的基本组成部分。\",\"chapterId\":null,\"sectionId\":null,\"style\":{\"color\":\"#8c00ff\",\"size\":35,\"shape\":null,\"fontSize\":null,\"highlighted\":false},\"position\":null,\"properties\":null},{\"id\":\"node_5\",\"name\":\"层次结构\",\"type\":\"concept\",\"level\":1,\"description\":\"介绍计算机系统的层次结构。\",\"chapterId\":null,\"sectionId\":null,\"style\":{\"color\":\"#8c00ff\",\"size\":35,\"shape\":null,\"fontSize\":null,\"highlighted\":false},\"position\":null,\"properties\":null},{\"id\":\"node_6\",\"name\":\"性能指标\",\"type\":\"concept\",\"level\":1,\"description\":\"介绍计算机系统的性能指标。\",\"chapterId\":null,\"sectionId\":null,\"style\":{\"color\":\"#8c00ff\",\"size\":35,\"shape\":null,\"fontSize\":null,\"highlighted\":false},\"position\":null,\"properties\":null}],\"edges\":[{\"id\":\"edge_1\",\"source\":\"node_1\",\"target\":\"node_2\",\"type\":\"contains\",\"description\":\"包含章节：计算机系统的组成\",\"weight\":1.0,\"style\":{\"color\":null,\"width\":null,\"lineType\":\"solid\",\"showArrow\":true},\"properties\":null},{\"id\":\"edge_2\",\"source\":\"node_1\",\"target\":\"node_3\",\"type\":\"contains\",\"description\":\"包含章节：数字逻辑基础\",\"weight\":1.0,\"style\":{\"color\":null,\"width\":null,\"lineType\":\"solid\",\"showArrow\":true},\"properties\":null},{\"id\":\"edge_3\",\"source\":\"node_2\",\"target\":\"node_4\",\"type\":\"contains\",\"description\":\"包含内容：计算机系统的组成\",\"weight\":1.0,\"style\":{\"color\":null,\"width\":null,\"lineType\":\"solid\",\"showArrow\":true},\"properties\":null},{\"id\":\"edge_4\",\"source\":\"node_2\",\"target\":\"node_5\",\"type\":\"contains\",\"description\":\"包含内容：层次结构\",\"weight\":1.0,\"style\":{\"color\":null,\"width\":null,\"lineType\":\"solid\",\"showArrow\":true},\"properties\":null},{\"id\":\"edge_5\",\"source\":\"node_2\",\"target\":\"node_6\",\"type\":\"contains\",\"description\":\"包含内容：性能指标\",\"weight\":1.0,\"style\":{\"color\":null,\"width\":null,\"lineType\":\"solid\",\"showArrow\":true},\"properties\":null}],\"metadata\":{\"edgeCount\":5,\"generatedAt\":\"2025-07-12T01:31:29.447817\",\"format\":\"echarts\",\"nodeCount\":6,\"rawData\":{\"title\":{\"text\":\"计算机组成原理知识图谱\"},\"tooltip\":{},\"legend\":{\"data\":[\"主题\",\"章节\",\"概念\"]},\"series\":[{\"name\":\"知识图谱\",\"type\":\"graph\",\"layout\":\"force\",\"data\":[{\"id\":\"node_1\",\"name\":\"计算机组成原理\",\"symbolSize\":50,\"category\":0,\"value\":3,\"draggable\":true,\"tooltip\":{\"formatter\":\"本课程介绍计算机系统的基本组成和工作原理。\"}},{\"id\":\"node_2\",\"name\":\"计算机系统的组成\",\"symbolSize\":40,\"category\":1,\"value\":2,\"draggable\":true,\"tooltip\":{\"formatter\":\"介绍计算机系统的基本概念和发展历史。\"}},{\"id\":\"node_3\",\"name\":\"数字逻辑基础\",\"symbolSize\":40,\"category\":1,\"value\":2,\"draggable\":true,\"tooltip\":{\"formatter\":\"介绍数字电路和逻辑设计的基本原理。\"}},{\"id\":\"node_4\",\"name\":\"计算机系统的组成\",\"symbolSize\":35,\"category\":2,\"value\":1,\"draggable\":true,\"tooltip\":{\"formatter\":\"介绍计算机系统的基本组成部分。\"}},{\"id\":\"node_5\",\"name\":\"层次结构\",\"symbolSize\":35,\"category\":2,\"value\":1,\"draggable\":true,\"tooltip\":{\"formatter\":\"介绍计算机系统的层次结构。\"}},{\"id\":\"node_6\",\"name\":\"性能指标\",\"symbolSize\":35,\"category\":2,\"value\":1,\"draggable\":true,\"tooltip\":{\"formatter\":\"介绍计算机系统的性能指标。\"}}],\"links\":[{\"source\":\"node_1\",\"target\":\"node_2\",\"value\":1,\"tooltip\":{\"formatter\":\"包含章节：计算机系统的组成\"}},{\"source\":\"node_1\",\"target\":\"node_3\",\"value\":1,\"tooltip\":{\"formatter\":\"包含章节：数字逻辑基础\"}},{\"source\":\"node_2\",\"target\":\"node_4\",\"value\":1,\"tooltip\":{\"formatter\":\"包含内容：计算机系统的组成\"}},{\"source\":\"node_2\",\"target\":\"node_5\",\"value\":1,\"tooltip\":{\"formatter\":\"包含内容：层次结构\"}},{\"source\":\"node_2\",\"target\":\"node_6\",\"value\":1,\"tooltip\":{\"formatter\":\"包含内容：性能指标\"}}],\"categories\":[{\"name\":\"主题\"},{\"name\":\"章节\"},{\"name\":\"概念\"}],\"roam\":true,\"label\":{\"show\":true,\"position\":\"right\"},\"force\":{\"repulsion\":1000,\"edgeLength\":[50,150]}}]},\"source\":\"dify_agent\"}}', 6, 'published', '2025-07-12 01:31:29', '2025-07-12 01:31:29');

-- ----------------------------
-- Table structure for question
-- ----------------------------
DROP TABLE IF EXISTS `question`;
CREATE TABLE `question`  (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `title` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '题干内容',
  `question_type` enum('single','multiple','true_false','blank','short','code') CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '题型',
  `difficulty` tinyint(4) NOT NULL DEFAULT 3 COMMENT '难度等级，1~5整数',
  `correct_answer` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL COMMENT '标准答案',
  `explanation` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL COMMENT '答案解析',
  `knowledge_point` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '知识点',
  `course_id` bigint(20) NOT NULL,
  `chapter_id` bigint(20) NOT NULL,
  `created_by` bigint(20) NOT NULL COMMENT '出题教师ID',
  `create_time` datetime NULL DEFAULT CURRENT_TIMESTAMP,
  `update_time` datetime NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`) USING BTREE,
  INDEX `course_id`(`course_id`) USING BTREE,
  INDEX `chapter_id`(`chapter_id`) USING BTREE,
  INDEX `question_ibfk_3`(`created_by`) USING BTREE,
  CONSTRAINT `question_ibfk_1` FOREIGN KEY (`course_id`) REFERENCES `course` (`id`) ON DELETE RESTRICT ON UPDATE RESTRICT,
  CONSTRAINT `question_ibfk_2` FOREIGN KEY (`chapter_id`) REFERENCES `chapter` (`id`) ON DELETE RESTRICT ON UPDATE RESTRICT,
  CONSTRAINT `question_ibfk_3` FOREIGN KEY (`created_by`) REFERENCES `user` (`id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE = InnoDB AUTO_INCREMENT = 25 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci COMMENT = '题目表' ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of question
-- ----------------------------
INSERT INTO `question` VALUES (12, '21', 'blank', 3, '2132', '132', 'Vue', 9, 11, 6, '2025-06-30 21:40:18', '2025-07-01 01:54:19');
INSERT INTO `question` VALUES (13, '1=1==1=1111', 'single', 3, 'B', '123123', 'JavaScript', 9, 11, 6, '2025-06-30 21:51:52', '2025-07-01 15:31:34');
INSERT INTO `question` VALUES (17, '我说的对不对（ ）', 'true_false', 3, 'T', '不对', '22231 ', 9, 11, 6, '2025-07-01 02:06:34', '2025-07-01 02:10:22');
INSERT INTO `question` VALUES (18, '??????????????', 'short', 3, '???????????????????????CPU??????????????????????', '?????????????????', '???????', 9, 11, 6, '2025-07-11 21:35:46', '2025-07-11 21:35:46');
INSERT INTO `question` VALUES (19, '?????ALU?????', 'single', 2, 'A', 'ALU????????????CPU?????????', '???????', 9, 11, 6, '2025-07-11 21:35:46', '2025-07-11 21:35:46');
INSERT INTO `question` VALUES (20, '????1101?????????', 'blank', 2, '13', '???1101 = 1*2^3 + 1*2^2 + 0*2^1 + 1*2^0 = 8 + 4 + 0 + 1 = 13', '????', 9, 12, 6, '2025-07-11 21:35:46', '2025-07-11 21:35:46');
INSERT INTO `question` VALUES (21, '????????????', 'multiple', 2, 'A,B,D', '??C??????????', '?????', 9, 11, 6, '2025-07-11 21:35:46', '2025-07-11 21:35:46');
INSERT INTO `question` VALUES (22, '计算机的中央处理器(CPU)的主要功能是什么？', 'short', 3, 'CPU是计算机的核心，主要功能是执行指令、处理数据和控制系统运行。它负责算术运算、逻辑运算、数据传送和程序控制等基本操作。', '中央处理器是计算机的大脑，负责执行指令和处理数据。', '计算机组成原理', 9, 11, 6, '2025-07-11 21:47:01', '2025-07-11 21:47:01');
INSERT INTO `question` VALUES (23, '以下哪项是输入设备？', 'multiple', 2, 'A,B,D', '键盘、鼠标和扫描仪都是输入设备，而打印机是输出设备。', '计算机外设', 9, 11, 6, '2025-07-11 21:47:01', '2025-07-11 21:47:01');
INSERT INTO `question` VALUES (24, '二进制数1010等于十进制数多少？', 'blank', 2, '10', '二进制1010 = 1*2^3 + 0*2^2 + 1*2^1 + 0*2^0 = 8 + 0 + 2 + 0 = 10', '数字逻辑', 9, 12, 6, '2025-07-11 21:47:01', '2025-07-11 21:47:01');

-- ----------------------------
-- Table structure for question_bank
-- ----------------------------
DROP TABLE IF EXISTS `question_bank`;
CREATE TABLE `question_bank`  (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `user_id` bigint(20) NOT NULL COMMENT '出题教师的用户ID',
  `title` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '题库名称',
  `description` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL COMMENT '题库说明',
  `create_time` datetime NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`) USING BTREE,
  INDEX `fk_question_bank_user`(`user_id`) USING BTREE,
  CONSTRAINT `fk_question_bank_user` FOREIGN KEY (`user_id`) REFERENCES `user` (`id`) ON DELETE CASCADE ON UPDATE RESTRICT
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci COMMENT = '题库表' ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of question_bank
-- ----------------------------

-- ----------------------------
-- Table structure for question_bank_item
-- ----------------------------
DROP TABLE IF EXISTS `question_bank_item`;
CREATE TABLE `question_bank_item`  (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `bank_id` bigint(20) NOT NULL,
  `question_id` bigint(20) NOT NULL,
  PRIMARY KEY (`id`) USING BTREE,
  INDEX `bank_id`(`bank_id`) USING BTREE,
  INDEX `question_id`(`question_id`) USING BTREE,
  CONSTRAINT `question_bank_item_ibfk_1` FOREIGN KEY (`bank_id`) REFERENCES `question_bank` (`id`) ON DELETE CASCADE ON UPDATE RESTRICT,
  CONSTRAINT `question_bank_item_ibfk_2` FOREIGN KEY (`question_id`) REFERENCES `question` (`id`) ON DELETE CASCADE ON UPDATE RESTRICT
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci COMMENT = '题库题目关联表' ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of question_bank_item
-- ----------------------------

-- ----------------------------
-- Table structure for question_image
-- ----------------------------
DROP TABLE IF EXISTS `question_image`;
CREATE TABLE `question_image`  (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `question_id` bigint(20) NOT NULL,
  `image_url` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '图片URL或路径',
  `description` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '图片说明',
  `sequence` int(11) NULL DEFAULT 1 COMMENT '图片显示顺序',
  `upload_time` datetime NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`) USING BTREE,
  INDEX `question_id`(`question_id`) USING BTREE,
  CONSTRAINT `question_image_ibfk_1` FOREIGN KEY (`question_id`) REFERENCES `question` (`id`) ON DELETE CASCADE ON UPDATE RESTRICT
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci COMMENT = '题目图片表' ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of question_image
-- ----------------------------

-- ----------------------------
-- Table structure for question_option
-- ----------------------------
DROP TABLE IF EXISTS `question_option`;
CREATE TABLE `question_option`  (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `question_id` bigint(20) NOT NULL,
  `option_label` char(1) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '选项标识 A/B/C/D/T/F',
  `option_text` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '选项内容',
  PRIMARY KEY (`id`) USING BTREE,
  INDEX `question_id`(`question_id`) USING BTREE,
  CONSTRAINT `question_option_ibfk_1` FOREIGN KEY (`question_id`) REFERENCES `question` (`id`) ON DELETE CASCADE ON UPDATE RESTRICT
) ENGINE = InnoDB AUTO_INCREMENT = 15 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci COMMENT = '题目选项表' ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of question_option
-- ----------------------------
INSERT INTO `question_option` VALUES (7, 12, 'A', '');
INSERT INTO `question_option` VALUES (8, 13, 'A', '1231');
INSERT INTO `question_option` VALUES (9, 13, 'B', '12321');
INSERT INTO `question_option` VALUES (10, 13, 'C', '3244');
INSERT INTO `question_option` VALUES (11, 13, 'D', '5234');
INSERT INTO `question_option` VALUES (13, 17, 'T', '正确1');
INSERT INTO `question_option` VALUES (14, 17, 'F', '错误23');
INSERT INTO `question_option` VALUES (15, 19, 'A', '??????');
INSERT INTO `question_option` VALUES (16, 19, 'B', '??????');
INSERT INTO `question_option` VALUES (17, 19, 'C', '??????');
INSERT INTO `question_option` VALUES (18, 19, 'D', '??????');
INSERT INTO `question_option` VALUES (19, 21, 'A', '??');
INSERT INTO `question_option` VALUES (20, 21, 'B', '??');
INSERT INTO `question_option` VALUES (21, 21, 'C', '???');
INSERT INTO `question_option` VALUES (22, 21, 'D', '???');
INSERT INTO `question_option` VALUES (23, 23, 'A', '键盘');
INSERT INTO `question_option` VALUES (24, 23, 'B', '鼠标');
INSERT INTO `question_option` VALUES (25, 23, 'C', '打印机');
INSERT INTO `question_option` VALUES (26, 23, 'D', '扫描仪');

-- ----------------------------
-- Table structure for question_stats
-- ----------------------------
DROP TABLE IF EXISTS `question_stats`;
CREATE TABLE `question_stats`  (
  `question_id` bigint(20) NOT NULL,
  `answer_count` int(11) NULL DEFAULT 0 COMMENT '答题总数',
  `correct_count` int(11) NULL DEFAULT 0 COMMENT '正确人数',
  `accuracy` decimal(5, 2) NULL DEFAULT 0.00 COMMENT '正确率（百分比）',
  PRIMARY KEY (`question_id`) USING BTREE,
  CONSTRAINT `question_stats_ibfk_1` FOREIGN KEY (`question_id`) REFERENCES `question` (`id`) ON DELETE CASCADE ON UPDATE RESTRICT
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci COMMENT = '题目统计信息表' ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of question_stats
-- ----------------------------

-- ----------------------------
-- Table structure for section
-- ----------------------------
DROP TABLE IF EXISTS `section`;
CREATE TABLE `section`  (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '小节ID，主键',
  `chapter_id` bigint(20) NOT NULL COMMENT '所属章节ID，外键',
  `title` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '小节名称',
  `description` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL COMMENT '小节简介',
  `video_url` varchar(1024) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '视频播放地址，可对接OSS',
  `duration` int(11) NULL DEFAULT 0 COMMENT '视频时长(秒)',
  `sort_order` int(11) NOT NULL DEFAULT 0 COMMENT '小节顺序，用于排序',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`) USING BTREE,
  INDEX `idx_chapter_id`(`chapter_id`) USING BTREE,
  INDEX `idx_sort_order`(`sort_order`) USING BTREE,
  CONSTRAINT `fk_section_chapter` FOREIGN KEY (`chapter_id`) REFERENCES `chapter` (`id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE = InnoDB AUTO_INCREMENT = 20 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci COMMENT = '课程小节表' ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of section
-- ----------------------------
INSERT INTO `section` VALUES (14, 11, '计算机系统的组成', '计算机系统的组成', 'D:\\大二下实训4\\SmartClass\\resource\\video\\202506\\2025-07-11 194655.mp4', 45, 1, '2025-06-30 12:33:31', '2025-07-11 19:50:08');
INSERT INTO `section` VALUES (15, 11, '计算机系统的层次结构', '计算机系统的层次结构', NULL, 30, 2, '2025-06-30 14:23:30', '2025-07-03 01:42:04');
INSERT INTO `section` VALUES (18, 11, '计算机的性能指标', '计算机的性能指标', NULL, 25, 3, '2025-07-03 01:42:20', '2025-07-03 01:42:20');

-- ----------------------------
-- Table structure for section_comment
-- ----------------------------
DROP TABLE IF EXISTS `section_comment`;
CREATE TABLE `section_comment`  (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '评论ID，主键',
  `section_id` bigint(20) NOT NULL COMMENT '所属小节ID，外键',
  `user_id` bigint(20) NOT NULL COMMENT '评论人ID，外键',
  `content` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '评论内容',
  `parent_id` bigint(20) NULL DEFAULT NULL COMMENT '父评论ID，为NULL表示一级评论',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`) USING BTREE,
  INDEX `idx_section_id`(`section_id`) USING BTREE,
  INDEX `idx_user_id`(`user_id`) USING BTREE,
  INDEX `idx_parent_id`(`parent_id`) USING BTREE,
  CONSTRAINT `fk_comment_parent` FOREIGN KEY (`parent_id`) REFERENCES `section_comment` (`id`) ON DELETE SET NULL ON UPDATE CASCADE,
  CONSTRAINT `fk_comment_section` FOREIGN KEY (`section_id`) REFERENCES `section` (`id`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `fk_comment_user` FOREIGN KEY (`user_id`) REFERENCES `user` (`id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE = InnoDB AUTO_INCREMENT = 76 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci COMMENT = '小节评论表(讨论区)' ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of section_comment
-- ----------------------------
INSERT INTO `section_comment` VALUES (71, 14, 6, '评论1', NULL, '2025-06-30 14:25:28', '2025-06-30 14:25:28');
INSERT INTO `section_comment` VALUES (72, 14, 6, '12', 71, '2025-06-30 14:25:57', '2025-06-30 14:25:57');
INSERT INTO `section_comment` VALUES (73, 14, 6, '2', NULL, '2025-06-30 14:26:03', '2025-06-30 14:26:03');
INSERT INTO `section_comment` VALUES (74, 14, 6, '12', 73, '2025-06-30 14:26:05', '2025-06-30 14:26:05');
INSERT INTO `section_comment` VALUES (75, 14, 12, '1', NULL, '2025-07-03 16:18:19', '2025-07-03 16:18:19');

-- ----------------------------
-- Table structure for section_progress
-- ----------------------------
DROP TABLE IF EXISTS `section_progress`;
CREATE TABLE `section_progress`  (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '进度记录ID，主键',
  `student_id` bigint(20) NOT NULL COMMENT '学生ID，外键',
  `section_id` bigint(20) NOT NULL COMMENT '小节ID，外键',
  `watched_time` int(11) NOT NULL DEFAULT 0 COMMENT '已观看时间(秒)',
  `is_finished` tinyint(1) NOT NULL DEFAULT 0 COMMENT '是否看完，0-未完成，1-已完成',
  `last_watch_time` datetime NULL DEFAULT NULL COMMENT '上次观看时间',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE INDEX `uk_student_section`(`student_id`, `section_id`) USING BTREE COMMENT '学生+小节唯一约束',
  INDEX `idx_section_id`(`section_id`) USING BTREE,
  CONSTRAINT `fk_progress_section` FOREIGN KEY (`section_id`) REFERENCES `section` (`id`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `fk_progress_student` FOREIGN KEY (`student_id`) REFERENCES `user` (`id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci COMMENT = '学生观看进度记录表' ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of section_progress
-- ----------------------------

-- ----------------------------
-- Table structure for student
-- ----------------------------
DROP TABLE IF EXISTS `student`;
CREATE TABLE `student`  (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '学生ID',
  `user_id` bigint(20) NOT NULL COMMENT '用户ID',
  `enrollment_status` varchar(20) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT 'ENROLLED' COMMENT '学籍状态',
  `gpa` decimal(3, 2) NULL DEFAULT NULL COMMENT 'GPA',
  `gpa_level` varchar(5) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT 'GPA等级',
  `create_time` datetime NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE INDEX `uk_user_id`(`user_id`) USING BTREE,
  CONSTRAINT `fk_student_user` FOREIGN KEY (`user_id`) REFERENCES `user` (`id`) ON DELETE CASCADE ON UPDATE RESTRICT
) ENGINE = InnoDB AUTO_INCREMENT = 20250006 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_unicode_ci COMMENT = '学生表' ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of student
-- ----------------------------
INSERT INTO `student` VALUES (20250005, 12, 'ENROLLED', NULL, NULL, '2025-07-01 21:05:06', '2025-07-01 21:05:06');

-- ----------------------------
-- Table structure for student_answer
-- ----------------------------
DROP TABLE IF EXISTS `student_answer`;
CREATE TABLE `student_answer`  (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `student_id` bigint(20) NOT NULL,
  `assignment_id` bigint(20) NOT NULL,
  `question_id` bigint(20) NOT NULL,
  `answer_content` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL COMMENT '学生答案',
  `is_correct` tinyint(1) NULL DEFAULT NULL COMMENT '是否正确',
  `score` int(11) NULL DEFAULT 0,
  `answer_time` datetime NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`) USING BTREE,
  INDEX `student_id`(`student_id`) USING BTREE,
  INDEX `assignment_id`(`assignment_id`) USING BTREE,
  INDEX `question_id`(`question_id`) USING BTREE,
  CONSTRAINT `student_answer_ibfk_1` FOREIGN KEY (`student_id`) REFERENCES `student` (`id`) ON DELETE RESTRICT ON UPDATE RESTRICT,
  CONSTRAINT `student_answer_ibfk_2` FOREIGN KEY (`assignment_id`) REFERENCES `assignment` (`id`) ON DELETE RESTRICT ON UPDATE RESTRICT,
  CONSTRAINT `student_answer_ibfk_3` FOREIGN KEY (`question_id`) REFERENCES `question` (`id`) ON DELETE RESTRICT ON UPDATE RESTRICT
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci COMMENT = '学生答题记录表' ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of student_answer
-- ----------------------------

-- ----------------------------
-- Table structure for teacher
-- ----------------------------
DROP TABLE IF EXISTS `teacher`;
CREATE TABLE `teacher`  (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '教师ID',
  `user_id` bigint(20) NOT NULL COMMENT '用户ID',
  `department` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '所属院系',
  `title` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '职称',
  `education` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '学历',
  `specialty` varchar(200) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '专业领域',
  `introduction` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL COMMENT '个人简介',
  `office_location` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '办公地点',
  `office_hours` varchar(200) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '办公时间',
  `contact_email` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '联系邮箱',
  `contact_phone` varchar(20) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '联系电话',
  `status` varchar(20) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT 'ACTIVE' COMMENT '状态',
  `hire_date` datetime NULL DEFAULT NULL COMMENT '入职日期',
  `create_time` datetime NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE INDEX `uk_user_id`(`user_id`) USING BTREE,
  INDEX `idx_department`(`department`) USING BTREE,
  CONSTRAINT `fk_teacher_user` FOREIGN KEY (`user_id`) REFERENCES `user` (`id`) ON DELETE CASCADE ON UPDATE RESTRICT
) ENGINE = InnoDB AUTO_INCREMENT = 2025005 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_unicode_ci COMMENT = '教师表' ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of teacher
-- ----------------------------
INSERT INTO `teacher` VALUES (2025000, 2, '计算机科学与技术学院', '教授', '博士', '人工智能', '张教授是人工智能领域的专家', 'A栋201', '周一至周五 9:00-17:00', 'teacher1@example.com', NULL, 'ACTIVE', NULL, '2025-06-29 02:09:12', '2025-06-29 02:09:12');
INSERT INTO `teacher` VALUES (2025001, 3, '数学学院', '副教授', '博士', '应用数学', '李教授专注于应用数学研究', 'B栋305', '周一至周五 10:00-16:00', 'teacher2@example.com', NULL, 'ACTIVE', NULL, '2025-06-29 02:09:12', '2025-06-29 02:09:12');
INSERT INTO `teacher` VALUES (2025002, 6, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, 'ACTIVE', NULL, '2025-06-29 02:12:20', '2025-06-29 02:12:20');
INSERT INTO `teacher` VALUES (2025003, 7, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, 'ACTIVE', NULL, '2025-06-29 12:29:23', '2025-06-29 12:29:23');
INSERT INTO `teacher` VALUES (2025004, 9, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, 'ACTIVE', NULL, '2025-06-29 18:16:10', '2025-06-29 18:16:10');

-- ----------------------------
-- Table structure for user
-- ----------------------------
DROP TABLE IF EXISTS `user`;
CREATE TABLE `user`  (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '用户ID',
  `username` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '用户名',
  `password` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '密码(加密)',
  `email` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '邮箱',
  `real_name` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '真实姓名',
  `avatar` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '头像URL',
  `role` varchar(20) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '用户角色',
  `status` varchar(20) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT 'ACTIVE' COMMENT '状态',
  `create_time` datetime NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE INDEX `uk_username`(`username`) USING BTREE,
  UNIQUE INDEX `uk_email`(`email`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 13 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_unicode_ci COMMENT = '用户表' ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of user
-- ----------------------------
INSERT INTO `user` VALUES (1, 'admin', '$2a$10$N.zmdr9k7uOCQb376NoUnuTJ8iAt6Z5EHsNSEYy4k4onaZDe17ZOS', 'admin@example.com', '系统管理员', NULL, 'ADMIN', 'ACTIVE', '2025-06-29 02:09:12', '2025-06-29 02:09:12');
INSERT INTO `user` VALUES (2, 'teacher1', '$2a$10$N.zmdr9k7uOCQb376NoUnuTJ8iAt6Z5EHsNSEYy4k4onaZDe17ZOS', 'teacher1@example.com', '张教授', NULL, 'TEACHER', 'ACTIVE', '2025-06-29 02:09:12', '2025-06-29 02:09:12');
INSERT INTO `user` VALUES (3, 'teacher2', '$2a$10$N.zmdr9k7uOCQb376NoUnuTJ8iAt6Z5EHsNSEYy4k4onaZDe17ZOS', 'teacher2@example.com', '李教授', NULL, 'TEACHER', 'ACTIVE', '2025-06-29 02:09:12', '2025-06-29 02:09:12');
INSERT INTO `user` VALUES (6, 'test01', '$2a$10$yinM6zXGIZ7AWSXA8QquqOE6v28EEcb7hpaOoz53ikmbV7bT8jWJe', '2825507827@qq.com', '张教授', NULL, 'TEACHER', 'ACTIVE', '2025-06-29 02:12:20', '2025-07-03 01:16:42');
INSERT INTO `user` VALUES (7, 'test02', '$2a$10$KoS8uj6MG4W.9W3lRL78yuFtC9rYZmFbQIxX/BmIX7nAJLkfME7he', '2825507825@qq.com', '测试教师', NULL, 'TEACHER', 'ACTIVE', '2025-06-29 12:29:23', '2025-06-29 12:29:23');
INSERT INTO `user` VALUES (9, 'th1', '$2a$10$jkqmHsAkW.soiGElTZyk8OU2mC6YEUxCWizcgGsOly2wfQGljHY6.', '28254437825@qq.com', '测试', NULL, 'TEACHER', 'ACTIVE', '2025-06-29 18:16:10', '2025-06-29 18:16:10');
INSERT INTO `user` VALUES (12, 'stu01', '$2a$10$hwXoEKCcsQXLLSM5AoHPh.qvWzWSKalBGq8XcCNxSsWMCy.9CQHvS', '20234042@stu.neu.edu.cn', '测试学生', NULL, 'STUDENT', 'ACTIVE', '2025-07-01 21:05:06', '2025-07-01 21:05:06');

-- ----------------------------
-- Table structure for wrong_question
-- ----------------------------
DROP TABLE IF EXISTS `wrong_question`;
CREATE TABLE `wrong_question`  (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '主键',
  `student_id` bigint(20) NOT NULL COMMENT '学生ID',
  `question_id` bigint(20) NOT NULL COMMENT '题目ID',
  `assignment_id` bigint(20) NULL DEFAULT NULL COMMENT '所属作业或考试ID',
  `submission_id` bigint(20) NULL DEFAULT NULL COMMENT '作业提交ID',
  `wrong_time` datetime NULL DEFAULT CURRENT_TIMESTAMP COMMENT '记录时间',
  `student_answer` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL COMMENT '学生当时的答案',
  `correct_answer` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL COMMENT '正确答案（冗余存储）',
  PRIMARY KEY (`id`) USING BTREE,
  INDEX `student_id`(`student_id`) USING BTREE,
  INDEX `question_id`(`question_id`) USING BTREE,
  CONSTRAINT `fk_wrong_question_question` FOREIGN KEY (`question_id`) REFERENCES `question` (`id`) ON DELETE CASCADE ON UPDATE RESTRICT,
  CONSTRAINT `fk_wrong_question_student` FOREIGN KEY (`student_id`) REFERENCES `student` (`id`) ON DELETE CASCADE ON UPDATE RESTRICT
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci COMMENT = '学生错题本' ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of wrong_question
-- ----------------------------

SET FOREIGN_KEY_CHECKS = 1;
