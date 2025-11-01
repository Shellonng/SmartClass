/*
SQLyog Ultimate v8.71 
MySQL - 8.0.13 : Database - education_platform
*********************************************************************
*/

/*!40101 SET NAMES utf8 */;

/*!40101 SET SQL_MODE=''*/;

/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;
CREATE DATABASE /*!32312 IF NOT EXISTS*/`education_platform` /*!40100 DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci */;

USE `education_platform`;

/*Table structure for table `assignment` */

DROP TABLE IF EXISTS `assignment`;

CREATE TABLE `assignment` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `title` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '作业或考试标题',
  `course_id` bigint(20) NOT NULL,
  `user_id` bigint(20) NOT NULL COMMENT '发布作业的用户ID',
  `type` enum('homework','exam') CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci DEFAULT 'homework',
  `description` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci,
  `start_time` datetime DEFAULT NULL,
  `end_time` datetime DEFAULT NULL,
  `create_time` datetime DEFAULT CURRENT_TIMESTAMP,
  `status` tinyint(4) NOT NULL DEFAULT '0' COMMENT '发布状态：0未发布，1已发布',
  `update_time` datetime DEFAULT NULL,
  `mode` enum('question','file') CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL DEFAULT 'question' COMMENT '作业模式：question-答题型，file-上传型',
  `time_limit` int(11) DEFAULT NULL COMMENT '时间限制（分钟）',
  `total_score` int(11) DEFAULT '100' COMMENT '总分',
  `duration` int(11) DEFAULT NULL COMMENT '考试时长（分钟）',
  `allowed_file_types` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci COMMENT '允许的文件类型（JSON格式）',
  `max_file_size` int(11) DEFAULT '10' COMMENT '最大文件大小（MB）',
  `reference_answer` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci COMMENT '参考答案（用于智能批改）',
  PRIMARY KEY (`id`) USING BTREE,
  KEY `course_id` (`course_id`) USING BTREE,
  KEY `fk_assignment_user` (`user_id`) USING BTREE,
  CONSTRAINT `assignment_ibfk_1` FOREIGN KEY (`course_id`) REFERENCES `course` (`id`) ON DELETE RESTRICT ON UPDATE RESTRICT,
  CONSTRAINT `fk_assignment_user` FOREIGN KEY (`user_id`) REFERENCES `user` (`id`) ON DELETE CASCADE ON UPDATE RESTRICT
) ENGINE=InnoDB AUTO_INCREMENT=32 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci ROW_FORMAT=DYNAMIC COMMENT='作业或考试表';

/*Data for the table `assignment` */

insert  into `assignment`(`id`,`title`,`course_id`,`user_id`,`type`,`description`,`start_time`,`end_time`,`create_time`,`status`,`update_time`,`mode`,`time_limit`,`total_score`,`duration`,`allowed_file_types`,`max_file_size`,`reference_answer`) values (30,'计算机组成原理-期中测试',9,6,'exam','严禁作弊','2025-07-11 14:00:08','2025-07-13 16:00:00','2025-07-12 04:50:52',1,'2025-07-12 06:45:15','question',NULL,100,NULL,NULL,10,NULL),(31,'完成第一章实验并提交实验报告',9,6,'homework','按照规范提交报告文档','2025-07-11 05:15:39','2025-07-13 05:15:41','2025-07-12 04:58:18',1,'2025-07-12 06:05:04','file',NULL,100,NULL,NULL,10,NULL);

/*Table structure for table `assignment_config` */

DROP TABLE IF EXISTS `assignment_config`;

CREATE TABLE `assignment_config` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `assignment_id` bigint(20) NOT NULL COMMENT '作业ID',
  `knowledge_points` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci COMMENT '知识点范围（JSON格式）',
  `difficulty` enum('EASY','MEDIUM','HARD') CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci DEFAULT 'MEDIUM' COMMENT '难度级别',
  `question_count` int(11) DEFAULT '10' COMMENT '题目总数',
  `question_types` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci COMMENT '题型分布（JSON格式）',
  `additional_requirements` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci COMMENT '额外要求',
  `create_time` datetime DEFAULT CURRENT_TIMESTAMP,
  `update_time` datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE KEY `uk_assignment_config` (`assignment_id`) USING BTREE,
  CONSTRAINT `fk_assignment_config_assignment` FOREIGN KEY (`assignment_id`) REFERENCES `assignment` (`id`) ON DELETE CASCADE ON UPDATE RESTRICT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci ROW_FORMAT=DYNAMIC COMMENT='作业配置表';

/*Data for the table `assignment_config` */

/*Table structure for table `assignment_question` */

DROP TABLE IF EXISTS `assignment_question`;

CREATE TABLE `assignment_question` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `assignment_id` bigint(20) NOT NULL,
  `question_id` bigint(20) NOT NULL,
  `score` int(11) DEFAULT '5' COMMENT '该题满分',
  `sequence` int(11) DEFAULT '1' COMMENT '题目顺序',
  PRIMARY KEY (`id`) USING BTREE,
  KEY `assignment_id` (`assignment_id`) USING BTREE,
  KEY `question_id` (`question_id`) USING BTREE,
  CONSTRAINT `assignment_question_ibfk_1` FOREIGN KEY (`assignment_id`) REFERENCES `assignment` (`id`) ON DELETE CASCADE ON UPDATE RESTRICT,
  CONSTRAINT `assignment_question_ibfk_2` FOREIGN KEY (`question_id`) REFERENCES `question` (`id`) ON DELETE CASCADE ON UPDATE RESTRICT
) ENGINE=InnoDB AUTO_INCREMENT=64 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci ROW_FORMAT=DYNAMIC COMMENT='作业题目关联表';

/*Data for the table `assignment_question` */

insert  into `assignment_question`(`id`,`assignment_id`,`question_id`,`score`,`sequence`) values (42,30,25,5,1),(43,30,26,5,2),(44,30,27,5,3),(45,30,28,5,4),(46,30,29,5,5),(47,30,23,5,6),(48,30,30,5,7),(49,30,31,5,8),(50,30,32,5,9),(51,30,33,5,10),(52,30,34,5,11),(53,30,35,5,12),(54,30,36,5,13),(55,30,37,5,14),(56,30,39,5,15),(57,30,40,5,16),(58,30,41,5,17),(59,30,43,5,18),(60,30,46,10,19);

/*Table structure for table `assignment_submission` */

DROP TABLE IF EXISTS `assignment_submission`;

CREATE TABLE `assignment_submission` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '主键ID',
  `assignment_id` bigint(20) NOT NULL COMMENT '作业ID',
  `student_id` bigint(20) NOT NULL COMMENT '学生ID',
  `status` int(11) NOT NULL DEFAULT '0' COMMENT '状态：0-未提交，1-已提交未批改，2-已批改',
  `score` int(11) DEFAULT NULL COMMENT '得分',
  `feedback` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci COMMENT '教师反馈',
  `submit_time` datetime DEFAULT NULL COMMENT '提交时间',
  `grade_time` datetime DEFAULT NULL COMMENT '批改时间',
  `graded_by` bigint(20) DEFAULT NULL COMMENT '批改人ID',
  `content` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci COMMENT '提交内容',
  `create_time` datetime DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  `file_name` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci DEFAULT NULL COMMENT '文件名称',
  `file_path` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci DEFAULT NULL COMMENT '文件路径',
  PRIMARY KEY (`id`) USING BTREE,
  KEY `idx_assignment_id` (`assignment_id`) USING BTREE,
  KEY `idx_student_id` (`student_id`) USING BTREE,
  KEY `idx_status` (`status`) USING BTREE,
  CONSTRAINT `fk_submission_assignment` FOREIGN KEY (`assignment_id`) REFERENCES `assignment` (`id`) ON DELETE CASCADE ON UPDATE RESTRICT,
  CONSTRAINT `fk_submission_student` FOREIGN KEY (`student_id`) REFERENCES `user` (`id`) ON DELETE CASCADE ON UPDATE RESTRICT
) ENGINE=InnoDB AUTO_INCREMENT=48 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci ROW_FORMAT=DYNAMIC COMMENT='作业提交记录表';

/*Data for the table `assignment_submission` */

insert  into `assignment_submission`(`id`,`assignment_id`,`student_id`,`status`,`score`,`feedback`,`submit_time`,`grade_time`,`graded_by`,`content`,`create_time`,`update_time`,`file_name`,`file_path`) values (46,31,12,2,90,'再接再厉！','2025-07-12 06:18:09','2025-07-12 06:40:10',1,NULL,'2025-07-12 06:05:04','2025-07-12 06:40:10','2511760.pdf','D:/my_git_code/SmartClass/resource/file/assignments/31/12/9310c9db-77d6-4c2d-b3ee-4c3e822ba700.pdf'),(47,30,12,1,10,NULL,'2025-07-12 09:11:31',NULL,NULL,NULL,'2025-07-12 06:45:43','2025-07-12 09:11:31',NULL,NULL);

/*Table structure for table `assignment_submission_answer` */

DROP TABLE IF EXISTS `assignment_submission_answer`;

CREATE TABLE `assignment_submission_answer` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '主键',
  `submission_id` bigint(20) NOT NULL COMMENT '提交ID，关联assignment_submission',
  `question_id` bigint(20) NOT NULL COMMENT '题目ID',
  `student_answer` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci COMMENT '学生的答案',
  `is_correct` tinyint(1) DEFAULT NULL COMMENT '是否正确：1-正确，0-错误，NULL-未批改',
  `score` int(11) DEFAULT NULL COMMENT '得分（教师批改后记录）',
  `comment` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci COMMENT '教师对该题的点评（可选）',
  `create_time` datetime DEFAULT CURRENT_TIMESTAMP,
  `update_time` datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`) USING BTREE,
  KEY `submission_id` (`submission_id`) USING BTREE,
  KEY `question_id` (`question_id`) USING BTREE,
  CONSTRAINT `fk_submission_answer_question` FOREIGN KEY (`question_id`) REFERENCES `question` (`id`) ON DELETE CASCADE ON UPDATE RESTRICT,
  CONSTRAINT `fk_submission_answer_submission` FOREIGN KEY (`submission_id`) REFERENCES `assignment_submission` (`id`) ON DELETE CASCADE ON UPDATE RESTRICT
) ENGINE=InnoDB AUTO_INCREMENT=93 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci ROW_FORMAT=DYNAMIC COMMENT='学生答题记录表';

/*Data for the table `assignment_submission_answer` */

insert  into `assignment_submission_answer`(`id`,`submission_id`,`question_id`,`student_answer`,`is_correct`,`score`,`comment`,`create_time`,`update_time`) values (23,47,25,'A',0,0,NULL,'2025-07-12 08:03:04','2025-07-12 09:11:31'),(24,47,23,'A,C,D',0,0,NULL,'2025-07-12 08:56:54','2025-07-12 09:11:31'),(25,47,30,'A,B,D',0,0,NULL,'2025-07-12 09:01:17','2025-07-12 09:11:31'),(26,47,31,'B,C,E',0,0,NULL,'2025-07-12 09:11:31','2025-07-12 09:11:31'),(27,47,27,'B',0,0,NULL,'2025-07-12 09:11:31','2025-07-12 09:11:31'),(28,47,29,'D',1,5,NULL,'2025-07-12 09:11:31','2025-07-12 09:11:31'),(29,47,28,'B',0,0,NULL,'2025-07-12 09:11:31','2025-07-12 09:11:31'),(30,47,26,'B',0,0,NULL,'2025-07-12 09:11:31','2025-07-12 09:11:31'),(31,47,34,'F',1,5,NULL,'2025-07-12 09:11:31','2025-07-12 09:11:31'),(32,47,32,'B,C,D',0,0,NULL,'2025-07-12 09:11:31','2025-07-12 09:11:31'),(33,47,33,'F',0,0,NULL,'2025-07-12 09:11:31','2025-07-12 09:11:31'),(34,47,35,'T',0,0,NULL,'2025-07-12 09:11:31','2025-07-12 09:11:31'),(35,47,41,'123',0,0,NULL,'2025-07-12 09:11:31','2025-07-12 09:11:31'),(36,47,40,'不知',0,0,NULL,'2025-07-12 09:11:31','2025-07-12 09:11:31'),(37,47,39,'不知',0,0,NULL,'2025-07-12 09:11:31','2025-07-12 09:11:31'),(38,47,37,'1',0,0,NULL,'2025-07-12 09:11:31','2025-07-12 09:11:31'),(39,47,36,'F',0,0,NULL,'2025-07-12 09:11:31','2025-07-12 09:11:31'),(40,47,46,'123',0,0,NULL,'2025-07-12 09:11:31','2025-07-12 09:11:31'),(41,47,43,'321',0,0,NULL,'2025-07-12 09:11:31','2025-07-12 09:11:31');

/*Table structure for table `chapter` */

DROP TABLE IF EXISTS `chapter`;

CREATE TABLE `chapter` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '章节ID，主键',
  `course_id` bigint(20) NOT NULL COMMENT '所属课程ID，外键',
  `title` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '章节名称',
  `description` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci,
  `sort_order` int(11) NOT NULL,
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`) USING BTREE,
  KEY `idx_course_id` (`course_id`) USING BTREE,
  CONSTRAINT `fk_chapter_course` FOREIGN KEY (`course_id`) REFERENCES `course` (`id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=27 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci ROW_FORMAT=DYNAMIC COMMENT='课程章节表';

/*Data for the table `chapter` */

insert  into `chapter`(`id`,`course_id`,`title`,`description`,`sort_order`,`create_time`,`update_time`) values (11,9,'计算机系统的组成','介绍计算机系统的基本概念和发展历史',1,'2025-06-29 16:22:39','2025-07-03 01:41:18'),(12,9,'数字逻辑基础','介绍数字电路和逻辑设计的基本原理',2,'2025-06-29 21:07:41','2025-07-03 01:42:55'),(13,9,'指令系统','介绍计算机的指令系统架构和设计',3,'2025-07-12 03:52:26','2025-07-12 03:52:26'),(14,9,'CPU的结构与功能','详细分析中央处理器的结构组成和功能原理',4,'2025-07-12 03:52:26','2025-07-12 03:52:26'),(15,9,'存储器层次结构','介绍计算机存储系统的层次结构和工作原理',5,'2025-07-12 03:52:26','2025-07-12 03:52:26'),(16,9,'输入输出系统','讲解计算机输入输出系统的基本原理和实现方式',6,'2025-07-12 03:52:26','2025-07-12 03:52:26'),(17,9,'总线结构','介绍计算机系统内部的总线结构和通信机制',7,'2025-07-12 03:52:26','2025-07-12 03:52:26'),(18,9,'并行与多处理器系统','讲解计算机系统中的并行处理技术和多处理器架构',8,'2025-07-12 03:52:26','2025-07-12 03:52:26'),(19,9,'计算机系统基础','介绍计算机系统的基本概念、发展历史和基础架构',1,'2025-07-12 03:59:23','2025-07-12 03:59:23'),(20,9,'数据的表示与运算','讲解数据在计算机中的表示方法和运算原理',2,'2025-07-12 03:59:23','2025-07-12 03:59:23');

/*Table structure for table `class_student` */

DROP TABLE IF EXISTS `class_student`;

CREATE TABLE `class_student` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `class_id` bigint(20) NOT NULL,
  `student_id` bigint(20) NOT NULL,
  `join_time` datetime DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE KEY `uk_class_student` (`class_id`,`student_id`) USING BTREE,
  KEY `student_id` (`student_id`) USING BTREE,
  CONSTRAINT `class_student_ibfk_1` FOREIGN KEY (`class_id`) REFERENCES `course_class` (`id`) ON DELETE CASCADE ON UPDATE RESTRICT,
  CONSTRAINT `class_student_ibfk_2` FOREIGN KEY (`student_id`) REFERENCES `student` (`id`) ON DELETE CASCADE ON UPDATE RESTRICT
) ENGINE=InnoDB AUTO_INCREMENT=2 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci ROW_FORMAT=DYNAMIC COMMENT='班级学生关联表';

/*Data for the table `class_student` */

insert  into `class_student`(`id`,`class_id`,`student_id`,`join_time`) values (1,10,20250005,NULL);

/*Table structure for table `course` */

DROP TABLE IF EXISTS `course`;

CREATE TABLE `course` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '课程ID',
  `title` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '课程名称',
  `description` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci COMMENT '课程描述',
  `cover_image` varchar(500) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '课程封面图片',
  `credit` decimal(3,1) DEFAULT '3.0' COMMENT '学分',
  `course_type` varchar(20) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT '必修课' COMMENT '课程类型',
  `start_time` datetime DEFAULT NULL COMMENT '开始时间',
  `end_time` datetime DEFAULT NULL COMMENT '结束时间',
  `teacher_id` bigint(20) NOT NULL COMMENT '教师ID',
  `status` varchar(20) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT '未开始' COMMENT '课程状态',
  `term` varchar(20) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '学期',
  `student_count` int(11) DEFAULT '0' COMMENT '学生数量',
  `average_score` decimal(5,2) DEFAULT NULL COMMENT '平均分数',
  `create_time` datetime DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`) USING BTREE,
  KEY `idx_teacher_id` (`teacher_id`) USING BTREE,
  KEY `idx_term` (`term`) USING BTREE,
  KEY `idx_status` (`status`) USING BTREE,
  CONSTRAINT `fk_course_teacher` FOREIGN KEY (`teacher_id`) REFERENCES `teacher` (`id`) ON DELETE RESTRICT ON UPDATE RESTRICT
) ENGINE=InnoDB AUTO_INCREMENT=30 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci ROW_FORMAT=DYNAMIC COMMENT='课程表';

/*Data for the table `course` */

insert  into `course`(`id`,`title`,`description`,`cover_image`,`credit`,`course_type`,`start_time`,`end_time`,`teacher_id`,`status`,`term`,`student_count`,`average_score`,`create_time`,`update_time`) values (1,'Java编程基础','Java编程语言入门课程',NULL,'3.0','必修课','2024-09-01 08:00:00','2024-12-31 17:00:00',2025000,'未开始','2024-2025-1',0,NULL,'2025-06-29 02:09:12','2025-06-29 02:09:12'),(2,'高等数学','高等数学基础课程',NULL,'4.0','必修课','2024-09-01 08:00:00','2024-12-31 17:00:00',2025001,'已结束','2024-2025-1',0,NULL,'2025-06-29 02:09:12','2025-06-29 02:09:12'),(3,'人工智能导论','人工智能基础理论与应用',NULL,'3.0','选修课','2024-09-01 08:00:00','2024-12-31 17:00:00',2025000,'未开始','2024-2025-1',0,NULL,'2025-06-29 02:09:12','2025-06-29 02:09:12'),(5,'测试课程','123','','3.0','必修课','2025-07-03 02:15:34','2025-07-11 02:15:36',2025002,'未开始','2024-2025-1',0,NULL,'2025-06-29 02:15:41','2025-06-29 02:15:41'),(6,'123','123','/api/photo/202507/799f6631a8604aeaa5a59b4416cf43da.png','3.0','必修课','2025-07-23 02:15:34','2025-08-01 02:15:36',2025002,'未开始','2024-2025-1',0,NULL,'2025-06-29 02:15:59','2025-07-02 11:51:33'),(9,'计算机组成原理','本课程介绍计算机系统的基本组成和工作原理，包括数字逻辑、CPU结构、存储系统、输入输出系统等内容。','https://img.freepik.com/free-vector/english-school-landing-page-template_23-2148475038.jpg','4.0','必修课','2025-06-04 02:52:20','2025-06-26 02:52:22',2025002,'已结束','2024-2025-1',1,NULL,'2025-06-29 02:52:54','2025-07-03 01:13:39'),(19,'Java程序设计','Java编程基础课程，零基础向对象编程思想和Java核心技术，为开发企业级应用打下基础','https://img.freepik.com/free-vector/programming-concept-illustration_114360-1670.jpg','3.0','必修课','2024-05-01 00:00:00','2024-12-31 00:00:00',2025002,'已结束','2024-2025-1',16421,NULL,'2025-07-02 02:13:16','2025-07-02 02:13:16'),(20,'数据结构与算法','深入学习常用数据结构和算法设计技巧，包括数组、链表、栈、队列、树、图以及常见算法','https://img.freepik.com/free-vector/programming-concept-illustration_114360-1213.jpg','4.0','必修课','2024-05-01 00:00:00','2024-12-31 00:00:00',2025002,'已结束','2024-2025-1',9850,NULL,'2025-07-02 02:13:16','2025-07-02 02:13:16'),(21,'Python程序基础','零基础入门Python编程，掌握Python基本语法、数据类型、控制结构、函数和模块开发','https://img.freepik.com/free-vector/programming-concept-illustration_114360-1351.jpg','3.0','必修课','2024-05-01 00:00:00','2024-12-31 00:00:00',2025002,'已结束','2024-2025-1',23190,NULL,'2025-07-02 02:13:16','2025-07-02 02:13:16'),(22,'微观经济学原理','介绍微观经济学的基本理论，包括供需关系、市场结构、消费行为、生产理论等','https://img.freepik.com/free-vector/economy-concept-illustration_114360-7385.jpg','3.0','必修课','2024-06-01 00:00:00','2024-12-31 00:00:00',2025002,'已结束','2024-2025-1',11201,NULL,'2025-07-02 02:13:16','2025-07-02 02:13:16'),(23,'线性代数','系统学习线性代数的基本概念和方法，包括向量运算、行列式、向量空间、特征值和特征向量','https://img.freepik.com/free-vector/mathematics-concept-illustration_114360-3972.jpg','4.0','必修课','2024-06-01 00:00:00','2024-12-31 00:00:00',2025002,'已结束','2024-2025-1',13580,NULL,'2025-07-02 02:13:16','2025-07-02 02:13:16'),(24,'大学物理（力学部分）','系统讲授经典力学的基本概念、定律和方法，包括刚体力学、振动力学、流体力学等','https://img.freepik.com/free-vector/physics-concept-illustration_114360-3972.jpg','3.0','必修课','2024-06-01 00:00:00','2024-12-31 00:00:00',2025002,'已结束','2024-2025-1',12680,NULL,'2025-07-02 02:13:16','2025-07-02 02:13:16'),(25,'高等数学（上）','本课程系统讲授高等数学的基本概念、理论和方法，包括极限、导数、微积分等','https://img.freepik.com/free-vector/hand-drawn-mathematics-background_23-2148157511.jpg','4.0','必修课','2025-07-01 11:57:20','2025-07-17 11:57:24',2025002,'进行中','2024-2025-1',15420,NULL,'2025-07-02 02:13:16','2025-07-02 11:57:30'),(26,'大学英语综合教程','提升英语听说读写综合能力，涵盖语法、词汇、阅读理解和写作技巧，为四六级考试做准备','https://img.freepik.com/free-vector/english-school-landing-page-template_23-2148475038.jpg','3.0','必修课','2024-06-01 00:00:00','2024-12-31 00:00:00',2025002,'已结束','2024-2025-1',18700,NULL,'2025-07-02 02:13:16','2025-07-02 02:13:16');

/*Table structure for table `course_class` */

DROP TABLE IF EXISTS `course_class`;

CREATE TABLE `course_class` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '班级ID',
  `course_id` bigint(20) DEFAULT NULL,
  `teacher_id` bigint(20) NOT NULL,
  `name` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `description` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci,
  `is_default` tinyint(1) DEFAULT '0',
  `create_time` datetime DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`) USING BTREE,
  KEY `course_class_ibfk_1` (`course_id`) USING BTREE,
  KEY `course_class_ibfk_2` (`teacher_id`) USING BTREE,
  CONSTRAINT `course_class_ibfk_1` FOREIGN KEY (`course_id`) REFERENCES `course` (`id`) ON DELETE SET NULL ON UPDATE RESTRICT,
  CONSTRAINT `course_class_ibfk_2` FOREIGN KEY (`teacher_id`) REFERENCES `user` (`id`) ON DELETE CASCADE ON UPDATE RESTRICT
) ENGINE=InnoDB AUTO_INCREMENT=17 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci ROW_FORMAT=DYNAMIC COMMENT='课程班级表';

/*Data for the table `course_class` */

insert  into `course_class`(`id`,`course_id`,`teacher_id`,`name`,`description`,`is_default`,`create_time`) values (10,9,6,'软件工程2306班','123',0,'2025-07-02 14:08:30');

/*Table structure for table `course_enrollment_request` */

DROP TABLE IF EXISTS `course_enrollment_request`;

CREATE TABLE `course_enrollment_request` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '申请ID',
  `student_id` bigint(20) NOT NULL COMMENT '学生ID',
  `course_id` bigint(20) NOT NULL COMMENT '申请加入的课程ID',
  `status` tinyint(4) DEFAULT '0' COMMENT '申请状态：0=待审核 1=已通过 2=已拒绝',
  `reason` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci COMMENT '学生申请理由',
  `review_comment` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci COMMENT '教师审核意见',
  `submit_time` datetime DEFAULT CURRENT_TIMESTAMP,
  `review_time` datetime DEFAULT NULL COMMENT '审核时间',
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE KEY `uk_student_course` (`student_id`,`course_id`) USING BTREE,
  KEY `course_id` (`course_id`) USING BTREE,
  CONSTRAINT `course_enrollment_request_ibfk_1` FOREIGN KEY (`student_id`) REFERENCES `student` (`id`) ON DELETE CASCADE ON UPDATE RESTRICT,
  CONSTRAINT `course_enrollment_request_ibfk_2` FOREIGN KEY (`course_id`) REFERENCES `course` (`id`) ON DELETE CASCADE ON UPDATE RESTRICT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci ROW_FORMAT=DYNAMIC COMMENT='学生选课申请表';

/*Data for the table `course_enrollment_request` */

/*Table structure for table `course_resource` */

DROP TABLE IF EXISTS `course_resource`;

CREATE TABLE `course_resource` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '资源ID',
  `course_id` bigint(20) NOT NULL COMMENT '所属课程ID',
  `name` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '资源名称',
  `file_type` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '文件类型，如pdf、doc、ppt等',
  `file_size` bigint(20) NOT NULL COMMENT '文件大小(字节)',
  `file_url` varchar(500) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '文件URL',
  `description` varchar(500) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci DEFAULT NULL COMMENT '资源描述',
  `download_count` int(11) DEFAULT '0' COMMENT '下载次数',
  `upload_user_id` bigint(20) NOT NULL COMMENT '上传用户ID',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`) USING BTREE,
  KEY `idx_course_id` (`course_id`) USING BTREE COMMENT '课程ID索引',
  KEY `idx_upload_user_id` (`upload_user_id`) USING BTREE COMMENT '上传用户ID索引'
) ENGINE=InnoDB AUTO_INCREMENT=6 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci ROW_FORMAT=DYNAMIC COMMENT='课程资源表';

/*Data for the table `course_resource` */

insert  into `course_resource`(`id`,`course_id`,`name`,`file_type`,`file_size`,`file_url`,`description`,`download_count`,`upload_user_id`,`create_time`,`update_time`) values (1,9,'1','pdf',325858,'/files/resources/9/202506/7acbf820bec64795bedaca556c235c4a.pdf','1',4,6,'2025-06-30 16:18:14','2025-06-30 16:18:14'),(5,9,'测试','png',48173,'/files/resources/9/202507/a54ec5a02dba4fc9882aaa1f23caf063.png','1',2,6,'2025-07-02 20:39:01','2025-07-02 20:39:01');

/*Table structure for table `course_student` */

DROP TABLE IF EXISTS `course_student`;

CREATE TABLE `course_student` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `course_id` bigint(20) NOT NULL,
  `student_id` bigint(20) NOT NULL,
  `enroll_time` datetime DEFAULT CURRENT_TIMESTAMP,
  `collected` int(11) DEFAULT '0' COMMENT '课程是否被该学生收藏，1为被收藏，0为未被收藏',
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE KEY `uk_course_student` (`course_id`,`student_id`) USING BTREE,
  KEY `student_id` (`student_id`) USING BTREE,
  CONSTRAINT `course_student_ibfk_1` FOREIGN KEY (`course_id`) REFERENCES `course` (`id`) ON DELETE CASCADE ON UPDATE RESTRICT,
  CONSTRAINT `course_student_ibfk_2` FOREIGN KEY (`student_id`) REFERENCES `student` (`id`) ON DELETE CASCADE ON UPDATE RESTRICT
) ENGINE=InnoDB AUTO_INCREMENT=5 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci ROW_FORMAT=DYNAMIC COMMENT='学生选课表';

/*Data for the table `course_student` */

insert  into `course_student`(`id`,`course_id`,`student_id`,`enroll_time`,`collected`) values (1,9,20250005,NULL,0),(2,19,20250005,NULL,0),(3,22,20250005,NULL,0);

/*Table structure for table `knowledge_graph` */

DROP TABLE IF EXISTS `knowledge_graph`;

CREATE TABLE `knowledge_graph` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '知识图谱ID',
  `title` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '图谱标题',
  `description` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci COMMENT '图谱描述',
  `course_id` bigint(20) NOT NULL COMMENT '关联课程ID',
  `graph_type` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci DEFAULT 'COURSE' COMMENT '图谱类型：COURSE-课程图谱，CHAPTER-章节图谱',
  `graph_data` mediumtext CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci COMMENT '图谱数据（JSON格式）',
  `creator_id` bigint(20) NOT NULL COMMENT '创建者ID',
  `status` varchar(20) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci DEFAULT 'active' COMMENT '状态：active-活跃，archived-归档',
  `create_time` datetime DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`) USING BTREE,
  KEY `idx_course_id` (`course_id`) USING BTREE,
  KEY `idx_creator_id` (`creator_id`) USING BTREE,
  KEY `idx_status` (`status`) USING BTREE,
  CONSTRAINT `fk_knowledge_graph_course` FOREIGN KEY (`course_id`) REFERENCES `course` (`id`) ON DELETE CASCADE ON UPDATE RESTRICT,
  CONSTRAINT `fk_knowledge_graph_creator` FOREIGN KEY (`creator_id`) REFERENCES `user` (`id`) ON DELETE CASCADE ON UPDATE RESTRICT
) ENGINE=InnoDB AUTO_INCREMENT=3 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci ROW_FORMAT=DYNAMIC COMMENT='知识图谱表';

/*Data for the table `knowledge_graph` */

insert  into `knowledge_graph`(`id`,`title`,`description`,`course_id`,`graph_type`,`graph_data`,`creator_id`,`status`,`create_time`,`update_time`) values (1,'计算机组成原理知识图谱','知识图谱',9,'comprehensive','{\"id\":null,\"title\":\"计算机组成原理知识图谱\",\"description\":\"知识图谱\",\"nodes\":[{\"id\":\"node_1\",\"name\":\"计算机组成原理\",\"type\":\"topic\",\"level\":3,\"description\":\"本课程介绍计算机系统的基本组成和工作原理。\",\"chapterId\":null,\"sectionId\":null,\"style\":{\"color\":\"#6400ff\",\"size\":50,\"shape\":null,\"fontSize\":null,\"highlighted\":false},\"position\":null,\"properties\":null},{\"id\":\"node_2\",\"name\":\"计算机系统的组成\",\"type\":\"chapter\",\"level\":2,\"description\":\"介绍计算机系统的基本概念和发展历史。\",\"chapterId\":null,\"sectionId\":null,\"style\":{\"color\":\"#7800ff\",\"size\":40,\"shape\":null,\"fontSize\":null,\"highlighted\":false},\"position\":null,\"properties\":null},{\"id\":\"node_3\",\"name\":\"数字逻辑基础\",\"type\":\"chapter\",\"level\":2,\"description\":\"介绍数字电路和逻辑设计的基本原理。\",\"chapterId\":null,\"sectionId\":null,\"style\":{\"color\":\"#7800ff\",\"size\":40,\"shape\":null,\"fontSize\":null,\"highlighted\":false},\"position\":null,\"properties\":null},{\"id\":\"node_4\",\"name\":\"计算机系统的组成\",\"type\":\"concept\",\"level\":1,\"description\":\"介绍计算机系统的基本组成部分。\",\"chapterId\":null,\"sectionId\":null,\"style\":{\"color\":\"#8c00ff\",\"size\":35,\"shape\":null,\"fontSize\":null,\"highlighted\":false},\"position\":null,\"properties\":null},{\"id\":\"node_5\",\"name\":\"层次结构\",\"type\":\"concept\",\"level\":1,\"description\":\"介绍计算机系统的层次结构。\",\"chapterId\":null,\"sectionId\":null,\"style\":{\"color\":\"#8c00ff\",\"size\":35,\"shape\":null,\"fontSize\":null,\"highlighted\":false},\"position\":null,\"properties\":null},{\"id\":\"node_6\",\"name\":\"性能指标\",\"type\":\"concept\",\"level\":1,\"description\":\"介绍计算机系统的性能指标。\",\"chapterId\":null,\"sectionId\":null,\"style\":{\"color\":\"#8c00ff\",\"size\":35,\"shape\":null,\"fontSize\":null,\"highlighted\":false},\"position\":null,\"properties\":null}],\"edges\":[{\"id\":\"edge_1\",\"source\":\"node_1\",\"target\":\"node_2\",\"type\":\"contains\",\"description\":\"包含章节：计算机系统的组成\",\"weight\":1.0,\"style\":{\"color\":null,\"width\":null,\"lineType\":\"solid\",\"showArrow\":true},\"properties\":null},{\"id\":\"edge_2\",\"source\":\"node_1\",\"target\":\"node_3\",\"type\":\"contains\",\"description\":\"包含章节：数字逻辑基础\",\"weight\":1.0,\"style\":{\"color\":null,\"width\":null,\"lineType\":\"solid\",\"showArrow\":true},\"properties\":null},{\"id\":\"edge_3\",\"source\":\"node_2\",\"target\":\"node_4\",\"type\":\"contains\",\"description\":\"包含内容：计算机系统的组成\",\"weight\":1.0,\"style\":{\"color\":null,\"width\":null,\"lineType\":\"solid\",\"showArrow\":true},\"properties\":null},{\"id\":\"edge_4\",\"source\":\"node_2\",\"target\":\"node_5\",\"type\":\"contains\",\"description\":\"包含内容：层次结构\",\"weight\":1.0,\"style\":{\"color\":null,\"width\":null,\"lineType\":\"solid\",\"showArrow\":true},\"properties\":null},{\"id\":\"edge_5\",\"source\":\"node_2\",\"target\":\"node_6\",\"type\":\"contains\",\"description\":\"包含内容：性能指标\",\"weight\":1.0,\"style\":{\"color\":null,\"width\":null,\"lineType\":\"solid\",\"showArrow\":true},\"properties\":null}],\"metadata\":{\"edgeCount\":5,\"generatedAt\":\"2025-07-12T01:31:29.447817\",\"format\":\"echarts\",\"nodeCount\":6,\"rawData\":{\"title\":{\"text\":\"计算机组成原理知识图谱\"},\"tooltip\":{},\"legend\":{\"data\":[\"主题\",\"章节\",\"概念\"]},\"series\":[{\"name\":\"知识图谱\",\"type\":\"graph\",\"layout\":\"force\",\"data\":[{\"id\":\"node_1\",\"name\":\"计算机组成原理\",\"symbolSize\":50,\"category\":0,\"value\":3,\"draggable\":true,\"tooltip\":{\"formatter\":\"本课程介绍计算机系统的基本组成和工作原理。\"}},{\"id\":\"node_2\",\"name\":\"计算机系统的组成\",\"symbolSize\":40,\"category\":1,\"value\":2,\"draggable\":true,\"tooltip\":{\"formatter\":\"介绍计算机系统的基本概念和发展历史。\"}},{\"id\":\"node_3\",\"name\":\"数字逻辑基础\",\"symbolSize\":40,\"category\":1,\"value\":2,\"draggable\":true,\"tooltip\":{\"formatter\":\"介绍数字电路和逻辑设计的基本原理。\"}},{\"id\":\"node_4\",\"name\":\"计算机系统的组成\",\"symbolSize\":35,\"category\":2,\"value\":1,\"draggable\":true,\"tooltip\":{\"formatter\":\"介绍计算机系统的基本组成部分。\"}},{\"id\":\"node_5\",\"name\":\"层次结构\",\"symbolSize\":35,\"category\":2,\"value\":1,\"draggable\":true,\"tooltip\":{\"formatter\":\"介绍计算机系统的层次结构。\"}},{\"id\":\"node_6\",\"name\":\"性能指标\",\"symbolSize\":35,\"category\":2,\"value\":1,\"draggable\":true,\"tooltip\":{\"formatter\":\"介绍计算机系统的性能指标。\"}}],\"links\":[{\"source\":\"node_1\",\"target\":\"node_2\",\"value\":1,\"tooltip\":{\"formatter\":\"包含章节：计算机系统的组成\"}},{\"source\":\"node_1\",\"target\":\"node_3\",\"value\":1,\"tooltip\":{\"formatter\":\"包含章节：数字逻辑基础\"}},{\"source\":\"node_2\",\"target\":\"node_4\",\"value\":1,\"tooltip\":{\"formatter\":\"包含内容：计算机系统的组成\"}},{\"source\":\"node_2\",\"target\":\"node_5\",\"value\":1,\"tooltip\":{\"formatter\":\"包含内容：层次结构\"}},{\"source\":\"node_2\",\"target\":\"node_6\",\"value\":1,\"tooltip\":{\"formatter\":\"包含内容：性能指标\"}}],\"categories\":[{\"name\":\"主题\"},{\"name\":\"章节\"},{\"name\":\"概念\"}],\"roam\":true,\"label\":{\"show\":true,\"position\":\"right\"},\"force\":{\"repulsion\":1000,\"edgeLength\":[50,150]}}]},\"source\":\"dify_agent\"}}',6,'published','2025-07-12 01:31:29','2025-07-12 01:31:29'),(2,'计算机组成原理知识图谱','知识图谱',9,'comprehensive','{\"id\":null,\"title\":\"计算机组成原理知识图谱\",\"description\":\"知识图谱\",\"nodes\":[{\"id\":\"node_1\",\"name\":\"计算机组成原理\",\"type\":\"topic\",\"level\":1,\"description\":\"课程主题：介绍计算机系统的基本组成和工作原理\",\"chapterId\":null,\"sectionId\":null,\"style\":{\"color\":\"#6400ff\",\"size\":40,\"shape\":null,\"fontSize\":null,\"highlighted\":false},\"position\":null,\"properties\":null},{\"id\":\"node_2\",\"name\":\"计算机系统的组成\",\"type\":\"chapter\",\"level\":2,\"description\":\"章节：介绍计算机系统的基本概念和发展历史\",\"chapterId\":null,\"sectionId\":null,\"style\":{\"color\":\"#7800ff\",\"size\":50,\"shape\":null,\"fontSize\":null,\"highlighted\":false},\"position\":null,\"properties\":null},{\"id\":\"node_3\",\"name\":\"数字逻辑基础\",\"type\":\"chapter\",\"level\":2,\"description\":\"章节：介绍数字电路和逻辑设计的基本原理\",\"chapterId\":null,\"sectionId\":null,\"style\":{\"color\":\"#7800ff\",\"size\":50,\"shape\":null,\"fontSize\":null,\"highlighted\":false},\"position\":null,\"properties\":null},{\"id\":\"node_4\",\"name\":\"计算机系统的组成\",\"type\":\"concept\",\"level\":3,\"description\":\"概念：介绍计算机系统的基本组成部分\",\"chapterId\":null,\"sectionId\":null,\"style\":{\"color\":\"#8c00ff\",\"size\":60,\"shape\":null,\"fontSize\":null,\"highlighted\":false},\"position\":null,\"properties\":null},{\"id\":\"node_5\",\"name\":\"计算机系统的层次结构\",\"type\":\"concept\",\"level\":3,\"description\":\"概念：介绍计算机系统的层次化结构\",\"chapterId\":null,\"sectionId\":null,\"style\":{\"color\":\"#8c00ff\",\"size\":60,\"shape\":null,\"fontSize\":null,\"highlighted\":false},\"position\":null,\"properties\":null},{\"id\":\"node_6\",\"name\":\"计算机的性能指标\",\"type\":\"concept\",\"level\":3,\"description\":\"概念：介绍计算机性能评估的关键指标\",\"chapterId\":null,\"sectionId\":null,\"style\":{\"color\":\"#8c00ff\",\"size\":60,\"shape\":null,\"fontSize\":null,\"highlighted\":false},\"position\":null,\"properties\":null},{\"id\":\"node_7\",\"name\":\"数字逻辑基础知识\",\"type\":\"concept\",\"level\":3,\"description\":\"概念：介绍数字电路和逻辑设计的基本原理\",\"chapterId\":null,\"sectionId\":null,\"style\":{\"color\":\"#8c00ff\",\"size\":60,\"shape\":null,\"fontSize\":null,\"highlighted\":false},\"position\":null,\"properties\":null},{\"id\":\"node_8\",\"name\":\"CPU设计\",\"type\":\"concept\",\"level\":3,\"description\":\"应用：数字逻辑基础在CPU设计中的应用\",\"chapterId\":null,\"sectionId\":null,\"style\":{\"color\":\"#8c00ff\",\"size\":60,\"shape\":null,\"fontSize\":null,\"highlighted\":false},\"position\":null,\"properties\":null},{\"id\":\"node_9\",\"name\":\"存储系统设计\",\"type\":\"concept\",\"level\":3,\"description\":\"应用：计算机系统组成在存储系统设计中的应用\",\"chapterId\":null,\"sectionId\":null,\"style\":{\"color\":\"#8c00ff\",\"size\":60,\"shape\":null,\"fontSize\":null,\"highlighted\":false},\"position\":null,\"properties\":null}],\"edges\":[{\"id\":\"edge_1\",\"source\":\"node_1\",\"target\":\"node_2\",\"type\":\"contains\",\"description\":\"包含\",\"weight\":1.0,\"style\":{\"color\":null,\"width\":null,\"lineType\":\"solid\",\"showArrow\":true},\"properties\":null},{\"id\":\"edge_2\",\"source\":\"node_1\",\"target\":\"node_3\",\"type\":\"contains\",\"description\":\"包含\",\"weight\":1.0,\"style\":{\"color\":null,\"width\":null,\"lineType\":\"solid\",\"showArrow\":true},\"properties\":null},{\"id\":\"edge_3\",\"source\":\"node_2\",\"target\":\"node_4\",\"type\":\"contains\",\"description\":\"包含\",\"weight\":1.0,\"style\":{\"color\":null,\"width\":null,\"lineType\":\"solid\",\"showArrow\":true},\"properties\":null},{\"id\":\"edge_4\",\"source\":\"node_2\",\"target\":\"node_5\",\"type\":\"contains\",\"description\":\"包含\",\"weight\":1.0,\"style\":{\"color\":null,\"width\":null,\"lineType\":\"solid\",\"showArrow\":true},\"properties\":null},{\"id\":\"edge_5\",\"source\":\"node_2\",\"target\":\"node_6\",\"type\":\"contains\",\"description\":\"包含\",\"weight\":1.0,\"style\":{\"color\":null,\"width\":null,\"lineType\":\"solid\",\"showArrow\":true},\"properties\":null},{\"id\":\"edge_6\",\"source\":\"node_3\",\"target\":\"node_7\",\"type\":\"contains\",\"description\":\"包含\",\"weight\":1.0,\"style\":{\"color\":null,\"width\":null,\"lineType\":\"solid\",\"showArrow\":true},\"properties\":null},{\"id\":\"edge_7\",\"source\":\"node_4\",\"target\":\"node_8\",\"type\":\"prerequisite\",\"description\":\"先修\",\"weight\":2.0,\"style\":{\"color\":null,\"width\":null,\"lineType\":\"dashed\",\"showArrow\":true},\"properties\":null},{\"id\":\"edge_8\",\"source\":\"node_7\",\"target\":\"node_8\",\"type\":\"application\",\"description\":\"应用\",\"weight\":3.0,\"style\":{\"color\":null,\"width\":null,\"lineType\":\"dotted\",\"showArrow\":true},\"properties\":null},{\"id\":\"edge_9\",\"source\":\"node_4\",\"target\":\"node_9\",\"type\":\"application\",\"description\":\"应用\",\"weight\":3.0,\"style\":{\"color\":null,\"width\":null,\"lineType\":\"dotted\",\"showArrow\":true},\"properties\":null}],\"metadata\":{\"edgeCount\":9,\"generatedAt\":\"2025-07-12T01:48:16.022170900\",\"format\":\"echarts\",\"nodeCount\":9,\"rawData\":{\"title\":{\"text\":\"计算机组成原理知识图谱\"},\"tooltip\":{},\"legend\":{\"data\":[\"主题\",\"章节\",\"概念\"]},\"series\":[{\"name\":\"知识图谱\",\"type\":\"graph\",\"layout\":\"force\",\"data\":[{\"id\":\"node_1\",\"name\":\"计算机组成原理\",\"symbolSize\":40,\"category\":0,\"value\":1,\"draggable\":true,\"tooltip\":{\"formatter\":\"课程主题：介绍计算机系统的基本组成和工作原理\"}},{\"id\":\"node_2\",\"name\":\"计算机系统的组成\",\"symbolSize\":50,\"category\":1,\"value\":2,\"draggable\":true,\"tooltip\":{\"formatter\":\"章节：介绍计算机系统的基本概念和发展历史\"}},{\"id\":\"node_3\",\"name\":\"数字逻辑基础\",\"symbolSize\":50,\"category\":1,\"value\":2,\"draggable\":true,\"tooltip\":{\"formatter\":\"章节：介绍数字电路和逻辑设计的基本原理\"}},{\"id\":\"node_4\",\"name\":\"计算机系统的组成\",\"symbolSize\":60,\"category\":2,\"value\":3,\"draggable\":true,\"tooltip\":{\"formatter\":\"概念：介绍计算机系统的基本组成部分\"}},{\"id\":\"node_5\",\"name\":\"计算机系统的层次结构\",\"symbolSize\":60,\"category\":2,\"value\":3,\"draggable\":true,\"tooltip\":{\"formatter\":\"概念：介绍计算机系统的层次化结构\"}},{\"id\":\"node_6\",\"name\":\"计算机的性能指标\",\"symbolSize\":60,\"category\":2,\"value\":3,\"draggable\":true,\"tooltip\":{\"formatter\":\"概念：介绍计算机性能评估的关键指标\"}},{\"id\":\"node_7\",\"name\":\"数字逻辑基础知识\",\"symbolSize\":60,\"category\":2,\"value\":3,\"draggable\":true,\"tooltip\":{\"formatter\":\"概念：介绍数字电路和逻辑设计的基本原理\"}},{\"id\":\"node_8\",\"name\":\"CPU设计\",\"symbolSize\":60,\"category\":2,\"value\":3,\"draggable\":true,\"tooltip\":{\"formatter\":\"应用：数字逻辑基础在CPU设计中的应用\"}},{\"id\":\"node_9\",\"name\":\"存储系统设计\",\"symbolSize\":60,\"category\":2,\"value\":3,\"draggable\":true,\"tooltip\":{\"formatter\":\"应用：计算机系统组成在存储系统设计中的应用\"}}],\"links\":[{\"source\":\"node_1\",\"target\":\"node_2\",\"value\":1,\"tooltip\":{\"formatter\":\"包含\"}},{\"source\":\"node_1\",\"target\":\"node_3\",\"value\":1,\"tooltip\":{\"formatter\":\"包含\"}},{\"source\":\"node_2\",\"target\":\"node_4\",\"value\":1,\"tooltip\":{\"formatter\":\"包含\"}},{\"source\":\"node_2\",\"target\":\"node_5\",\"value\":1,\"tooltip\":{\"formatter\":\"包含\"}},{\"source\":\"node_2\",\"target\":\"node_6\",\"value\":1,\"tooltip\":{\"formatter\":\"包含\"}},{\"source\":\"node_3\",\"target\":\"node_7\",\"value\":1,\"tooltip\":{\"formatter\":\"包含\"}},{\"source\":\"node_4\",\"target\":\"node_8\",\"value\":2,\"tooltip\":{\"formatter\":\"先修\"}},{\"source\":\"node_7\",\"target\":\"node_8\",\"value\":3,\"tooltip\":{\"formatter\":\"应用\"}},{\"source\":\"node_4\",\"target\":\"node_9\",\"value\":3,\"tooltip\":{\"formatter\":\"应用\"}}],\"categories\":[{\"name\":\"主题\"},{\"name\":\"章节\"},{\"name\":\"概念\"}],\"roam\":true,\"label\":{\"show\":true,\"position\":\"right\"},\"force\":{\"repulsion\":1000,\"edgeLength\":[100,300]}}]},\"source\":\"dify_agent\"}}',6,'published','2025-07-12 01:48:16','2025-07-12 01:48:16');

/*Table structure for table `learning_records` */

DROP TABLE IF EXISTS `learning_records`;

CREATE TABLE `learning_records` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '记录ID',
  `student_id` bigint(20) NOT NULL COMMENT '学生ID',
  `course_id` bigint(20) NOT NULL COMMENT '课程ID',
  `section_id` bigint(20) DEFAULT NULL COMMENT '章节ID',
  `resource_id` bigint(20) DEFAULT NULL COMMENT '资源ID',
  `resource_type` varchar(50) DEFAULT NULL COMMENT '资源类型：video, document, quiz等',
  `start_time` datetime NOT NULL COMMENT '开始学习时间',
  `end_time` datetime DEFAULT NULL COMMENT '结束学习时间',
  `duration` int(11) DEFAULT '0' COMMENT '学习时长(秒)',
  `progress` int(11) DEFAULT '0' COMMENT '学习进度(百分比)',
  `completed` tinyint(1) DEFAULT '0' COMMENT '是否完成',
  `device_info` varchar(255) DEFAULT NULL COMMENT '设备信息',
  `ip_address` varchar(50) DEFAULT NULL COMMENT 'IP地址',
  `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `updated_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  KEY `idx_student_course` (`student_id`,`course_id`),
  KEY `idx_student_time` (`student_id`,`start_time`),
  KEY `idx_section` (`section_id`),
  KEY `idx_resource` (`resource_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='学习记录表';

/*Data for the table `learning_records` */

/*Table structure for table `learning_statistics` */

DROP TABLE IF EXISTS `learning_statistics`;

CREATE TABLE `learning_statistics` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '统计ID',
  `student_id` bigint(20) NOT NULL COMMENT '学生ID',
  `course_id` bigint(20) NOT NULL COMMENT '课程ID',
  `date` date NOT NULL COMMENT '日期',
  `total_duration` int(11) DEFAULT '0' COMMENT '总学习时长(秒)',
  `sections_completed` int(11) DEFAULT '0' COMMENT '完成章节数',
  `resources_viewed` int(11) DEFAULT '0' COMMENT '查看资源数',
  `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `updated_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_student_course_date` (`student_id`,`course_id`,`date`),
  KEY `idx_student_date` (`student_id`,`date`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='学习统计表';

/*Data for the table `learning_statistics` */

/*Table structure for table `question` */

DROP TABLE IF EXISTS `question`;

CREATE TABLE `question` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `title` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '题干内容',
  `question_type` enum('single','multiple','true_false','blank','short','code') CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '题型',
  `difficulty` tinyint(4) NOT NULL DEFAULT '3' COMMENT '难度等级，1~5整数',
  `correct_answer` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci COMMENT '标准答案',
  `explanation` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci COMMENT '答案解析',
  `knowledge_point` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci DEFAULT NULL COMMENT '知识点',
  `course_id` bigint(20) NOT NULL,
  `chapter_id` bigint(20) NOT NULL,
  `created_by` bigint(20) NOT NULL COMMENT '出题教师ID',
  `create_time` datetime DEFAULT CURRENT_TIMESTAMP,
  `update_time` datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`) USING BTREE,
  KEY `course_id` (`course_id`) USING BTREE,
  KEY `chapter_id` (`chapter_id`) USING BTREE,
  KEY `question_ibfk_3` (`created_by`) USING BTREE,
  CONSTRAINT `question_ibfk_1` FOREIGN KEY (`course_id`) REFERENCES `course` (`id`) ON DELETE RESTRICT ON UPDATE RESTRICT,
  CONSTRAINT `question_ibfk_2` FOREIGN KEY (`chapter_id`) REFERENCES `chapter` (`id`) ON DELETE RESTRICT ON UPDATE RESTRICT,
  CONSTRAINT `question_ibfk_3` FOREIGN KEY (`created_by`) REFERENCES `user` (`id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=50 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci ROW_FORMAT=DYNAMIC COMMENT='题目表';

/*Data for the table `question` */

insert  into `question`(`id`,`title`,`question_type`,`difficulty`,`correct_answer`,`explanation`,`knowledge_point`,`course_id`,`chapter_id`,`created_by`,`create_time`,`update_time`) values (22,'计算机的中央处理器(CPU)的主要功能是什么？','short',3,'CPU是计算机的核心，主要功能是执行指令、处理数据和控制系统运行。它负责算术运算、逻辑运算、数据传送和程序控制等基本操作。','中央处理器是计算机的大脑，负责执行指令和处理数据。','计算机组成原理',9,11,6,'2025-07-11 21:47:01','2025-07-11 21:47:01'),(23,'以下哪项是输入设备？','multiple',2,'A,B,D','键盘、鼠标和扫描仪都是输入设备，而打印机是输出设备。','计算机外设',9,11,6,'2025-07-11 21:47:01','2025-07-11 21:47:01'),(24,'二进制数1010等于十进制数多少？','blank',2,'10','二进制1010 = 1*2^3 + 0*2^2 + 1*2^1 + 0*2^0 = 8 + 0 + 2 + 0 = 10','数字逻辑',9,12,6,'2025-07-11 21:47:01','2025-07-11 21:47:01'),(25,'在计算机系统中，CPU的主要功能是什么？','single',3,'B','CPU(中央处理器)的主要功能是执行指令和处理数据，是计算机的运算和控制核心。','计算机系统组成',9,11,6,'2025-07-12 04:42:06','2025-07-12 04:42:06'),(26,'以下哪种存储器具有最快的访问速度？','single',2,'A','寄存器是CPU内部的存储单元，具有最快的访问速度，其次是缓存、内存和硬盘。','存储器层次结构',9,15,6,'2025-07-12 04:42:06','2025-07-12 04:42:06'),(27,'在IEEE 754标准中，32位单精度浮点数的指数偏移值是多少？','single',4,'C','IEEE 754标准中，32位单精度浮点数的指数偏移值是127，用于将实际指数转换为无符号表示。','数据表示',9,20,6,'2025-07-12 04:42:06','2025-07-12 04:42:06'),(28,'以下哪种总线主要用于CPU与内存之间的数据传输？','single',3,'A','数据总线用于在CPU与内存以及其他设备之间传输数据，地址总线用于传输内存地址，控制总线用于传输控制信号。','总线结构',9,17,6,'2025-07-12 04:42:06','2025-07-12 04:42:06'),(29,'在流水线CPU中，以下哪个不是典型的五级流水线阶段？','single',4,'D','典型的五级流水线包括：取指令(IF)、指令译码(ID)、执行(EX)、访存(MEM)和写回(WB)，而不包括指令优化阶段。','CPU结构',9,14,6,'2025-07-12 04:42:06','2025-07-12 04:42:06'),(30,'以下哪些是计算机系统的主要组成部分？','multiple',2,'A,B,C,E','计算机系统主要由CPU、内存、输入设备、输出设备和存储设备组成，而显卡是输出设备的一种，不是主要组成部分。','计算机系统组成',9,11,6,'2025-07-12 04:42:06','2025-07-12 04:42:06'),(31,'以下哪些是RISC处理器的特点？','multiple',4,'A,C,D','RISC(精简指令集计算机)的特点包括指令数量少、指令长度固定、寻址方式简单和使用大量寄存器，而不是复杂的寻址方式。','指令系统',9,13,6,'2025-07-12 04:42:06','2025-07-12 04:42:06'),(32,'以下哪些因素会影响CPU的性能？','multiple',3,'A,B,C,D','CPU性能受时钟频率、指令集架构、缓存大小和流水线深度等因素影响。','CPU性能',9,14,6,'2025-07-12 04:42:06','2025-07-12 04:42:06'),(33,'在冯·诺依曼结构中，程序和数据存储在同一个存储器中。','true_false',2,'T','冯·诺依曼结构的一个核心特点就是程序和数据存储在同一个存储器中，这使得计算机可以像处理数据一样处理指令。','计算机体系结构',9,11,6,'2025-07-12 04:42:06','2025-07-12 04:42:06'),(34,'CISC处理器比RISC处理器的指令数量更少。','true_false',3,'F','CISC(复杂指令集计算机)的指令数量比RISC(精简指令集计算机)更多，而不是更少。','指令系统',9,13,6,'2025-07-12 04:42:06','2025-07-12 04:42:06'),(35,'在计算机中，1KB等于1000字节。','true_false',1,'F','在计算机中，1KB(千字节)等于1024字节，而不是1000字节。这是因为计算机使用二进制，1KB = 2^10 字节。','数据表示',9,20,6,'2025-07-12 04:42:06','2025-07-12 04:42:06'),(36,'高速缓存(Cache)的主要目的是弥补CPU和内存之间的速度差异。','true_false',2,'T','高速缓存的主要目的确实是弥补CPU和内存之间的速度差异，通过存储频繁使用的数据来提高系统性能。','存储器层次结构',9,15,6,'2025-07-12 04:42:06','2025-07-12 04:42:06'),(37,'在二进制数系统中，十进制数15表示为________。','blank',2,'1111','十进制数15转换为二进制是1111，计算过程：15 = 8 + 4 + 2 + 1 = 2^3 + 2^2 + 2^1 + 2^0 = 1111(二进制)','数制转换',9,20,6,'2025-07-12 04:42:06','2025-07-12 04:42:06'),(38,'计算机中的主频通常以________为单位。','blank',1,'Hz','计算机主频是CPU时钟频率，通常以赫兹(Hz)为单位，常见的有MHz(兆赫)和GHz(吉赫)。','CPU性能',9,14,6,'2025-07-12 04:42:06','2025-07-12 04:42:06'),(39,'在计算机存储层次结构中，从上到下依次是寄存器、________、内存和外存。','blank',3,'高速缓存','计算机存储层次结构从上到下依次是：寄存器、高速缓存(Cache)、内存(RAM)和外存(硬盘等)，速度依次降低，容量依次增大。','存储器层次结构',9,15,6,'2025-07-12 04:42:06','2025-07-12 04:42:06'),(40,'在计算机中，将二进制数据转换为人类可读的十进制数的过程称为________。','blank',2,'解码','解码是将二进制数据转换为人类可读形式的过程，是计算机内部数据表示与人类理解之间的桥梁。','数据表示',9,20,6,'2025-07-12 04:42:06','2025-07-12 04:42:06'),(41,'简述计算机的冯·诺依曼结构及其主要特点。','short',3,'冯·诺依曼结构是现代计算机的基本结构，其主要特点包括：1. 计算机由运算器、控制器、存储器、输入设备和输出设备五大部分组成；2. 程序和数据存储在同一个存储器中；3. 指令和数据均以二进制形式表示；4. 指令按地址顺序存放，通常按顺序执行；5. 采用存储程序原理，程序可以像数据一样存取。','冯·诺依曼结构是现代计算机的基础，理解其特点对理解计算机工作原理至关重要。','计算机体系结构',9,11,6,'2025-07-12 04:42:06','2025-07-12 04:42:06'),(42,'解释CPU中的流水线技术原理及其优缺点。','short',4,'CPU流水线技术是将指令执行过程分解为多个阶段，使多条指令可以同时在不同阶段执行，从而提高CPU的吞吐率。典型的五级流水线包括：取指令(IF)、指令译码(ID)、执行(EX)、访存(MEM)和写回(WB)。\n\n优点：1. 提高CPU的吞吐率；2. 提高硬件资源利用率；3. 减少平均指令执行时间。\n\n缺点：1. 可能产生数据相关、控制相关和结构相关等冒险；2. 增加了硬件复杂度；3. 流水线越深，分支预测失败的惩罚越大。','流水线技术是现代CPU提高性能的重要手段，理解其原理有助于理解CPU的工作方式。','CPU结构',9,14,6,'2025-07-12 04:42:06','2025-07-12 04:42:06'),(43,'比较CISC和RISC架构的主要区别及各自的优缺点。','short',5,'CISC(复杂指令集计算机)和RISC(精简指令集计算机)是两种不同的处理器设计理念：\n\nCISC特点：1. 指令数量多且复杂；2. 指令长度可变；3. 寻址方式多样；4. 指令执行时间不等；5. 硬件实现复杂，软件实现简单。\n\nRISC特点：1. 指令数量少且简单；2. 指令长度固定；3. 寻址方式简单；4. 使用大量寄存器；5. 强调优化编译器；6. 硬件实现简单，软件实现复杂。\n\nCISC优点：代码密度高，适合内存受限系统；缺点：硬件复杂，功耗高，流水线实现困难。\n\nRISC优点：硬件简单，功耗低，易于实现流水线；缺点：代码密度低，对编译器要求高。\n\n现代处理器通常融合了两种架构的特点，如x86处理器内部采用RISC微架构，但对外提供CISC指令集。','CISC和RISC代表了处理器设计的两种不同思路，理解它们的区别有助于理解计算机体系结构的发展。','指令系统',9,13,6,'2025-07-12 04:42:06','2025-07-12 04:42:06'),(44,'描述计算机存储层次结构，并解释为什么要采用这种层次化设计。','short',3,'计算机存储层次结构从上到下依次是：\n1. 寄存器：CPU内部，容量最小(几KB)，速度最快，成本最高\n2. 高速缓存(Cache)：分为L1、L2、L3等级，容量适中(几MB)，速度很快，成本高\n3. 主存(RAM)：容量较大(几GB)，速度中等，成本中等\n4. 外存(硬盘、SSD等)：容量最大(几TB)，速度最慢，成本最低\n\n采用层次化设计的原因：\n1. 平衡速度与容量的矛盾：高速存储器成本高，容量小；大容量存储器速度慢，成本低\n2. 利用程序的局部性原理：时间局部性(最近访问的数据很可能再次被访问)和空间局部性(最近访问的数据附近的数据很可能被访问)\n3. 通过缓存机制，使系统在大部分情况下能以接近高速存储器的速度工作，同时拥有大容量存储器的容量\n\n这种层次化设计是计算机系统性能与成本平衡的重要手段。','存储层次结构是计算机系统设计的重要概念，理解其原理有助于理解计算机性能优化的方法。','存储器层次结构',9,15,6,'2025-07-12 04:42:06','2025-07-12 04:42:06'),(45,'请编写一个简单的C语言程序，实现两个32位无符号整数的二进制加法，并处理可能的溢出情况。','code',4,'#include <stdio.h>\n#include <stdint.h>\n\ntypedef struct {\n    uint32_t result;\n    uint8_t overflow; // 0表示无溢出，1表示有溢出\n} AddResult;\n\nAddResult add_with_overflow(uint32_t a, uint32_t b) {\n    AddResult res;\n    res.result = a + b;\n    // 如果结果小于任一操作数，则发生了溢出\n    res.overflow = (res.result < a || res.result < b) ? 1 : 0;\n    return res;\n}\n\nint main() {\n    uint32_t a = 4294967290; // 接近uint32_t最大值\n    uint32_t b = 10;\n    \n    AddResult result = add_with_overflow(a, b);\n    \n    printf(\"a = %u\\n\", a);\n    printf(\"b = %u\\n\", b);\n    printf(\"a + b = %u\\n\", result.result);\n    printf(\"溢出标志: %s\\n\", result.overflow ? \"是\" : \"否\");\n    \n    return 0;\n}','此程序演示了如何在C语言中实现二进制加法并检测溢出。在计算机组成原理中，理解整数运算的溢出处理是很重要的概念。','计算机算术运算',9,20,6,'2025-07-12 04:42:06','2025-07-12 04:42:06'),(46,'请编写一个简单的汇编语言程序(MIPS指令集)，实现两个整数的加法并将结果存储到内存中。','code',5,'# MIPS汇编程序：两数相加\n# 假设$a0和$a1中存储着要相加的两个整数\n# 结果将存储在内存地址result中\n\n.data\nresult: .word 0    # 分配一个字(4字节)用于存储结果\n\n.text\n.globl main\nmain:\n    # 假设要相加的两个数已在$a0和$a1中\n    li $a0, 25      # 加载第一个数\n    li $a1, 30      # 加载第二个数\n    \n    # 执行加法\n    add $t0, $a0, $a1   # $t0 = $a0 + $a1\n    \n    # 将结果存储到内存\n    sw $t0, result      # 存储结果到内存\n    \n    # 打印结果(系统调用)\n    li $v0, 1           # 系统调用代码1表示打印整数\n    move $a0, $t0       # 将结果移到$a0用于打印\n    syscall             # 执行系统调用\n    \n    # 退出程序\n    li $v0, 10          # 系统调用代码10表示退出\n    syscall             # 执行系统调用','此程序演示了MIPS汇编语言中如何执行基本的加法运算并与内存交互。理解汇编语言是理解计算机如何执行指令的基础。','汇编语言编程',9,13,6,'2025-07-12 04:42:06','2025-07-12 04:42:06'),(47,'请简要描述计算机系统的层次结构','short',3,'计算机系统的层次结构从低到高包括硬件、机器语言层、操作系统层、汇编语言层、高级语言层和应用层。每一层都为上一层提供服务，隐藏下层的具体实现细节。','这个问题考察学生对计算机系统层次化设计的理解。','计算机系统结构',9,11,6,'2025-07-12 09:26:29','2025-07-12 09:26:29'),(48,'在计算机中，整数的二进制补码表示法主要用于解决什么问题？','single',2,'B','补码表示法使得负数可以用二进制表示，并且加法和减法可以用统一的电路实现。','数据表示',9,12,6,'2025-07-12 09:26:29','2025-07-12 09:26:29'),(49,'以下哪些属于计算机的易失性存储器？','multiple',2,'A,C','易失性存储器在断电后数据会丢失，包括RAM和Cache等。','存储器',9,15,6,'2025-07-12 09:26:29','2025-07-12 09:26:29');

/*Table structure for table `question_bank` */

DROP TABLE IF EXISTS `question_bank`;

CREATE TABLE `question_bank` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `user_id` bigint(20) NOT NULL COMMENT '出题教师的用户ID',
  `title` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '题库名称',
  `description` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci COMMENT '题库说明',
  `create_time` datetime DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`) USING BTREE,
  KEY `fk_question_bank_user` (`user_id`) USING BTREE,
  CONSTRAINT `fk_question_bank_user` FOREIGN KEY (`user_id`) REFERENCES `user` (`id`) ON DELETE CASCADE ON UPDATE RESTRICT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci ROW_FORMAT=DYNAMIC COMMENT='题库表';

/*Data for the table `question_bank` */

/*Table structure for table `question_bank_item` */

DROP TABLE IF EXISTS `question_bank_item`;

CREATE TABLE `question_bank_item` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `bank_id` bigint(20) NOT NULL,
  `question_id` bigint(20) NOT NULL,
  PRIMARY KEY (`id`) USING BTREE,
  KEY `bank_id` (`bank_id`) USING BTREE,
  KEY `question_id` (`question_id`) USING BTREE,
  CONSTRAINT `question_bank_item_ibfk_1` FOREIGN KEY (`bank_id`) REFERENCES `question_bank` (`id`) ON DELETE CASCADE ON UPDATE RESTRICT,
  CONSTRAINT `question_bank_item_ibfk_2` FOREIGN KEY (`question_id`) REFERENCES `question` (`id`) ON DELETE CASCADE ON UPDATE RESTRICT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci ROW_FORMAT=DYNAMIC COMMENT='题库题目关联表';

/*Data for the table `question_bank_item` */

/*Table structure for table `question_image` */

DROP TABLE IF EXISTS `question_image`;

CREATE TABLE `question_image` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `question_id` bigint(20) NOT NULL,
  `image_url` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '图片URL或路径',
  `description` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci DEFAULT NULL COMMENT '图片说明',
  `sequence` int(11) DEFAULT '1' COMMENT '图片显示顺序',
  `upload_time` datetime DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`) USING BTREE,
  KEY `question_id` (`question_id`) USING BTREE,
  CONSTRAINT `question_image_ibfk_1` FOREIGN KEY (`question_id`) REFERENCES `question` (`id`) ON DELETE CASCADE ON UPDATE RESTRICT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci ROW_FORMAT=DYNAMIC COMMENT='题目图片表';

/*Data for the table `question_image` */

/*Table structure for table `question_option` */

DROP TABLE IF EXISTS `question_option`;

CREATE TABLE `question_option` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `question_id` bigint(20) NOT NULL,
  `option_label` char(1) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '选项标识 A/B/C/D/T/F',
  `option_text` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '选项内容',
  PRIMARY KEY (`id`) USING BTREE,
  KEY `question_id` (`question_id`) USING BTREE,
  CONSTRAINT `question_option_ibfk_1` FOREIGN KEY (`question_id`) REFERENCES `question` (`id`) ON DELETE CASCADE ON UPDATE RESTRICT
) ENGINE=InnoDB AUTO_INCREMENT=78 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci ROW_FORMAT=DYNAMIC COMMENT='题目选项表';

/*Data for the table `question_option` */

insert  into `question_option`(`id`,`question_id`,`option_label`,`option_text`) values (23,23,'A','键盘'),(24,23,'B','鼠标'),(25,23,'C','打印机'),(26,23,'D','扫描仪'),(27,25,'A','存储大量数据'),(28,25,'B','执行指令和处理数据'),(29,25,'C','显示图形界面'),(30,25,'D','连接外部设备'),(31,26,'A','寄存器(Register)'),(32,26,'B','高速缓存(Cache)'),(33,26,'C','内存(RAM)'),(34,26,'D','硬盘(HDD)'),(35,27,'A','63'),(36,27,'B','64'),(37,27,'C','127'),(38,27,'D','1023'),(39,28,'A','数据总线'),(40,28,'B','地址总线'),(41,28,'C','控制总线'),(42,28,'D','外部总线'),(43,29,'A','取指令(IF)'),(44,29,'B','指令译码(ID)'),(45,29,'C','执行(EX)'),(46,29,'D','指令优化(IO)'),(47,30,'A','CPU(中央处理器)'),(48,30,'B','内存(RAM)'),(49,30,'C','输入设备'),(50,30,'D','显卡'),(51,30,'E','存储设备'),(52,31,'A','指令数量较少'),(53,31,'B','复杂的寻址方式'),(54,31,'C','指令长度固定'),(55,31,'D','使用大量寄存器'),(56,31,'E','每条指令执行时间不同'),(57,32,'A','时钟频率'),(58,32,'B','指令集架构'),(59,32,'C','缓存大小'),(60,32,'D','流水线深度'),(61,32,'E','主板颜色'),(62,33,'T','正确'),(63,33,'F','错误'),(64,34,'T','正确'),(65,34,'F','错误'),(66,35,'T','正确'),(67,35,'F','错误'),(68,36,'T','正确'),(69,36,'F','错误'),(70,48,'A','提高数据存储效率'),(71,48,'B','实现负数表示和简化加减法电路设计'),(72,48,'C','加快数据处理速度'),(73,48,'D','减少存储空间占用'),(74,49,'A','内存(RAM)'),(75,49,'B','硬盘'),(76,49,'C','缓存(Cache)'),(77,49,'D','固态硬盘(SSD)');

/*Table structure for table `question_stats` */

DROP TABLE IF EXISTS `question_stats`;

CREATE TABLE `question_stats` (
  `question_id` bigint(20) NOT NULL,
  `answer_count` int(11) DEFAULT '0' COMMENT '答题总数',
  `correct_count` int(11) DEFAULT '0' COMMENT '正确人数',
  `accuracy` decimal(5,2) DEFAULT '0.00' COMMENT '正确率（百分比）',
  PRIMARY KEY (`question_id`) USING BTREE,
  CONSTRAINT `question_stats_ibfk_1` FOREIGN KEY (`question_id`) REFERENCES `question` (`id`) ON DELETE CASCADE ON UPDATE RESTRICT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci ROW_FORMAT=DYNAMIC COMMENT='题目统计信息表';

/*Data for the table `question_stats` */

/*Table structure for table `section` */

DROP TABLE IF EXISTS `section`;

CREATE TABLE `section` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '小节ID，主键',
  `chapter_id` bigint(20) NOT NULL COMMENT '所属章节ID，外键',
  `title` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '小节名称',
  `description` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci COMMENT '小节简介',
  `video_url` varchar(1024) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci DEFAULT NULL COMMENT '视频播放地址，可对接OSS',
  `duration` int(11) DEFAULT '0' COMMENT '视频时长(秒)',
  `sort_order` int(11) NOT NULL DEFAULT '0' COMMENT '小节顺序，用于排序',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`) USING BTREE,
  KEY `idx_chapter_id` (`chapter_id`) USING BTREE,
  KEY `idx_sort_order` (`sort_order`) USING BTREE,
  CONSTRAINT `fk_section_chapter` FOREIGN KEY (`chapter_id`) REFERENCES `chapter` (`id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=71 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci ROW_FORMAT=DYNAMIC COMMENT='课程小节表';

/*Data for the table `section` */

insert  into `section`(`id`,`chapter_id`,`title`,`description`,`video_url`,`duration`,`sort_order`,`create_time`,`update_time`) values (20,11,'计算机系统概述','介绍计算机系统的基本概念、发展历史和应用领域','202507/f61e05fc-f5ec-4e06-b268-877d3ed8de16.mp4',40,1,'2025-07-12 03:52:26','2025-07-12 04:01:43'),(21,11,'计算机系统的层次结构','详细讲解计算机系统的层次结构模型和各层之间的关系',NULL,30,2,'2025-07-12 03:52:26','2025-07-12 04:01:43'),(22,11,'计算机的性能指标','介绍评估计算机系统性能的各种指标和测量方法',NULL,25,3,'2025-07-12 03:52:26','2025-07-12 04:01:43'),(23,11,'计算机系统的发展趋势','分析计算机系统的未来发展方向和技术趋势',NULL,35,4,'2025-07-12 03:52:26','2025-07-12 04:01:43'),(24,12,'数字逻辑基础概念','介绍数字逻辑的基本概念和理论基础',NULL,30,1,'2025-07-12 03:52:26','2025-07-12 04:01:43'),(25,12,'逻辑门电路','讲解各种逻辑门电路的工作原理和应用',NULL,35,2,'2025-07-12 03:52:26','2025-07-12 04:01:43'),(26,12,'组合逻辑电路','介绍组合逻辑电路的设计和分析方法',NULL,40,3,'2025-07-12 03:52:26','2025-07-12 04:01:43'),(27,12,'时序逻辑电路','讲解时序逻辑电路的工作原理和设计方法',NULL,45,4,'2025-07-12 03:52:26','2025-07-12 04:01:43'),(28,12,'数字系统设计基础','介绍数字系统的设计方法和工具',NULL,40,5,'2025-07-12 03:52:26','2025-07-12 04:01:43'),(29,13,'指令系统概述','介绍指令系统的基本概念和设计原则',NULL,30,1,'2025-07-12 03:52:26','2025-07-12 04:01:43'),(30,13,'指令格式与寻址方式','讲解指令格式的设计和各种寻址方式',NULL,40,2,'2025-07-12 03:52:26','2025-07-12 04:01:43'),(31,13,'RISC与CISC指令集','比较精简指令集和复杂指令集的特点和应用',NULL,45,3,'2025-07-12 03:52:26','2025-07-12 04:01:43'),(32,13,'指令系统的性能评估','介绍评估指令系统性能的方法和指标',NULL,35,4,'2025-07-12 03:52:26','2025-07-12 04:01:43'),(33,14,'CPU的基本功能与结构','介绍中央处理器的基本功能和内部结构',NULL,40,1,'2025-07-12 03:52:26','2025-07-12 04:01:43'),(34,14,'运算器的设计','讲解算术逻辑单元的设计原理和实现方法',NULL,35,2,'2025-07-12 03:52:26','2025-07-12 04:01:43'),(35,14,'控制器的设计','介绍控制单元的设计方法和实现技术',NULL,40,3,'2025-07-12 03:52:26','2025-07-12 04:01:43'),(36,14,'流水线技术','讲解CPU流水线技术的原理和实现方法',NULL,45,4,'2025-07-12 03:52:26','2025-07-12 04:01:43'),(37,14,'超标量与超流水技术','介绍现代CPU中的高级处理技术',NULL,40,5,'2025-07-12 03:52:26','2025-07-12 04:01:43'),(38,15,'存储器层次结构概述','介绍计算机存储系统的层次结构模型',NULL,30,1,'2025-07-12 03:52:26','2025-07-12 04:01:43'),(39,15,'缓存存储器','讲解缓存存储器的工作原理和设计方法',NULL,40,2,'2025-07-12 03:52:26','2025-07-12 04:01:43'),(40,15,'主存储器','介绍主存储器的类型、特性和组织方式',NULL,35,3,'2025-07-12 03:52:26','2025-07-12 04:01:43'),(41,15,'辅助存储器','讲解辅助存储设备的类型和工作原理',NULL,35,4,'2025-07-12 03:52:26','2025-07-12 04:01:43'),(42,15,'虚拟存储器','介绍虚拟存储技术的原理和实现方法',NULL,45,5,'2025-07-12 03:52:26','2025-07-12 04:01:43'),(43,16,'输入输出系统概述','介绍计算机输入输出系统的基本概念和组成',NULL,30,1,'2025-07-12 03:52:26','2025-07-12 04:01:43'),(44,16,'I/O接口','讲解I/O接口的设计原理和功能',NULL,35,2,'2025-07-12 03:52:26','2025-07-12 04:01:43'),(45,16,'I/O方式','介绍程序查询、中断和DMA等I/O控制方式',NULL,40,3,'2025-07-12 03:52:26','2025-07-12 04:01:43'),(46,16,'外部设备','讲解各种常见外部设备的工作原理',NULL,45,4,'2025-07-12 03:52:26','2025-07-12 04:01:43'),(47,17,'总线概述','介绍计算机系统中总线的基本概念和分类',NULL,30,1,'2025-07-12 03:52:26','2025-07-12 04:01:43'),(48,17,'总线仲裁','讲解总线仲裁的方法和实现机制',NULL,35,2,'2025-07-12 03:52:26','2025-07-12 04:01:43'),(49,17,'总线操作和定时','介绍总线操作的时序和定时控制',NULL,40,3,'2025-07-12 03:52:26','2025-07-12 04:01:43'),(50,17,'标准总线','讲解各种常见计算机总线标准',NULL,35,4,'2025-07-12 03:52:26','2025-07-12 04:01:43'),(51,18,'并行处理概述','介绍并行处理的基本概念和分类',NULL,35,1,'2025-07-12 03:52:26','2025-07-12 04:01:43'),(52,18,'并行处理机构','讲解各种并行处理机构的设计和特点',NULL,40,2,'2025-07-12 03:52:26','2025-07-12 04:01:43'),(53,18,'多处理器系统','介绍多处理器系统的组织方式和通信机制',NULL,45,3,'2025-07-12 03:52:26','2025-07-12 04:01:43'),(54,18,'多核处理器','讲解多核处理器的架构和设计技术',NULL,40,4,'2025-07-12 03:52:26','2025-07-12 04:01:43'),(55,18,'并行编程基础','介绍并行程序设计的基本原理和方法',NULL,45,5,'2025-07-12 03:52:26','2025-07-12 04:01:43'),(57,19,'冯·诺依曼结构','详细讲解冯·诺依曼计算机体系结构的核心思想','',40,2,'2025-07-12 03:59:23','2025-07-12 03:59:23'),(58,19,'计算机的层次结构','介绍现代计算机的层次结构模型','',30,3,'2025-07-12 03:59:23','2025-07-12 03:59:23'),(59,19,'计算机性能评估','讲解衡量计算机性能的主要指标和方法','',35,4,'2025-07-12 03:59:23','2025-07-12 03:59:23'),(60,19,'计算机分类与发展历程','介绍计算机的分类方式和发展历史','',45,5,'2025-07-12 03:59:23','2025-07-12 03:59:23'),(61,19,'CISC与RISC架构','比较复杂指令集和精简指令集计算机架构','',40,6,'2025-07-12 03:59:23','2025-07-12 03:59:23'),(62,19,'计算机发展趋势','分析计算机技术的未来发展方向','',25,7,'2025-07-12 03:59:23','2025-07-12 03:59:23'),(63,20,'数制与码制','介绍二进制、八进制、十进制、十六进制及其转换','',35,1,'2025-07-12 03:59:23','2025-07-12 03:59:23'),(64,20,'数据在计算机中的表示','讲解定点数和浮点数的表示方法','',40,2,'2025-07-12 03:59:23','2025-07-12 03:59:23'),(65,20,'整数的表示与运算','介绍原码、反码、补码及其加减运算','',45,3,'2025-07-12 03:59:23','2025-07-12 03:59:23'),(66,20,'浮点数的表示与运算','详细讲解IEEE 754标准和浮点运算','',50,4,'2025-07-12 03:59:23','2025-07-12 03:59:23'),(67,20,'定点运算器设计','介绍加法器、减法器、乘法器和除法器的设计','',45,5,'2025-07-12 03:59:23','2025-07-12 03:59:23'),(68,20,'浮点运算器设计','讲解浮点数加、减、乘、除运算器的实现','',50,6,'2025-07-12 03:59:23','2025-07-12 03:59:23'),(69,20,'算术逻辑单元(ALU)','详解ALU的功能和内部结构设计','',40,7,'2025-07-12 03:59:23','2025-07-12 03:59:23'),(70,20,'数值计算中的精度与误差','分析数值计算中的精度问题和误差控制','',30,8,'2025-07-12 03:59:23','2025-07-12 03:59:23');

/*Table structure for table `section_comment` */

DROP TABLE IF EXISTS `section_comment`;

CREATE TABLE `section_comment` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '评论ID，主键',
  `section_id` bigint(20) NOT NULL COMMENT '所属小节ID，外键',
  `user_id` bigint(20) NOT NULL COMMENT '评论人ID，外键',
  `content` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '评论内容',
  `parent_id` bigint(20) DEFAULT NULL COMMENT '父评论ID，为NULL表示一级评论',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`) USING BTREE,
  KEY `idx_section_id` (`section_id`) USING BTREE,
  KEY `idx_user_id` (`user_id`) USING BTREE,
  KEY `idx_parent_id` (`parent_id`) USING BTREE,
  CONSTRAINT `fk_comment_parent` FOREIGN KEY (`parent_id`) REFERENCES `section_comment` (`id`) ON DELETE SET NULL ON UPDATE CASCADE,
  CONSTRAINT `fk_comment_section` FOREIGN KEY (`section_id`) REFERENCES `section` (`id`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `fk_comment_user` FOREIGN KEY (`user_id`) REFERENCES `user` (`id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=77 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci ROW_FORMAT=DYNAMIC COMMENT='小节评论表(讨论区)';

/*Data for the table `section_comment` */

insert  into `section_comment`(`id`,`section_id`,`user_id`,`content`,`parent_id`,`create_time`,`update_time`) values (76,20,6,'讲的不错',NULL,'2025-07-12 07:57:24','2025-07-12 07:57:24');

/*Table structure for table `section_progress` */

DROP TABLE IF EXISTS `section_progress`;

CREATE TABLE `section_progress` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '进度记录ID，主键',
  `student_id` bigint(20) NOT NULL COMMENT '学生ID，外键',
  `section_id` bigint(20) NOT NULL COMMENT '小节ID，外键',
  `watched_time` int(11) NOT NULL DEFAULT '0' COMMENT '已观看时间(秒)',
  `is_finished` tinyint(1) NOT NULL DEFAULT '0' COMMENT '是否看完，0-未完成，1-已完成',
  `last_watch_time` datetime DEFAULT NULL COMMENT '上次观看时间',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE KEY `uk_student_section` (`student_id`,`section_id`) USING BTREE COMMENT '学生+小节唯一约束',
  KEY `idx_section_id` (`section_id`) USING BTREE,
  CONSTRAINT `fk_progress_section` FOREIGN KEY (`section_id`) REFERENCES `section` (`id`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `fk_progress_student` FOREIGN KEY (`student_id`) REFERENCES `user` (`id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci ROW_FORMAT=DYNAMIC COMMENT='学生观看进度记录表';

/*Data for the table `section_progress` */

/*Table structure for table `student` */

DROP TABLE IF EXISTS `student`;

CREATE TABLE `student` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '学生ID',
  `user_id` bigint(20) NOT NULL COMMENT '用户ID',
  `enrollment_status` varchar(20) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT 'ENROLLED' COMMENT '学籍状态',
  `gpa` decimal(3,2) DEFAULT NULL COMMENT 'GPA',
  `gpa_level` varchar(5) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT 'GPA等级',
  `create_time` datetime DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE KEY `uk_user_id` (`user_id`) USING BTREE,
  CONSTRAINT `fk_student_user` FOREIGN KEY (`user_id`) REFERENCES `user` (`id`) ON DELETE CASCADE ON UPDATE RESTRICT
) ENGINE=InnoDB AUTO_INCREMENT=20250006 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci ROW_FORMAT=DYNAMIC COMMENT='学生表';

/*Data for the table `student` */

insert  into `student`(`id`,`user_id`,`enrollment_status`,`gpa`,`gpa_level`,`create_time`,`update_time`) values (20250005,12,'ENROLLED',NULL,NULL,'2025-07-01 21:05:06','2025-07-01 21:05:06');

/*Table structure for table `student_answer` */

DROP TABLE IF EXISTS `student_answer`;

CREATE TABLE `student_answer` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `student_id` bigint(20) NOT NULL,
  `assignment_id` bigint(20) NOT NULL,
  `question_id` bigint(20) NOT NULL,
  `answer_content` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci COMMENT '学生答案',
  `is_correct` tinyint(1) DEFAULT NULL COMMENT '是否正确',
  `score` int(11) DEFAULT '0',
  `answer_time` datetime DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`) USING BTREE,
  KEY `student_id` (`student_id`) USING BTREE,
  KEY `assignment_id` (`assignment_id`) USING BTREE,
  KEY `question_id` (`question_id`) USING BTREE,
  CONSTRAINT `student_answer_ibfk_1` FOREIGN KEY (`student_id`) REFERENCES `student` (`id`) ON DELETE RESTRICT ON UPDATE RESTRICT,
  CONSTRAINT `student_answer_ibfk_2` FOREIGN KEY (`assignment_id`) REFERENCES `assignment` (`id`) ON DELETE RESTRICT ON UPDATE RESTRICT,
  CONSTRAINT `student_answer_ibfk_3` FOREIGN KEY (`question_id`) REFERENCES `question` (`id`) ON DELETE RESTRICT ON UPDATE RESTRICT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci ROW_FORMAT=DYNAMIC COMMENT='学生答题记录表';

/*Data for the table `student_answer` */

/*Table structure for table `teacher` */

DROP TABLE IF EXISTS `teacher`;

CREATE TABLE `teacher` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '教师ID',
  `user_id` bigint(20) NOT NULL COMMENT '用户ID',
  `department` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '所属院系',
  `title` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '职称',
  `education` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '学历',
  `specialty` varchar(200) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '专业领域',
  `introduction` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci COMMENT '个人简介',
  `office_location` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '办公地点',
  `office_hours` varchar(200) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '办公时间',
  `contact_email` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '联系邮箱',
  `contact_phone` varchar(20) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '联系电话',
  `status` varchar(20) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT 'ACTIVE' COMMENT '状态',
  `hire_date` datetime DEFAULT NULL COMMENT '入职日期',
  `create_time` datetime DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE KEY `uk_user_id` (`user_id`) USING BTREE,
  KEY `idx_department` (`department`) USING BTREE,
  CONSTRAINT `fk_teacher_user` FOREIGN KEY (`user_id`) REFERENCES `user` (`id`) ON DELETE CASCADE ON UPDATE RESTRICT
) ENGINE=InnoDB AUTO_INCREMENT=2025005 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci ROW_FORMAT=DYNAMIC COMMENT='教师表';

/*Data for the table `teacher` */

insert  into `teacher`(`id`,`user_id`,`department`,`title`,`education`,`specialty`,`introduction`,`office_location`,`office_hours`,`contact_email`,`contact_phone`,`status`,`hire_date`,`create_time`,`update_time`) values (2025000,2,'计算机科学与技术学院','教授','博士','人工智能','张教授是人工智能领域的专家','A栋201','周一至周五 9:00-17:00','teacher1@example.com',NULL,'ACTIVE',NULL,'2025-06-29 02:09:12','2025-06-29 02:09:12'),(2025001,3,'数学学院','副教授','博士','应用数学','李教授专注于应用数学研究','B栋305','周一至周五 10:00-16:00','teacher2@example.com',NULL,'ACTIVE',NULL,'2025-06-29 02:09:12','2025-06-29 02:09:12'),(2025002,6,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,'ACTIVE',NULL,'2025-06-29 02:12:20','2025-06-29 02:12:20'),(2025003,7,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,'ACTIVE',NULL,'2025-06-29 12:29:23','2025-06-29 12:29:23'),(2025004,9,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,'ACTIVE',NULL,'2025-06-29 18:16:10','2025-06-29 18:16:10');

/*Table structure for table `user` */

DROP TABLE IF EXISTS `user`;

CREATE TABLE `user` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '用户ID',
  `username` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '用户名',
  `password` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '密码(加密)',
  `email` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '邮箱',
  `real_name` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '真实姓名',
  `avatar` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '头像URL',
  `role` varchar(20) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '用户角色',
  `status` varchar(20) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT 'ACTIVE' COMMENT '状态',
  `create_time` datetime DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE KEY `uk_username` (`username`) USING BTREE,
  UNIQUE KEY `uk_email` (`email`) USING BTREE
) ENGINE=InnoDB AUTO_INCREMENT=13 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci ROW_FORMAT=DYNAMIC COMMENT='用户表';

/*Data for the table `user` */

insert  into `user`(`id`,`username`,`password`,`email`,`real_name`,`avatar`,`role`,`status`,`create_time`,`update_time`) values (1,'admin','$2a$10$N.zmdr9k7uOCQb376NoUnuTJ8iAt6Z5EHsNSEYy4k4onaZDe17ZOS','admin@example.com','系统管理员',NULL,'ADMIN','ACTIVE','2025-06-29 02:09:12','2025-06-29 02:09:12'),(2,'teacher1','$2a$10$N.zmdr9k7uOCQb376NoUnuTJ8iAt6Z5EHsNSEYy4k4onaZDe17ZOS','teacher1@example.com','张教授',NULL,'TEACHER','ACTIVE','2025-06-29 02:09:12','2025-06-29 02:09:12'),(3,'teacher2','$2a$10$N.zmdr9k7uOCQb376NoUnuTJ8iAt6Z5EHsNSEYy4k4onaZDe17ZOS','teacher2@example.com','李教授',NULL,'TEACHER','ACTIVE','2025-06-29 02:09:12','2025-06-29 02:09:12'),(6,'test01','$2a$10$yinM6zXGIZ7AWSXA8QquqOE6v28EEcb7hpaOoz53ikmbV7bT8jWJe','2825507827@qq.com','张教授',NULL,'TEACHER','ACTIVE','2025-06-29 02:12:20','2025-07-03 01:16:42'),(7,'test02','$2a$10$KoS8uj6MG4W.9W3lRL78yuFtC9rYZmFbQIxX/BmIX7nAJLkfME7he','2825507825@qq.com','测试教师',NULL,'TEACHER','ACTIVE','2025-06-29 12:29:23','2025-06-29 12:29:23'),(9,'th1','$2a$10$jkqmHsAkW.soiGElTZyk8OU2mC6YEUxCWizcgGsOly2wfQGljHY6.','28254437825@qq.com','测试',NULL,'TEACHER','ACTIVE','2025-06-29 18:16:10','2025-06-29 18:16:10'),(12,'stu01','$2a$10$hwXoEKCcsQXLLSM5AoHPh.qvWzWSKalBGq8XcCNxSsWMCy.9CQHvS','20234042@stu.neu.edu.cn','测试学生',NULL,'STUDENT','ACTIVE','2025-07-01 21:05:06','2025-07-01 21:05:06');

/*Table structure for table `wrong_question` */

DROP TABLE IF EXISTS `wrong_question`;

CREATE TABLE `wrong_question` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '主键',
  `student_id` bigint(20) NOT NULL COMMENT '学生ID',
  `question_id` bigint(20) NOT NULL COMMENT '题目ID',
  `assignment_id` bigint(20) DEFAULT NULL COMMENT '所属作业或考试ID',
  `submission_id` bigint(20) DEFAULT NULL COMMENT '作业提交ID',
  `wrong_time` datetime DEFAULT CURRENT_TIMESTAMP COMMENT '记录时间',
  `student_answer` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci COMMENT '学生当时的答案',
  `correct_answer` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci COMMENT '正确答案（冗余存储）',
  PRIMARY KEY (`id`) USING BTREE,
  KEY `student_id` (`student_id`) USING BTREE,
  KEY `question_id` (`question_id`) USING BTREE,
  CONSTRAINT `fk_wrong_question_question` FOREIGN KEY (`question_id`) REFERENCES `question` (`id`) ON DELETE CASCADE ON UPDATE RESTRICT,
  CONSTRAINT `fk_wrong_question_student` FOREIGN KEY (`student_id`) REFERENCES `student` (`id`) ON DELETE CASCADE ON UPDATE RESTRICT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci ROW_FORMAT=DYNAMIC COMMENT='学生错题本';

/*Data for the table `wrong_question` */

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;
