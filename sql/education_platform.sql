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
  `title` varchar(100) NOT NULL COMMENT '作业或考试标题',
  `course_id` bigint(20) NOT NULL,
  `user_id` bigint(20) NOT NULL COMMENT '发布作业的用户ID',
  `type` enum('homework','exam') DEFAULT 'homework',
  `description` text,
  `start_time` datetime DEFAULT NULL,
  `end_time` datetime DEFAULT NULL,
  `create_time` datetime DEFAULT CURRENT_TIMESTAMP,
  `status` tinyint(4) NOT NULL DEFAULT '0' COMMENT '发布状态：0未发布，1已发布',
  `update_time` datetime DEFAULT NULL,
  `mode` enum('question','file') NOT NULL DEFAULT 'question' COMMENT '作业模式：question-答题型，file-上传型',
  `time_limit` int(11) DEFAULT NULL COMMENT '时间限制（分钟）',
  PRIMARY KEY (`id`),
  KEY `course_id` (`course_id`),
  KEY `fk_assignment_user` (`user_id`),
  CONSTRAINT `assignment_ibfk_1` FOREIGN KEY (`course_id`) REFERENCES `course` (`id`),
  CONSTRAINT `fk_assignment_user` FOREIGN KEY (`user_id`) REFERENCES `user` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=26 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='作业或考试表';

/*Data for the table `assignment` */

insert  into `assignment`(`id`,`title`,`course_id`,`user_id`,`type`,`description`,`start_time`,`end_time`,`create_time`,`status`,`update_time`,`mode`,`time_limit`) values (9,'测试1',9,6,'exam','测试1的说明','2025-07-01 14:00:00','2025-07-10 16:00:00','2025-07-01 14:53:32',1,'2025-07-03 01:46:44','question',NULL),(15,'12',9,3,'exam','132','2025-07-01 00:00:00','2025-07-01 01:00:00','2025-07-01 18:19:09',0,'2025-07-01 18:19:09','question',NULL),(17,'作业2',9,3,'homework','','2025-07-01 00:00:00','2025-07-01 01:00:00','2025-07-01 18:41:05',0,'2025-07-02 19:35:30','file',NULL),(20,'测试2',9,6,'exam','','2025-07-01 01:00:00','2025-07-01 02:00:00','2025-07-01 19:05:51',0,'2025-07-01 19:05:59','question',NULL),(23,'3232',9,6,'homework','','2025-07-02 23:56:06','2025-07-03 23:07:00','2025-07-01 23:55:16',1,'2025-07-03 01:48:05','question',NULL),(24,'1',9,6,'homework','','2025-07-02 20:07:35','2025-07-02 21:00:00','2025-07-02 20:07:41',1,'2025-07-02 20:07:41','file',NULL),(25,'2',9,6,'homework','','2025-07-03 20:12:48','2025-07-05 20:12:55','2025-07-02 20:12:58',0,'2025-07-02 20:12:58','file',NULL);

/*Table structure for table `assignment_question` */

DROP TABLE IF EXISTS `assignment_question`;

CREATE TABLE `assignment_question` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `assignment_id` bigint(20) NOT NULL,
  `question_id` bigint(20) NOT NULL,
  `score` int(11) DEFAULT '5' COMMENT '该题满分',
  `sequence` int(11) DEFAULT '1' COMMENT '题目顺序',
  PRIMARY KEY (`id`),
  KEY `assignment_id` (`assignment_id`),
  KEY `question_id` (`question_id`),
  CONSTRAINT `assignment_question_ibfk_1` FOREIGN KEY (`assignment_id`) REFERENCES `assignment` (`id`) ON DELETE CASCADE,
  CONSTRAINT `assignment_question_ibfk_2` FOREIGN KEY (`question_id`) REFERENCES `question` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=31 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='作业题目关联表';

/*Data for the table `assignment_question` */

insert  into `assignment_question`(`id`,`assignment_id`,`question_id`,`score`,`sequence`) values (18,9,13,5,1),(19,9,17,5,2),(20,9,12,5,3),(21,9,12,0,4),(28,17,13,5,1),(29,17,12,5,2),(30,23,13,5,1);

/*Table structure for table `assignment_submission` */

DROP TABLE IF EXISTS `assignment_submission`;

CREATE TABLE `assignment_submission` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '主键ID',
  `assignment_id` bigint(20) NOT NULL COMMENT '作业ID',
  `student_id` bigint(20) NOT NULL COMMENT '学生ID',
  `status` int(11) NOT NULL DEFAULT '0' COMMENT '状态：0-未提交，1-已提交未批改，2-已批改',
  `score` int(11) DEFAULT NULL COMMENT '得分',
  `feedback` text COMMENT '教师反馈',
  `submit_time` datetime DEFAULT NULL COMMENT '提交时间',
  `grade_time` datetime DEFAULT NULL COMMENT '批改时间',
  `graded_by` bigint(20) DEFAULT NULL COMMENT '批改人ID',
  `content` text COMMENT '提交内容',
  `create_time` datetime DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  KEY `idx_assignment_id` (`assignment_id`),
  KEY `idx_student_id` (`student_id`),
  KEY `idx_status` (`status`),
  CONSTRAINT `fk_submission_assignment` FOREIGN KEY (`assignment_id`) REFERENCES `assignment` (`id`) ON DELETE CASCADE,
  CONSTRAINT `fk_submission_student` FOREIGN KEY (`student_id`) REFERENCES `user` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=13 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='作业提交记录表';

/*Data for the table `assignment_submission` */

insert  into `assignment_submission`(`id`,`assignment_id`,`student_id`,`status`,`score`,`feedback`,`submit_time`,`grade_time`,`graded_by`,`content`,`create_time`,`update_time`) values (4,23,12,1,NULL,NULL,'2025-07-02 00:21:05',NULL,NULL,'这是学生提交的作业内容','2025-07-02 00:21:05','2025-07-02 00:21:05'),(5,23,12,2,85,'做得不错，但有些地方需要改进','2025-07-01 23:21:05','2025-07-02 00:21:05',3,'这是另一个学生提交的作业内容','2025-07-02 00:21:05','2025-07-02 00:21:05'),(12,9,12,0,NULL,NULL,NULL,NULL,NULL,NULL,'2025-07-03 02:53:30','2025-07-03 02:53:29');

/*Table structure for table `assignment_submission_answer` */

DROP TABLE IF EXISTS `assignment_submission_answer`;

CREATE TABLE `assignment_submission_answer` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '主键',
  `submission_id` bigint(20) NOT NULL COMMENT '提交ID，关联assignment_submission',
  `question_id` bigint(20) NOT NULL COMMENT '题目ID',
  `student_answer` text COMMENT '学生的答案',
  `is_correct` tinyint(1) DEFAULT NULL COMMENT '是否正确：1-正确，0-错误，NULL-未批改',
  `score` int(11) DEFAULT NULL COMMENT '得分（教师批改后记录）',
  `comment` text COMMENT '教师对该题的点评（可选）',
  `create_time` datetime DEFAULT CURRENT_TIMESTAMP,
  `update_time` datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `submission_id` (`submission_id`),
  KEY `question_id` (`question_id`),
  CONSTRAINT `fk_submission_answer_question` FOREIGN KEY (`question_id`) REFERENCES `question` (`id`) ON DELETE CASCADE,
  CONSTRAINT `fk_submission_answer_submission` FOREIGN KEY (`submission_id`) REFERENCES `assignment_submission` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='学生答题记录表';

/*Data for the table `assignment_submission_answer` */

/*Table structure for table `assignment_submission_file` */

DROP TABLE IF EXISTS `assignment_submission_file`;

CREATE TABLE `assignment_submission_file` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '主键ID',
  `student_id` bigint(20) NOT NULL COMMENT '学生ID',
  `assignment_id` bigint(20) NOT NULL COMMENT '作业ID',
  `file_url` varchar(255) NOT NULL COMMENT '上传文件路径',
  `file_type` varchar(50) DEFAULT NULL COMMENT '文件类型（docx/pdf/ppt/mp4/jpg/png等）',
  `upload_time` datetime DEFAULT CURRENT_TIMESTAMP COMMENT '上传时间',
  `comment` text COMMENT '教师评语',
  `score` int(11) DEFAULT '0' COMMENT '评分',
  PRIMARY KEY (`id`),
  KEY `student_id` (`student_id`),
  KEY `assignment_id` (`assignment_id`),
  CONSTRAINT `assignment_submission_file_ibfk_1` FOREIGN KEY (`student_id`) REFERENCES `student` (`id`) ON DELETE CASCADE,
  CONSTRAINT `assignment_submission_file_ibfk_2` FOREIGN KEY (`assignment_id`) REFERENCES `assignment` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='学生作业上传文件表';

/*Data for the table `assignment_submission_file` */

/*Table structure for table `chapter` */

DROP TABLE IF EXISTS `chapter`;

CREATE TABLE `chapter` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '章节ID，主键',
  `course_id` bigint(20) NOT NULL COMMENT '所属课程ID，外键',
  `title` varchar(255) NOT NULL COMMENT '章节名称',
  `description` text,
  `sort_order` int(11) NOT NULL,
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  KEY `idx_course_id` (`course_id`),
  CONSTRAINT `fk_chapter_course` FOREIGN KEY (`course_id`) REFERENCES `course` (`id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=13 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='课程章节表';

/*Data for the table `chapter` */

insert  into `chapter`(`id`,`course_id`,`title`,`description`,`sort_order`,`create_time`,`update_time`) values (11,9,'计算机系统的组成','介绍计算机系统的基本概念和发展历史',1,'2025-06-29 16:22:39','2025-07-03 01:41:18'),(12,9,'数字逻辑基础','介绍数字电路和逻辑设计的基本原理',2,'2025-06-29 21:07:41','2025-07-03 01:42:55');

/*Table structure for table `class_student` */

DROP TABLE IF EXISTS `class_student`;

CREATE TABLE `class_student` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `class_id` bigint(20) NOT NULL,
  `student_id` bigint(20) NOT NULL,
  `join_time` datetime DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_class_student` (`class_id`,`student_id`),
  KEY `student_id` (`student_id`),
  CONSTRAINT `class_student_ibfk_1` FOREIGN KEY (`class_id`) REFERENCES `course_class` (`id`) ON DELETE CASCADE,
  CONSTRAINT `class_student_ibfk_2` FOREIGN KEY (`student_id`) REFERENCES `student` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=2 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='班级学生关联表';

/*Data for the table `class_student` */

insert  into `class_student`(`id`,`class_id`,`student_id`,`join_time`) values (1,10,20250005,NULL);

/*Table structure for table `course` */

DROP TABLE IF EXISTS `course`;

CREATE TABLE `course` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '课程ID',
  `title` varchar(100) COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '课程名称',
  `description` text COLLATE utf8mb4_unicode_ci COMMENT '课程描述',
  `cover_image` varchar(500) COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '课程封面图片',
  `credit` decimal(3,1) DEFAULT '3.0' COMMENT '学分',
  `course_type` varchar(20) COLLATE utf8mb4_unicode_ci DEFAULT '必修课' COMMENT '课程类型',
  `start_time` datetime DEFAULT NULL COMMENT '开始时间',
  `end_time` datetime DEFAULT NULL COMMENT '结束时间',
  `teacher_id` bigint(20) NOT NULL COMMENT '教师ID',
  `status` varchar(20) COLLATE utf8mb4_unicode_ci DEFAULT '未开始' COMMENT '课程状态',
  `term` varchar(20) COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '学期',
  `student_count` int(11) DEFAULT '0' COMMENT '学生数量',
  `average_score` decimal(5,2) DEFAULT NULL COMMENT '平均分数',
  `create_time` datetime DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  KEY `idx_teacher_id` (`teacher_id`),
  KEY `idx_term` (`term`),
  KEY `idx_status` (`status`),
  CONSTRAINT `fk_course_teacher` FOREIGN KEY (`teacher_id`) REFERENCES `teacher` (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=30 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='课程表';

/*Data for the table `course` */

insert  into `course`(`id`,`title`,`description`,`cover_image`,`credit`,`course_type`,`start_time`,`end_time`,`teacher_id`,`status`,`term`,`student_count`,`average_score`,`create_time`,`update_time`) values (1,'Java编程基础','Java编程语言入门课程',NULL,'3.0','必修课','2024-09-01 08:00:00','2024-12-31 17:00:00',2025000,'未开始','2024-2025-1',0,NULL,'2025-06-29 02:09:12','2025-06-29 02:09:12'),(2,'高等数学','高等数学基础课程',NULL,'4.0','必修课','2024-09-01 08:00:00','2024-12-31 17:00:00',2025001,'未开始','2024-2025-1',0,NULL,'2025-06-29 02:09:12','2025-06-29 02:09:12'),(3,'人工智能导论','人工智能基础理论与应用',NULL,'3.0','选修课','2024-09-01 08:00:00','2024-12-31 17:00:00',2025000,'未开始','2024-2025-1',0,NULL,'2025-06-29 02:09:12','2025-06-29 02:09:12'),(5,'测试课程','123','','3.0','必修课','2025-07-03 02:15:34','2025-07-11 02:15:36',2025002,'未开始','2024-2025-1',0,NULL,'2025-06-29 02:15:41','2025-06-29 02:15:41'),(6,'123','123','/api/photo/202507/799f6631a8604aeaa5a59b4416cf43da.png','3.0','必修课','2025-07-23 02:15:34','2025-08-01 02:15:36',2025002,'未开始','2024-2025-1',0,NULL,'2025-06-29 02:15:59','2025-07-02 11:51:33'),(9,'计算机组成原理','本课程介绍计算机系统的基本组成和工作原理，包括数字逻辑、CPU结构、存储系统、输入输出系统等内容。','https://img.freepik.com/free-vector/english-school-landing-page-template_23-2148475038.jpg','4.0','必修课','2025-06-04 02:52:20','2025-06-26 02:52:22',2025002,'已结束','2024-2025-1',1,NULL,'2025-06-29 02:52:54','2025-07-03 01:13:39'),(19,'Java程序设计','Java编程基础课程，零基础向对象编程思想和Java核心技术，为开发企业级应用打下基础','https://img.freepik.com/free-vector/programming-concept-illustration_114360-1670.jpg','3.0','必修课','2024-05-01 00:00:00','2024-12-31 00:00:00',2025002,'已结束','2024-2025-1',16421,NULL,'2025-07-02 02:13:16','2025-07-02 02:13:16'),(20,'数据结构与算法','深入学习常用数据结构和算法设计技巧，包括数组、链表、栈、队列、树、图以及常见算法','https://img.freepik.com/free-vector/programming-concept-illustration_114360-1213.jpg','4.0','必修课','2024-05-01 00:00:00','2024-12-31 00:00:00',2025002,'已结束','2024-2025-1',9850,NULL,'2025-07-02 02:13:16','2025-07-02 02:13:16'),(21,'Python程序基础','零基础入门Python编程，掌握Python基本语法、数据类型、控制结构、函数和模块开发','https://img.freepik.com/free-vector/programming-concept-illustration_114360-1351.jpg','3.0','必修课','2024-05-01 00:00:00','2024-12-31 00:00:00',2025002,'已结束','2024-2025-1',23190,NULL,'2025-07-02 02:13:16','2025-07-02 02:13:16'),(22,'微观经济学原理','介绍微观经济学的基本理论，包括供需关系、市场结构、消费行为、生产理论等','https://img.freepik.com/free-vector/economy-concept-illustration_114360-7385.jpg','3.0','必修课','2024-06-01 00:00:00','2024-12-31 00:00:00',2025002,'已结束','2024-2025-1',11201,NULL,'2025-07-02 02:13:16','2025-07-02 02:13:16'),(23,'线性代数','系统学习线性代数的基本概念和方法，包括向量运算、行列式、向量空间、特征值和特征向量','https://img.freepik.com/free-vector/mathematics-concept-illustration_114360-3972.jpg','4.0','必修课','2024-06-01 00:00:00','2024-12-31 00:00:00',2025002,'已结束','2024-2025-1',13580,NULL,'2025-07-02 02:13:16','2025-07-02 02:13:16'),(24,'大学物理（力学部分）','系统讲授经典力学的基本概念、定律和方法，包括刚体力学、振动力学、流体力学等','https://img.freepik.com/free-vector/physics-concept-illustration_114360-3972.jpg','3.0','必修课','2024-06-01 00:00:00','2024-12-31 00:00:00',2025002,'已结束','2024-2025-1',12680,NULL,'2025-07-02 02:13:16','2025-07-02 02:13:16'),(25,'高等数学（上）','本课程系统讲授高等数学的基本概念、理论和方法，包括极限、导数、微积分等','https://img.freepik.com/free-vector/hand-drawn-mathematics-background_23-2148157511.jpg','4.0','必修课','2025-07-01 11:57:20','2025-07-17 11:57:24',2025002,'进行中','2024-2025-1',15420,NULL,'2025-07-02 02:13:16','2025-07-02 11:57:30'),(26,'大学英语综合教程','提升英语听说读写综合能力，涵盖语法、词汇、阅读理解和写作技巧，为四六级考试做准备','https://img.freepik.com/free-vector/english-school-landing-page-template_23-2148475038.jpg','3.0','必修课','2024-06-01 00:00:00','2024-12-31 00:00:00',2025002,'已结束','2024-2025-1',18700,NULL,'2025-07-02 02:13:16','2025-07-02 02:13:16');

/*Table structure for table `course_class` */

DROP TABLE IF EXISTS `course_class`;

CREATE TABLE `course_class` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '班级ID',
  `course_id` bigint(20) DEFAULT NULL,
  `teacher_id` bigint(20) NOT NULL,
  `name` varchar(100) NOT NULL,
  `description` text,
  `is_default` tinyint(1) DEFAULT '0',
  `create_time` datetime DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `course_class_ibfk_1` (`course_id`),
  KEY `course_class_ibfk_2` (`teacher_id`),
  CONSTRAINT `course_class_ibfk_1` FOREIGN KEY (`course_id`) REFERENCES `course` (`id`) ON DELETE SET NULL,
  CONSTRAINT `course_class_ibfk_2` FOREIGN KEY (`teacher_id`) REFERENCES `user` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=17 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='课程班级表';

/*Data for the table `course_class` */

insert  into `course_class`(`id`,`course_id`,`teacher_id`,`name`,`description`,`is_default`,`create_time`) values (10,9,6,'软件工程2306班','123',0,'2025-07-02 14:08:30');

/*Table structure for table `course_enrollment_request` */

DROP TABLE IF EXISTS `course_enrollment_request`;

CREATE TABLE `course_enrollment_request` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '申请ID',
  `student_id` bigint(20) NOT NULL COMMENT '学生ID',
  `course_id` bigint(20) NOT NULL COMMENT '申请加入的课程ID',
  `status` tinyint(4) DEFAULT '0' COMMENT '申请状态：0=待审核 1=已通过 2=已拒绝',
  `reason` text COMMENT '学生申请理由',
  `review_comment` text COMMENT '教师审核意见',
  `submit_time` datetime DEFAULT CURRENT_TIMESTAMP,
  `review_time` datetime DEFAULT NULL COMMENT '审核时间',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_student_course` (`student_id`,`course_id`),
  KEY `course_id` (`course_id`),
  CONSTRAINT `course_enrollment_request_ibfk_1` FOREIGN KEY (`student_id`) REFERENCES `student` (`id`) ON DELETE CASCADE,
  CONSTRAINT `course_enrollment_request_ibfk_2` FOREIGN KEY (`course_id`) REFERENCES `course` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='学生选课申请表';

/*Data for the table `course_enrollment_request` */

/*Table structure for table `course_resource` */

DROP TABLE IF EXISTS `course_resource`;

CREATE TABLE `course_resource` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '资源ID',
  `course_id` bigint(20) NOT NULL COMMENT '所属课程ID',
  `name` varchar(255) NOT NULL COMMENT '资源名称',
  `file_type` varchar(50) NOT NULL COMMENT '文件类型，如pdf、doc、ppt等',
  `file_size` bigint(20) NOT NULL COMMENT '文件大小(字节)',
  `file_url` varchar(500) NOT NULL COMMENT '文件URL',
  `description` varchar(500) DEFAULT NULL COMMENT '资源描述',
  `download_count` int(11) DEFAULT '0' COMMENT '下载次数',
  `upload_user_id` bigint(20) NOT NULL COMMENT '上传用户ID',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  KEY `idx_course_id` (`course_id`) COMMENT '课程ID索引',
  KEY `idx_upload_user_id` (`upload_user_id`) COMMENT '上传用户ID索引'
) ENGINE=InnoDB AUTO_INCREMENT=6 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='课程资源表';

/*Data for the table `course_resource` */

insert  into `course_resource`(`id`,`course_id`,`name`,`file_type`,`file_size`,`file_url`,`description`,`download_count`,`upload_user_id`,`create_time`,`update_time`) values (1,9,'1','pdf',325858,'/files/resources/9/202506/7acbf820bec64795bedaca556c235c4a.pdf','1',4,6,'2025-06-30 16:18:14','2025-06-30 16:18:14'),(5,9,'测试','png',48173,'/files/resources/9/202507/a54ec5a02dba4fc9882aaa1f23caf063.png','1',1,6,'2025-07-02 20:39:01','2025-07-02 20:39:01');

/*Table structure for table `course_student` */

DROP TABLE IF EXISTS `course_student`;

CREATE TABLE `course_student` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `course_id` bigint(20) NOT NULL,
  `student_id` bigint(20) NOT NULL,
  `enroll_time` datetime DEFAULT CURRENT_TIMESTAMP,
  `collected` int(11) DEFAULT '0' COMMENT '课程是否被该学生收藏，1为被收藏，0为未被收藏',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_course_student` (`course_id`,`student_id`),
  KEY `student_id` (`student_id`),
  CONSTRAINT `course_student_ibfk_1` FOREIGN KEY (`course_id`) REFERENCES `course` (`id`) ON DELETE CASCADE,
  CONSTRAINT `course_student_ibfk_2` FOREIGN KEY (`student_id`) REFERENCES `student` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=5 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='学生选课表';

/*Data for the table `course_student` */

insert  into `course_student`(`id`,`course_id`,`student_id`,`enroll_time`,`collected`) values (1,9,20250005,NULL,0),(2,19,20250005,NULL,0),(3,22,20250005,NULL,0);

/*Table structure for table `question` */

DROP TABLE IF EXISTS `question`;

CREATE TABLE `question` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `title` text NOT NULL COMMENT '题干内容',
  `question_type` enum('single','multiple','true_false','blank','short','code') NOT NULL COMMENT '题型',
  `difficulty` tinyint(4) NOT NULL DEFAULT '3' COMMENT '难度等级，1~5整数',
  `correct_answer` text COMMENT '标准答案',
  `explanation` text COMMENT '答案解析',
  `knowledge_point` varchar(100) DEFAULT NULL COMMENT '知识点',
  `course_id` bigint(20) NOT NULL,
  `chapter_id` bigint(20) NOT NULL,
  `created_by` bigint(20) NOT NULL COMMENT '出题教师ID',
  `create_time` datetime DEFAULT CURRENT_TIMESTAMP,
  `update_time` datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `course_id` (`course_id`),
  KEY `chapter_id` (`chapter_id`),
  KEY `question_ibfk_3` (`created_by`),
  CONSTRAINT `question_ibfk_1` FOREIGN KEY (`course_id`) REFERENCES `course` (`id`),
  CONSTRAINT `question_ibfk_2` FOREIGN KEY (`chapter_id`) REFERENCES `chapter` (`id`),
  CONSTRAINT `question_ibfk_3` FOREIGN KEY (`created_by`) REFERENCES `user` (`id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=18 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='题目表';

/*Data for the table `question` */

insert  into `question`(`id`,`title`,`question_type`,`difficulty`,`correct_answer`,`explanation`,`knowledge_point`,`course_id`,`chapter_id`,`created_by`,`create_time`,`update_time`) values (12,'21','blank',3,'2132','132','Vue',9,11,6,'2025-06-30 21:40:18','2025-07-01 01:54:19'),(13,'1=1==1=1111','single',3,'B','123123','JavaScript',9,11,6,'2025-06-30 21:51:52','2025-07-01 15:31:34'),(17,'我说的对不对（ ）','true_false',3,'T','不对','22231 ',9,11,6,'2025-07-01 02:06:34','2025-07-01 02:10:22');

/*Table structure for table `question_bank` */

DROP TABLE IF EXISTS `question_bank`;

CREATE TABLE `question_bank` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `user_id` bigint(20) NOT NULL COMMENT '出题教师的用户ID',
  `title` varchar(100) NOT NULL COMMENT '题库名称',
  `description` text COMMENT '题库说明',
  `create_time` datetime DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `fk_question_bank_user` (`user_id`),
  CONSTRAINT `fk_question_bank_user` FOREIGN KEY (`user_id`) REFERENCES `user` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='题库表';

/*Data for the table `question_bank` */

/*Table structure for table `question_bank_item` */

DROP TABLE IF EXISTS `question_bank_item`;

CREATE TABLE `question_bank_item` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `bank_id` bigint(20) NOT NULL,
  `question_id` bigint(20) NOT NULL,
  PRIMARY KEY (`id`),
  KEY `bank_id` (`bank_id`),
  KEY `question_id` (`question_id`),
  CONSTRAINT `question_bank_item_ibfk_1` FOREIGN KEY (`bank_id`) REFERENCES `question_bank` (`id`) ON DELETE CASCADE,
  CONSTRAINT `question_bank_item_ibfk_2` FOREIGN KEY (`question_id`) REFERENCES `question` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='题库题目关联表';

/*Data for the table `question_bank_item` */

/*Table structure for table `question_image` */

DROP TABLE IF EXISTS `question_image`;

CREATE TABLE `question_image` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `question_id` bigint(20) NOT NULL,
  `image_url` varchar(255) NOT NULL COMMENT '图片URL或路径',
  `description` varchar(255) DEFAULT NULL COMMENT '图片说明',
  `sequence` int(11) DEFAULT '1' COMMENT '图片显示顺序',
  `upload_time` datetime DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `question_id` (`question_id`),
  CONSTRAINT `question_image_ibfk_1` FOREIGN KEY (`question_id`) REFERENCES `question` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='题目图片表';

/*Data for the table `question_image` */

/*Table structure for table `question_option` */

DROP TABLE IF EXISTS `question_option`;

CREATE TABLE `question_option` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `question_id` bigint(20) NOT NULL,
  `option_label` char(1) NOT NULL COMMENT '选项标识 A/B/C/D/T/F',
  `option_text` text NOT NULL COMMENT '选项内容',
  PRIMARY KEY (`id`),
  KEY `question_id` (`question_id`),
  CONSTRAINT `question_option_ibfk_1` FOREIGN KEY (`question_id`) REFERENCES `question` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=15 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='题目选项表';

/*Data for the table `question_option` */

insert  into `question_option`(`id`,`question_id`,`option_label`,`option_text`) values (7,12,'A',''),(8,13,'A','1231'),(9,13,'B','12321'),(10,13,'C','3244'),(11,13,'D','5234'),(13,17,'T','正确1'),(14,17,'F','错误23');

/*Table structure for table `question_stats` */

DROP TABLE IF EXISTS `question_stats`;

CREATE TABLE `question_stats` (
  `question_id` bigint(20) NOT NULL,
  `answer_count` int(11) DEFAULT '0' COMMENT '答题总数',
  `correct_count` int(11) DEFAULT '0' COMMENT '正确人数',
  `accuracy` decimal(5,2) DEFAULT '0.00' COMMENT '正确率（百分比）',
  PRIMARY KEY (`question_id`),
  CONSTRAINT `question_stats_ibfk_1` FOREIGN KEY (`question_id`) REFERENCES `question` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='题目统计信息表';

/*Data for the table `question_stats` */

/*Table structure for table `section` */

DROP TABLE IF EXISTS `section`;

CREATE TABLE `section` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '小节ID，主键',
  `chapter_id` bigint(20) NOT NULL COMMENT '所属章节ID，外键',
  `title` varchar(255) NOT NULL COMMENT '小节名称',
  `description` text COMMENT '小节简介',
  `video_url` varchar(1024) DEFAULT NULL COMMENT '视频播放地址，可对接OSS',
  `duration` int(11) DEFAULT '0' COMMENT '视频时长(秒)',
  `sort_order` int(11) NOT NULL DEFAULT '0' COMMENT '小节顺序，用于排序',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  KEY `idx_chapter_id` (`chapter_id`),
  KEY `idx_sort_order` (`sort_order`),
  CONSTRAINT `fk_section_chapter` FOREIGN KEY (`chapter_id`) REFERENCES `chapter` (`id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=19 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='课程小节表';

/*Data for the table `section` */

insert  into `section`(`id`,`chapter_id`,`title`,`description`,`video_url`,`duration`,`sort_order`,`create_time`,`update_time`) values (14,11,'计算机系统的组成','计算机系统的组成','202506/d8f77cf2-597d-48d9-bc78-7010e3a915ec.mp4',45,1,'2025-06-30 12:33:31','2025-07-03 01:41:44'),(15,11,'计算机系统的层次结构','计算机系统的层次结构',NULL,30,2,'2025-06-30 14:23:30','2025-07-03 01:42:04'),(18,11,'计算机的性能指标','计算机的性能指标',NULL,25,3,'2025-07-03 01:42:20','2025-07-03 01:42:20');

/*Table structure for table `section_comment` */

DROP TABLE IF EXISTS `section_comment`;

CREATE TABLE `section_comment` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '评论ID，主键',
  `section_id` bigint(20) NOT NULL COMMENT '所属小节ID，外键',
  `user_id` bigint(20) NOT NULL COMMENT '评论人ID，外键',
  `content` text NOT NULL COMMENT '评论内容',
  `parent_id` bigint(20) DEFAULT NULL COMMENT '父评论ID，为NULL表示一级评论',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  KEY `idx_section_id` (`section_id`),
  KEY `idx_user_id` (`user_id`),
  KEY `idx_parent_id` (`parent_id`),
  CONSTRAINT `fk_comment_parent` FOREIGN KEY (`parent_id`) REFERENCES `section_comment` (`id`) ON DELETE SET NULL ON UPDATE CASCADE,
  CONSTRAINT `fk_comment_section` FOREIGN KEY (`section_id`) REFERENCES `section` (`id`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `fk_comment_user` FOREIGN KEY (`user_id`) REFERENCES `user` (`id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=75 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='小节评论表(讨论区)';

/*Data for the table `section_comment` */

insert  into `section_comment`(`id`,`section_id`,`user_id`,`content`,`parent_id`,`create_time`,`update_time`) values (71,14,6,'评论1',NULL,'2025-06-30 14:25:28','2025-06-30 14:25:28'),(72,14,6,'12',71,'2025-06-30 14:25:57','2025-06-30 14:25:57'),(73,14,6,'2',NULL,'2025-06-30 14:26:03','2025-06-30 14:26:03'),(74,14,6,'12',73,'2025-06-30 14:26:05','2025-06-30 14:26:05');

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
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_student_section` (`student_id`,`section_id`) COMMENT '学生+小节唯一约束',
  KEY `idx_section_id` (`section_id`),
  CONSTRAINT `fk_progress_section` FOREIGN KEY (`section_id`) REFERENCES `section` (`id`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `fk_progress_student` FOREIGN KEY (`student_id`) REFERENCES `user` (`id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='学生观看进度记录表';

/*Data for the table `section_progress` */

/*Table structure for table `student` */

DROP TABLE IF EXISTS `student`;

CREATE TABLE `student` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '学生ID',
  `user_id` bigint(20) NOT NULL COMMENT '用户ID',
  `enrollment_status` varchar(20) COLLATE utf8mb4_unicode_ci DEFAULT 'ENROLLED' COMMENT '学籍状态',
  `gpa` decimal(3,2) DEFAULT NULL COMMENT 'GPA',
  `gpa_level` varchar(5) COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT 'GPA等级',
  `create_time` datetime DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_user_id` (`user_id`),
  CONSTRAINT `fk_student_user` FOREIGN KEY (`user_id`) REFERENCES `user` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=20250006 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='学生表';

/*Data for the table `student` */

insert  into `student`(`id`,`user_id`,`enrollment_status`,`gpa`,`gpa_level`,`create_time`,`update_time`) values (20250005,12,'ENROLLED',NULL,NULL,'2025-07-01 21:05:06','2025-07-01 21:05:06');

/*Table structure for table `student_answer` */

DROP TABLE IF EXISTS `student_answer`;

CREATE TABLE `student_answer` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `student_id` bigint(20) NOT NULL,
  `assignment_id` bigint(20) NOT NULL,
  `question_id` bigint(20) NOT NULL,
  `answer_content` text COMMENT '学生答案',
  `is_correct` tinyint(1) DEFAULT NULL COMMENT '是否正确',
  `score` int(11) DEFAULT '0',
  `answer_time` datetime DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `student_id` (`student_id`),
  KEY `assignment_id` (`assignment_id`),
  KEY `question_id` (`question_id`),
  CONSTRAINT `student_answer_ibfk_1` FOREIGN KEY (`student_id`) REFERENCES `student` (`id`),
  CONSTRAINT `student_answer_ibfk_2` FOREIGN KEY (`assignment_id`) REFERENCES `assignment` (`id`),
  CONSTRAINT `student_answer_ibfk_3` FOREIGN KEY (`question_id`) REFERENCES `question` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='学生答题记录表';

/*Data for the table `student_answer` */

/*Table structure for table `teacher` */

DROP TABLE IF EXISTS `teacher`;

CREATE TABLE `teacher` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '教师ID',
  `user_id` bigint(20) NOT NULL COMMENT '用户ID',
  `department` varchar(100) COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '所属院系',
  `title` varchar(50) COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '职称',
  `education` varchar(50) COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '学历',
  `specialty` varchar(200) COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '专业领域',
  `introduction` text COLLATE utf8mb4_unicode_ci COMMENT '个人简介',
  `office_location` varchar(100) COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '办公地点',
  `office_hours` varchar(200) COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '办公时间',
  `contact_email` varchar(100) COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '联系邮箱',
  `contact_phone` varchar(20) COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '联系电话',
  `status` varchar(20) COLLATE utf8mb4_unicode_ci DEFAULT 'ACTIVE' COMMENT '状态',
  `hire_date` datetime DEFAULT NULL COMMENT '入职日期',
  `create_time` datetime DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_user_id` (`user_id`),
  KEY `idx_department` (`department`),
  CONSTRAINT `fk_teacher_user` FOREIGN KEY (`user_id`) REFERENCES `user` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=2025005 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='教师表';

/*Data for the table `teacher` */

insert  into `teacher`(`id`,`user_id`,`department`,`title`,`education`,`specialty`,`introduction`,`office_location`,`office_hours`,`contact_email`,`contact_phone`,`status`,`hire_date`,`create_time`,`update_time`) values (2025000,2,'计算机科学与技术学院','教授','博士','人工智能','张教授是人工智能领域的专家','A栋201','周一至周五 9:00-17:00','teacher1@example.com',NULL,'ACTIVE',NULL,'2025-06-29 02:09:12','2025-06-29 02:09:12'),(2025001,3,'数学学院','副教授','博士','应用数学','李教授专注于应用数学研究','B栋305','周一至周五 10:00-16:00','teacher2@example.com',NULL,'ACTIVE',NULL,'2025-06-29 02:09:12','2025-06-29 02:09:12'),(2025002,6,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,'ACTIVE',NULL,'2025-06-29 02:12:20','2025-06-29 02:12:20'),(2025003,7,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,'ACTIVE',NULL,'2025-06-29 12:29:23','2025-06-29 12:29:23'),(2025004,9,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,'ACTIVE',NULL,'2025-06-29 18:16:10','2025-06-29 18:16:10');

/*Table structure for table `user` */

DROP TABLE IF EXISTS `user`;

CREATE TABLE `user` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '用户ID',
  `username` varchar(50) COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '用户名',
  `password` varchar(255) COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '密码(加密)',
  `email` varchar(100) COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '邮箱',
  `real_name` varchar(50) COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '真实姓名',
  `avatar` varchar(255) COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '头像URL',
  `role` varchar(20) COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '用户角色',
  `status` varchar(20) COLLATE utf8mb4_unicode_ci DEFAULT 'ACTIVE' COMMENT '状态',
  `create_time` datetime DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_username` (`username`),
  UNIQUE KEY `uk_email` (`email`)
) ENGINE=InnoDB AUTO_INCREMENT=13 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='用户表';

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
  `student_answer` text COMMENT '学生当时的答案',
  `correct_answer` text COMMENT '正确答案（冗余存储）',
  PRIMARY KEY (`id`),
  KEY `student_id` (`student_id`),
  KEY `question_id` (`question_id`),
  CONSTRAINT `fk_wrong_question_question` FOREIGN KEY (`question_id`) REFERENCES `question` (`id`) ON DELETE CASCADE,
  CONSTRAINT `fk_wrong_question_student` FOREIGN KEY (`student_id`) REFERENCES `student` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='学生错题本';

/*Data for the table `wrong_question` */

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;
