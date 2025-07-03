CREATE TABLE IF NOT EXISTS `assignment_submission_answer` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '主键ID',
  `submission_id` bigint(20) NOT NULL COMMENT '提交记录ID',
  `question_id` bigint(20) NOT NULL COMMENT '题目ID',
  `student_id` bigint(20) NOT NULL COMMENT '学生ID',
  `answer_content` text COMMENT '答案内容',
  `is_correct` tinyint(1) DEFAULT NULL COMMENT '是否正确',
  `score` int(11) DEFAULT '0' COMMENT '得分',
  `create_time` datetime DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  KEY `idx_submission_id` (`submission_id`),
  KEY `idx_question_id` (`question_id`),
  KEY `idx_student_id` (`student_id`),
  CONSTRAINT `fk_answer_submission` FOREIGN KEY (`submission_id`) REFERENCES `assignment_submission` (`id`) ON DELETE CASCADE,
  CONSTRAINT `fk_answer_question` FOREIGN KEY (`question_id`) REFERENCES `question` (`id`) ON DELETE CASCADE,
  CONSTRAINT `fk_answer_user` FOREIGN KEY (`student_id`) REFERENCES `user` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='作业题目答案表'; 