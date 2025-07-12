-- 完整修复作业功能的数据库补丁
-- 执行此脚本来修复assignment表的字段缺失问题

USE education_platform;

-- 1. 添加缺失的字段到assignment表（如果字段已存在会报错，可忽略）
ALTER TABLE `assignment` 
ADD COLUMN `total_score` INT DEFAULT 100 COMMENT '总分' AFTER `time_limit`;

ALTER TABLE `assignment` 
ADD COLUMN `duration` INT DEFAULT NULL COMMENT '考试时长（分钟）' AFTER `total_score`;

ALTER TABLE `assignment` 
ADD COLUMN `allowed_file_types` TEXT DEFAULT NULL COMMENT '允许的文件类型（JSON格式）' AFTER `duration`;

ALTER TABLE `assignment` 
ADD COLUMN `max_file_size` INT DEFAULT 10 COMMENT '最大文件大小（MB）' AFTER `allowed_file_types`;

ALTER TABLE `assignment` 
ADD COLUMN `reference_answer` TEXT DEFAULT NULL COMMENT '参考答案（用于智能批改）' AFTER `max_file_size`;

-- 2. 创建作业配置表，用于存储智能组卷配置
CREATE TABLE IF NOT EXISTS `assignment_config` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `assignment_id` bigint(20) NOT NULL COMMENT '作业ID',
  `knowledge_points` TEXT DEFAULT NULL COMMENT '知识点范围（JSON格式）',
  `difficulty` ENUM('EASY','MEDIUM','HARD') DEFAULT 'MEDIUM' COMMENT '难度级别',
  `question_count` INT DEFAULT 10 COMMENT '题目总数',
  `question_types` TEXT DEFAULT NULL COMMENT '题型分布（JSON格式）',
  `additional_requirements` TEXT DEFAULT NULL COMMENT '额外要求',
  `create_time` datetime DEFAULT CURRENT_TIMESTAMP,
  `update_time` datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_assignment_config` (`assignment_id`),
  CONSTRAINT `fk_assignment_config_assignment` FOREIGN KEY (`assignment_id`) REFERENCES `assignment` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='作业配置表';

-- 3. 验证字段添加成功
SHOW COLUMNS FROM `assignment` WHERE Field IN ('total_score', 'duration', 'allowed_file_types', 'max_file_size', 'reference_answer');

-- 4. 验证配置表创建成功
SHOW TABLES LIKE 'assignment_config';

-- 显示执行结果
SELECT '作业功能数据库修复完成！' AS result;