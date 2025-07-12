-- 安全版本的作业功能数据库修复脚本
-- 此脚本会检查字段是否存在，避免重复添加导致的错误

USE education_platform;

-- 创建临时存储过程来安全添加字段
DELIMITER //

CREATE PROCEDURE AddColumnIfNotExists(
    IN tableName VARCHAR(128),
    IN columnName VARCHAR(128),
    IN columnDefinition TEXT
)
BEGIN
    DECLARE columnExists INT DEFAULT 0;
    
    SELECT COUNT(*) INTO columnExists
    FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_SCHEMA = DATABASE()
    AND TABLE_NAME = tableName
    AND COLUMN_NAME = columnName;
    
    IF columnExists = 0 THEN
        SET @sql = CONCAT('ALTER TABLE `', tableName, '` ADD COLUMN `', columnName, '` ', columnDefinition);
        PREPARE stmt FROM @sql;
        EXECUTE stmt;
        DEALLOCATE PREPARE stmt;
        SELECT CONCAT('字段 ', columnName, ' 已成功添加到表 ', tableName) AS result;
    ELSE
        SELECT CONCAT('字段 ', columnName, ' 已存在于表 ', tableName, '，跳过添加') AS result;
    END IF;
END//

DELIMITER ;

-- 1. 安全添加缺失的字段到assignment表
CALL AddColumnIfNotExists('assignment', 'total_score', 'INT DEFAULT 100 COMMENT \'总分\' AFTER `time_limit`');
CALL AddColumnIfNotExists('assignment', 'duration', 'INT DEFAULT NULL COMMENT \'考试时长（分钟）\' AFTER `total_score`');
CALL AddColumnIfNotExists('assignment', 'allowed_file_types', 'TEXT DEFAULT NULL COMMENT \'允许的文件类型（JSON格式）\' AFTER `duration`');
CALL AddColumnIfNotExists('assignment', 'max_file_size', 'INT DEFAULT 10 COMMENT \'最大文件大小（MB）\' AFTER `allowed_file_types`');
CALL AddColumnIfNotExists('assignment', 'reference_answer', 'TEXT DEFAULT NULL COMMENT \'参考答案（用于智能批改）\' AFTER `max_file_size`');

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

-- 3. 清理临时存储过程
DROP PROCEDURE IF EXISTS AddColumnIfNotExists;

-- 4. 验证字段添加成功
SHOW COLUMNS FROM `assignment` WHERE Field IN ('total_score', 'duration', 'allowed_file_types', 'max_file_size', 'reference_answer');

-- 5. 验证配置表创建成功
SHOW TABLES LIKE 'assignment_config';

-- 显示执行结果
SELECT '作业功能数据库修复完成！所有字段和表已安全创建或验证存在。' AS result;