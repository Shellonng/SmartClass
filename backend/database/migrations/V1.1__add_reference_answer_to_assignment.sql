-- 添加参考答案字段到 assignment 表
-- 用于智能批改功能

-- 为 assignment 表添加 reference_answer 字段
ALTER TABLE `assignment` 
ADD COLUMN `reference_answer` TEXT NULL COMMENT '参考答案（用于智能批改）' AFTER `description`;

-- 添加索引以提高查询性能
CREATE INDEX `idx_assignment_reference_answer` ON `assignment`(`reference_answer`(100));

-- 更新现有记录的注释
ALTER TABLE `assignment` 
MODIFY COLUMN `reference_answer` TEXT NULL COMMENT '参考答案，用于AI智能批改功能，支持文本格式的标准答案';

-- 记录迁移完成
INSERT INTO `migration_log` (`version`, `description`, `executed_at`) 
VALUES ('V1.1', '添加智能批改功能所需的参考答案字段', NOW())
ON DUPLICATE KEY UPDATE `executed_at` = NOW(); 