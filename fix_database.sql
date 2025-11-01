-- 修复智能批改功能所需的数据库字段
-- 执行此脚本来添加缺少的 reference_answer 字段

USE education_platform;

-- 添加 reference_answer 字段到 assignment 表
ALTER TABLE `assignment` 
ADD COLUMN `reference_answer` TEXT NULL COMMENT '参考答案（用于智能批改）' AFTER `description`;

-- 验证字段添加成功
SHOW COLUMNS FROM `assignment` LIKE 'reference_answer';

-- 显示执行结果
SELECT 'reference_answer 字段添加成功！' AS result; 