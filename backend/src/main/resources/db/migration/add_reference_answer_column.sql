-- 添加参考答案字段到 assignment 表
-- 用于支持智能批改功能

ALTER TABLE assignment 
ADD COLUMN reference_answer TEXT COMMENT '参考答案，用于智能批改';

-- 更新已有数据的默认值（可选）
UPDATE assignment 
SET reference_answer = '' 
WHERE reference_answer IS NULL; 