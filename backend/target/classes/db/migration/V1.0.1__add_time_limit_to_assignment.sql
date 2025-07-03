-- 向 assignment 表添加 time_limit 字段
ALTER TABLE assignment ADD COLUMN time_limit INT COMMENT '时间限制（分钟）'; 