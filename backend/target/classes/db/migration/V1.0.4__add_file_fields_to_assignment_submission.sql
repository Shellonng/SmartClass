-- 添加文件字段到作业提交表
ALTER TABLE assignment_submission ADD COLUMN file_name VARCHAR(255) COMMENT '文件名称';
ALTER TABLE assignment_submission ADD COLUMN file_path VARCHAR(255) COMMENT '文件路径'; 