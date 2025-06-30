-- 修改question表的外键约束，将created_by字段的外键引用从teacher表改为user表

-- 首先删除现有的外键约束
ALTER TABLE question DROP FOREIGN KEY question_ibfk_3;

-- 然后添加新的外键约束，引用user表
ALTER TABLE question ADD CONSTRAINT question_ibfk_3 FOREIGN KEY (created_by) REFERENCES user(id);

-- 添加注释，说明修改原因
-- 这个修改是因为created_by字段应该存储的是用户ID，而不是教师ID
-- 用户表中包含所有用户（包括教师、学生和管理员），而教师表只包含教师
-- 在系统中，创建题目的用户不一定是教师，也可能是管理员或其他角色 