-- 快速恢复用户数据脚本
-- 用于恢复被误删的用户数据
-- 使用方法: 在MySQL中执行此脚本

USE education_platform;

-- 清空现有用户数据（如果需要）
-- DELETE FROM user;
-- ALTER TABLE user AUTO_INCREMENT = 1;

-- 插入管理员用户
INSERT INTO user (username, password, email, real_name, role, status) VALUES 
('root', 'root123', 'root@education.com', 'Root管理员', 'ADMIN', 1),
('admin', 'admin123', 'admin@education.com', '系统管理员', 'ADMIN', 1);

-- 插入测试教师用户
INSERT INTO user (username, password, email, real_name, role, status) VALUES
('teacher1', 'teacher123', 'teacher1@education.com', '张教授', 'TEACHER', 1),
('teacher2', 'teacher123', 'teacher2@education.com', '李教授', 'TEACHER', 1),
('teacher3', 'teacher123', 'teacher3@education.com', '王老师', 'TEACHER', 1);

-- 插入测试学生用户
INSERT INTO user (username, password, email, real_name, role, status) VALUES
('student1', 'student123', 'student1@education.com', '张三', 'STUDENT', 1),
('student2', 'student123', 'student2@education.com', '李四', 'STUDENT', 1),
('student3', 'student123', 'student3@education.com', '王五', 'STUDENT', 1),
('student4', 'student123', 'student4@education.com', '赵六', 'STUDENT', 1),
('student5', 'student123', 'student5@education.com', '陈七', 'STUDENT', 1);

-- 验证插入结果
SELECT COUNT(*) as '用户总数' FROM user;
SELECT role as '角色', COUNT(*) as '数量' FROM user GROUP BY role;

SELECT '用户数据恢复完成！可以使用以下账号登录:' as message;
SELECT '管理员账号: root/root123 或 admin/admin123' as admin_accounts;
SELECT '教师账号: teacher1/teacher123 (还有teacher2, teacher3)' as teacher_accounts;
SELECT '学生账号: student1/student123 (还有student2-student5)' as student_accounts;