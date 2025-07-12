-- 为学生ID=20250005(user_id=12)添加与课程9的关联数据
-- 创建日期: 2025-07-15

-- 检查是否已存在关联
SELECT * FROM course_student WHERE student_id = 20250005 AND course_id = 9;

-- 如果不存在，则插入关联数据
INSERT INTO course_student (course_id, student_id, enroll_time)
SELECT 9, 20250005, NOW()
FROM dual
WHERE NOT EXISTS (
    SELECT 1 FROM course_student WHERE student_id = 20250005 AND course_id = 9
);

-- 验证插入结果
SELECT * FROM course_student WHERE student_id = 20250005 AND course_id = 9;

-- 查看学生可见的作业
SELECT a.*, c.title AS course_name
FROM assignment a
JOIN course c ON a.course_id = c.id
JOIN course_student cs ON c.id = cs.course_id
WHERE cs.student_id = 20250005 AND a.status = 1; 