-- =============================================
-- 教育平台数据库初始化脚本
-- 版本: 1.0
-- 创建日期: 2024
-- 描述: 包含完整的表结构创建和初始化数据
-- =============================================

-- 创建数据库
CREATE DATABASE IF NOT EXISTS education_platform 
CHARACTER SET utf8mb4 
COLLATE utf8mb4_unicode_ci;

USE education_platform;

-- =============================================
-- 表结构创建
-- =============================================

-- 用户表
CREATE TABLE user (
    id BIGINT PRIMARY KEY AUTO_INCREMENT COMMENT '用户ID',
    username VARCHAR(50) NOT NULL UNIQUE COMMENT '用户名',
    password VARCHAR(255) NOT NULL COMMENT '密码',
    email VARCHAR(100) UNIQUE COMMENT '邮箱',
    phone VARCHAR(20) COMMENT '手机号',
    avatar VARCHAR(255) COMMENT '头像URL',
    status TINYINT DEFAULT 1 COMMENT '状态: 0-禁用, 1-启用',
    role ENUM('ADMIN', 'TEACHER', 'STUDENT') NOT NULL COMMENT '角色',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    INDEX idx_username (username),
    INDEX idx_email (email),
    INDEX idx_role (role)
) COMMENT='用户基础信息表';

-- 教师表
CREATE TABLE teacher (
    id BIGINT PRIMARY KEY AUTO_INCREMENT COMMENT '教师ID',
    user_id BIGINT NOT NULL UNIQUE COMMENT '关联用户ID',
    employee_id VARCHAR(20) UNIQUE COMMENT '工号',
    department VARCHAR(100) COMMENT '所属部门',
    title VARCHAR(50) COMMENT '职称',
    bio TEXT COMMENT '个人简介',
    specialization VARCHAR(200) COMMENT '专业领域',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    FOREIGN KEY (user_id) REFERENCES user(id) ON DELETE CASCADE,
    INDEX idx_employee_id (employee_id),
    INDEX idx_department (department)
) COMMENT='教师信息表';

-- 学生表
CREATE TABLE student (
    id BIGINT PRIMARY KEY AUTO_INCREMENT COMMENT '学生ID',
    user_id BIGINT NOT NULL UNIQUE COMMENT '关联用户ID',
    student_id VARCHAR(20) UNIQUE COMMENT '学号',
    class_id BIGINT COMMENT '班级ID',
    grade VARCHAR(10) COMMENT '年级',
    major VARCHAR(100) COMMENT '专业',
    enrollment_year YEAR COMMENT '入学年份',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    FOREIGN KEY (user_id) REFERENCES user(id) ON DELETE CASCADE,
    INDEX idx_student_id (student_id),
    INDEX idx_class_id (class_id),
    INDEX idx_grade (grade)
) COMMENT='学生信息表';

-- 班级表
CREATE TABLE class (
    id BIGINT PRIMARY KEY AUTO_INCREMENT COMMENT '班级ID',
    name VARCHAR(100) NOT NULL COMMENT '班级名称',
    description TEXT COMMENT '班级描述',
    teacher_id BIGINT COMMENT '班主任ID',
    grade VARCHAR(10) COMMENT '年级',
    major VARCHAR(100) COMMENT '专业',
    student_count INT DEFAULT 0 COMMENT '学生人数',
    status TINYINT DEFAULT 1 COMMENT '状态: 0-停用, 1-启用',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    FOREIGN KEY (teacher_id) REFERENCES teacher(id) ON DELETE SET NULL,
    INDEX idx_name (name),
    INDEX idx_teacher_id (teacher_id),
    INDEX idx_grade (grade)
) COMMENT='班级信息表';

-- 课程表
CREATE TABLE course (
    id BIGINT PRIMARY KEY AUTO_INCREMENT COMMENT '课程ID',
    name VARCHAR(100) NOT NULL COMMENT '课程名称',
    code VARCHAR(20) UNIQUE COMMENT '课程代码',
    description TEXT COMMENT '课程描述',
    teacher_id BIGINT NOT NULL COMMENT '授课教师ID',
    credit DECIMAL(3,1) COMMENT '学分',
    hours INT COMMENT '学时',
    category VARCHAR(50) COMMENT '课程类别',
    difficulty ENUM('BEGINNER', 'INTERMEDIATE', 'ADVANCED') DEFAULT 'BEGINNER' COMMENT '难度等级',
    status TINYINT DEFAULT 1 COMMENT '状态: 0-停用, 1-启用',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    FOREIGN KEY (teacher_id) REFERENCES teacher(id) ON DELETE CASCADE,
    INDEX idx_name (name),
    INDEX idx_code (code),
    INDEX idx_teacher_id (teacher_id),
    INDEX idx_category (category)
) COMMENT='课程信息表';

-- 任务表
CREATE TABLE task (
    id BIGINT PRIMARY KEY AUTO_INCREMENT COMMENT '任务ID',
    title VARCHAR(200) NOT NULL COMMENT '任务标题',
    description TEXT COMMENT '任务描述',
    course_id BIGINT NOT NULL COMMENT '所属课程ID',
    teacher_id BIGINT NOT NULL COMMENT '发布教师ID',
    type ENUM('HOMEWORK', 'EXAM', 'PROJECT', 'QUIZ') NOT NULL COMMENT '任务类型',
    difficulty ENUM('EASY', 'MEDIUM', 'HARD') DEFAULT 'MEDIUM' COMMENT '难度等级',
    max_score DECIMAL(5,2) DEFAULT 100.00 COMMENT '满分',
    start_time TIMESTAMP COMMENT '开始时间',
    end_time TIMESTAMP COMMENT '截止时间',
    allow_late TINYINT DEFAULT 0 COMMENT '是否允许迟交',
    late_penalty DECIMAL(3,2) DEFAULT 0.00 COMMENT '迟交扣分比例',
    status TINYINT DEFAULT 1 COMMENT '状态: 0-草稿, 1-发布, 2-结束',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    FOREIGN KEY (course_id) REFERENCES course(id) ON DELETE CASCADE,
    FOREIGN KEY (teacher_id) REFERENCES teacher(id) ON DELETE CASCADE,
    INDEX idx_course_id (course_id),
    INDEX idx_teacher_id (teacher_id),
    INDEX idx_type (type),
    INDEX idx_status (status),
    INDEX idx_end_time (end_time)
) COMMENT='任务信息表';

-- 提交表
CREATE TABLE submission (
    id BIGINT PRIMARY KEY AUTO_INCREMENT COMMENT '提交ID',
    task_id BIGINT NOT NULL COMMENT '任务ID',
    student_id BIGINT NOT NULL COMMENT '学生ID',
    content TEXT COMMENT '提交内容',
    file_url VARCHAR(500) COMMENT '附件URL',
    submit_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '提交时间',
    is_late TINYINT DEFAULT 0 COMMENT '是否迟交',
    status ENUM('SUBMITTED', 'GRADED', 'RETURNED') DEFAULT 'SUBMITTED' COMMENT '状态',
    version INT DEFAULT 1 COMMENT '版本号',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    FOREIGN KEY (task_id) REFERENCES task(id) ON DELETE CASCADE,
    FOREIGN KEY (student_id) REFERENCES student(id) ON DELETE CASCADE,
    UNIQUE KEY uk_task_student (task_id, student_id),
    INDEX idx_task_id (task_id),
    INDEX idx_student_id (student_id),
    INDEX idx_submit_time (submit_time)
) COMMENT='作业提交表';

-- 成绩表
CREATE TABLE grade (
    id BIGINT PRIMARY KEY AUTO_INCREMENT COMMENT '成绩ID',
    submission_id BIGINT NOT NULL UNIQUE COMMENT '提交ID',
    task_id BIGINT NOT NULL COMMENT '任务ID',
    student_id BIGINT NOT NULL COMMENT '学生ID',
    teacher_id BIGINT NOT NULL COMMENT '评分教师ID',
    score DECIMAL(5,2) COMMENT '得分',
    feedback TEXT COMMENT '评语',
    graded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '评分时间',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    FOREIGN KEY (submission_id) REFERENCES submission(id) ON DELETE CASCADE,
    FOREIGN KEY (task_id) REFERENCES task(id) ON DELETE CASCADE,
    FOREIGN KEY (student_id) REFERENCES student(id) ON DELETE CASCADE,
    FOREIGN KEY (teacher_id) REFERENCES teacher(id) ON DELETE CASCADE,
    INDEX idx_task_id (task_id),
    INDEX idx_student_id (student_id),
    INDEX idx_teacher_id (teacher_id)
) COMMENT='成绩评定表';

-- 资源表
CREATE TABLE resource (
    id BIGINT PRIMARY KEY AUTO_INCREMENT COMMENT '资源ID',
    title VARCHAR(200) NOT NULL COMMENT '资源标题',
    description TEXT COMMENT '资源描述',
    type ENUM('VIDEO', 'DOCUMENT', 'IMAGE', 'AUDIO', 'LINK', 'OTHER') NOT NULL COMMENT '资源类型',
    url VARCHAR(500) NOT NULL COMMENT '资源URL',
    file_size BIGINT COMMENT '文件大小(字节)',
    course_id BIGINT COMMENT '关联课程ID',
    teacher_id BIGINT NOT NULL COMMENT '上传教师ID',
    download_count INT DEFAULT 0 COMMENT '下载次数',
    status TINYINT DEFAULT 1 COMMENT '状态: 0-禁用, 1-启用',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    FOREIGN KEY (course_id) REFERENCES course(id) ON DELETE SET NULL,
    FOREIGN KEY (teacher_id) REFERENCES teacher(id) ON DELETE CASCADE,
    INDEX idx_course_id (course_id),
    INDEX idx_teacher_id (teacher_id),
    INDEX idx_type (type)
) COMMENT='教学资源表';

-- 知识图谱表
CREATE TABLE knowledge_graph (
    id BIGINT PRIMARY KEY AUTO_INCREMENT COMMENT '知识点ID',
    name VARCHAR(100) NOT NULL COMMENT '知识点名称',
    description TEXT COMMENT '知识点描述',
    course_id BIGINT NOT NULL COMMENT '所属课程ID',
    parent_id BIGINT COMMENT '父知识点ID',
    level INT DEFAULT 1 COMMENT '层级',
    order_num INT DEFAULT 0 COMMENT '排序号',
    difficulty ENUM('EASY', 'MEDIUM', 'HARD') DEFAULT 'MEDIUM' COMMENT '难度',
    mastery_threshold DECIMAL(3,2) DEFAULT 0.80 COMMENT '掌握阈值',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    FOREIGN KEY (course_id) REFERENCES course(id) ON DELETE CASCADE,
    FOREIGN KEY (parent_id) REFERENCES knowledge_graph(id) ON DELETE SET NULL,
    INDEX idx_course_id (course_id),
    INDEX idx_parent_id (parent_id),
    INDEX idx_level (level)
) COMMENT='知识图谱表';

-- 题库表
CREATE TABLE question_bank (
    id BIGINT PRIMARY KEY AUTO_INCREMENT COMMENT '题目ID',
    title VARCHAR(500) NOT NULL COMMENT '题目标题',
    content TEXT NOT NULL COMMENT '题目内容',
    type ENUM('SINGLE_CHOICE', 'MULTIPLE_CHOICE', 'TRUE_FALSE', 'FILL_BLANK', 'ESSAY') NOT NULL COMMENT '题目类型',
    options JSON COMMENT '选项(JSON格式)',
    answer TEXT NOT NULL COMMENT '正确答案',
    explanation TEXT COMMENT '答案解析',
    difficulty ENUM('EASY', 'MEDIUM', 'HARD') DEFAULT 'MEDIUM' COMMENT '难度等级',
    knowledge_point_id BIGINT COMMENT '关联知识点ID',
    course_id BIGINT NOT NULL COMMENT '所属课程ID',
    teacher_id BIGINT NOT NULL COMMENT '创建教师ID',
    usage_count INT DEFAULT 0 COMMENT '使用次数',
    correct_rate DECIMAL(5,4) DEFAULT 0.0000 COMMENT '正确率',
    status TINYINT DEFAULT 1 COMMENT '状态: 0-禁用, 1-启用',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    FOREIGN KEY (knowledge_point_id) REFERENCES knowledge_graph(id) ON DELETE SET NULL,
    FOREIGN KEY (course_id) REFERENCES course(id) ON DELETE CASCADE,
    FOREIGN KEY (teacher_id) REFERENCES teacher(id) ON DELETE CASCADE,
    INDEX idx_course_id (course_id),
    INDEX idx_teacher_id (teacher_id),
    INDEX idx_type (type),
    INDEX idx_difficulty (difficulty),
    INDEX idx_knowledge_point_id (knowledge_point_id)
) COMMENT='题库表';

-- AI功能表
CREATE TABLE ai_feature (
    id BIGINT PRIMARY KEY AUTO_INCREMENT COMMENT 'AI功能ID',
    user_id BIGINT NOT NULL COMMENT '用户ID',
    feature_type ENUM('CHAT', 'ANALYSIS', 'RECOMMENDATION', 'GRADING') NOT NULL COMMENT '功能类型',
    input_data JSON COMMENT '输入数据',
    output_data JSON COMMENT '输出数据',
    processing_time INT COMMENT '处理时间(毫秒)',
    status ENUM('PENDING', 'PROCESSING', 'COMPLETED', 'FAILED') DEFAULT 'PENDING' COMMENT '状态',
    error_message TEXT COMMENT '错误信息',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    FOREIGN KEY (user_id) REFERENCES user(id) ON DELETE CASCADE,
    INDEX idx_user_id (user_id),
    INDEX idx_feature_type (feature_type),
    INDEX idx_status (status),
    INDEX idx_created_at (created_at)
) COMMENT='AI功能使用记录表';

-- 学生选课表
CREATE TABLE student_course (
    id BIGINT PRIMARY KEY AUTO_INCREMENT COMMENT '选课ID',
    student_id BIGINT NOT NULL COMMENT '学生ID',
    course_id BIGINT NOT NULL COMMENT '课程ID',
    enrollment_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '选课时间',
    status ENUM('ENROLLED', 'COMPLETED', 'DROPPED') DEFAULT 'ENROLLED' COMMENT '选课状态',
    final_grade DECIMAL(5,2) COMMENT '最终成绩',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    FOREIGN KEY (student_id) REFERENCES student(id) ON DELETE CASCADE,
    FOREIGN KEY (course_id) REFERENCES course(id) ON DELETE CASCADE,
    UNIQUE KEY uk_student_course (student_id, course_id),
    INDEX idx_student_id (student_id),
    INDEX idx_course_id (course_id),
    INDEX idx_status (status)
) COMMENT='学生选课关系表';

-- =============================================
-- 索引优化
-- =============================================

-- 复合索引
CREATE INDEX idx_task_course_status ON task(course_id, status);
CREATE INDEX idx_submission_task_student ON submission(task_id, student_id);
CREATE INDEX idx_grade_task_student ON grade(task_id, student_id);
CREATE INDEX idx_resource_course_type ON resource(course_id, type);

-- 全文索引
CREATE FULLTEXT INDEX ft_course_name_desc ON course(name, description);
CREATE FULLTEXT INDEX ft_task_title_desc ON task(title, description);
CREATE FULLTEXT INDEX ft_question_content ON question_bank(title, content);

-- =============================================
-- 初始化数据
-- =============================================

-- 插入默认管理员用户
INSERT INTO user (username, password, email, role) VALUES 
('admin', 'admin123', 'admin@education.com', 'ADMIN');

-- 插入测试教师
INSERT INTO user (username, password, email, phone, role) VALUES 
('teacher1', 'teacher123', 'teacher1@education.com', '13800138001', 'TEACHER'),
('teacher2', 'teacher123', 'teacher2@education.com', '13800138002', 'TEACHER');

INSERT INTO teacher (user_id, employee_id, department, title, specialization) VALUES 
(2, 'T001', '计算机学院', '副教授', 'Java开发,数据库设计'),
(3, 'T002', '计算机学院', '讲师', 'Web前端,用户体验设计');

-- 插入测试学生
INSERT INTO user (username, password, email, phone, role) VALUES 
('student1', 'student123', 'student1@education.com', '13900139001', 'STUDENT'),
('student2', 'student123', 'student2@education.com', '13900139002', 'STUDENT'),
('student3', 'student123', 'student3@education.com', '13900139003', 'STUDENT');

-- 插入测试班级
INSERT INTO class (name, description, teacher_id, grade, major) VALUES 
('计算机2022-1班', '计算机科学与技术专业2022级1班', 1, '2022', '计算机科学与技术'),
('软件工程2022-1班', '软件工程专业2022级1班', 2, '2022', '软件工程');

INSERT INTO student (user_id, student_id, class_id, grade, major, enrollment_year) VALUES 
(4, 'S2022001', 1, '2022', '计算机科学与技术', 2022),
(5, 'S2022002', 1, '2022', '计算机科学与技术', 2022),
(6, 'S2022003', 2, '2022', '软件工程', 2022);

-- 插入测试课程
INSERT INTO course (name, code, description, teacher_id, credit, hours, category, difficulty) VALUES 
('Java程序设计', 'CS101', 'Java编程语言基础课程，包含面向对象编程思想', 1, 4.0, 64, '专业核心课', 'INTERMEDIATE'),
('Web前端开发', 'CS201', 'HTML、CSS、JavaScript及现代前端框架开发', 2, 3.0, 48, '专业选修课', 'BEGINNER'),
('数据库系统原理', 'CS301', '关系数据库理论与MySQL实践应用', 1, 3.5, 56, '专业核心课', 'ADVANCED');

-- 插入学生选课记录
INSERT INTO student_course (student_id, course_id) VALUES 
(1, 1), (1, 2), (1, 3),
(2, 1), (2, 2),
(3, 2), (3, 3);

-- 插入测试任务
INSERT INTO task (title, description, course_id, teacher_id, type, difficulty, max_score, start_time, end_time) VALUES 
('Java基础语法练习', '完成Java基本语法的编程练习，包括变量、循环、条件判断等', 1, 1, 'HOMEWORK', 'EASY', 100.00, NOW(), DATE_ADD(NOW(), INTERVAL 7 DAY)),
('前端页面设计', '使用HTML和CSS设计一个响应式网页', 2, 2, 'PROJECT', 'MEDIUM', 100.00, NOW(), DATE_ADD(NOW(), INTERVAL 14 DAY)),
('数据库设计期末考试', '数据库系统原理期末考试', 3, 1, 'EXAM', 'HARD', 100.00, DATE_ADD(NOW(), INTERVAL 30 DAY), DATE_ADD(NOW(), INTERVAL 30 DAY));

-- 插入知识图谱示例
INSERT INTO knowledge_graph (name, description, course_id, parent_id, level, order_num, difficulty) VALUES 
('Java基础', 'Java编程语言基础知识', 1, NULL, 1, 1, 'EASY'),
('变量与数据类型', 'Java中的变量声明和基本数据类型', 1, 1, 2, 1, 'EASY'),
('控制结构', '条件判断和循环控制', 1, 1, 2, 2, 'MEDIUM'),
('面向对象编程', 'Java面向对象编程思想', 1, NULL, 1, 2, 'MEDIUM'),
('类与对象', '类的定义和对象的创建', 1, 4, 2, 1, 'MEDIUM');

-- 插入题库示例
INSERT INTO question_bank (title, content, type, options, answer, explanation, difficulty, knowledge_point_id, course_id, teacher_id) VALUES 
('Java数据类型', '以下哪个是Java的基本数据类型？', 'SINGLE_CHOICE', 
 '["String", "int", "ArrayList", "Object"]', 
 'int', 'int是Java的基本数据类型，而String、ArrayList、Object都是引用类型', 
 'EASY', 2, 1, 1),
('循环语句', '在Java中，以下哪个循环语句可以保证至少执行一次？', 'SINGLE_CHOICE',
 '["for循环", "while循环", "do-while循环", "增强for循环"]',
 'do-while循环', 'do-while循环先执行循环体，再判断条件，所以至少执行一次',
 'MEDIUM', 3, 1, 1);

-- =============================================
-- 数据库初始化完成
-- =============================================

SELECT 'Database initialization completed successfully!' AS message;