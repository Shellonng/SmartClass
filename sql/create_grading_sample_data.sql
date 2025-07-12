-- Create sample assignment data for intelligent grading
-- This script adds assignments, questions, student submissions, and answers

-- Add more questions for testing
INSERT INTO `question` (`title`, `question_type`, `difficulty`, `correct_answer`, `explanation`, `knowledge_point`, `course_id`, `chapter_id`, `created_by`) 
VALUES 
('什么是计算机组成的基本部件？', 'short', 3, '计算机的基本组成部件包括：输入设备、输出设备、CPU（中央处理器）、内存（主存）和辅助存储设备。', '计算机的基本组成遵循冯·诺依曼架构', '计算机组成原理', 9, 11, 6),
('计算机中的ALU是指什么？', 'single', 2, 'A', 'ALU是算术逻辑单元的缩写，是CPU的核心组成部分之一', '计算机组成原理', 9, 11, 6),
('二进制数1101等于多少十进制数？', 'blank', 2, '13', '二进制1101 = 1*2^3 + 1*2^2 + 0*2^1 + 1*2^0 = 8 + 4 + 0 + 1 = 13', '数字逻辑', 9, 12, 6),
('哪些是计算机的输入设备？', 'multiple', 2, 'A,B,D', '选项C（打印机）是输出设备', '计算机外设', 9, 11, 6);

-- Add options for single choice and multiple choice questions
INSERT INTO `question_option` (`question_id`, `option_label`, `option_text`)
SELECT id, 'A', '算术逻辑单元' FROM `question` WHERE `title` = '计算机中的ALU是指什么？';
INSERT INTO `question_option` (`question_id`, `option_label`, `option_text`)
SELECT id, 'B', '自动加载单元' FROM `question` WHERE `title` = '计算机中的ALU是指什么？';
INSERT INTO `question_option` (`question_id`, `option_label`, `option_text`)
SELECT id, 'C', '地址定位单元' FROM `question` WHERE `title` = '计算机中的ALU是指什么？';
INSERT INTO `question_option` (`question_id`, `option_label`, `option_text`)
SELECT id, 'D', '辅助逻辑单元' FROM `question` WHERE `title` = '计算机中的ALU是指什么？';

INSERT INTO `question_option` (`question_id`, `option_label`, `option_text`)
SELECT id, 'A', '键盘' FROM `question` WHERE `title` = '哪些是计算机的输入设备？';
INSERT INTO `question_option` (`question_id`, `option_label`, `option_text`)
SELECT id, 'B', '鼠标' FROM `question` WHERE `title` = '哪些是计算机的输入设备？';
INSERT INTO `question_option` (`question_id`, `option_label`, `option_text`)
SELECT id, 'C', '打印机' FROM `question` WHERE `title` = '哪些是计算机的输入设备？';
INSERT INTO `question_option` (`question_id`, `option_label`, `option_text`)
SELECT id, 'D', '扫描仪' FROM `question` WHERE `title` = '哪些是计算机的输入设备？';

-- Create a grading assignment 
INSERT INTO `assignment` (`title`, `course_id`, `user_id`, `type`, `description`, `start_time`, `end_time`, `create_time`, `status`, `mode`, `total_score`)
VALUES ('计算机组成原理基础测验', 9, 6, 'homework', '这是一次关于计算机组成原理基础知识的测验，请认真作答。', 
        '2025-07-01 08:00:00', '2025-07-20 23:59:59', NOW(), 1, 'question', 100);

-- Add questions to this assignment
INSERT INTO `assignment_question` (`assignment_id`, `question_id`, `score`, `sequence`)
SELECT 
    (SELECT id FROM assignment WHERE title = '计算机组成原理基础测验'), 
    id, 
    CASE 
        WHEN question_type = 'short' THEN 40
        WHEN question_type = 'multiple' THEN 25
        ELSE 15
    END, 
    CASE 
        WHEN title = '什么是计算机组成的基本部件？' THEN 1
        WHEN title = '计算机中的ALU是指什么？' THEN 2
        WHEN title = '二进制数1101等于多少十进制数？' THEN 3
        WHEN title = '哪些是计算机的输入设备？' THEN 4
    END
FROM question 
WHERE title IN ('什么是计算机组成的基本部件？', '计算机中的ALU是指什么？', '二进制数1101等于多少十进制数？', '哪些是计算机的输入设备？');

-- Create student submissions
INSERT INTO `assignment_submission` (`assignment_id`, `student_id`, `status`, `submit_time`, `create_time`)
VALUES (
    (SELECT id FROM assignment WHERE title = '计算机组成原理基础测验'),
    12, 1, NOW(), NOW()
);

-- Add student answers to the submissions
INSERT INTO `assignment_submission_answer` (`submission_id`, `question_id`, `student_answer`, `is_correct`, `score`, `comment`, `create_time`, `update_time`)
VALUES
(
    (SELECT MAX(id) FROM assignment_submission WHERE student_id = 12),
    (SELECT id FROM question WHERE title = '什么是计算机组成的基本部件？'),
    '计算机组成部件主要包括输入设备、中央处理器（CPU）、内存、输出设备和存储设备。它们共同工作来处理和存储数据。',
    NULL, NULL, NULL, NOW(), NOW()
),
(
    (SELECT MAX(id) FROM assignment_submission WHERE student_id = 12),
    (SELECT id FROM question WHERE title = '计算机中的ALU是指什么？'),
    'A',
    NULL, NULL, NULL, NOW(), NOW()
),
(
    (SELECT MAX(id) FROM assignment_submission WHERE student_id = 12),
    (SELECT id FROM question WHERE title = '二进制数1101等于多少十进制数？'),
    '13',
    NULL, NULL, NULL, NOW(), NOW()
),
(
    (SELECT MAX(id) FROM assignment_submission WHERE student_id = 12),
    (SELECT id FROM question WHERE title = '哪些是计算机的输入设备？'),
    'A,B,C',
    NULL, NULL, NULL, NOW(), NOW()
);

-- Create another student's submission with different answers
INSERT INTO `assignment_submission` (`assignment_id`, `student_id`, `status`, `submit_time`, `create_time`)
VALUES (
    (SELECT id FROM assignment WHERE title = '计算机组成原理基础测验'),
    12, 1, NOW(), NOW()
);

-- Add second student's answers
INSERT INTO `assignment_submission_answer` (`submission_id`, `question_id`, `student_answer`, `is_correct`, `score`, `comment`, `create_time`, `update_time`)
VALUES
(
    (SELECT MAX(id) FROM assignment_submission WHERE student_id = 12),
    (SELECT id FROM question WHERE title = '什么是计算机组成的基本部件？'),
    '计算机组成部件包括CPU、内存、硬盘',
    NULL, NULL, NULL, NOW(), NOW()
),
(
    (SELECT MAX(id) FROM assignment_submission WHERE student_id = 12),
    (SELECT id FROM question WHERE title = '计算机中的ALU是指什么？'),
    'B',
    NULL, NULL, NULL, NOW(), NOW()
),
(
    (SELECT MAX(id) FROM assignment_submission WHERE student_id = 12),
    (SELECT id FROM question WHERE title = '二进制数1101等于多少十进制数？'),
    '14',
    NULL, NULL, NULL, NOW(), NOW()
),
(
    (SELECT MAX(id) FROM assignment_submission WHERE student_id = 12),
    (SELECT id FROM question WHERE title = '哪些是计算机的输入设备？'),
    'A,D',
    NULL, NULL, NULL, NOW(), NOW()
); 