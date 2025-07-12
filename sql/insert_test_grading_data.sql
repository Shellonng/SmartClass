-- 插入测试数据用于智能批改功能
-- 确保作业存在
INSERT INTO `assignment` (`title`, `course_id`, `user_id`, `type`, `description`, `start_time`, `end_time`, `create_time`, `status`, `mode`, `total_score`)
SELECT '智能批改测试作业', 9, 6, 'homework', '这是一个用于测试智能批改功能的作业', 
       DATE_SUB(NOW(), INTERVAL 7 DAY), DATE_ADD(NOW(), INTERVAL 7 DAY), NOW(), 1, 'question', 100
FROM dual
WHERE NOT EXISTS (SELECT 1 FROM `assignment` WHERE `title` = '智能批改测试作业');

-- 获取刚插入的作业ID
SET @assignment_id = LAST_INSERT_ID();
SELECT @assignment_id := id FROM `assignment` WHERE `title` = '智能批改测试作业' LIMIT 1;

-- 确保有题目
INSERT INTO `question` (`title`, `question_type`, `difficulty`, `correct_answer`, `explanation`, `knowledge_point`, `course_id`, `chapter_id`, `created_by`) 
VALUES 
('计算机的中央处理器(CPU)的主要功能是什么？', 'short', 3, 'CPU是计算机的核心，主要功能是执行指令、处理数据和控制系统运行。它负责算术运算、逻辑运算、数据传送和程序控制等基本操作。', '中央处理器是计算机的大脑，负责执行指令和处理数据。', '计算机组成原理', 9, 11, 6),
('以下哪项是输入设备？', 'multiple', 2, 'A,B,D', '键盘、鼠标和扫描仪都是输入设备，而打印机是输出设备。', '计算机外设', 9, 11, 6),
('二进制数1010等于十进制数多少？', 'blank', 2, '10', '二进制1010 = 1*2^3 + 0*2^2 + 1*2^1 + 0*2^0 = 8 + 0 + 2 + 0 = 10', '数字逻辑', 9, 12, 6);

-- 添加选项
INSERT INTO `question_option` (`question_id`, `option_label`, `option_text`)
SELECT id, 'A', '键盘' FROM `question` WHERE `title` = '以下哪项是输入设备？';
INSERT INTO `question_option` (`question_id`, `option_label`, `option_text`)
SELECT id, 'B', '鼠标' FROM `question` WHERE `title` = '以下哪项是输入设备？';
INSERT INTO `question_option` (`question_id`, `option_label`, `option_text`)
SELECT id, 'C', '打印机' FROM `question` WHERE `title` = '以下哪项是输入设备？';
INSERT INTO `question_option` (`question_id`, `option_label`, `option_text`)
SELECT id, 'D', '扫描仪' FROM `question` WHERE `title` = '以下哪项是输入设备？';

-- 将题目添加到作业中
INSERT INTO `assignment_question` (`assignment_id`, `question_id`, `score`, `sequence`)
SELECT @assignment_id, id, 40, 1 FROM `question` WHERE `title` = '计算机的中央处理器(CPU)的主要功能是什么？';
INSERT INTO `assignment_question` (`assignment_id`, `question_id`, `score`, `sequence`)
SELECT @assignment_id, id, 30, 2 FROM `question` WHERE `title` = '以下哪项是输入设备？';
INSERT INTO `assignment_question` (`assignment_id`, `question_id`, `score`, `sequence`)
SELECT @assignment_id, id, 30, 3 FROM `question` WHERE `title` = '二进制数1010等于十进制数多少？';

-- 创建学生提交记录（已提交未批改状态）
INSERT INTO `assignment_submission` (`assignment_id`, `student_id`, `status`, `submit_time`, `create_time`)
VALUES 
(@assignment_id, 12, 1, NOW(), NOW()),
(@assignment_id, 12, 1, DATE_SUB(NOW(), INTERVAL 1 DAY), DATE_SUB(NOW(), INTERVAL 1 DAY)),
(@assignment_id, 12, 1, DATE_SUB(NOW(), INTERVAL 2 DAY), DATE_SUB(NOW(), INTERVAL 2 DAY));

-- 获取提交ID
SET @submission_id1 = LAST_INSERT_ID();
SET @submission_id2 = @submission_id1 - 1;
SET @submission_id3 = @submission_id1 - 2;

-- 添加学生答案
-- 第一个提交
INSERT INTO `assignment_submission_answer` (`submission_id`, `question_id`, `student_answer`, `create_time`, `update_time`)
SELECT @submission_id1, id, 'CPU是计算机的核心部件，负责执行指令和处理数据，控制计算机的运行。它执行算术运算和逻辑运算，并管理数据的流动。', NOW(), NOW()
FROM `question` WHERE `title` = '计算机的中央处理器(CPU)的主要功能是什么？';

INSERT INTO `assignment_submission_answer` (`submission_id`, `question_id`, `student_answer`, `create_time`, `update_time`)
SELECT @submission_id1, id, 'A,B,D', NOW(), NOW()
FROM `question` WHERE `title` = '以下哪项是输入设备？';

INSERT INTO `assignment_submission_answer` (`submission_id`, `question_id`, `student_answer`, `create_time`, `update_time`)
SELECT @submission_id1, id, '10', NOW(), NOW()
FROM `question` WHERE `title` = '二进制数1010等于十进制数多少？';

-- 第二个提交
INSERT INTO `assignment_submission_answer` (`submission_id`, `question_id`, `student_answer`, `create_time`, `update_time`)
SELECT @submission_id2, id, 'CPU负责执行计算机程序的指令，进行数据处理和运算。', NOW(), NOW()
FROM `question` WHERE `title` = '计算机的中央处理器(CPU)的主要功能是什么？';

INSERT INTO `assignment_submission_answer` (`submission_id`, `question_id`, `student_answer`, `create_time`, `update_time`)
SELECT @submission_id2, id, 'A,C,D', NOW(), NOW()
FROM `question` WHERE `title` = '以下哪项是输入设备？';

INSERT INTO `assignment_submission_answer` (`submission_id`, `question_id`, `student_answer`, `create_time`, `update_time`)
SELECT @submission_id2, id, '8', NOW(), NOW()
FROM `question` WHERE `title` = '二进制数1010等于十进制数多少？';

-- 第三个提交
INSERT INTO `assignment_submission_answer` (`submission_id`, `question_id`, `student_answer`, `create_time`, `update_time`)
SELECT @submission_id3, id, 'CPU是计算机的大脑，主要功能是运行程序、处理数据和协调各部件工作。', NOW(), NOW()
FROM `question` WHERE `title` = '计算机的中央处理器(CPU)的主要功能是什么？';

INSERT INTO `assignment_submission_answer` (`submission_id`, `question_id`, `student_answer`, `create_time`, `update_time`)
SELECT @submission_id3, id, 'A,B', NOW(), NOW()
FROM `question` WHERE `title` = '以下哪项是输入设备？';

INSERT INTO `assignment_submission_answer` (`submission_id`, `question_id`, `student_answer`, `create_time`, `update_time`)
SELECT @submission_id3, id, '10', NOW(), NOW()
FROM `question` WHERE `title` = '二进制数1010等于十进制数多少？';

-- 添加一个已批改的提交记录，用于测试查看批改结果
INSERT INTO `assignment_submission` (`assignment_id`, `student_id`, `status`, `score`, `feedback`, `submit_time`, `grade_time`, `graded_by`, `create_time`)
VALUES 
(@assignment_id, 12, 2, 85, '整体表现良好，对计算机基础知识掌握较好，但在多选题上有小错误。', DATE_SUB(NOW(), INTERVAL 3 DAY), NOW(), 6, DATE_SUB(NOW(), INTERVAL 3 DAY));

SET @submission_id4 = LAST_INSERT_ID();

-- 添加已批改的答案
INSERT INTO `assignment_submission_answer` (`submission_id`, `question_id`, `student_answer`, `is_correct`, `score`, `comment`, `create_time`, `update_time`)
SELECT @submission_id4, id, 'CPU是计算机的核心，负责执行指令、处理数据和控制计算机运行。', 1, 40, '回答全面准确，包含了CPU的核心功能。', NOW(), NOW()
FROM `question` WHERE `title` = '计算机的中央处理器(CPU)的主要功能是什么？';

INSERT INTO `assignment_submission_answer` (`submission_id`, `question_id`, `student_answer`, `is_correct`, `score`, `comment`, `create_time`, `update_time`)
SELECT @submission_id4, id, 'A,B,C', 0, 20, '部分正确，但C选项(打印机)是输出设备而非输入设备。', NOW(), NOW()
FROM `question` WHERE `title` = '以下哪项是输入设备？';

INSERT INTO `assignment_submission_answer` (`submission_id`, `question_id`, `student_answer`, `is_correct`, `score`, `comment`, `create_time`, `update_time`)
SELECT @submission_id4, id, '10', 1, 25, '答案正确。', NOW(), NOW()
FROM `question` WHERE `title` = '二进制数1010等于十进制数多少？'; 