-- 为数据分析页面生成模拟数据

-- 插入更多的题目答案数据
-- 注意：运行前请先确认数据库中已有相应的题目、学生和作业提交记录

-- 为题目ID=22（CPU功能）的简答题添加更多回答
INSERT INTO assignment_submission_answer (submission_id, question_id, student_answer, is_correct, score, comment, create_time, update_time)
VALUES 
(42, 22, 'CPU是计算机的核心处理器，负责执行指令和控制计算机操作。', 1, 38, '基本描述正确', NOW(), NOW()),
(42, 22, 'CPU的主要功能是执行指令集、控制数据流和处理运算。', 1, 40, '回答全面且准确', NOW(), NOW()),
(42, 22, '中央处理器负责执行程序中的指令并进行数据处理。', 1, 36, '描述较简单', NOW(), NOW()),
(42, 22, 'CPU集成了控制单元和运算单元，是电子计算机的运算核心。', 1, 39, '回答较全面', NOW(), NOW()),
(42, 22, '电脑的大脑，处理所有指令。', 0, 25, '描述过于简单', NOW(), NOW()),
(42, 22, 'CPU主要处理系统中的各种运算，同时也控制其他硬件的工作。', 1, 37, '回答基本正确', NOW(), NOW()),
(42, 22, 'CPU是计算机最重要的硬件，没有它计算机无法工作。', 0, 20, '没有具体说明功能', NOW(), NOW()),
(42, 22, '中央处理器(CPU)是计算机的核心部件，负责执行程序中的指令以及处理数据。CPU包含控制单元、算术逻辑单元和寄存器等组件，共同工作完成复杂的计算任务。', 1, 40, '回答非常全面', NOW(), NOW()),
(42, 22, '执行计算机程序的指令，完成数据处理、程序控制等功能。', 1, 35, '回答正确但不够详细', NOW(), NOW()),
(42, 22, 'CPU通过取指令、译码、执行和写回等步骤完成程序的运行，是计算机的核心组件。', 1, 40, '回答准确且有深度', NOW(), NOW());

-- 为题目ID=23（输入设备）的多选题添加更多回答
INSERT INTO assignment_submission_answer (submission_id, question_id, student_answer, is_correct, score, comment, create_time, update_time)
VALUES 
(42, 23, 'A,B,D', 1, 30, '回答正确', NOW(), NOW()),
(42, 23, 'A,B', 0, 20, '缺少部分正确选项', NOW(), NOW()),
(42, 23, 'A,C', 0, 10, '包含错误选项', NOW(), NOW()),
(42, 23, 'A,B,C,D', 0, 15, '包含错误选项', NOW(), NOW()),
(42, 23, 'A,D', 0, 20, '缺少部分正确选项', NOW(), NOW()),
(42, 23, 'B,D', 0, 20, '缺少部分正确选项', NOW(), NOW()),
(42, 23, 'C,D', 0, 10, '包含错误选项', NOW(), NOW()),
(42, 23, 'A,B,D', 1, 30, '回答正确', NOW(), NOW()),
(42, 23, 'A,B,D', 1, 30, '回答正确', NOW(), NOW()),
(42, 23, 'A,B,D', 1, 30, '回答正确', NOW(), NOW());

-- 为题目ID=24（二进制转换）的填空题添加更多回答
INSERT INTO assignment_submission_answer (submission_id, question_id, student_answer, is_correct, score, comment, create_time, update_time)
VALUES 
(42, 24, '10', 1, 30, '回答正确', NOW(), NOW()),
(42, 24, '8', 0, 0, '计算错误', NOW(), NOW()),
(42, 24, '12', 0, 0, '计算错误', NOW(), NOW()),
(42, 24, '2', 0, 0, '计算错误', NOW(), NOW()),
(42, 24, '10', 1, 30, '回答正确', NOW(), NOW()),
(42, 24, '10', 1, 30, '回答正确', NOW(), NOW()),
(42, 24, '1010', 0, 10, '未完成进制转换', NOW(), NOW()),
(42, 24, '10', 1, 30, '回答正确', NOW(), NOW()),
(42, 24, '10', 1, 30, '回答正确', NOW(), NOW()),
(42, 24, '10', 1, 30, '回答正确', NOW(), NOW());

-- 为简答题添加更多题目和答案（计算机系统基础）
INSERT INTO question (title, question_type, difficulty, correct_answer, explanation, knowledge_point, course_id, chapter_id, created_by, create_time)
VALUES 
('请简要描述计算机系统的层次结构', 'short', 3, '计算机系统的层次结构从低到高包括硬件、机器语言层、操作系统层、汇编语言层、高级语言层和应用层。每一层都为上一层提供服务，隐藏下层的具体实现细节。', '这个问题考察学生对计算机系统层次化设计的理解。', '计算机系统结构', 9, 11, 6, NOW());

SET @new_question_id = LAST_INSERT_ID();

-- 为新添加的简答题生成答案
INSERT INTO assignment_submission_answer (submission_id, question_id, student_answer, is_correct, score, comment, create_time, update_time)
VALUES 
(42, @new_question_id, '计算机系统从底层到高层依次是硬件、操作系统、应用软件。', 0, 15, '回答过于简单', NOW(), NOW()),
(42, @new_question_id, '计算机系统层次结构包括硬件层、操作系统层、应用层。硬件层负责基本运算，操作系统层管理资源，应用层提供用户接口。', 0, 25, '回答基本正确但不够详细', NOW(), NOW()),
(42, @new_question_id, '计算机系统分为硬件层次和软件层次，硬件包括CPU、内存、I/O设备等，软件包括系统软件和应用软件。', 0, 20, '描述不够系统', NOW(), NOW()),
(42, @new_question_id, '计算机系统层次结构从低到高包括：硬件层、机器语言层、操作系统层、汇编语言层、高级语言层、应用层。每层都为上层提供服务并屏蔽下层细节。', 1, 40, '回答全面准确', NOW(), NOW()),
(42, @new_question_id, '计算机系统的层次结构主要包括硬件层、操作系统层、编程语言层和应用层。', 0, 30, '回答基本正确但不够详细', NOW(), NOW());

-- 为单选题添加更多题目和答案（数据表示）
INSERT INTO question (title, question_type, difficulty, correct_answer, explanation, knowledge_point, course_id, chapter_id, created_by, create_time)
VALUES 
('在计算机中，整数的二进制补码表示法主要用于解决什么问题？', 'single', 2, 'B', '补码表示法使得负数可以用二进制表示，并且加法和减法可以用统一的电路实现。', '数据表示', 9, 12, 6, NOW());

SET @new_single_id = LAST_INSERT_ID();

-- 为单选题添加选项
INSERT INTO question_option (question_id, option_label, option_text)
VALUES 
(@new_single_id, 'A', '提高数据存储效率'),
(@new_single_id, 'B', '实现负数表示和简化加减法电路设计'),
(@new_single_id, 'C', '加快数据处理速度'),
(@new_single_id, 'D', '减少存储空间占用');

-- 为单选题生成答案
INSERT INTO assignment_submission_answer (submission_id, question_id, student_answer, is_correct, score, comment, create_time, update_time)
VALUES 
(42, @new_single_id, 'A', 0, 0, '选择错误', NOW(), NOW()),
(42, @new_single_id, 'B', 1, 20, '回答正确', NOW(), NOW()),
(42, @new_single_id, 'C', 0, 0, '选择错误', NOW(), NOW()),
(42, @new_single_id, 'D', 0, 0, '选择错误', NOW(), NOW()),
(42, @new_single_id, 'B', 1, 20, '回答正确', NOW(), NOW()),
(42, @new_single_id, 'B', 1, 20, '回答正确', NOW(), NOW()),
(42, @new_single_id, 'A', 0, 0, '选择错误', NOW(), NOW()),
(42, @new_single_id, 'B', 1, 20, '回答正确', NOW(), NOW());

-- 为多选题添加更多题目和答案（存储器）
INSERT INTO question (title, question_type, difficulty, correct_answer, explanation, knowledge_point, course_id, chapter_id, created_by, create_time)
VALUES 
('以下哪些属于计算机的易失性存储器？', 'multiple', 2, 'A,C', '易失性存储器在断电后数据会丢失，包括RAM和Cache等。', '存储器', 9, 15, 6, NOW());

SET @new_multiple_id = LAST_INSERT_ID();

-- 为多选题添加选项
INSERT INTO question_option (question_id, option_label, option_text)
VALUES 
(@new_multiple_id, 'A', '内存(RAM)'),
(@new_multiple_id, 'B', '硬盘'),
(@new_multiple_id, 'C', '缓存(Cache)'),
(@new_multiple_id, 'D', '固态硬盘(SSD)');

-- 为多选题生成答案
INSERT INTO assignment_submission_answer (submission_id, question_id, student_answer, is_correct, score, comment, create_time, update_time)
VALUES 
(42, @new_multiple_id, 'A', 0, 15, '不完整', NOW(), NOW()),
(42, @new_multiple_id, 'A,B', 0, 10, '包含错误选项', NOW(), NOW()),
(42, @new_multiple_id, 'A,C', 1, 30, '回答正确', NOW(), NOW()),
(42, @new_multiple_id, 'A,B,C,D', 0, 5, '包含错误选项', NOW(), NOW()),
(42, @new_multiple_id, 'A,C', 1, 30, '回答正确', NOW(), NOW()),
(42, @new_multiple_id, 'A,C,D', 0, 15, '包含错误选项', NOW(), NOW()),
(42, @new_multiple_id, 'B,D', 0, 0, '完全错误', NOW(), NOW()),
(42, @new_multiple_id, 'A,C', 1, 30, '回答正确', NOW(), NOW());

-- 更新assignment_question表，关联新题目到作业
INSERT INTO assignment_question (assignment_id, question_id, score, sequence)
VALUES 
(29, @new_question_id, 40, 4),
(29, @new_single_id, 20, 5),
(29, @new_multiple_id, 30, 6); 