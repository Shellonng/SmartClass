-- 添加作业批改功能所需的测试数据

-- 1. 添加新的作业记录
INSERT INTO `assignment` 
(`title`, `course_id`, `user_id`, `type`, `description`, `start_time`, `end_time`, `status`, `mode`, `time_limit`, `total_score`, `reference_answer`)
VALUES 
('Java编程基础测验', 19, 6, 'homework', '本次作业主要测试Java基础知识，包括变量、循环和条件语句等', 
 DATE_SUB(NOW(), INTERVAL 10 DAY), DATE_ADD(NOW(), INTERVAL 5 DAY), 1, 'question', 60, 100, 
 '{"referenceAnswers":[{"questionId":1,"answer":"public class是Java中的访问修饰符，用于定义可以被其他类访问的类"},{"questionId":2,"answer":"B"},{"questionId":3,"answer":"T"}]}'),
 
('数据结构期中考试', 20, 6, 'exam', '数据结构与算法期中考试，考察基本数据结构的理解和应用', 
 DATE_SUB(NOW(), INTERVAL 5 DAY), DATE_ADD(NOW(), INTERVAL 2 DAY), 1, 'question', 120, 100, 
 '{"referenceAnswers":[{"questionId":4,"answer":"栈是一种后进先出(LIFO)的数据结构"},{"questionId":5,"answer":"A"},{"questionId":6,"answer":"F"}]}'),
 
('Python程序设计作业', 21, 6, 'homework', '请完成Python基础语法练习，包括列表、字典和函数的使用', 
 DATE_SUB(NOW(), INTERVAL 15 DAY), DATE_SUB(NOW(), INTERVAL 2 DAY), 1, 'question', 0, 100, 
 '{"referenceAnswers":[{"questionId":7,"answer":"Python中的列表是可变的，而元组是不可变的"},{"questionId":8,"answer":"C"},{"questionId":9,"answer":"T"}]}');

-- 获取新插入的作业ID
SET @java_assignment_id = LAST_INSERT_ID();
SET @ds_assignment_id = @java_assignment_id + 1;
SET @python_assignment_id = @java_assignment_id + 2;

-- 2. 创建一些题目用于关联到作业
INSERT INTO `question` 
(`title`, `question_type`, `difficulty`, `correct_answer`, `explanation`, `knowledge_point`, `course_id`, `chapter_id`, `created_by`)
VALUES
-- Java课程题目
('在Java中，public class关键字的作用是什么？', 'short', 3, 'public class是Java中的访问修饰符，用于定义可以被其他类访问的类', 
 'public是访问修饰符，表示该类可以被任何其他类访问；class关键字用于声明一个类', 'Java基础语法', 19, 11, 6),
 
('以下哪个不是Java的基本数据类型？', 'single', 2, 'B', 
 'String是引用类型，不是基本数据类型。Java的基本数据类型有byte、short、int、long、float、double、char和boolean', 
 'Java数据类型', 19, 11, 6),
 
('Java中的变量名区分大小写。（判断）', 'true_false', 1, 'T', 
 'Java是大小写敏感的语言，变量名User和user会被视为两个不同的变量', 'Java基础语法', 19, 11, 6),

-- 数据结构课程题目
('请简述栈(Stack)的特点和基本操作', 'short', 3, '栈是一种后进先出(LIFO)的数据结构，基本操作包括push(入栈)和pop(出栈)', 
 '栈是一种线性数据结构，遵循后进先出原则，主要操作有push、pop、peek等', '栈', 20, 11, 6),
 
('以下哪种数据结构适合用于实现广度优先搜索(BFS)？', 'single', 3, 'A', 
 '队列的先进先出特性适合实现广度优先搜索的层次遍历方式', '图算法', 20, 11, 6),
 
('在链表中，删除节点的时间复杂度始终是O(1)。（判断）', 'true_false', 3, 'F', 
 '在单链表中，删除节点通常需要O(n)时间复杂度，因为需要找到前驱节点；只有在已知前驱节点的情况下才是O(1)', 
 '链表', 20, 11, 6),

-- Python课程题目
('请解释Python中列表(List)和元组(Tuple)的区别', 'short', 2, 'Python中的列表是可变的，而元组是不可变的', 
 '列表可以在创建后修改，而元组一旦创建就不能更改。列表使用方括号[]，元组使用圆括号()', 'Python数据类型', 21, 11, 6),
 
('在Python中，以下哪个方法用于向列表末尾添加元素？', 'single', 1, 'C', 
 'append()方法用于在列表末尾添加一个元素；extend()用于合并列表；insert()用于在指定位置插入元素', 
 'Python列表操作', 21, 11, 6),
 
('Python中的字典是无序集合。（判断）', 'true_false', 2, 'T', 
 '在Python 3.7之前，字典确实是无序的集合。从Python 3.7开始，字典会保持插入顺序，但这是实现细节，不应依赖', 
 'Python字典', 21, 11, 6);

-- 获取新插入的题目ID
SET @java_q1_id = LAST_INSERT_ID();
SET @java_q2_id = @java_q1_id + 1;
SET @java_q3_id = @java_q1_id + 2;
SET @ds_q1_id = @java_q1_id + 3;
SET @ds_q2_id = @java_q1_id + 4;
SET @ds_q3_id = @java_q1_id + 5;
SET @python_q1_id = @java_q1_id + 6;
SET @python_q2_id = @java_q1_id + 7;
SET @python_q3_id = @java_q1_id + 8;

-- 为单选题添加选项
INSERT INTO `question_option` (`question_id`, `option_label`, `option_text`)
VALUES
-- Java单选题选项
(@java_q2_id, 'A', 'int'),
(@java_q2_id, 'B', 'String'),
(@java_q2_id, 'C', 'boolean'),
(@java_q2_id, 'D', 'char'),

-- 数据结构单选题选项
(@ds_q2_id, 'A', '队列(Queue)'),
(@ds_q2_id, 'B', '栈(Stack)'),
(@ds_q2_id, 'C', '哈希表(Hash Table)'),
(@ds_q2_id, 'D', '堆(Heap)'),

-- Python单选题选项
(@python_q2_id, 'A', 'insert()'),
(@python_q2_id, 'B', 'add()'),
(@python_q2_id, 'C', 'append()'),
(@python_q2_id, 'D', 'extend()');

-- 为判断题添加选项
INSERT INTO `question_option` (`question_id`, `option_label`, `option_text`)
VALUES
(@java_q3_id, 'T', '正确'),
(@java_q3_id, 'F', '错误'),
(@ds_q3_id, 'T', '正确'),
(@ds_q3_id, 'F', '错误'),
(@python_q3_id, 'T', '正确'),
(@python_q3_id, 'F', '错误');

-- 3. 关联题目到作业
INSERT INTO `assignment_question` (`assignment_id`, `question_id`, `score`, `sequence`)
VALUES
-- Java作业题目
(@java_assignment_id, @java_q1_id, 40, 1),
(@java_assignment_id, @java_q2_id, 30, 2),
(@java_assignment_id, @java_q3_id, 30, 3),

-- 数据结构作业题目
(@ds_assignment_id, @ds_q1_id, 40, 1),
(@ds_assignment_id, @ds_q2_id, 30, 2),
(@ds_assignment_id, @ds_q3_id, 30, 3),

-- Python作业题目
(@python_assignment_id, @python_q1_id, 40, 1),
(@python_assignment_id, @python_q2_id, 30, 2),
(@python_assignment_id, @python_q3_id, 30, 3);

-- 4. 添加学生提交记录
-- 假设学生ID为12（在user表中已存在的学生）

-- 学生提交Java作业
INSERT INTO `assignment_submission` 
(`assignment_id`, `student_id`, `status`, `submit_time`, `content`)
VALUES
(@java_assignment_id, 12, 1, DATE_SUB(NOW(), INTERVAL 8 DAY), '这是我的Java作业提交');

SET @java_submission_id = LAST_INSERT_ID();

-- 学生提交数据结构作业
INSERT INTO `assignment_submission` 
(`assignment_id`, `student_id`, `status`, `submit_time`, `content`)
VALUES
(@ds_assignment_id, 12, 1, DATE_SUB(NOW(), INTERVAL 4 DAY), '这是我的数据结构作业提交');

SET @ds_submission_id = LAST_INSERT_ID();

-- 学生提交Python作业（已批改状态）
INSERT INTO `assignment_submission` 
(`assignment_id`, `student_id`, `status`, `score`, `feedback`, `submit_time`, `grade_time`, `graded_by`, `content`)
VALUES
(@python_assignment_id, 12, 2, 85, '整体不错，但第一题解释不够全面', 
 DATE_SUB(NOW(), INTERVAL 10 DAY), DATE_SUB(NOW(), INTERVAL 1 DAY), 6, '这是我的Python作业提交');

SET @python_submission_id = LAST_INSERT_ID();

-- 5. 添加学生答题记录
INSERT INTO `assignment_submission_answer` 
(`submission_id`, `question_id`, `student_answer`, `is_correct`, `score`, `comment`)
VALUES
-- Java作业答题
(@java_submission_id, @java_q1_id, 'public class是Java中定义公共类的关键字，表示该类可以被其他包中的类访问', NULL, NULL, NULL),
(@java_submission_id, @java_q2_id, 'B', NULL, NULL, NULL),
(@java_submission_id, @java_q3_id, 'T', NULL, NULL, NULL),

-- 数据结构作业答题
(@ds_submission_id, @ds_q1_id, '栈是一种后进先出的数据结构，主要操作有push和pop', NULL, NULL, NULL),
(@ds_submission_id, @ds_q2_id, 'A', NULL, NULL, NULL),
(@ds_submission_id, @ds_q3_id, 'T', NULL, NULL, NULL),

-- Python作业答题（已批改）
(@python_submission_id, @python_q1_id, '列表是可变的，元组是不可变的数据类型', 1, 35, '解释基本正确，但不够全面'),
(@python_submission_id, @python_q2_id, 'C', 1, 30, '正确'),
(@python_submission_id, @python_q3_id, 'T', 1, 20, '正确，但注意Python 3.7+中的变化');

-- 6. 添加错题记录
INSERT INTO `wrong_question` 
(`student_id`, `question_id`, `assignment_id`, `submission_id`, `student_answer`, `correct_answer`)
VALUES
(20250005, @ds_q3_id, @ds_assignment_id, @ds_submission_id, 'T', 'F');

COMMIT; 