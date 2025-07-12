-- 计算机组成原理课程题目数据 (course_id=9, created_by=6)
-- 创建日期: 2025-07-15

-- 清理已有数据（如果需要）
-- DELETE FROM question_option WHERE question_id IN (SELECT id FROM question WHERE course_id = 9);
-- DELETE FROM question WHERE course_id = 9;

-- 插入单选题
INSERT INTO question (title, question_type, difficulty, correct_answer, explanation, knowledge_point, course_id, chapter_id, created_by, create_time)
VALUES 
('在计算机系统中，CPU的主要功能是什么？', 'single', 3, 'B', 'CPU(中央处理器)的主要功能是执行指令和处理数据，是计算机的运算和控制核心。', '计算机系统组成', 9, 11, 6, NOW()),
('以下哪种存储器具有最快的访问速度？', 'single', 2, 'A', '寄存器是CPU内部的存储单元，具有最快的访问速度，其次是缓存、内存和硬盘。', '存储器层次结构', 9, 15, 6, NOW()),
('在IEEE 754标准中，32位单精度浮点数的指数偏移值是多少？', 'single', 4, 'C', 'IEEE 754标准中，32位单精度浮点数的指数偏移值是127，用于将实际指数转换为无符号表示。', '数据表示', 9, 20, 6, NOW()),
('以下哪种总线主要用于CPU与内存之间的数据传输？', 'single', 3, 'A', '数据总线用于在CPU与内存以及其他设备之间传输数据，地址总线用于传输内存地址，控制总线用于传输控制信号。', '总线结构', 9, 17, 6, NOW()),
('在流水线CPU中，以下哪个不是典型的五级流水线阶段？', 'single', 4, 'D', '典型的五级流水线包括：取指令(IF)、指令译码(ID)、执行(EX)、访存(MEM)和写回(WB)，而不包括指令优化阶段。', 'CPU结构', 9, 14, 6, NOW());

-- 插入单选题选项
INSERT INTO question_option (question_id, option_label, option_text)
VALUES
((SELECT id FROM question WHERE title = '在计算机系统中，CPU的主要功能是什么？' AND course_id = 9), 'A', '存储大量数据'),
((SELECT id FROM question WHERE title = '在计算机系统中，CPU的主要功能是什么？' AND course_id = 9), 'B', '执行指令和处理数据'),
((SELECT id FROM question WHERE title = '在计算机系统中，CPU的主要功能是什么？' AND course_id = 9), 'C', '显示图形界面'),
((SELECT id FROM question WHERE title = '在计算机系统中，CPU的主要功能是什么？' AND course_id = 9), 'D', '连接外部设备'),

((SELECT id FROM question WHERE title = '以下哪种存储器具有最快的访问速度？' AND course_id = 9), 'A', '寄存器(Register)'),
((SELECT id FROM question WHERE title = '以下哪种存储器具有最快的访问速度？' AND course_id = 9), 'B', '高速缓存(Cache)'),
((SELECT id FROM question WHERE title = '以下哪种存储器具有最快的访问速度？' AND course_id = 9), 'C', '内存(RAM)'),
((SELECT id FROM question WHERE title = '以下哪种存储器具有最快的访问速度？' AND course_id = 9), 'D', '硬盘(HDD)'),

((SELECT id FROM question WHERE title = '在IEEE 754标准中，32位单精度浮点数的指数偏移值是多少？' AND course_id = 9), 'A', '63'),
((SELECT id FROM question WHERE title = '在IEEE 754标准中，32位单精度浮点数的指数偏移值是多少？' AND course_id = 9), 'B', '64'),
((SELECT id FROM question WHERE title = '在IEEE 754标准中，32位单精度浮点数的指数偏移值是多少？' AND course_id = 9), 'C', '127'),
((SELECT id FROM question WHERE title = '在IEEE 754标准中，32位单精度浮点数的指数偏移值是多少？' AND course_id = 9), 'D', '1023'),

((SELECT id FROM question WHERE title = '以下哪种总线主要用于CPU与内存之间的数据传输？' AND course_id = 9), 'A', '数据总线'),
((SELECT id FROM question WHERE title = '以下哪种总线主要用于CPU与内存之间的数据传输？' AND course_id = 9), 'B', '地址总线'),
((SELECT id FROM question WHERE title = '以下哪种总线主要用于CPU与内存之间的数据传输？' AND course_id = 9), 'C', '控制总线'),
((SELECT id FROM question WHERE title = '以下哪种总线主要用于CPU与内存之间的数据传输？' AND course_id = 9), 'D', '外部总线'),

((SELECT id FROM question WHERE title = '在流水线CPU中，以下哪个不是典型的五级流水线阶段？' AND course_id = 9), 'A', '取指令(IF)'),
((SELECT id FROM question WHERE title = '在流水线CPU中，以下哪个不是典型的五级流水线阶段？' AND course_id = 9), 'B', '指令译码(ID)'),
((SELECT id FROM question WHERE title = '在流水线CPU中，以下哪个不是典型的五级流水线阶段？' AND course_id = 9), 'C', '执行(EX)'),
((SELECT id FROM question WHERE title = '在流水线CPU中，以下哪个不是典型的五级流水线阶段？' AND course_id = 9), 'D', '指令优化(IO)');

-- 插入多选题
INSERT INTO question (title, question_type, difficulty, correct_answer, explanation, knowledge_point, course_id, chapter_id, created_by, create_time)
VALUES 
('以下哪些是计算机系统的主要组成部分？', 'multiple', 2, 'A,B,C,E', '计算机系统主要由CPU、内存、输入设备、输出设备和存储设备组成，而显卡是输出设备的一种，不是主要组成部分。', '计算机系统组成', 9, 11, 6, NOW()),
('以下哪些是RISC处理器的特点？', 'multiple', 4, 'A,C,D', 'RISC(精简指令集计算机)的特点包括指令数量少、指令长度固定、寻址方式简单和使用大量寄存器，而不是复杂的寻址方式。', '指令系统', 9, 13, 6, NOW()),
('以下哪些因素会影响CPU的性能？', 'multiple', 3, 'A,B,C,D', 'CPU性能受时钟频率、指令集架构、缓存大小和流水线深度等因素影响。', 'CPU性能', 9, 14, 6, NOW());

-- 插入多选题选项
INSERT INTO question_option (question_id, option_label, option_text)
VALUES
((SELECT id FROM question WHERE title = '以下哪些是计算机系统的主要组成部分？' AND course_id = 9), 'A', 'CPU(中央处理器)'),
((SELECT id FROM question WHERE title = '以下哪些是计算机系统的主要组成部分？' AND course_id = 9), 'B', '内存(RAM)'),
((SELECT id FROM question WHERE title = '以下哪些是计算机系统的主要组成部分？' AND course_id = 9), 'C', '输入设备'),
((SELECT id FROM question WHERE title = '以下哪些是计算机系统的主要组成部分？' AND course_id = 9), 'D', '显卡'),
((SELECT id FROM question WHERE title = '以下哪些是计算机系统的主要组成部分？' AND course_id = 9), 'E', '存储设备'),

((SELECT id FROM question WHERE title = '以下哪些是RISC处理器的特点？' AND course_id = 9), 'A', '指令数量较少'),
((SELECT id FROM question WHERE title = '以下哪些是RISC处理器的特点？' AND course_id = 9), 'B', '复杂的寻址方式'),
((SELECT id FROM question WHERE title = '以下哪些是RISC处理器的特点？' AND course_id = 9), 'C', '指令长度固定'),
((SELECT id FROM question WHERE title = '以下哪些是RISC处理器的特点？' AND course_id = 9), 'D', '使用大量寄存器'),
((SELECT id FROM question WHERE title = '以下哪些是RISC处理器的特点？' AND course_id = 9), 'E', '每条指令执行时间不同'),

((SELECT id FROM question WHERE title = '以下哪些因素会影响CPU的性能？' AND course_id = 9), 'A', '时钟频率'),
((SELECT id FROM question WHERE title = '以下哪些因素会影响CPU的性能？' AND course_id = 9), 'B', '指令集架构'),
((SELECT id FROM question WHERE title = '以下哪些因素会影响CPU的性能？' AND course_id = 9), 'C', '缓存大小'),
((SELECT id FROM question WHERE title = '以下哪些因素会影响CPU的性能？' AND course_id = 9), 'D', '流水线深度'),
((SELECT id FROM question WHERE title = '以下哪些因素会影响CPU的性能？' AND course_id = 9), 'E', '主板颜色');

-- 插入判断题
INSERT INTO question (title, question_type, difficulty, correct_answer, explanation, knowledge_point, course_id, chapter_id, created_by, create_time)
VALUES 
('在冯·诺依曼结构中，程序和数据存储在同一个存储器中。', 'true_false', 2, 'T', '冯·诺依曼结构的一个核心特点就是程序和数据存储在同一个存储器中，这使得计算机可以像处理数据一样处理指令。', '计算机体系结构', 9, 11, 6, NOW()),
('CISC处理器比RISC处理器的指令数量更少。', 'true_false', 3, 'F', 'CISC(复杂指令集计算机)的指令数量比RISC(精简指令集计算机)更多，而不是更少。', '指令系统', 9, 13, 6, NOW()),
('在计算机中，1KB等于1000字节。', 'true_false', 1, 'F', '在计算机中，1KB(千字节)等于1024字节，而不是1000字节。这是因为计算机使用二进制，1KB = 2^10 字节。', '数据表示', 9, 20, 6, NOW()),
('高速缓存(Cache)的主要目的是弥补CPU和内存之间的速度差异。', 'true_false', 2, 'T', '高速缓存的主要目的确实是弥补CPU和内存之间的速度差异，通过存储频繁使用的数据来提高系统性能。', '存储器层次结构', 9, 15, 6, NOW());

-- 插入判断题选项
INSERT INTO question_option (question_id, option_label, option_text)
VALUES
((SELECT id FROM question WHERE title = '在冯·诺依曼结构中，程序和数据存储在同一个存储器中。' AND course_id = 9), 'T', '正确'),
((SELECT id FROM question WHERE title = '在冯·诺依曼结构中，程序和数据存储在同一个存储器中。' AND course_id = 9), 'F', '错误'),

((SELECT id FROM question WHERE title = 'CISC处理器比RISC处理器的指令数量更少。' AND course_id = 9), 'T', '正确'),
((SELECT id FROM question WHERE title = 'CISC处理器比RISC处理器的指令数量更少。' AND course_id = 9), 'F', '错误'),

((SELECT id FROM question WHERE title = '在计算机中，1KB等于1000字节。' AND course_id = 9), 'T', '正确'),
((SELECT id FROM question WHERE title = '在计算机中，1KB等于1000字节。' AND course_id = 9), 'F', '错误'),

((SELECT id FROM question WHERE title = '高速缓存(Cache)的主要目的是弥补CPU和内存之间的速度差异。' AND course_id = 9), 'T', '正确'),
((SELECT id FROM question WHERE title = '高速缓存(Cache)的主要目的是弥补CPU和内存之间的速度差异。' AND course_id = 9), 'F', '错误');

-- 插入填空题
INSERT INTO question (title, question_type, difficulty, correct_answer, explanation, knowledge_point, course_id, chapter_id, created_by, create_time)
VALUES 
('在二进制数系统中，十进制数15表示为________。', 'blank', 2, '1111', '十进制数15转换为二进制是1111，计算过程：15 = 8 + 4 + 2 + 1 = 2^3 + 2^2 + 2^1 + 2^0 = 1111(二进制)', '数制转换', 9, 20, 6, NOW()),
('计算机中的主频通常以________为单位。', 'blank', 1, 'Hz', '计算机主频是CPU时钟频率，通常以赫兹(Hz)为单位，常见的有MHz(兆赫)和GHz(吉赫)。', 'CPU性能', 9, 14, 6, NOW()),
('在计算机存储层次结构中，从上到下依次是寄存器、________、内存和外存。', 'blank', 3, '高速缓存', '计算机存储层次结构从上到下依次是：寄存器、高速缓存(Cache)、内存(RAM)和外存(硬盘等)，速度依次降低，容量依次增大。', '存储器层次结构', 9, 15, 6, NOW()),
('在计算机中，将二进制数据转换为人类可读的十进制数的过程称为________。', 'blank', 2, '解码', '解码是将二进制数据转换为人类可读形式的过程，是计算机内部数据表示与人类理解之间的桥梁。', '数据表示', 9, 20, 6, NOW());

-- 插入简答题
INSERT INTO question (title, question_type, difficulty, correct_answer, explanation, knowledge_point, course_id, chapter_id, created_by, create_time)
VALUES 
('简述计算机的冯·诺依曼结构及其主要特点。', 'short', 3, '冯·诺依曼结构是现代计算机的基本结构，其主要特点包括：1. 计算机由运算器、控制器、存储器、输入设备和输出设备五大部分组成；2. 程序和数据存储在同一个存储器中；3. 指令和数据均以二进制形式表示；4. 指令按地址顺序存放，通常按顺序执行；5. 采用存储程序原理，程序可以像数据一样存取。', '冯·诺依曼结构是现代计算机的基础，理解其特点对理解计算机工作原理至关重要。', '计算机体系结构', 9, 11, 6, NOW()),
('解释CPU中的流水线技术原理及其优缺点。', 'short', 4, 'CPU流水线技术是将指令执行过程分解为多个阶段，使多条指令可以同时在不同阶段执行，从而提高CPU的吞吐率。典型的五级流水线包括：取指令(IF)、指令译码(ID)、执行(EX)、访存(MEM)和写回(WB)。\n\n优点：1. 提高CPU的吞吐率；2. 提高硬件资源利用率；3. 减少平均指令执行时间。\n\n缺点：1. 可能产生数据相关、控制相关和结构相关等冒险；2. 增加了硬件复杂度；3. 流水线越深，分支预测失败的惩罚越大。', '流水线技术是现代CPU提高性能的重要手段，理解其原理有助于理解CPU的工作方式。', 'CPU结构', 9, 14, 6, NOW()),
('比较CISC和RISC架构的主要区别及各自的优缺点。', 'short', 5, 'CISC(复杂指令集计算机)和RISC(精简指令集计算机)是两种不同的处理器设计理念：\n\nCISC特点：1. 指令数量多且复杂；2. 指令长度可变；3. 寻址方式多样；4. 指令执行时间不等；5. 硬件实现复杂，软件实现简单。\n\nRISC特点：1. 指令数量少且简单；2. 指令长度固定；3. 寻址方式简单；4. 使用大量寄存器；5. 强调优化编译器；6. 硬件实现简单，软件实现复杂。\n\nCISC优点：代码密度高，适合内存受限系统；缺点：硬件复杂，功耗高，流水线实现困难。\n\nRISC优点：硬件简单，功耗低，易于实现流水线；缺点：代码密度低，对编译器要求高。\n\n现代处理器通常融合了两种架构的特点，如x86处理器内部采用RISC微架构，但对外提供CISC指令集。', 'CISC和RISC代表了处理器设计的两种不同思路，理解它们的区别有助于理解计算机体系结构的发展。', '指令系统', 9, 13, 6, NOW()),
('描述计算机存储层次结构，并解释为什么要采用这种层次化设计。', 'short', 3, '计算机存储层次结构从上到下依次是：\n1. 寄存器：CPU内部，容量最小(几KB)，速度最快，成本最高\n2. 高速缓存(Cache)：分为L1、L2、L3等级，容量适中(几MB)，速度很快，成本高\n3. 主存(RAM)：容量较大(几GB)，速度中等，成本中等\n4. 外存(硬盘、SSD等)：容量最大(几TB)，速度最慢，成本最低\n\n采用层次化设计的原因：\n1. 平衡速度与容量的矛盾：高速存储器成本高，容量小；大容量存储器速度慢，成本低\n2. 利用程序的局部性原理：时间局部性(最近访问的数据很可能再次被访问)和空间局部性(最近访问的数据附近的数据很可能被访问)\n3. 通过缓存机制，使系统在大部分情况下能以接近高速存储器的速度工作，同时拥有大容量存储器的容量\n\n这种层次化设计是计算机系统性能与成本平衡的重要手段。', '存储层次结构是计算机系统设计的重要概念，理解其原理有助于理解计算机性能优化的方法。', '存储器层次结构', 9, 15, 6, NOW());

-- 插入代码题
INSERT INTO question (title, question_type, difficulty, correct_answer, explanation, knowledge_point, course_id, chapter_id, created_by, create_time)
VALUES 
('请编写一个简单的C语言程序，实现两个32位无符号整数的二进制加法，并处理可能的溢出情况。', 'code', 4, '#include <stdio.h>\n#include <stdint.h>\n\ntypedef struct {\n    uint32_t result;\n    uint8_t overflow; // 0表示无溢出，1表示有溢出\n} AddResult;\n\nAddResult add_with_overflow(uint32_t a, uint32_t b) {\n    AddResult res;\n    res.result = a + b;\n    // 如果结果小于任一操作数，则发生了溢出\n    res.overflow = (res.result < a || res.result < b) ? 1 : 0;\n    return res;\n}\n\nint main() {\n    uint32_t a = 4294967290; // 接近uint32_t最大值\n    uint32_t b = 10;\n    \n    AddResult result = add_with_overflow(a, b);\n    \n    printf("a = %u\\n", a);\n    printf("b = %u\\n", b);\n    printf("a + b = %u\\n", result.result);\n    printf("溢出标志: %s\\n", result.overflow ? "是" : "否");\n    \n    return 0;\n}', '此程序演示了如何在C语言中实现二进制加法并检测溢出。在计算机组成原理中，理解整数运算的溢出处理是很重要的概念。', '计算机算术运算', 9, 20, 6, NOW()),
('请编写一个简单的汇编语言程序(MIPS指令集)，实现两个整数的加法并将结果存储到内存中。', 'code', 5, '# MIPS汇编程序：两数相加\n# 假设$a0和$a1中存储着要相加的两个整数\n# 结果将存储在内存地址result中\n\n.data\nresult: .word 0    # 分配一个字(4字节)用于存储结果\n\n.text\n.globl main\nmain:\n    # 假设要相加的两个数已在$a0和$a1中\n    li $a0, 25      # 加载第一个数\n    li $a1, 30      # 加载第二个数\n    \n    # 执行加法\n    add $t0, $a0, $a1   # $t0 = $a0 + $a1\n    \n    # 将结果存储到内存\n    sw $t0, result      # 存储结果到内存\n    \n    # 打印结果(系统调用)\n    li $v0, 1           # 系统调用代码1表示打印整数\n    move $a0, $t0       # 将结果移到$a0用于打印\n    syscall             # 执行系统调用\n    \n    # 退出程序\n    li $v0, 10          # 系统调用代码10表示退出\n    syscall             # 执行系统调用', '此程序演示了MIPS汇编语言中如何执行基本的加法运算并与内存交互。理解汇编语言是理解计算机如何执行指令的基础。', '汇编语言编程', 9, 13, 6, NOW()); 