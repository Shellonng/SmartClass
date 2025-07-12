package com.education.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.education.dto.AssignmentDTO;
import com.education.entity.Assignment;
import com.education.entity.AssignmentSubmission;
import com.education.entity.Course;
import com.education.entity.CourseStudent;
import com.education.entity.Student;
import com.education.entity.User;
import com.education.mapper.AssignmentMapper;
import com.education.mapper.AssignmentSubmissionMapper;
import com.education.mapper.CourseStudentMapper;
import com.education.mapper.StudentMapper;
import com.education.mapper.UserMapper;
import com.education.service.AssignmentService;
import com.education.service.teacher.CourseService;
import com.education.service.teacher.QuestionService;
import com.education.service.common.UserService;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.BeanUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.util.StringUtils;

import java.time.LocalDateTime;
import java.util.*;
import java.util.stream.Collectors;

/**
 * 作业服务实现类
 */
@Slf4j
@Service
public class AssignmentServiceImpl extends ServiceImpl<AssignmentMapper, Assignment> implements AssignmentService {
    
    @Autowired
    private CourseService courseService;
    
    @Autowired
    private UserService userService;
    
    @Autowired
    private QuestionService questionService;
    
    @Autowired
    private CourseStudentMapper courseStudentMapper;
    
    @Autowired
    private AssignmentSubmissionMapper assignmentSubmissionMapper;
    
    @Autowired
    private StudentMapper studentMapper;
    
    @Autowired
    private UserMapper userMapper;
    
    private final ObjectMapper objectMapper = new ObjectMapper();
    
    @Override
    public IPage<AssignmentDTO> pageAssignments(Page<Assignment> page, Long courseId, Long userId, String keyword, Integer status) {
        QueryWrapper<Assignment> queryWrapper = new QueryWrapper<>();
        
        // 构建查询条件
        if (courseId != null) {
            queryWrapper.eq("course_id", courseId);
        }
        if (userId != null) {
            queryWrapper.eq("user_id", userId);
        }
        if (StringUtils.hasText(keyword)) {
            queryWrapper.like("title", keyword)
                    .or().like("description", keyword);
        }
        if (status != null) {
            queryWrapper.eq("status", status);
        }
        
        // 只查询类型为homework的作业
        queryWrapper.eq("type", "homework");
        
        // 按创建时间倒序
        queryWrapper.orderByDesc("create_time");
        
        IPage<Assignment> assignmentPage = this.page(page, queryWrapper);
        
        // 转换为DTO并填充额外信息
        IPage<AssignmentDTO> dtoPage = assignmentPage.convert(this::convertToDTO);
        
        // 批量查询课程名称和教师名称
        fillCourseAndTeacherNames(dtoPage.getRecords());
        
        // 计算提交率
        fillSubmissionRates(dtoPage.getRecords());
        
        return dtoPage;
    }
    
    @Override
    public AssignmentDTO getAssignmentById(Long id) {
        Assignment assignment = this.getById(id);
        if (assignment == null) {
            return null;
        }
        
        AssignmentDTO dto = convertToDTO(assignment);
        
        // 填充课程和教师信息
        fillCourseAndTeacherNames(Collections.singletonList(dto));
        
        // 填充提交率
        fillSubmissionRates(Collections.singletonList(dto));
        
        // 如果是答题型作业，加载题目列表
        if ("question".equals(dto.getMode())) {
            // TODO: 加载题目列表
            dto.setQuestions(new ArrayList<>());
        }
        
        return dto;
    }
    
    @Override
    @Transactional
    public Long createAssignment(AssignmentDTO assignmentDTO) {
        // 基本验证
        validateAssignmentDTO(assignmentDTO);
        
        Assignment assignment = new Assignment();
        BeanUtils.copyProperties(assignmentDTO, assignment);
        
        // 手动转换LocalDateTime到Date
        if (assignmentDTO.getStartTime() != null) {
            assignment.setStartTime(java.sql.Timestamp.valueOf(assignmentDTO.getStartTime()));
        }
        if (assignmentDTO.getEndTime() != null) {
            assignment.setEndTime(java.sql.Timestamp.valueOf(assignmentDTO.getEndTime()));
        }
        
        // 设置默认值
        assignment.setType("homework"); // 固定为作业类型
        assignment.setStatus(0); // 默认未发布
        assignment.setCreateTime(new Date()); // 使用当前日期
        assignment.setUpdateTime(new Date()); // 使用当前日期
        
        // 处理文件类型（转换为JSON字符串）
        if (assignmentDTO.getAllowedFileTypes() != null && !assignmentDTO.getAllowedFileTypes().isEmpty()) {
            try {
                assignment.setAllowedFileTypes(objectMapper.writeValueAsString(assignmentDTO.getAllowedFileTypes()));
            } catch (Exception e) {
                log.error("转换文件类型失败", e);
                throw new RuntimeException("文件类型配置错误");
            }
        }
        
        // 保存作业
        this.save(assignment);
        
        // 如果有智能组卷配置，保存配置信息
        if (assignmentDTO.getConfig() != null) {
            saveAssignmentConfig(assignment.getId(), assignmentDTO.getConfig());
        }
        
        return assignment.getId();
    }
    
    @Override
    @Transactional
    public Boolean updateAssignment(Long id, AssignmentDTO assignmentDTO) {
        if (assignmentDTO == null || id == null) {
            throw new IllegalArgumentException("作业ID和作业信息不能为空");
        }
        
        Assignment existingAssignment = this.getById(id);
        if (existingAssignment == null) {
            throw new RuntimeException("作业不存在");
        }
        
        // 检查是否可以修改
        if (existingAssignment.getStatus() == 1 && assignmentDTO.getStatus() != 0) {
            throw new RuntimeException("已发布的作业不能修改，只能取消发布");
        }
        
        // 基本验证
        validateAssignmentDTO(assignmentDTO);
        
        Assignment assignment = new Assignment();
        BeanUtils.copyProperties(assignmentDTO, assignment);
        assignment.setId(id);
        assignment.setUpdateTime(new Date()); // 使用当前日期
        
        // 手动转换LocalDateTime到Date
        if (assignmentDTO.getStartTime() != null) {
            assignment.setStartTime(java.sql.Timestamp.valueOf(assignmentDTO.getStartTime()));
        }
        if (assignmentDTO.getEndTime() != null) {
            assignment.setEndTime(java.sql.Timestamp.valueOf(assignmentDTO.getEndTime()));
        }
        
        // 处理文件类型
        if (assignmentDTO.getAllowedFileTypes() != null && !assignmentDTO.getAllowedFileTypes().isEmpty()) {
            try {
                assignment.setAllowedFileTypes(objectMapper.writeValueAsString(assignmentDTO.getAllowedFileTypes()));
            } catch (Exception e) {
                log.error("转换文件类型失败", e);
                throw new RuntimeException("文件类型配置错误");
            }
        }
        
        // 更新作业
        boolean result = this.updateById(assignment);
        
        // 更新配置信息
        if (assignmentDTO.getConfig() != null) {
            saveAssignmentConfig(id, assignmentDTO.getConfig());
        }
        
        return result;
    }
    
    @Override
    public Boolean deleteAssignment(Long id) {
        Assignment assignment = this.getById(id);
        if (assignment == null) {
            throw new RuntimeException("作业不存在");
        }
        
        if (assignment.getStatus() == 1) {
            throw new RuntimeException("已发布的作业不能删除");
        }
        
        return this.removeById(id);
    }
    
    @Override
    @Transactional
    public Boolean publishAssignment(Long id) {
        Assignment assignment = this.getById(id);
        if (assignment == null) {
            throw new RuntimeException("作业不存在");
        }
        
        assignment.setStatus(1);
        assignment.setUpdateTime(new Date()); // 使用当前日期
        
        // 获取该课程的所有学生
        QueryWrapper<CourseStudent> queryWrapper = new QueryWrapper<>();
        queryWrapper.eq("course_id", assignment.getCourseId());
        List<CourseStudent> courseStudents = courseStudentMapper.selectList(queryWrapper);
        
        // 为每个学生创建作业提交记录
        for (CourseStudent courseStudent : courseStudents) {
            // 查询学生对应的用户ID
            Student student = studentMapper.selectById(courseStudent.getStudentId());
            if (student != null) {
                AssignmentSubmission submission = new AssignmentSubmission();
                submission.setAssignmentId(id);
                submission.setStudentId(student.getUserId()); // 使用用户ID而不是学生ID
                submission.setStatus(0); // 未提交
                submission.setCreateTime(new Date());
                submission.setUpdateTime(new Date());
                
                assignmentSubmissionMapper.insert(submission);
            }
        }
        
        return this.updateById(assignment);
    }
    
    @Override
    @Transactional
    public Boolean unpublishAssignment(Long id) {
        Assignment assignment = this.getById(id);
        if (assignment == null) {
            throw new RuntimeException("作业不存在");
        }
        
        assignment.setStatus(0);
        assignment.setUpdateTime(new Date()); // 使用当前日期
        
        // 删除所有与该作业相关的提交记录
        QueryWrapper<AssignmentSubmission> queryWrapper = new QueryWrapper<>();
        queryWrapper.eq("assignment_id", id);
        assignmentSubmissionMapper.delete(queryWrapper);
        
        return this.updateById(assignment);
    }
    
    @Override
    public List<AssignmentDTO.AssignmentQuestionDTO> generatePaper(AssignmentDTO assignmentDTO) {
        // TODO: 实现智能组卷逻辑
        // 1. 根据知识点和难度筛选题目
        // 2. 按照题型分布要求选择题目
        // 3. 随机排序或按难度排序
        
        log.info("智能组卷功能待实现，配置：{}", assignmentDTO.getConfig());
        return new ArrayList<>();
    }
    
    @Override
    public Boolean selectQuestions(Long assignmentId, List<Long> questionIds) {
        // TODO: 实现手动选题逻辑
        // 1. 验证题目是否存在
        // 2. 保存作业题目关联关系
        
        log.info("手动选题功能待实现，作业ID：{}，题目IDs：{}", assignmentId, questionIds);
        return true;
    }
    
    @Override
    public Double getSubmissionRate(Long assignmentId) {
        // TODO: 实现提交率计算
        return 0.0;
    }
    
    @Override
    public IPage<Map<String, Object>> getAssignmentSubmissions(Long assignmentId, Page<Object> page, Integer status) {
        log.info("获取作业提交记录: assignmentId={}, status={}", assignmentId, status);
        
        // 创建一个Page对象来存储结果
        Page<Map<String, Object>> resultPage = new Page<>(page.getCurrent(), page.getSize());
        List<Map<String, Object>> records = new ArrayList<>();
        
        try {
            // 查询作业信息
            Assignment assignment = this.getById(assignmentId);
            if (assignment == null) {
                log.error("作业不存在: {}", assignmentId);
                resultPage.setRecords(records);
                resultPage.setTotal(0);
                return resultPage;
            }
            
            // 查询该作业的所有提交记录
            QueryWrapper<AssignmentSubmission> queryWrapper = new QueryWrapper<>();
            queryWrapper.eq("assignment_id", assignmentId);
            
            // 如果指定了状态，添加状态过滤条件
            if (status != null) {
                queryWrapper.eq("status", status);
            }
            
            // 使用MyBatis-Plus的分页查询
            Page<AssignmentSubmission> submissionPage = new Page<>(page.getCurrent(), page.getSize());
            IPage<AssignmentSubmission> submissionIPage = assignmentSubmissionMapper.selectPage(submissionPage, queryWrapper);
            
            // 获取学生信息，将user_id映射到学生姓名
            Map<Long, String> studentNameMap = new HashMap<>();
            
            // 遍历提交记录，转换为前端需要的格式
            for (AssignmentSubmission submission : submissionIPage.getRecords()) {
                Map<String, Object> record = new HashMap<>();
                record.put("id", submission.getId());
                record.put("assignmentId", submission.getAssignmentId());
                record.put("studentId", submission.getStudentId());
                record.put("status", submission.getStatus());
                record.put("score", submission.getScore());
                record.put("feedback", submission.getFeedback());
                record.put("submitTime", submission.getSubmitTime());
                record.put("createTime", submission.getCreateTime());
                record.put("updateTime", submission.getUpdateTime());
                
                // 获取学生姓名
                String studentName = studentNameMap.get(submission.getStudentId());
                if (studentName == null) {
                    // 如果缓存中没有，则查询数据库
                    User user = userMapper.selectById(submission.getStudentId());
                    if (user != null) {
                        studentName = user.getDisplayName();
                        studentNameMap.put(submission.getStudentId(), studentName);
                    } else {
                        studentName = "未知学生";
                    }
                }
                record.put("studentName", studentName);
                
                records.add(record);
            }
            
            // 设置返回结果
            resultPage.setRecords(records);
            resultPage.setTotal(submissionIPage.getTotal());
            
            log.info("获取到{}条提交记录", records.size());
        } catch (Exception e) {
            log.error("获取作业提交记录异常", e);
        }
        
        return resultPage;
    }
    
    @Override
    public Boolean setReferenceAnswer(Long assignmentId, String referenceAnswer) {
        Assignment assignment = this.getById(assignmentId);
        if (assignment == null) {
            throw new RuntimeException("作业不存在");
        }
        
        assignment.setReferenceAnswer(referenceAnswer);
        assignment.setUpdateTime(new Date()); // 使用当前日期
        
        return this.updateById(assignment);
    }
    
    @Override
    public String aiGradeBatch(Long assignmentId) {
        // TODO: 实现批量智能批改
        return "task-" + UUID.randomUUID().toString();
    }
    
    @Override
    public Map<String, Object> getGradingStatus(String taskId) {
        // 现有实现保持不变
        Map<String, Object> result = new HashMap<>();
        result.put("taskId", taskId);
        result.put("status", "completed");
        result.put("progress", 100);
        return result;
    }
    
    @Override
    public List<String> getKnowledgePoints(Long courseId, Long createdBy) {
        log.info("获取知识点列表: courseId={}, createdBy={}", courseId, createdBy);
        
        // 从题库中获取知识点列表
        List<String> knowledgePoints = new ArrayList<>();
        
        try {
            // 这里应该是从数据库中查询知识点
            // 简化实现，返回模拟数据
            if (courseId != null && courseId == 1) {
                // Java课程知识点
                knowledgePoints.add("Java基础");
                knowledgePoints.add("面向对象");
                knowledgePoints.add("集合框架");
                knowledgePoints.add("异常处理");
                knowledgePoints.add("IO流");
                knowledgePoints.add("多线程");
                knowledgePoints.add("反射");
                knowledgePoints.add("注解");
            } else if (courseId != null && courseId == 2) {
                // 数据结构知识点
                knowledgePoints.add("线性表");
                knowledgePoints.add("栈和队列");
                knowledgePoints.add("树");
                knowledgePoints.add("图");
                knowledgePoints.add("排序算法");
                knowledgePoints.add("查找算法");
                knowledgePoints.add("哈希表");
            } else if (courseId != null && courseId == 3) {
                // Python课程知识点
                knowledgePoints.add("Python基础");
                knowledgePoints.add("数据类型");
                knowledgePoints.add("控制流");
                knowledgePoints.add("函数");
                knowledgePoints.add("模块和包");
                knowledgePoints.add("面向对象");
                knowledgePoints.add("文件操作");
            } else {
                // 默认知识点
                knowledgePoints.add("基础知识");
                knowledgePoints.add("进阶知识");
                knowledgePoints.add("高级知识");
                knowledgePoints.add("综合应用");
            }
            
            log.info("获取到{}个知识点", knowledgePoints.size());
        } catch (Exception e) {
            log.error("获取知识点列表异常", e);
        }
        
        return knowledgePoints;
    }
    
    @Override
    public Map<String, List<Map<String, Object>>> getQuestionsByType(Long courseId, String questionType, 
                                                                    Integer difficulty, String knowledgePoint, 
                                                                    Long createdBy, String keyword) {
        log.info("获取题目列表: courseId={}, type={}, difficulty={}, knowledgePoint={}, createdBy={}, keyword={}",
                courseId, questionType, difficulty, knowledgePoint, createdBy, keyword);
        
        Map<String, List<Map<String, Object>>> result = new HashMap<>();
        
        try {
            // 这里应该是从数据库中查询题目
            // 简化实现，返回模拟数据
            
            // 单选题
            List<Map<String, Object>> singleChoiceQuestions = new ArrayList<>();
            for (int i = 1; i <= 5; i++) {
                Map<String, Object> question = new HashMap<>();
                question.put("id", 100 + i);
                question.put("title", "单选题" + i + ": 以下哪个选项是正确的？");
                question.put("type", "single_choice");
                question.put("difficulty", difficulty != null ? difficulty : (i % 3 + 1));
                question.put("knowledgePoint", knowledgePoint != null ? knowledgePoint : "基础知识");
                question.put("score", 10);
                
                List<Map<String, Object>> options = new ArrayList<>();
                for (int j = 0; j < 4; j++) {
                    Map<String, Object> option = new HashMap<>();
                    option.put("id", "option" + j);
                    option.put("content", "选项" + (char)('A' + j));
                    options.add(option);
                }
                question.put("options", options);
                
                singleChoiceQuestions.add(question);
            }
            result.put("single_choice", singleChoiceQuestions);
            
            // 多选题
            List<Map<String, Object>> multiChoiceQuestions = new ArrayList<>();
            for (int i = 1; i <= 3; i++) {
                Map<String, Object> question = new HashMap<>();
                question.put("id", 200 + i);
                question.put("title", "多选题" + i + ": 以下哪些选项是正确的？");
                question.put("type", "multi_choice");
                question.put("difficulty", difficulty != null ? difficulty : (i % 3 + 1));
                question.put("knowledgePoint", knowledgePoint != null ? knowledgePoint : "进阶知识");
                question.put("score", 15);
                
                List<Map<String, Object>> options = new ArrayList<>();
                for (int j = 0; j < 4; j++) {
                    Map<String, Object> option = new HashMap<>();
                    option.put("id", "option" + j);
                    option.put("content", "选项" + (char)('A' + j));
                    options.add(option);
                }
                question.put("options", options);
                
                multiChoiceQuestions.add(question);
            }
            result.put("multi_choice", multiChoiceQuestions);
            
            // 判断题
            List<Map<String, Object>> trueFalseQuestions = new ArrayList<>();
            for (int i = 1; i <= 4; i++) {
                Map<String, Object> question = new HashMap<>();
                question.put("id", 300 + i);
                question.put("title", "判断题" + i + ": 这是一道判断题。");
                question.put("type", "true_false");
                question.put("difficulty", difficulty != null ? difficulty : (i % 3 + 1));
                question.put("knowledgePoint", knowledgePoint != null ? knowledgePoint : "基础知识");
                question.put("score", 5);
                trueFalseQuestions.add(question);
            }
            result.put("true_false", trueFalseQuestions);
            
            // 填空题
            List<Map<String, Object>> fillBlankQuestions = new ArrayList<>();
            for (int i = 1; i <= 3; i++) {
                Map<String, Object> question = new HashMap<>();
                question.put("id", 400 + i);
                question.put("title", "填空题" + i + ": 请填写正确答案。");
                question.put("type", "fill_blank");
                question.put("difficulty", difficulty != null ? difficulty : (i % 3 + 1));
                question.put("knowledgePoint", knowledgePoint != null ? knowledgePoint : "进阶知识");
                question.put("score", 10);
                fillBlankQuestions.add(question);
            }
            result.put("fill_blank", fillBlankQuestions);
            
            // 简答题
            List<Map<String, Object>> shortAnswerQuestions = new ArrayList<>();
            for (int i = 1; i <= 2; i++) {
                Map<String, Object> question = new HashMap<>();
                question.put("id", 500 + i);
                question.put("title", "简答题" + i + ": 请简要回答以下问题。");
                question.put("type", "short_answer");
                question.put("difficulty", difficulty != null ? difficulty : (i % 3 + 1));
                question.put("knowledgePoint", knowledgePoint != null ? knowledgePoint : "高级知识");
                question.put("score", 20);
                shortAnswerQuestions.add(question);
            }
            result.put("short_answer", shortAnswerQuestions);
            
            // 如果指定了题型，只返回对应题型的题目
            if (StringUtils.hasText(questionType)) {
                Map<String, List<Map<String, Object>>> filteredResult = new HashMap<>();
                if (result.containsKey(questionType)) {
                    filteredResult.put(questionType, result.get(questionType));
                }
                return filteredResult;
            }
            
            log.info("获取到{}种题型，共{}道题目", result.size(), 
                    result.values().stream().mapToInt(List::size).sum());
            
        } catch (Exception e) {
            log.error("获取题目列表异常", e);
        }
        
        return result;
    }
    
    private AssignmentDTO convertToDTO(Assignment assignment) {
        AssignmentDTO dto = new AssignmentDTO();
        BeanUtils.copyProperties(assignment, dto);
        
        // 手动转换Date到LocalDateTime
        if (assignment.getStartTime() != null) {
            dto.setStartTime(assignment.getStartTime().toInstant()
                .atZone(java.time.ZoneId.systemDefault())
                .toLocalDateTime());
        }
        
        if (assignment.getEndTime() != null) {
            dto.setEndTime(assignment.getEndTime().toInstant()
                .atZone(java.time.ZoneId.systemDefault())
                .toLocalDateTime());
        }
        
        if (assignment.getCreateTime() != null) {
            dto.setCreateTime(assignment.getCreateTime().toInstant()
                .atZone(java.time.ZoneId.systemDefault())
                .toLocalDateTime());
        }
        
        if (assignment.getUpdateTime() != null) {
            dto.setUpdateTime(assignment.getUpdateTime().toInstant()
                .atZone(java.time.ZoneId.systemDefault())
                .toLocalDateTime());
        }
        
        // 如果有文件类型，转换为List
        if (StringUtils.hasText(assignment.getAllowedFileTypes())) {
            try {
                List<String> fileTypes = objectMapper.readValue(assignment.getAllowedFileTypes(), new TypeReference<List<String>>() {});
                dto.setAllowedFileTypes(fileTypes);
            } catch (Exception e) {
                log.error("解析文件类型失败", e);
            }
        }
        
        return dto;
    }
    
    private void fillCourseAndTeacherNames(List<AssignmentDTO> assignments) {
        if (assignments == null || assignments.isEmpty()) {
            return;
        }
        
        // 收集所有课程ID
        Set<Long> courseIds = assignments.stream()
                .map(AssignmentDTO::getCourseId)
                .filter(Objects::nonNull)
                .collect(Collectors.toSet());
        
        // 收集所有用户ID
        Set<Long> userIds = assignments.stream()
                .map(AssignmentDTO::getUserId)
                .filter(Objects::nonNull)
                .collect(Collectors.toSet());
        
        // 批量查询课程信息
        Map<Long, String> courseNameMap = new HashMap<>();
        if (!courseIds.isEmpty()) {
            // 使用getCoursesByIds方法替代listByIds
            List<Course> courses = getCoursesByIds(courseIds);
            if (courses != null) {
                for (Course course : courses) {
                    courseNameMap.put(course.getId(), course.getTitle()); // 使用title字段代替getName()
                }
            }
        }
        
        // 批量查询教师信息
        Map<Long, String> teacherNameMap = new HashMap<>();
        if (!userIds.isEmpty()) {
            // 使用getUsersByIds方法替代listByIds
            List<User> users = getUsersByIds(userIds);
            if (users != null) {
                for (User user : users) {
                    teacherNameMap.put(user.getId(), user.getDisplayName()); // 使用getDisplayName()方法代替getName()
                }
            }
        }
        
        // 填充课程名称和教师名称
        for (AssignmentDTO dto : assignments) {
            if (dto.getCourseId() != null && courseNameMap.containsKey(dto.getCourseId())) {
                dto.setCourseName(courseNameMap.get(dto.getCourseId()));
            }
            
            if (dto.getUserId() != null && teacherNameMap.containsKey(dto.getUserId())) {
                dto.setTeacherName(teacherNameMap.get(dto.getUserId()));
            }
        }
    }
    
    // 辅助方法：根据ID列表获取课程信息
    private List<Course> getCoursesByIds(Set<Long> courseIds) {
        // 这里需要根据实际的CourseService接口实现
        // 由于CourseService接口没有listByIds方法，我们需要自行实现或调用其他方法
        try {
            // 假设CourseService有一个根据ID列表获取课程的方法
            // 实际实现可能需要根据CourseService的接口调整
            return new ArrayList<>(); // 临时返回空列表
        } catch (Exception e) {
            log.error("获取课程信息失败", e);
            return new ArrayList<>();
        }
    }
    
    // 辅助方法：根据ID列表获取用户信息
    private List<User> getUsersByIds(Set<Long> userIds) {
        // 这里需要根据实际的UserService接口实现
        // 由于UserService接口没有listByIds方法，我们需要自行实现或调用其他方法
        try {
            // 假设UserService有一个根据ID列表获取用户的方法
            // 实际实现可能需要根据UserService的接口调整
            return new ArrayList<>(); // 临时返回空列表
        } catch (Exception e) {
            log.error("获取用户信息失败", e);
            return new ArrayList<>();
        }
    }
    
    private void fillSubmissionRates(List<AssignmentDTO> assignments) {
        // TODO: 实现提交率计算
        for (AssignmentDTO dto : assignments) {
            dto.setSubmissionRate(0.0);
        }
    }
    
    private void validateAssignmentDTO(AssignmentDTO assignmentDTO) {
        if (assignmentDTO == null) {
            throw new IllegalArgumentException("作业信息不能为空");
        }
        
        if (!StringUtils.hasText(assignmentDTO.getTitle())) {
            throw new IllegalArgumentException("作业标题不能为空");
        }
        
        if (assignmentDTO.getCourseId() == null) {
            throw new IllegalArgumentException("所属课程不能为空");
        }
        
        if (assignmentDTO.getStartTime() == null) {
            throw new IllegalArgumentException("开始时间不能为空");
        }
        
        if (assignmentDTO.getEndTime() == null) {
            throw new IllegalArgumentException("结束时间不能为空");
        }
        
        if (assignmentDTO.getStartTime().isAfter(assignmentDTO.getEndTime())) {
            throw new IllegalArgumentException("开始时间不能晚于结束时间");
        }
    }
    
    private void saveAssignmentConfig(Long assignmentId, AssignmentDTO.AssignmentConfig config) {
        // TODO: 实现保存作业配置到assignment_config表
    }
}