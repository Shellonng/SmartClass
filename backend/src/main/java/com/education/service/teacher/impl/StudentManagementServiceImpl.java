package com.education.service.teacher.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.education.dto.common.PageResponse;
import com.education.entity.*;
import com.education.exception.BusinessException;
import com.education.exception.ResultCode;
import com.education.mapper.*;
import com.education.security.SecurityUtil;
import com.education.service.teacher.StudentManagementService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.util.StringUtils;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * 学生管理服务实现类
 */
@Service
@Slf4j
@RequiredArgsConstructor
public class StudentManagementServiceImpl implements StudentManagementService {

    private final StudentMapper studentMapper;
    private final UserMapper userMapper;
    private final CourseClassMapper courseClassMapper;
    private final ClassStudentMapper classStudentMapper;
    private final CourseMapper courseMapper;
    private final CourseStudentMapper courseStudentMapper;
    private final TeacherMapper teacherMapper;
    private final CourseEnrollmentRequestMapper courseEnrollmentRequestMapper;
    private final PasswordEncoder passwordEncoder;
    private final SecurityUtil securityUtil;

    @Override
    public PageResponse<Student> getStudents(int page, int size, String keyword, Long classId, Long courseId) {
        // 获取当前教师ID
        Long teacherId = securityUtil.getCurrentUserId();
        
        // 查询分页数据
        Page<Student> pageParam = new Page<>(page + 1, size); // 后端是1-based索引
        IPage<Student> result;
        
        if (classId != null) {
            // 按班级查询
            CourseClass courseClass = courseClassMapper.selectById(classId);
            if (courseClass == null || !teacherId.equals(courseClass.getTeacherId())) {
                throw new BusinessException(ResultCode.FORBIDDEN, "无权操作该班级");
            }
            
            result = studentMapper.selectPageByClassId(pageParam, classId, keyword);
        } else if (courseId != null) {
            // 按课程查询
            Course course = courseMapper.selectById(courseId);
            if (course == null) {
                throw new BusinessException(ResultCode.DATA_NOT_FOUND, "课程不存在");
            }
            
            // 检查教师是否有权限操作该课程
            Long courseTeacherId = course.getTeacherId();
            Teacher teacher = teacherMapper.selectByUserId(teacherId);
            if (teacher == null || !teacher.getId().equals(courseTeacherId)) {
                throw new BusinessException(ResultCode.FORBIDDEN, "无权操作该课程");
            }
            
            result = studentMapper.selectPageByCourseId(pageParam, courseId, keyword);
        } else {
            // 查询所有学生（仅限教师有权限的班级和课程的学生）
            Teacher teacher = teacherMapper.selectByUserId(teacherId);
            if (teacher == null) {
                throw new BusinessException(ResultCode.FORBIDDEN, "无教师权限");
            }
            
            result = studentMapper.selectPageByTeacherId(pageParam, teacher.getId(), keyword);
        }
        
        // 填充用户信息
        for (Student student : result.getRecords()) {
            User user = userMapper.selectById(student.getUserId());
            if (user != null) {
                student.setUser(user);
            }
        }
        
        return PageResponse.of(result);
    }

    @Override
    public List<Map<String, Object>> searchStudents(String keyword) {
        // 创建返回结果列表
        List<Map<String, Object>> result = new ArrayList<>();
        
        try {
            if (StringUtils.hasText(keyword)) {
                // 方法1：先从用户表中查找真实姓名匹配的STUDENT角色用户
                LambdaQueryWrapper<User> userWrapper = new LambdaQueryWrapper<>();
                userWrapper.eq(User::getRole, "STUDENT")
                          .like(User::getRealName, keyword);
                List<User> users = userMapper.selectList(userWrapper);
                
                // 收集用户ID
                List<Long> userIds = users.stream().map(User::getId).collect(Collectors.toList());
                Map<Long, User> userMap = users.stream().collect(Collectors.toMap(User::getId, user -> user));
                
                // 如果通过真实姓名找到了学生用户
                if (!userIds.isEmpty()) {
                    // 查询对应的学生信息
                    LambdaQueryWrapper<Student> studentWrapperByUser = new LambdaQueryWrapper<>();
                    studentWrapperByUser.in(Student::getUserId, userIds);
                    List<Student> students = studentMapper.selectList(studentWrapperByUser);
                    
                    // 组装结果
                    for (Student student : students) {
                        User user = userMap.get(student.getUserId());
                        if (user != null) {
                            Map<String, Object> studentInfo = new HashMap<>();
                            studentInfo.put("id", student.getId());
                            studentInfo.put("userId", student.getUserId());
                            studentInfo.put("studentId", student.getId());
                            studentInfo.put("realName", user.getRealName());
                            result.add(studentInfo);
                        }
                    }
                }
                
                // 方法2：从学生表中查找学号(id)匹配的学生
                LambdaQueryWrapper<Student> studentWrapperById = new LambdaQueryWrapper<>();
                studentWrapperById.like(Student::getId, keyword);
                List<Student> studentsByNumber = studentMapper.selectList(studentWrapperById);
                
                if (!studentsByNumber.isEmpty()) {
                    // 收集这些学生的用户ID
                    List<Long> studentUserIds = studentsByNumber.stream()
                                                              .map(Student::getUserId)
                                                              .filter(id -> !userIds.contains(id)) // 排除已通过姓名查询到的
                                                              .collect(Collectors.toList());
                    
                    if (!studentUserIds.isEmpty()) {
                        // 查询对应的用户信息
                        LambdaQueryWrapper<User> studentUserWrapper = new LambdaQueryWrapper<>();
                        studentUserWrapper.in(User::getId, studentUserIds);
                        List<User> studentUsers = userMapper.selectList(studentUserWrapper);
                        Map<Long, User> studentUserMap = studentUsers.stream().collect(Collectors.toMap(User::getId, user -> user));
                        
                        // 添加通过学号找到的学生到结果
                        for (Student student : studentsByNumber) {
                            if (!userIds.contains(student.getUserId())) { // 避免重复添加
                                User user = studentUserMap.get(student.getUserId());
                                if (user != null) {
                                    Map<String, Object> studentInfo = new HashMap<>();
                                    studentInfo.put("id", student.getId());
                                    studentInfo.put("userId", student.getUserId());
                                    studentInfo.put("studentId", student.getId());
                                    studentInfo.put("realName", user.getRealName());
                                    result.add(studentInfo);
                                }
                            }
                        }
                    }
                }
            } else {
                // 如果没有关键词，返回所有学生(限制最多50条)
                LambdaQueryWrapper<User> userWrapper = new LambdaQueryWrapper<>();
                userWrapper.eq(User::getRole, "STUDENT").last("LIMIT 50");
                List<User> users = userMapper.selectList(userWrapper);
                
                // 收集用户ID
                List<Long> userIds = users.stream().map(User::getId).collect(Collectors.toList());
                
                if (!userIds.isEmpty()) {
                    // 查询对应的学生信息
                    LambdaQueryWrapper<Student> studentWrapper = new LambdaQueryWrapper<>();
                    studentWrapper.in(Student::getUserId, userIds);
                    List<Student> students = studentMapper.selectList(studentWrapper);
                    Map<Long, Student> studentMap = students.stream().collect(Collectors.toMap(Student::getUserId, student -> student, (s1, s2) -> s1));
                    
                    // 组装结果
                    for (User user : users) {
                        Student student = studentMap.get(user.getId());
                        if (student != null) {
                            Map<String, Object> studentInfo = new HashMap<>();
                            studentInfo.put("id", student.getId());
                            studentInfo.put("userId", student.getUserId());
                            studentInfo.put("studentId", student.getId());
                            studentInfo.put("realName", user.getRealName());
                            result.add(studentInfo);
                        }
                    }
                }
            }
        } catch (Exception e) {
            log.error("搜索学生失败", e);
        }
        
        return result;
    }

    @Override
    public Student getStudentById(Long id) {
        Student student = studentMapper.selectById(id);
        if (student == null) {
            throw new BusinessException(ResultCode.DATA_NOT_FOUND, "学生不存在");
        }
        
        // 填充用户信息
        User user = userMapper.selectById(student.getUserId());
        if (user != null) {
            student.setUser(user);
        }
        
        return student;
    }

    @Override
    public List<CourseClass> getClassesByTeacherId(Long teacherId) {
        return courseClassMapper.selectListByTeacherId(teacherId);
    }

    @Override
    @Transactional
    public void addStudentToClass(Long studentId, Long classId) {
        // 检查学生是否存在
        Student student = studentMapper.selectById(studentId);
        if (student == null) {
            throw new BusinessException(ResultCode.DATA_NOT_FOUND, "学生不存在");
        }
        
        // 检查班级是否存在
        CourseClass courseClass = courseClassMapper.selectById(classId);
        if (courseClass == null) {
            throw new BusinessException(ResultCode.DATA_NOT_FOUND, "班级不存在");
        }
        
        // 检查权限
        Long currentUserId = securityUtil.getCurrentUserId();
        if (!currentUserId.equals(courseClass.getTeacherId())) {
            throw new BusinessException(ResultCode.FORBIDDEN, "无权操作该班级");
        }
        
        // 检查学生是否已在班级中
        LambdaQueryWrapper<ClassStudent> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(ClassStudent::getClassId, classId)
               .eq(ClassStudent::getStudentId, studentId);
        if (classStudentMapper.selectCount(wrapper) > 0) {
            throw new BusinessException(ResultCode.BUSINESS_ERROR, "学生已在班级中");
        }
        
        // 添加学生到班级
        ClassStudent classStudent = new ClassStudent();
        classStudent.setClassId(classId);
        classStudent.setStudentId(studentId);
        classStudentMapper.insert(classStudent);
        
        // 如果班级绑定了课程，同时将学生添加到课程
        if (courseClass.getCourseId() != null) {
            try {
                addStudentToCourse(studentId, courseClass.getCourseId());
            } catch (Exception e) {
                log.warn("将学生添加到课程时出错", e);
                // 不影响主流程
            }
        }
    }

    @Override
    @Transactional
    public void removeStudentFromClass(Long studentId, Long classId) {
        // 检查班级是否存在
        CourseClass courseClass = courseClassMapper.selectById(classId);
        if (courseClass == null) {
            throw new BusinessException(ResultCode.DATA_NOT_FOUND, "班级不存在");
        }
        
        // 检查权限
        Long currentUserId = securityUtil.getCurrentUserId();
        if (!currentUserId.equals(courseClass.getTeacherId())) {
            throw new BusinessException(ResultCode.FORBIDDEN, "无权操作该班级");
        }
        
        // 删除班级学生关联
        LambdaQueryWrapper<ClassStudent> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(ClassStudent::getClassId, classId)
               .eq(ClassStudent::getStudentId, studentId);
        classStudentMapper.delete(wrapper);
    }

    @Override
    @Transactional
    public void addStudentToCourse(Long studentId, Long courseId) {
        // 检查学生是否存在
        Student student = studentMapper.selectById(studentId);
        if (student == null) {
            throw new BusinessException(ResultCode.DATA_NOT_FOUND, "学生不存在");
        }
        
        // 检查课程是否存在
        Course course = courseMapper.selectById(courseId);
        if (course == null) {
            throw new BusinessException(ResultCode.DATA_NOT_FOUND, "课程不存在");
        }
        
        // 检查权限
        Long currentUserId = securityUtil.getCurrentUserId();
        Teacher teacher = teacherMapper.selectByUserId(currentUserId);
        if (teacher == null || !teacher.getId().equals(course.getTeacherId())) {
            throw new BusinessException(ResultCode.FORBIDDEN, "无权操作该课程");
        }
        
        // 检查学生是否已选课
        LambdaQueryWrapper<CourseStudent> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(CourseStudent::getCourseId, courseId)
               .eq(CourseStudent::getStudentId, studentId);
        if (courseStudentMapper.selectCount(wrapper) > 0) {
            throw new BusinessException(ResultCode.BUSINESS_ERROR, "学生已选该课程");
        }
        
        // 添加学生到课程
        CourseStudent courseStudent = new CourseStudent();
        courseStudent.setCourseId(courseId);
        courseStudent.setStudentId(studentId);
        courseStudentMapper.insert(courseStudent);
        
        // 更新课程学生数量
        course.setStudentCount(course.getStudentCount() + 1);
        courseMapper.updateById(course);
        
        // 查找课程的默认班级
        LambdaQueryWrapper<CourseClass> classWrapper = new LambdaQueryWrapper<>();
        classWrapper.eq(CourseClass::getCourseId, courseId)
                   .eq(CourseClass::getIsDefault, true);
        CourseClass defaultClass = courseClassMapper.selectOne(classWrapper);
        
        // 如果没有默认班级，查找教师创建的任意一个关联到该课程的班级
        if (defaultClass == null) {
            classWrapper = new LambdaQueryWrapper<>();
            classWrapper.eq(CourseClass::getCourseId, courseId)
                       .eq(CourseClass::getTeacherId, currentUserId);
            defaultClass = courseClassMapper.selectOne(classWrapper);
        }
        
        // 如果找到班级，将学生添加到班级中
        if (defaultClass != null) {
            try {
                // 检查学生是否已在班级
                LambdaQueryWrapper<ClassStudent> classStudentWrapper = new LambdaQueryWrapper<>();
                classStudentWrapper.eq(ClassStudent::getClassId, defaultClass.getId())
                                  .eq(ClassStudent::getStudentId, studentId);
                if (classStudentMapper.selectCount(classStudentWrapper) == 0) {
                    // 添加学生到班级
                    ClassStudent classStudent = new ClassStudent();
                    classStudent.setClassId(defaultClass.getId());
                    classStudent.setStudentId(studentId);
                    classStudentMapper.insert(classStudent);
                    log.info("学生 {} 已添加到班级 {}", studentId, defaultClass.getId());
                }
            } catch (Exception e) {
                log.warn("将学生添加到班级时出错", e);
                // 不影响主流程
            }
        } else {
            log.warn("未找到课程 {} 的默认班级或任何班级", courseId);
        }
    }

    @Override
    @Transactional
    public void removeStudentFromCourse(Long studentId, Long courseId) {
        // 检查课程是否存在
        Course course = courseMapper.selectById(courseId);
        if (course == null) {
            throw new BusinessException(ResultCode.DATA_NOT_FOUND, "课程不存在");
        }
        
        // 检查权限
        Long currentUserId = securityUtil.getCurrentUserId();
        Teacher teacher = teacherMapper.selectByUserId(currentUserId);
        if (teacher == null || !teacher.getId().equals(course.getTeacherId())) {
            throw new BusinessException(ResultCode.FORBIDDEN, "无权操作该课程");
        }
        
        // 删除课程学生关联
        LambdaQueryWrapper<CourseStudent> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(CourseStudent::getCourseId, courseId)
               .eq(CourseStudent::getStudentId, studentId);
        int count = courseStudentMapper.delete(wrapper);
        
        // 更新课程学生数量
        if (count > 0 && course.getStudentCount() > 0) {
            course.setStudentCount(course.getStudentCount() - 1);
            courseMapper.updateById(course);
        }
    }

    @Override
    @Transactional
    public void processEnrollmentRequest(Long requestId, Boolean approved, String comment) {
        // 检查选课申请是否存在
        CourseEnrollmentRequest request = courseEnrollmentRequestMapper.selectById(requestId);
        if (request == null) {
            throw new BusinessException(ResultCode.DATA_NOT_FOUND, "选课申请不存在");
        }
        
        // 检查课程是否存在
        Course course = courseMapper.selectById(request.getCourseId());
        if (course == null) {
            throw new BusinessException(ResultCode.DATA_NOT_FOUND, "课程不存在");
        }
        
        // 检查权限
        Long currentUserId = securityUtil.getCurrentUserId();
        Teacher teacher = teacherMapper.selectByUserId(currentUserId);
        if (teacher == null || !teacher.getId().equals(course.getTeacherId())) {
            throw new BusinessException(ResultCode.FORBIDDEN, "无权操作该课程");
        }
        
        // 更新申请状态
        request.setStatus(approved ? 1 : 2); // 1=已通过 2=已拒绝
        request.setReviewComment(comment);
        request.setReviewTime(LocalDateTime.now());
        courseEnrollmentRequestMapper.updateById(request);
        
        // 如果通过申请，将学生添加到课程
        if (approved) {
            try {
                addStudentToCourse(request.getStudentId(), request.getCourseId());
            } catch (BusinessException e) {
                if (e.getCode() == ResultCode.BUSINESS_ERROR.getCode() && "学生已选该课程".equals(e.getMessage())) {
                    // 学生已经在课程中，忽略错误
                    log.info("学生已在课程中，无需重复添加");
                } else {
                    throw e;
                }
            }
        }
    }

    @Override
    public PageResponse<Map<String, Object>> getEnrollmentRequests(int page, int size, Long courseId) {
        // 获取当前教师ID
        Long teacherId = securityUtil.getCurrentUserId();
        Teacher teacher = teacherMapper.selectByUserId(teacherId);
        if (teacher == null) {
            throw new BusinessException(ResultCode.FORBIDDEN, "无教师权限");
        }
        
        // 查询分页数据
        Page<CourseEnrollmentRequest> pageParam = new Page<>(page + 1, size); // 后端是1-based索引
        IPage<CourseEnrollmentRequest> result;
        
        if (courseId != null) {
            // 检查课程是否属于当前教师
            Course course = courseMapper.selectById(courseId);
            if (course == null || !teacher.getId().equals(course.getTeacherId())) {
                throw new BusinessException(ResultCode.FORBIDDEN, "无权操作该课程");
            }
            
            // 按课程查询
            LambdaQueryWrapper<CourseEnrollmentRequest> wrapper = new LambdaQueryWrapper<>();
            wrapper.eq(CourseEnrollmentRequest::getCourseId, courseId)
                   .eq(CourseEnrollmentRequest::getStatus, 0) // 0=待审核
                   .orderByDesc(CourseEnrollmentRequest::getSubmitTime);
            result = courseEnrollmentRequestMapper.selectPage(pageParam, wrapper);
        } else {
            // 查询教师所有课程的申请
            result = courseEnrollmentRequestMapper.selectPageByTeacherId(pageParam, teacher.getId());
        }
        
        // 转换为Map列表，包含学生和课程信息
        List<Map<String, Object>> records = new ArrayList<>();
        for (CourseEnrollmentRequest request : result.getRecords()) {
            Map<String, Object> item = new HashMap<>();
            item.put("id", request.getId());
            item.put("studentId", request.getStudentId());
            item.put("courseId", request.getCourseId());
            item.put("status", request.getStatus());
            item.put("reason", request.getReason());
            item.put("submitTime", request.getSubmitTime());
            
            // 填充学生信息
            Student student = studentMapper.selectById(request.getStudentId());
            if (student != null) {
                User user = userMapper.selectById(student.getUserId());
                Map<String, Object> studentInfo = new HashMap<>();
                studentInfo.put("id", student.getId());
                studentInfo.put("studentId", student.getStudentId());
                studentInfo.put("name", user != null ? user.getRealName() : "未知");
                item.put("student", studentInfo);
            }
            
            // 填充课程信息
            Course course = courseMapper.selectById(request.getCourseId());
            if (course != null) {
                Map<String, Object> courseInfo = new HashMap<>();
                courseInfo.put("id", course.getId());
                courseInfo.put("title", course.getTitle());
                item.put("course", courseInfo);
            }
            
            records.add(item);
        }
        
        // 构建分页响应
        PageResponse<Map<String, Object>> response = new PageResponse<>();
        response.setRecords(records);
        response.setTotal(result.getTotal());
        response.setCurrent((int)result.getCurrent());
        response.setSize((int)result.getSize());
        
        return response;
    }

    @Override
    @Transactional
    public User createStudent(User user) {
        // 检查用户名是否已存在
        LambdaQueryWrapper<User> userWrapper = new LambdaQueryWrapper<>();
        userWrapper.eq(User::getUsername, user.getUsername());
        if (userMapper.selectCount(userWrapper) > 0) {
            throw new BusinessException(ResultCode.BUSINESS_ERROR, "用户名已存在");
        }
        
        // 检查邮箱是否已存在
        if (StringUtils.hasText(user.getEmail())) {
            LambdaQueryWrapper<User> emailWrapper = new LambdaQueryWrapper<>();
            emailWrapper.eq(User::getEmail, user.getEmail());
            if (userMapper.selectCount(emailWrapper) > 0) {
                throw new BusinessException(ResultCode.BUSINESS_ERROR, "邮箱已存在");
            }
        }
        
        // 设置角色为学生
        user.setRole(User.Role.STUDENT.getCode());
        user.setStatus(User.Status.ACTIVE.getCode());
        
        // 加密密码
        if (StringUtils.hasText(user.getPassword())) {
            user.setPassword(passwordEncoder.encode(user.getPassword()));
        } else {
            // 默认密码为123456
            user.setPassword(passwordEncoder.encode("123456"));
        }
        
        // 保存用户
        userMapper.insert(user);
        
        // 创建学生记录
        Student student = new Student();
        student.setUserId(user.getId());
        student.setStudentId(user.getUsername()); // 默认学号与用户名相同
        student.setEnrollmentStatus(Student.EnrollmentStatus.ENROLLED.getCode());
        studentMapper.insert(student);
        
        return user;
    }

    @Override
    @Transactional
    public Student updateStudent(Student student) {
        // 检查学生是否存在
        Student existingStudent = studentMapper.selectById(student.getId());
        if (existingStudent == null) {
            throw new BusinessException(ResultCode.DATA_NOT_FOUND, "学生不存在");
        }
        
        // 更新学生信息
        studentMapper.updateById(student);
        
        // 如果提供了用户信息，更新用户信息
        if (student.getUser() != null) {
            User user = student.getUser();
            user.setId(existingStudent.getUserId());
            
            // 不更新敏感字段
            user.setUsername(null);
            user.setPassword(null);
            user.setRole(null);
            
            userMapper.updateById(user);
        }
        
        return getStudentById(student.getId());
    }

    @Override
    public List<Map<String, Object>> getStudentClasses(Long studentId) {
        log.info("获取学生班级信息, 学生ID: {}", studentId);
        
        // 检查学生是否存在
        Student student = studentMapper.selectById(studentId);
        if (student == null) {
            log.error("学生不存在, ID: {}", studentId);
            throw new BusinessException(ResultCode.DATA_NOT_FOUND, "学生不存在");
        }
        
        List<Map<String, Object>> result = new ArrayList<>();
        
        try {
            // 查询学生所属的班级关联记录
            LambdaQueryWrapper<ClassStudent> wrapper = new LambdaQueryWrapper<>();
            wrapper.eq(ClassStudent::getStudentId, studentId);
            List<ClassStudent> classStudents = classStudentMapper.selectList(wrapper);
            
            if (classStudents.isEmpty()) {
                log.info("学生未加入任何班级, 学生ID: {}", studentId);
                return result;
            }
            
            // 收集班级ID
            List<Long> classIds = classStudents.stream()
                                            .map(ClassStudent::getClassId)
                                            .collect(Collectors.toList());
            
            // 查询班级信息
            LambdaQueryWrapper<CourseClass> classWrapper = new LambdaQueryWrapper<>();
            classWrapper.in(CourseClass::getId, classIds);
            List<CourseClass> classes = courseClassMapper.selectList(classWrapper);
            Map<Long, CourseClass> classMap = classes.stream()
                                                .collect(Collectors.toMap(CourseClass::getId, cls -> cls));
            
            // 收集课程ID
            List<Long> courseIds = classes.stream()
                                        .map(CourseClass::getCourseId)
                                        .filter(id -> id != null)
                                        .distinct()
                                        .collect(Collectors.toList());
            
            // 查询课程信息
            Map<Long, Course> courseMap = new HashMap<>();
            if (!courseIds.isEmpty()) {
                LambdaQueryWrapper<Course> courseWrapper = new LambdaQueryWrapper<>();
                courseWrapper.in(Course::getId, courseIds);
                List<Course> courses = courseMapper.selectList(courseWrapper);
                courseMap = courses.stream().collect(Collectors.toMap(Course::getId, course -> course));
            }
            
            // 组装结果
            for (ClassStudent classStudent : classStudents) {
                CourseClass courseClass = classMap.get(classStudent.getClassId());
                if (courseClass == null) {
                    continue;
                }
                
                Map<String, Object> classInfo = new HashMap<>();
                classInfo.put("id", courseClass.getId());
                classInfo.put("name", courseClass.getName());
                classInfo.put("description", courseClass.getDescription());
                classInfo.put("teacherId", courseClass.getTeacherId());
                classInfo.put("joinTime", classStudent.getJoinTime());
                
                // 添加课程信息
                if (courseClass.getCourseId() != null) {
                    Course course = courseMap.get(courseClass.getCourseId());
                    if (course != null) {
                        Map<String, Object> courseInfo = new HashMap<>();
                        courseInfo.put("id", course.getId());
                        courseInfo.put("title", course.getTitle());
                        courseInfo.put("courseType", course.getCourseType());
                        courseInfo.put("term", course.getTerm());
                        classInfo.put("course", courseInfo);
                    }
                }
                
                result.add(classInfo);
            }
            
            log.info("获取到学生班级信息 {} 条记录, 学生ID: {}", result.size(), studentId);
        } catch (Exception e) {
            log.error("获取学生班级信息失败, 学生ID: {}", studentId, e);
        }
        
        return result;
    }
    
    @Override
    public List<Map<String, Object>> getStudentCourses(Long studentId) {
        log.info("获取学生课程信息, 学生ID: {}", studentId);
        
        // 检查学生是否存在
        Student student = studentMapper.selectById(studentId);
        if (student == null) {
            log.error("学生不存在, ID: {}", studentId);
            throw new BusinessException(ResultCode.DATA_NOT_FOUND, "学生不存在");
        }
        
        List<Map<String, Object>> result = new ArrayList<>();
        
        try {
            // 查询学生所属的课程关联记录
            LambdaQueryWrapper<CourseStudent> wrapper = new LambdaQueryWrapper<>();
            wrapper.eq(CourseStudent::getStudentId, studentId);
            List<CourseStudent> courseStudents = courseStudentMapper.selectList(wrapper);
            
            if (courseStudents.isEmpty()) {
                log.info("学生未选修任何课程, 学生ID: {}", studentId);
                return result;
            }
            
            // 收集课程ID
            List<Long> courseIds = courseStudents.stream()
                                             .map(CourseStudent::getCourseId)
                                             .collect(Collectors.toList());
            
            // 查询课程信息
            LambdaQueryWrapper<Course> courseWrapper = new LambdaQueryWrapper<>();
            courseWrapper.in(Course::getId, courseIds);
            List<Course> courses = courseMapper.selectList(courseWrapper);
            Map<Long, Course> courseMap = courses.stream()
                                              .collect(Collectors.toMap(Course::getId, course -> course));
            
            // 组装结果
            for (CourseStudent courseStudent : courseStudents) {
                Course course = courseMap.get(courseStudent.getCourseId());
                if (course == null) {
                    continue;
                }
                
                Map<String, Object> courseInfo = new HashMap<>();
                courseInfo.put("id", course.getId());
                courseInfo.put("title", course.getTitle());
                courseInfo.put("courseType", course.getCourseType());
                courseInfo.put("term", course.getTerm());
                courseInfo.put("description", course.getDescription());
                courseInfo.put("enrollTime", courseStudent.getEnrollTime());
                
                // 获取教师信息
                if (course.getTeacherId() != null) {
                    Teacher teacher = teacherMapper.selectById(course.getTeacherId());
                    if (teacher != null && teacher.getUserId() != null) {
                        User teacherUser = userMapper.selectById(teacher.getUserId());
                        if (teacherUser != null) {
                            courseInfo.put("teacherName", teacherUser.getRealName());
                        }
                    }
                }
                
                result.add(courseInfo);
            }
            
            log.info("获取到学生课程信息 {} 条记录, 学生ID: {}", result.size(), studentId);
        } catch (Exception e) {
            log.error("获取学生课程信息失败, 学生ID: {}", studentId, e);
        }
        
        return result;
    }
} 