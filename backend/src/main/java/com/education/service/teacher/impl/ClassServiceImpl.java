package com.education.service.teacher.impl;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.education.dto.ClassDTO;
import com.education.dto.common.PageRequest;
import com.education.dto.common.PageResponse;
import com.education.entity.Class;
import com.education.entity.Student;
import com.education.entity.User;
import com.education.mapper.ClassMapper;
import com.education.mapper.StudentMapper;
import com.education.mapper.UserMapper;
import com.education.service.teacher.ClassService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.BeanUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.stream.Collectors;

/**
 * 教师端班级服务实现类
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
@Slf4j
@Service
public class ClassServiceImpl implements ClassService {

    @Autowired
    private ClassMapper classMapper;
    
    @Autowired
    private UserMapper userMapper;
    
    @Autowired
    private StudentMapper studentMapper;

    @Override
    public ClassDTO.ClassResponse createClass(ClassDTO.ClassCreateRequest createRequest, Long teacherId) {
        // 验证教师权限
        User teacher = userMapper.selectById(teacherId);
        if (teacher == null || !"TEACHER".equals(teacher.getRole())) {
            throw new RuntimeException("无效的教师ID");
        }
        
        // 检查班级代码是否已存在
        if (createRequest.getClassCode() != null) {
            Class existingClass = classMapper.selectByClassCode(createRequest.getClassCode());
            if (existingClass != null) {
                throw new RuntimeException("班级代码已存在");
            }
        }
        
        // 创建班级实体
        Class classEntity = new Class();
        classEntity.setClassName(createRequest.getClassName());
        classEntity.setDescription(createRequest.getDescription());
        classEntity.setHeadTeacherId(teacherId);
        classEntity.setClassCode(createRequest.getClassCode() != null ? 
            createRequest.getClassCode() : generateClassCode());
        classEntity.setMaxStudentCount(createRequest.getMaxStudents() != null ? 
            createRequest.getMaxStudents() : 50);
        classEntity.setStudentCount(0);
        classEntity.setStatus(createRequest.getIsActive() != null && createRequest.getIsActive() ? 
            "ACTIVE" : "INACTIVE");
        classEntity.setSemester(createRequest.getSemester());
        classEntity.setGrade(createRequest.getAcademicYear() != null ? 
            Integer.parseInt(createRequest.getAcademicYear()) : null);
        classEntity.setCreateTime(LocalDateTime.now());
        classEntity.setUpdateTime(LocalDateTime.now());
        
        // 保存到数据库
        int result = classMapper.insert(classEntity);
        if (result <= 0) {
            throw new RuntimeException("创建班级失败");
        }
        
        // 转换为响应DTO
        return convertToClassResponse(classEntity, teacher);
    }

    @Override
    public PageResponse<ClassDTO.ClassResponse> getClassList(Long teacherId, PageRequest pageRequest) {
        // 验证教师权限
        User teacher = userMapper.selectById(teacherId);
        if (teacher == null || !"TEACHER".equals(teacher.getRole())) {
            throw new RuntimeException("无效的教师ID");
        }
        
        // 设置分页参数
        Page<Class> page = new Page<>(pageRequest.getPageNum(), pageRequest.getPageSize());
        
        // 构建查询条件
        QueryWrapper<Class> queryWrapper = new QueryWrapper<>();
        queryWrapper.eq("head_teacher_id", teacherId)
                   .eq("is_deleted", false)
                   .orderByDesc("create_time");
        
        // 执行分页查询
        Page<Class> classPage = classMapper.selectPage(page, queryWrapper);
        
        // 转换为响应DTO列表
        List<ClassDTO.ClassResponse> responseList = classPage.getRecords().stream()
            .map(classEntity -> convertToClassResponse(classEntity, teacher))
            .collect(Collectors.toList());
        
        // 构建分页响应
        PageResponse<ClassDTO.ClassResponse> pageResponse = new PageResponse<>();
        pageResponse.setList(responseList);
        pageResponse.setTotal(classPage.getTotal());
        pageResponse.setPageNum(pageRequest.getPage());
        pageResponse.setPageSize(pageRequest.getSize());
        pageResponse.setPages((long) Math.ceil((double) classPage.getTotal() / pageRequest.getSize()));
        
        return pageResponse;
    }

    @Override
    public ClassDTO.ClassResponse getClassDetail(Long classId, Long teacherId) {
        // 验证教师权限
        User teacher = userMapper.selectById(teacherId);
        if (teacher == null || !"TEACHER".equals(teacher.getRole())) {
            throw new RuntimeException("无效的教师ID");
        }
        
        // 查询班级信息
        Class classEntity = classMapper.selectById(classId);
        if (classEntity == null) {
            throw new RuntimeException("班级不存在");
        }
        
        // 验证教师是否有权限访问该班级
        if (!classEntity.getHeadTeacherId().equals(teacherId)) {
            throw new RuntimeException("无权限访问该班级");
        }
        
        return convertToClassResponse(classEntity, teacher);
    }

    @Override
    public ClassDTO.ClassResponse updateClass(Long classId, ClassDTO.ClassUpdateRequest updateRequest, Long teacherId) {
        // 验证教师权限
        User teacher = userMapper.selectById(teacherId);
        if (teacher == null || !"TEACHER".equals(teacher.getRole())) {
            throw new RuntimeException("无效的教师ID");
        }
        
        // 查询班级信息
        Class classEntity = classMapper.selectById(classId);
        if (classEntity == null) {
            throw new RuntimeException("班级不存在");
        }
        
        // 验证教师是否有权限修改该班级
        if (!classEntity.getHeadTeacherId().equals(teacherId)) {
            throw new RuntimeException("无权限修改该班级");
        }
        
        // 更新班级信息
        if (updateRequest.getClassName() != null) {
            classEntity.setClassName(updateRequest.getClassName());
        }
        if (updateRequest.getDescription() != null) {
            classEntity.setDescription(updateRequest.getDescription());
        }
        if (updateRequest.getMaxStudents() != null) {
            classEntity.setMaxStudentCount(updateRequest.getMaxStudents());
        }
        if (updateRequest.getIsActive() != null) {
            classEntity.setStatus(updateRequest.getIsActive() ? "ACTIVE" : "INACTIVE");
        }
        if (updateRequest.getSemester() != null) {
            classEntity.setSemester(updateRequest.getSemester());
        }
        classEntity.setUpdateTime(LocalDateTime.now());
        
        // 保存更新
        int result = classMapper.updateById(classEntity);
        if (result <= 0) {
            throw new RuntimeException("更新班级失败");
        }
        
        return convertToClassResponse(classEntity, teacher);
    }

    @Override
    public Boolean deleteClass(Long classId, Long teacherId) {
        // 验证教师权限
        User teacher = userMapper.selectById(teacherId);
        if (teacher == null || !"TEACHER".equals(teacher.getRole())) {
            throw new RuntimeException("无效的教师ID");
        }
        
        // 查询班级信息
        Class classEntity = classMapper.selectById(classId);
        if (classEntity == null) {
            throw new RuntimeException("班级不存在");
        }
        
        // 验证教师是否有权限删除该班级
        if (!classEntity.getHeadTeacherId().equals(teacherId)) {
            throw new RuntimeException("无权限删除该班级");
        }
        
        // 检查班级是否有学生
        if (classEntity.getStudentCount() != null && classEntity.getStudentCount() > 0) {
            throw new RuntimeException("班级中还有学生，无法删除");
        }
        
        // 软删除班级
        classEntity.setIsDeleted(true);
        classEntity.setUpdateTime(LocalDateTime.now());
        
        int result = classMapper.updateById(classEntity);
        return result > 0;
    }



    @Override
    public PageResponse<Object> getClassStudents(Long classId, Long teacherId, PageRequest pageRequest) {
        log.info("获取班级学生列表，班级ID: {}, 教师ID: {}", classId, teacherId);
        
        // 验证班级和权限
        Class classEntity = validateClassAccess(classId, teacherId);
        
        // 构建分页对象
        Page<Student> page = new Page<>(pageRequest.getPageNum(), pageRequest.getPageSize());
        
        // 构建查询条件
        QueryWrapper<Student> wrapper = new QueryWrapper<>();
        wrapper.eq("class_id", classId)
               .eq("is_deleted", false)
               .orderByAsc("student_number");
        
        // 执行分页查询
        IPage<Student> studentPage = studentMapper.selectPage(page, wrapper);
        
        // 转换为响应对象
        List<ClassDTO.StudentResponse> studentResponses = studentPage.getRecords().stream()
                .map(this::convertToStudentResponse)
                .collect(Collectors.toList());
        
        return new PageResponse<Object>(
                studentPage.getCurrent(),
                studentPage.getSize(),
                studentPage.getTotal(),
                new ArrayList<Object>(studentResponses)
        );
    }

    @Override
    public ClassDTO.ClassStatisticsResponse getClassStatistics(Long classId, Long teacherId) {
        log.info("获取班级统计信息，班级ID: {}, 教师ID: {}", classId, teacherId);
        
        // 验证班级和权限
        Class classEntity = validateClassAccess(classId, teacherId);
        
        // 构建统计响应
        ClassDTO.ClassStatisticsResponse statistics = new ClassDTO.ClassStatisticsResponse();
        statistics.setClassId(classId);
        statistics.setClassName(classEntity.getClassName());
        statistics.setTotalStudents(classEntity.getStudentCount() != null ? classEntity.getStudentCount() : 0);
        statistics.setMaxStudents(classEntity.getMaxStudentCount());
        
        // 统计学生状态
        QueryWrapper<Student> wrapper = new QueryWrapper<>();
        wrapper.eq("class_id", classId).eq("is_deleted", false);
        List<Student> students = studentMapper.selectList(wrapper);
        
        statistics.setActiveStudents((int) students.stream().filter(s -> "ACTIVE".equals(s.getStatus())).count());
        statistics.setInactiveStudents(statistics.getTotalStudents() - statistics.getActiveStudents());
        
        // TODO: 添加更多统计信息，如平均成绩、作业完成率等
        statistics.setAverageGrade(0.0);
        statistics.setAssignmentCompletionRate(0.0);
        
        return statistics;
    }

    @Override
    public List<Object> getClassCourses(Long classId, Long teacherId) {
        // TODO: 实现获取班级课程列表逻辑
        return null;
    }

    @Override
    public PageResponse<Object> getClassTasks(Long classId, Long teacherId, PageRequest pageRequest) {
        // TODO: 实现获取班级任务列表逻辑
        return null;
    }

    @Override
    @Transactional
    public Boolean archiveClass(Long classId, Long teacherId) {
        log.info("归档班级，班级ID: {}, 教师ID: {}", classId, teacherId);
        
        return setClassStatus(classId, "ARCHIVED", teacherId);
    }



    @Override
    @Transactional
    public Boolean removeStudent(Long classId, Long studentId, Long teacherId) {
        log.info("移除学生，班级ID: {}, 学生ID: {}, 教师ID: {}", classId, studentId, teacherId);
        
        // 验证班级和权限
        Class classEntity = validateClassAccess(classId, teacherId);
        
        // 查询学生
        Student student = studentMapper.selectById(studentId);
        if (student == null || student.getIsDeleted()) {
            throw new RuntimeException("学生不存在");
        }
        
        // 验证学生是否在该班级
        if (!classId.equals(student.getClassId())) {
            throw new RuntimeException("学生不在该班级中");
        }
        
        // 移除学生（设置班级ID为空）
        student.setClassId(null);
        student.setUpdateTime(LocalDateTime.now());
        studentMapper.updateById(student);
        
        // 更新班级学生数量
        if (classEntity.getStudentCount() != null && classEntity.getStudentCount() > 0) {
            classEntity.setStudentCount(classEntity.getStudentCount() - 1);
            classEntity.setUpdateTime(LocalDateTime.now());
            classMapper.updateById(classEntity);
        }
        
        return true;
    }

    @Override
    @Transactional
    public Boolean removeStudents(Long classId, List<Long> studentIds, Long teacherId) {
        log.info("批量移除学生，班级ID: {}, 学生ID列表: {}, 教师ID: {}", classId, studentIds, teacherId);
        
        if (studentIds == null || studentIds.isEmpty()) {
            throw new RuntimeException("学生ID列表不能为空");
        }
        
        // 验证班级和权限
        Class classEntity = validateClassAccess(classId, teacherId);
        
        int removedCount = 0;
        for (Long studentId : studentIds) {
            try {
                // 查询学生
                Student student = studentMapper.selectById(studentId);
                if (student == null || student.getIsDeleted()) {
                    continue;
                }
                
                // 验证学生是否在该班级
                if (!classId.equals(student.getClassId())) {
                    continue;
                }
                
                // 移除学生
                student.setClassId(null);
                student.setUpdateTime(LocalDateTime.now());
                studentMapper.updateById(student);
                removedCount++;
                
            } catch (Exception e) {
                log.error("移除学生失败，学生ID: {}", studentId, e);
            }
        }
        
        // 更新班级学生数量
        if (removedCount > 0 && classEntity.getStudentCount() != null) {
            classEntity.setStudentCount(Math.max(0, classEntity.getStudentCount() - removedCount));
            classEntity.setUpdateTime(LocalDateTime.now());
            classMapper.updateById(classEntity);
        }
        
        return removedCount > 0;
    }

    @Override
    @Transactional
    public Object generateInviteCode(Long classId, Long teacherId, Integer expireHours) {
        log.info("生成邀请码，班级ID: {}, 教师ID: {}, 过期小时: {}", classId, teacherId, expireHours);
        
        // 验证班级和权限
        Class classEntity = validateClassAccess(classId, teacherId);
        
        // 生成邀请码
        String inviteCode = "INV" + System.currentTimeMillis() + UUID.randomUUID().toString().substring(0, 8).toUpperCase();
        
        // 计算过期时间
        LocalDateTime expireTime = LocalDateTime.now().plusHours(expireHours != null ? expireHours : 24);
        
        // TODO: 保存邀请码到数据库（需要创建邀请码实体和Mapper）
        
        // 构建响应
        ClassDTO.InviteCodeResponse response = new ClassDTO.InviteCodeResponse();
        response.setInviteCode(inviteCode);
        response.setClassId(classId);
        response.setClassName(classEntity.getClassName());
        response.setExpireTime(expireTime);
        response.setCreateTime(LocalDateTime.now());
        
        return response;
    }

    @Override
    public String exportClassStudents(Long classId, Long teacherId) {
        log.info("导出班级学生列表，班级ID: {}, 教师ID: {}", classId, teacherId);
        
        // 验证班级和权限
        Class classEntity = validateClassAccess(classId, teacherId);
        
        // 获取班级学生列表
        QueryWrapper<Student> queryWrapper = new QueryWrapper<>();
        queryWrapper.eq("class_id", classId)
                   .eq("is_deleted", false)
                   .orderByAsc("create_time");
        
        List<Student> students = studentMapper.selectList(queryWrapper);
        
        // TODO: 实现导出逻辑（Excel、CSV等）
        // 这里可以使用EasyExcel或Apache POI来实现导出功能
        // 暂时返回导出文件路径
        
        log.info("班级学生列表导出完成，共{}名学生", students.size());
        return "/exports/class_" + classId + "_students.xlsx";
    }

    @Override
    @Transactional
    public Boolean addStudents(Long classId, List<Long> studentIds, Long teacherId) {
        log.info("添加学生到班级，班级ID: {}, 学生ID列表: {}, 教师ID: {}", classId, studentIds, teacherId);
        
        if (studentIds == null || studentIds.isEmpty()) {
            throw new RuntimeException("学生ID列表不能为空");
        }
        
        // 验证班级和权限
        Class classEntity = validateClassAccess(classId, teacherId);
        
        // 检查班级容量
        int currentCount = classEntity.getStudentCount() != null ? classEntity.getStudentCount() : 0;
        if (currentCount + studentIds.size() > classEntity.getMaxStudentCount()) {
            throw new RuntimeException("班级人数已达上限");
        }
        
        int addedCount = 0;
        for (Long studentId : studentIds) {
            try {
                // 查询学生
                Student student = studentMapper.selectById(studentId);
                if (student == null || student.getIsDeleted()) {
                    continue;
                }
                
                // 检查学生是否已在其他班级
                if (student.getClassId() != null) {
                    continue;
                }
                
                // 添加学生到班级
                student.setClassId(classId);
                student.setUpdateTime(LocalDateTime.now());
                studentMapper.updateById(student);
                addedCount++;
                
            } catch (Exception e) {
                log.error("添加学生失败，学生ID: {}", studentId, e);
            }
        }
        
        // 更新班级学生数量
        if (addedCount > 0) {
            classEntity.setStudentCount(currentCount + addedCount);
            classEntity.setUpdateTime(LocalDateTime.now());
            classMapper.updateById(classEntity);
        }
        
        return addedCount > 0;
    }

    @Override
    @Transactional
    public Boolean joinClassByInviteCode(String inviteCode, Long studentId) {
        log.info("学生通过邀请码加入班级，邀请码: {}, 学生ID: {}", inviteCode, studentId);
        
        // 验证学生
        Student student = studentMapper.selectById(studentId);
        if (student == null || student.getIsDeleted()) {
            throw new RuntimeException("学生不存在");
        }
        
        // 检查学生是否已在班级中
        if (student.getClassId() != null) {
            throw new RuntimeException("学生已在其他班级中");
        }
        
        // TODO: 验证邀请码有效性和查找对应班级
        // 这里需要实现邀请码的存储和验证逻辑
        // 暂时返回false，等待邀请码表的实现
        
        return false;
    }

    @Override
    public List<Object> getInviteCodes(Long classId, Long teacherId) {
        log.info("获取班级邀请码列表，班级ID: {}, 教师ID: {}", classId, teacherId);
        
        // 验证班级和权限
        validateClassAccess(classId, teacherId);
        
        // TODO: 查询邀请码列表
        // 这里需要实现邀请码表的查询逻辑
        // 暂时返回空列表，等待邀请码表的实现
        
        return new ArrayList<>();
    }

    @Override
    @Transactional
    public Boolean disableInviteCode(Long inviteCodeId, Long teacherId) {
        log.info("禁用邀请码，邀请码ID: {}, 教师ID: {}", inviteCodeId, teacherId);
        
        // 验证教师权限
        User teacher = userMapper.selectById(teacherId);
        if (teacher == null || !"TEACHER".equals(teacher.getRole())) {
            throw new RuntimeException("无效的教师ID");
        }
        
        // TODO: 禁用邀请码
        // 这里需要实现邀请码的禁用逻辑
        // 暂时返回false，等待邀请码表的实现
        
        return false;
    }

    @Override
    @Transactional
    public Boolean setClassStatus(Long classId, String status, Long teacherId) {
        log.info("设置班级状态，班级ID: {}, 状态: {}, 教师ID: {}", classId, status, teacherId);
        
        // 验证班级和权限
        Class classEntity = validateClassAccess(classId, teacherId);
        
        // 验证状态值
        if (!"ACTIVE".equals(status) && !"INACTIVE".equals(status) && !"ARCHIVED".equals(status)) {
            throw new RuntimeException("无效的班级状态");
        }
        
        // 更新班级状态
        classEntity.setStatus(status);
        classEntity.setUpdateTime(LocalDateTime.now());
        
        int result = classMapper.updateById(classEntity);
        return result > 0;
    }

    @Override
    @Transactional
    public ClassDTO.ClassResponse copyClass(Long classId, String newClassName, Long teacherId) {
        log.info("复制班级，原班级ID: {}, 新班级名: {}, 教师ID: {}", classId, newClassName, teacherId);
        
        // 验证原班级和权限
        Class originalClass = validateClassAccess(classId, teacherId);
        
        // 创建新班级
        Class newClass = new Class();
        BeanUtils.copyProperties(originalClass, newClass);
        newClass.setId(null);
        newClass.setClassName(newClassName);
        newClass.setClassCode(generateClassCode());
        newClass.setStudentCount(0); // 新班级学生数量为0
        newClass.setCreateTime(LocalDateTime.now());
        newClass.setUpdateTime(LocalDateTime.now());
        
        // 保存新班级
        int result = classMapper.insert(newClass);
        if (result <= 0) {
            throw new RuntimeException("复制班级失败");
        }
        
        // 获取教师信息
        User teacher = userMapper.selectById(teacherId);
        
        return convertToClassResponse(newClass, teacher);
    }

    @Override
    public Boolean assignCourses(Long classId, List<Long> courseIds, Long teacherId) {
        // TODO: 实现分配课程逻辑
        return null;
    }

    @Override
    public Boolean removeCourse(Long classId, Long courseId, Long teacherId) {
        // TODO: 实现移除课程逻辑
        return null;
    }

    @Override
    public Object getClassGradeOverview(Long classId, Long teacherId) {
        // TODO: 实现获取班级成绩概览逻辑
        return null;
    }

    @Override
    public Boolean sendClassNotification(Long classId, String title, String content, Long teacherId) {
        // TODO: 实现发送班级通知逻辑
        return null;
    }

    @Override
    public PageResponse<Object> getClassNotifications(Long classId, Long teacherId, PageRequest pageRequest) {
        // TODO: 实现获取班级通知列表逻辑
        return null;
    }

    @Override
    @Transactional
    public Boolean restoreClass(Long classId, Long teacherId) {
        log.info("恢复归档班级，班级ID: {}, 教师ID: {}", classId, teacherId);
        
        return setClassStatus(classId, "ACTIVE", teacherId);
    }
    
    /**
     * 生成班级代码
     */
    private String generateClassCode() {
        return "CLS" + System.currentTimeMillis();
    }
    
    /**
     * 验证班级访问权限
     */
    private Class validateClassAccess(Long classId, Long teacherId) {
        // 验证教师权限
        User teacher = userMapper.selectById(teacherId);
        if (teacher == null || !"TEACHER".equals(teacher.getRole())) {
            throw new RuntimeException("无效的教师ID");
        }
        
        // 查询班级信息
        Class classEntity = classMapper.selectById(classId);
        if (classEntity == null || classEntity.getIsDeleted()) {
            throw new RuntimeException("班级不存在");
        }
        
        // 验证教师是否有权限访问该班级
        if (!classEntity.getHeadTeacherId().equals(teacherId)) {
            throw new RuntimeException("无权限访问该班级");
        }
        
        return classEntity;
    }
    
    /**
     * 转换学生实体为响应DTO
     */
    private ClassDTO.StudentResponse convertToStudentResponse(Student student) {
        ClassDTO.StudentResponse response = new ClassDTO.StudentResponse();
        BeanUtils.copyProperties(student, response);
        
        // 获取学生用户信息
        User user = userMapper.selectById(student.getUserId());
        if (user != null) {
            response.setRealName(user.getRealName());
            response.setEmail(user.getEmail());
        }
        
        return response;
    }
    
    /**
     * 转换班级实体为响应DTO
     */
    private ClassDTO.ClassResponse convertToClassResponse(Class classEntity, User teacher) {
        ClassDTO.ClassResponse response = new ClassDTO.ClassResponse();
        response.setClassId(classEntity.getId());
        response.setClassName(classEntity.getClassName());
        response.setDescription(classEntity.getDescription());
        response.setClassCode(classEntity.getClassCode());
        response.setMaxStudents(classEntity.getMaxStudentCount());
        response.setCurrentStudents(classEntity.getStudentCount());
        response.setIsActive("ACTIVE".equals(classEntity.getStatus()));
        response.setSemester(classEntity.getSemester());
        response.setAcademicYear(classEntity.getGrade() != null ? classEntity.getGrade().toString() : null);
        response.setCreateTime(classEntity.getCreateTime());
        response.setUpdateTime(classEntity.getUpdateTime());
        response.setTeacherId(teacher.getId());
        response.setTeacherName(teacher.getRealName());
        return response;
    }
    
}