package com.education.service.teacher.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.education.dto.common.PageResponse;
import com.education.entity.ClassStudent;
import com.education.entity.Course;
import com.education.entity.CourseClass;
import com.education.entity.Student;
import com.education.entity.Teacher;
import com.education.exception.BusinessException;
import com.education.exception.ResultCode;
import com.education.mapper.ClassStudentMapper;
import com.education.mapper.CourseClassMapper;
import com.education.mapper.CourseMapper;
import com.education.mapper.StudentMapper;
import com.education.mapper.TeacherMapper;
import com.education.security.SecurityUtil;
import com.education.service.teacher.ClassService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.stream.Collectors;

/**
 * 班级管理服务实现类
 */
@Service
@Slf4j
@RequiredArgsConstructor
public class ClassServiceImpl implements ClassService {

    private final CourseClassMapper courseClassMapper;
    private final CourseMapper courseMapper;
    private final StudentMapper studentMapper;
    private final ClassStudentMapper classStudentMapper;
    private final TeacherMapper teacherMapper;
    private final SecurityUtil securityUtil;

    @Override
    public PageResponse<CourseClass> getClassesByTeacher(int page, int size, String keyword, Long courseId) {
        // 获取当前登录教师ID
        Long teacherId = securityUtil.getCurrentUserId();
        
        // 查询分页数据
        Page<CourseClass> pageParam = new Page<>(page + 1, size); // 后端是1-based索引
        IPage<CourseClass> result = courseClassMapper.selectPageByTeacherId(pageParam, teacherId, keyword, courseId);
        
        // 直接使用IPage转换
        return PageResponse.of(result);
    }

    @Override
    public CourseClass getClassById(Long id) {
        // 获取班级信息
        CourseClass courseClass = courseClassMapper.selectById(id);
        if (courseClass == null) {
            throw new BusinessException(ResultCode.DATA_NOT_FOUND, "班级不存在");
        }
        
        // 检查权限
        checkPermission(courseClass);
        
        // 获取关联的课程信息
        if (courseClass.getCourseId() != null) {
            Course course = courseMapper.selectById(courseClass.getCourseId());
            courseClass.setCourse(course);
        }
        
        // 获取班级学生数量
        Integer studentCount = classStudentMapper.countByClassId(id);
        courseClass.setStudentCount(studentCount);
        
        return courseClass;
    }

    @Override
    @Transactional
    public CourseClass createClass(CourseClass courseClass) {
        // 设置当前教师ID
        Long teacherId = securityUtil.getCurrentUserId();
        courseClass.setTeacherId(teacherId);
        
        // 如果设置了课程ID，检查课程是否存在
        if (courseClass.getCourseId() != null) {
            Course course = courseMapper.selectById(courseClass.getCourseId());
            if (course == null) {
                throw new BusinessException(ResultCode.PARAM_ERROR, "所选课程不存在");
            }
            
            // 检查课程是否属于当前教师
            if (!teacherId.equals(course.getTeacherId())) {
                throw new BusinessException(ResultCode.FORBIDDEN, "无权操作该课程");
            }
            
            // 如果设置为默认班级，需要取消其他班级的默认设置
            if (Boolean.TRUE.equals(courseClass.getIsDefault())) {
                CourseClass defaultClass = courseClassMapper.selectDefaultClassByCourseId(courseClass.getCourseId());
                if (defaultClass != null) {
                    defaultClass.setIsDefault(false);
                    courseClassMapper.updateById(defaultClass);
                }
            }
        } else {
            // 如果没有指定课程ID，确保isDefault为false
            courseClass.setIsDefault(false);
        }
        
        try {
            // 插入班级记录
            courseClassMapper.insertWithNullCourseId(courseClass);
        } catch (Exception e) {
            log.error("创建班级失败", e);
            throw new BusinessException(ResultCode.SYSTEM_ERROR, "创建班级失败：" + e.getMessage());
        }
        
        return courseClass;
    }

    @Override
    @Transactional
    public CourseClass updateClass(CourseClass courseClass) {
        // 获取原班级信息
        CourseClass existingClass = courseClassMapper.selectById(courseClass.getId());
        if (existingClass == null) {
            log.warn("更新班级失败：班级不存在，ID: {}", courseClass.getId());
            throw new BusinessException(ResultCode.DATA_NOT_FOUND, "班级不存在");
        }
        
        // 记录原始信息和更新信息
        log.info("更新班级开始，班级ID: {}, 原课程ID: {}, 新课程ID: {}", 
            courseClass.getId(), existingClass.getCourseId(), courseClass.getCourseId());
        
        // 获取当前用户ID
        Long currentUserId = securityUtil.getCurrentUserId();
        log.info("当前用户ID: {}, 班级所属教师ID: {}", currentUserId, existingClass.getTeacherId());
        
        // 检查权限 - 直接比较用户ID和班级的teacherId
        if (!currentUserId.equals(existingClass.getTeacherId())) {
            log.warn("权限检查失败：当前用户ID: {}, 班级所属教师ID: {}", currentUserId, existingClass.getTeacherId());
            throw new BusinessException(ResultCode.FORBIDDEN, "无权限");
        }
        
        // 如果修改了课程绑定
        if (courseClass.getCourseId() != null && !courseClass.getCourseId().equals(existingClass.getCourseId())) {
            log.info("检测到课程ID变更，从 {} 变为 {}", existingClass.getCourseId(), courseClass.getCourseId());
            
            // 检查课程是否存在
            Course course = courseMapper.selectById(courseClass.getCourseId());
            if (course == null) {
                log.warn("更新班级失败：指定的课程不存在，课程ID: {}", courseClass.getCourseId());
                throw new BusinessException(ResultCode.PARAM_ERROR, "所选课程不存在");
            }
            
            // 查询用户对应的教师ID
            Long teacherId = getTeacherIdByUserId(currentUserId);
            log.info("当前用户ID {} 对应的教师ID: {}", currentUserId, teacherId);
            
            // 检查课程是否属于当前教师
            if (!teacherId.equals(course.getTeacherId())) {
                log.warn("更新班级失败：课程不属于当前教师，课程ID: {}, 课程教师ID: {}, 当前教师ID: {}", 
                    courseClass.getCourseId(), course.getTeacherId(), teacherId);
                throw new BusinessException(ResultCode.FORBIDDEN, "无权操作该课程");
            }
            
            log.info("课程绑定检查通过，课程ID: {}, 课程名称: {}", 
                course.getId(), course.getTitle() != null ? course.getTitle() : course.getCourseName());
            
            // 单独更新课程ID，确保能正确保存
            try {
                int updateResult = courseClassMapper.updateCourseId(courseClass.getId(), courseClass.getCourseId());
                log.info("单独更新课程ID结果: {}, 影响行数: {}", updateResult > 0 ? "成功" : "失败", updateResult);
            } catch (Exception e) {
                log.error("单独更新课程ID失败", e);
                throw new BusinessException(ResultCode.SYSTEM_ERROR, "更新课程ID失败：" + e.getMessage());
            }
        }
        
        // 如果设置为默认班级，需要取消其他班级的默认设置
        if (Boolean.TRUE.equals(courseClass.getIsDefault()) && 
            (existingClass.getIsDefault() == null || !existingClass.getIsDefault())) {
            CourseClass defaultClass = courseClassMapper.selectDefaultClassByCourseId(
                courseClass.getCourseId() != null ? courseClass.getCourseId() : existingClass.getCourseId());
            if (defaultClass != null && !defaultClass.getId().equals(courseClass.getId())) {
                defaultClass.setIsDefault(false);
                courseClassMapper.updateById(defaultClass);
                log.info("取消了课程ID: {}的原默认班级: {}", 
                    courseClass.getCourseId() != null ? courseClass.getCourseId() : existingClass.getCourseId(), 
                    defaultClass.getId());
            }
        }
        
        // 确保teacherId字段不会被更新
        courseClass.setTeacherId(existingClass.getTeacherId());
        
        try {
            // 更新班级信息
            log.info("准备更新班级信息，班级ID: {}, 班级名称: {}, 课程ID: {}", 
                courseClass.getId(), courseClass.getName(), courseClass.getCourseId());
            int result = courseClassMapper.updateById(courseClass);
            log.info("班级更新结果: {}, 影响行数: {}", result > 0 ? "成功" : "失败", result);
            
            // 验证更新是否成功
            CourseClass updatedClass = courseClassMapper.selectById(courseClass.getId());
            if (updatedClass != null) {
                log.info("验证更新结果 - 班级ID: {}, 课程ID: {}", updatedClass.getId(), updatedClass.getCourseId());
            }
        } catch (Exception e) {
            log.error("更新班级失败", e);
            throw new BusinessException(ResultCode.SYSTEM_ERROR, "更新班级失败：" + e.getMessage());
        }
        
        // 返回更新后的完整信息
        CourseClass updatedClass = getClassById(courseClass.getId());
        log.info("班级更新成功，ID: {}, 课程ID: {}", updatedClass.getId(), updatedClass.getCourseId());
        return updatedClass;
    }

    @Override
    @Transactional
    public void deleteClass(Long id) {
        // 获取班级信息
        CourseClass courseClass = courseClassMapper.selectById(id);
        if (courseClass == null) {
            throw new BusinessException(ResultCode.DATA_NOT_FOUND, "班级不存在");
        }
        
        // 检查权限
        checkPermission(courseClass);
        
        // 删除班级关联的学生记录
        LambdaQueryWrapper<ClassStudent> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(ClassStudent::getClassId, id);
        classStudentMapper.delete(wrapper);
        
        // 删除班级
        courseClassMapper.deleteById(id);
    }

    @Override
    public PageResponse<Student> getStudentsByClassId(Long classId, int page, int size, String keyword) {
        // 获取班级信息
        CourseClass courseClass = courseClassMapper.selectById(classId);
        if (courseClass == null) {
            throw new BusinessException(ResultCode.DATA_NOT_FOUND, "班级不存在");
        }
        
        // 检查权限
        checkPermission(courseClass);
        
        // 查询分页数据
        Page<Student> pageParam = new Page<>(page + 1, size); // 后端是1-based索引
        IPage<Student> result = studentMapper.selectPageByClassId(pageParam, classId, keyword);
        
        // 直接使用IPage转换
        return PageResponse.of(result);
    }

    @Override
    @Transactional
    public void addStudentsToClass(Long classId, List<Long> studentIds) {
        if (studentIds == null || studentIds.isEmpty()) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "学生ID列表不能为空");
        }
        
        // 获取班级信息
        CourseClass courseClass = courseClassMapper.selectById(classId);
        if (courseClass == null) {
            throw new BusinessException(ResultCode.DATA_NOT_FOUND, "班级不存在");
        }
        
        // 检查权限
        checkPermission(courseClass);
        
        // 批量添加学生到班级
        List<ClassStudent> classStudents = studentIds.stream()
            .map(studentId -> {
                // 检查学生是否已经在班级中
                LambdaQueryWrapper<ClassStudent> wrapper = new LambdaQueryWrapper<>();
                wrapper.eq(ClassStudent::getClassId, classId)
                       .eq(ClassStudent::getStudentId, studentId);
                if (classStudentMapper.selectCount(wrapper) > 0) {
                    return null; // 已存在，跳过
                }
                
                // 创建新关联
                ClassStudent classStudent = new ClassStudent();
                classStudent.setClassId(classId);
                classStudent.setStudentId(studentId);
                return classStudent;
            })
            .filter(cs -> cs != null) // 过滤掉已存在的
            .collect(Collectors.toList());
        
        // 批量插入
        if (!classStudents.isEmpty()) {
            for (ClassStudent cs : classStudents) {
                classStudentMapper.insert(cs);
            }
        }
    }

    @Override
    @Transactional
    public void removeStudentFromClass(Long classId, Long studentId) {
        // 获取班级信息
        CourseClass courseClass = courseClassMapper.selectById(classId);
        if (courseClass == null) {
            throw new BusinessException(ResultCode.DATA_NOT_FOUND, "班级不存在");
        }
        
        // 检查权限
        checkPermission(courseClass);
        
        // 删除关联
        LambdaQueryWrapper<ClassStudent> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(ClassStudent::getClassId, classId)
               .eq(ClassStudent::getStudentId, studentId);
        classStudentMapper.delete(wrapper);
    }
    
    /**
     * 检查当前用户是否有权限操作班级
     */
    private void checkPermission(CourseClass courseClass) {
        Long currentUserId = securityUtil.getCurrentUserId();
        if (!currentUserId.equals(courseClass.getTeacherId())) {
            log.warn("权限检查失败：当前用户ID: {}, 班级所属教师ID: {}", currentUserId, courseClass.getTeacherId());
            throw new BusinessException(ResultCode.FORBIDDEN, "无权操作该班级");
        }
    }

    /**
     * 根据用户ID查询教师ID
     * 
     * @param userId 用户ID
     * @return 教师ID，如果不存在则返回null
     */
    @Override
    public Long getTeacherIdByUserId(Long userId) {
        if (userId == null) {
            log.warn("传入的用户ID为null");
            return null;
        }
        
        log.info("查询用户ID为{}的教师信息", userId);
        
        // 使用TeacherMapper查询教师信息
        Teacher teacher = teacherMapper.selectByUserId(userId);
        
        if (teacher == null) {
            log.warn("未找到用户ID为{}的教师信息", userId);
            return null;
        }
        
        log.info("查询到用户ID为{}的教师ID为{}", userId, teacher.getId());
        return teacher.getId();
    }
} 