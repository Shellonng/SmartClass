package com.education.service.teacher.impl;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.education.dto.ClassDTO;
import com.education.dto.common.PageRequest;
import com.education.dto.common.PageResponse;
import com.education.dto.clazz.*;
import com.education.entity.Class;
import com.education.entity.Student;
import com.education.entity.User;
import com.education.mapper.ClassMapper;
import com.education.mapper.StudentMapper;
import com.education.mapper.UserMapper;
import com.education.service.teacher.ClassService;
import com.education.exception.BusinessException;
import com.education.exception.ResultCode;
import com.education.utils.SecurityUtils;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.BeanUtils;
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
@RequiredArgsConstructor
public class ClassServiceImpl implements ClassService {

    private final ClassMapper classMapper;
    private final UserMapper userMapper;
    private final StudentMapper studentMapper;

    @Override
    public PageResponse<ClassResponse> getClassList(PageRequest pageRequest, String name, String grade, String status) {
        Long teacherId = SecurityUtils.getCurrentUserId();
        
        // 构建查询条件
        Long offset = (pageRequest.getCurrent() - 1) * pageRequest.getPageSize();
        List<Class> classes = classMapper.selectClassesByTeacherWithConditions(
                teacherId, name, grade, status, offset, pageRequest.getPageSize());
        
        Long total = classMapper.countClassesByTeacherWithConditions(teacherId, name, grade, status);
        
        List<ClassResponse> responses = classes.stream()
                .map(this::convertToClassResponse)
                .collect(Collectors.toList());
        
        return PageResponse.<ClassResponse>builder()
                .records(responses)
                .total(total)
                .current(pageRequest.getCurrent())
                .pageSize(pageRequest.getPageSize())
                .build();
    }

    @Override
    @Transactional
    public ClassResponse createClass(ClassCreateRequest request) {
        Long teacherId = SecurityUtils.getCurrentUserId();
        
        // 检查班级名称是否重复
        if (classMapper.existsByNameAndTeacherId(request.getName(), teacherId)) {
            throw new BusinessException(ResultCode.BUSINESS_ERROR, "班级名称已存在");
        }
        
        Class clazz = new Class();
        clazz.setName(request.getName());
        clazz.setGrade(request.getGrade());
        clazz.setMajor(request.getMajor());
        clazz.setDescription(request.getDescription());
        clazz.setCapacity(request.getCapacity());
        clazz.setSemester(request.getSemester());
        clazz.setTeacherId(teacherId);
        clazz.setStatus("ACTIVE");
        clazz.setCreateTime(LocalDateTime.now());
        clazz.setUpdateTime(LocalDateTime.now());
        
        classMapper.insert(clazz);
        
        log.info("班级创建成功，班级ID：{}，名称：{}", clazz.getId(), clazz.getName());
        return convertToClassResponse(clazz);
    }

    @Override
    public ClassDetailResponse getClassDetail(Long classId) {
        Long teacherId = SecurityUtils.getCurrentUserId();
        
        Class clazz = classMapper.selectByIdAndTeacherId(classId, teacherId);
        if (clazz == null) {
            throw new BusinessException(ResultCode.NOT_FOUND, "班级不存在");
        }
        
        ClassDetailResponse response = convertToClassDetailResponse(clazz);
        
        // 获取班级统计信息
        ClassDetailResponse.ClassStatistics statistics = getClassStatisticsData(classId);
        response.setStatistics(statistics);
        
        // 获取最近活动
        List<ClassDetailResponse.RecentActivity> activities = getRecentActivities(classId);
        response.setRecentActivities(activities);
        
        return response;
    }

    @Override
    @Transactional
    public ClassResponse updateClass(Long classId, ClassUpdateRequest request) {
        Long teacherId = SecurityUtils.getCurrentUserId();
        
        Class clazz = classMapper.selectByIdAndTeacherId(classId, teacherId);
        if (clazz == null) {
            throw new BusinessException(ResultCode.NOT_FOUND, "班级不存在");
        }
        
        // 更新字段
        if (request.getName() != null) {
            clazz.setName(request.getName());
        }
        if (request.getGrade() != null) {
            clazz.setGrade(request.getGrade());
        }
        if (request.getMajor() != null) {
            clazz.setMajor(request.getMajor());
        }
        if (request.getDescription() != null) {
            clazz.setDescription(request.getDescription());
        }
        if (request.getCapacity() != null) {
            clazz.setCapacity(request.getCapacity());
        }
        if (request.getSemester() != null) {
            clazz.setSemester(request.getSemester());
        }
        if (request.getStatus() != null) {
            clazz.setStatus(request.getStatus());
        }
        
        clazz.setUpdateTime(LocalDateTime.now());
        classMapper.updateById(clazz);
        
        log.info("班级更新成功，班级ID：{}", classId);
        return convertToClassResponse(clazz);
    }

    @Override
    @Transactional
    public void deleteClass(Long classId) {
        Long teacherId = SecurityUtils.getCurrentUserId();
        
        Class clazz = classMapper.selectByIdAndTeacherId(classId, teacherId);
        if (clazz == null) {
            throw new BusinessException(ResultCode.NOT_FOUND, "班级不存在");
        }
        
        // 检查是否有学生
        Integer studentCount = classMapper.getStudentCount(classId);
        if (studentCount > 0) {
            throw new BusinessException(ResultCode.BUSINESS_ERROR, "班级内还有学生，无法删除");
        }
        
        classMapper.deleteById(classId);
        log.info("班级删除成功，班级ID：{}", classId);
    }

    @Override
    public PageResponse<ClassStudentResponse> getClassStudents(Long classId, PageRequest pageRequest, String keyword) {
        Long teacherId = SecurityUtils.getCurrentUserId();
        
        // 验证班级权限
        if (!classMapper.existsByIdAndTeacherId(classId, teacherId)) {
            throw new BusinessException(ResultCode.FORBIDDEN, "无权限访问该班级");
        }
        
        Long offset = (pageRequest.getCurrent() - 1) * pageRequest.getPageSize();
        List<User> students = classMapper.selectStudentsByClassId(classId, keyword, offset, pageRequest.getPageSize());
        Long total = classMapper.countStudentsByClassId(classId, keyword);
        
        List<ClassStudentResponse> responses = students.stream()
                .map(this::convertToClassStudentResponse)
                .collect(Collectors.toList());
        
        return PageResponse.<ClassStudentResponse>builder()
                .records(responses)
                .total(total)
                .current(pageRequest.getCurrent())
                .pageSize(pageRequest.getPageSize())
                .build();
    }

    @Override
    @Transactional
    public void addStudentsToClass(Long classId, List<Long> studentIds) {
        Long teacherId = SecurityUtils.getCurrentUserId();
        
        // 验证班级权限
        if (!classMapper.existsByIdAndTeacherId(classId, teacherId)) {
            throw new BusinessException(ResultCode.FORBIDDEN, "无权限访问该班级");
        }
        
        // 批量添加学生
        classMapper.batchAddStudents(classId, studentIds);
        log.info("批量添加学生到班级成功，班级ID：{}，学生数量：{}", classId, studentIds.size());
    }

    @Override
    @Transactional
    public void removeStudentFromClass(Long classId, Long studentId) {
        Long teacherId = SecurityUtils.getCurrentUserId();
        
        // 验证班级权限
        if (!classMapper.existsByIdAndTeacherId(classId, teacherId)) {
            throw new BusinessException(ResultCode.FORBIDDEN, "无权限访问该班级");
        }
        
        classMapper.removeStudent(classId, studentId);
        log.info("从班级移除学生成功，班级ID：{}，学生ID：{}", classId, studentId);
    }

    @Override
    @Transactional
    public void removeStudentsFromClass(Long classId, List<Long> studentIds) {
        Long teacherId = SecurityUtils.getCurrentUserId();
        
        // 验证班级权限
        if (!classMapper.existsByIdAndTeacherId(classId, teacherId)) {
            throw new BusinessException(ResultCode.FORBIDDEN, "无权限访问该班级");
        }
        
        classMapper.batchRemoveStudents(classId, studentIds);
        log.info("批量从班级移除学生成功，班级ID：{}，学生数量：{}", classId, studentIds.size());
    }

    @Override
    public Object getClassStatistics(Long classId) {
        Long teacherId = SecurityUtils.getCurrentUserId();
        
        // 验证班级权限
        if (!classMapper.existsByIdAndTeacherId(classId, teacherId)) {
            throw new BusinessException(ResultCode.FORBIDDEN, "无权限访问该班级");
        }
        
        return getClassStatisticsData(classId);
    }

    @Override
    @Transactional
    public void updateClassStatus(Long classId, String status) {
        Long teacherId = SecurityUtils.getCurrentUserId();
        
        Class clazz = classMapper.selectByIdAndTeacherId(classId, teacherId);
        if (clazz == null) {
            throw new BusinessException(ResultCode.NOT_FOUND, "班级不存在");
        }
        
        clazz.setStatus(status);
        clazz.setUpdateTime(LocalDateTime.now());
        classMapper.updateById(clazz);
        
        log.info("班级状态更新成功，班级ID：{}，状态：{}", classId, status);
    }

    @Override
    @Transactional
    public ClassResponse copyClass(Long classId, String newName) {
        Long teacherId = SecurityUtils.getCurrentUserId();
        
        Class originalClass = classMapper.selectByIdAndTeacherId(classId, teacherId);
        if (originalClass == null) {
            throw new BusinessException(ResultCode.NOT_FOUND, "班级不存在");
        }
        
        // 检查新名称是否重复
        if (classMapper.existsByNameAndTeacherId(newName, teacherId)) {
            throw new BusinessException(ResultCode.BUSINESS_ERROR, "班级名称已存在");
        }
        
        Class newClass = new Class();
        newClass.setName(newName);
        newClass.setGrade(originalClass.getGrade());
        newClass.setMajor(originalClass.getMajor());
        newClass.setDescription(originalClass.getDescription());
        newClass.setCapacity(originalClass.getCapacity());
        newClass.setSemester(originalClass.getSemester());
        newClass.setTeacherId(teacherId);
        newClass.setStatus("ACTIVE");
        newClass.setCreateTime(LocalDateTime.now());
        newClass.setUpdateTime(LocalDateTime.now());
        
        classMapper.insert(newClass);
        
        log.info("班级复制成功，原班级ID：{}，新班级ID：{}", classId, newClass.getId());
        return convertToClassResponse(newClass);
    }

    @Override
    public String exportClassStudents(Long classId) {
        Long teacherId = SecurityUtils.getCurrentUserId();
        
        // 验证班级权限
        if (!classMapper.existsByIdAndTeacherId(classId, teacherId)) {
            throw new BusinessException(ResultCode.FORBIDDEN, "无权限访问该班级");
        }
        
        // TODO: 实现Excel导出逻辑
        // 这里应该调用文件服务生成Excel文件并返回下载URL
        String downloadUrl = "/api/downloads/class-students-" + classId + ".xlsx";
        
        log.info("班级学生名单导出成功，班级ID：{}，下载URL：{}", classId, downloadUrl);
        return downloadUrl;
    }

    // 私有辅助方法
    private ClassResponse convertToClassResponse(Class clazz) {
        ClassResponse response = new ClassResponse();
        response.setId(clazz.getId());
        response.setName(clazz.getName());
        response.setGrade(clazz.getGrade());
        response.setMajor(clazz.getMajor());
        response.setDescription(clazz.getDescription());
        response.setCapacity(clazz.getCapacity());
        response.setStudentCount(classMapper.getStudentCount(clazz.getId()));
        response.setSemester(clazz.getSemester());
        response.setStatus(clazz.getStatus());
        response.setCreateTime(clazz.getCreateTime());
        response.setUpdateTime(clazz.getUpdateTime());
        
        // 获取教师姓名
        User teacher = userMapper.selectById(clazz.getTeacherId());
        if (teacher != null) {
            response.setTeacherName(teacher.getUsername());
        }
        
        return response;
    }

    private ClassDetailResponse convertToClassDetailResponse(Class clazz) {
        ClassDetailResponse response = new ClassDetailResponse();
        response.setId(clazz.getId());
        response.setName(clazz.getName());
        response.setGrade(clazz.getGrade());
        response.setMajor(clazz.getMajor());
        response.setDescription(clazz.getDescription());
        response.setCapacity(clazz.getCapacity());
        response.setStudentCount(classMapper.getStudentCount(clazz.getId()));
        response.setSemester(clazz.getSemester());
        response.setStatus(clazz.getStatus());
        response.setCreateTime(clazz.getCreateTime());
        response.setUpdateTime(clazz.getUpdateTime());
        
        // 获取教师姓名
        User teacher = userMapper.selectById(clazz.getTeacherId());
        if (teacher != null) {
            response.setTeacherName(teacher.getUsername());
        }
        
        return response;
    }

    private ClassStudentResponse convertToClassStudentResponse(User student) {
        ClassStudentResponse response = new ClassStudentResponse();
        response.setId(student.getId());
        response.setStudentId(student.getStudentId());
        response.setName(student.getUsername());
        response.setEmail(student.getEmail());
        response.setPhone(student.getPhone());
        response.setGender(student.getGender());
        response.setJoinTime(student.getCreateTime());
        response.setStatus("ACTIVE");
        
        // TODO: 获取学习统计数据
        response.setCompletedAssignments(0);
        response.setTotalAssignments(0);
        response.setAverageScore(0.0);
        response.setAttendance(0);
        response.setLastActiveTime(LocalDateTime.now());
        
        return response;
    }

    private ClassDetailResponse.ClassStatistics getClassStatisticsData(Long classId) {
        ClassDetailResponse.ClassStatistics statistics = new ClassDetailResponse.ClassStatistics();
        
        // TODO: 实现真实的统计逻辑
        statistics.setTotalStudents(classMapper.getStudentCount(classId));
        statistics.setActiveStudents(classMapper.getActiveStudentCount(classId));
        statistics.setTotalCourses(0);
        statistics.setTotalAssignments(0);
        statistics.setAverageScore(0.0);
        statistics.setCompletedAssignments(0);
        
        return statistics;
    }

    private List<ClassDetailResponse.RecentActivity> getRecentActivities(Long classId) {
        // TODO: 实现获取最近活动的逻辑
        return List.of();
    }
}