package com.education.service.teacher;

import com.education.dto.ClassDTO;
import com.education.dto.common.PageRequest;
import com.education.dto.common.PageResponse;

import java.util.List;

/**
 * 教师端班级服务接口
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
public interface ClassService {

    /**
     * 创建班级
     * 
     * @param createRequest 创建班级请求
     * @param teacherId 教师ID
     * @return 班级信息
     */
    ClassDTO.ClassResponse createClass(ClassDTO.ClassCreateRequest createRequest, Long teacherId);

    /**
     * 获取教师的班级列表
     * 
     * @param teacherId 教师ID
     * @param pageRequest 分页请求
     * @return 班级列表
     */
    PageResponse<ClassDTO.ClassResponse> getClassList(Long teacherId, PageRequest pageRequest);

    /**
     * 获取班级详情
     * 
     * @param classId 班级ID
     * @param teacherId 教师ID
     * @return 班级详情
     */
    ClassDTO.ClassResponse getClassDetail(Long classId, Long teacherId);

    /**
     * 更新班级信息
     * 
     * @param classId 班级ID
     * @param updateRequest 更新请求
     * @param teacherId 教师ID
     * @return 更新后的班级信息
     */
    ClassDTO.ClassResponse updateClass(Long classId, ClassDTO.ClassUpdateRequest updateRequest, Long teacherId);

    /**
     * 删除班级
     * 
     * @param classId 班级ID
     * @param teacherId 教师ID
     * @return 操作结果
     */
    Boolean deleteClass(Long classId, Long teacherId);

    /**
     * 获取班级学生列表
     * 
     * @param classId 班级ID
     * @param teacherId 教师ID
     * @param pageRequest 分页请求
     * @return 学生列表
     */
    PageResponse<Object> getClassStudents(Long classId, Long teacherId, PageRequest pageRequest);

    /**
     * 从班级移除学生
     * 
     * @param classId 班级ID
     * @param studentId 学生ID
     * @param teacherId 教师ID
     * @return 操作结果
     */
    Boolean removeStudent(Long classId, Long studentId, Long teacherId);

    /**
     * 批量移除学生
     * 
     * @param classId 班级ID
     * @param studentIds 学生ID列表
     * @param teacherId 教师ID
     * @return 操作结果
     */
    Boolean removeStudents(Long classId, List<Long> studentIds, Long teacherId);

    /**
     * 生成班级邀请码
     * 
     * @param classId 班级ID
     * @param teacherId 教师ID
     * @param expireHours 过期小时数
     * @return 邀请码信息
     */
    Object generateInviteCode(Long classId, Long teacherId, Integer expireHours);

    /**
     * 获取班级统计信息
     * 
     * @param classId 班级ID
     * @param teacherId 教师ID
     * @return 统计信息
     */
    Object getClassStatistics(Long classId, Long teacherId);

    /**
     * 导出班级学生信息
     * 
     * @param classId 班级ID
     * @param teacherId 教师ID
     * @return 导出文件路径
     */
    String exportClassStudents(Long classId, Long teacherId);

    /**
     * 添加学生到班级
     * 
     * @param classId 班级ID
     * @param studentIds 学生ID列表
     * @param teacherId 教师ID
     * @return 操作结果
     */
    Boolean addStudents(Long classId, List<Long> studentIds, Long teacherId);

    /**
     * 通过邀请码加入班级
     * 
     * @param inviteCode 邀请码
     * @param studentId 学生ID
     * @return 操作结果
     */
    Boolean joinClassByInviteCode(String inviteCode, Long studentId);

    /**
     * 获取班级邀请码列表
     * 
     * @param classId 班级ID
     * @param teacherId 教师ID
     * @return 邀请码列表
     */
    List<Object> getInviteCodes(Long classId, Long teacherId);

    /**
     * 禁用邀请码
     * 
     * @param inviteCodeId 邀请码ID
     * @param teacherId 教师ID
     * @return 操作结果
     */
    Boolean disableInviteCode(Long inviteCodeId, Long teacherId);

    /**
     * 设置班级状态
     * 
     * @param classId 班级ID
     * @param status 状态
     * @param teacherId 教师ID
     * @return 操作结果
     */
    Boolean setClassStatus(Long classId, String status, Long teacherId);

    /**
     * 复制班级
     * 
     * @param classId 班级ID
     * @param newClassName 新班级名称
     * @param teacherId 教师ID
     * @return 新班级信息
     */
    ClassDTO.ClassResponse copyClass(Long classId, String newClassName, Long teacherId);

    /**
     * 获取班级课程列表
     * 
     * @param classId 班级ID
     * @param teacherId 教师ID
     * @return 课程列表
     */
    List<Object> getClassCourses(Long classId, Long teacherId);

    /**
     * 为班级分配课程
     * 
     * @param classId 班级ID
     * @param courseIds 课程ID列表
     * @param teacherId 教师ID
     * @return 操作结果
     */
    Boolean assignCourses(Long classId, List<Long> courseIds, Long teacherId);

    /**
     * 移除班级课程
     * 
     * @param classId 班级ID
     * @param courseId 课程ID
     * @param teacherId 教师ID
     * @return 操作结果
     */
    Boolean removeCourse(Long classId, Long courseId, Long teacherId);

    /**
     * 获取班级任务列表
     * 
     * @param classId 班级ID
     * @param teacherId 教师ID
     * @param pageRequest 分页请求
     * @return 任务列表
     */
    PageResponse<Object> getClassTasks(Long classId, Long teacherId, PageRequest pageRequest);

    /**
     * 获取班级成绩概览
     * 
     * @param classId 班级ID
     * @param teacherId 教师ID
     * @return 成绩概览
     */
    Object getClassGradeOverview(Long classId, Long teacherId);

    /**
     * 发送班级通知
     * 
     * @param classId 班级ID
     * @param title 通知标题
     * @param content 通知内容
     * @param teacherId 教师ID
     * @return 操作结果
     */
    Boolean sendClassNotification(Long classId, String title, String content, Long teacherId);

    /**
     * 获取班级通知列表
     * 
     * @param classId 班级ID
     * @param teacherId 教师ID
     * @param pageRequest 分页请求
     * @return 通知列表
     */
    PageResponse<Object> getClassNotifications(Long classId, Long teacherId, PageRequest pageRequest);

    /**
     * 归档班级
     * 
     * @param classId 班级ID
     * @param teacherId 教师ID
     * @return 操作结果
     */
    Boolean archiveClass(Long classId, Long teacherId);

    /**
     * 恢复归档班级
     * 
     * @param classId 班级ID
     * @param teacherId 教师ID
     * @return 操作结果
     */
    Boolean restoreClass(Long classId, Long teacherId);
}