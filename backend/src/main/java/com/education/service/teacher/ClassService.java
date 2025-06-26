package com.education.service.teacher;

import com.education.dto.ClassDTO;
import com.education.dto.common.PageRequest;
import com.education.dto.common.PageResponse;
import com.education.dto.clazz.*;

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
     * 分页查询班级列表
     */
    PageResponse<ClassResponse> getClassList(PageRequest pageRequest, String name, String grade, String status);

    /**
     * 创建班级
     */
    ClassResponse createClass(ClassCreateRequest request);

    /**
     * 获取班级详情
     */
    ClassDetailResponse getClassDetail(Long classId);

    /**
     * 更新班级信息
     */
    ClassResponse updateClass(Long classId, ClassUpdateRequest request);

    /**
     * 删除班级
     */
    void deleteClass(Long classId);

    /**
     * 获取班级学生列表
     */
    PageResponse<ClassStudentResponse> getClassStudents(Long classId, PageRequest pageRequest, String keyword);

    /**
     * 添加学生到班级
     */
    void addStudentsToClass(Long classId, List<Long> studentIds);

    /**
     * 从班级移除学生
     */
    void removeStudentFromClass(Long classId, Long studentId);

    /**
     * 批量从班级移除学生
     */
    void removeStudentsFromClass(Long classId, List<Long> studentIds);

    /**
     * 获取班级统计信息
     */
    Object getClassStatistics(Long classId);

    /**
     * 设置班级状态
     */
    void updateClassStatus(Long classId, String status);

    /**
     * 复制班级
     */
    ClassResponse copyClass(Long classId, String newName);

    /**
     * 导出班级学生名单
     */
    String exportClassStudents(Long classId);

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