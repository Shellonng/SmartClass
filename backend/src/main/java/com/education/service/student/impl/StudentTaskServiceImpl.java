package com.education.service.student.impl;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.education.dto.TaskDTO;
import com.education.dto.common.PageRequest;
import com.education.dto.common.PageResponse;
import com.education.entity.*;
import com.education.exception.BusinessException;
import com.education.exception.ResultCode;
import com.education.mapper.*;
import com.education.service.student.StudentTaskService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.BeanUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.util.StringUtils;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.time.LocalDateTime;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

/**
 * 学生端任务服务实现类
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
@Service
@Slf4j
public class StudentTaskServiceImpl implements StudentTaskService {

    @Autowired
    private TaskMapper taskMapper;
    
    @Autowired
    private TaskSubmissionMapper taskSubmissionMapper;
    
    @Autowired
    private StudentMapper studentMapper;
    
    @Autowired
    private CourseMapper courseMapper;
    
    @Autowired
    private ResourceMapper resourceMapper;

    @Override
    public PageResponse<TaskDTO.TaskListResponse> getStudentTasks(Long studentId, PageRequest pageRequest) {
        log.info("获取学生任务列表，学生ID: {}", studentId);
        
        // 验证学生是否存在
        Student student = studentMapper.selectById(studentId);
        if (student == null) {
            throw new BusinessException(ResultCode.USER_NOT_FOUND, "学生不存在");
        }
        
        // 构建分页查询
        Page<Task> page = new Page<>(pageRequest.getPageNum(), pageRequest.getPageSize());
        
        // 查询学生所在班级的课程任务
        QueryWrapper<Task> queryWrapper = new QueryWrapper<>();
        queryWrapper.exists("SELECT 1 FROM course c WHERE c.id = task.course_id AND c.class_id = {0}", student.getClassId())
                   .orderByDesc("create_time");
        
        IPage<Task> taskPage = taskMapper.selectPage(page, queryWrapper);
        
        // 转换为响应对象
        List<TaskDTO.TaskListResponse> responseList = taskPage.getRecords().stream()
                .map(this::convertToTaskListResponse)
                .collect(Collectors.toList());
        
        return PageResponse.of(pageRequest.getPageNum(), pageRequest.getPageSize(), taskPage.getTotal(), responseList);
    }

    @Override
    public TaskDTO.TaskDetailResponse getTaskDetail(Long taskId, Long studentId) {
        log.info("获取任务详情，任务ID: {}, 学生ID: {}", taskId, studentId);
        
        // 验证任务是否存在
        Task task = taskMapper.selectById(taskId);
        if (task == null) {
            throw new BusinessException(ResultCode.TASK_NOT_FOUND, "任务不存在");
        }
        
        // 验证学生是否有权限访问该任务
        if (!hasTaskAccess(taskId, studentId)) {
            throw new BusinessException(ResultCode.ACCESS_DENIED, "无权限访问该任务");
        }
        
        // 查询提交记录
        QueryWrapper<TaskSubmission> submissionWrapper = new QueryWrapper<>();
        submissionWrapper.eq("task_id", taskId).eq("student_id", studentId);
        TaskSubmission submission = taskSubmissionMapper.selectOne(submissionWrapper);
        
        // 转换为响应对象
        TaskDTO.TaskDetailResponse response = new TaskDTO.TaskDetailResponse();
        BeanUtils.copyProperties(task, response);
        
        if (submission != null) {
            response.setHasSubmitted(true);
            response.setSubmitTime(submission.getSubmitTime());
            response.setMyScore(submission.getScore());
            response.setFeedback(submission.getFeedback());
        } else {
            response.setHasSubmitted(false);
        }
        
        return response;
    }

    @Override
    @Transactional
    public Long submitTask(TaskDTO.TaskSubmissionRequest submissionRequest, Long studentId) {
        log.info("提交任务，任务ID: {}, 学生ID: {}", submissionRequest.getTaskId(), studentId);
        
        // 验证任务是否存在
        Task task = taskMapper.selectById(submissionRequest.getTaskId());
        if (task == null) {
            throw new BusinessException(ResultCode.TASK_NOT_FOUND, "任务不存在");
        }
        
        // 验证是否超过截止时间
        LocalDateTime now = LocalDateTime.now();
        boolean isLate = task.getDueTime() != null && now.isAfter(task.getDueTime());
        
        // 检查是否已经提交过
        QueryWrapper<TaskSubmission> existWrapper = new QueryWrapper<>();
        existWrapper.eq("task_id", submissionRequest.getTaskId()).eq("student_id", studentId);
        TaskSubmission existSubmission = taskSubmissionMapper.selectOne(existWrapper);
        
        if (existSubmission != null && !"DRAFT".equals(existSubmission.getStatus())) {
            throw new BusinessException(ResultCode.TASK_ALREADY_SUBMITTED, "任务已提交");
        }
        
        // 创建或更新提交记录
        TaskSubmission submission = existSubmission != null ? existSubmission : new TaskSubmission();
        submission.setTaskId(submissionRequest.getTaskId());
        submission.setStudentId(studentId);
         submission.setContent(submissionRequest.getContent());
        submission.setFiles(submissionRequest.getAttachments() != null ? 
                String.join(",", submissionRequest.getAttachments()) : null);
        submission.setLinks(submissionRequest.getLinks() != null ? 
                String.join(",", submissionRequest.getLinks()) : null);
        submission.setSubmitTime(now);
        submission.setIsLate(isLate);
        submission.setStatus("SUBMITTED");
        
        if (isLate && task.getDueTime() != null) {
            long lateDays = java.time.Duration.between(task.getDueTime(), now).toDays();
            submission.setLateDays((int) lateDays);
        }
        
        if (existSubmission != null) {
            taskSubmissionMapper.updateById(submission);
        } else {
            taskSubmissionMapper.insert(submission);
        }
        
        return submission.getId();
    }

    @Override
    @Transactional
    public Boolean saveDraft(TaskDTO.TaskDraftRequest draftRequest, Long studentId) {
        log.info("保存任务草稿，任务ID: {}, 学生ID: {}", draftRequest.getTaskId(), studentId);
        
        // 验证任务是否存在
        Task task = taskMapper.selectById(draftRequest.getTaskId());
        if (task == null) {
            throw new BusinessException(ResultCode.TASK_NOT_FOUND, "任务不存在");
        }
        
        // 查询是否已有草稿
        QueryWrapper<TaskSubmission> wrapper = new QueryWrapper<>();
        wrapper.eq("task_id", draftRequest.getTaskId()).eq("student_id", studentId);
        TaskSubmission existSubmission = taskSubmissionMapper.selectOne(wrapper);
        
        // 如果已提交，不能保存草稿
        if (existSubmission != null && "SUBMITTED".equals(existSubmission.getStatus())) {
            throw new BusinessException(ResultCode.TASK_ALREADY_SUBMITTED, "任务已提交，无法保存草稿");
        }
        
        // 创建或更新草稿
        TaskSubmission submission = existSubmission != null ? existSubmission : new TaskSubmission();
        submission.setTaskId(draftRequest.getTaskId());
        submission.setStudentId(studentId);
        submission.setContent(draftRequest.getContent());
        submission.setFiles(draftRequest.getAttachments() != null ? 
                String.join(",", draftRequest.getAttachments()) : null);
        submission.setLinks(draftRequest.getLinks() != null ? 
                String.join(",", draftRequest.getLinks()) : null);
        submission.setStatus("DRAFT");
        submission.setUpdateTime(LocalDateTime.now());
        
        if (existSubmission != null) {
            taskSubmissionMapper.updateById(submission);
        } else {
            submission.setCreateTime(LocalDateTime.now());
            taskSubmissionMapper.insert(submission);
        }
        
        return true;
    }

    @Override
    public TaskDTO.TaskDraftResponse getDraft(Long taskId, Long studentId) {
        log.info("获取任务草稿，任务ID: {}, 学生ID: {}", taskId, studentId);
        
        QueryWrapper<TaskSubmission> wrapper = new QueryWrapper<>();
        wrapper.eq("task_id", taskId).eq("student_id", studentId).eq("status", "DRAFT");
        TaskSubmission submission = taskSubmissionMapper.selectOne(wrapper);
        
        if (submission == null) {
            return null;
        }
        
        TaskDTO.TaskDraftResponse response = new TaskDTO.TaskDraftResponse();
        response.setTaskId(taskId);
        response.setContent(submission.getContent());
        response.setAttachments(submission.getFiles() != null ? 
                Arrays.asList(submission.getFiles().split(",")) : List.of());
        response.setLinks(submission.getLinks() != null ? 
                Arrays.asList(submission.getLinks().split(",")) : List.of());
        response.setSaveTime(submission.getUpdateTime());
        
        return response;
    }

    @Override
    public TaskDTO.SubmissionResponse getSubmission(Long taskId, Long studentId) {
        log.info("获取任务提交记录，任务ID: {}, 学生ID: {}", taskId, studentId);
        
        QueryWrapper<TaskSubmission> wrapper = new QueryWrapper<>();
        wrapper.eq("task_id", taskId).eq("student_id", studentId);
        TaskSubmission submission = taskSubmissionMapper.selectOne(wrapper);
        
        if (submission == null) {
            return null;
        }
        
        TaskDTO.SubmissionResponse response = new TaskDTO.SubmissionResponse();
        BeanUtils.copyProperties(submission, response);
        
        return response;
    }

    @Override
    @Transactional
    public Boolean updateSubmission(Long submissionId, TaskDTO.SubmissionUpdateRequest updateRequest, Long studentId) {
        log.info("更新任务提交，提交ID: {}, 学生ID: {}", submissionId, studentId);
        
        TaskSubmission submission = taskSubmissionMapper.selectById(submissionId);
        if (submission == null) {
            throw new BusinessException(ResultCode.SUBMISSION_NOT_FOUND, "提交记录不存在");
        }
        
        if (!submission.getStudentId().equals(studentId)) {
            throw new BusinessException(ResultCode.ACCESS_DENIED, "无权限修改该提交");
        }
        
        if ("GRADED".equals(submission.getStatus())) {
            throw new BusinessException(ResultCode.SUBMISSION_ALREADY_GRADED, "已评分的提交无法修改");
        }
        
        submission.setContent(updateRequest.getContent());
        submission.setFiles(updateRequest.getAttachments() != null ? 
                String.join(",", updateRequest.getAttachments()) : null);
        submission.setLinks(updateRequest.getLinks() != null ? 
                String.join(",", updateRequest.getLinks()) : null);
        submission.setUpdateTime(LocalDateTime.now());
        
        taskSubmissionMapper.updateById(submission);
        return true;
    }

    @Override
    @Transactional
    public Boolean withdrawSubmission(Long submissionId, Long studentId) {
        log.info("撤回任务提交，提交ID: {}, 学生ID: {}", submissionId, studentId);
        
        TaskSubmission submission = taskSubmissionMapper.selectById(submissionId);
        if (submission == null) {
            throw new BusinessException(ResultCode.SUBMISSION_NOT_FOUND, "提交记录不存在");
        }
        
        if (!submission.getStudentId().equals(studentId)) {
            throw new BusinessException(ResultCode.ACCESS_DENIED, "无权限撤回该提交");
        }
        
        if ("GRADED".equals(submission.getStatus())) {
            throw new BusinessException(ResultCode.SUBMISSION_ALREADY_GRADED, "已评分的提交无法撤回");
        }
        
        submission.setStatus("WITHDRAWN");
        submission.setUpdateTime(LocalDateTime.now());
        
        taskSubmissionMapper.updateById(submission);
        return true;
    }

    @Override
    public TaskDTO.TaskGradeResponse getTaskGrade(Long taskId, Long studentId) {
        log.info("获取任务成绩，任务ID: {}, 学生ID: {}", taskId, studentId);
        
        // 验证权限
        if (!hasTaskAccess(taskId, studentId)) {
            throw new BusinessException(ResultCode.ACCESS_DENIED, "无权限访问该任务");
        }
        
        QueryWrapper<TaskSubmission> wrapper = new QueryWrapper<>();
        wrapper.eq("task_id", taskId).eq("student_id", studentId).eq("status", "GRADED");
        TaskSubmission submission = taskSubmissionMapper.selectOne(wrapper);
        
        if (submission == null || submission.getScore() == null) {
            return null;
        }
        
        TaskDTO.TaskGradeResponse response = new TaskDTO.TaskGradeResponse();
        response.setTaskId(taskId);
        response.setScore(submission.getScore());
        Task task = taskMapper.selectById(taskId);
        response.setScore(submission.getScore());
        response.setTotalScore(task.getMaxScore());
        response.setGradeTime(submission.getGradeTime());
        response.setFeedback(submission.getFeedback());
        
        return response;
    }

    @Override
    public TaskDTO.TaskFeedbackResponse getTaskFeedback(Long taskId, Long studentId) {
        log.info("获取任务反馈，任务ID: {}, 学生ID: {}", taskId, studentId);
        
        // 验证权限
        if (!hasTaskAccess(taskId, studentId)) {
            throw new BusinessException(ResultCode.ACCESS_DENIED, "无权限访问该任务");
        }
        
        QueryWrapper<TaskSubmission> wrapper = new QueryWrapper<>();
        wrapper.eq("task_id", taskId).eq("student_id", studentId);
        TaskSubmission submission = taskSubmissionMapper.selectOne(wrapper);
        
        if (submission == null || !StringUtils.hasText(submission.getFeedback())) {
            return null;
        }
        
        TaskDTO.TaskFeedbackResponse response = new TaskDTO.TaskFeedbackResponse();
        response.setTaskId(taskId);
        response.setFeedback(submission.getFeedback());
        response.setGradeTime(submission.getGradeTime());
        response.setScore(submission.getScore());
        
        return response;
    }

    @Override
    public PageResponse<TaskDTO.TaskListResponse> getPendingTasks(Long studentId, PageRequest pageRequest) {
        log.info("获取待完成任务列表，学生ID: {}", studentId);
        
        Student student = studentMapper.selectById(studentId);
        if (student == null) {
            throw new BusinessException(ResultCode.USER_NOT_FOUND, "学生不存在");
        }
        
        Page<Task> page = new Page<>(pageRequest.getPageNum(), pageRequest.getPageSize());
        
        QueryWrapper<Task> queryWrapper = new QueryWrapper<>();
        queryWrapper.exists("SELECT 1 FROM course c WHERE c.id = task.course_id AND c.class_id = {0}", student.getClassId())
                   .notExists("SELECT 1 FROM task_submission ts WHERE ts.task_id = task.id AND ts.student_id = {0} AND ts.status = 'SUBMITTED'", studentId)
                   .orderByAsc("due_time");
        
        IPage<Task> taskPage = taskMapper.selectPage(page, queryWrapper);
        
        List<TaskDTO.TaskListResponse> responseList = taskPage.getRecords().stream()
                .map(this::convertToTaskListResponse)
                .collect(Collectors.toList());
        
        return PageResponse.of(pageRequest.getPageNum(), pageRequest.getPageSize(), taskPage.getTotal(), responseList);
    }

    @Override
    public PageResponse<TaskDTO.TaskListResponse> getCompletedTasks(Long studentId, PageRequest pageRequest) {
        log.info("获取已完成任务列表，学生ID: {}", studentId);
        
        Student student = studentMapper.selectById(studentId);
        if (student == null) {
            throw new BusinessException(ResultCode.USER_NOT_FOUND, "学生不存在");
        }
        
        Page<Task> page = new Page<>(pageRequest.getPageNum(), pageRequest.getPageSize());
        
        QueryWrapper<Task> queryWrapper = new QueryWrapper<>();
        queryWrapper.exists("SELECT 1 FROM course c WHERE c.id = task.course_id AND c.class_id = {0}", student.getClassId())
                   .exists("SELECT 1 FROM task_submission ts WHERE ts.task_id = task.id AND ts.student_id = {0} AND ts.status = 'SUBMITTED'", studentId)
                   .orderByDesc("create_time");
        
        IPage<Task> taskPage = taskMapper.selectPage(page, queryWrapper);
        
        List<TaskDTO.TaskListResponse> responseList = taskPage.getRecords().stream()
                .map(this::convertToTaskListResponse)
                .collect(Collectors.toList());
        
        return PageResponse.of(pageRequest.getPageNum(), pageRequest.getPageSize(), taskPage.getTotal(), responseList);
    }

    @Override
    public PageResponse<TaskDTO.TaskListResponse> getOverdueTasks(Long studentId, PageRequest pageRequest) {
        log.info("获取逾期任务列表，学生ID: {}", studentId);
        
        Student student = studentMapper.selectById(studentId);
        if (student == null) {
            throw new BusinessException(ResultCode.USER_NOT_FOUND, "学生不存在");
        }
        
        Page<Task> page = new Page<>(pageRequest.getPageNum(), pageRequest.getPageSize());
        LocalDateTime now = LocalDateTime.now();
        
        QueryWrapper<Task> queryWrapper = new QueryWrapper<>();
        queryWrapper.exists("SELECT 1 FROM course c WHERE c.id = task.course_id AND c.class_id = {0}", student.getClassId())
                   .lt("due_time", now)
                   .notExists("SELECT 1 FROM task_submission ts WHERE ts.task_id = task.id AND ts.student_id = {0} AND ts.status = 'SUBMITTED'", studentId)
                   .orderByDesc("due_time");
        
        IPage<Task> taskPage = taskMapper.selectPage(page, queryWrapper);
        
        List<TaskDTO.TaskListResponse> responseList = taskPage.getRecords().stream()
                .map(this::convertToTaskListResponse)
                .collect(Collectors.toList());
        
        return PageResponse.of(pageRequest.getPageNum(), pageRequest.getPageSize(), taskPage.getTotal(), responseList);
    }

    @Override
    public PageResponse<TaskDTO.TaskListResponse> searchTasks(TaskDTO.TaskSearchRequest searchRequest, Long studentId) {
        log.info("搜索任务，学生ID: {}, 关键词: {}", studentId, searchRequest.getKeyword());
        
        Student student = studentMapper.selectById(studentId);
        if (student == null) {
            throw new BusinessException(ResultCode.USER_NOT_FOUND, "学生不存在");
        }
        
        Page<Task> page = new Page<>(1, 10); // 默认第一页，每页10条
        
        QueryWrapper<Task> queryWrapper = new QueryWrapper<>();
        queryWrapper.exists("SELECT 1 FROM course c WHERE c.id = task.course_id AND c.class_id = {0}", student.getClassId());
        
        if (StringUtils.hasText(searchRequest.getKeyword())) {
            queryWrapper.and(wrapper -> wrapper.like("title", searchRequest.getKeyword())
                                              .or().like("description", searchRequest.getKeyword())
                                              .or().like("content", searchRequest.getKeyword()));
        }
        
        if (searchRequest.getCourseId() != null) {
            queryWrapper.eq("course_id", searchRequest.getCourseId());
        }
        
        if (StringUtils.hasText(searchRequest.getTaskType())) {
            queryWrapper.eq("task_type", searchRequest.getTaskType());
        }
        
        queryWrapper.orderByDesc("create_time");
        
        IPage<Task> taskPage = taskMapper.selectPage(page, queryWrapper);
        
        List<TaskDTO.TaskListResponse> responseList = taskPage.getRecords().stream()
                .map(this::convertToTaskListResponse)
                .collect(Collectors.toList());
        
        return PageResponse.of(
                1, // 默认第一页
                10, // 默认每页10条
                taskPage.getTotal(),
                responseList
        );
    }

    @Override
    public TaskDTO.TaskStatisticsResponse getTaskStatistics(Long studentId) {
        log.info("获取任务统计，学生ID: {}", studentId);
        
        Student student = studentMapper.selectById(studentId);
        if (student == null) {
            throw new BusinessException(ResultCode.USER_NOT_FOUND, "学生不存在");
        }
        
        // 统计总任务数
        QueryWrapper<Task> totalWrapper = new QueryWrapper<>();
        totalWrapper.exists("SELECT 1 FROM course c WHERE c.id = task.course_id AND c.class_id = {0}", student.getClassId());
        Long totalTasks = taskMapper.selectCount(totalWrapper);
        
        // 统计已完成任务数
        QueryWrapper<Task> completedWrapper = new QueryWrapper<>();
        completedWrapper.exists("SELECT 1 FROM course c WHERE c.id = task.course_id AND c.class_id = {0}", student.getClassId())
                       .exists("SELECT 1 FROM task_submission ts WHERE ts.task_id = task.id AND ts.student_id = {0} AND ts.status = 'SUBMITTED'", studentId);
        Long completedTasks = taskMapper.selectCount(completedWrapper);
        
        // 统计逾期任务数
        QueryWrapper<Task> overdueWrapper = new QueryWrapper<>();
        overdueWrapper.exists("SELECT 1 FROM course c WHERE c.id = task.course_id AND c.class_id = {0}", student.getClassId())
                     .lt("due_time", LocalDateTime.now())
                     .notExists("SELECT 1 FROM task_submission ts WHERE ts.task_id = task.id AND ts.student_id = {0} AND ts.status = 'SUBMITTED'", studentId);
        Long overdueTasks = taskMapper.selectCount(overdueWrapper);
        
        // 统计平均分
        QueryWrapper<TaskSubmission> gradeWrapper = new QueryWrapper<>();
        gradeWrapper.eq("student_id", studentId).eq("status", "GRADED").isNotNull("score");
        List<TaskSubmission> gradedSubmissions = taskSubmissionMapper.selectList(gradeWrapper);
        
        BigDecimal averageScore = BigDecimal.ZERO;
        if (!gradedSubmissions.isEmpty()) {
            BigDecimal totalScore = gradedSubmissions.stream()
                    .map(TaskSubmission::getScore)
                    .reduce(BigDecimal.ZERO, BigDecimal::add);
            averageScore = totalScore.divide(BigDecimal.valueOf(gradedSubmissions.size()), 2, RoundingMode.HALF_UP);
        }
        
        TaskDTO.TaskStatisticsResponse response = new TaskDTO.TaskStatisticsResponse();
        response.setTotalTasks(totalTasks.intValue());
        response.setCompletedTasks(completedTasks.intValue());
        response.setPendingTasks((int)(totalTasks - completedTasks));
        response.setOverdueTasks(overdueTasks.intValue());
        response.setAverageScore(averageScore.doubleValue());
        
        return response;
    }

    @Override
    public TaskDTO.TaskCalendarResponse getTaskCalendar(Long studentId, Integer year, Integer month) {
        log.info("获取任务日历，学生ID: {}, 年月: {}-{}", studentId, year, month);
        
        Student student = studentMapper.selectById(studentId);
        if (student == null) {
            throw new BusinessException(ResultCode.USER_NOT_FOUND, "学生不存在");
        }
        
        TaskDTO.TaskCalendarResponse response = new TaskDTO.TaskCalendarResponse();
        response.setYear(year);
        response.setMonth(month);
        response.setTasks(List.of());
        
        return response;
    }

    @Override
    @Transactional
    public Boolean favoriteTask(Long taskId, Long studentId) {
        log.info("收藏任务，任务ID: {}, 学生ID: {}", taskId, studentId);
        
        // 验证任务是否存在
        Task task = taskMapper.selectById(taskId);
        if (task == null) {
            throw new BusinessException(ResultCode.TASK_NOT_FOUND, "任务不存在");
        }
        
        // 验证权限
        if (!hasTaskAccess(taskId, studentId)) {
            throw new BusinessException(ResultCode.ACCESS_DENIED, "无权限访问该任务");
        }
        
        // 这里可以实现收藏逻辑，比如在用户偏好表中添加记录
        // 暂时返回true表示操作成功
        return true;
    }

    @Override
    @Transactional
    public Boolean unfavoriteTask(Long taskId, Long studentId) {
        log.info("取消收藏任务，任务ID: {}, 学生ID: {}", taskId, studentId);
        
        // 验证任务是否存在
        Task task = taskMapper.selectById(taskId);
        if (task == null) {
            throw new BusinessException(ResultCode.TASK_NOT_FOUND, "任务不存在");
        }
        
        // 这里可以实现取消收藏逻辑
        // 暂时返回true表示操作成功
        return true;
    }

    @Override
    public PageResponse<TaskDTO.TaskListResponse> getFavoriteTasks(Long studentId, PageRequest pageRequest) {
        log.info("获取收藏的任务列表，学生ID: {}", studentId);
        
        // 这里可以实现获取收藏任务的逻辑
        // 暂时返回空列表
        return PageResponse.of(
                pageRequest.getPageNum(),
                pageRequest.getPageSize(),
                0L,
                List.<TaskDTO.TaskListResponse>of()
        );
    }

    @Override
    public List<TaskDTO.TaskReminderResponse> getTaskReminders(Long studentId) {
        log.info("获取任务提醒列表，学生ID: {}", studentId);
        
        Student student = studentMapper.selectById(studentId);
        if (student == null) {
            throw new BusinessException(ResultCode.USER_NOT_FOUND, "学生不存在");
        }
        
        // 这里可以实现获取提醒的逻辑
        // 暂时返回空列表
        return List.of();
    }

    @Override
    @Transactional
    public Boolean setTaskReminder(TaskDTO.TaskReminderRequest reminderRequest, Long studentId) {
        log.info("设置任务提醒，任务ID: {}, 学生ID: {}", reminderRequest.getTaskId(), studentId);
        
        // 验证任务是否存在
        Task task = taskMapper.selectById(reminderRequest.getTaskId());
        if (task == null) {
            throw new BusinessException(ResultCode.TASK_NOT_FOUND, "任务不存在");
        }
        
        // 验证权限
        if (!hasTaskAccess(reminderRequest.getTaskId(), studentId)) {
            throw new BusinessException(ResultCode.ACCESS_DENIED, "无权限访问该任务");
        }
        
        // 这里可以实现设置提醒的逻辑
        // 暂时返回true表示操作成功
        return true;
    }

    @Override
    @Transactional
    public Boolean cancelTaskReminder(Long reminderId, Long studentId) {
        log.info("取消任务提醒，提醒ID: {}, 学生ID: {}", reminderId, studentId);
        
        // 这里可以实现取消提醒的逻辑
        // 暂时返回true表示操作成功
        return true;
    }

    @Override
    public PageResponse<TaskDTO.TaskDiscussionResponse> getTaskDiscussions(Long taskId, Long studentId, PageRequest pageRequest) {
        log.info("获取任务讨论列表，任务ID: {}, 学生ID: {}", taskId, studentId);
        
        // 验证权限
        if (!hasTaskAccess(taskId, studentId)) {
            throw new BusinessException(ResultCode.ACCESS_DENIED, "无权限访问该任务");
        }
        
        // 这里可以实现获取讨论的逻辑
        // 暂时返回空列表
        return PageResponse.of(
                pageRequest.getPageNum(),
                pageRequest.getPageSize(),
                0L,
                List.<TaskDTO.TaskDiscussionResponse>of()
        );
    }

    @Override
    @Transactional
    public Long createTaskDiscussion(TaskDTO.TaskDiscussionCreateRequest discussionRequest, Long studentId) {
        log.info("创建任务讨论，任务ID: {}, 学生ID: {}", discussionRequest.getTaskId(), studentId);
        
        // 验证权限
        if (!hasTaskAccess(discussionRequest.getTaskId(), studentId)) {
            throw new BusinessException(ResultCode.ACCESS_DENIED, "无权限访问该任务");
        }
        
        // 这里可以实现创建讨论的逻辑
        // 暂时返回1L表示操作成功
        return 1L;
    }

    @Override
    @Transactional
    public Long replyTaskDiscussion(Long discussionId, TaskDTO.TaskDiscussionReplyRequest replyRequest, Long studentId) {
        log.info("回复任务讨论，讨论ID: {}, 学生ID: {}", discussionId, studentId);
        
        // 这里可以实现回复讨论的逻辑
        // 暂时返回1L表示操作成功
        return 1L;
    }

    @Override
    public List<TaskDTO.TaskResourceResponse> getTaskResources(Long taskId, Long studentId) {
        log.info("获取任务资源列表，任务ID: {}, 学生ID: {}", taskId, studentId);
        
        // 验证权限
        if (!hasTaskAccess(taskId, studentId)) {
            throw new BusinessException(ResultCode.ACCESS_DENIED, "无权限访问该任务");
        }
        
        // 获取任务关联的资源
        QueryWrapper<Resource> queryWrapper = new QueryWrapper<>();
        queryWrapper.eq("task_id", taskId)
                   .eq("status", 1)
                   .orderByDesc("create_time");
        
        List<Resource> resources = resourceMapper.selectList(queryWrapper);
        
        return resources.stream().map(resource -> {
            TaskDTO.TaskResourceResponse response = new TaskDTO.TaskResourceResponse();
            response.setResourceId(resource.getId());
             response.setResourceName(resource.getResourceName());
             response.setResourceType(resource.getResourceType());
             response.setFileUrl(resource.getFileUrl());
             response.setFileSize(resource.getFileSize());
             response.setUploadTime(resource.getCreateTime());
            return response;
        }).collect(Collectors.toList());
    }

    @Override
    public TaskDTO.ResourceDownloadResponse downloadTaskResource(Long resourceId, Long studentId) {
        log.info("下载任务资源，资源ID: {}, 学生ID: {}", resourceId, studentId);
        
        Resource resource = resourceMapper.selectById(resourceId);
        if (resource == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "资源不存在");
        }
        
        // 验证权限（通过任务ID验证）
        if (resource.getTaskId() != null && !hasTaskAccess(resource.getTaskId(), studentId)) {
            throw new BusinessException(ResultCode.ACCESS_DENIED, "无权限下载该资源");
        }
        
        TaskDTO.ResourceDownloadResponse response = new TaskDTO.ResourceDownloadResponse();
        response.setFileName(resource.getResourceName());
        response.setDownloadUrl(resource.getFileUrl());
        response.setFileSize(resource.getFileSize());
        
        return response;
    }

    @Override
    public TaskDTO.TaskTemplateResponse getTaskTemplate(Long taskId, Long studentId) {
        log.info("获取任务模板，任务ID: {}, 学生ID: {}", taskId, studentId);
        
        // 验证权限
        if (!hasTaskAccess(taskId, studentId)) {
            throw new BusinessException(ResultCode.ACCESS_DENIED, "无权限访问该任务");
        }
        
        Task task = taskMapper.selectById(taskId);
        if (task == null) {
            throw new BusinessException(ResultCode.TASK_NOT_FOUND, "任务不存在");
        }
        
        TaskDTO.TaskTemplateResponse response = new TaskDTO.TaskTemplateResponse();
        response.setTemplateId(1L); // 默认模板ID
        response.setTemplateName("默认任务模板");
        response.setDescription("默认模板描述");
        response.setTaskType("HOMEWORK");
        
        return response;
    }

    @Override
    public TaskDTO.GradingCriteriaResponse getGradingCriteria(Long taskId, Long studentId) {
        log.info("获取评分标准，任务ID: {}, 学生ID: {}", taskId, studentId);
        
        // 验证权限
        if (!hasTaskAccess(taskId, studentId)) {
            throw new BusinessException(ResultCode.ACCESS_DENIED, "无权限访问该任务");
        }
        
        Task task = taskMapper.selectById(taskId);
        if (task == null) {
            throw new BusinessException(ResultCode.TASK_NOT_FOUND, "任务不存在");
        }
        
        TaskDTO.GradingCriteriaResponse response = new TaskDTO.GradingCriteriaResponse();
        response.setCriteriaId(1L); // 默认标准ID
        response.setCriteriaName("默认评分标准");
        response.setDescription("评分标准描述");
        response.setMaxScore(new BigDecimal("100"));
        response.setWeight(new BigDecimal("1.0"));
        response.setGradingType("NUMERIC");
        
        return response;
    }

    @Override
    public PageResponse<TaskDTO.PeerReviewTaskResponse> getPeerReviewTasks(Long studentId, PageRequest pageRequest) {
        log.info("获取同伴评价任务列表，学生ID: {}", studentId);
        
        Student student = studentMapper.selectById(studentId);
        if (student == null) {
            throw new BusinessException(ResultCode.USER_NOT_FOUND, "学生不存在");
        }
        
        // 这里可以实现获取需要同伴评价的任务逻辑
        // 暂时返回空列表
        return PageResponse.of(
                pageRequest.getPageNum(),
                pageRequest.getPageSize(),
                0L,
                List.<TaskDTO.PeerReviewTaskResponse>of()
        );
    }

    @Override
    @Transactional
    public Boolean submitPeerReview(TaskDTO.PeerReviewRequest reviewRequest, Long studentId) {
        log.info("提交同伴评价，任务ID: {}, 学生ID: {}", reviewRequest.getTaskId(), studentId);
        
        // 验证任务是否存在
        Task task = taskMapper.selectById(reviewRequest.getTaskId());
        if (task == null) {
            throw new BusinessException(ResultCode.TASK_NOT_FOUND, "任务不存在");
        }
        
        // 验证权限
        if (!hasTaskAccess(reviewRequest.getTaskId(), studentId)) {
            throw new BusinessException(ResultCode.ACCESS_DENIED, "无权限访问该任务");
        }
        
        // 这里可以实现提交同伴评价的逻辑
        // 暂时返回true表示操作成功
        return true;
    }

    @Override
    public List<TaskDTO.PeerReviewResponse> getMyPeerReviews(Long taskId, Long studentId) {
        log.info("获取我的同伴评价，任务ID: {}, 学生ID: {}", taskId, studentId);
        
        // 验证权限
        if (!hasTaskAccess(taskId, studentId)) {
            throw new BusinessException(ResultCode.ACCESS_DENIED, "无权限访问该任务");
        }
        
        // 这里可以实现获取同伴评价的逻辑
        // 暂时返回空列表
        return List.of();
    }

    @Override
    @Transactional
    public Boolean requestExtension(TaskDTO.TaskExtensionRequest extensionRequest, Long studentId) {
        log.info("申请任务延期，任务ID: {}, 学生ID: {}", extensionRequest.getTaskId(), studentId);
        
        // 验证任务是否存在
        Task task = taskMapper.selectById(extensionRequest.getTaskId());
        if (task == null) {
            throw new BusinessException(ResultCode.TASK_NOT_FOUND, "任务不存在");
        }
        
        // 验证权限
        if (!hasTaskAccess(extensionRequest.getTaskId(), studentId)) {
            throw new BusinessException(ResultCode.ACCESS_DENIED, "无权限访问该任务");
        }
        
        // 检查是否已过截止时间
        if (task.getDueTime() != null && task.getDueTime().isBefore(LocalDateTime.now())) {
            throw new BusinessException(ResultCode.TASK_OVERDUE, "任务已过期，无法申请延期");
        }
        
        // 这里可以实现申请延期的逻辑
        // 暂时返回true表示操作成功
        return true;
    }

    @Override
    public TaskDTO.ExtensionStatusResponse getExtensionStatus(Long taskId, Long studentId) {
        log.info("获取延期申请状态，任务ID: {}, 学生ID: {}", taskId, studentId);
        
        // 验证权限
        if (!hasTaskAccess(taskId, studentId)) {
            throw new BusinessException(ResultCode.ACCESS_DENIED, "无权限访问该任务");
        }
        
        TaskDTO.ExtensionStatusResponse response = new TaskDTO.ExtensionStatusResponse();
        response.setTaskId(taskId);
        response.setStatus("NONE"); // 默认无延期申请
        response.setReason("");
        
        return response;
    }

    @Override
    public TaskDTO.TaskProgressResponse getTaskProgress(Long taskId, Long studentId) {
        log.info("获取任务进度，任务ID: {}, 学生ID: {}", taskId, studentId);
        
        // 验证权限
        if (!hasTaskAccess(taskId, studentId)) {
            throw new BusinessException(ResultCode.ACCESS_DENIED, "无权限访问该任务");
        }
        
        // 获取任务提交记录
        QueryWrapper<TaskSubmission> queryWrapper = new QueryWrapper<>();
        queryWrapper.eq("task_id", taskId)
                   .eq("student_id", studentId)
                   .orderByDesc("create_time")
                   .last("LIMIT 1");
        
        TaskSubmission submission = taskSubmissionMapper.selectOne(queryWrapper);
        
        TaskDTO.TaskProgressResponse response = new TaskDTO.TaskProgressResponse();
        response.setTaskId(taskId);
        response.setStudentId(studentId);
        
        if (submission != null) {
            response.setProgressPercentage(submission.getStatus().equals("SUBMITTED") ? 100 : 50);
            response.setStatus(submission.getStatus());
            response.setLastUpdateTime(submission.getUpdateTime());
        } else {
            response.setProgressPercentage(0);
            response.setStatus("NOT_STARTED");
            response.setLastUpdateTime(null);
        }
        
        return response;
    }

    @Override
    @Transactional
    public Boolean updateTaskProgress(TaskDTO.TaskProgressUpdateRequest progressRequest, Long studentId) {
        log.info("更新任务进度，任务ID: {}, 学生ID: {}, 进度: {}%", progressRequest.getTaskId(), studentId, progressRequest.getProgressPercentage());
        
        // 验证权限
        if (!hasTaskAccess(progressRequest.getTaskId(), studentId)) {
            throw new BusinessException(ResultCode.ACCESS_DENIED, "无权限访问该任务");
        }
        
        // 这里可以实现更新进度的逻辑
        // 暂时返回true表示操作成功
        return true;
    }

    @Override
    public TaskDTO.TaskSuggestionResponse getTaskSuggestion(Long taskId, Long studentId) {
        log.info("获取任务建议，任务ID: {}, 学生ID: {}", taskId, studentId);
        
        // 验证权限
        if (!hasTaskAccess(taskId, studentId)) {
            throw new BusinessException(ResultCode.ACCESS_DENIED, "无权限访问该任务");
        }
        
        Task task = taskMapper.selectById(taskId);
        if (task == null) {
            throw new BusinessException(ResultCode.TASK_NOT_FOUND, "任务不存在");
        }
        
        TaskDTO.TaskSuggestionResponse response = new TaskDTO.TaskSuggestionResponse();
        response.setSuggestionId(1L);
        response.setTitle("任务改进建议");
        response.setContent("建议增加练习时间，可以参考相关资料，注意任务截止时间");
        response.setSuggestionType("IMPROVEMENT");
        response.setPriority("MEDIUM");
        response.setStatus("PENDING");
        
        return response;
    }

    @Override
    public List<TaskDTO.RecommendedMaterialResponse> getRecommendedMaterials(Long taskId, Long studentId) {
        log.info("获取推荐材料，任务ID: {}, 学生ID: {}", taskId, studentId);
        
        // 验证权限
        if (!hasTaskAccess(taskId, studentId)) {
            throw new BusinessException(ResultCode.ACCESS_DENIED, "无权限访问该任务");
        }
        
        // 这里可以实现获取推荐材料的逻辑
        // 暂时返回空列表
        return List.of();
    }

    @Override
    public TaskDTO.TaskCompletionReportResponse getTaskCompletionReport(Long studentId, String timeRange) {
        log.info("获取任务完成报告，学生ID: {}, 时间范围: {}", studentId, timeRange);
        
        Student student = studentMapper.selectById(studentId);
        if (student == null) {
            throw new BusinessException(ResultCode.USER_NOT_FOUND, "学生不存在");
        }
        
        TaskDTO.TaskCompletionReportResponse response = new TaskDTO.TaskCompletionReportResponse();
        response.setTaskId(1L);
        response.setTaskTitle("任务完成报告");
        response.setTotalStudents(30);
        response.setCompletedStudents(25);
        response.setSubmittedStudents(28);
        response.setGradedStudents(20);
        response.setAverageScore(BigDecimal.ZERO);
        
        return response;
    }

    @Override
    public TaskDTO.ExportResponse exportTaskData(Long studentId, TaskDTO.TaskDataExportRequest exportRequest) {
        log.info("导出任务数据，学生ID: {}", studentId);
        
        Student student = studentMapper.selectById(studentId);
        if (student == null) {
            throw new BusinessException(ResultCode.USER_NOT_FOUND, "学生不存在");
        }
        
        TaskDTO.ExportResponse response = new TaskDTO.ExportResponse();
        response.setDownloadUrl("/api/export/download/" + studentId);
        response.setFileName("task_data_" + studentId + ".xlsx");
        response.setExpiryTime(LocalDateTime.now().plusDays(7));
        
        return response;
    }

    // 辅助方法：验证学生是否有权限访问任务
    private boolean hasTaskAccess(Long taskId, Long studentId) {
        Student student = studentMapper.selectById(studentId);
        if (student == null) {
            return false;
        }
        
        Task task = taskMapper.selectById(taskId);
        if (task == null) {
            return false;
        }
        
        // 通过课程和班级关联验证权限
        QueryWrapper<Course> courseQuery = new QueryWrapper<>();
        courseQuery.eq("id", task.getCourseId())
                  .eq("class_id", student.getClassId());
        
        return courseMapper.selectCount(courseQuery) > 0;
    }
    
    // 辅助方法：获取任务提交记录
    private TaskSubmission getTaskSubmissionByTaskAndStudent(Long taskId, Long studentId) {
        QueryWrapper<TaskSubmission> queryWrapper = new QueryWrapper<>();
        queryWrapper.eq("task_id", taskId)
                   .eq("student_id", studentId)
                   .orderByDesc("create_time")
                   .last("LIMIT 1");
        
        return taskSubmissionMapper.selectOne(queryWrapper);
    }
    
    // 辅助方法：转换任务为列表响应对象
    private TaskDTO.TaskListResponse convertToTaskListResponse(Task task) {
        TaskDTO.TaskListResponse response = new TaskDTO.TaskListResponse();
        response.setTaskId(task.getId());
        response.setTitle(task.getTitle());
        response.setTaskType(task.getTaskType());
        response.setDifficulty(task.getDifficultyLevel());
        response.setEndTime(task.getDueTime());
        response.setTotalScore(task.getMaxScore());
        response.setStatus("ACTIVE");
        return response;
    }
}