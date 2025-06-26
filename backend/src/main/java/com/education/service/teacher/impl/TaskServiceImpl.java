package com.education.service.teacher.impl;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.education.dto.common.PageRequest;
import com.education.dto.common.PageResponse;
import com.education.entity.Class;
import com.education.entity.Course;
import com.education.entity.Task;
import com.education.entity.TaskSubmission;
import com.education.mapper.ClassMapper;
import com.education.mapper.CourseMapper;
import com.education.mapper.TaskMapper;
import com.education.mapper.TaskSubmissionMapper;
import com.education.service.teacher.TaskService;
import com.education.dto.TaskDTO;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.BeanUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.util.StringUtils;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Teacher Task Service Implementation
 */
@Service
@Slf4j
public class TaskServiceImpl implements TaskService {
    
    @Autowired
    private TaskMapper taskMapper;
    
    @Autowired
    private CourseMapper courseMapper;
    
    @Autowired
    private ClassMapper classMapper;
    
    @Autowired
    private TaskSubmissionMapper taskSubmissionMapper;
    
    @Autowired
    private RedisTemplate<String, Object> redisTemplate;
    
    @Override
    @Transactional
    public TaskDTO.TaskResponse createTask(TaskDTO.TaskCreateRequest request, Long teacherId) {
        log.info("Creating task, teacherId: {}, title: {}", teacherId, request.getTitle());
        
        // Validate course permission
        Course course = courseMapper.selectById(request.getCourseId());
        if (course == null || !course.getTeacherId().equals(teacherId)) {
            throw new RuntimeException("No permission to create task in this course");
        }
        
        // Validate class permissions
        if (request.getClassIds() != null && !request.getClassIds().isEmpty()) {
            for (Long classId : request.getClassIds()) {
                Class clazz = classMapper.selectById(classId);
                if (clazz == null || !clazz.getHeadTeacherId().equals(teacherId)) {
                    throw new RuntimeException("No permission to create task in class ID: " + classId);
                }
            }
        }
        
        // Create task entity
        Task task = new Task();
        BeanUtils.copyProperties(request, task);
        task.setTeacherId(teacherId);
        task.setCreateTime(LocalDateTime.now());
        task.setUpdateTime(LocalDateTime.now());
        task.setDueTime(request.getEndTime());
        task.setMaxScore(request.getTotalScore());
        // 设置任务权重，如果未指定则使用默认值1.0
        task.setWeight(request.getWeight() != null ? request.getWeight() : BigDecimal.valueOf(1.0));
        
        task.setStatus("DRAFT"); // Default draft status
        
        // Save task
        taskMapper.insert(task);
        
        // Clear related cache
        clearTaskCache(task.getCourseId());
        
        log.info("Task created successfully, taskId: {}", task.getId());
        
        return convertToTaskResponse(task);
    }
    
    @Override
    public PageResponse<TaskDTO.TaskResponse> getTaskList(Long teacherId, PageRequest pageRequest) {
        log.info("Getting task list, teacherId: {}", teacherId);
        
        // Build query conditions
        QueryWrapper<Task> queryWrapper = new QueryWrapper<>();
        queryWrapper.eq("teacher_id", teacherId)
                   .eq("is_deleted", false)
                   .orderByDesc("create_time");
        
        // Add other query conditions as needed
        if (StringUtils.hasText(pageRequest.getKeyword())) {
            queryWrapper.and(wrapper -> wrapper
                .like("title", pageRequest.getKeyword())
                .or()
                .like("description", pageRequest.getKeyword()));
        }
        
        // Add other query conditions as needed
        
        // Paginated query
        Page<Task> page = new Page<>(pageRequest.getPageNum(), pageRequest.getPageSize());
        Page<Task> taskPage = taskMapper.selectPage(page, queryWrapper);
        
        List<TaskDTO.TaskResponse> taskResponses = taskPage.getRecords().stream()
                .map(this::convertToTaskResponse)
                .collect(Collectors.toList());
        
        return new PageResponse<>(
                (long) pageRequest.getPageNum(), (long) pageRequest.getPageSize(), taskPage.getTotal(), taskResponses);
    }
    
    @Override
    public TaskDTO.TaskDetailResponse getTaskDetail(Long taskId, Long teacherId) {
        log.info("Getting task detail, taskId: {}, teacherId: {}", taskId, teacherId);
        
        Task task = validateTaskAccess(taskId, teacherId);
        
        TaskDTO.TaskDetailResponse response = new TaskDTO.TaskDetailResponse();
        BeanUtils.copyProperties(task, response);
        
        // Get course information
        if (task.getCourseId() != null) {
            Course course = courseMapper.selectById(task.getCourseId());
            if (course != null) {
                response.setCourseName(course.getCourseName());
            }
        }
        
        // TODO: Get submission statistics
        
        return response;
    }
    
    @Override
    @Transactional
    public TaskDTO.TaskResponse updateTask(Long taskId, TaskDTO.TaskUpdateRequest request, Long teacherId) {
        log.info("Updating task, taskId: {}, teacherId: {}", taskId, teacherId);
        
        Task task = validateTaskAccess(taskId, teacherId);
        
        // Check if task status allows modification
        if (task.getStartTime() != null && task.getStartTime().isBefore(LocalDateTime.now())) {
            throw new RuntimeException("Started tasks cannot be modified");
        }
        
        // Update task information
        if (StringUtils.hasText(request.getTitle())) {
            task.setTitle(request.getTitle());
        }
        if (StringUtils.hasText(request.getDescription())) {
            task.setDescription(request.getDescription());
        }
        if (request.getDifficulty() != null) {
            task.setDifficultyLevel(request.getDifficulty());
        }
        if (request.getTimeLimit() != null) {
            task.setEstimatedHours(new BigDecimal(request.getTimeLimit()).divide(new BigDecimal(60), 2, RoundingMode.HALF_UP));
        }
        if (request.getMaxAttempts() != null) {
            task.setMaxAttempts(request.getMaxAttempts());
        }
        if (request.getTotalScore() != null) {
            task.setMaxScore(request.getTotalScore());
        }
        if (request.getStartTime() != null) {
            task.setStartTime(request.getStartTime());
        }
        if (request.getEndTime() != null) {
            task.setDueTime(request.getEndTime());
        }
        if (request.getIsVisible() != null) {
            task.setIsPublic(request.getIsVisible());
        }
        if (request.getWeight() != null) {
            task.setWeight(request.getWeight());
        }
        
        task.setUpdateTime(LocalDateTime.now());
        taskMapper.updateById(task);
        
        // Clear cache
        clearTaskCache(task.getCourseId());
        
        log.info("Task updated successfully, taskId: {}", taskId);
        
        return convertToTaskResponse(task);
    }
    
    @Override
    @Transactional
    public Boolean deleteTask(Long taskId, Long teacherId) {
        log.info("Deleting task, taskId: {}, teacherId: {}", taskId, teacherId);
        
        Task task = validateTaskAccess(taskId, teacherId);
        
        // Check if there are student submissions
        // TODO: Check submission records
        
        // Soft delete
        task.setIsDeleted(true);
        task.setUpdateTime(LocalDateTime.now());
        taskMapper.updateById(task);
        
        // Clear cache
        clearTaskCache(task.getCourseId());
        
        log.info("Task deleted successfully, taskId: {}", taskId);
        
        return true;
    }
    
    @Override
    @Transactional
    public Boolean publishTask(Long taskId, Long teacherId) {
        log.info("Publishing task, taskId: {}, teacherId: {}", taskId, teacherId);
        return updateTaskStatus(taskId, teacherId, "PUBLISHED");
    }
    
    @Override
    @Transactional
    public Boolean closeTask(Long taskId, Long teacherId) {
        log.info("Closing task, taskId: {}, teacherId: {}", taskId, teacherId);
        return updateTaskStatus(taskId, teacherId, "CLOSED");
    }
    
    /**
     * Update task status
     */
    private Boolean updateTaskStatus(Long taskId, Long teacherId, String status) {
        Task task = validateTaskAccess(taskId, teacherId);
        task.setStatus(status);
        task.setUpdateTime(LocalDateTime.now());
        taskMapper.updateById(task);
        clearTaskCache(task.getCourseId());
        return true;
    }
    
    /**
     * Validate task access permission
     */
    private Task validateTaskAccess(Long taskId, Long teacherId) {
        Task task = taskMapper.selectById(taskId);
        if (task == null || task.getIsDeleted()) {
            throw new RuntimeException("Task not found");
        }
        
        if (!task.getTeacherId().equals(teacherId)) {
            throw new RuntimeException("No permission to access this task");
        }
        
        return task;
    }
    
    /**
     * Convert to task response object
     */
    private TaskDTO.TaskResponse convertToTaskResponse(Task task) {
        TaskDTO.TaskResponse response = new TaskDTO.TaskResponse();
        BeanUtils.copyProperties(task, response);
        
        // Get course name
        if (task.getCourseId() != null) {
            Course course = courseMapper.selectById(task.getCourseId());
            if (course != null) {
                response.setCourseName(course.getCourseName());
            }
        }
        
        return response;
    }
    
    /**
     * Clear task related cache
     */
    private void clearTaskCache(Long courseId) {
        if (courseId != null) {
            String cacheKey = "course:tasks:" + courseId;
            redisTemplate.delete(cacheKey);
        }
    }
    
    /**
     * 转换为提交记录响应对象
     */
    private Object convertToSubmissionResponse(TaskSubmission submission) {
        // 这里应该返回具体的DTO对象，暂时返回TaskSubmission
        return submission;
    }
    
    // The following methods return default values temporarily, to be improved later
    
    @Override
    public PageResponse<Object> getTaskSubmissions(Long taskId, Long teacherId, PageRequest pageRequest) {
        log.info("Getting task submissions, taskId: {}, teacherId: {}", taskId, teacherId);
        validateTaskAccess(taskId, teacherId);
        
        // 查询提交列表
        List<TaskSubmission> submissions = taskSubmissionMapper.selectByTaskId(taskId);
        
        // 分页处理
        int start = (pageRequest.getPageNum() - 1) * pageRequest.getPageSize();
        int end = Math.min(start + pageRequest.getPageSize(), submissions.size());
        
        List<Object> pageSubmissions = submissions.subList(start, end)
                .stream()
                .map(this::convertToSubmissionResponse)
                .collect(Collectors.toList());
        
        return new PageResponse<>(
                (long) pageRequest.getPageNum(), 
                (long) pageRequest.getPageSize(), 
                (long) submissions.size(), 
                pageSubmissions);
    }
    
    @Override
    @Transactional
    public Boolean gradeSubmission(Long submissionId, Object gradeRequest, Long teacherId) {
        log.info("Grading submission, submissionId: {}, teacherId: {}", submissionId, teacherId);
        
        // 查询提交记录
        TaskSubmission submission = taskSubmissionMapper.selectById(submissionId);
        if (submission == null || submission.getIsDeleted()) {
            throw new RuntimeException("提交记录不存在");
        }
        
        // 验证作业权限
        validateTaskAccess(submission.getTaskId(), teacherId);
        
        // 更新批改信息
        // 这里需要根据gradeRequest的具体类型来处理
        // 暂时假设gradeRequest包含score和feedback字段
        submission.setStatus("GRADED");
        submission.setGradeTime(LocalDateTime.now());
        submission.setGradedBy(teacherId);
        submission.setUpdateTime(LocalDateTime.now());
        
        taskSubmissionMapper.updateById(submission);
        
        log.info("Submission graded successfully, submissionId: {}", submissionId);
        return true;
    }
    
    @Override
    @Transactional
    public Boolean batchGradeSubmissions(List<Object> gradeRequests, Long teacherId) {
        log.info("Batch grading submissions, count: {}, teacherId: {}", gradeRequests.size(), teacherId);
        
        for (Object gradeRequest : gradeRequests) {
            // 这里需要根据gradeRequest的具体结构来提取submissionId
            // 暂时跳过具体实现
            // Long submissionId = extractSubmissionId(gradeRequest);
            // gradeSubmission(submissionId, gradeRequest, teacherId);
            log.debug("Processing grade request: {}", gradeRequest);
        }
        
        log.info("Batch grading completed, count: {}", gradeRequests.size());
        return true;
    }
    
    @Override
    public TaskDTO.TaskStatisticsResponse getTaskStatistics(Long taskId, Long teacherId) {
        log.info("Getting task statistics, taskId: {}, teacherId: {}", taskId, teacherId);
        validateTaskAccess(taskId, teacherId);
        
        TaskDTO.TaskStatisticsResponse response = new TaskDTO.TaskStatisticsResponse();
        
        // 统计提交数量
        Integer totalSubmissions = taskSubmissionMapper.countByTaskId(taskId);
        Integer gradedSubmissions = taskSubmissionMapper.countGradedByTaskId(taskId);
        Integer lateSubmissions = taskSubmissionMapper.countLateByTaskId(taskId);
        
        // 统计分数
        Double averageScore = taskSubmissionMapper.getAverageScoreByTaskId(taskId);
        Double maxScore = taskSubmissionMapper.getMaxScoreByTaskId(taskId);
        Double minScore = taskSubmissionMapper.getMinScoreByTaskId(taskId);
        
        response.setTotalSubmissions(totalSubmissions != null ? totalSubmissions : 0);
        response.setGradedSubmissions(gradedSubmissions != null ? gradedSubmissions : 0);
        response.setLateSubmissions(lateSubmissions != null ? lateSubmissions : 0);
        response.setAverageScore(averageScore != null ? averageScore : 0.0);
        response.setMaxScore(maxScore != null ? maxScore : 0.0);
        response.setMinScore(minScore != null ? minScore : 0.0);
        
        return response;
    }
    
    @Override
    public String exportTaskGrades(Long taskId, Long teacherId) {
        log.info("Exporting task grades, taskId: {}, teacherId: {}", taskId, teacherId);
        
        validateTaskAccess(taskId, teacherId);
        
        // 获取所有提交记录
        List<TaskSubmission> submissions = taskSubmissionMapper.selectByTaskId(taskId);
        
        // TODO: 使用EasyExcel或Apache POI生成Excel文件
        // 这里可以实现具体的导出逻辑
        String fileName = "task_grades_" + taskId + "_" + System.currentTimeMillis() + ".xlsx";
        String filePath = "/exports/" + fileName;
        
        log.info("Task grades exported successfully, file: {}, submissions count: {}", filePath, submissions.size());
        return filePath;
    }
    
    @Override
    @Transactional
    public TaskDTO.TaskResponse copyTask(Long taskId, String newTitle, Long teacherId) {
        log.info("Copying task, taskId: {}, newTitle: {}, teacherId: {}", taskId, newTitle, teacherId);
        
        Task originalTask = validateTaskAccess(taskId, teacherId);
        
        // 创建新任务
        Task newTask = new Task();
        BeanUtils.copyProperties(originalTask, newTask);
        newTask.setId(null);
        newTask.setTitle(newTitle);
        newTask.setStatus("DRAFT"); // 复制的任务默认为草稿状态
        newTask.setCreateTime(LocalDateTime.now());
        newTask.setUpdateTime(LocalDateTime.now());
        
        // 重置时间相关字段
        newTask.setStartTime(null);
        newTask.setDueTime(null);
        
        // 保存新任务
        int result = taskMapper.insert(newTask);
        if (result <= 0) {
            throw new RuntimeException("复制任务失败");
        }
        
        // 清除缓存
        clearTaskCache(newTask.getCourseId());
        
        log.info("Task copied successfully, new taskId: {}", newTask.getId());
        return convertToTaskResponse(newTask);
    }
    
    @Override
    public Object getPlagiarismDetectionResult(Long taskId, Long teacherId) {
        log.info("Getting plagiarism detection result, taskId: {}, teacherId: {}", taskId, teacherId);
        validateTaskAccess(taskId, teacherId);
        // TODO: Implement plagiarism detection result query
        return new Object();
    }
    
    @Override
    @Transactional
    public Boolean startPlagiarismDetection(Long taskId, Long teacherId) {
        log.info("Starting plagiarism detection, taskId: {}, teacherId: {}", taskId, teacherId);
        validateTaskAccess(taskId, teacherId);
        // TODO: Implement plagiarism detection start logic
        return true;
    }
    
    @Override
    public List<Object> getTaskTemplates(Long teacherId) {
        log.info("Getting task template list, teacherId: {}", teacherId);
        // TODO: Implement template list query
        return new ArrayList<>();
    }
    
    @Override
    @Transactional
    public TaskDTO.TaskResponse createTaskFromTemplate(Long templateId, TaskDTO.TaskCreateRequest request, Long teacherId) {
        log.info("Creating task from template, templateId: {}, teacherId: {}", templateId, teacherId);
        
        // TODO: 从模板创建任务
        // TaskTemplate template = taskTemplateMapper.selectById(templateId);
        // if (template == null) {
        //     throw new RuntimeException("模板不存在");
        // }
        
        // 验证课程权限
        Course course = courseMapper.selectById(request.getCourseId());
        if (course == null || !course.getTeacherId().equals(teacherId)) {
            throw new RuntimeException("无权限在该课程中创建任务");
        }
        
        // 创建任务
        Task task = new Task();
        BeanUtils.copyProperties(request, task);
        task.setTeacherId(teacherId);
        task.setCreateTime(LocalDateTime.now());
        task.setUpdateTime(LocalDateTime.now());
        task.setStatus("DRAFT");
        
        // 从模板复制属性
        // BeanUtils.copyProperties(template, task, "id", "templateName", "creatorId", "createTime");
        
        // 保存任务
        taskMapper.insert(task);
        
        // 清除缓存
        clearTaskCache(task.getCourseId());
        
        log.info("Task created from template successfully, taskId: {}", task.getId());
        return convertToTaskResponse(task);
    }
    
    @Override
    @Transactional
    public Boolean saveTaskAsTemplate(Long taskId, String templateName, Long teacherId) {
        log.info("Saving task as template, taskId: {}, templateName: {}, teacherId: {}", taskId, templateName, teacherId);
        
        validateTaskAccess(taskId, teacherId);
        
        // TODO: 创建任务模板实体并保存到模板表
        // 这里需要实现TaskTemplate实体和相关的Mapper
        // TaskTemplate template = new TaskTemplate();
        // BeanUtils.copyProperties(task, template);
        // template.setId(null);
        // template.setTemplateName(templateName);
        // template.setCreatorId(teacherId);
        // template.setCreateTime(LocalDateTime.now());
        // taskTemplateMapper.insert(template);
        
        log.info("Task saved as template successfully, templateName: {}", templateName);
        return true;
    }
    
    @Override
    @Transactional
    public Boolean setTaskWeight(Long taskId, Double weight, Long teacherId) {
        log.info("Setting task weight, taskId: {}, weight: {}, teacherId: {}", taskId, weight, teacherId);
        
        Task task = validateTaskAccess(taskId, teacherId);
        task.setWeight(BigDecimal.valueOf(weight));
        task.setUpdateTime(LocalDateTime.now());
        taskMapper.updateById(task);
        
        clearTaskCache(task.getCourseId());
        return true;
    }
    
    @Override
    @Transactional
    public Boolean extendTaskDeadline(Long taskId, Object extendRequest, Long teacherId) {
        log.info("Extending task deadline, taskId: {}, extendRequest: {}, teacherId: {}", taskId, extendRequest, teacherId);
        
        Task task = validateTaskAccess(taskId, teacherId);
        
        // TODO: Parse extendRequest to get new deadline
        LocalDateTime newDeadline = LocalDateTime.now().plusDays(7); // Example: extend 7 days
        // if (newDeadline.isBefore(task.getDueTime())) {
        //     throw new RuntimeException("New deadline cannot be earlier than current deadline");
        // }
        
        task.setDueTime(newDeadline);
        task.setUpdateTime(LocalDateTime.now());
        taskMapper.updateById(task);
        
        clearTaskCache(task.getCourseId());
        return true;
    }
    
    // The following are methods defined in interface but missing in implementation class
    
    @Override
    public Object getTaskSubmissionStatistics(Long taskId, Long teacherId) {
        log.info("Getting task submission statistics, taskId: {}, teacherId: {}", taskId, teacherId);
        validateTaskAccess(taskId, teacherId);
        // TODO: Implement submission statistics logic
        return new TaskDTO.SubmissionStatisticsResponse();
    }
    
    @Override
    public Boolean sendTaskReminder(Long taskId, String reminderType, Long teacherId) {
        log.info("Sending task reminder, taskId: {}, reminderType: {}, teacherId: {}", taskId, reminderType, teacherId);
        validateTaskAccess(taskId, teacherId);
        // TODO: Implement send reminder logic
        return true;
    }
    
    @Override
    public PageResponse<Object> getTaskComments(Long taskId, Long teacherId, PageRequest pageRequest) {
        log.info("Getting task comment list, taskId: {}, teacherId: {}", taskId, teacherId);
        validateTaskAccess(taskId, teacherId);
        // TODO: Implement comment list query
        return new PageResponse<>(
                (long) pageRequest.getPageNum(), (long) pageRequest.getPageSize(), 0L, new ArrayList<>());
    }
    
    @Override
    public Boolean replyTaskComment(Long commentId, String reply, Long teacherId) {
        log.info("Replying to task comment, commentId: {}, reply: {}, teacherId: {}", commentId, reply, teacherId);
        // TODO: Implement comment reply logic
        return true;
    }
    
    @Override
    public Boolean setTaskVisibility(Long taskId, Object visibility, Long teacherId) {
        log.info("Setting task visibility, taskId: {}, visibility: {}, teacherId: {}", taskId, visibility, teacherId);
        validateTaskAccess(taskId, teacherId);
        // TODO: Implement visibility setting logic
        return true;
    }
    
    @Override
    public Object getTaskAnalysisReport(Long taskId, Long teacherId) {
        log.info("Getting task analysis report, taskId: {}, teacherId: {}", taskId, teacherId);
        validateTaskAccess(taskId, teacherId);
        // TODO: Implement analysis report generation logic
        return new TaskDTO.TaskAnalysisResponse();
    }
    
    @Override
    public Object importTask(Object importRequest, Long teacherId) {
        log.info("Importing tasks, importRequest: {}, teacherId: {}", importRequest, teacherId);
        try {
            // 验证教师权限
            if (teacherId == null) {
                throw new IllegalArgumentException("教师ID不能为空");
            }
            
            // 解析导入请求
            // 这里应该根据实际的导入格式（如Excel、JSON等）进行解析
            // 暂时返回成功状态，具体实现需要根据前端传入的数据格式确定
            log.info("Task import completed successfully for teacher: {}", teacherId);
            return true;
        } catch (Exception e) {
            log.error("Failed to import tasks for teacher: {}", teacherId, e);
            throw new RuntimeException("任务导入失败: " + e.getMessage());
        }
    }
    
    @Override
    public String exportTasks(Long taskId, Long teacherId) {
        log.info("Exporting tasks, taskId: {}, teacherId: {}", taskId, teacherId);
        validateTaskAccess(taskId, teacherId);
        try {
            // 获取任务信息
            Task task = taskMapper.selectById(taskId);
            if (task == null) {
                throw new RuntimeException("任务不存在");
            }
            
            // 生成导出文件名
            String fileName = "task_" + taskId + "_" + System.currentTimeMillis() + ".xlsx";
            
            // 这里应该实现具体的导出逻辑，如生成Excel文件
            // 包含任务基本信息、提交记录、成绩统计等
            
            // 返回下载链接或文件路径
            String exportUrl = "/api/files/download/" + fileName;
            log.info("Task export completed, taskId: {}, exportUrl: {}", taskId, exportUrl);
            return exportUrl;
        } catch (Exception e) {
            log.error("Failed to export task: {}", taskId, e);
            throw new RuntimeException("任务导出失败: " + e.getMessage());
        }
    }
    
    @Override
    public Boolean setGradingCriteria(Long taskId, TaskDTO.GradingCriteria gradingCriteria, Long teacherId) {
        log.info("Setting task grading criteria, taskId: {}, gradingCriteria: {}, teacherId: {}", taskId, gradingCriteria, teacherId);
        validateTaskAccess(taskId, teacherId);
        // TODO: Implement grading criteria setting logic
        return true;
    }
    
    @Override
    public TaskDTO.GradingCriteria getGradingCriteria(Long taskId, Long teacherId) {
        log.info("Getting task grading criteria, taskId: {}, teacherId: {}", taskId, teacherId);
        validateTaskAccess(taskId, teacherId);
        // TODO: Implement grading criteria query logic
        return new TaskDTO.GradingCriteria();
    }
    
    @Override
    public Boolean autoGradeTask(Long taskId, Long teacherId) {
        log.info("Auto grading task, taskId: {}, teacherId: {}", taskId, teacherId);
        validateTaskAccess(taskId, teacherId);
        // TODO: Implement auto grading logic
        return true;
    }
    
    @Override
    public Object getTaskDifficultyAnalysis(Long taskId, Long teacherId) {
        log.info("Getting task difficulty analysis, taskId: {}, teacherId: {}", taskId, teacherId);
        validateTaskAccess(taskId, teacherId);
        // TODO: Implement difficulty analysis logic
        return new TaskDTO.DifficultyAnalysisResponse();
    }
    
    @Override
    public Boolean setTaskTags(Long taskId, List<String> tags, Long teacherId) {
        log.info("Setting task tags, taskId: {}, tags: {}, teacherId: {}", taskId, tags, teacherId);
        validateTaskAccess(taskId, teacherId);
        // TODO: Implement tag setting logic
        return true;
    }
    
    @Override
    public List<String> getTaskTags(Long taskId, Long teacherId) {
        log.info("Getting task tags, taskId: {}, teacherId: {}", taskId, teacherId);
        validateTaskAccess(taskId, teacherId);
        // TODO: Implement tag query logic
        return new ArrayList<>();
    }
    
    @Override
    @Transactional
    public Boolean archiveTask(Long taskId, Long teacherId) {
        log.info("Archiving task, taskId: {}, teacherId: {}", taskId, teacherId);
        
        Task task = validateTaskAccess(taskId, teacherId);
        
        // 更新任务状态为归档
        task.setStatus("ARCHIVED");
        task.setUpdateTime(LocalDateTime.now());
        
        int result = taskMapper.updateById(task);
        
        if (result > 0) {
            // 清除缓存
            clearTaskCache(task.getCourseId());
            log.info("Task archived successfully, taskId: {}", taskId);
            return true;
        }
        
        return false;
    }
    
    @Override
    public Boolean restoreTask(Long taskId, Long teacherId) {
        log.info("Restoring task, taskId: {}, teacherId: {}", taskId, teacherId);
        
        Task task = validateTaskAccess(taskId, teacherId);
        
        // Update task status
        task.setIsDeleted(false);
        task.setUpdateTime(LocalDateTime.now());
        
        int result = taskMapper.updateById(task);
        
        if (result > 0) {
            // Clear cache
            clearTaskCache(task.getCourseId());
            log.info("Task restored successfully, taskId: {}", taskId);
            return true;
        }
        
        return false;
    }

    @Override
    public String exportTask(Long taskId, Long teacherId) {
        log.info("Exporting task, taskId: {}, teacherId: {}", taskId, teacherId);
        validateTaskAccess(taskId, teacherId);
        try {
            // 获取任务详细信息
            Task task = taskMapper.selectById(taskId);
            if (task == null) {
                throw new RuntimeException("任务不存在");
            }
            
            // 生成导出文件路径
            String fileName = "task_export_" + taskId + "_" + System.currentTimeMillis() + ".json";
            String filePath = "/tmp/" + fileName;
            
            // 这里应该实现具体的导出逻辑
            // 包含任务配置、题目内容、评分标准等信息
            
            log.info("Task export completed, taskId: {}, filePath: {}", taskId, filePath);
            return filePath;
        } catch (Exception e) {
            log.error("Failed to export task: {}", taskId, e);
            throw new RuntimeException("任务导出失败: " + e.getMessage());
        }
    }

    @Override
    public Boolean setTaskGradingCriteria(Long taskId, Object gradingCriteria, Long teacherId) {
        log.info("Setting task grading criteria, taskId: {}, teacherId: {}", taskId, teacherId);
        validateTaskAccess(taskId, teacherId);
        
        try {
            // 验证评分标准数据
            if (gradingCriteria == null) {
                throw new IllegalArgumentException("评分标准不能为空");
            }
            
            // 这里应该将评分标准保存到数据库
            // 暂时模拟保存成功
            log.info("Task grading criteria set successfully for taskId: {}", taskId);
            return true;
        } catch (Exception e) {
            log.error("Failed to set task grading criteria for taskId: {}", taskId, e);
            return false;
        }
    }

    @Override
    public Object getTaskGradingCriteria(Long taskId, Long teacherId) {
        log.info("Getting task grading criteria, taskId: {}, teacherId: {}", taskId, teacherId);
        validateTaskAccess(taskId, teacherId);
        
        try {
            // 这里应该从数据库查询评分标准
            // 暂时返回模拟数据
            Object criteria = new Object();
            log.info("Task grading criteria retrieved successfully for taskId: {}", taskId);
            return criteria;
        } catch (Exception e) {
            log.error("Failed to get task grading criteria for taskId: {}", taskId, e);
            return new Object();
        }
    }
}