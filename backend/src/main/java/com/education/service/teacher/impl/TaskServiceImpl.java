package com.education.service.teacher.impl;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.education.dto.common.PageRequest;
import com.education.dto.common.PageResponse;
import com.education.entity.Class;
import com.education.entity.Course;
import com.education.entity.Task;
import com.education.mapper.ClassMapper;
import com.education.mapper.CourseMapper;
import com.education.mapper.TaskMapper;
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
        // TODO: Add weight field to TaskCreateRequest or use other way to set weight
        task.setWeight(BigDecimal.valueOf(1.0)); // Default weight 1.0
        
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
        
        // TODO: Add weight field to TaskUpdateRequest or use other way to set weight
        // task.setWeight(request.getWeight());
        
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
    
    // The following methods return default values temporarily, to be improved later
    
    @Override
    public PageResponse<Object> getTaskSubmissions(Long taskId, Long teacherId, PageRequest pageRequest) {
        log.info("Getting task submissions, taskId: {}, teacherId: {}", taskId, teacherId);
        validateTaskAccess(taskId, teacherId);
        // TODO: Implement submission list query
        return new PageResponse<>(
                (long) pageRequest.getPageNum(), (long) pageRequest.getPageSize(), 0L, new ArrayList<>());
    }
    
    @Override
    @Transactional
    public Boolean gradeSubmission(Long submissionId, Object gradeRequest, Long teacherId) {
        log.info("Grading submission, submissionId: {}, teacherId: {}", submissionId, teacherId);
        // TODO: Implement submission grading logic
        return true;
    }
    
    @Override
    @Transactional
    public Boolean batchGradeSubmissions(List<Object> gradeRequests, Long teacherId) {
        log.info("Batch grading submissions, count: {}, teacherId: {}", gradeRequests.size(), teacherId);
        // TODO: Implement batch grading logic
        return true;
    }
    
    @Override
    public TaskDTO.TaskStatisticsResponse getTaskStatistics(Long taskId, Long teacherId) {
        log.info("Getting task statistics, taskId: {}, teacherId: {}", taskId, teacherId);
        validateTaskAccess(taskId, teacherId);
        // TODO: Implement task statistics logic
        return new TaskDTO.TaskStatisticsResponse();
    }
    
    @Override
    public String exportTaskGrades(Long taskId, Long teacherId) {
        log.info("Exporting task grades, taskId: {}, teacherId: {}", taskId, teacherId);
        validateTaskAccess(taskId, teacherId);
        // TODO: Implement grade export logic
        return "/tmp/task_grades_" + taskId + ".xlsx";
    }
    
    @Override
    @Transactional
    public TaskDTO.TaskResponse copyTask(Long taskId, String newTitle, Long teacherId) {
        log.info("Copying task, taskId: {}, newTitle: {}, teacherId: {}", taskId, newTitle, teacherId);
        validateTaskAccess(taskId, teacherId);
        // TODO: Implement task copy logic
        return new TaskDTO.TaskResponse();
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
        // TODO: Implement create task from template logic
        return new TaskDTO.TaskResponse();
    }
    
    @Override
    @Transactional
    public Boolean saveTaskAsTemplate(Long taskId, String templateName, Long teacherId) {
        log.info("Saving task as template, taskId: {}, templateName: {}, teacherId: {}", taskId, templateName, teacherId);
        validateTaskAccess(taskId, teacherId);
        // TODO: Implement save template logic
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
        // TODO: Implement task import logic
        return true;
    }
    
    @Override
    public String exportTasks(Long taskId, Long teacherId) {
        log.info("Exporting tasks, taskId: {}, teacherId: {}", taskId, teacherId);
        validateTaskAccess(taskId, teacherId);
        // TODO: Implement task export logic
        return "export_url";
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
    public Boolean archiveTask(Long taskId, Long teacherId) {
        log.info("Archiving task, taskId: {}, teacherId: {}", taskId, teacherId);
        validateTaskAccess(taskId, teacherId);
        // TODO: Implement task archive logic
        return true;
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
        // TODO: Implement task export logic
        return "/tmp/task_export_" + taskId + ".json";
    }

    @Override
    public Boolean setTaskGradingCriteria(Long taskId, Object gradingCriteria, Long teacherId) {
        log.info("Setting task grading criteria, taskId: {}, teacherId: {}", taskId, teacherId);
        validateTaskAccess(taskId, teacherId);
        // TODO: Implement grading criteria setting logic
        return true;
    }

    @Override
    public Object getTaskGradingCriteria(Long taskId, Long teacherId) {
        log.info("Getting task grading criteria, taskId: {}, teacherId: {}", taskId, teacherId);
        validateTaskAccess(taskId, teacherId);
        // TODO: Implement grading criteria query logic
        return new Object();
    }
}