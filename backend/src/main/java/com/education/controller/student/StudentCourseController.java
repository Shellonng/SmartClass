package com.education.controller.student;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.education.dto.common.PageResponse;
import com.education.dto.common.Result;
import com.education.dto.CourseResourceDTO;
import com.education.entity.Course;
import com.education.entity.CourseStudent;
import com.education.entity.Student;
import com.education.entity.User;
import com.education.entity.Chapter;
import com.education.entity.Section;
import com.education.entity.Teacher;
import com.education.entity.Assignment;
import com.education.mapper.CourseMapper;
import com.education.mapper.CourseStudentMapper;
import com.education.mapper.StudentMapper;
import com.education.mapper.UserMapper;
import com.education.mapper.ChapterMapper;
import com.education.mapper.SectionMapper;
import com.education.mapper.TeacherMapper;
import com.education.mapper.AssignmentMapper;
import com.education.service.teacher.CourseResourceService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.web.bind.annotation.*;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;
import java.util.HashMap;
import java.util.Map;
import java.util.Date;

/**
 * 学生课程控制器
 */
@Tag(name = "学生-课程管理", description = "学生课程相关接口")
@RestController
@RequestMapping("/api/student/courses")
public class StudentCourseController {

    private static final Logger logger = LoggerFactory.getLogger(StudentCourseController.class);

    @Autowired
    private CourseMapper courseMapper;

    @Autowired
    private UserMapper userMapper;

    @Autowired
    private StudentMapper studentMapper;
    
    @Autowired
    private CourseStudentMapper courseStudentMapper;

    @Autowired
    private ChapterMapper chapterMapper;
    
    @Autowired
    private SectionMapper sectionMapper;

    @Autowired
    private TeacherMapper teacherMapper;

    @Autowired
    private AssignmentMapper assignmentMapper;

    @Autowired
    private CourseResourceService courseResourceService;

    @Operation(summary = "获取学生课程列表", description = "获取当前登录学生已选的课程列表")
    @GetMapping
    public Result<PageResponse<Course>> getStudentCourses(
            @Parameter(description = "页码") @RequestParam(defaultValue = "0") Integer page,
            @Parameter(description = "每页大小") @RequestParam(defaultValue = "10") Integer size,
            @Parameter(description = "关键词搜索") @RequestParam(required = false) String keyword) {
        
        logger.info("获取学生课程列表 - 页码: {}, 每页大小: {}, 关键词: {}", page, size, keyword);
        
        try {
            // 获取当前登录用户
            Authentication authentication = SecurityContextHolder.getContext().getAuthentication();
            String username = authentication.getName();
            
            // 获取用户ID
            User user = userMapper.selectOne(new LambdaQueryWrapper<User>().eq(User::getUsername, username));
            if (user == null) {
                return Result.error("用户不存在");
            }
            
            // 获取学生ID
            Student student = studentMapper.selectOne(new LambdaQueryWrapper<Student>().eq(Student::getUserId, user.getId()));
            if (student == null) {
                return Result.error("学生信息不存在");
            }
            
            // 从course_student表中查询该学生已选课程的课程ID
            LambdaQueryWrapper<CourseStudent> csQueryWrapper = new LambdaQueryWrapper<>();
            csQueryWrapper.eq(CourseStudent::getStudentId, student.getId());
            List<CourseStudent> courseStudents = courseStudentMapper.selectList(csQueryWrapper);
            
            // 如果学生没有选课
            if (courseStudents.isEmpty()) {
                PageResponse<Course> emptyResponse = new PageResponse<>();
                emptyResponse.setRecords(Collections.emptyList());
                emptyResponse.setTotal(0);
                emptyResponse.setCurrent(page);
                emptyResponse.setPageSize(size);
                emptyResponse.setPages(0);
                return Result.success(emptyResponse);
            }
            
            // 提取课程ID列表
            List<Long> courseIds = courseStudents.stream()
                .map(CourseStudent::getCourseId)
                .collect(Collectors.toList());
            
            logger.info("查询到学生选课ID: {}", courseIds);
            
            // 构建课程查询条件
            LambdaQueryWrapper<Course> courseQueryWrapper = new LambdaQueryWrapper<>();
            courseQueryWrapper.in(Course::getId, courseIds);
            
            // 添加关键词搜索条件
            if (keyword != null && !keyword.isEmpty()) {
                courseQueryWrapper.and(wrapper -> 
                    wrapper.like(Course::getTitle, keyword)
                           .or()
                           .like(Course::getDescription, keyword)
                );
            }
            
            // 按创建时间降序排序
            courseQueryWrapper.orderByDesc(Course::getCreateTime);
            
            // 计算总数
            int total = courseMapper.selectCount(courseQueryWrapper).intValue();
            
            // 设置分页
            Page<Course> coursePage = new Page<>(page + 1, size);  // MyBatisPlus分页从1开始
            
            // 查询数据并检查结果
            Page<Course> result = courseMapper.selectPage(coursePage, courseQueryWrapper);
            List<Course> records = result.getRecords();
            
            // 构建响应对象
            PageResponse<Course> response = new PageResponse<>();
            response.setRecords(records);
            response.setTotal(total);
            response.setCurrent(page);
            response.setPageSize(size);
            response.setPages((int) Math.ceil((double) total / size));
            // 移除不存在的方法
            // response.setHasNext(page < Math.ceil((double) total / size) - 1);
            // response.setHasPrevious(page > 0);
            response.setFirst(page == 0);
            response.setLast(page >= Math.ceil((double) total / size) - 1);
            response.setEmpty(records == null || records.isEmpty());
            
            logger.info("查询到学生课程总数: {}, 实际返回课程数: {}", total, records != null ? records.size() : 0);
            
            return Result.success(response);
        } catch (Exception e) {
            logger.error("获取学生课程列表异常", e);
            return Result.error("获取课程列表失败: " + e.getMessage());
        }
    }

    /**
     * 获取学生已选课程列表（不分页，用于下拉选择）
     * @return 课程列表
     */
    @Operation(summary = "获取学生已选课程列表", description = "获取当前登录学生已选的所有课程列表，不分页")
    @GetMapping("/enrolled")
    public Result<List<Course>> getEnrolledCourses() {
        logger.info("获取学生已选课程列表");
        
        try {
            // 获取当前登录用户ID
            Authentication authentication = SecurityContextHolder.getContext().getAuthentication();
            String username = authentication.getName();
            
            // 获取用户ID
            User user = userMapper.selectOne(new LambdaQueryWrapper<User>().eq(User::getUsername, username));
            if (user == null) {
                return Result.error("用户不存在");
            }
            
            // 获取学生ID
            Student student = studentMapper.selectOne(new LambdaQueryWrapper<Student>().eq(Student::getUserId, user.getId()));
            if (student == null) {
                return Result.error("学生信息不存在");
            }
            
            // 从course_student表中查询该学生已选课程的课程ID
            LambdaQueryWrapper<CourseStudent> csQueryWrapper = new LambdaQueryWrapper<>();
            csQueryWrapper.eq(CourseStudent::getStudentId, student.getId());
            List<CourseStudent> courseStudents = courseStudentMapper.selectList(csQueryWrapper);
            
            // 如果学生没有选课
            if (courseStudents.isEmpty()) {
                return Result.success(Collections.emptyList());
            }
            
            // 提取课程ID列表
            List<Long> courseIds = courseStudents.stream()
                .map(CourseStudent::getCourseId)
                .collect(Collectors.toList());
            
            // 查询课程信息
            LambdaQueryWrapper<Course> courseQueryWrapper = new LambdaQueryWrapper<>();
            courseQueryWrapper.in(Course::getId, courseIds);
            courseQueryWrapper.orderByDesc(Course::getCreateTime);
            
            List<Course> courses = courseMapper.selectList(courseQueryWrapper);
            
            logger.info("查询到学生已选课程数: {}", courses.size());
            
            return Result.success(courses);
        } catch (Exception e) {
            logger.error("获取学生已选课程列表异常", e);
            return Result.error("获取已选课程列表失败: " + e.getMessage());
        }
    }

    /**
     * 获取课程详情，包括章节和小节信息
     * @param courseId 课程ID
     * @return 课程详情
     */
    @GetMapping("/{courseId}")
    public Result<Map<String, Object>> getCourseDetail(@PathVariable Long courseId) {
        logger.info("获取学生课程详情，课程ID: {}", courseId);
        
        try {
            // 1. 获取课程基本信息
            Course course = courseMapper.selectById(courseId);
            if (course == null) {
                logger.error("课程不存在，课程ID: {}", courseId);
                return Result.error("课程不存在");
            }
            
            // 2. 获取教师真实姓名
            if (course.getTeacherId() != null) {
                // 根据教师ID获取教师信息
                Teacher teacher = teacherMapper.selectById(course.getTeacherId());
                if (teacher != null && teacher.getUserId() != null) {
                    // 根据用户ID获取用户信息
                    User user = userMapper.selectById(teacher.getUserId());
                    if (user != null && user.getRealName() != null) {
                        // 将教师姓名添加到课程对象中
                        course.setTeacherName(user.getRealName());
                        logger.info("获取到教师真实姓名: {}, 教师ID: {}, 用户ID: {}", 
                                   user.getRealName(), course.getTeacherId(), teacher.getUserId());
                    } else {
                        logger.warn("未找到教师对应的用户信息或用户没有真实姓名, 教师ID: {}", course.getTeacherId());
                    }
                } else {
                    logger.warn("未找到教师信息, 教师ID: {}", course.getTeacherId());
                }
            }
            
            // 3. 获取课程章节
            List<Chapter> chapters = chapterMapper.selectList(
                new LambdaQueryWrapper<Chapter>()
                    .eq(Chapter::getCourseId, courseId)
                    .orderByAsc(Chapter::getSortOrder)
            );
            
            List<Map<String, Object>> chapterList = new ArrayList<>();
            
            // 4. 遍历章节，获取每个章节的小节
            for (Chapter chapter : chapters) {
                Map<String, Object> chapterMap = new HashMap<>();
                chapterMap.put("id", chapter.getId());
                chapterMap.put("title", chapter.getTitle());
                chapterMap.put("description", chapter.getDescription());
                chapterMap.put("sortOrder", chapter.getSortOrder());
                
                // 获取当前章节的所有小节
                List<Section> sections = sectionMapper.selectList(
                    new LambdaQueryWrapper<Section>()
                        .eq(Section::getChapterId, chapter.getId())
                        .orderByAsc(Section::getSortOrder)
                );
                
                chapterMap.put("sections", sections);
                chapterList.add(chapterMap);
            }
            
            // 5. 构建返回数据
            Map<String, Object> result = new HashMap<>();
            result.put("course", course);
            result.put("chapters", chapterList);
            
            logger.info("成功获取课程详情，课程ID: {}, 章节数: {}", courseId, chapters.size());
            return Result.success(result);
            
        } catch (Exception e) {
            logger.error("获取课程详情失败", e);
            return Result.error("获取课程详情失败: " + e.getMessage());
        }
    }

    /**
     * 获取课程资源列表
     * @param courseId 课程ID
     * @return 课程资源列表
     */
    @GetMapping("/{courseId}/resources")
    public Result<List<CourseResourceDTO>> getCourseResources(@PathVariable Long courseId) {
        logger.info("获取学生课程资源，课程ID: {}", courseId);
        
        try {
            List<CourseResourceDTO> resources = courseResourceService.listResources(courseId);
            return Result.success(resources);
        } catch (Exception e) {
            logger.error("获取学生课程资源失败: {}", e.getMessage(), e);
            return Result.error("获取课程资源失败: " + e.getMessage());
        }
    }

    /**
     * 获取课程任务列表（作业和考试）
     * @param courseId 课程ID
     * @return 课程任务列表
     */
    @GetMapping("/{courseId}/assignments")
    public Result<List<Map<String, Object>>> getCourseTasks(@PathVariable Long courseId) {
        logger.info("获取学生课程任务，课程ID: {}", courseId);
        
        try {
            // 查询该课程的所有已发布任务（作业和考试）
            LambdaQueryWrapper<Assignment> queryWrapper = new LambdaQueryWrapper<>();
            queryWrapper.eq(Assignment::getCourseId, courseId);
            // 只查询已发布的任务（status=1）
            queryWrapper.eq(Assignment::getStatus, 1);
            queryWrapper.orderByDesc(Assignment::getCreateTime);
            
            List<Assignment> assignments = assignmentMapper.selectList(queryWrapper);
            
            // 转换为前端需要的格式
            List<Map<String, Object>> taskList = new ArrayList<>();
            
            for (Assignment assignment : assignments) {
                Map<String, Object> task = new HashMap<>();
                task.put("id", assignment.getId());
                task.put("title", assignment.getTitle());
                task.put("description", assignment.getDescription());
                task.put("type", assignment.getType());
                task.put("mode", assignment.getMode());
                task.put("startTime", assignment.getStartTime());
                task.put("endTime", assignment.getEndTime());
                task.put("createTime", assignment.getCreateTime());
                
                // 根据当前时间和截止时间判断状态
                Date now = new Date();
                String status;
                
                if (assignment.getEndTime() != null && now.after(assignment.getEndTime())) {
                    status = "completed"; // 已截止
                } else if (assignment.getStartTime() != null && now.before(assignment.getStartTime())) {
                    status = "pending"; // 未开始
                } else {
                    status = "in_progress"; // 进行中
                }
                
                task.put("status", status);
                
                // 添加到列表
                taskList.add(task);
            }
            
            return Result.success(taskList);
        } catch (Exception e) {
            logger.error("获取学生课程任务失败: {}", e.getMessage(), e);
            return Result.error("获取课程任务失败: " + e.getMessage());
        }
    }

    /**
     * 获取学生所有已选课程的资源列表
     * @return 资源列表
     */
    @GetMapping("/resources")
    public Result<List<CourseResourceDTO>> getAllStudentResources() {
        logger.info("获取学生所有课程资源");
        
        try {
            // 获取当前登录用户
            Authentication authentication = SecurityContextHolder.getContext().getAuthentication();
            String username = authentication.getName();
            
            // 获取用户ID
            User user = userMapper.selectOne(new LambdaQueryWrapper<User>().eq(User::getUsername, username));
            if (user == null) {
                return Result.error("用户不存在");
            }
            
            // 获取学生ID
            Student student = studentMapper.selectOne(new LambdaQueryWrapper<Student>().eq(Student::getUserId, user.getId()));
            if (student == null) {
                return Result.error("学生信息不存在");
            }
            
            // 从course_student表中查询该学生已选课程的课程ID
            LambdaQueryWrapper<CourseStudent> csQueryWrapper = new LambdaQueryWrapper<>();
            csQueryWrapper.eq(CourseStudent::getStudentId, student.getId());
            List<CourseStudent> courseStudents = courseStudentMapper.selectList(csQueryWrapper);
            
            // 如果学生没有选课
            if (courseStudents.isEmpty()) {
                return Result.success(Collections.emptyList());
            }
            
            // 提取课程ID列表
            List<Long> courseIds = courseStudents.stream()
                .map(CourseStudent::getCourseId)
                .collect(Collectors.toList());
            
            logger.info("查询到学生选课ID: {}", courseIds);
            
            // 存储所有课程的资源
            List<CourseResourceDTO> allResources = new ArrayList<>();
            
            // 依次获取每个课程的资源
            for (Long courseId : courseIds) {
                List<CourseResourceDTO> resources = courseResourceService.listResources(courseId);
                if (resources != null && !resources.isEmpty()) {
                    // 设置课程名称
                    Course course = courseMapper.selectById(courseId);
                    if (course != null) {
                        for (CourseResourceDTO resource : resources) {
                            // 确保课程ID和课程名称被正确设置
                            resource.setCourseId(courseId);
                        }
                    }
                    allResources.addAll(resources);
                }
            }
            
            return Result.success(allResources);
        } catch (Exception e) {
            logger.error("获取学生所有课程资源失败: {}", e.getMessage(), e);
            return Result.error("获取学生所有课程资源失败: " + e.getMessage());
        }
    }
} 