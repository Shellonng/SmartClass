package com.education.service.teacher.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.education.dto.common.PageRequest;
import com.education.dto.common.PageResponse;
import com.education.entity.Course;
import com.education.entity.Teacher;
import com.education.exception.BusinessException;
import com.education.exception.ResultCode;
import com.education.mapper.CourseMapper;
import com.education.mapper.TeacherMapper;
import com.education.service.teacher.CourseService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.util.StringUtils;

import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * 教师课程服务实现类
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
@Service
public class CourseServiceImpl implements CourseService {

    private static final Logger logger = LoggerFactory.getLogger(CourseServiceImpl.class);

    @Autowired
    private CourseMapper courseMapper;
    
    @Autowired
    private TeacherMapper teacherMapper;
    
    @Override
    public PageResponse<Course> getTeacherCourses(String username, PageRequest pageRequest, String keyword, String status, String term) {
        logger.info("获取教师课程列表 - 用户名: {}, 页码: {}, 每页大小: {}, 关键词: {}, 状态: {}, 学期: {}", 
                username, pageRequest.getPage(), pageRequest.getSize(), keyword, status, term);
        
        try {
            // 根据用户名获取教师ID
            Teacher teacher = getTeacherByUsername(username);
            logger.info("找到教师信息 - 教师ID: {}, 用户ID: {}", teacher.getId(), teacher.getUserId());
            
            // 构建查询条件
            LambdaQueryWrapper<Course> queryWrapper = new LambdaQueryWrapper<>();
            queryWrapper.eq(Course::getTeacherId, teacher.getId());
            logger.info("设置查询条件 - 教师ID: {}", teacher.getId());
            
            // 添加关键词搜索
            if (StringUtils.hasText(keyword)) {
                queryWrapper.like(Course::getTitle, keyword);
                logger.info("添加关键词搜索条件: {}", keyword);
            }
            
            // 添加状态筛选
            if (StringUtils.hasText(status)) {
                queryWrapper.eq(Course::getStatus, status);
                logger.info("添加状态筛选条件: {}", status);
            }
            
            // 添加学期筛选
            if (StringUtils.hasText(term)) {
                queryWrapper.eq(Course::getTerm, term);
                logger.info("添加学期筛选条件: {}", term);
            }
            
            // 排序
            queryWrapper.orderByDesc(Course::getCreateTime);
            
            // 分页查询
            Page<Course> page = new Page<>(pageRequest.getPage(), pageRequest.getSize());
            Page<Course> coursePage = courseMapper.selectPage(page, queryWrapper);
            
            // 更新课程状态
            List<Course> updatedCourses = coursePage.getRecords().stream()
                    .map(this::updateCourseStatus)
                    .collect(Collectors.toList());
            
            // 使用简化的方式构建分页响应
            PageResponse<Course> response = new PageResponse<>();
            response.setRecords(updatedCourses);
            response.setTotal(coursePage.getTotal());
            response.setPageSize((int) coursePage.getSize());
            response.setCurrent((int) coursePage.getCurrent());

            // 计算总页数
            long total = coursePage.getTotal();
            int size = (int) coursePage.getSize();
            long pages = size > 0 ? (total + size - 1) / size : 0;
            response.setPages(pages);

            logger.info("查询到教师课程 - 总数: {}, 当前页: {}, 每页大小: {}", 
                    total, response.getCurrent(), response.getPageSize());
            
            return response;
        } catch (BusinessException e) {
            logger.error("获取教师课程列表业务异常: {}", e.getMessage());
            throw e;
        } catch (Exception e) {
            logger.error("获取教师课程列表异常", e);
            throw new BusinessException(ResultCode.INTERNAL_SERVER_ERROR, "获取课程列表失败: " + e.getMessage());
        }
    }
    
    @Override
    @Transactional
    public Course createCourse(String username, Course course) {
        logger.info("开始创建课程，用户名: {}, 课程信息: {}", username, course);
        
        try {
            // 根据用户名获取教师ID
            logger.info("开始根据用户名[{}]获取教师信息", username);
            Teacher teacher = getTeacherByUsername(username);
            logger.info("成功获取到教师信息: id={}, userId={}, department={}, title={}", 
                    teacher.getId(), teacher.getUserId(), teacher.getDepartment(), teacher.getTitle());
            
            // 设置教师ID
            course.setTeacherId(teacher.getId());
            logger.info("设置教师ID: {}", teacher.getId());
            
            // 处理前端兼容字段
            if (course.getTitle() == null && course.getCourseName() != null) {
                course.setTitle(course.getCourseName());
                logger.info("从courseName字段获取标题: {}", course.getTitle());
            }
            
            if (course.getCourseType() == null && course.getCategory() != null) {
                course.setCourseType(course.getCategory());
                logger.info("从category字段获取课程类型: {}", course.getCourseType());
            }
            
            if (course.getTerm() == null && course.getSemester() != null) {
                course.setTerm(course.getSemester());
                logger.info("从semester字段获取学期: {}", course.getTerm());
            }
            
            // 设置初始状态
            if (course.getStatus() == null || course.getStatus().isEmpty()) {
                course.setStatus("未开始");
                logger.info("设置课程初始状态: 未开始");
            } else {
                logger.info("使用提供的课程状态: {}", course.getStatus());
            }
            
            // 设置初始学生数量
            if (course.getStudentCount() == null) {
                course.setStudentCount(0);
                logger.info("设置初始学生数量: 0");
            } else {
                logger.info("使用提供的学生数量: {}", course.getStudentCount());
            }
            
            // 确保必填字段不为空
            if (course.getTitle() == null || course.getTitle().isEmpty()) {
                logger.error("课程名称为空，无法创建课程");
                throw new BusinessException(ResultCode.VALIDATION_ERROR, "课程名称不能为空");
            }
            logger.info("课程名称有效: {}", course.getTitle());
            
            // 处理开始和结束时间
            if (course.getStartTime() == null) {
                logger.warn("课程开始时间为空，设置为当前时间");
                course.setStartTime(LocalDateTime.now());
            } else {
                logger.info("使用提供的开始时间: {}", course.getStartTime());
            }
            
            if (course.getEndTime() == null) {
                logger.warn("课程结束时间为空，设置为开始时间后3个月");
                course.setEndTime(course.getStartTime().plusMonths(3));
            } else {
                logger.info("使用提供的结束时间: {}", course.getEndTime());
            }
            
            // 处理学期信息
            if (course.getTerm() == null || course.getTerm().isEmpty()) {
                logger.warn("课程学期为空，设置为默认学期");
                course.setTerm("2024-2025-1");
            } else {
                logger.info("使用提供的学期: {}", course.getTerm());
            }
            
            // 处理学分信息
            if (course.getCredit() == null) {
                logger.warn("课程学分为空，设置为默认值3.0");
                course.setCredit(new BigDecimal("3.0"));
            } else {
                logger.info("使用提供的学分: {}", course.getCredit());
            }
            
            // 设置创建时间和更新时间
            LocalDateTime now = LocalDateTime.now();
            course.setCreateTime(now);
            course.setUpdateTime(now);
            logger.info("设置创建时间和更新时间: {}", now);
            
            logger.info("准备插入数据库的课程信息: {}", course);
            
            // 插入数据库前检查所有必要字段
            logger.info("检查课程字段映射 - title: {}, description: {}, coverImage: {}, credit: {}, courseType: {}, startTime: {}, endTime: {}, teacherId: {}, status: {}, term: {}, studentCount: {}",
                    course.getTitle(), course.getDescription(), course.getCoverImage(), course.getCredit(), 
                    course.getCourseType(), course.getStartTime(), course.getEndTime(), course.getTeacherId(),
                    course.getStatus(), course.getTerm(), course.getStudentCount());
            
            // 插入数据库
            try {
                int result = courseMapper.insert(course);
                logger.info("数据库插入结果: {}", result);
                
                if (result > 0) {
                    logger.info("课程创建成功，ID: {}", course.getId());
                    return course;
                } else {
                    logger.error("课程创建失败，插入数据库返回结果为0");
                    throw new BusinessException(ResultCode.INTERNAL_SERVER_ERROR, "课程创建失败，请稍后再试");
                }
            } catch (Exception e) {
                logger.error("数据库插入异常", e);
                throw new BusinessException(ResultCode.INTERNAL_SERVER_ERROR, "数据库操作异常: " + e.getMessage());
            }
        } catch (BusinessException e) {
            logger.error("创建课程业务异常: {}", e.getMessage());
            throw e;
        } catch (Exception e) {
            logger.error("创建课程异常", e);
            throw new BusinessException(ResultCode.INTERNAL_SERVER_ERROR, "创建课程失败: " + e.getMessage());
        }
    }
    
    @Override
    public Course getCourseDetail(String username, Long courseId) {
        // 根据用户名获取教师ID
        Teacher teacher = getTeacherByUsername(username);
        
        // 查询课程
        Course course = courseMapper.selectById(courseId);
        if (course == null) {
            throw new BusinessException(ResultCode.NOT_FOUND, "课程不存在");
        }
        
        // 验证权限
        if (!course.getTeacherId().equals(teacher.getId())) {
            throw new BusinessException(ResultCode.FORBIDDEN, "无权访问该课程");
        }
        
        // 更新课程状态
        return updateCourseStatus(course);
    }
    
    @Override
    @Transactional
    public Course updateCourse(String username, Course course) {
        // 获取原课程信息
        Course originalCourse = getCourseDetail(username, course.getId());
        
        // 更新课程信息
        course.setTeacherId(originalCourse.getTeacherId());  // 保持教师ID不变
        courseMapper.updateById(course);
        
        return courseMapper.selectById(course.getId());
    }
    
    @Override
    @Transactional
    public boolean deleteCourse(String username, Long courseId) {
        logger.info("开始删除课程，用户名: {}, 课程ID: {}", username, courseId);
        
        try {
            // 检查课程是否存在
            Course course = courseMapper.selectById(courseId);
            if (course == null) {
                logger.error("课程不存在，ID: {}", courseId);
                throw new BusinessException(ResultCode.NOT_FOUND, "课程不存在");
            }
            
            // 根据用户名获取教师ID
            Teacher teacher = getTeacherByUsername(username);
            
            // 验证权限
            if (!course.getTeacherId().equals(teacher.getId())) {
                logger.error("无权删除该课程，教师ID: {}, 课程所属教师ID: {}", teacher.getId(), course.getTeacherId());
                throw new BusinessException(ResultCode.FORBIDDEN, "无权删除该课程");
            }
            
            // 删除课程
            int result = courseMapper.deleteById(courseId);
            
            if (result > 0) {
                logger.info("课程删除成功，ID: {}", courseId);
                return true;
            } else {
                logger.error("课程删除失败，数据库操作返回结果为0，ID: {}", courseId);
                throw new BusinessException(ResultCode.INTERNAL_SERVER_ERROR, "课程删除失败，请稍后再试");
            }
        } catch (BusinessException e) {
            logger.error("删除课程业务异常: {}", e.getMessage());
            throw e;
        } catch (Exception e) {
            logger.error("删除课程异常", e);
            throw new BusinessException(ResultCode.INTERNAL_SERVER_ERROR, "删除课程失败: " + e.getMessage());
        }
    }
    
    @Override
    @Transactional
    public Course publishCourse(String username, Long courseId) {
        // 获取课程信息
        Course course = getCourseDetail(username, courseId);
        
        // 更新状态
        course.setStatus(Course.Status.IN_PROGRESS.getCode());
        courseMapper.updateById(course);
        
        return courseMapper.selectById(courseId);
    }
    
    @Override
    @Transactional
    public Course unpublishCourse(String username, Long courseId) {
        // 获取课程信息
        Course course = getCourseDetail(username, courseId);
        
        // 更新状态
        course.setStatus(Course.Status.NOT_STARTED.getCode());
        courseMapper.updateById(course);
        
        return courseMapper.selectById(courseId);
    }
    
    @Override
    public Map<String, Object> getCourseStatistics(String username, Long courseId) {
        // 获取课程信息
        Course course = getCourseDetail(username, courseId);
        
        // 构建统计信息
        Map<String, Object> statistics = new HashMap<>();
        statistics.put("studentCount", course.getStudentCount());
        statistics.put("averageScore", course.getAverageScore());
        statistics.put("progress", course.getProgress());
        
        // 可以添加更多统计信息...
        
        return statistics;
    }
    
    /**
     * 根据用户名获取教师信息
     * 
     * @param username 用户名
     * @return 教师信息
     */
    private Teacher getTeacherByUsername(String username) {
        logger.info("开始根据用户名获取教师信息: {}", username);
        try {
            // 检查用户名是否为匿名用户
            if ("anonymousUser".equals(username)) {
                logger.error("匿名用户无法获取教师信息，请先登录");
                throw new BusinessException(ResultCode.UNAUTHORIZED, "请先登录");
            }
            
            Teacher teacher = null;
            
            // 尝试将username解析为userId
            if (username.matches("\\d+")) {
                try {
                    Long userId = Long.parseLong(username);
                    logger.info("尝试通过userId={}查找教师信息", userId);
                    teacher = teacherMapper.selectByUserId(userId);
                    if (teacher != null) {
                        logger.info("通过userId={}找到教师信息: id={}, department={}", 
                                userId, teacher.getId(), teacher.getDepartment());
                    } else {
                        logger.warn("通过userId={}未找到教师信息", userId);
                    }
                } catch (NumberFormatException e) {
                    logger.error("无法将username解析为userId: {}", username);
                }
            }
            
            // 如果通过userId未找到，则尝试通过用户名查找
            if (teacher == null) {
                // 查询教师信息
                logger.debug("调用teacherMapper.selectByUsername查询教师信息，用户名: {}", username);
                teacher = teacherMapper.selectByUsername(username);
            }
            
            if (teacher == null) {
                logger.error("未找到教师信息，用户名: {}", username);
                throw new BusinessException(ResultCode.NOT_FOUND, "未找到教师信息，请确认您是否有教师身份");
            }
            
            logger.info("成功获取教师信息: id={}, userId={}, department={}, title={}", 
                    teacher.getId(), teacher.getUserId(), teacher.getDepartment(), teacher.getTitle());
            
            return teacher;
        } catch (BusinessException e) {
            logger.error("获取教师信息业务异常: {}", e.getMessage());
            throw e;
        } catch (Exception e) {
            logger.error("获取教师信息异常", e);
            throw new BusinessException(ResultCode.INTERNAL_SERVER_ERROR, "获取教师信息失败: " + e.getMessage());
        }
    }

    /**
     * 根据课程的开始时间和结束时间更新课程状态
     * 
     * @param course 课程对象
     * @return 更新后的课程对象
     */
    private Course updateCourseStatus(Course course) {
        LocalDateTime now = LocalDateTime.now();
        String currentStatus = course.getStatus();
        String newStatus = currentStatus;
        
        // 根据时间判断课程状态
        if (course.getStartTime() != null && course.getEndTime() != null) {
            if (now.isBefore(course.getStartTime())) {
                newStatus = Course.Status.NOT_STARTED.getCode(); // 未开始
            } else if (now.isAfter(course.getEndTime())) {
                newStatus = Course.Status.FINISHED.getCode(); // 已结束
            } else {
                newStatus = Course.Status.IN_PROGRESS.getCode(); // 进行中
            }
        }
        
        // 如果状态有变化，更新数据库
        if (!currentStatus.equals(newStatus)) {
            logger.info("更新课程状态 - 课程ID: {}, 旧状态: {}, 新状态: {}", course.getId(), currentStatus, newStatus);
            course.setStatus(newStatus);
            courseMapper.updateById(course);
        }
        
        return course;
    }
} 