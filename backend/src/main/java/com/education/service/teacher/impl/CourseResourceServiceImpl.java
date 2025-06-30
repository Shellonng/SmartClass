package com.education.service.teacher.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.education.dto.common.PageRequest;
import com.education.dto.common.PageResponse;
import com.education.entity.Course;
import com.education.entity.CourseResource;
import com.education.entity.User;
import com.education.exception.BusinessException;
import com.education.exception.ResultCode;
import com.education.mapper.CourseMapper;
import com.education.mapper.CourseResourceMapper;
import com.education.mapper.UserMapper;
import com.education.service.common.FileService;
import com.education.service.teacher.CourseResourceService;
import com.education.utils.FileUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.multipart.MultipartFile;

import java.util.Arrays;
import java.util.List;

/**
 * 课程资源服务实现类
 */
@Service
public class CourseResourceServiceImpl implements CourseResourceService {

    private static final Logger logger = LoggerFactory.getLogger(CourseResourceServiceImpl.class);
    
    private static final List<String> ALLOWED_FILE_TYPES = Arrays.asList(
        "pdf", "doc", "docx", "ppt", "pptx", "xls", "xlsx", "txt", "zip", "rar", "7z", 
        "jpg", "jpeg", "png", "gif", "mp4", "mp3", "wav"
    );

    @Autowired
    private CourseResourceMapper courseResourceMapper;

    @Autowired
    private CourseMapper courseMapper;

    @Autowired
    private UserMapper userMapper;

    @Autowired
    private FileService fileService;

    @Override
    @Transactional
    public CourseResource uploadResource(String username, Long courseId, MultipartFile file, String name, String description) {
        // 获取当前用户
        User user = userMapper.selectByUsername(username);
        if (user == null) {
            throw new BusinessException(ResultCode.USER_NOT_FOUND);
        }

        // 检查课程是否存在，以及当前用户是否有权限
        Course course = courseMapper.selectById(courseId);
        if (course == null) {
            throw new BusinessException(ResultCode.COURSE_NOT_FOUND);
        }

        // 检查是否为课程教师或管理员
        if (!course.getTeacherId().equals(user.getId()) && !"ROLE_ADMIN".equals(user.getRole())) {
            throw new BusinessException(ResultCode.PERMISSION_DENIED, "只有课程教师或管理员可以上传资源");
        }

        // 检查文件类型
        String originalFilename = file.getOriginalFilename();
        String fileType = FileUtils.getFileExtension(originalFilename);
        
        if (!ALLOWED_FILE_TYPES.contains(fileType.toLowerCase())) {
            throw new BusinessException(ResultCode.FILE_TYPE_ERROR, 
                    "不支持的文件类型，仅支持: " + String.join(", ", ALLOWED_FILE_TYPES));
        }

        // 上传文件
        String fileUrl = fileService.uploadFile(file, "resources/" + courseId);

        // 创建资源记录
        CourseResource resource = new CourseResource()
                .setCourseId(courseId)
                .setName(name != null && !name.isEmpty() ? name : originalFilename)
                .setFileType(fileType)
                .setFileSize(file.getSize())
                .setFileUrl(fileUrl)
                .setDescription(description)
                .setDownloadCount(0)
                .setUploadUserId(user.getId());

        // 保存到数据库
        courseResourceMapper.insert(resource);
        
        // 设置上传用户名
        resource.setUploadUserName(user.getUsername());

        return resource;
    }

    @Override
    public List<CourseResource> getCourseResources(String username, Long courseId) {
        // 验证用户权限
        validateUserCourseAccess(username, courseId);
        
        // 查询课程资源列表
        return courseResourceMapper.selectByCourseId(courseId);
    }

    @Override
    public PageResponse<CourseResource> getCourseResourcesPage(String username, Long courseId, PageRequest pageRequest) {
        // 验证用户权限
        validateUserCourseAccess(username, courseId);
        
        // 分页查询
        Page<CourseResource> page = new Page<>(pageRequest.getPage(), pageRequest.getSize());
        LambdaQueryWrapper<CourseResource> wrapper = new LambdaQueryWrapper<CourseResource>()
                .eq(CourseResource::getCourseId, courseId)
                .orderByDesc(CourseResource::getCreateTime);
        
        IPage<CourseResource> resultPage = courseResourceMapper.selectPage(page, wrapper);
        
        // 查询上传用户名
        for (CourseResource resource : resultPage.getRecords()) {
            User user = userMapper.selectById(resource.getUploadUserId());
            if (user != null) {
                resource.setUploadUserName(user.getUsername());
            }
        }
        
        return PageResponse.fromIPage(resultPage);
    }

    @Override
    @Transactional
    public boolean deleteResource(String username, Long resourceId) {
        // 获取资源信息
        CourseResource resource = courseResourceMapper.selectById(resourceId);
        if (resource == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND);
        }

        // 验证用户权限
        User user = userMapper.selectByUsername(username);
        if (user == null) {
            throw new BusinessException(ResultCode.USER_NOT_FOUND);
        }

        // 检查是否为资源上传者、课程教师或管理员
        Course course = courseMapper.selectById(resource.getCourseId());
        if (course == null) {
            throw new BusinessException(ResultCode.COURSE_NOT_FOUND);
        }

        boolean isResourceOwner = resource.getUploadUserId().equals(user.getId());
        boolean isCourseTeacher = course.getTeacherId().equals(user.getId());
        boolean isAdmin = "ROLE_ADMIN".equals(user.getRole());

        if (!isResourceOwner && !isCourseTeacher && !isAdmin) {
            throw new BusinessException(ResultCode.PERMISSION_DENIED, "只有资源上传者、课程教师或管理员可以删除资源");
        }

        // 删除文件
        fileService.deleteFile(resource.getFileUrl());

        // 删除数据库记录
        return courseResourceMapper.deleteById(resourceId) > 0;
    }

    @Override
    public CourseResource getResourceDetail(String username, Long resourceId) {
        // 获取资源信息
        CourseResource resource = courseResourceMapper.selectById(resourceId);
        if (resource == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND);
        }

        // 验证用户权限
        validateUserCourseAccess(username, resource.getCourseId());
        
        // 查询上传用户名
        User user = userMapper.selectById(resource.getUploadUserId());
        if (user != null) {
            resource.setUploadUserName(user.getUsername());
        }
        
        return resource;
    }

    @Override
    public boolean incrementDownloadCount(Long resourceId) {
        return courseResourceMapper.incrementDownloadCount(resourceId) > 0;
    }

    /**
     * 验证用户对课程的访问权限
     *
     * @param username 用户名
     * @param courseId 课程ID
     */
    private void validateUserCourseAccess(String username, Long courseId) {
        User user = userMapper.selectByUsername(username);
        if (user == null) {
            throw new BusinessException(ResultCode.USER_NOT_FOUND);
        }

        Course course = courseMapper.selectById(courseId);
        if (course == null) {
            throw new BusinessException(ResultCode.COURSE_NOT_FOUND);
        }

        // 检查是否为课程教师、学生或管理员
        boolean isCourseTeacher = course.getTeacherId().equals(user.getId());
        boolean isAdmin = "ROLE_ADMIN".equals(user.getRole());
        
        // 这里可以添加检查学生是否选修了该课程的逻辑
        // boolean isStudent = studentCourseMapper.isStudentInCourse(user.getId(), courseId);
        
        if (!isCourseTeacher && !isAdmin) {
            throw new BusinessException(ResultCode.PERMISSION_DENIED, "没有访问该课程资源的权限");
        }
    }
} 