package com.education.service.teacher.impl;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.education.dto.CourseResourceDTO;
import com.education.dto.common.PageRequest;
import com.education.dto.common.PageResponse;
import com.education.entity.Course;
import com.education.entity.CourseResource;
import com.education.exception.BusinessException;
import com.education.exception.ResultCode;
import com.education.mapper.CourseMapper;
import com.education.mapper.CourseResourceMapper;
import com.education.service.common.FileService;
import com.education.service.teacher.CourseResourceService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.multipart.MultipartFile;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

/**
 * 课程资源服务实现类
 */
@Service
public class CourseResourceServiceImpl implements CourseResourceService {

    private static final Logger logger = LoggerFactory.getLogger(CourseResourceServiceImpl.class);
    
    @Autowired
    private CourseResourceMapper courseResourceMapper;
    
    @Autowired
    private CourseMapper courseMapper;
    
    @Autowired
    private FileService fileService;
    
    @Override
    public CourseResourceDTO uploadResource(Long userId, Long courseId, MultipartFile file, String name, String description) {
        logger.info("上传课程资源，用户ID: {}, 课程ID: {}, 文件名: {}", userId, courseId, file.getOriginalFilename());
        
        // 检查课程是否存在
        Course course = courseMapper.selectById(courseId);
        if (course == null) {
            throw new BusinessException(ResultCode.COURSE_NOT_FOUND);
        }
        
        // 上传文件
        String fileUrl = fileService.uploadFile(file, "resources/" + courseId);
        
        // 创建资源记录
        CourseResource resource = new CourseResource();
        resource.setCourseId(courseId);
        resource.setName(name);
        resource.setFileType(getFileExtension(file.getOriginalFilename()));
        resource.setFileSize(file.getSize());
        resource.setFileUrl(fileUrl);
        resource.setDescription(description);
        resource.setDownloadCount(0);
        resource.setUploadUserId(userId);
        
        // 保存到数据库
        courseResourceMapper.insert(resource);
        
        return convertToDTO(resource);
    }

    @Override
    public List<CourseResourceDTO> listResources(Long courseId) {
        logger.info("获取课程资源列表，课程ID: {}", courseId);
        
        // 检查课程是否存在
        Course course = courseMapper.selectById(courseId);
        if (course == null) {
            throw new BusinessException(ResultCode.COURSE_NOT_FOUND);
        }
        
        // 查询资源列表
        QueryWrapper<CourseResource> queryWrapper = new QueryWrapper<>();
        queryWrapper.eq("course_id", courseId);
        queryWrapper.orderByDesc("create_time");
        
        List<CourseResource> resources = courseResourceMapper.selectList(queryWrapper);
        
        // 转换为DTO
        return resources.stream()
            .map(this::convertToDTO)
            .collect(Collectors.toList());
    }

    @Override
    public PageResponse<CourseResourceDTO> listByCourse(Long courseId, PageRequest pageRequest) {
        logger.info("获取课程资源列表，课程ID: {}, 页码: {}, 每页大小: {}", courseId, pageRequest.getCurrent(), pageRequest.getSize());
        
        // 检查课程是否存在
        Course course = courseMapper.selectById(courseId);
        if (course == null) {
            throw new BusinessException(ResultCode.COURSE_NOT_FOUND);
        }
        
        // 构建分页查询条件
        Page<CourseResource> page = new Page<>(pageRequest.getCurrent(), pageRequest.getSize());
        QueryWrapper<CourseResource> queryWrapper = new QueryWrapper<>();
        queryWrapper.eq("course_id", courseId);
        queryWrapper.orderByDesc("create_time"); // 按创建时间降序排序
        
        // 执行查询
        IPage<CourseResource> resourcePage = courseResourceMapper.selectPage(page, queryWrapper);
        
        // 转换为DTO
        List<CourseResourceDTO> resourceDTOs = resourcePage.getRecords().stream()
            .map(this::convertToDTO)
            .collect(Collectors.toList());
        
        // 构建分页响应
        PageResponse<CourseResourceDTO> response = new PageResponse<>();
        response.setCurrent((int)resourcePage.getCurrent());
        response.setPageSize((int)resourcePage.getSize());
        response.setTotal(resourcePage.getTotal());
        response.setPages(resourcePage.getPages());
        response.setRecords(resourceDTOs);
        
        return response;
    }

    @Override
    public boolean deleteCourseResource(Long resourceId, Long userId) {
        logger.info("删除课程资源，资源ID: {}, 用户ID: {}", resourceId, userId);
        
        // 查询资源信息
        CourseResource resource = courseResourceMapper.selectById(resourceId);
        if (resource == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND);
        }
        
        // 检查课程是否存在
        Course course = courseMapper.selectById(resource.getCourseId());
        if (course == null) {
            throw new BusinessException(ResultCode.COURSE_NOT_FOUND);
        }
        
        // 检查权限（只有课程创建者或资源上传者可以删除）
        if (!userId.equals(course.getTeacherId()) && !userId.equals(resource.getUploadUserId())) {
            throw new BusinessException(ResultCode.PERMISSION_DENIED);
        }
        
        // 删除资源文件
        try {
            String filePath = resource.getFileUrl();
            if (filePath != null && !filePath.isEmpty()) {
                fileService.deleteFile(filePath);
            }
        } catch (Exception e) {
            logger.error("删除资源文件失败: {}", e.getMessage());
            // 继续删除数据库记录
        }
        
        // 删除数据库记录
        int result = courseResourceMapper.deleteById(resourceId);
        return result > 0;
    }

    @Override
    public CourseResourceDTO getResourceInfo(Long resourceId, Long userId) {
        logger.info("获取课程资源信息，资源ID: {}, 用户ID: {}", resourceId, userId);
        
        // 查询资源信息
        CourseResource resource = courseResourceMapper.selectById(resourceId);
        if (resource == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND);
        }
        
        // 检查课程是否存在
        Course course = courseMapper.selectById(resource.getCourseId());
        if (course == null) {
            throw new BusinessException(ResultCode.COURSE_NOT_FOUND);
        }
        
        // 检查用户是否有权限访问该资源
        // TODO: 根据业务需求检查权限
        
        return convertToDTO(resource);
    }
    
    @Override
    public boolean incrementDownloadCount(Long resourceId) {
        logger.info("更新资源下载次数，资源ID: {}", resourceId);
        
        // 查询资源信息
        CourseResource resource = courseResourceMapper.selectById(resourceId);
        if (resource == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND);
        }
        
        // 更新下载次数
        resource.setDownloadCount(resource.getDownloadCount() + 1);
        int result = courseResourceMapper.updateById(resource);
        
        return result > 0;
    }
    
    /**
     * 将实体对象转换为DTO
     * 
     * @param resource 课程资源实体
     * @return 课程资源DTO
     */
    private CourseResourceDTO convertToDTO(CourseResource resource) {
        if (resource == null) {
            return null;
        }
        
        CourseResourceDTO dto = new CourseResourceDTO();
        dto.setId(resource.getId());
        dto.setCourseId(resource.getCourseId());
        dto.setName(resource.getName());
        dto.setFileType(resource.getFileType());
        dto.setFileSize(resource.getFileSize());
        dto.setFileUrl(resource.getFileUrl());
        dto.setDescription(resource.getDescription());
        dto.setDownloadCount(resource.getDownloadCount());
        dto.setUploadUserId(resource.getUploadUserId());
        dto.setCreateTime(resource.getCreateTime());
        dto.setUpdateTime(resource.getUpdateTime());
        
        // 格式化文件大小
        dto.setFormattedSize(formatFileSize(resource.getFileSize()));
        
        return dto;
    }
    
    /**
     * 获取文件扩展名
     * 
     * @param filename 文件名
     * @return 文件扩展名
     */
    private String getFileExtension(String filename) {
        if (filename == null || filename.isEmpty()) {
            return "";
        }
        
        int dotIndex = filename.lastIndexOf('.');
        if (dotIndex > 0 && dotIndex < filename.length() - 1) {
            return filename.substring(dotIndex + 1).toLowerCase();
        }
        
        return "";
    }
    
    /**
     * 格式化文件大小
     * 
     * @param size 文件大小（字节）
     * @return 格式化后的文件大小
     */
    private String formatFileSize(Long size) {
        if (size == null) {
            return "0 B";
        }
        
        if (size < 1024) {
            return size + " B";
        } else if (size < 1024 * 1024) {
            return String.format("%.2f KB", size / 1024.0);
        } else if (size < 1024 * 1024 * 1024) {
            return String.format("%.2f MB", size / (1024.0 * 1024));
        } else {
            return String.format("%.2f GB", size / (1024.0 * 1024 * 1024));
        }
    }
} 